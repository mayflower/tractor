"""Clairic request protocol types and strict validation; no ML imports.

Pinned contract: cloudflare/cloudflare-docs@e2dab6cf6f9038b866b96174cb4c447b2f834c10
src/content/workers-ai-models/clairic.json. Parsing a type does not make it
executable; capability checks happen in the request compiler. External field
names are kept exactly; internal Tractor key rules never rewrite them.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .capabilities import InvalidInput

MODEL_ALIASES = ("clairic", "@cf/cloudflare/clairic")
MODEL_ID = "@cf/cloudflare/clairic"
FIELD_TYPES = ("text", "number", "date", "time", "datetime", "duration", "list", "estimate", "forecast")
LIST_ITEMS = ("text", "number", "date")
DURATION_UNITS = ("seconds", "minutes", "hours", "days", "weeks", "months", "years")
SCALES = ("linear", "log")
INTERVALS = ("step", "minute", "hour", "day", "week", "month", "quarter", "year")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Keys a field spec may carry, per type. The published schema documents these
# restrictions in descriptions only; locally they are enforced, not ignored.
COMMON = {"type", "instructions", "null_threshold"}
ALLOWED = {
    "text": COMMON,
    "number": COMMON | {"unit", "integer", "decimals"},
    "date": COMMON,
    "time": COMMON,
    "datetime": COMMON,
    "duration": COMMON | {"unit", "integer", "decimals"},
    # Lists have no scalar null decision, so a null threshold has no meaning.
    "list": {"type", "instructions", "items", "decimals"},
    "estimate": COMMON | {"unit", "min", "max", "step", "scale"},
    "forecast": COMMON | {"unit", "min", "max", "step", "scale", "horizon", "interval"},
}
TOP_LEVEL = {"model", "state", "fields", "images", "reference_date", "options"}
# rejectIfBusy is added by the official client's transport, not by the schema.
OPTIONS = {"include_quantiles", "return_state_text", "rejectIfBusy"}
IMAGE_KEYS = {"content_type", "base64"}


def _number(value):
    """JSON numbers only: bool is not a number and NaN/Infinity never pass."""
    if isinstance(value, bool) or not isinstance(value, int | Decimal | float):
        return None
    value = Decimal(value) if not isinstance(value, Decimal) else value
    return value if value.is_finite() else None


def _iso_date(value, path):
    if not isinstance(value, str) or not ISO_DATE.fullmatch(value):
        raise InvalidInput(f"{path} must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise InvalidInput(f"{path} is not a valid calendar date") from None


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str
    instructions: str | None = None
    unit: str | None = None
    integer: bool | None = None
    decimals: int | None = None
    items: str | None = None
    min: Decimal | date | None = None
    max: Decimal | date | None = None
    step: Decimal | str | None = None
    scale: str | None = None
    null_threshold: Decimal | None = None
    horizon: int | None = None
    interval: str | None = None
    # Keys that were explicitly present, to distinguish defaults from requests.
    explicit: frozenset = frozenset()

    @property
    def threshold(self):
        """The effective threshold; the documented default applies when absent."""
        return self.null_threshold if self.null_threshold is not None else Decimal("0.5")


@dataclass(frozen=True)
class Options:
    include_quantiles: bool = False
    return_state_text: bool = False
    reject_if_busy: bool = False
    explicit: frozenset = frozenset()


@dataclass(frozen=True)
class ClairicRequest:
    model: str
    state: str
    fields: dict
    images: tuple = ()
    reference_date: date | None = None
    options: Options = Options()


def parse_field(name, spec):
    path = f"fields.{name}"
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise InvalidInput(f"Invalid field name: {name!r}", field=name if isinstance(name, str) else None)
    if not isinstance(spec, dict):
        raise InvalidInput(f"{path} must be an object", field=name)
    kind = spec.get("type")
    if kind not in FIELD_TYPES:
        raise InvalidInput(f"{path}.type must be one of {', '.join(FIELD_TYPES)}", field=name)
    unknown = set(spec) - ALLOWED[kind]
    if unknown:
        raise InvalidInput(f"{path} does not accept {', '.join(sorted(unknown))} for type {kind}", field=name)
    values = {"name": name, "type": kind, "explicit": frozenset(spec)}
    for key in spec:
        if spec[key] is None and key != "type":
            # Explicit JSON null is not the same as omission; never treat it as a default.
            raise InvalidInput(f"{path}.{key} must not be null", field=name)
    if "instructions" in spec:
        if not isinstance(spec["instructions"], str) or not spec["instructions"].strip():
            raise InvalidInput(f"{path}.instructions must be a nonempty string", field=name)
        values["instructions"] = spec["instructions"]
    if "unit" in spec:
        if not isinstance(spec["unit"], str) or not spec["unit"].strip():
            raise InvalidInput(f"{path}.unit must be a nonempty string", field=name)
        if kind == "duration" and spec["unit"] not in DURATION_UNITS:
            raise InvalidInput(f"{path}.unit must be one of {', '.join(DURATION_UNITS)}", field=name)
        values["unit"] = spec["unit"]
    if "integer" in spec:
        if not isinstance(spec["integer"], bool):
            raise InvalidInput(f"{path}.integer must be a boolean", field=name)
        values["integer"] = spec["integer"]
    if "decimals" in spec:
        decimals = spec["decimals"]
        if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 6:
            raise InvalidInput(f"{path}.decimals must be an integer from 0 to 6", field=name)
        if kind == "list" and spec.get("items") != "number":
            raise InvalidInput(f"{path}.decimals applies only to number lists", field=name)
        if spec.get("integer") is True and decimals:
            raise InvalidInput(f"{path}: integer and nonzero decimals conflict", field=name)
        values["decimals"] = decimals
    if kind == "list":
        if spec.get("items") not in LIST_ITEMS:
            raise InvalidInput(f"{path}.items is required: one of {', '.join(LIST_ITEMS)}", field=name)
        values["items"] = spec["items"]
    if "null_threshold" in spec:
        threshold = _number(spec["null_threshold"])
        if threshold is None or not Decimal(0) < threshold <= Decimal(1):
            raise InvalidInput(f"{path}.null_threshold must be a number in (0, 1]", field=name)
        values["null_threshold"] = threshold
    if kind in {"estimate", "forecast"}:
        values.update(_grid(spec, path, name, kind))
    return FieldSpec(**values)


def _grid(spec, path, name, kind):
    values = {}
    if "min" not in spec or "max" not in spec:
        raise InvalidInput(f"{path} requires min and max for {kind}", field=name)
    bounds = []
    for key in ("min", "max"):
        raw = spec[key]
        number = _number(raw)
        bounds.append(number if number is not None else _iso_date(raw, f"{path}.{key}"))
    if type(bounds[0]) is not type(bounds[1]):
        raise InvalidInput(f"{path}: min and max must both be numbers or both ISO dates", field=name)
    if not bounds[0] < bounds[1]:
        raise InvalidInput(f"{path}: min must be smaller than max", field=name)
    values["min"], values["max"] = bounds
    dated = isinstance(bounds[0], date)
    if "step" in spec:
        step = spec["step"]
        if step in ("month", "year"):
            if not dated:
                raise InvalidInput(f"{path}.step {step!r} applies only to date bounds", field=name)
            values["step"] = step
        else:
            number = _number(step)
            if number is None or number <= 0:
                raise InvalidInput(f"{path}.step must be a positive number, 'month' or 'year'", field=name)
            if dated and number != number.to_integral_value():
                raise InvalidInput(f"{path}.step must be whole days for date bounds", field=name)
            values["step"] = number
    if "scale" in spec:
        if spec["scale"] not in SCALES:
            raise InvalidInput(f"{path}.scale must be linear or log", field=name)
        if spec["scale"] == "log" and (dated or bounds[0] <= 0):
            raise InvalidInput(f"{path}: a log scale requires positive numeric bounds", field=name)
        values["scale"] = spec["scale"]
    if "horizon" in spec:
        horizon = spec["horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 64:
            raise InvalidInput(f"{path}.horizon must be an integer from 1 to 64", field=name)
        values["horizon"] = horizon
    if "interval" in spec:
        if spec["interval"] not in INTERVALS:
            raise InvalidInput(f"{path}.interval must be one of {', '.join(INTERVALS)}", field=name)
        values["interval"] = spec["interval"]
    return values


def parse_request(payload):
    """Validate one raw JSON payload (already decoded with Decimal numbers)."""
    if not isinstance(payload, dict):
        raise InvalidInput("Request body must be a JSON object")
    unknown = set(payload) - TOP_LEVEL
    if unknown:
        raise InvalidInput(f"Unknown request keys: {', '.join(sorted(unknown))}")
    # Declared as required with a default; the default does not make it optional.
    if "model" not in payload:
        raise InvalidInput("model is required")
    if payload["model"] not in MODEL_ALIASES:
        raise InvalidInput("model must be 'clairic' or '@cf/cloudflare/clairic'")
    state = payload.get("state", "")
    if not isinstance(state, str):
        raise InvalidInput("state must be a string")
    images = payload.get("images", [])
    if not isinstance(images, list) or len(images) > 4:
        raise InvalidInput("images must be a list of at most 4 images")
    for index, image in enumerate(images):
        valid = (isinstance(image, str) and image[:5].lower() == "data:") or (
            isinstance(image, dict) and set(image) == IMAGE_KEYS
            and all(isinstance(image[k], str) for k in IMAGE_KEYS))
        if not valid:
            raise InvalidInput(f"images[{index}] must be a data URL or {{content_type, base64}}")
    if not state.strip() and not images:
        raise InvalidInput("state must be nonempty unless images are sent")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise InvalidInput("fields must be an object with at least one field")
    parsed = {name: parse_field(name, spec) for name, spec in fields.items()}
    reference = payload.get("reference_date")
    if "reference_date" in payload:
        reference = _iso_date(reference, "reference_date")
    options = payload.get("options", {})
    if not isinstance(options, dict):
        raise InvalidInput("options must be an object")
    unknown = set(options) - OPTIONS
    if unknown:
        raise InvalidInput(f"Unknown options: {', '.join(sorted(unknown))}")
    if any(not isinstance(value, bool) for value in options.values()):
        raise InvalidInput("options values must be booleans")
    return ClairicRequest(
        model=payload["model"], state=state, fields=parsed, images=tuple(images),
        reference_date=reference,
        options=Options(include_quantiles=options.get("include_quantiles", False),
                        return_state_text=options.get("return_state_text", False),
                        reject_if_busy=options.get("rejectIfBusy", False),
                        explicit=frozenset(options)))
