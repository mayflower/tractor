"""Compile a validated Clairic request into an explicit plan on the production core.

No inference happens here. Every capability is checked before the extractor is
called, so an unsupported request never consumes a model pass. External field
names stay unchanged; internal IDs are collision-free and anonymous. The
semantic part of a field (its instructions or name) reaches the model as the
field description.

Instructions are free text. They are accepted only in the documented forms
below; anything else is `unknown` and rejected, never silently extracted.
"""

import re
from dataclasses import dataclass, field

from .capabilities import (
    CapabilityUnavailable,
    CapacityExceeded,
    ClairicError,
    UnsupportedOperation,
)
from .clairic_compute import Operand, Operation
from .types import (
    UNITS,
    Date,
    Datetime,
    Duration,
    ExtractRequest,
    Money,
    Number,
    Quantity,
    Text,
    Time,
)

CURRENCIES = ("EUR", "GBP", "USD", "CAD", "AUD", "CHF")
MAX_INTERNAL_FIELDS = 32  # ExtractRequest's limit; the backend's token limit applies too.
MAX_DESCRIPTION = 240
COMPILER_VERSION = "clairic-compiler-v1"

WORD = r"[^\W\d_][\w'’&/-]*"
LABEL = rf"{WORD}(?: {WORD}){{0,7}}"
# Words that turn a label into a derivation. This is defense in depth inside an
# already allowlisted sentence form, not a classifier of arbitrary instructions.
OPERATOR_WORDS = frozenset("""
plus minus times divided multiplied sum difference product average mean median ratio percentage
including excluding incl excl after before until since between per estimate estimated predicted
forecast expected likely projected remaining due-in
summe differenz produkt durchschnitt inklusive inkl zuzüglich zzgl abzüglich abzgl nach vor bis
seit zwischen pro geschätzt geschätzte geschätzten erwartet erwartete voraussichtlich voraussichtliche
""".split())
NULL_EN = r"(?: Return null if (?:it is )?(?:absent|missing|not stated)\.)?"
NULL_DE = r"(?: Gib null zurück, wenn (?:er|sie|es) fehlt\.)?"
SOURCE_FORMS = (
    re.compile(rf"(?:Extract|Copy|Return) the (?:explicitly|literally) (?:stated|written|printed|given|mentioned) "
               rf"(?P<label>{LABEL})(?: exactly as written)?\.?{NULL_EN}"),
    re.compile(rf"(?:Extrahiere|Übernimm) (?:den|die|das) (?:ausdrücklich|wörtlich) "
               rf"(?:genannten?|angegebenen?|geschriebenen?) (?P<label>{LABEL})\.?{NULL_DE}"),
)
COMPUTE_FORMS = (
    ("binary", re.compile(rf"(?P<op>Sum|Difference|Product) of the explicitly stated (?P<a>{LABEL}) "
                          rf"and the explicitly stated (?P<b>{LABEL})\.?")),
    ("date", re.compile(rf"The explicitly stated (?P<a>{LABEL}) plus (?P<n>[0-9]{{1,4}}) "
                        rf"(?P<unit>days?|weeks?|months?|years?)\.?")),
)


@dataclass(frozen=True)
class Policy:
    """Server-side processing policy; Clairic itself has no locale field.

    These are local extensions configured at server start, never request keys.
    `None` means only unambiguous forms are accepted.
    """

    locale: str | None = None
    date_order: str | None = None
    compute: bool = False


@dataclass(frozen=True)
class FieldPlan:
    name: str
    spec: object
    operation: str  # supported_source_only or supported_compute
    label: str
    internal_id: str | None = None
    compute: Operation | None = None
    capabilities: tuple = ()


@dataclass(frozen=True)
class OperationPlan:
    request: object
    fields: tuple
    extract_request: ExtractRequest
    capabilities: tuple = ()
    internal_names: dict = field(default_factory=dict)
    version: str = COMPILER_VERSION


def valid_label(label):
    return label is not None and not any(w.casefold() in OPERATOR_WORDS for w in label.split())


def classify(spec):
    """(operation, label, match) for the documented forms; ('unknown', ...) otherwise."""
    text = spec.instructions
    if text is None:
        # The field name is the semantic description, e.g. "invoice_number".
        label = " ".join(re.split(r"[_.\-]+", spec.name)).strip() or spec.name
        return "supported_source_only", label, None
    text = " ".join(text.split())
    for form in SOURCE_FORMS:
        match = form.fullmatch(text)
        if match and valid_label(match["label"]):
            return "supported_source_only", match["label"], match
    for kind, form in COMPUTE_FORMS:
        match = form.fullmatch(text)
        if match and all(valid_label(match[k]) for k in ("a", "b") if k in match.groupdict()):
            return "supported_compute", kind, match
    return "unknown", None, None


def unit_kind(unit):
    if unit is None:
        return "number"
    if unit in CURRENCIES:
        return "money"
    if unit in UNITS:
        return "quantity"
    return "number"  # An output label only; nothing is converted.


def tractor_field(kind, description, policy, *, unit=None, cardinality="optional_one"):
    """One Clairic value type to one existing Tractor field contract."""
    common = {"cardinality": cardinality}
    if kind == "text":
        return Text(description, **common)
    if kind == "number":
        target = unit_kind(unit)
        if target == "money":
            return Money(description, currency=unit, locale=policy.locale, **common)
        if target == "quantity":
            return Quantity(description, unit=unit, locale=policy.locale, **common)
        return Number(description, locale=policy.locale, **common)
    if kind == "date":
        return Date(description, date_order=policy.date_order, **common)
    if kind == "time":
        return Time(description, **common)
    if kind == "datetime":
        return Datetime(description, date_order=policy.date_order, **common)
    if kind == "duration":
        return Duration(description, **common)
    raise UnsupportedOperation(f"No extraction contract for {kind}")


def description_for(name, label):
    if len(label) > MAX_DESCRIPTION:
        raise CapacityExceeded(f"Field description exceeds {MAX_DESCRIPTION} characters", field=name)
    return label


DEFAULT_POLICY = Policy()


def compile_request(request, capabilities, policy=DEFAULT_POLICY):
    """Check every requirement first; raise one error listing all rejected fields."""
    errors = []

    def reject(error):
        errors.append(error)

    if request.images:
        try:
            capabilities.require(["images"])
        except ClairicError as exc:
            reject(exc)
        else:
            reject(UnsupportedOperation("Image input is not wired to a local runtime", capability="images"))
    temporal = {"date", "datetime"}
    if request.reference_date is not None and any(
            s.type in temporal or s.items == "date" for s in request.fields.values()):
        reject(UnsupportedOperation(
            "reference_date year completion is not implemented; yearless dates are rejected"))
    internal, plans, ids = {}, [], {}

    def allocate(spec_field):
        key = f"f{len(internal)}"
        internal[key] = spec_field
        return key

    for name, spec in request.fields.items():
        needed = ["source_spans", "typed_extraction"]
        try:
            if spec.type in {"estimate", "forecast"}:
                capability = "numeric_distribution" if spec.type == "estimate" else "forecast"
                capabilities.require([capability], field=name)
                raise UnsupportedOperation(f"{spec.type} is not wired to a released runtime head",
                                           field=name, capability=capability)
            if "null_threshold" in spec.explicit:
                # An explicit threshold can only be honored with a real p_null.
                capabilities.require(["typed_answerability"], field=name)
            operation, label, match = classify(spec)
            if operation == "unknown":
                raise UnsupportedOperation(
                    "Instructions are not in a supported form; see docs/clairic-compatibility.md",
                    field=name)
            if operation == "supported_compute":
                plans.append(compile_compute(name, spec, label, match, capabilities, policy, allocate))
                continue
            if spec.type == "list":
                needed.append("typed_lists")
                contract = tractor_field(spec.items, description_for(name, label), policy,
                                         cardinality="unique_list")
            else:
                contract = tractor_field(spec.type, description_for(name, label), policy, unit=spec.unit)
            capabilities.require(needed, field=name)
            key = allocate(contract)
            ids[key] = name
            plans.append(FieldPlan(name, spec, operation, label, internal_id=key, capabilities=tuple(needed)))
        except ClairicError as exc:
            reject(exc)
    if len(internal) > MAX_INTERNAL_FIELDS:
        reject(CapacityExceeded(f"At most {MAX_INTERNAL_FIELDS} internal fields fit one local pass"))
    if errors:
        first = errors[0]
        if len(errors) > 1:
            first.details = [e.describe() for e in errors]
        raise first
    extract = ExtractRequest(text=request.state, fields=internal, locale=policy.locale)
    return OperationPlan(request=request, fields=tuple(plans), extract_request=extract,
                         capabilities=tuple(sorted({c for p in plans for c in p.capabilities})),
                         internal_names=ids)


def compile_compute(name, spec, kind, match, capabilities, policy, allocate):
    if not policy.compute:
        raise CapabilityUnavailable("Deterministic compute is disabled on this server",
                                    field=name, capability="deterministic_compute")
    capabilities.require(["deterministic_compute", "source_spans", "typed_extraction"], field=name)
    if kind == "binary":
        if spec.type != "number":
            raise UnsupportedOperation("Arithmetic results must be number fields", field=name)
        operator = match["op"].lower()
        unit = spec.unit if operator != "product" else None
        first_kind = unit_kind(unit)
        second_unit = spec.unit if operator == "product" else unit
        operands = []
        for label, operand_unit in ((match["a"], unit), (match["b"], second_unit)):
            contract = tractor_field("number", description_for(name, label), policy, unit=operand_unit)
            operands.append(Operand(allocate(contract), label, unit_kind(operand_unit)))
        if operator == "product" and first_kind != "number":
            raise UnsupportedOperation("Products take a plain count and one unit operand", field=name)
        operation = Operation(operator, tuple(operands))
    else:
        if spec.type != "date":
            raise UnsupportedOperation("Date addition results must be date fields", field=name)
        contract = tractor_field("date", description_for(name, match["a"]), policy)
        operand = Operand(allocate(contract), match["a"], "date")
        unit = match["unit"].rstrip("s")
        operation = Operation(f"add_{unit}s", (operand,), constant=int(match["n"]))
    return FieldPlan(name, spec, "supported_compute", kind, compute=operation,
                     capabilities=("deterministic_compute", "source_spans", "typed_extraction"))
