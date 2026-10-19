"""Conservative scalar parsing on already selected original source spans."""

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, InvalidOperation, localcontext

import pint
from babel.numbers import NumberFormatError, parse_decimal

from .types import UNITS

CONTEXT = Context(prec=50, rounding=ROUND_HALF_EVEN)
PINT_UNITS = {u: u for u in UNITS} | {
    "in": "inch",
    "m3": "meter**3",
    "mm2": "millimeter**2",
    "cm2": "centimeter**2",
    "m2": "meter**2",
    "mile": "mile",
    "nmi": "nautical_mile",
}
with localcontext(CONTEXT):
    REGISTRY = pint.UnitRegistry(non_int_type=Decimal)
    TARGETS = {key: REGISTRY.Unit(value) for key, value in PINT_UNITS.items()}
ALIASES = {u: u for u in UNITS}
for unit, aliases in {
    "nm": "Nanometer nanometer nanometers",
    "um": "µm μm Mikrometer micrometer micrometers",
    "ug": "µg μg Mikrogramm microgram micrograms",
    "Pa": "Pascal pascal pascals",
    "kPa": "Kilopascal kilopascal kilopascals",
    "MPa": "Megapascal megapascal megapascals",
    "K": "Kelvin kelvin",
    "degC": "°C Celsius",
    "degF": "°F Fahrenheit",
    "mm": "Millimeter millimeter millimeters",
    "cm": "Zentimeter zentimeter centimeter centimeters",
    "m": "Meter meter meters",
    "km": "Kilometer kilometer kilometers",
    "in": "inch inches Zoll zoll",
    "ft": "foot feet Fuß",
    "yd": "yard yards",
    "mile": "mi miles Meile Meilen meile meilen",
    "nmi": "Seemeile Seemeilen",
    "mg": "Milligramm milligram milligrams",
    "g": "Gramm gram grams",
    "kg": "Kilogramm kilogram kilograms",
    "lb": "lbs pound pounds",
    "ms": "Millisekunden milliseconds",
    "s": "sec Sekunde Sekunden second seconds",
    "min": "Minute Minuten minute minutes",
    "h": "Stunde Stunden hour hours",
    "mL": "ml Milliliter milliliter milliliters",
    "L": "l Liter liter liters litre litres",
    "m3": "m³",
    "mm2": "mm²",
    "cm2": "cm²",
    "m2": "m²",
}.items():
    ALIASES.update({alias: unit for alias in aliases.split()})
ALIASES["nautical miles"] = "nmi"
ALIASES["nautical mile"] = "nmi"
QUALIFIER = r"(?:\b(?:bis\s+zu|mindestens|höchstens|maximal|minimal|ungefähr|circa|ca\.?|etwa|about|approximately|at\s+least|at\s+most|up\s+to)(?!\w)|[<>≤≥~≈])"
NUMBER = re.compile(r"^([+\-−]?[0-9][0-9., \u00a0\u202f]*(?:[eE][+\-−]?[0-9]+)?)\s*(.*?)$")


@dataclass(frozen=True)
class Normalized:
    value: Decimal | None = None
    issue: str | None = None
    source_unit: str | None = None


def number(literal, locale):
    literal = literal.replace("−", "-")
    pieces = re.split("[eE]", literal)
    mantissa = pieces[0].strip()
    exponent_digits = pieces[1].lstrip("+-0") if len(pieces) == 2 else ""
    if len(exponent_digits) > 3:
        return None, "numeric_out_of_range"
    exponent = int(pieces[1]) if len(pieces) == 2 else 0
    digits = re.sub(r"\D", "", mantissa).lstrip("0")
    if len(digits) > 24 or abs(exponent) > 100:
        return None, "numeric_out_of_range"
    if re.search(r"[ \u00a0\u202f]", mantissa):
        # Space grouping is allowed only in complete groups of three.
        if not re.fullmatch(r"[+-]?\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?", mantissa):
            return None, "invalid_number_format"
        mantissa = re.sub(r"[ \u00a0\u202f]", "", mantissa)
    locales = [locale.replace("-", "_")] if locale else ["de_DE", "en_US"]
    values = set()
    for candidate in locales:
        try:
            values.add(parse_decimal(mantissa, locale=candidate, strict=True))
        except (NumberFormatError, InvalidOperation):
            pass
    if not values:
        return None, "invalid_number_format"
    if len(values) != 1:
        return None, "ambiguous_number_format"
    return values.pop() * (Decimal(10) ** exponent), None


def normalize(text, span, target_unit, locale=None):
    start, end = span.get("start"), span.get("end")
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(text)
        or text[start:end] != span.get("text")
    ):
        return Normalized(issue="invalid_span")
    source = span["text"].strip()
    prefix, suffix = text[:start], text[end:]
    # A closing markup delimiter is not a greater-than qualifier. This only
    # checks the evidence boundary; it does not interpret rows or HTML layout.
    prefix = re.sub(r"(?:</?[A-Za-z][^<>]*>\s*)+$", "", prefix)
    if (
        re.search(QUALIFIER, source, re.I)
        or re.search(QUALIFIER + r"\s*$", prefix, re.I)
        or re.search(r"^[^\S\r\n]*(?:±|[×x]|[-–—]\s*\d|bis\s+\d|to\s+\d)", suffix, re.I)
        or re.search(r"±|×|\d\s*[x–—]\s*\d|\d\s*-\s*\d|\d\s+(?:bis|to)\s+\d", source, re.I)
    ):
        return Normalized(issue="non_scalar_quantity")
    if re.search(r"\d\s*/\s*\d", source):
        return Normalized(issue="unsupported_number_format")
    if re.match(r"[+-]?(?:nan|inf(?:inity)?)\b", source, re.I):
        return Normalized(issue="invalid_number_format")
    match = NUMBER.fullmatch(source)
    if not match:
        return Normalized(issue="unsupported_number_format")
    literal, unit = match.groups()
    if re.search(r"\d", unit) and unit not in ALIASES:
        return Normalized(issue="non_scalar_quantity")
    with localcontext(CONTEXT):
        value, issue = number(literal, locale)
        if issue:
            return Normalized(issue=issue)
        if not unit:
            return Normalized(issue="missing_unit")
        if unit in {"gal", "gallon", "gallons", "Pfund", "pfund"}:
            return Normalized(issue="ambiguous_unit")
        source_unit = ALIASES.get(unit)
        if source_unit is None:
            return Normalized(issue="unsupported_unit")
        try:
            converted = (
                REGISTRY.Quantity(value, TARGETS[source_unit]).to(TARGETS[target_unit]).magnitude
            )
        except pint.DimensionalityError:
            return Normalized(issue="incompatible_dimension", source_unit=source_unit)
        with localcontext(Context(prec=34, rounding=ROUND_HALF_EVEN)):
            converted = +converted
        if not converted.is_finite():
            return Normalized(issue="numeric_out_of_range")
        return Normalized(value=converted, source_unit=source_unit)


def compatible_policy(policy):
    """A pinned bundle's declared conversions must keep exactly their meaning.

    Additional supported units/aliases do not invalidate existing weights.
    Removal or reinterpretation of any declared conversion still fails closed.
    """
    return (policy.get("contract") == "tractor-v1"
            and isinstance(policy.get("aliases"), dict) and bool(policy["aliases"])
            and isinstance(policy.get("units"), dict) and bool(policy["units"])
            and all(ALIASES.get(k) == v for k, v in policy["aliases"].items())
            and all(PINT_UNITS.get(k) == v for k, v in policy["units"].items()))
