"""Local normalization of explicit contact values; no lookups or identity inference."""
import re
import unicodedata
from functools import cache

from .field_normalization import Unresolved

NAME_KINDS = frozenset({"person", "organization", "place"})


def clean_text(value):
    value = unicodedata.normalize("NFC", value)
    if any(unicodedata.category(c).startswith("C") and not c.isspace() for c in value):
        raise Unresolved("invalid_text_character")
    value = " ".join(value.split())
    if not value:
        raise Unresolved("empty_value")
    return value


def comparison_key(value, kind):
    """Type-specific equality shared by production deduplication and evaluation.

    Preserve spelling, qualifiers, punctuation, accents and identifier case.
    Only declared name/address types ignore case. Never infer identities.
    """
    if isinstance(value, str) and kind in NAME_KINDS | {"address"}:
        return " ".join(unicodedata.normalize("NFC", value).split()).lower()
    return value


@cache
def postal_rules(country):
    from i18naddress import get_validation_rules
    try:
        rules = get_validation_rules({"country_code": country})
    except (ValueError, KeyError) as exc:
        raise Unresolved("unsupported_postal_country") from exc
    if not rules.postal_code_matchers:
        raise Unresolved("unsupported_postal_country")
    return rules


def postal_code(value, country):
    rules = postal_rules(country)
    source = clean_text(value).upper()
    if not source.isascii() and not any(c.isascii() and c.isdigit() for c in source):
        raise Unresolved("invalid_postal_code")

    def canonical(candidate):
        if country in {"GB", "CA", "NL"}:
            compact = candidate.replace(" ", "")
            split = -3 if country in {"GB", "CA"} else -2
            return compact[:split] + " " + compact[split:]
        if country == "US" and re.fullmatch(r"[0-9]{9}", candidate):
            return candidate[:5] + "-" + candidate[5:]
        return candidate

    def valid(candidate):
        return candidate.isascii() and all(m.fullmatch(candidate) for m in rules.postal_code_matchers)

    direct = canonical(source)
    if valid(direct):
        return direct
    # A detected field span may contain postcode + town. Parse only that
    # evidence, using the country's bundled pattern and strict token boundaries.
    # Multiple distinct codes remain ambiguous; never consult the document/gold.
    pattern = rules.postal_code_matchers[0].pattern.removeprefix("^").removesuffix("$")
    matches = re.finditer(r"(?<![\w-])(?:" + pattern + r")(?![\w-])", source)
    codes = {canonical(m[0]) for m in matches if valid(canonical(m[0]))}
    if len(codes) > 1:
        raise Unresolved("ambiguous_postal_code")
    if codes:
        return codes.pop()
    raise Unresolved("invalid_postal_code")


def normalize_contact(value, rule):
    if rule.kind == "address":
        value = re.sub(r"\s*[\r\n]+\s*", ", ", value)
    value = clean_text(value)
    if rule.kind in NAME_KINDS:
        if not any(c.isalpha() for c in value):
            raise Unresolved("invalid_name")
        return value
    if rule.kind == "address":
        # Format only. Do not parse/invent international address components.
        result = re.sub(r"\s*,\s*", ", ", value).strip(" ,")
        if not result:
            raise Unresolved("empty_value")
        return result
    if rule.kind == "postcode":
        return postal_code(value, rule.country)
    if rule.kind == "email":
        from email_validator import EmailNotValidError, validate_email
        try:
            return validate_email(value, check_deliverability=False).normalized
        except EmailNotValidError as exc:
            raise Unresolved("invalid_email") from exc
    if rule.kind == "phone":
        import phonenumbers as pn
        if rule.country and rule.country not in pn.SUPPORTED_REGIONS:
            raise Unresolved("unsupported_phone_country")
        value = re.sub(r"^tel:", "", value, flags=re.I).strip()
        if not re.fullmatch(r"[+()0-9\s./-]+(?:\s*(?:ext\.?|x|#|;ext=)\s*[0-9]+)?", value, re.I):
            raise Unresolved("invalid_phone")
        if not value.startswith("+") and not rule.country:
            raise Unresolved("missing_phone_country")
        try:
            parsed = pn.parse(value, rule.country)
        except pn.NumberParseException as exc:
            raise Unresolved("invalid_phone") from exc
        if not pn.is_valid_number(parsed):
            raise Unresolved("invalid_phone")
        result = pn.format_number(parsed, pn.PhoneNumberFormat.E164)
        # E.164 has no extension; keep it explicitly rather than dropping it.
        return result + (";ext=" + parsed.extension if parsed.extension else "")
    raise Unresolved("unsupported_processing")
