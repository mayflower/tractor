"""Explicit numeric values, labels and magnitudes without selecting arbitrary numbers."""
import re
from decimal import Decimal, localcontext
from functools import cache

from .field_normalization import Unresolved

NUMERIC = r"[+\-−]?\d[\d., \u00a0\u202f]*(?:[eE][+\-−]?\d+)?"
COUNTS = r"(?:participants?|subjects?|employees?|staff|people|persons?|units?|(?:pouch )?cells?|specimens?|samples?|soil cores|Teilnehmende|Teilnehmer|Mitarbeitende|Mitarbeiter(?:innen)?|Beschäftigte|Personen|Proben|Probanden|Messpunkte|Probenflächen|Einheiten|Stück|Stk\.?|cartons?|packages?|Kolli|Kanister|Set|reams|cartridges)"
LABEL = r"(?:n\s*=|(?:sample size|Stichprobengröße|employees|Mitarbeitende|headcount)\s*:)"
MAGNITUDES = {"thousand": 1000, "thousands": 1000, "tausend": 1000,
              "million": 1000000, "millions": 1000000, "millionen": 1000000, "mio.": 1000000,
              "milliarde": 1000000000, "milliarden": 1000000000,
              "billionen": 1000000000000, "trillion": 1000000000000, "trillions": 1000000000000}
@cache
def currency_vocabulary():
    """CLDR spellings for supported currencies, plus explicit colloquial aliases."""
    from babel.numbers import get_currency_name, get_currency_symbol

    names = {}
    for code in ('EUR', 'GBP', 'USD', 'CAD', 'AUD', 'CHF'):
        variants = {code}
        for locale in ('de_DE', 'en_US', 'en_GB'):
            variants.add(get_currency_symbol(code, locale=locale))
            variants.update(get_currency_name(code, count=n, locale=locale) for n in (1, 2))
        for name in variants:
            names.setdefault(name.casefold(), set()).add(code)
    # A bare dollar symbol/name needs the caller's explicit currency; CLDR's
    # default symbol for one locale is not evidence of the source currency.
    for name in ('$', 'dollar', 'dollars'):
        names[name] = {'USD', 'CAD', 'AUD'}
    for name in ('pound', 'pounds', 'pound sterling', 'pounds sterling'):
        names[name] = {'GBP'}
    alternatives = '|'.join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(r'(?<![^\W\d_])(?:' + alternatives + r')(?![^\W\d_])', re.I), names


def decimal_value(value, locale):
    from .normalize import CONTEXT, number
    if not re.fullmatch(NUMERIC, value):
        raise Unresolved("non_scalar_number")
    with localcontext(CONTEXT):
        parsed, issue = number(value, locale)
    if issue:
        raise Unresolved(issue)
    return parsed


def count(value, locale):
    if re.fullmatch(NUMERIC, value):
        return decimal_value(value, locale)
    # Explicit n= identifies the reported count. A description or parenthesized
    # breakdown does not request arithmetic on the subgroup numbers. Without
    # that label we still require the entire value to be one scalar/count.
    labelled = re.fullmatch(r"n\s*=\s*(\d+)(?:\s+([A-Za-zÄÖÜäöü][A-Za-zÄÖÜäöüß\s-]*))?"
                            r"(?:\s+(?:aus|across|from)\s+\d+\s+[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß -]*)?"
                            r"(?:\s*\(([A-Za-zÄÖÜäöüß0-9 ,×*-]+)\))?", value, re.I)
    uncertain = re.search(r"\b(?:or|oder|and|und|plus|minus|to|bis|about|around|approximately|"
                          r"ca|rund|estimate[ds]?|not|nicht|at least|at most|mindestens|höchstens)\b", value, re.I)
    if labelled and not uncertain:
        return decimal_value(labelled[1], locale)
    match = re.fullmatch(r"(?:" + LABEL + r"\s*)?(" + NUMERIC + r")(?:\s+" + COUNTS + r")?", value, re.I)
    if not match:
        raise Unresolved("non_scalar_number")
    parsed = decimal_value(match[1].strip(), locale)
    if parsed < 0 or parsed % 1:
        raise Unresolved("invalid_count")
    return parsed


def money(value, currency, locale):
    value = ' '.join(value.split())
    value = re.sub(r"^(?:MSRP|Price|Preis|Amount|Betrag)\s*:?\s+", "", value, flags=re.I)
    # The money type requests the stated monetary amount, never a derived rate.
    value = re.sub(r"\s+(?:per (?:calendar )?(?:month|year|annum|day|hour|visit|inspection)|monthly|annually|"
                   r"monatlich|jährlich|pro (?:Monat|Jahr|Tag|Stunde))$", "", value, flags=re.I)
    currency_pattern, accepted = currency_vocabulary()
    tokens = list(currency_pattern.finditer(value))
    if len(tokens) != 1:
        raise Unresolved("missing_or_multiple_currency")
    token = tokens[0]
    key = token[0].casefold()
    if currency not in accepted.get(key, {key.upper()}):
        raise Unresolved("currency_mismatch")
    if token.start() == 0:
        scalar = value[token.end():].strip()
    elif token.end() == len(value):
        scalar = value[:token.start()].strip()
    else:
        raise Unresolved("invalid_money_format")
    magnitude = re.fullmatch(r"(" + NUMERIC + r")\s+([A-Za-zäöüÄÖÜ.]+)", scalar)
    factor = 1
    if magnitude:
        word = magnitude[2].casefold()
        if word in {"billion", "billions"}:
            if locale is None:
                raise Unresolved("ambiguous_magnitude")
            factor = 1000000000000 if locale == "de-DE" else 1000000000
        else:
            factor = MAGNITUDES.get(word)
        if factor is None:
            raise Unresolved("unsupported_magnitude")
        scalar = magnitude[1].strip()
    from .normalize import CONTEXT
    with localcontext(CONTEXT):
        result = decimal_value(scalar, locale) * Decimal(factor)
        if not result.is_finite() or abs(result) > Decimal("1e100"):
            raise Unresolved("numeric_out_of_range")
        return result


def percent(value, locale):
    match = re.fullmatch(r"(" + NUMERIC + r")\s*(?:%|percent|per cent|Prozent)", value, re.I)
    if not match:
        raise Unresolved("invalid_percent")
    return decimal_value(match[1].strip(), locale)
