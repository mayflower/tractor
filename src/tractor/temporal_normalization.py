"""Explicit temporal values only: no reference dates, inferred bounds or timezones."""
import re
from datetime import datetime, time
from decimal import Decimal, DecimalException, localcontext

from .field_normalization import Unresolved

CLOCK = r"\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d{1,6})?)?(?:\s*(?:[AaPp][Mm]|Uhr))?(?:\s*(?:Z|UTC|GMT|[+-]\d{2}:\d{2}))?"


def clock(value):
    value = value.strip()
    if not re.fullmatch(CLOCK, value):
        raise Unresolved("invalid_time")
    value = re.sub(r"\s*(UTC|GMT)$", "+00:00", value)
    value = re.sub(r"\s*Uhr", "", value)
    meridiem = re.search(r"\s*([AaPp][Mm])", value)
    if meridiem:
        hour = int(value.split(":")[0])
        if not 1 <= hour <= 12:
            raise Unresolved("invalid_time")
        hour = hour % 12 + (12 if meridiem[1].upper() == "PM" else 0)
        value = re.sub(r"^\d{1,2}", f"{hour:02d}", value)
        value = re.sub(r"\s*[AaPp][Mm]", "", value)
    value = re.sub(r"\s+", "", value).replace(",", ".")
    value = re.sub(r"^(\d):", r"0\1:", value)
    # Offset hours must stay within 23; datetime validates minutes and magnitude.
    try:
        parsed = time.fromisoformat(value)
        offset = re.search(r"([+-])(\d{2}):(\d{2})$", value)
        if offset and (int(offset[2]) > 23 or int(offset[3]) > 59):
            raise ValueError("invalid offset")
    except ValueError as exc:
        raise Unresolved("invalid_time") from exc
    local = re.split(r"Z|[+-]", value)[0]
    precision = "microseconds" if "." in local else "seconds" if local.count(":") == 2 else "minutes"
    return parsed.isoformat(timespec=precision)


def split_datetime(value):
    return re.fullmatch(r"(.+?)(?:T|,\s*|\s+)(" + CLOCK + r")", value.strip())


def timestamp(value, order):
    from .field_contracts import Processing
    from .field_normalization import normalize_value
    match = split_datetime(value)
    if not match:
        raise Unresolved("invalid_datetime")
    day = normalize_value(match[1], Processing(kind="date", date_order=order))[0]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise Unresolved("incomplete_datetime")
    return day + "T" + clock(match[2])


def interval(value, order):
    from .field_contracts import Processing
    from .field_normalization import normalize_value
    value = re.sub(r"^(?:from|von)\s+", "", value.strip(), flags=re.I)
    separators = list(re.finditer(r"\s+(?:to|bis|-)\s+|\s*[–—]\s*|/", value, re.I))
    if not 1 <= len(separators) <= 8:
        raise Unresolved("invalid_interval")

    def endpoint(part):
        part = part.strip()
        if split_datetime(part):
            return "datetime", timestamp(part, order)
        if re.fullmatch(CLOCK, part):
            return "time", clock(part)
        result = normalize_value(part, Processing(kind="date", date_order=order))[0]
        return "date", result

    choices = set()
    for separator in separators:
        try:
            kind_a, a = endpoint(value[:separator.start()])
            kind_b, b = endpoint(value[separator.end():])
            if kind_a != kind_b:
                continue
            if kind_a == "time":
                start, end = time.fromisoformat(a), time.fromisoformat(b)
            else:
                start, end = datetime.fromisoformat(a), datetime.fromisoformat(b)
            # Mixed explicit/absent timezones cannot be ordered without inference.
            if (start.tzinfo is None) != (end.tzinfo is None) or end < start:
                continue
            choices.add(a + "/" + b)
        except (ValueError, TypeError):
            continue
    if len(choices) != 1:
        raise Unresolved("ambiguous_or_invalid_interval")
    return choices.pop()


UNITS = {
    "year": "Y", "years": "Y", "jahr": "Y", "jahre": "Y", "jahren": "Y",
    "month": "M", "months": "M", "monat": "M", "monate": "M", "monaten": "M",
    "week": "W", "weeks": "W", "woche": "W", "wochen": "W",
    "day": "D", "days": "D", "tag": "D", "tage": "D", "tagen": "D",
    "hour": "TH", "hours": "TH", "stunde": "TH", "stunden": "TH", "h": "TH",
    "minute": "TM", "minutes": "TM", "minuten": "TM", "min": "TM",
    "second": "TS", "seconds": "TS", "sekunde": "TS", "sekunden": "TS", "s": "TS",
}

def iso_duration_parts(value):
    from isoduration import DurationParsingException, parse_duration

    # The library also accepts signs/exponents. Our API requests a nonnegative
    # stated duration in ordinary ISO notation, without calendar arithmetic.
    if not re.fullmatch(r'P[0-9YMWDTHS.,]+', value):
        raise Unresolved('invalid_duration')
    try:
        with localcontext() as context:
            context.prec = 256  # bounded input, preserve all decimal digits
            parsed = parse_duration(value)
    except (DurationParsingException, DecimalException, ValueError) as exc:
        raise Unresolved('invalid_duration') from exc
    day, _, clockpart = value.partition('T')
    # Retain explicitly stated zero components and units; no months -> days or
    # fractional hours -> seconds conversion. The library validates ordering.
    parts = {key: getattr(parsed.date, name) for key, name in
             [('Y', 'years'), ('M', 'months'), ('W', 'weeks'), ('D', 'days')] if key in day}
    parts.update({'T' + key: getattr(parsed.time, name) for key, name in
                  [('H', 'hours'), ('M', 'minutes'), ('S', 'seconds')] if key in clockpart})
    return parts


def duration(value):
    if len(value) > 256:
        raise Unresolved("invalid_duration")
    keys = ["Y", "M", "W", "D", "TH", "TM", "TS"]
    if value.startswith('P'):
        parts = iso_duration_parts(value)
    else:
        pattern = re.compile(r"(\d+(?:[.,]\d+)?)\s+(?:calendar\s+)?(" +
                             "|".join(sorted(UNITS, key=len, reverse=True)) + r")(?!\w)", re.I)
        parts = {}
        end = 0
        for match in pattern.finditer(value):
            gap = value[end:match.start()].strip()
            if gap and (end == 0 or gap.casefold() not in {",", "and", "und", ", and"}):
                raise Unresolved("invalid_duration")
            key = UNITS[match[2].casefold()]
            if key in parts:
                raise Unresolved("ambiguous_duration")
            parts[key] = Decimal(match[1].replace(",", "."))
            end = match.end()
        if value[end:].strip():
            raise Unresolved("invalid_duration")
    if not parts or ("W" in parts and len(parts) != 1):
        raise Unresolved("invalid_duration")
    # ISO fractions are allowed only on the smallest specified component.
    ordered = [k for k in keys if k in parts]
    if any(parts[k] != parts[k].to_integral_value() for k in ordered[:-1]):
        raise Unresolved("invalid_duration")
    def formatted(v):
        return format(v, "f").rstrip("0").rstrip(".") if v != v.to_integral_value() else str(int(v))
    datepart = "".join(formatted(parts[k]) + k for k in keys[:4] if k in parts)
    clockpart = "".join(formatted(parts[k]) + k[1] for k in keys[4:] if k in parts)
    return "P" + datepart + ("T" + clockpart if clockpart else "")
