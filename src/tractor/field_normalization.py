"""Conservative normalizers for selected evidence. No role guessing or gold access."""
import re
from datetime import date, datetime
from functools import cache


@cache
def month_names():
    # Babel's pinned CLDR data supplies locale vocabulary. Import lazily so the
    # HTTP client does not need server dependencies merely to validate a request.
    from babel.dates import get_month_names

    names = {}
    for locale in ('de_DE', 'en_GB', 'en_US'):
        for width in ('wide', 'abbreviated'):
            for number, name in get_month_names(width=width, locale=locale).items():
                key = name.casefold().rstrip('.')
                if key in names and names[key] != number:
                    raise ValueError('Ambiguous month vocabulary')
                names[key] = number
    return names


class Unresolved(ValueError):
    pass


def named_date(value):
    """Parse complete DE/EN dates; verify every stated component and qualifier.

    dateparser's strict mode still accepts 'about' and two-digit years. These
    checks are our no-inference contract, not a second date parser.
    """
    import dateparser

    words = re.findall(r"[^\W\d_]+", value.casefold())
    months = month_names()
    named = [months[word] for word in words if word in months]
    numbers = re.findall(r"[0-9]+", value)
    if (len(value) > 128 or len(named) != 1 or len(numbers) != 2
            or any(word not in months and word not in {'st', 'nd', 'rd', 'th', 'of'} for word in words)
            or re.search(r"[^\w\s.,/-]", value) or '_' in value):
        raise Unresolved('ambiguous_or_unsupported_date')
    years = [part for part in numbers if len(part) == 4]
    days = [part for part in numbers if len(part) <= 2]
    if len(years) != 1 or len(days) != 1:
        raise Unresolved('ambiguous_year')
    parsed = dateparser.parse(value, languages=['de', 'en'], settings={
        'STRICT_PARSING': True, 'REQUIRE_PARTS': ['year', 'month', 'day'],
        'PARSERS': ['absolute-time'], 'SKIP_TOKENS': [],
    })
    if parsed is None or (parsed.year, parsed.month, parsed.day) != (int(years[0]), named[0], int(days[0])):
        raise Unresolved('invalid_date')
    return parsed.date().isoformat()


def normalize_value(text, rule, locale=None):
    """Return unique-list elements; raise a typed ambiguity/failure code."""
    value = text.strip()
    for prefix in rule.prefixes:
        if value.casefold().startswith(prefix.casefold()):
            value = value[len(prefix):].strip()
            break
    if not value:
        raise Unresolved('empty_value')
    if rule.kind in {'person', 'organization', 'place', 'phone', 'email', 'postcode', 'address'}:
        from .contact_normalization import normalize_contact
        return [normalize_contact(value, rule)]
    if rule.kind in {'time', 'datetime', 'interval', 'duration'}:
        from .temporal_normalization import clock, duration, interval, timestamp
        handler = {'time': lambda: clock(value), 'datetime': lambda: timestamp(value, rule.date_order),
                   'interval': lambda: interval(value, rule.date_order), 'duration': lambda: duration(value)}
        return [handler[rule.kind]()]
    if rule.kind in {'iban', 'url'}:
        from .identifier_normalization import iban, url
        return [iban(value) if rule.kind == 'iban' else url(value)]
    if rule.kind == 'percent':
        from .scalar_normalization import percent
        return [percent(value, rule.locale or locale)]
    if rule.kind == 'literal':
        return [value if rule.prefixes else text]
    if rule.kind == 'identifier':
        if rule.format:
            from .identifier_normalization import standard_identifier
            return [standard_identifier(value, rule.format, rule.country)]
        # Whitespace removal/case folding are explicit; punctuation and zeros stay.
        value = re.sub(r'\s+', '', value) if rule.remove_spaces else value
        return [value.upper() if rule.uppercase else value]
    if rule.kind == 'list':
        parts = re.split(r'\r?\n' if rule.separator == 'newline' else ';', value)
        if any(not p.strip() for p in parts):
            raise Unresolved('empty_list_item')
        return [p.strip() for p in parts]
    if rule.kind == 'date':
        from .temporal_normalization import split_datetime, timestamp
        if split_datetime(value):
            return [timestamp(value, rule.date_order).split('T')[0]]
        iso = re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})', value)
        if iso:
            year, month, day = map(int, iso.groups())
        else:
            months = month_names()
            if rule.allow_partial:
                numeric_month = re.fullmatch(r'(\d{1,2})[./](\d{4})', value)
                if numeric_month:
                    month, year = map(int, numeric_month.groups())
                    if not 1 <= month <= 12 or not 1 <= year <= 9999:
                        raise Unresolved('invalid_date')
                    return [f'{year:04d}-{month:02d}']
                partial = re.fullmatch(r'([A-Za-zäÄ]+\.?)\s+(\d{4})', value)
                if partial and months.get(partial[1].casefold().rstrip('.')) and 1 <= int(partial[2]) <= 9999:
                    return [f'{int(partial[2]):04d}-{months[partial[1].casefold().rstrip('.')]:02d}']
                if re.fullmatch(r'\d{4}(?:-\d{2})?', value):
                    parts = value.split('-')
                    if 1 <= int(parts[0]) <= 9999 and (len(parts) == 1 or 1 <= int(parts[1]) <= 12):
                        return [value]
            if any(c.isalpha() for c in value):
                return [named_date(value)]
            match = re.fullmatch(r'(\d{1,4})([./-])(\d{1,2})\2(\d{1,4})', value)
            if not match or not rule.date_order:
                raise Unresolved('ambiguous_or_unsupported_date')
            values = dict(zip(rule.date_order, (match[1], match[3], match[4]), strict=True))
            if len(values['Y']) != 4:
                raise Unresolved('ambiguous_year')
            fmt = match[2].join({'D': '%d', 'M': '%m', 'Y': '%Y'}[p] for p in rule.date_order)
            try:
                return [datetime.strptime(value, fmt).date().isoformat()]
            except ValueError as exc:
                raise Unresolved('invalid_date') from exc
        try:
            return [date(year, month, day).isoformat()]
        except ValueError as exc:
            raise Unresolved('invalid_date') from exc
    if rule.kind in {'number', 'money'}:
        from .scalar_normalization import count, money
        return [count(value, rule.locale or locale) if rule.kind == 'number' else
                money(value, rule.currency, rule.locale or locale)]
    raise Unresolved('unsupported_processing')
