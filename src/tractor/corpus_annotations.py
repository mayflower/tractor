"""Independent final-target annotation, never imports production normalizers.

This deliberately bounded annotation grammar is not a copy of inference parsing.
It uses original annotated spans, stdlib/validation libraries and explicit arithmetic.
Unsupported or invalid evidence stays an annotation gap, never an empty gold answer.
"""
import re
import unicodedata
from datetime import datetime, time
from decimal import Decimal, localcontext

from .pipeline_contract_data import oracle as original_date_oracle


def decimal_literal(text, locale=None):
    value = text.strip().replace('−', '-')
    if not re.fullmatch(r'[+-]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?', value):
        raise ValueError('unsupported_numeric_annotation')
    if ',' in value and '.' in value:
        decimal = ',' if value.rfind(',') > value.rfind('.') else '.'
        group = '.' if decimal == ',' else ','
        left, right = value.rsplit(decimal, 1)
        if not re.fullmatch(r'[+-]?\d{1,3}(?:'+re.escape(group)+r'\d{3})+', left):
            raise ValueError('invalid_numeric_grouping')
        value = left.replace(group, '') + '.' + right
    elif ',' in value or '.' in value:
        sep = ',' if ',' in value else '.'
        parts = value.split(sep)
        if len(parts) > 2:
            if not re.fullmatch(r'[+-]?\d{1,3}(?:'+re.escape(sep)+r'\d{3})+', value):
                raise ValueError('invalid_numeric_grouping')
            value = ''.join(parts)
        elif len(parts[1]) == 3 and 1 <= len(parts[0].lstrip('+-')) <= 3:
            if locale is None:
                raise ValueError('ambiguous_numeric_annotation')
            value = ''.join(parts) if (locale == 'de-DE') == (sep == '.') else '.'.join(parts)
        else:
            value = value.replace(',', '.')
    number = Decimal(value)
    if not number.is_finite() or abs(number) > Decimal('1e100'):
        raise ValueError('numeric_annotation_out_of_range')
    return number


def scalar_string(value):
    value = format(value, 'f')
    return value.rstrip('0').rstrip('.') if '.' in value else value


def date_annotation(value, rule):
    if rule.get('allow_partial'):
        for fmt in ('%m/%Y', '%m.%Y'):
            try:
                return datetime.strptime(value, fmt).strftime('%Y-%m')
            except ValueError:
                pass
    return original_date_oracle(value, 'date', date_order=rule.get('date_order'),
                                allow_partial=rule.get('allow_partial', False))


def time_annotation(value):
    value = value.strip().removesuffix(' Uhr').replace(' UTC', '+00:00').replace(' GMT', '+00:00')
    for fmt in ('%I:%M %p', '%I:%M:%S %p'):
        try:
            return datetime.strptime(value.upper(), fmt).strftime('%H:%M' if fmt == '%I:%M %p' else '%H:%M:%S')
        except ValueError:
            pass
    if not re.fullmatch(r'\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})?', value):
        raise ValueError('unsupported_time_annotation')
    parsed = time.fromisoformat(value)
    precision = 'microseconds' if '.' in value else 'seconds' if re.match(r'\d{2}:\d{2}:', value) else 'minutes'
    return parsed.isoformat(timespec=precision)


def timestamp_annotation(value, rule):
    match = re.fullmatch(r'(.+?)(?:T|,\s*|\s+)(\d{1,2}:\d{2}.*)', value)
    if not match:
        raise ValueError('unsupported_datetime_annotation')
    day = date_annotation(match[1], {**rule, 'allow_partial': False})
    return day + 'T' + time_annotation(match[2])


def interval_annotation(value, rule):
    pieces = re.split(r'\s+(?:to|bis|[-–—])\s+|[–—]', value)
    if len(pieces) != 2:
        raise ValueError('interval_requires_two_explicit_endpoints')
    first, last = [date_annotation(p.strip(), {**rule, 'allow_partial': False}) for p in pieces]
    if first > last:
        raise ValueError('reversed_interval_annotation')
    return first + '/' + last


def duration_annotation(value):
    # Annotate an explicit numeral and unit even if prose surrounds the duration.
    # No converting calendar months to days or calculating a date difference.
    match = re.fullmatch(r'(\d+)\s+(days?|Tage[n]?|weeks?|Wochen?|months?|Monate[n]?|years?|Jahre[n]?)', value, re.I)
    if not match:
        raise ValueError('unsupported_duration_annotation')
    unit = match[2].lower()
    code = 'D' if unit.startswith(('day', 'tag')) else 'W' if unit.startswith(('week', 'woch')) else 'M' if unit.startswith(('month', 'monat')) else 'Y'
    return f'P{int(match[1])}{code}'


def money_annotation(value, rule):
    aliases = {
        'EUR': r'€|EUR|Euros?', 'GBP': r'£|GBP|pounds?(?: sterling)?',
        'USD': r'USD|US dollars?|US\$|\$', 'CAD': r'CAD|Canadian dollars?|CA\$|\$',
        'AUD': r'AUD|Australian dollars?|AU\$|\$', 'CHF': r'CHF|Swiss francs?|Schweizer Franken',
    }
    value = re.sub(r'^(?:MSRP|Price|Preis|Amount|Betrag)\s*:?\s*', '', value, flags=re.I)
    value = re.sub(r'\s+(?:per (?:calendar )?(?:month|year|annum|day|hour|visit|inspection)|monthly|annually|monatlich|jährlich|pro (?:Monat|Jahr))$', '', value, flags=re.I)
    token = aliases[rule['currency']]
    match = re.fullmatch(r'(?:'+token+r')\s*(.+)|(.+?)\s*(?:'+token+r')', value, re.I)
    if not match:
        raise ValueError('unsupported_money_annotation')
    scalar = match[1] or match[2]
    factors = {'thousand': 1000, 'tausend': 1000, 'million': 10**6, 'millionen': 10**6,
               'millions': 10**6, 'mio.': 10**6, 'milliarden': 10**9, 'milliarde': 10**9}
    pieces = scalar.rsplit(' ', 1)
    factor = factors.get(pieces[-1].lower())
    if factor:
        scalar = pieces[0]
    with localcontext() as ctx:
        ctx.prec = 120
        return scalar_string(decimal_literal(scalar, rule.get('locale')) * (factor or 1))


def annotate(value, rule):
    kind = rule['kind']
    if kind == 'literal' and not rule.get('prefixes'):
        return [value]
    value = value.strip()
    for prefix in rule.get('prefixes', []):
        if value.casefold().startswith(prefix.casefold()):
            value = value[len(prefix):].strip()
            break
    if not value:
        raise ValueError('empty_annotation')
    if kind in {'person', 'organization', 'place', 'address'}:
        if any(unicodedata.category(c).startswith('C') and not c.isspace() for c in value):
            raise ValueError('invalid_annotation_character')
        if kind == 'address' and rule.get('component') == 'postcode':
            compact = ''.join(value.split()).upper()
            if not re.fullmatch(r'[A-Z]{1,2}\d[A-Z\d]?\d[A-Z]{2}', compact):
                raise ValueError('invalid_gb_postcode_annotation')
            return [compact[:-3] + ' ' + compact[-3:]]
        if kind == 'address' and rule.get('component', 'full') == 'full':
            value = ', '.join(line.strip() for line in value.splitlines() if line.strip())
            value = re.sub(r'\s*,\s*', ', ', value).strip(' ,')
        result = unicodedata.normalize('NFC', ' '.join(value.split()))
        if not result or (kind != 'address' and not any(c.isalpha() for c in result)):
            raise ValueError('invalid_name_annotation')
        return [result]
    if kind in {'literal', 'identifier'}:
        if rule.get('remove_spaces'):
            value = ''.join(value.split())
        return [value.upper() if rule.get('uppercase') else value]
    if kind == 'list':
        values = value.splitlines() if rule['separator'] == 'newline' else value.split(';')
        if any(not v.strip() for v in values):
            raise ValueError('empty_list_annotation')
        return [v.strip() for v in values]
    if kind == 'date':
        return [date_annotation(value, rule)]
    if kind == 'datetime':
        return [timestamp_annotation(value, rule)]
    if kind == 'time':
        return [time_annotation(value)]
    if kind == 'interval':
        return [interval_annotation(value, rule)]
    if kind == 'duration':
        return [duration_annotation(value)]
    if kind == 'money':
        return [money_annotation(value, rule)]
    if kind == 'number':
        source = value
        # An explicitly labelled n is the annotation, not a calculation from
        # subgroups. Multiple n labels or an arithmetic expression remain open.
        labelled = re.fullmatch(r'n\s*=\s*(\d+)(?:\s+([A-Za-zÄÖÜäöü(].*))?', value, re.I)
        uncertain = re.search(r'\b(?:or|oder|and|und|plus|minus|to|bis|about|around|approximately|ca|rund|estimate[ds]?)\b', value, re.I)
        if labelled and not uncertain and len(re.findall(r'\bn\s*=', value, re.I)) == 1:
            return [str(int(labelled[1]))]
        value = re.sub(r'^(?:n\s*=|(?:sample size|Stichprobengröße|headcount|employees|Mitarbeitende)\s*:)\s*', '', value, flags=re.I)
        value = re.sub(r'\s+(?:participants?|subjects?|employees?|staff|people|persons?|units?|(?:pouch )?cells?|specimens?|samples?|soil cores|Teilnehmende|Teilnehmer|Mitarbeitende|Mitarbeiter(?:innen)?|Beschäftigte|Personen|Proben|Probanden|Messpunkte|Probenflächen|Einheiten|Stück|Stk\.?|cartons?|packages?|Kolli|Kanister|Set|reams|cartridges)$', '', value, flags=re.I)
        number = decimal_literal(value, rule.get('locale'))
        if source != value and (number < 0 or number != number.to_integral_value()):
            raise ValueError('invalid_count_annotation')
        return [scalar_string(number)]
    if kind == 'percent':
        match = re.fullmatch(r'(.+?)\s*(?:%|percent|Prozent)', value, re.I)
        if not match:
            raise ValueError('unsupported_percent_annotation')
        return [scalar_string(decimal_literal(match[1], rule.get('locale')))]
    if kind == 'quantity':
        # Independent SI arithmetic, not Pint or tractor's conversion map.
        factors = {'kg': ('kg', '1'), 'g': ('kg', '.001'), 'mg': ('kg', '.000001'),
                   'lb': ('kg', '.45359237'), 'lbs': ('kg', '.45359237'),
                   'm': ('m', '1'), 'cm': ('m', '.01'), 'mm': ('m', '.001'),
                   'ft': ('m', '.3048'), 'in': ('m', '.0254')}
        match = re.fullmatch(r'([+-]?[\d.,]+)\s*([A-Za-z]+)', value)
        if not match or match[2] not in factors or factors[match[2]][0] != rule['unit']:
            raise ValueError('unsupported_quantity_annotation')
        with localcontext() as ctx:
            ctx.prec = 120
            return [scalar_string(decimal_literal(match[1], rule.get('locale')) * Decimal(factors[match[2]][1]))]
    if kind == 'iban':
        value = re.sub(r'^IBAN\s*:\s*', '', value, flags=re.I)
        value = ''.join(value.split()).upper()
        # Independent ISO 13616 mod-97 check and supported national BBAN shapes.
        patterns = {'DE': r'DE\d{20}', 'GB': r'GB\d{2}[A-Z]{4}\d{14}',
                    'FR': r'FR\d{12}[A-Z0-9]{11}\d{2}', 'NL': r'NL\d{2}[A-Z]{4}\d{10}'}
        if not re.fullmatch(patterns.get(value[:2], r'(?!)'), value):
            raise ValueError('unsupported_or_invalid_iban_annotation')
        digits = ''.join(str(ord(c) - 55) if c.isalpha() else c for c in value[4:] + value[:4])
        if int(digits) % 97 != 1:
            raise ValueError('invalid_iban_checksum_annotation')
        return [value]
    if kind == 'phone':
        import phonenumbers
        if not value.startswith('+') and not rule.get('country'):
            raise ValueError('phone_annotation_requires_country')
        if not re.fullmatch(r'[+()0-9\s./-]+(?:\s*(?:ext\.?|x|#|;ext=)\s*\d+)?', value, re.I):
            raise ValueError('unsupported_phone_annotation')
        try:
            number = phonenumbers.parse(value, rule.get('country'))
        except phonenumbers.NumberParseException as exc:
            raise ValueError('invalid_phone_annotation') from exc
        if not phonenumbers.is_valid_number(number):
            raise ValueError('invalid_phone_annotation')
        # Assemble from parsed components; keep national leading zeros/extensions.
        national = phonenumbers.national_significant_number(number)
        return ['+' + str(number.country_code) + national + (';ext=' + number.extension if number.extension else '')]
    if kind == 'email':
        from email_validator import validate_email
        return [validate_email(value, check_deliverability=False).normalized]
    if kind == 'url':
        import idna
        match = re.fullmatch(r'((?i:https?://)|//)?([^/?#:]+)(:\d+)?([/?#].*)?', value)
        if not match or re.search(r'\s|[\\\x00-\x1f\x7f]|%(?![0-9a-fA-F]{2})', value):
            raise ValueError('unsupported_url_annotation')
        if '.' not in match[2] or (match[3] and not 1 <= int(match[3][1:]) <= 65535):
            raise ValueError('invalid_url_annotation')
        host = idna.encode(match[2].lower(), uts46=True).decode('ascii')
        return [(match[1] or '').lower() + host + (match[3] or '') + (match[4] or '')]
    raise ValueError('unsupported_annotation_kind')
