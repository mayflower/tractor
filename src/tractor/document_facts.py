"""Internal monetary consistency before document authoring; never normalize extracted text."""

import re
from decimal import ROUND_HALF_UP, Decimal

NUMBER = re.compile(r'[-+]?\d[\d.,]*')


def number(value):
    matches = NUMBER.findall(value)
    if len(matches) != 1:
        raise ValueError('Expected one numeric fact')
    value = matches[0]
    if ',' in value and '.' in value:
        decimal = ',' if value.rfind(',') > value.rfind('.') else '.'
        value = value.replace('.' if decimal == ',' else ',', '').replace(',', '.')
    elif ',' in value:
        value = value.replace(',', '.' if len(value.rsplit(',', 1)[1]) != 3 else '')
    elif '.' in value and len(value.rsplit('.', 1)[1]) == 3:
        value = value.replace('.', '')
    return Decimal(value)


def money(value, language):
    result = f'{value.quantize(Decimal(".01"), rounding=ROUND_HALF_UP):,.2f}'
    if language == 'de':
        result = result.translate(str.maketrans(',.', '.,'))
    return result + ' EUR'


def coherent_invoice(item):
    if item['domain'] != 'invoices':
        return item
    facts = [dict(f) for f in item['facts']]
    by_path = {f['path']: f for f in facts}
    header = {f['path'].split('.')[-1]: f for f in facts if f['path'].startswith(('document.', 'context.'))}
    quantities, prices = {}, {}
    for f in facts:
        parts = f['path'].split('.')
        if len(parts) == 3 and parts[0] == 'records':
            if parts[-1] == 'quantity':
                quantities[parts[1]] = number(f['value'])
            if parts[-1] == 'unit_price':
                prices[parts[1]] = number(f['value'])
    if not quantities or quantities.keys() != prices.keys():
        raise ValueError('Invoice needs complete internal quantity and price facts')
    net = sum((quantities[k] * prices[k]).quantize(Decimal('.01'), rounding=ROUND_HALF_UP) for k in quantities)
    for key, value in prices.items():
        by_path[f'records.{key}.unit_price']['value'] = money(value, item['language'])
        facts.append({'marker': f'v{len(facts)}', 'path': f'records.{key}.line_total',
                      'description': 'Printed net line total (quantity times unit price)',
                      'value': money(quantities[key] * value, item['language'])})
    rate = number(header['tax_rate']['value'])
    if not 0 <= rate <= 25:
        raise ValueError('VAT rate outside this fictional invoice profile')
    tax = (net * rate / 100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    for key, value in [('net', net), ('tax', tax), ('gross', net+tax)]:
        if key in header:
            header[key]['value'] = money(value, item['language'])
    result = dict(item, facts=facts, monetary_policy=f'Fictional EUR invoice, {rate}% VAT; amounts computed before authoring. Use only supplied positions and amounts.')
    return result
