from decimal import Decimal
from pathlib import Path

from tractor.document_build import plan_item
from tractor.document_facts import coherent_invoice, number
from tractor.jsonio import loads


def test_internal_invoice_arithmetic_precedes_rendering_and_preserves_task_schema():
    spec = loads(Path('schemas/extraction/invoices.json').read_text())
    item = plan_item(spec, 'train', 0, 42)
    updated = coherent_invoice(item)
    values = {f['path'].split('.')[-1]: f['value'] for f in updated['facts'] if not f['path'].startswith('records.')}
    assert number(values['net']) + number(values['tax']) == number(values['gross'])
    assert updated['tasks'] == item['tasks'] and updated['rows'] == item['rows']
    assert any(f['path'].endswith('line_total') for f in updated['facts'])
    assert len({f['marker'] for f in updated['facts']}) == len(updated['facts'])


def test_locale_amount_parsing_does_not_round_float():
    assert number('1.234,56 EUR') == Decimal('1234.56')
    assert number('EUR 1,234.56') == Decimal('1234.56')
