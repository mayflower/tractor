"""Independent final-value cases, including library defaults we must reject."""
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tractor import Date, Datetime, Duration, Identifier, Interval, Money
from tractor.field_normalization import Unresolved, normalize_value
from tractor.jsonio import loads
from tractor.server import create_app
from tractor.types import ExtractRequest

VALID = [
    (Date('Date'), 'October 14th, 2026', '2026-10-14'),
    (Date('Date'), '14th of October 2026', '2026-10-14'),
    (Date('Date'), '29. Februar 2024', '2024-02-29'),
    (Date('Date'), 'October 1st, 2026', '2026-10-01'),
    (Datetime('Time'), 'October 14th, 2026 2:35 PM', '2026-10-14T14:35'),
    (Interval('Period'), 'October 1st, 2026 to October 14th, 2026', '2026-10-01/2026-10-14'),
    (Money('Price', currency='USD', locale='en-US'), 'US$ 1,250.00', Decimal('1250.00')),
    (Money('Price', currency='USD', locale='de-DE'), '1.250,00 US-Dollar', Decimal('1250.00')),
    (Money('Price', currency='CAD', locale='en-US'), 'CA$ 1,250.00', Decimal('1250.00')),
    (Money('Price', currency='AUD', locale='en-US'), 'A$ 1250.00', Decimal('1250.00')),
    (Money('Price', currency='AUD', locale='de-DE'), 'AU$ 1250,00', Decimal('1250.00')),
    (Money('Price', currency='GBP', locale='de-DE'), '12 Britische Pfund', Decimal(12)),
    (Money('Price', currency='CHF', locale='de-DE'), '12 Schweizer Franken', Decimal(12)),
    (Money('Price', currency='CAD', locale='de-DE'), '12 Kanadische Dollar', Decimal(12)),
    (Money('Price', currency='USD', locale='en-US'), '1.5 million US dollars', Decimal(1500000)),
    (Money('Price', currency='EUR', locale='de-DE'), '1.234,567890123456789 EUR', Decimal('1234.567890123456789')),
    (Duration('Term'), 'P1,5Y', 'P1.5Y'),
    (Duration('Term'), 'PT0.000000000123S', 'PT0.000000000123S'),
    (Duration('Term'), 'PT0.123456789012345678901234567890123456789S',
     'PT0.123456789012345678901234567890123456789S'),
    (Duration('Term'), 'P0D', 'P0D'),
    (Duration('Term'), 'P1Y0M', 'P1Y0M'),
    (Duration('Term'), 'P2W', 'P2W'),
    (Identifier('Book', format='isbn'), '978-0-471-11709-4', '9780471117094'),
    (Identifier('Book', format='isbn'), '0-306-40615-2', '0306406152'),
    (Identifier('Barcode', format='ean'), '4006381333931', '4006381333931'),
    (Identifier('VAT', format='vat', country='DE'), 'DE136695976', '136695976'),
    (Identifier('Reference'), 'DE136695976', 'DE136695976'),
]


@pytest.mark.parametrize('field,source,expected', VALID)
def test_library_normalization(field, source, expected):
    assert normalize_value(source, field.processing_rule()) == [expected]


@pytest.mark.parametrize('field,source', [
    (Date('Date'), 'about 14 October 2026'),
    (Date('Date'), 'not 14 October 2026'),
    (Date('Date'), 'before 14 October 2026'),
    (Date('Date'), '14 October 26'),
    (Date('Date'), 'October 2026'),
    (Date('Date'), 'October 14'),
    (Date('Date'), '32 December 2026'),
    (Date('Date'), '29 February 2026'),
    (Date('Date'), 'October 14th or 15th, 2026'),
    (Date('Date'), 'tomorrow'),
    (Date('Date'), '03/04/2026'),
    (Date('Date', date_order='MDY'), '14/10/2026'),
    (Money('Price', currency='USD', locale='en-US'), 'CA$ 1,250.00'),
    (Money('Price', currency='USD', locale='en-US'), 'about US$ 1,250.00'),
    (Money('Price', currency='USD', locale='en-US'), 'US$ 1,250.00 or 1,300.00'),
    (Money('Price', currency='USD'), 'US$ 1,250'),
    (Money('Price', currency='USD'), '1250'),
    (Money('Price', currency='EUR'), 'USD 12 EUR'),
    (Duration('Term'), 'P1.5Y0M'),
    (Duration('Term'), 'P2W1D'),
    (Duration('Term'), 'PT1e3S'),
    (Duration('Term'), 'PT1e1000000S'),
    (Duration('Term'), 'P1YT'),
    (Duration('Term'), '-P1D'),
    (Duration('Term'), 'PT1H1H'),
    (Identifier('Book', format='isbn'), '9780471117095'),
    (Identifier('Book', format='isbn'), '9780471117094 / 9789024538270'),
    (Identifier('Barcode', format='ean'), '4006381333932'),
    (Identifier('VAT', format='vat', country='DE'), 'DE136695977'),
    (Identifier('VAT', format='vat', country='DE'), 'FR136695976'),
    (Identifier('VAT', format='vat', country='ZZ'), '136695976'),
])
def test_library_inference_and_invalid_values_rejected(field, source):
    with pytest.raises(Unresolved):
        normalize_value(source, field.processing_rule())


@pytest.mark.parametrize('options', [
    {'format': 'vat'}, {'country': 'DE'}, {'format': 'isbn', 'country': 'DE'},
    {'format': 'unknown'}, {'format': 'isbn', 'uppercase': True},
    {'format': 'ean', 'remove_spaces': True},
])
def test_identifier_contract_is_explicit(options):
    with pytest.raises(ValidationError):
        Identifier('Identifier', **options)


def test_final_numeric_equality_preserves_identifier_semantics():
    from tractor.pipeline_evaluation import field_value_key
    for kind in ('number', 'money', 'quantity', 'percent'):
        assert field_value_key('1250.00', kind) == field_value_key(Decimal('1250'), kind)
        assert field_value_key('-0.00', kind) == field_value_key(Decimal(0), kind)
        assert field_value_key('1.25e3', kind) == field_value_key(Decimal('1250'), kind)
        assert field_value_key('1,250', kind) != field_value_key(Decimal('1250'), kind)
    for kind in ('identifier', 'literal'):
        assert field_value_key('0012', kind) != field_value_key('12', kind)
        assert field_value_key('1250.00', kind) != field_value_key('1250', kind)


def test_library_results_use_public_http_and_evaluation():
    from tractor.document_sources import envelope
    from tractor.pipeline_evaluation import infer, task_counts

    selected = VALID[-5:] + VALID[:2] + VALID[6:8] + VALID[16:19]
    text = ' | '.join(source for _, source, _ in selected)
    fields, row, finals = {}, {}, {}
    cursor = 0
    for i, (field, source, expected) in enumerate(selected):
        key = f'f{i}'
        start = text.index(source, cursor)
        cursor = start + len(source)
        fields[key] = field
        row[key] = [{'text': source, 'start': start, 'end': cursor}]
        finals[key] = expected

    class Backend:
        def candidates(self, request):
            if request.text == text:
                return [row]
            return [] if request.anchor else [{key: [] for key in request.fields}]

    request = ExtractRequest(text=text, fields=fields)
    with TestClient(create_app(extractor=Backend())) as http:
        result = http.post('/v1/extract', json=request.model_dump())
        assert result.status_code == 200
        assert loads(result.content) == {'data': finals, 'issues': []}

    episode = envelope('example', 'library-normalization', 'dev', 'metadata', 'en', text,
        [{'id': 'document', 'kind': 'entities', 'description': 'Explicit fields',
          'fields': {key: field.description for key, field in fields.items()}}],
        {'document': {'complete': True, 'records': [{key: {
            'status': 'present', 'items': row[key], 'final_values': [str(finals[key])],
        } for key in fields}]}})
    episode.processing = {'document': {key: field.processing_rule() for key, field in fields.items()}}
    predicted = infer(Backend(), episode)
    assert task_counts(episode, episode.tasks[0], predicted)['exact_fields'] == len(fields)
