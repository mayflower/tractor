from decimal import Decimal

import pytest
from pydantic import ValidationError

from tractor.document_pipeline import DocumentRequest, address_parts, resolve_document
from tractor.field_contracts import Processing
from tractor.field_normalization import Unresolved, normalize_value


@pytest.mark.parametrize('text,rule,expected', [
    ('1.234,50', {'kind': 'number', 'locale': 'de-DE'}, [Decimal('1234.50')]),
    ('1,234.50', {'kind': 'number', 'locale': 'en-US'}, [Decimal('1234.50')]),
    ('EUR 1.234,50', {'kind': 'money', 'currency': 'EUR', 'locale': 'de-DE'}, [Decimal('1234.50')]),
    ('-12.50 USD', {'kind': 'money', 'currency': 'USD', 'locale': 'en-US'}, [Decimal('-12.50')]),
    ('29.02.2024', {'kind': 'date', 'date_order': 'DMY'}, ['2024-02-29']),
    ('2024-02-29', {'kind': 'date'}, ['2024-02-29']),
    ('03/04/2024', {'kind': 'date', 'date_order': 'MDY'}, ['2024-03-04']),
    (' de 001-02 ', {'kind': 'identifier', 'remove_spaces': True, 'uppercase': True}, ['DE001-02']),
    ('Jason Friedberg\nAaron Seltzer', {'kind': 'list', 'separator': 'newline'}, ['Jason Friedberg', 'Aaron Seltzer']),
    ('Jones, Peter; Smith, Jane', {'kind': 'list', 'separator': 'semicolon'}, ['Jones, Peter', 'Smith, Jane']),
])
def test_explicit_contracts(text, rule, expected):
    assert normalize_value(text, Processing(**rule)) == expected


@pytest.mark.parametrize('text,rule', [
    ('1,234', {'kind': 'number'}),
    ('1-2', {'kind': 'number'}),
    ('approximately 12', {'kind': 'number'}),
    ('NaN', {'kind': 'number'}),
    ('1e10000', {'kind': 'number'}),
    ('12 EUR', {'kind': 'money', 'currency': 'USD'}),
    ('12', {'kind': 'money', 'currency': 'EUR'}),
    ('1 EUR 2', {'kind': 'money', 'currency': 'EUR'}),
    ('03/04/2024', {'kind': 'date'}),
    ('29.02.2023', {'kind': 'date', 'date_order': 'DMY'}),
    ('03/04/24', {'kind': 'date', 'date_order': 'DMY'}),
    ('May 2024', {'kind': 'date'}),
    ('Alice;;Bob', {'kind': 'list', 'separator': 'semicolon'}),
])
def test_ambiguity_and_unsupported_forms_remain_unresolved(text, rule):
    with pytest.raises(Unresolved):
        normalize_value(text, Processing(**rule))


@pytest.mark.parametrize('rule', [
    {'kind': 'date', 'unit': 'mm'}, {'kind': 'money'}, {'kind': 'list'},
    {'kind': 'address', 'country': 'GB', 'component': 'city', 'group': ''},
])
def test_contradictory_or_incomplete_contracts_rejected(rule):
    with pytest.raises(ValidationError):
        Processing(**rule)


def test_streets_without_numbers_and_no_city_guess():
    assert address_parts('Spencer Road', 'street') == {'street': 'Spencer Road'}
    assert address_parts('Spencer Road, London EC1A 9BU', 'street')['street'] == 'Spencer Road'
    assert 'city' not in address_parts('Spencer Road EC1A 9BU', 'street')


def test_scalar_ambiguity_preserves_evidence_and_never_picks_first():
    text = '001 002'
    req = DocumentRequest(text=text, tasks=[{'id': 'doc', 'kind': 'entities',
        'description': 'Numbers', 'fields': {'id': 'Registration identifier'}}],
        processing={'doc': {'id': {'kind': 'identifier', 'cardinality': 'optional_one'}}}, include_details=True)
    raw = {'entities': {'id': [{'text': '001', 'start': 0, 'end': 3}, {'text': '002', 'start': 4, 'end': 7}]}}
    result = resolve_document(req, raw)
    assert result['data']['doc']['id'] == []
    assert result['issues'][0]['code'] == 'ambiguous_value'
    assert len(result['details']['doc']['id']) == 2


def test_record_processing_does_not_merge_identical_records():
    text = 'A 12 EUR; A 12 EUR'
    req = DocumentRequest(text=text, tasks=[{'id': 'rows', 'kind': 'records', 'description': 'Payments',
        'fields': {'name': 'Name', 'amount': 'Amount'}, 'anchor': 'name'}],
        processing={'rows': {'amount': {'kind': 'money', 'currency': 'EUR'}}})
    raw = {'rows': [{'name': [{'text': 'A', 'start': i, 'end': i+1}],
                     'amount': [{'text': '12 EUR', 'start': i+2, 'end': i+8}]} for i in (0, 10)]}
    result = resolve_document(req, raw)
    assert result['data']['rows'] == [{'name': ['A'], 'amount': [Decimal(12)]}]*2


@pytest.mark.parametrize('text,expected', [('16. Okt. 2026', '2026-10-16'), ('Sept 7, 2025', '2025-09-07'), ('7 Sep 2025', '2025-09-07')])
def test_babel_month_vocabulary(text, expected):
    assert normalize_value(text, Processing(kind='date')) == [expected]


def test_explicit_source_label_is_not_part_of_value():
    rule = Processing(kind='literal', prefixes=['Phone:', ':'])
    assert normalize_value('Phone: 530-752-2971', rule) == ['530-752-2971']
    assert normalize_value('530-752-2971', rule) == ['530-752-2971']
    assert normalize_value('Customer: 530-752-2971', rule) == ['Customer: 530-752-2971']


def test_failed_normalization_keeps_original_evidence():
    text = '03/04/2026'
    req = DocumentRequest(text=text, tasks=[{'id': 'doc', 'kind': 'entities', 'description': 'Dates',
        'fields': {'date': 'Date'}}], processing={'doc': {'date': {'kind': 'date'}}}, include_details=True)
    evidence = {'text': text, 'start': 0, 'end': len(text)}
    result = resolve_document(req, {'entities': {'date': [evidence]}})
    assert result['data']['doc']['date'] == []
    assert result['issues'][0]['code'] == 'ambiguous_or_unsupported_date'
    assert result['details']['doc']['date'] == [evidence]
