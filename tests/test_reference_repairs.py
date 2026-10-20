import hashlib

import pytest

from tractor.document_sources import answer, envelope
from tractor.pipeline_evaluation import episode_counts, typed_requests_for
from tractor.reference_repairs import acceptance_context, apply_accepted_answers, apply_reviews
from tractor.service import resolve
from tractor.types import ExtractRequest


def example(text, target, kind='literal'):
    e = envelope('authored', 'review:1', 'dev', 'metadata', 'en', text,
        [{'id': 'document', 'kind': 'entities', 'description': 'Explicit fields',
          'fields': {'value': 'Named value'}}],
        {'document': {'complete': True, 'records': [{'value': answer(text, text.index(target), text.index(target)+len(target))}]}})
    # Validate using serialization so tests use real contracts.
    from tractor.episodes import Episode
    row = e.model_dump()
    row['processing'] = {'document': {'value': {'kind': kind}}}
    row['gold']['document']['records'][0]['value']['final_values'] = [target]
    return Episode.model_validate(row)


def decision(e, actual, category='book_title_variants', conclusion='semantic_false_negative'):
    return dict(id=e.id, task='document', record=0, field='value', category=category,
                conclusion=conclusion, source_sha256=hashlib.sha256(e.text.encode()).hexdigest(),
                raw_expected=[e.gold['document'].records[0]['value'].final_values], raw_actual=actual)


def test_adjudication_changes_scoring_without_changing_request_or_runtime():
    e = example('Rocket Boys: A Memoir', 'Rocket Boys')
    repaired, _ = apply_reviews(e, [decision(e, ['Rocket Boys: A Memoir'])])
    assert typed_requests_for(e) == typed_requests_for(repaired)
    for values in [['Rocket Boys'], ['Rocket Boys: A Memoir'], ['Rocket Boys', 'Rocket Boys: A Memoir']]:
        assert episode_counts(repaired, {'data': {'document': {'value': values}}})['exact_fields'] == 1
    for values in [['Rocket'], ['invented'], []]:
        assert episode_counts(repaired, {'data': {'document': {'value': values}}})['exact_fields'] == 0
    with pytest.raises(ValueError, match='source changed'):
        apply_reviews(e.model_copy(update={'text': e.text+'x'}), [decision(e, ['Rocket Boys: A Memoir'])])


def test_two_authors_are_two_values_not_an_equivalence_group():
    e = example('Authors: Dr. Helena Vogt, Dr. Marcus Oyelaran', 'Dr. Helena Vogt, Dr. Marcus Oyelaran', 'person')
    values = ['Dr. Helena Vogt', 'Dr. Marcus Oyelaran']
    repaired, _ = apply_reviews(e, [decision(e, values, 'people_merged_in_gold', 'annotation_error')])
    a = repaired.gold['document'].records[0]['value']
    assert a.final_values == values
    assert [s.text for s in a.items] == values
    assert not a.value_equivalences
    assert episode_counts(repaired, {'data': {'document': {'value': values}}})['exact_fields'] == 1
    assert not episode_counts(repaired, {'data': {'document': {'value': [', '.join(values)]}}})['exact_fields']


def test_meaningful_repetition_stays_significant():
    e = example('America America', 'America America')
    assert not episode_counts(e, {'data': {'document': {'value': ['America']}}})['exact_fields']


def acceptance(e, alternatives):
    return dict(id=e.id, task='document', record=0, field='value', audit_id='review-1',
                conclusion='semantic_equivalent',
                context_sha256=acceptance_context(e, 'document', 0, 'value'),
                approved_alternatives=alternatives)


def test_accepted_answer_sets_preserve_completeness_and_issue_checks():
    e = example('Cuisine: Asian, Sushi. Browse Asian restaurants.', 'Asian, Sushi')
    review = acceptance(e, [['Asian', 'Asian, Sushi']])
    fixed, changes = apply_accepted_answers(e, [review])
    assert changes
    assert typed_requests_for(fixed) == typed_requests_for(e)
    assert fixed.gold['document'].records[0]['value'].items == e.gold['document'].records[0]['value'].items
    for values, correct in [(['Asian, Sushi'], True), (['Asian', 'Asian, Sushi'], True),
                            (['Asian'], False), ([], False), (['French'], False)]:
        c = episode_counts(fixed, {'data': {'document': {'value': values}}})
        assert bool(c['exact_fields']) == correct
    c = episode_counts(fixed, {'data': {'document': {'value': ['Asian', 'Asian, Sushi']}},
                              'issues': [{'task': 'document', 'field': 'value',
                                          'index': 0, 'code': 'ambiguous_value'}]})
    assert c['issue_mismatches'] == 1 and c['exact_fields'] == 0
    twice, changes = apply_accepted_answers(fixed, [review])
    assert twice == fixed and changes == []


def test_reviewed_identifier_variants_do_not_make_distinct_codes_equivalent():
    e = example('NYSE: HRVL; other ticker OTHER', 'NYSE: HRVL', 'identifier')
    fixed, _ = apply_accepted_answers(e, [acceptance(e, [['HRVL'], ['NYSE: HRVL', 'HRVL']])])
    for values, correct in [(['HRVL'], True), (['NYSE: HRVL', 'HRVL'], True),
                            (['OTHER'], False), (['HRVL', 'OTHER'], False), (['HR'], False)]:
        assert bool(episode_counts(fixed, {'data': {'document': {'value': values}}})['exact_fields']) == correct


@pytest.mark.parametrize('mutation', ['text', 'description', 'type', 'evidence'])
def test_accepted_answers_fail_closed_when_source_or_contract_changes(mutation):
    e = example('Author: Dr. Helena Vogt', 'Helena Vogt', 'person')
    review = acceptance(e, [['Dr. Helena Vogt']])
    altered = e.model_copy(deep=True)
    if mutation == 'text':
        altered.text += ' changed'
    elif mutation == 'description':
        altered.tasks[0].fields['value'] = 'A different person'
    elif mutation == 'type':
        altered.processing['document']['value'].kind = 'organization'
    else:
        altered.gold['document'].records[0]['value'].items[0].start += 1
    with pytest.raises(ValueError, match='source or field contract changed'):
        apply_accepted_answers(altered, [review])


def test_identifier_label_repair_preserves_code_and_native_offsets():
    e = example('Our ref: HVPS/2026/0417', 'Our ref: HVPS/2026/0417', 'identifier')
    row = e.model_dump()
    row.update(source='tractor_authored_documents', family='metadata')
    row['tasks'][0]['fields'] = {'case': 'File reference'}
    row['processing']['document'] = {'case': row['processing']['document']['value']}
    row['gold']['document']['records'][0] = {'case': row['gold']['document']['records'][0]['value']}
    from tractor.episodes import Episode
    fixed, _ = apply_reviews(Episode.model_validate(row), [])
    a = fixed.gold['document'].records[0]['case']
    assert a.final_values == ['HVPS/2026/0417']
    assert fixed.text[a.items[0].start:a.items[0].end] == 'HVPS/2026/0417'
    request = typed_requests_for(fixed)['document']
    output = resolve(request, [{'case': [{'start': 0, 'end': len(e.text), 'text': e.text}]}])
    assert output['data']['case'] == ['HVPS/2026/0417']
    assert not output['issues']


@pytest.mark.parametrize('separator,joined', [('\n', True), ('\n\n', False), ('\nOther office\n', False)])
def test_address_annotation_uses_source_attribute_without_runtime_parser(separator, joined, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Production parser must not manufacture gold')
    monkeypatch.setattr('tractor.address_normalization.parsed_parts', forbidden)
    text = '503 SW Alder St.' + separator + 'Portland, OR 97204'
    e = envelope('swde', 'restaurant:opentable:review', 'dev', 'contacts', 'en', text,
        [{'id': 'document', 'kind': 'entities', 'description': 'Restaurant attributes',
          'fields': {'address': 'Restaurant address'}}],
        {'document': {'complete': True, 'records': [{'address': {
            'status': 'present', 'items': [dict(start=0, end=16, text=text[:16]),
                dict(start=text.index('Portland'), end=len(text), text='Portland, OR 97204')],
            'final_values': ['503 SW Alder St.', 'Portland, OR 97204']}}]}})
    row = e.model_dump()
    row['processing'] = {'document': {'address': {'kind': 'address'}}}
    from tractor.episodes import Episode
    fixed, _ = apply_reviews(Episode.model_validate(row), [])
    a = fixed.gold['document'].records[0]['address']
    assert len(a.items) == (1 if joined else 2)
    assert len(a.final_values) == (1 if joined else 2)
    twice, _ = apply_reviews(fixed, [])
    assert twice == fixed


@pytest.mark.parametrize('text,fragments,expected', [
    ('5840 Legacy Circle\nSuite D100\nPlano, TX 75024',
     ['5840 Legacy Circle\nSuite', 'D100\nPlano, TX 75024'],
     ['5840 Legacy Circle, Suite D100, Plano, TX 75024']),
    ('20753 N. Pima Road\nBldg. R\nScottsdale, AZ 85255',
     ['20753 N. Pima Road\nBldg. R', 'Scottsdale, AZ 85255'],
     ['20753 N. Pima Road, Bldg. R, Scottsdale, AZ 85255']),
    ('10 Downing Street, London\n11 High Street, Oxford',
     ['10 Downing Street, London', '11 High Street, Oxford'],
     ['10 Downing Street, London', '11 High Street, Oxford']),
    ('10 Downing Street\n\nLondon', ['10 Downing Street', 'London'], ['10 Downing Street', 'London']),
    ('10 Downing Street\nUnselected company\nLondon', ['10 Downing Street', 'London'], ['10 Downing Street', 'London']),
])
def test_address_fragments_through_public_service(text, fragments, expected):
    request = ExtractRequest(text=text, fields={'address': {'type': 'address', 'description': 'Full address',
                                                          'cardinality': 'unique_list'}})
    spans = [{'start': text.index(v), 'end': text.index(v)+len(v), 'text': v} for v in fragments]
    output = resolve(request, [{'address': spans}])
    assert output['data']['address'] == expected
    assert not output['issues']
