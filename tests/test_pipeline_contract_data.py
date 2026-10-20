from copy import deepcopy
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tractor.document_sources import envelope
from tractor.episodes import Episode, schema_variant
from tractor.field_contracts import Processing
from tractor.field_normalization import normalize_value
from tractor.pipeline_contract_data import adapt, oracle
from tractor.pipeline_diagnostics import diagnostics
from tractor.pipeline_evaluation import episode_counts


def example(text='24.10.2026'):
    return envelope('tractor_authored_documents', 'test', 'dev', 'metadata', 'de', text,
        [{'id': 'doc', 'kind': 'entities', 'description': 'Document date', 'fields': {'date': 'Document date'}}],
        {'doc': {'complete': True, 'records': [{'date': {'status': 'present',
            'items': [{'start': 0, 'end': len(text), 'text': text}]}}]}})


def test_independent_final_targets_preserve_spans_and_schema_variants():
    e, gaps = adapt(example())
    assert not gaps
    assert e.gold['doc'].records[0]['date'].final_values == ['2026-10-24']
    assert e.gold['doc'].records[0]['date'].items[0].text == '24.10.2026'
    renamed = schema_variant(e, 4)
    assert set(renamed.processing['doc']) == set(renamed.tasks[0].fields)
    assert episode_counts(e, {'data': {'doc': {'date': ['2026-10-24']}}, 'issues': []})['exact_requests'] == 1


def test_request_contract_does_not_depend_on_gold_support():
    complete, _ = adapt(example())
    unavailable, gaps = adapt(example('next week'))
    assert complete.processing == unavailable.processing
    assert gaps and unavailable.gold['doc'].records[0]['date'].final_annotation_missing
    score = episode_counts(unavailable, {'data': {'doc': {'date': []}}, 'issues': []})
    assert not score['exact_requests']
    assert score['unscored_fields'] == 1


def test_missing_independent_target_rejected_for_normalized_fields():
    e = example().model_dump()
    e['processing'] = {'doc': {'date': {'kind': 'date'}}}
    with pytest.raises(ValidationError, match='independently annotated'):
        Episode.model_validate(e)


def test_partial_date_preserves_precision():
    assert normalize_value('March 2025', Processing(kind='date', allow_partial=True)) == ['2025-03']
    assert oracle('March 2025', 'date', allow_partial=True) == '2025-03'
    assert normalize_value('2025', Processing(kind='date', allow_partial=True)) == ['2025']
    assert oracle('October 16, 2026', 'date') == '2026-10-16'


def test_numeric_expected_uses_decimal_without_rounding():
    e = example('100').model_dump()
    e['processing'] = {'doc': {'date': {'kind': 'number'}}}
    e['gold']['doc']['records'][0]['date']['final_values'] = ['100']
    e = Episode.model_validate(e)
    assert episode_counts(e, {'data': {'doc': {'date': [Decimal('1E2')]}}, 'issues': []})['exact_requests'] == 1


def test_per_field_reporting_exposes_missing_wrong_values_and_small_support():
    e, _ = adapt(example())
    other = deepcopy(e)
    other.id += '-other'
    predictions = {e.id: {'prediction': {'data': {'doc': {'date': []}}}},
                   other.id: {'prediction': {'data': {'doc': {'date': ['2025-01-01']}}}}}
    result = diagnostics([e, other], predictions, minimum_support=2)
    field = next(iter(result['field_scores'].values()))
    assert field['counts']['missing_value'] == 1
    assert field['counts']['wrong_or_incomplete_value'] == 1
    assert next(iter(result['field_checks'].values()))['status'] == 'fail'
    small = diagnostics([e], {e.id: predictions[e.id]})
    assert next(iter(small['field_checks'].values()))['status'] == 'insufficient_evidence'


def test_source_cell_labels_are_corrected_without_changing_document():
    e = envelope('swde', 'university:test:1', 'dev', 'contacts', 'en', 'Phone: 530-752-2971',
        [{'id': 'doc', 'kind': 'entities', 'description': 'Contact', 'fields': {'phone': 'Phone number'}}],
        {'doc': {'complete': True, 'records': [{'phone': {'status': 'present',
            'items': [{'start': 0, 'end': 19, 'text': 'Phone: 530-752-2971'}]}}]}})
    converted, audit = adapt(e)
    target = converted.gold['doc'].records[0]['phone']
    assert converted.text == e.text
    assert target.items[0].start == 7
    assert target.items[0].text == '530-752-2971'
    assert target.final_values == ['530-752-2971']
    assert audit[0]['before']['text'] == 'Phone: 530-752-2971'
    from tractor.document_pipeline import resolve_document
    from tractor.pipeline_evaluation import request_for
    raw = {'entities': {'phone': [{'start': 0, 'end': 19, 'text': e.text}]}}
    result = resolve_document(request_for(converted), raw)
    assert result['data']['doc']['phone'] == ['530-752-2971']
    assert episode_counts(converted, result)['exact_requests'] == 1


def test_movie_source_boundary_and_inference_contract_agree_without_gold_at_runtime():
    text = 'Sphere (1998)'
    e = envelope('swde', 'movie:imdb:boundary', 'train', 'metadata', 'en', text,
        [{'id': 'doc', 'kind': 'entities', 'description': 'Film', 'fields': {'title': 'Film title'}}],
        {'doc': {'complete': True, 'records': [{'title': {'status': 'present',
            'items': [{'start': 0, 'end': 6, 'text': 'Sphere'}]}}]}})
    revised, audit = adapt(e)
    assert not revised.processing['doc']['title'].include_adjacent_year
    assert revised.gold['doc'].records[0]['title'].items[0].text == 'Sphere'
    assert revised.gold['doc'].records[0]['title'].final_values == ['Sphere']
    assert revised.gold['doc'].records[0]['title'].value_equivalences == [['Sphere', text]]
    assert not audit
    twice, audit = adapt(revised)
    assert not audit
    assert twice == revised


def test_title_boundary_repair_keeps_distinct_years_for_equal_bare_titles():
    text = 'Sphere (1998); Sphere (2000)'
    e = envelope('swde', 'movie:imdb:two-years', 'train', 'metadata', 'en', text,
        [{'id': 'doc', 'kind': 'entities', 'description': 'Films', 'fields': {'title': 'Displayed titles'}}],
        {'doc': {'complete': True, 'records': [{'title': {'status': 'present',
            'items': [{'start': 0, 'end': 13, 'text': 'Sphere (1998)'},
                      {'start': 15, 'end': 28, 'text': 'Sphere (2000)'}],
            'final_values': ['Sphere (1998)', 'Sphere (2000)']}}]}})
    revised, audit = adapt(e)
    assert not audit
    assert not revised.gold['doc'].records[0]['title'].value_equivalences
    assert len(revised.gold['doc'].records[0]['title'].items) == 2
    assert revised.gold['doc'].records[0]['title'].final_values == ['Sphere (1998)', 'Sphere (2000)']
    assert not episode_counts(revised, {'data': {'doc': {'title': ['Sphere']}}})['exact_fields']
