import pytest
from pydantic import ValidationError

from tractor.document_sources import answer, envelope
from tractor.document_training import regression_guard, selection_score
from tractor.episodes import Episode, schema_variant
from tractor.pipeline_evaluation import episode_counts, evaluate_pipeline, typed_requests_for


def example(text, target, equivalents=(), source='swde', field_description=None):
    e = envelope(source, 'auto:example:1', 'dev', 'products', 'en', text,
        [{'id': 'document', 'kind': 'entities', 'description': 'Web attributes',
          'fields': {'model': field_description or 'auto model explicitly written on the page'}}],
        {'document': {'complete': True, 'records': [
            {'model': answer(text, text.index(target), text.index(target) + len(target))}]}})
    row = e.model_dump()
    row['processing'] = {'document': {'model': {'kind': 'literal'}}}
    row['gold']['document']['records'][0]['model']['value_equivalences'] = list(equivalents)
    return Episode.model_validate(row)


def response(values, field='model', issues=()):
    return {'data': {'document': {field: values}}, 'issues': list(issues)}


@pytest.mark.parametrize('values', [
    ['2011 Nissan Versa Sedan 1.6'],
    ['2011 Nissan Versa Sedan 1.6 Base'],
    ['2011 Nissan Versa Sedan 1.6', '2011 Nissan Versa Sedan 1.6 Base'],
])
def test_reviewed_variants_are_equally_correct_without_deduplicating_api_output(values):
    target = '2011 Nissan Versa Sedan'
    e = example(target + ' 1.6 Base', target, [[target, target+' 1.6', target+' 1.6 Base']])
    counts = episode_counts(e, response(values))
    assert counts['exact_fields'] == counts['exact_requests'] == 1
    assert counts['fp'] == counts['fn'] == 0
    assert response(values)['data']['document']['model'] == values
    original = e.model_copy(deep=True)
    original.gold['document'].records[0]['model'].value_equivalences = []
    assert typed_requests_for(e) == typed_requests_for(original)


def test_optional_year_and_whitespace_variants_are_reviewed_not_global_year_stripping():
    target = '2010 Toyota Venza'
    e = example(target + '\nToyota\nVenza', target, [[target, 'Toyota Venza']])
    c = episode_counts(e, response(['Toyota Venza', 'Toyota\nVenza']))
    assert c['field_tp'] == 1 and c['fp'] == c['fn'] == 0
    strict = example(target + '\nToyota Venza', target, source='authored',
                     field_description='Full model designation including year')
    c = episode_counts(strict, response(['Toyota Venza']))
    assert c['field_fn'] == c['field_fp'] == 1


def test_incomplete_reference_does_not_make_unreviewed_output_a_failure_or_a_success():
    e = example('2010 Nissan Versa Sedan 1.8 SL', '2010 Nissan Versa Sedan')
    for value in ['2010 Nissan Versa Sedan 1.8 SL', 'Nissan', 'invented model']:
        c = episode_counts(e, response([value]))
        assert c['unresolved_reference_fields'] == c['unscored_fields'] == 1
        assert c['field_tp'] == c['field_fp'] == c['field_fn'] == 0
        assert c['exact_requests'] == c['scored_requests'] == 0


def test_empty_output_is_still_a_proven_failure():
    e = example('New 2010 Hyundai Tucson GLS Auto FWD', '2010 Hyundai Tucson')
    c = episode_counts(e, response([], issues=[
        {'task': 'document', 'field': 'model', 'index': 0, 'code': 'not_found'}]))
    assert c['field_fn'] == 1 and c['scored_requests'] == 1
    assert c['unscored_fields'] == c['exact_requests'] == 0


def test_incomplete_references_cannot_qualify_or_drive_model_selection():
    e = example('2010 Nissan Versa Sedan', '2010 Nissan Versa Sedan')
    policy = {'domains': ['products'], 'shapes': ['single_object'],
              'required_origins': ['real/human'], 'minimum_exclusive': .7,
              'minimum_field_support': 1, 'qualification': ''}
    report = evaluate_pipeline([e], {e.id: {'prediction': response(['2010 Nissan Versa Sedan'])}}, policy)
    assert not report['passed']
    assert all(v['score'] is None for v in report['checks'].values())
    score = report['field_scores']['products:document:model:real/human']
    assert score['field_f1'] is None
    assert score['reviewed_subset_scores']['field_f1'] == 1
    assert report['field_checks']['products:document:model:real/human']['score'] is None
    with pytest.raises(ValueError, match='No finite'):
        selection_score(report)
    assert not regression_guard(report, dict(report, evaluation_contract='old'))['passed']


def test_policy_survives_field_renaming():
    e = schema_variant(example('Nissan Versa 1.8', 'Nissan Versa'), 42)
    key = next(iter(e.tasks[0].fields))
    c = episode_counts(e, response(['Nissan Versa 1.8'], field=key))
    assert c['unresolved_reference_fields'] == 1


@pytest.mark.parametrize('groups', [
    [['Nissan Versa', 'not in source']],
    [['Nissan Versa', 'Nissan'], ['Nissan', 'Versa']],
    [['Nissan', 'Versa']],
])
def test_invalid_adjudication_is_rejected(groups):
    with pytest.raises(ValidationError):
        example('Nissan Versa', 'Nissan Versa', groups)


def test_wrong_values_and_issues_remain_failures_with_complete_references():
    e = example('Model A; Model B', 'Model A', source='authored',
                field_description='The first explicitly named model')
    c = episode_counts(e, response(['Model B']))
    assert c['field_fn'] == c['field_fp'] == 1
    c = episode_counts(e, response(['Model A'], issues=[
        {'task': 'document', 'field': 'model', 'code': 'ambiguous_value', 'index': 0}]))
    assert c['issue_mismatches'] == c['field_fn'] == 1
