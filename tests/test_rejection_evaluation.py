from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tractor.document_sources import envelope
from tractor.episodes import Episode, native_episode
from tractor.pipeline_attribution import attribute, gold_probe
from tractor.pipeline_diagnostics import diagnostics
from tractor.pipeline_evaluation import episode_counts, infer, typed_requests_for
from tractor.rejection_annotations import adapt
from tractor.server import create_app
from tractor.typed_corpus import migrate


def fixture(text='DE45 5001 0517 5407 3249 31'):
    row = envelope('tractor_authored_documents', 'rejection:test', 'dev', 'invoices', 'de', text,
        [{'id': 'doc', 'kind': 'entities', 'description': 'Invoice', 'fields': {'iban': 'Supplier IBAN'}}],
        {'doc': {'complete': True, 'records': [{'iban': {'status': 'present',
            'items': [{'start': 0, 'end': len(text), 'text': text}]}}]}})
    return migrate(row)[0]


def test_correct_rejection_is_not_confused_with_missing_extraction():
    original = fixture()
    annotated, audit = adapt(original)
    assert audit[0]['expected_issues'] == ['invalid_iban']
    assert typed_requests_for(original) == typed_requests_for(annotated)
    assert native_episode(original) == native_episode(annotated)
    result = gold_probe(annotated)
    counts = episode_counts(annotated, result)
    assert counts['exact_requests'] == 1 and counts['correct_rejections'] == 1
    assert counts['field_tp'] == counts['answerable'] == 0  # Never inflate positive extraction F1.
    result['issues'][0]['code'] = 'not_found'
    counts = episode_counts(annotated, result)
    assert counts['exact_requests'] == 0 and counts['correct_rejections'] == 0
    assert counts['issue_mismatches'] == 1


def test_actual_http_and_evaluation_share_rejection_outcome():
    episode, _ = adapt(fixture())
    class Backend:
        def candidates(self, request):
            if request.text != episode.text:
                return [] if request.anchor else [{}]
            return [{'iban': [s.model_dump() for s in episode.gold['doc'].records[0]['iban'].items]}]
    backend = Backend()
    request = typed_requests_for(episode)['doc']
    with TestClient(create_app(extractor=backend)) as client:
        result = client.post('/v1/extract', json=request.model_dump()).json()
    assert result == {'data': {'iban': []}, 'issues': [{'field': 'iban', 'code': 'invalid_iban'}]}
    assert episode_counts(episode, infer(backend, episode))['exact_requests'] == 1


@pytest.mark.parametrize('change', ['value', 'extra_issue', 'wrong_index', 'wrong_field', 'no_issue'])
def test_rejection_requires_empty_value_and_exact_scoped_issue(change):
    e, _ = adapt(fixture())
    result = gold_probe(e)
    if change == 'value':
        result['data']['doc']['iban'] = ['DE45500105175407324931']
    elif change == 'extra_issue':
        result['issues'].append(dict(result['issues'][0], code='invalid_span'))
    elif change == 'wrong_index':
        result['issues'][0]['index'] = 8
    elif change == 'wrong_field':
        result['issues'][0]['field'] = 'other'
    else:
        result['issues'] = []
    assert episode_counts(e, result)['exact_requests'] == 0


def test_absent_field_correct_not_found_is_a_success():
    row = fixture().model_dump()
    row['gold']['doc']['records'][0]['iban'] = {'status': 'absent'}
    e = Episode.model_validate(row)
    assert episode_counts(e, gold_probe(e))['exact_requests'] == 1


def test_annotation_cannot_bless_missing_or_unknown_as_rejection():
    row = fixture().model_dump()
    answer = row['gold']['doc']['records'][0]['iban']
    answer.update(expected_issues=['not_found'], final_values=[], final_annotation_missing=False)
    with pytest.raises(ValidationError, match='not correct rejection'):
        Episode.model_validate(row)
    answer.update(expected_issues=['invalid_iban'], final_annotation_missing=True)
    with pytest.raises(ValidationError, match='complete extraction outcome'):
        Episode.model_validate(row)


def test_diagnostics_and_attribution_use_same_issue_and_name_equality():
    e, _ = adapt(fixture())
    result = gold_probe(e)
    report = diagnostics([e], {e.id: {'prediction': result}}, minimum_support=1)
    group = 'invoices:doc:iban:real/human'
    assert report['rejection_checks'][group]['status'] == 'pass'
    assert attribute([e], {e.id: {'prediction': result}})['fields'][group]['correct'] == 1
    wrong = deepcopy(result)
    wrong['issues'][0]['code'] = 'not_found'
    report = diagnostics([e], {e.id: {'prediction': wrong}}, minimum_support=1)
    assert report['field_error_examples'][group][0]['diagnosis'] == 'issue_mismatch'
    assert attribute([e], {e.id: {'prediction': wrong}})['counts']['issue_mismatch'] == 1


def test_expected_issues_never_enter_inference_request():
    e, _ = adapt(fixture())
    before = typed_requests_for(e)
    e.gold = {}
    assert typed_requests_for(e) == before


def test_saved_prediction_attribution_preserves_decimal_values(tmp_path, monkeypatch):
    from decimal import Decimal

    from tractor.jsonio import dumps, loads
    from tractor.pipeline_attribution import main
    e = fixture('100').model_dump()
    e['processing']['doc']['iban'] = {'kind': 'number'}
    e['gold']['doc']['records'][0]['iban'].update(final_annotation_missing=False, final_values=['100'])
    e = Episode.model_validate(e)
    episodes, predictions, report = (tmp_path / n for n in ('episodes.jsonl', 'predictions.jsonl', 'report.json'))
    episodes.write_text(dumps(e.model_dump()) + '\n')
    predictions.write_text(dumps({'id': e.id, 'prediction': {
        'data': {'doc': {'iban': [Decimal('100.0')]}}, 'issues': []}}) + '\n')
    monkeypatch.setattr('sys.argv', ['attribution', '--episodes', str(episodes),
                                   '--predictions', str(predictions), '--output', str(report)])
    main()
    assert loads(report.read_text())['fields']['invoices:doc:iban:real/human']['correct'] == 1


def test_reordered_records_match_issue_to_predicted_record_not_gold_index():
    text = 'A DE45 5001 0517 5407 3249 31 B DE89 3704 0044 0532 0130 00'
    def span(start, end):
        return {'status': 'present', 'items': [{'text': text[start:end], 'start': start, 'end': end}]}
    e = envelope('tractor_authored_documents', 'records', 'dev', 'invoices', 'de', text,
        [{'id': 'rows', 'kind': 'records', 'description': 'Invoices', 'anchor': 'number',
          'fields': {'number': 'Invoice number', 'iban': 'IBAN'}}],
        {'rows': {'complete': True, 'records': [
            {'number': span(0, 1), 'iban': span(2, 29)},
            {'number': span(30, 31), 'iban': span(32, len(text))}]}})
    e = adapt(migrate(e)[0])[0]
    result = gold_probe(e)
    assert episode_counts(e, result)['exact_requests'] == 1
    result['data']['rows'].reverse()
    result['issues'][0]['index'] = 1
    assert episode_counts(e, result)['exact_requests'] == 1
    result['issues'][0]['index'] = 0
    assert episode_counts(e, result)['exact_requests'] == 0
