from copy import deepcopy

import pytest

from tractor.document_sources import envelope
from tractor.pipeline_attribution import attribute, gold_probe
from tractor.pipeline_contract_data import adapt


def fixture():
    text = 'A 24.10.2026 B 25.10.2026'
    def value(start, end):
        return {'status': 'present', 'items': [{'text': text[start:end], 'start': start, 'end': end}]}
    e = envelope('tractor_authored_documents', 'attribution', 'dev', 'metadata', 'de', text,
        [{'id': 'rows', 'kind': 'records', 'description': 'Dated records',
          'anchor': 'name', 'fields': {'name': 'Name', 'date': 'Date'}}],
        {'rows': {'complete': True, 'records': [
            {'name': value(0, 1), 'date': value(2, 12)},
            {'name': value(13, 14), 'date': value(15, 25)}]}})
    return adapt(e)[0]


def test_swapped_fields_are_binding_symptom_and_probe_is_separate():
    e = fixture()
    prediction = gold_probe(e)
    a, b = prediction['data']['rows']
    a['date'], b['date'] = b['date'], a['date']
    original = deepcopy(prediction)
    report = attribute([e], {e.id: {'prediction': prediction}})
    assert report['counts']['correct_value_in_other_record'] == 2
    assert prediction == original
    field = report['fields']['metadata:rows:date:real/human']
    assert field['gold_evidence_probe_pass'] == 2


def test_unmatched_anchor_not_misreported_as_missing_field():
    e = fixture()
    prediction = gold_probe(e)
    prediction['data']['rows'][0]['name'] = ['Other']
    report = attribute([e], {e.id: {'prediction': prediction}})
    assert report['counts']['unmatched_anchor'] == 2
    assert report['record_anchor_audit']['unmatched_gold_records'] == 1


def test_test_split_cannot_be_used_for_attribution():
    e = fixture().model_copy(update={'split': 'test'})
    with pytest.raises(ValueError, match='Dev-only'):
        attribute([e], {})


def test_reapplying_contract_preserves_independent_final_annotations():
    e = fixture()
    again, gaps = adapt(e)
    assert not gaps
    assert again.gold == e.gold


def test_reapplying_contract_keeps_missing_annotations_missing():
    e = fixture().model_dump()
    answer = e['gold']['rows']['records'][0]['date']
    answer['final_values'] = None
    answer['final_annotation_missing'] = True
    from tractor.episodes import Episode
    original = Episode.model_validate(e)
    again, _ = adapt(original)
    assert again.gold == original.gold
