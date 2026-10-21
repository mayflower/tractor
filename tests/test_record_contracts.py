from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tractor.api_training import examples
from tractor.document_sources import envelope
from tractor.model import schema_for
from tractor.pipeline_evaluation import typed_requests_for
from tractor.record_contracts import cardinality, migrate
from tractor.service import resolve
from tractor.types import ExtractRequest, Person, Quantity, Text


def span(text, value, start=0):
    offset = text.index(value, start)
    return {'start': offset, 'end': offset + len(value), 'text': value}


def contact_episode():
    text = 'Anna, Einkauf. Max, Vertrieb.'
    return envelope('example', 'record-contract-test', 'train', 'contacts', 'de', text,
        [{'id': 'contacts', 'kind': 'records', 'description': 'Contact details',
          'fields': {'name': 'Person', 'role': 'Function of this person'}, 'anchor': 'name',
          'cardinalities': {'name': 'required_one', 'role': 'optional_one'}}],
        {'contacts': {'complete': True, 'records': [
            {key: {'status': 'present', 'items': [span(text, value)]} for key, value in row.items()}
            for row in [{'name': 'Anna', 'role': 'Einkauf'}, {'name': 'Max', 'role': 'Vertrieb'}]]}})


def test_legacy_default_cannot_override_declared_scalar_in_any_policy_branch():
    from tractor.field_contracts import Processing
    from tractor.typed_corpus_policy import choose
    e = contact_episode()
    e.processing = {'contacts': {'role': Processing(kind='literal')}}
    request = typed_requests_for(e)['contacts']
    assert request.fields['role'].cardinality == 'optional_one'
    schema = dict(source='tractor_authored_documents', source_group='test', family='contacts',
                  text=e.text, language='de', locale=None)
    task = e.tasks[0].model_dump()
    for rule in ({'kind': 'literal'}, {'kind': 'literal', 'cardinality': 'unique_list'}, {'kind': 'person'}):
        new, _ = choose(schema, task, 'role', rule)
        assert new['cardinality'] == 'optional_one'
    migrated, changes = migrate(e.model_dump())
    assert migrated['gold'] == e.model_dump()['gold']
    assert migrated['text'] == e.text and changes
    assert migrate(migrated)[0] == migrated and not migrate(migrated)[1]
    no_gold = deepcopy(e.model_dump())
    no_gold['gold'] = {}
    assert migrate(no_gold)[0]['processing'] == migrated['processing']
    with pytest.raises(ValueError, match='Conflicting'):
        cardinality('optional_one', 'one_or_more')
    assert cardinality('zero_or_more', 'unique_list') == 'unique_list'


def test_public_schema_and_native_training_preserve_evidence_and_scalar_ownership():
    rows, rejected = examples(contact_episode())
    assert not rejected
    _, request, native = rows[0]
    metadata = schema_for(request).build()['record_metadata']
    assert native['output']['record_metadata'] == metadata
    fields = metadata['record']['fields']
    assert fields['f1'] == {'cardinality': 'zero_or_more', 'exclusive': True}
    request.fields['role'] = Text('Functions', cardinality='unique_list')
    assert schema_for(request).build()['record_metadata']['record']['fields']['f1'] == {'cardinality': 'zero_or_more'}


def test_native_assignment_does_not_copy_one_occurrence_to_two_contacts():
    from gliner2.models.boundary.records import decode_group
    specs = [SimpleNamespace(query_id=i, exclusive=i == 1, cardinality=SimpleNamespace(is_scalar=False))
             for i in range(2)]
    group = SimpleNamespace(num_instances=2, object_logits=torch.tensor([5., 5.]),
        spec=SimpleNamespace(mode='natural', anchor_query_id=0), field_specs=specs,
        field_query_ids=[0, 1], instance_seed=[0, 1], instance_spans=[(0, 1), (4, 5)],
        field_spans=[torch.tensor([[0, 1], [4, 5]]), torch.tensor([[2, 3], [6, 7]])],
        assign_logits=[torch.zeros(2, 3), torch.logit(torch.tensor([[.1, .9, .8], [.1, .2, .95]]))])
    decoded = decode_group(group, field_threshold=.3)
    assert [record.fields[1] for record in decoded] == [[(2, 3)], [(6, 7)]]
    # A genuinely list-valued field keeps its declared sharing behavior.
    specs[1].exclusive = False
    assert decode_group(group, field_threshold=.3)[0].fields[1] == [(2, 3), (6, 7)]


def test_normalized_record_values_preserve_duplicates_and_detect_conflicts():
    text = 'Anna: 12 cm, 120 mm. Max: 12 cm, 13 cm.'
    request = ExtractRequest(text=text, many=True, anchor='name', fields={
        'name': Person('Person', cardinality='required_one'), 'length': Quantity('Length', unit='mm')})
    rows = [{'name': [span(text, 'Anna')], 'length': [span(text, '12 cm'), span(text, '120 mm')]},
            {'name': [span(text, 'Max')], 'length': [span(text, '12 cm', 20), span(text, '13 cm')]}]
    result = resolve(request, rows)
    assert result['data'][0]['length'] == 120
    assert result['data'][1]['length'] == [120, 130]
    assert {'field': 'length', 'index': 1, 'code': 'ambiguous_value'} in result['issues']


def test_identical_job_titles_at_distinct_offsets_preserve_both_contacts():
    text = 'Anna: Einkauf. Max: Einkauf.'
    request = ExtractRequest(text=text, many=True, anchor='name', fields={
        'name': Person('Person', cardinality='required_one'), 'role': Text('Function')})
    result = resolve(request, [
        {'name': [span(text, 'Anna')], 'role': [span(text, 'Einkauf')]},
        {'name': [span(text, 'Max')], 'role': [span(text, 'Einkauf', 14)]}])
    assert result['data'] == [{'name': 'Anna', 'role': 'Einkauf'}, {'name': 'Max', 'role': 'Einkauf'}]
    assert result['issues'] == []
