from tractor.api_training import examples
from tractor.document_sources import envelope
from tractor.model import schema_for


def test_native_training_uses_api_names_record_policy_and_separate_requests():
    e = envelope('example', 'api-schema-test', 'train', 'contacts', 'en', 'Ada, Leeds',
        [{'id': 'people', 'kind': 'entities', 'description': 'People', 'fields': {'name': 'Person'}},
         {'id': 'offices', 'kind': 'records', 'description': 'Offices', 'fields': {'who': 'Person', 'town': 'City'},
          'anchor': 'who', 'cardinalities': {'who': 'required_one', 'town': 'optional_one'}}],
        {'people': {'complete': True, 'records': [{'name': {'status': 'present', 'items': [{'start': 0, 'end': 3, 'text': 'Ada'}]}}]},
         'offices': {'complete': True, 'records': [{'who': {'status': 'present', 'items': [{'start': 0, 'end': 3, 'text': 'Ada'}]},
                                                  'town': {'status': 'present', 'items': [{'start': 5, 'end': 10, 'text': 'Leeds'}]}}]}})
    rows, rejected = examples(e)
    assert not rejected and len(rows) == 2
    for _, request, native in rows:
        schema = schema_for(request).build()
        for key in ('entity_descriptions', 'json_descriptions', 'record_metadata'):
            assert native['output'].get(key) == schema.get(key)
        assert native['input'] == e.text
    assert rows[0][2]['output']['entities'] == {'f0': [{'start': 0, 'end': 3, 'text': 'Ada'}]}
    assert rows[1][2]['output']['json_structures'] == [{'record': {
        'f0': [{'start': 0, 'end': 3, 'text': 'Ada'}], 'f1': [{'start': 5, 'end': 10, 'text': 'Leeds'}]}}]
