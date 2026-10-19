import pytest

from tractor.document_eval import run


@pytest.mark.parametrize('threshold', [0, -0.1, 1.1, True, float('nan'), float('inf')])
def test_invalid_dev_threshold_rejected_before_loading_weights(threshold):
    with pytest.raises(ValueError, match='threshold'):
        run('no-model', 'no-data', 'no-output', threshold=threshold)


@pytest.mark.parametrize('threshold', [0, -0.1, 1.1, True, float('nan'), float('inf')])
def test_invalid_record_threshold_rejected_before_loading_weights(threshold):
    with pytest.raises(ValueError, match='threshold'):
        run('no-model', 'no-data', 'no-output', record_threshold=threshold)


def test_record_threshold_does_not_change_entity_thresholds():
    from tractor.episodes import Task, compile_schema
    tasks = [Task(id='doc', kind='entities', description='Header', fields={'title': 'Title'}),
             Task(id='rows', kind='records', description='Records',
                  fields={'name': 'Name', 'role': 'Role'}, anchor='name')]
    schema = compile_schema(tasks, threshold=.5, record_threshold=.3).to_dict()
    assert schema['entities']['title']['threshold'] == .5
    assert all(f['threshold'] == .3 for f in schema['structures']['rows']['fields'])
    default = compile_schema(tasks, threshold=.5).to_dict()
    assert all(f['threshold'] == .5 for f in default['structures']['rows']['fields'])
