from collections import Counter
from types import SimpleNamespace

import pytest

from tractor.classification_coverage import (
    DOMAINS,
    INTENTS,
    massive_episode,
    split_conflicts,
    synthetic_episode,
)
from tractor.generalist_data import validate_classification_targets
from tractor.jsonio import loads


@pytest.fixture
def config():
    from pathlib import Path
    return loads(Path('tests/fixtures/legacy-classification-config.json').read_text())


def test_synthetic_classes_are_balanced_in_both_languages(config):
    for axis, spec in config['axes'].items():
        counts = Counter()
        for i in range(200):
            e = synthetic_episode(config, axis, 'train', i)
            answer = e.gold['category'].records[0]['label'].label
            description = e.tasks[0].labels[answer]
            counts[e.language, description] += 1
        assert set(counts.values()) == {25}
        assert len(counts) == 2 * len(spec['classes'])


def test_contrasts_and_mixed_tasks_keep_groups_and_offsets(config):
    for axis in config['axes']:
        a, b = (synthetic_episode(config, axis, 'train', i) for i in (0, 1))
        assert a.source_group == b.source_group
        assert a.text != b.text
        assert len(a.tasks) == len(b.tasks) == 2
        for e in (a, b):
            for record in e.gold['details'].records:
                for answer in record.values():
                    for span in answer.items:
                        assert e.text[span.start:span.end] == span.text


def test_classification_gold_recovered_after_label_permutation(config):
    e = synthetic_episode(config, 'approval', 'train', 0)
    labels = list(reversed(e.tasks[0].labels))
    label = e.gold['category'].records[0]['label'].label
    tokens = ['(', '[P]', 'category: question'] + [t for key in labels for t in ('[L]', key)]
    b = SimpleNamespace(task_types=[['classifications']], schema_tokens_list=[[tokens]],
                        structure_labels=[[[int(key == label) for key in labels]]])
    validate_classification_targets(e, b)
    b.structure_labels = [[[0] * len(labels)]]
    with pytest.raises(ValueError, match='differ from gold'):
        validate_classification_targets(e, b)
    b.task_types = [[]]
    b.schema_tokens_list = [[]]
    b.structure_labels = [[]]
    with pytest.raises(ValueError, match='missing'):
        validate_classification_targets(e, b)


def test_parallel_public_rows_share_split_and_gold():
    base = {'id': '456', 'partition': 'test', 'scenario': list(DOMAINS).index('alarm'),
            'intent': list(INTENTS).index('alarm_set'), 'utt': 'set an alarm', 'locale': 'en-US'}
    a = massive_episode(base, list(DOMAINS), list(INTENTS), 907)
    b = massive_episode(dict(base, locale='de-DE', utt='stelle einen wecker'), list(DOMAINS), list(INTENTS), 907)
    assert a.source_group == b.source_group
    assert a.split == b.split
    assert a.tasks == b.tasks
    assert a.gold == b.gold
    assert a.original_split == 'test'


def test_cross_split_duplicate_removes_parallel_group(config):
    a = synthetic_episode(config, 'approval', 'train', 0)
    b = synthetic_episode(config, 'approval', 'dev', 0).model_copy(update={'text': a.text})
    assert split_conflicts([a, b]) == {a.source_group, b.source_group}


def test_classification_metrics_are_not_hidden_by_correct_spans(config, tmp_path, monkeypatch):
    from tractor import evaluate
    from tractor.jsonio import dumps

    e = synthetic_episode(config, 'approval', 'dev', 0)
    gold_label = e.gold['category'].records[0]['label'].label
    wrong = next(label for label in e.tasks[0].labels if label != gold_label)
    raw = {'category': wrong, 'entities': {key: [s.model_dump() for s in answer.items]
                                         for key, answer in e.gold['details'].records[0].items()}}

    class Extractor:
        model_revision = 'fixture'

        def __init__(self, *args):
            pass

        def extract_tasks(self, text, tasks):
            return raw

    monkeypatch.setattr(evaluate, 'Extractor', Extractor)
    source = tmp_path / 'dev.jsonl'
    source.write_text(dumps(e.model_dump()) + '\n')
    output = tmp_path / 'metrics'
    evaluate.evaluate_episodes(SimpleNamespace(calibrate=False, output=str(output), model='fixture',
                                              device='cpu', precision='fp32', episodes=str(source), split='dev'))
    report = loads((output / 'summary.json').read_text())
    assert report['task_kinds']['classification']['field_f1'] == 0
    assert report['task_kinds']['entities']['field_f1'] == 1
    assert report['overall']['field_f1'] > 0
    assert next(iter(report['classification_tasks'].values()))['request_exact'] == 0


def test_conflicting_public_labels_exclude_all_parallel_identities():
    from tractor.classification_coverage import public_annotation_conflicts

    rows = [{'id': '1', 'locale': 'en-US', 'utt': 'Turn it off', 'scenario': 1, 'intent': 2},
            {'id': '2', 'locale': 'en-US', 'utt': ' turn  IT off ', 'scenario': 1, 'intent': 3},
            {'id': '1', 'locale': 'de-DE', 'utt': 'Schalte es aus', 'scenario': 1, 'intent': 2}]
    assert public_annotation_conflicts(rows) == {'massive:utterance:1', 'massive:utterance:2'}
