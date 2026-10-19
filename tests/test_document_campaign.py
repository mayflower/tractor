from pathlib import Path

import pytest

from tractor.document_campaign import quotas
from tractor.document_duplicates import Index, masked
from tractor.document_training import selection_score
from tractor.jsonio import loads


def test_accepted_targets_are_30000_train_and_separate_balanced_evals():
    targets = quotas(loads(Path('configs/document-build.json').read_text()))
    assert sum(v for k, v in targets.items() if k[0] == 'train') == 30000
    assert sum(targets.values()) == 39600
    for split in ('dev', 'calibration', 'test'):
        assert sum(v for k, v in targets.items() if k[0] == split) == 3200
    for shape, expected in [('single_object', 18000), ('repeated_records', 9000), ('entity_list', 3000)]:
        assert sum(v for k, v in targets.items() if k[0] == 'train' and k[2] == shape) == expected


def test_missing_real_evidence_does_not_turn_into_passing_score():
    report = {'passed': False, 'checks': {'real': {'score': None}, 'invoices': {'score': .8}, 'contracts': {'score': .4}}}
    assert selection_score(report) == pytest.approx(.9*.4 + .1*.6)
    assert not report['passed'] and report['checks']['real']['score'] is None
    with pytest.raises(ValueError):
        selection_score({'checks': {'missing': {'score': None}}})


def test_weak_silver_domain_is_not_hidden_by_human_only_release_checks():
    report = {'checks': {'synthetic': {'score': .85}, 'human_missing': {'score': None}},
              'scores': {'domain:contacts:real/deterministic': {'field_f1': .31}}}
    assert selection_score(report) == pytest.approx(.9*.31 + .1*(.85+.31)/2)


def test_masked_template_detection_keeps_original_source_identity():
    def row(name, split):
        text = f'Invoice recipient {name} has requested that the attached receipt be sent to the purchasing office before Friday.'
        start = text.index(name)
        return {'id': name, 'source_group': name, 'split': split, 'text': text,
                'gold': {'header': {'records': [{'recipient': {'items': [{'start': start, 'end': start+len(name)}]}}]}}}
    first, second = row('Cedar Ltd', 'test'), row('Willow PLC', 'train')
    assert masked(first) == masked(second)
    index = Index()
    assert index.check(first) == []
    matches = index.check(second)
    assert len(matches) == 1 and matches[0]['masked_jaccard'] == 1
    assert matches[0]['prior_split'] == 'test' and matches[0]['split'] == 'train'
