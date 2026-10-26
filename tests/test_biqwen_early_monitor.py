import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    'biqwen_early_monitor', Path(__file__).parents[1] / 'scripts/monitor-biqwen-training.py')
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def test_monitor_selection_covers_sparse_strata_without_gold_or_order_dependence():
    rows = [dict(id=str(i), split='dev', language='de' if i < 2 else 'en',
                 family='contacts', artifact='single_object', text_origin='real',
                 label_origin='human', gold={'irrelevant': i}) for i in range(100)]
    first = monitor.select_rows(rows, 12)
    assert len(first) == len({r['id'] for r in first}) == 12
    assert {r['language'] for r in first} == {'de', 'en'}
    other = [dict(r, gold={}) for r in reversed(rows)]
    assert [r['id'] for r in first] == [r['id'] for r in monitor.select_rows(other, 12)]
    with pytest.raises(ValueError, match='at least'):
        monitor.select_rows(rows, 1)
    with pytest.raises(ValueError, match='unique Dev'):
        monitor.select_rows([dict(rows[0], split='train')], 12)
    with pytest.raises(ValueError, match='unique Dev'):
        monitor.select_rows([rows[0], rows[0]], 12)


def test_snapshot_survives_trainer_pruning_and_rejects_corruption(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    weights = source / 'weights'
    weights.write_bytes(b'immutable checkpoint')
    manifest = {'files': {'weights': monitor.digest(weights)}}
    (source / 'checkpoint.json').write_text(json.dumps(manifest))
    target = tmp_path / 'pinned'
    monitor.snapshot(source, target)
    assert weights.stat().st_ino == (target / 'weights').stat().st_ino
    shutil.rmtree(source)
    assert monitor.digest(target / 'weights') == manifest['files']['weights']
    (target / 'weights').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        monitor.snapshot(target, tmp_path / 'bad')
    assert not (tmp_path / 'bad.writing').exists()


def test_incomplete_references_do_not_become_correctness_evidence():
    report = {'scores': {
        'shape:entities:real/human': {'counts': {'field_tp': 3, 'field_fp': 1, 'field_fn': 1}},
        'shape:records:real/human': {'reference_complete': False, 'counts': {'field_tp': 100}},
        'domain:contacts:real/human': {'counts': {'field_tp': 3}},
    }}
    result = monitor.aggregate(report)
    assert result['field_f1'] == .75
    assert result['excluded_incomplete_groups'] == ['shape:records:real/human']


def test_source_change_fails_closed(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('version = 1')
    manifest = {'bindings': {str(source): monitor.digest(source)}}
    monitor.check_bindings(manifest)
    source.write_text('version = 2')
    with pytest.raises(ValueError, match='changed'):
        monitor.check_bindings(manifest)


def test_many_synthetic_strata_do_not_crowd_out_real_documents():
    rows = [dict(id=f'{origin}-{i}', split='dev', language='en',
                 family=f'domain-{i % 20}' if origin == 'synthetic' else 'contacts',
                 artifact='single_object', text_origin=origin, label_origin='human')
            for origin in ['real', 'synthetic'] for i in range(100)]
    result = monitor.select_rows(rows, 80)
    assert sum(row['text_origin'] == 'real' for row in result) == 40
    assert len({row['family'] for row in result}) == 21


def test_monitor_evaluates_early_checkpoint_and_bounds_backlog(tmp_path, monkeypatch):
    root = tmp_path / 'monitor'
    run = tmp_path / 'run'
    (root / 'snapshots').mkdir(parents=True)
    (run / 'checkpoints').mkdir(parents=True)

    def checkpoint(step, directory):
        path = directory / f'step-{step:08d}'
        path.mkdir()
        (path / 'weights').write_bytes(str(step).encode())
        monitor.write(path / 'checkpoint.json', {'step': step, 'files': {'weights': monitor.digest(path / 'weights')}})

    checkpoint(0, root / 'snapshots')
    monitor.write(root / 'manifest.json', {'run': str(run), 'bindings': {}, 'threads': 2,
                                         'checkpoint_interval': 1000, 'trainer_pid': 123,
                                         'trainer_start_ticks': 'identity'})
    monitor.write(root / 'status.json', {'status': 'prepared', 'evaluated': [], 'skipped': []})
    monitor.write(run / 'status.json', {'status': 'running', 'step': 0})
    commands, comparisons = [], []

    def launch(command, **options):
        commands.append(command)
        assert options['env']['CUDA_VISIBLE_DEVICES'] == ''
        return SimpleNamespace(pid=456, returncode=0, poll=lambda: 0)

    def tick(_):
        if not (run / 'checkpoints/step-00001000').exists():
            for step in [1000, 2000, 3000]:
                checkpoint(step, run / 'checkpoints')
            monitor.write(run / 'status.json', {'status': 'completed', 'step': 3000})

    monkeypatch.setattr(monitor.subprocess, 'Popen', launch)
    monkeypatch.setattr(monitor.time, 'sleep', tick)
    monkeypatch.setattr(monitor, 'compare', lambda _, step: comparisons.append(step))
    monkeypatch.setattr(monitor, 'process_identity', lambda _: 'identity')
    monkeypatch.setattr(monitor.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 * 1024**3))
    monitor.monitor(root)
    state = monitor.read(root / 'status.json')
    assert state['status'] == 'completed'
    assert state['evaluated'] == [0, 1000, 3000]
    assert state['skipped'] == [2000]
    assert comparisons == [1000, 3000]
    assert len(commands) == 3
    assert list((root / 'snapshots').iterdir()) == []
    assert len(list((run / 'checkpoints').iterdir())) == 3
