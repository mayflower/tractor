import hashlib
import os
import shutil
from pathlib import Path

import pytest

from tractor.biqwen_attention import BIDIRECTIONAL, CAUSAL
from tractor.biqwen_followup import completed_predecessor, process_identity
from tractor.biqwen_training import BIDIRECTIONAL_RECIPE, validate_config
from tractor.coverage_assembly import sha
from tractor.jsonio import dumps, loads


def test_bidirectional_recipe_is_explicit_and_keeps_full_training_requirements():
    config = loads(Path('configs/biqwen-full.json').read_text())
    config.update(recipe=BIDIRECTIONAL_RECIPE, attention_mode=BIDIRECTIONAL)
    validate_config(config)
    config['attention_mode'] = CAUSAL
    with pytest.raises(ValueError, match='attention contract'):
        validate_config(config)
    config['attention_mode'] = BIDIRECTIONAL
    config['max_steps'] = 1000
    with pytest.raises(ValueError, match='forbids'):
        validate_config(config)


def predecessor(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'manifest.json').write_text('{}')
    (data / 'dev.episodes.jsonl').write_text('{"id":"a"}\n{"id":"b"}\n')
    config = {'dataset': {'path': str(data)}, 'model': 'base', 'model_revision': 'pinned',
              'num_epochs': 4, 'policy': 'policy.json', 'baseline_sha256': 'baseline'}
    plan = {'config_sha256': hashlib.sha256(dumps(config, sort_keys=True).encode()).hexdigest(),
            'total_optimizer_updates': 4, 'sampling': False, 'step_limit': None,
            'native_requests_per_epoch': 2, 'dataset_manifest_sha256': sha(data / 'manifest.json'),
            'full_dev_episodes': 2, 'dataset_files': {
                'dev.episodes.jsonl': sha(data / 'dev.episodes.jsonl'), 'train.identities.jsonl': 'identities'}}
    for name, value in [('config', config), ('plan', plan), ('status', {
            'status': 'completed', 'training_complete': True, 'epoch': 4, 'cursor': 0, 'step': 4})]:
        (root / f'{name}.json').write_text(dumps(value))
    for epoch in range(1, 5):
        (root / f'epoch-{epoch}-coverage.json').write_text(dumps({
            'epoch': epoch, 'seen_requests': 2, 'expected_requests': 2,
            'all_requests_seen_exactly_once': True, 'identity_sha256': 'identities',
            'coverage_sha256': hashlib.sha256(bytes([1, 1])).hexdigest()}))
        (root / f'epoch-{epoch}-dev.json').write_text(dumps({
            'epoch': epoch, 'regression_guard': {'passed': False}, 'selection_metric': .4}))
        (root / f'epoch-{epoch}-dev.predictions.jsonl').write_text('{"id":"a"}\n{"id":"b"}\n')
    return root, config


def test_followup_requires_complete_coverage_and_dev_even_when_quality_is_poor(tmp_path):
    root, config = predecessor(tmp_path)
    assert completed_predecessor(root, config)['full_dev_episodes'] == 2
    # Poor quality is permitted as comparative evidence, missing computation is not.
    (root / 'epoch-2-dev.predictions.jsonl').write_text('{"id":"a"}\n{"id":"a"}\n')
    with pytest.raises(ValueError, match='exactly once'):
        completed_predecessor(root, config)


@pytest.mark.parametrize('state', ['running', 'interrupted_or_failed'])
def test_no_followup_from_live_or_failed_training(tmp_path, state):
    root, config = predecessor(tmp_path)
    path = root / 'status.json'
    status = loads(path.read_text())
    status['status'] = state
    path.write_text(dumps(status))
    with pytest.raises(ValueError, match='four full epochs'):
        completed_predecessor(root, config)


def test_missing_requests_and_changed_contract_block_followup(tmp_path):
    root, config = predecessor(tmp_path)
    with pytest.raises(ValueError, match='comparison contract'):
        completed_predecessor(root, {**config, 'model_revision': 'other'})
    path = root / 'epoch-3-coverage.json'
    coverage = loads(path.read_text())
    coverage['seen_requests'] = 1
    path.write_text(dumps(coverage))
    with pytest.raises(ValueError, match='incomplete epoch coverage'):
        completed_predecessor(root, config)


def test_process_identity_uses_live_process_start_time():
    assert process_identity(os.getpid()) == process_identity(os.getpid()) is not None
    assert process_identity(2147483647) is None


@pytest.mark.parametrize('gpu_passes', [True, False])
def test_automatic_sequence_waits_then_checks_gpu_before_full_training(tmp_path, monkeypatch, gpu_passes):
    from tractor import biqwen_followup as followup

    parent, config = predecessor(tmp_path)
    config.update(recipe=BIDIRECTIONAL_RECIPE, attention_mode=BIDIRECTIONAL,
                  output_dir=str(tmp_path / 'next-run'))
    work = tmp_path / 'worktree'
    (work / 'configs').mkdir(parents=True)
    active = work / 'configs/active-training.json'
    active.write_text('{"status":"queued"}')
    config_path = work / 'configs/run.json'
    config_path.write_text(dumps(config))
    monkeypatch.chdir(work)
    monkeypatch.setattr(followup, 'prepare_plan', lambda _: {'fixture': True})
    monkeypatch.setattr(followup, 'validate_process', lambda *args: 'start-1')
    identities = iter(['start-1', None])
    monkeypatch.setattr(followup, 'process_identity', lambda _: next(identities))
    stages = []
    monkeypatch.setattr(followup.time, 'sleep', lambda _: stages.append('wait'))

    def execute(command, **kwargs):
        assert stages[0] == 'wait'
        if 'tractor.biqwen_preflight' in command:
            stages.append('gpu')
            assert loads(active.read_text())['status'] == 'queued'
            output = Path(command[command.index('--output') + 1])
            output.mkdir()
            (output / 'report.json').write_text(dumps({
                'status': 'passed' if gpu_passes else 'failed', 'all_parameters_unchanged': True,
                'optimizer_updates': 0, 'attention': {'mode': BIDIRECTIONAL},
                'future_context_probe': {'passed': True}, 'config': config}))
        else:
            assert command[-1] == '--train' and 'tractor.biqwen_training' in command
            stages.append('train')
            assert loads(active.read_text())['binding'] == config['dataset']
            output = Path(config['output_dir'])
            shutil.copytree(parent, output)
            (output / 'config.json').write_text(dumps(config))
            plan = loads((output / 'plan.json').read_text())
            plan['config_sha256'] = hashlib.sha256(dumps(config, sort_keys=True).encode()).hexdigest()
            (output / 'plan.json').write_text(dumps(plan))

    monkeypatch.setattr(followup.subprocess, 'run', execute)
    job = tmp_path / 'job'
    if gpu_passes:
        followup.run(config_path, parent, 42, job, interval=0)
        assert stages == ['wait', 'gpu', 'train']
        assert loads((job / 'status.json').read_text())['status'] == 'training_completed_evaluation_pending'
    else:
        with pytest.raises(ValueError, match='GPU preflight'):
            followup.run(config_path, parent, 42, job, interval=0)
        assert stages == ['wait', 'gpu']
        assert loads(active.read_text())['status'] == 'queued'
        assert not Path(config['output_dir']).exists()
        assert loads((job / 'status.json').read_text())['status'] == 'failed'
