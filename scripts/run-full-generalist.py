"""Durable local full training, followed by Dev-only evaluation of the selected model."""

import importlib.util
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from tractor.coverage_assembly import sha
from tractor.jsonio import dumps, loads

ROOT = Path('reports/generalist/full-training')


def status(state, **extra):
    path = ROOT / 'status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(dumps({'state': state, 'pid': os.getpid(),
                          'updated_at': datetime.now(UTC).isoformat(), **extra}, indent=2))
    tmp.replace(path)


def main():
    status('waiting_for_idle_gpu')
    spec = importlib.util.spec_from_file_location('pilot', 'scripts/run-generalist-pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_idle_gpu()
    config = ROOT / 'train-config.json'
    data = Path('data/generalist-full')
    model = Path('runs/generalist-coverage/final')
    design = {'initial_model': str(model), 'initial_weights_sha256': sha(model / 'model.safetensors'),
              'data_manifest_sha256': sha(data / 'manifest.json'), 'config_sha256': sha(config),
              'quality_policy_sha256': sha('configs/generalist-quality.json'),
              'expected_steps': 59000, 'expected_training_samples': 1888000,
              'qualification': 'All 402613 unique training examples, balanced repetitions, two epochs. '
              'Weights-only warm start. Dev only; no final test, latency measurement or release promotion.'}
    (ROOT / 'design.json').write_text(dumps(design, indent=2))
    status('training', expected_steps=59000)
    with (ROOT / 'train.log').open('x') as log:
        subprocess.run([sys.executable, '-m', 'tractor.cli', 'train', '--model', str(model),
                        '--data', str(data), '--config', str(config), '--device', 'cuda'],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    evidence = loads(Path('runs/generalist-full/training-evidence.json').read_text())
    if evidence['result']['total_steps'] != design['expected_steps'] or evidence['workload']['training_samples'] != design['expected_training_samples']:
        raise ValueError('Incomplete training workload')
    status('evaluating_selected_checkpoint')
    with (ROOT / 'selected-dev.log').open('x') as log:
        subprocess.run([sys.executable, '-m', 'tractor.cli', 'evaluate', '--model', 'runs/generalist-full/best',
                        '--episodes', 'data/generalist-extraction-scoped/dev.episodes.jsonl',
                        '--split', 'dev', '--device', 'cuda', '--output', str(ROOT / 'selected-dev')],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    from tractor.full_training import score_general_dev
    from tractor.generalist_data import read_episodes
    from tractor.training_report import rescore_predictions

    episodes = list(read_episodes('data/generalist-extraction-scoped/dev.episodes.jsonl'))
    rows = [loads(line) for line in (ROOT / 'selected-dev/predictions.jsonl').read_text().splitlines()]
    if len(rows) != len(episodes) or {r['id'] for r in rows} != {e.id for e in episodes}:
        raise ValueError('Incomplete full Dev evaluation')
    if any(r['error'] for r in rows):
        raise ValueError('Full Dev inference errors; inspect selected-dev predictions')
    predictions = {r['id']: r['prediction'] for r in rows}
    scores, gate, minimum = score_general_dev(episodes, [predictions[e.id] for e in episodes],
                                            loads(Path('configs/generalist-quality.json').read_text()))
    rescore_predictions('data/generalist-extraction-scoped/dev.episodes.jsonl', ROOT / 'selected-dev')
    (ROOT / 'quality.json').write_text(dumps({'scores': scores, 'gate': gate, 'minimum': minimum}, indent=2))
    status('complete', quality_gate_passed=gate['passed'], weakest_required_score=minimum)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        raise
