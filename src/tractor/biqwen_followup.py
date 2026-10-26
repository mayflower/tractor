"""Run the explicit full bidirectional experiment after a verified predecessor."""

import argparse
import fcntl
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from .biqwen_attention import BIDIRECTIONAL
from .biqwen_training import BIDIRECTIONAL_RECIPE, prepare_plan
from .coverage_assembly import sha
from .jsonio import dumps, loads


def process_identity(pid):
    """PID plus Linux start time, avoiding accidental attachment to a reused PID."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] in {'Z', 'X'}:
        return None
    return fields[19]


def validate_process(pid, predecessor):
    identity = process_identity(pid)
    if identity is None:
        raise ValueError('The exact predecessor process must be live when queuing')
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if b'tractor.biqwen_training' not in command:
        raise ValueError('Expected the actual BiQwen trainer process')
    parent_config = loads((Path(predecessor) / 'config.json').read_text())
    if Path(parent_config['output_dir']).resolve() != Path(predecessor).resolve():
        raise ValueError('Predecessor run identity differs')
    parent_cwd = Path(f'/proc/{pid}/cwd').resolve()
    if parent_cwd == Path.cwd().resolve():
        raise ValueError('Follow-up must have its own worktree and active configuration')
    arguments = [os.fsdecode(arg) for arg in command if arg]
    if '--config' not in arguments:
        raise ValueError('Predecessor must explicitly name its training configuration')
    process_config = parent_cwd / arguments[arguments.index('--config') + 1]
    if loads(process_config.read_text()) != parent_config:
        raise ValueError('Predecessor process does not belong to the specified run')
    return identity


def completed_predecessor(root, config):
    root = Path(root)
    status, plan, previous = [loads((root / name).read_text())
                              for name in ['status.json', 'plan.json', 'config.json']]
    if (status.get('status') != 'completed' or status.get('training_complete') is not True
            or status['epoch'] != 4 or status['cursor'] != 0
            or status['step'] != plan['total_optimizer_updates'] or previous['num_epochs'] != 4):
        raise ValueError('Predecessor must finish all four full epochs')
    if hashlib.sha256(dumps(previous, sort_keys=True).encode()).hexdigest() != plan['config_sha256']:
        raise ValueError('Predecessor configuration changed')
    for key in ['dataset', 'model', 'model_revision', 'policy', 'baseline_sha256']:
        if previous[key] != config[key]:
            raise ValueError(f'Unmatched predecessor comparison contract: {key}')
    if plan['sampling'] is not False or plan['step_limit'] is not None:
        raise ValueError('Predecessor must be the full unsampled training run')
    count = plan['native_requests_per_epoch']
    coverage_hash = hashlib.sha256(bytes([1]) * count).hexdigest()
    data = Path(config['dataset']['path'])
    if sha(data / 'manifest.json') != plan['dataset_manifest_sha256']:
        raise ValueError('Predecessor dataset changed')
    if sha(data / 'dev.episodes.jsonl') != plan['dataset_files']['dev.episodes.jsonl']:
        raise ValueError('Predecessor Dev changed')
    with (data / 'dev.episodes.jsonl').open() as stream:
        dev_ids = {loads(line)['id'] for line in stream}
    if len(dev_ids) != plan['full_dev_episodes']:
        raise ValueError('Predecessor Dev inventory differs')
    for epoch in range(1, 5):
        coverage = loads((root / f'epoch-{epoch}-coverage.json').read_text())
        if (coverage['epoch'] != epoch or coverage['seen_requests'] != count
                or coverage['expected_requests'] != count or coverage['all_requests_seen_exactly_once'] is not True
                or coverage['coverage_sha256'] != coverage_hash
                or coverage['identity_sha256'] != plan['dataset_files']['train.identities.jsonl']):
            raise ValueError('Predecessor has incomplete epoch coverage')
        metrics = loads((root / f'epoch-{epoch}-dev.json').read_text())
        if metrics['epoch'] != epoch or 'regression_guard' not in metrics or 'selection_metric' not in metrics:
            raise ValueError('Predecessor full Dev decision is missing')
        with (root / f'epoch-{epoch}-dev.predictions.jsonl').open() as stream:
            ids = [loads(line)['id'] for line in stream]
        if len(ids) != len(dev_ids) or set(ids) != dev_ids:
            raise ValueError('Predecessor did not evaluate every Dev document exactly once')
    return plan


def run(config_path, predecessor, pid, output, interval=30):
    """Separate process/worktree; never stop, mutate or resume the predecessor."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = Path(config_path).resolve()
        config = loads(config_path.read_text())
        if config['recipe'] != BIDIRECTIONAL_RECIPE or config.get('attention_mode') != BIDIRECTIONAL:
            raise ValueError('Follow-up requires the explicit bidirectional recipe')
        if config['num_epochs'] != 4 or Path(config['output_dir']).exists():
            raise ValueError('Follow-up requires four full epochs and a fresh output directory')
        if Path(config['output_dir']).resolve() == Path(predecessor).resolve():
            raise ValueError('Follow-up may not overwrite its predecessor')
        plan = prepare_plan(config)
        identity = validate_process(pid, predecessor)
        status = {'status': 'waiting', 'predecessor': str(Path(predecessor).resolve()),
                  'predecessor_pid': pid, 'predecessor_start_time': identity,
                  'config': str(config_path), 'config_sha256': sha(config_path),
                  'worktree': str(Path.cwd().resolve()), 'plan': plan,
                  'initialization': 'same pinned merged base as predecessor; fresh optimizer',
                  'adaptation': 'supervised extraction on all eligible DE/EN requests; no MNTP claim'}

        def persist():
            temporary = root / 'status.json.writing'
            temporary.write_text(dumps(status, indent=2) + '\n')
            temporary.replace(root / 'status.json')

        persist()
        try:
            while True:
                alive = process_identity(pid)
                current = loads((Path(predecessor) / 'status.json').read_text())
                if current['status'] == 'completed' and alive is None:
                    break
                if alive != identity or current['status'] not in {'running', 'completed'}:
                    raise RuntimeError('Predecessor stopped or failed; no automatic restart or follow-up')
                time.sleep(interval)
            completed_predecessor(predecessor, config)
            if sha(config_path) != status['config_sha256'] or prepare_plan(config) != plan:
                raise ValueError('Queued configuration, corpus or implementation changed')
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))

            def execute(stage, arguments):
                status['status'] = stage
                persist()
                with (root / f'{stage}.log').open('x') as log:
                    subprocess.run([sys.executable, '-m', *arguments], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)

            # Real full-sized forward/backward checks, no optimizer steps or latency tests.
            gpu = root / 'gpu-preflight'
            execute('gpu_preflight', ['tractor.biqwen_preflight', '--config', str(config_path), '--output', str(gpu)])
            report = loads((gpu / 'report.json').read_text())
            if (report['status'] != 'passed' or report['all_parameters_unchanged'] is not True
                    or report['optimizer_updates'] != 0 or report['attention']['mode'] != BIDIRECTIONAL
                    or report['future_context_probe']['passed'] is not True or report['config'] != config):
                raise ValueError('Actual bidirectional GPU preflight did not pass')
            if prepare_plan(config) != plan:
                raise ValueError('Inputs changed during GPU preflight')
            active = {'backend': 'biqwen', 'status': 'prepared', 'binding': config['dataset'],
                      'training_config': str(config_path), 'training_started': False, 'training_complete': False}
            Path('configs/active-training.json').write_text(dumps(active, indent=2) + '\n')
            execute('training', ['tractor.biqwen_training', '--config', str(config_path), '--train'])
            result = loads((Path(config['output_dir']) / 'status.json').read_text())
            if result['status'] != 'completed' or result['training_complete'] is not True:
                raise ValueError('Follow-up process exited without completing full training')
            completed_predecessor(config['output_dir'], config)
            status['status'] = 'training_completed_evaluation_pending'
            status['quality_proven'] = False
        except BaseException as exc:
            status.update(status='failed', error=type(exc).__name__, message=str(exc))
            raise
        finally:
            persist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--predecessor', required=True)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.config, args.predecessor, args.pid, args.output)
