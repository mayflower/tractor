"""Read-only early Dev monitoring alongside full training; never selects weights.

Runs on CPU with its own matched step-zero baseline. The GPU trainer, its source
bindings, full data coverage and complete-epoch selection remain untouched.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select_rows(rows, count):
    """Cover all strata, then balance real/synthetic origins and their strata."""
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        if row['split'] != 'dev' or row['id'] in ids:
            raise ValueError('Expected unique Dev documents only')
        ids.add(row['id'])
        key = tuple(row[k] for k in ('language', 'family', 'artifact', 'text_origin', 'label_origin'))
        groups[key].append(row)
    if not groups or count < len(groups):
        raise ValueError(f'Need at least {len(groups)} documents to cover every observed stratum')
    for group in groups.values():
        group.sort(key=lambda row: hashlib.sha256(('early-dev-v1:' + row['id']).encode()).digest())
    keys = sorted(groups, key=lambda key: hashlib.sha256('|'.join(key).encode()).digest())
    selected = [groups[key][0] for key in keys]
    used = Counter({key: 1 for key in keys})
    origins = Counter(key[3] for key in keys)
    while len(selected) < min(count, len(rows)):
        available = [key for key in keys if used[key] < len(groups[key])]
        key = min(available, key=lambda key: (origins[key[3]], used[key]))
        selected.append(groups[key][used[key]])
        used[key] += 1
        origins[key[3]] += 1
    return selected


def snapshot(source, destination):
    """Pin immutable files against trainer pruning, without duplicating their data."""
    destination = Path(destination)
    if destination.exists():
        return
    temporary = destination.with_name(destination.name + '.writing')
    try:
        metadata = read(Path(source) / 'checkpoint.json')
        temporary.mkdir(parents=True)
        for name in ['checkpoint.json', *metadata['files']]:
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(Path(source) / name, target)
        for name, expected in metadata['files'].items():
            if digest(temporary / name) != expected:
                raise ValueError(f'Checkpoint hash mismatch: {name}')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def check_bindings(manifest):
    for path, expected in manifest['bindings'].items():
        if digest(path) != expected:
            raise ValueError(f'Monitor input/source changed: {path}')


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] in {'Z', 'X'} else fields[19]


def prepare(config_path, output, count, threads, trainer_pid):
    from tractor.generalist_data import read_episodes

    config = read(config_path)
    identity = process_identity(trainer_pid)
    if identity is None:
        raise ValueError('Actual trainer PID must be live')
    command = Path(f'/proc/{trainer_pid}/cmdline').read_bytes().split(b'\0')
    if b'tractor.biqwen_training' not in command or b'--config' not in command:
        raise ValueError('Expected actual BiQwen trainer process')
    process_config = Path(f'/proc/{trainer_pid}/cwd').resolve() / os.fsdecode(command[command.index(b'--config') + 1])
    if read(process_config) != config:
        raise ValueError('Trainer process belongs to another run')
    run = Path(config['output_dir'])
    plan = read(run / 'plan.json')
    if config != read(run / 'config.json'):
        raise ValueError('Configuration differs from active run')
    source = Path(config['dataset']['path']) / 'dev.episodes.jsonl'
    if digest(source) != plan['dataset_files']['dev.episodes.jsonl']:
        raise ValueError('Dev differs from frozen training plan')
    rows = [e.model_dump(mode='json') for e in read_episodes(source)]
    selected = select_rows(rows, count)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'snapshots').mkdir()
    episodes = root / 'dev.episodes.jsonl'
    episodes.write_text(''.join(json.dumps(row) + '\n' for row in selected))
    bindings = {str(Path(p).resolve()): h for p, h in plan['source_sha256'].items()}
    for path in [source, episodes, Path(config_path), Path(config['policy']), Path(__file__)]:
        bindings[str(path.resolve())] = digest(path)
    manifest = {
        'scope': 'Early diagnostic Dev subset only; no selection, release, calibration or test',
        'run': str(run), 'episodes': str(episodes), 'documents': len(selected),
        'full_dev_documents': len(rows), 'policy': str(Path(config['policy']).resolve()),
        'device': 'cpu', 'precision': 'fp32', 'gdn_backend': 'torch', 'threads': threads,
        'checkpoint_interval': config['checkpoint_updates'],
        'trainer_pid': trainer_pid, 'trainer_start_ticks': identity,
        'baseline': 'same hybrid architecture and initial weights at step zero',
        'bindings': bindings,
        'strata': dict(Counter('|'.join(row[k] for k in
                       ('language', 'family', 'artifact', 'text_origin', 'label_origin')) for row in selected)),
        'queue_policy': 'One active evaluation; retain earliest pending and newest pending. Record skipped steps.',
    }
    check_bindings(manifest)
    snapshot(run / 'checkpoints/step-00000000', root / 'snapshots/step-00000000')
    write(root / 'manifest.json', manifest)
    write(root / 'status.json', {'status': 'prepared', 'evaluated': [], 'skipped': []})
    return manifest


def aggregate(report):
    counts = Counter()
    excluded = []
    for key, score in report['scores'].items():
        if key.startswith('shape:'):
            if not score.get('reference_complete', True):
                excluded.append(key)
            else:
                counts.update(score['counts'])
    denominator = 2 * counts['field_tp'] + counts['field_fp'] + counts['field_fn']
    return {'field_f1': 2 * counts['field_tp'] / denominator if denominator else None,
            'counts': dict(counts), 'excluded_incomplete_groups': excluded}


def worker(root, checkpoint, output):
    import torch

    from tractor.document_eval import run
    from tractor.generalist_data import read_episodes
    from tractor.jsonio import loads
    from tractor.pipeline_evaluation import evaluate_pipeline

    manifest = read(root / 'manifest.json')
    check_bindings(manifest)
    torch.set_num_threads(manifest['threads'])
    torch.set_num_interop_threads(1)
    report = run(str(checkpoint), manifest['episodes'], str(output), device='cpu',
                 precision='fp32', policy=manifest['policy'], gdn_backend='torch')
    rows = list(read_episodes(manifest['episodes']))
    predictions = {p['id']: p for p in map(loads, (output / 'predictions.jsonl').read_text().splitlines())}
    errors = sum('error' in p for p in predictions.values())
    policy = loads(Path(manifest['policy']).read_text())
    languages = {}
    for lang in sorted({e.language for e in rows}):
        group = [e for e in rows if e.language == lang]
        languages[lang] = aggregate(evaluate_pipeline(group, {e.id: predictions[e.id] for e in group}, policy))
    check_bindings(manifest)
    write(output / 'summary.json', {
        'scope': manifest['scope'], 'step': read(checkpoint / 'checkpoint.json')['step'],
        'documents': len(rows), 'inference_errors': errors,
        'aggregate': aggregate(report), 'languages': languages,
        'scores': report['scores'], 'field_checks': report['field_checks'],
        'rejection_checks': report['rejection_checks'],
    })
    if errors:
        raise RuntimeError(f'{errors} inference errors; diagnostic is incomplete')


def compare(root, step):
    from tractor.document_training import regression_guard

    before = read(root / 'evaluations/step-00000000/metrics.json')
    after = read(root / f'evaluations/step-{step:08d}/metrics.json')
    keys = ['input_sha256', 'policy_sha256', 'evaluation_precision', 'gdn_backend',
            'evaluation_contract', 'pipeline_sources']
    if any(before[k] != after[k] for k in keys):
        raise ValueError('Unmatched baseline and checkpoint evaluation contract')
    result = {'scope': 'Early diagnostic only; sparse/missing groups do not prove release quality',
              'step': step, 'baseline': aggregate(before), 'candidate': aggregate(after),
              'regression_guard': regression_guard(after, before), 'groups': {}}
    for key, value in after['scores'].items():
        if key.startswith(('domain:', 'shape:')):
            old = before['scores'].get(key, {})
            valid = value.get('reference_complete', True) and old.get('reference_complete', True)
            result['groups'][key] = {'before': old.get('field_f1'), 'after': value.get('field_f1'),
                                     'reference_complete': valid, 'counts': value['counts']}
    write(root / f'evaluations/step-{step:08d}/comparison.json', result)
    write(root / 'latest-comparison.json', result)


def monitor(root):
    manifest = read(root / 'manifest.json')
    run = Path(manifest['run'])
    state = read(root / 'status.json')
    if state['status'] != 'prepared':
        raise ValueError('Monitor already started; inspect its status before restarting')
    state.update(status='running', pid=os.getpid(), active_step=None)
    active = None
    log = None
    first = True
    (root / 'evaluations').mkdir()
    try:
        while True:
            check_bindings(manifest)
            free = shutil.disk_usage(root).free
            state['disk_free_bytes'] = free
            if free < 10 * 1024**3:
                raise RuntimeError('Less than 10 GiB free; stop monitor without affecting trainer')
            training = read(run / 'status.json')
            if (training['status'] == 'running'
                    and process_identity(manifest['trainer_pid']) != manifest['trainer_start_ticks']):
                training = {**training, 'status': 'trainer_process_disappeared'}
            state['training_step'] = training['step']
            # Capture each completed periodic checkpoint before the trainer prunes it.
            for source in sorted((run / 'checkpoints').glob('step-[0-9]*')):
                if source.name.endswith('.writing'):
                    continue
                step = int(source.name[5:])
                if step == 0 or step in state['evaluated'] or step in state['skipped'] or step == state['active_step']:
                    continue
                if step % manifest['checkpoint_interval'] and training['status'] != 'completed':
                    continue
                snapshot(source, root / 'snapshots' / source.name)
            pending = sorted(p for p in (root / 'snapshots').glob('step-[0-9]*')
                             if int(p.name[5:]) != state['active_step'])
            # Bound disk usage even if CPU evaluation takes longer than one interval.
            for path in pending[1:-1]:
                state['skipped'].append(int(path.name[5:]))
                shutil.rmtree(path)
            if active is not None and active.poll() is not None:
                log.close()
                step = state['active_step']
                if active.returncode:
                    raise RuntimeError(f'Dev worker failed for step {step}; inspect evaluation log')
                if step:
                    compare(root, step)
                state['evaluated'].append(step)
                shutil.rmtree(root / f'snapshots/step-{step:08d}')
                state['active_step'] = None
                active = None
            pending = sorted((root / 'snapshots').glob('step-[0-9]*'))
            if active is None and pending:
                path = pending[0]
                step = int(path.name[5:])
                if first and step != 0:
                    raise ValueError('Matched step-zero baseline is required first')
                first = False
                state['active_step'] = step
                output = root / 'evaluations' / path.name
                log = (root / 'evaluations' / (path.name + '.log')).open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(manifest['threads']),
                           MKL_NUM_THREADS=str(manifest['threads']), TOKENIZERS_PARALLELISM='false')
                active = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output', str(root),
                                           '--worker', str(path), '--eval-output', str(output)],
                                          env=env, stdout=log, stderr=subprocess.STDOUT)
                state['worker_pid'] = active.pid
            state['updated_at'] = time.time()
            write(root / 'status.json', state)
            if active is None and not pending and training['status'] != 'running':
                state['status'] = 'completed' if training['status'] == 'completed' else 'trainer_stopped'
                break
            time.sleep(20)
    except BaseException as exc:
        state.update(status='failed', error=type(exc).__name__, message=str(exc))
        if active is not None and active.poll() is None:
            active.terminate()
            active.wait(timeout=30)
        raise
    finally:
        if log is not None:
            log.close()
        write(root / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/biqwen-fast-full.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--trainer-pid', type=int)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--eval-output', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.worker, args.eval_output)
    elif args.prepare:
        if args.threads < 1 or args.count < 1 or args.trainer_pid is None:
            parser.error('Positive thread/document counts and --trainer-pid required')
        prepare(args.config, args.output, args.count, args.threads, args.trainer_pid)
    else:
        with (args.output / 'monitor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            monitor(args.output)


if __name__ == '__main__':
    main()
