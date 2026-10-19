"""Run prepared, bounded pilot arms; refuse a busy shared GPU and existing partial outputs."""

import argparse
import csv
import io
import subprocess
import sys
import time
from pathlib import Path

from tractor.jsonio import dumps, loads
from tractor.training_report import pilot_report, rescore_predictions


def require_idle_gpu():
    idle_samples = 0
    for _ in range(15):
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        busy = len(rows) != 1 or any(int(r[0]) > 1024 or int(r[1]) > 5 for r in rows)
        idle_samples = 0 if busy else idle_samples + 1
        if idle_samples == 3:
            return
        time.sleep(1)
    raise RuntimeError('Shared GPU is occupied; no GPU work started. Retry after the other job finishes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True)
    parser.add_argument('--phase', choices=['train', 'evaluate', 'report'], required=True)
    args = parser.parse_args()
    design_path = Path(args.design)
    root, design = design_path.parent, loads(design_path.read_text())
    if args.phase == 'report':
        print(dumps(pilot_report(root), indent=2))
        return

    def run(name, command):
        print('START', name, flush=True)
        with (root / (name + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-m', 'tractor.cli', *command],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print('DONE', name, flush=True)

    if args.phase == 'evaluate':
        models = dict(design.get('baseline_models', {'B0': 'models/base', 'B1': 'models/tractor'}))
        for arm, spec in design['arms'].items():
            config = loads(Path(spec['config']).read_text())
            run_root = Path(config['output_dir'])
            evidence = loads((run_root / 'training-evidence.json').read_text())
            if evidence['result']['total_steps'] != design['updates_per_arm'] or evidence['dataset_sha256'] != spec['train_sha256']:
                raise ValueError(f'{arm}: incomplete or mismatched training evidence')
            if any(not evidence['parameters'][key]['changed'] or
                   not evidence['parameters'][key]['nonzero_finite_gradient']
                   for key in evidence['required_learning_groups']):
                raise ValueError(f'{arm}: missing learning evidence')
            models[arm] = str(run_root / 'final')
        for arm, model in models.items():
            if design['evaluation_device'] == 'cuda':
                require_idle_gpu()
            for kind, extra in [
                ('native', ['--episodes', design['native_dev']]),
                ('product', ['--data', design['product_dev'], '--selection', str(root / 'product-policy.json')]),
            ]:
                output = root / arm / kind
                if (output / 'summary.json').exists():
                    continue
                if output.exists():
                    raise ValueError(f'Partial evaluation exists: {output}')
                run(arm + '-' + kind, ['evaluate', '--model', model, *extra, '--split', 'dev',
                    '--device', design['evaluation_device'], '--output', str(output)])
                if kind == 'native':
                    rescore_predictions(design['native_dev'], output)
        print(dumps(pilot_report(root), indent=2))
        return

    pending_evaluations = []
    for arm, spec in design['arms'].items():
        config = loads(Path(spec['config']).read_text())
        run_root = Path(config['output_dir'])
        evidence = run_root / 'training-evidence.json'
        if not evidence.exists():
            require_idle_gpu()
            run(arm + '-train', ['train', '--model', spec['initial_model'], '--data', spec['data'],
                                '--config', spec['config'], '--device', 'cuda'])
        evidence_data = loads(evidence.read_text())
        if evidence_data['result']['total_steps'] != design['updates_per_arm']:
            raise ValueError(f'{arm}: requested optimizer budget was not completed')
        if evidence_data['dataset_sha256'] != spec['train_sha256']:
            raise ValueError(f'{arm}: training evidence has a different dataset')
        if any(not evidence_data['parameters'][k]['changed'] or
               not evidence_data['parameters'][k]['nonzero_finite_gradient']
               for k in evidence_data['required_learning_groups']):
            raise ValueError(f'{arm}: incomplete learning evidence')
        model = str(run_root / 'final')
        for kind, extra in [
            ('native', ['--episodes', design['native_dev']]),
            ('product', ['--data', design['product_dev'], '--selection', str(root / 'product-policy.json')]),
        ]:
            output = root / arm / kind
            if (output / 'summary.json').exists():
                continue
            if output.exists():
                raise ValueError(f'Partial evaluation exists: {output}; preserve and inspect before retrying')
            command = [sys.executable, '-m', 'tractor.cli', 'evaluate', '--model', model,
                       *extra, '--split', 'dev', '--device', design['evaluation_device'],
                       '--output', str(output)]
            if design['evaluation_device'] != 'cpu':
                raise ValueError('Concurrent evaluation must use CPU while the next GPU arm trains')
            name = arm + '-' + kind
            print('START', name, flush=True)
            with (root / (name + '.log')).open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            pending_evaluations.append((process, kind, output, name))
    failures = []
    for process, kind, output, name in pending_evaluations:
        if process.wait() != 0:
            failures.append(name)
            continue
        if kind == 'native':
            rescore_predictions(design['native_dev'], output)
        print('DONE', name, flush=True)
    if failures:
        raise RuntimeError('Evaluation failed: ' + ', '.join(failures))
    print(dumps(pilot_report(root), indent=2))


if __name__ == '__main__':
    main()
