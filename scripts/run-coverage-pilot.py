"""Matched 0/20/40 percent coverage pilot; fixed updates and Dev-only scale gate."""

import argparse
import hashlib
import importlib.util
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tractor.jsonio import dumps, loads
from tractor.training_report import rescore_predictions

ROOT = Path('reports/generalist/synthetic-coverage')
DATA = Path('data/synthetic-coverage-pilot')
INITIAL = 'models/tractor'
LR_DIVISOR = 5.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_rows(root):
    with Path(root, 'train.episodes.jsonl').open() as ef, Path(root, 'train.native.jsonl').open() as nf:
        for e, n in zip(ef, nf, strict=True):
            row = loads(e)
            if row['split'] != 'train' or row['text'] != loads(n)['input']:
                raise ValueError('Unpaired or non-training input')
            yield row, e, n


def prepare():
    manifest = loads((DATA / 'manifest.json').read_text())
    if manifest['status'] != 'validated':
        raise ValueError('Synthetic native validation must pass first')
    ROOT.mkdir(parents=True, exist_ok=False)
    # Deterministic per-source reservoir; no evaluation data can enter any arm.
    rng = random.Random(731)
    pools, seen = defaultdict(list), Counter()
    for row, e, n in paired_rows('data/public-extraction-200k'):
        key = row['source']
        seen[key] += 1
        if len(pools[key]) < 2000:
            pools[key].append((e, n))
        else:
            j = rng.randrange(seen[key])
            if j < 2000:
                pools[key][j] = (e, n)
    public = []
    for i in range(2000):
        for key in sorted(pools):
            if i < len(pools[key]):
                public.append(pools[key][i])
    replay = [(e, n) for row, e, n in paired_rows('data/generalist-retention-30k')
              if row['source'] == 'product_replay']
    rng.shuffle(replay)
    synthetic = defaultdict(list)
    for row, e, n in paired_rows(DATA):
        synthetic[row['family']].append((e, n))
    for rows in synthetic.values():
        rng.shuffle(rows)
    config = loads(Path('configs/generalist-conservative.json').read_text())
    design = {'initial_model': INITIAL, 'initial_model_sha256': sha(Path(INITIAL, 'model.safetensors')),
              'updates_per_arm': 300, 'effective_batch_size': 32, 'seed': 42,
              'synthetic_manifest_sha256': sha(DATA / 'manifest.json'),
              'generator_sha256': manifest['generator_sha256'],
              'gates': manifest['config']['scale_requires'], 'arms': {},
              'qualification': '10% fixed product replay in every arm. 0/20/40% refer to NEW coverage data. '
              'The remaining 90/70/50% is public human extraction. Same source-balanced public ordering; '
              'same initialization, updates, optimizer, thresholds and Dev sets. Single-seed pilot.'}
    (ROOT / 'product-policy.json').write_text(dumps({'thresholds': {'text': .25, 'quantity': .25, 'anchor': .5}}))
    for percent in (0, 20, 40):
        arm = f'S{percent}'
        output = Path('data/synthetic-coverage-arms', arm)
        output.mkdir(parents=True, exist_ok=False)
        synth_count = percent * 100
        rows = public[:9000 - synth_count] + replay[:1000]
        for family, weight in manifest['config']['family_weights'].items():
            rows += synthetic[family][:synth_count * weight // 100]
        if len(rows) != 10000:
            raise ValueError('Insufficient unique source data')
        random.Random(42).shuffle(rows)
        for kind, pos in [('episodes', 0), ('native', 1)]:
            (output / f'train.{kind}.jsonl').write_text(''.join(row[pos] for row in rows))
            # Trainer eval disabled; retain a fixed native Dev file for its contract.
            (output / f'dev.{kind}.jsonl').write_bytes((DATA / f'dev.{kind}.jsonl').read_bytes())
        groups = ['encoder', 'boundary_head', 'record_decoder']
        if percent:
            groups.append('relation_scorer')
        recipe = dict(config, output_dir=f'runs/synthetic-coverage/{arm}', experiment_name=f'coverage-{arm}',
                      max_steps=300, num_epochs=2, tractor_required_groups=groups)
        recipe_path = ROOT / f'{arm}-config.json'
        recipe_path.write_text(dumps(recipe, indent=2))
        design['arms'][arm] = {'data': str(output), 'config': str(recipe_path),
                               'config_sha256': sha(recipe_path), 'train_sha256': sha(output / 'train.native.jsonl'),
                               'new_synthetic_percent': percent}
    design['evaluation'] = {
        'synthetic': {'episodes': str(DATA / 'dev.episodes.jsonl')},
        'public': {'episodes': 'data/public-extraction-200k/dev.episodes.jsonl'},
        'product': {'data': 'data/generalist-pilot-30k-product'},
    }
    design['evaluation_sha256'] = {k: sha(v['episodes']) for k, v in design['evaluation'].items() if 'episodes' in v}
    (ROOT / 'design.json').write_text(dumps(design, indent=2))


def refine():
    source = Path('reports/generalist/synthetic-coverage')
    if ROOT == source:
        raise ValueError('Refinement requires a new --root')
    design = loads((source / 'design.json').read_text())
    ROOT.mkdir(parents=True, exist_ok=False)
    design['recipe_change'] = f'Predeclared {LR_DIVISOR}x lower encoder/task learning rates than the initial pilot. Compare matched S0/S20; all data, seeds, update budgets and gates unchanged. Dev-only adaptive recipe search.'
    design['parent_design_sha256'] = sha(source / 'design.json')
    design['arms'] = {k: v for k, v in design['arms'].items() if k in {'S0', 'S20'}}
    for arm, spec in design['arms'].items():
        recipe = loads(Path(spec['config']).read_text())
        recipe['encoder_lr'] = float(recipe['encoder_lr']) / LR_DIVISOR
        recipe['task_lr'] = float(recipe['task_lr']) / LR_DIVISOR
        recipe['output_dir'] = f'runs/{ROOT.name}/{arm}'
        recipe['experiment_name'] = f'coverage-conservative-{arm}'
        path = ROOT / f'{arm}-config.json'
        path.write_text(dumps(recipe, indent=2))
        spec['config'] = str(path)
        spec['config_sha256'] = sha(path)
    (ROOT / 'product-policy.json').write_bytes((source / 'product-policy.json').read_bytes())
    (ROOT / 'design.json').write_text(dumps(design, indent=2))


def idle():
    spec = importlib.util.spec_from_file_location('pilot_runner', 'scripts/run-generalist-pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_idle_gpu()


def run_command(name, args):
    with (ROOT / (name + '.log')).open('x') as log:
        subprocess.run([sys.executable, '-m', 'tractor.cli', *args],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def execute():
    design = loads((ROOT / 'design.json').read_text())
    initial = design['initial_model']
    if sha(Path(initial, 'model.safetensors')) != design['initial_model_sha256']:
        raise ValueError('Initial checkpoint changed')
    for name, spec in design['evaluation'].items():
        if 'episodes' in spec and sha(spec['episodes']) != design['evaluation_sha256'][name]:
            raise ValueError('Dev data changed after preregistration')
    for arm, spec in design['arms'].items():
        recipe = loads(Path(spec['config']).read_text())
        run = Path(recipe['output_dir'])
        if sha(spec['config']) != spec['config_sha256'] or sha(Path(spec['data'], 'train.native.jsonl')) != spec['train_sha256']:
            raise ValueError('Pilot inputs changed after preregistration')
        if not (run / 'training-evidence.json').exists():
            idle()
            print('TRAIN', arm, flush=True)
            run_command(arm + '-train', ['train', '--model', initial, '--data', spec['data'],
                                        '--config', spec['config'], '--device', 'cuda'])
        evidence = loads((run / 'training-evidence.json').read_text())
        if evidence['result']['total_steps'] != design['updates_per_arm'] or evidence['dataset_sha256'] != spec['train_sha256']:
            raise ValueError('Incomplete or mismatched training evidence')
        for group in evidence['required_learning_groups']:
            if not all(evidence['parameters'][group][k] for k in ('changed', 'nonzero_finite_gradient')):
                raise ValueError('Required module did not learn')
        for name, spec_eval in design['evaluation'].items():
            output = ROOT / arm / name
            if (output / 'summary.json').exists():
                continue
            idle()
            print('EVALUATE', arm, name, flush=True)
            extra = ['--episodes', spec_eval['episodes']] if 'episodes' in spec_eval else [
                '--data', spec_eval['data'], '--selection', str(ROOT / 'product-policy.json')]
            run_command(arm + '-' + name, ['evaluate', '--model', str(run / 'final'), *extra,
                                          '--split', 'dev', '--device', 'cuda', '--output', str(output)])
            if 'episodes' in spec_eval:
                rescore_predictions(spec_eval['episodes'], output)
    report()


def report():
    design = loads((ROOT / 'design.json').read_text())
    scores = {}
    for arm in design['arms']:
        syn = loads((ROOT / arm / 'synthetic/summary.json').read_text())
        public = loads((ROOT / arm / 'public/summary.json').read_text())
        product = loads((ROOT / arm / 'product/summary.json').read_text())[0]['metrics']
        scores[arm] = {'synthetic_macro_f1': syn['macro_field_f1'], 'public_macro_f1': public['macro_field_f1'],
                       'product_exact_request': float(product['exact_request_rate']),
                       'synthetic_families': syn['families'], 'public_families': public['families'],
                       'synthetic_overall': syn['overall'], 'public_overall': public['overall']}
    base, gates = scores['S0'], design['gates']
    eligible = []
    for arm in design['arms']:
        if arm == 'S0':
            continue
        row = scores[arm]
        row['gates'] = {
            'synthetic_improved': row['synthetic_macro_f1'] >= base['synthetic_macro_f1'] + gates['synthetic_macro_f1_improvement'],
            'public_retained': row['public_macro_f1'] >= base['public_macro_f1'] - gates['public_macro_f1_max_drop'],
            'product_floor': row['product_exact_request'] >= gates['product_exact_request_floor'],
        }
        if all(row['gates'].values()):
            eligible.append(arm)
    selected = max(eligible, key=lambda a: scores[a]['synthetic_macro_f1']) if eligible else None
    result = {'status': 'complete', 'selected': selected, 'scale_allowed': bool(selected), 'scores': scores,
              'generator_sha256': design['generator_sha256'], 'design_sha256': sha(ROOT / 'design.json'),
              'pilot_data': str(DATA), 'pilot_manifest_sha256': design['synthetic_manifest_sha256'],
              'qualification': 'Dev-only single-seed gate. No calibration/test inference or model promotion.'}
    (ROOT / 'summary.json').write_text(dumps(result, indent=2))
    lines = ['# Synthetic coverage Dev pilot', '', design['qualification'], '',
             '| Arm | Synthetic macro F1 | Public macro F1 | Product exact requests | False answers on absent synthetic fields |',
             '|---|---:|---:|---:|---:|']
    for arm, row in scores.items():
        false_answers = row['synthetic_overall']['false_answer_rate_absent']
        lines.append(f"| {arm} | {row['synthetic_macro_f1']:.4f} | {row['public_macro_f1']:.4f} | "
                     f"{row['product_exact_request']:.4f} | {false_answers:.4f} |")
    lines.extend(['', f'Selected: {selected}. Scale allowed: {bool(selected)}.', '',
                  'These are bounded, single-seed Dev experiments. Synthetic templates are not proof of natural-document reliability. '
                  'The gate concerns dataset expansion, not deployment approval. Inspect absent-field errors and individual task families. '
                  'No calibration or test predictions were used.'])
    (ROOT / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run', 'report', 'refine'])
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--initial-model', default=INITIAL)
    parser.add_argument('--lr-divisor', type=float, default=LR_DIVISOR)
    args = parser.parse_args()
    ROOT = Path(args.root)
    INITIAL = args.initial_model
    LR_DIVISOR = args.lr_divisor
    if not 0 < LR_DIVISOR < float('inf'):
        parser.error('--lr-divisor must be positive and finite')
    {'prepare': prepare, 'run': execute, 'report': report, 'refine': refine}[args.phase]()
