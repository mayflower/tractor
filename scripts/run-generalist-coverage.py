"""Bounded generalist learning diagnostic; independent Dev and no product replay quota."""

import argparse
import importlib.util
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tractor.coverage_assembly import sha
from tractor.episode_metrics import score_episode, summarize
from tractor.generalist_data import read_episodes
from tractor.generalist_mixture import bucket
from tractor.generalist_quality import quality_gate
from tractor.jsonio import dumps, loads
from tractor.training_report import rescore_predictions


def idle():
    spec = importlib.util.spec_from_file_location('pilot_runner', 'scripts/run-generalist-pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_idle_gpu()


def balanced_score(episodes, output):
    rows = {e.id: e for e in read_episodes(episodes)}
    counts = defaultdict(Counter)
    predictions = [loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
    if len(predictions) != len(rows) or {p['id'] for p in predictions} != rows.keys():
        raise ValueError('Incomplete Dev predictions')
    for prediction in predictions:
        episode = rows[prediction['id']]
        key = bucket(episode.model_dump())
        if key == 'classification':
            tasks = [task for task in episode.tasks if task.kind == 'classification']
            episode = episode.model_copy(update={'tasks': tasks, 'gold': {t.id: episode.gold[t.id] for t in tasks}})
        counts[key].update(score_episode(episode, prediction['prediction'], prediction['error']))
    scores = {key: summarize(value) for key, value in counts.items()}
    if set(scores) != set(loads(Path('configs/generalist-quality.json').read_text())['required_buckets']) or any(v['field_f1'] is None for v in scores.values()):
        raise ValueError('Missing general-task bucket score')
    return {'buckets': scores, 'macro_f1': sum(v['field_f1'] for v in scores.values()) / len(scores)}


def run(args):
    root, data = Path(args.output), Path(args.data)
    root.mkdir(parents=True, exist_ok=False)
    recipe = loads(Path('configs/generalist-conservative.json').read_text())
    quality_policy = loads(Path('configs/generalist-quality.json').read_text())
    recipe.update(output_dir=args.run, experiment_name='generalist-coverage', max_steps=100,
                  encoder_lr=4e-7, task_lr=8e-6, logging_steps=25)
    config = root / 'train-config.json'
    config.write_text(dumps(recipe, indent=2))
    policy = root / 'product-policy.json'
    policy.write_bytes(Path('reports/generalist/synthetic-coverage-short/product-policy.json').read_bytes())
    design = {'initial_model': args.model, 'initial_model_sha256': sha(Path(args.model) / 'model.safetensors'),
              'data': str(data), 'data_manifest_sha256': sha(data / 'manifest.json'),
              'train_sha256': sha(data / 'train.native.jsonl'), 'dev_sha256': sha(data / 'dev.episodes.jsonl'),
              'classification_dev': args.classification_dev, 'classification_dev_sha256': sha(args.classification_dev),
              'config_sha256': sha(config), 'updates': 100, 'seed': 42,
              'quality_policy': quality_policy, 'quality_policy_sha256': sha('configs/generalist-quality.json'),
              'objective': 'Equal-weight macro F1 across the declared general buckets; classification scored without auxiliary spans.',
              'product_floor': 0.90, 'qualification': 'Single-seed before/after learning diagnostic; no matched control or final-test claim. '
              'No dedicated product replay, no calibration/test inference, no latency benchmark, no release promotion.'}
    (root / 'design.json').write_text(dumps(design, indent=2))

    def command(name, argv):
        idle()
        print('START', name, flush=True)
        with (root / f'{name}.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'tractor.cli', *argv], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        print('DONE', name, flush=True)

    command('train', ['train', '--model', args.model, '--data', str(data), '--config', str(config), '--device', 'cuda'])
    evidence = loads(Path(args.run, 'training-evidence.json').read_text())
    if evidence['result']['total_steps'] != 100 or evidence['dataset_sha256'] != design['train_sha256']:
        raise ValueError('Incomplete or mismatched training')
    for key in recipe['tractor_required_groups']:
        if not all(evidence['parameters'][key][k] for k in ('changed', 'nonzero_finite_gradient')):
            raise ValueError(f'No learning evidence for {key}')
    scores = {}
    for arm, model in [('initial', args.model), ('candidate', str(Path(args.run) / 'final'))]:
        for name, episodes in [('general', str(data / 'dev.episodes.jsonl')), ('classification', args.classification_dev)]:
            output = root / arm / name
            command(f'{arm}-{name}', ['evaluate', '--model', model, '--episodes', episodes,
                                      '--split', 'dev', '--device', 'cuda', '--output', str(output)])
            rescore_predictions(episodes, output)
        scores[arm] = balanced_score(data / 'dev.episodes.jsonl', root / arm / 'general')
        c = loads((root / arm / 'classification/summary.json').read_text())
        scores[arm]['classification_tasks'] = {k: v['request_exact'] for k, v in c['classification_tasks'].items()}
        scores[arm]['classification_accuracy'] = c['task_kinds']['classification']['counts']['exact_fields'] / c['task_kinds']['classification']['counts']['fields']
    command('candidate-product', ['evaluate', '--model', str(Path(args.run) / 'final'),
                                  '--data', 'data/generalist-pilot-30k-product', '--selection', str(policy),
                                  '--split', 'dev', '--device', 'cuda', '--output', str(root / 'candidate/product')])
    product = loads((root / 'candidate/product/summary.json').read_text())[0]['metrics']
    result = {'status': 'complete', 'design_sha256': sha(root / 'design.json'), 'scores': scores,
              'generalist_quality_gate': quality_gate(scores['candidate'], quality_policy),
              'product_exact_requests': float(product['exact_request_rate']),
              'general_improved': scores['candidate']['macro_f1'] > scores['initial']['macro_f1'],
              'product_floor_passed': float(product['exact_request_rate']) >= .90,
              'qualification': design['qualification']}
    (root / 'summary.json').write_text(dumps(result, indent=2))
    print(dumps({k: v for k, v in result.items() if k != 'scores'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--classification-dev', default='data/generalist-extraction-scoped/dev.episodes.jsonl')
    run(parser.parse_args())
