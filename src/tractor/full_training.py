"""Full-corpus balancing and in-training generalist Dev measurements."""

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .coverage_assembly import sha
from .episode_metrics import score_episode, summarize
from .episodes import compile_schema
from .extraction_scope import in_scope
from .generalist_data import read_episodes
from .generalist_mixture import bucket
from .generalist_quality import quality_gate
from .jsonio import dumps, loads


def exposure_counts(counts, weights, multiple=800):
    if set(counts) != set(weights) or sum(weights.values()) != 100 or any(w <= 0 for w in weights.values()):
        raise ValueError('Positive quotas must cover exactly the present buckets and sum to 100')
    total = math.ceil(max(counts[k] * 100 / weights[k] for k in counts) / multiple) * multiple
    target = {k: total * weights[k] // 100 for k in counts}
    if sum(target.values()) != total or any(target[k] < counts[k] for k in counts):
        raise ValueError('Invalid full-coverage exposure schedule')
    return target


def prepare(source, output, weights_path, dev_source):
    source, root = Path(source), Path(output)
    manifest = loads((source / 'manifest.json').read_text())
    weights = loads(Path(weights_path).read_text())['weights']
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete full corpus')
    if manifest.get('training_ready') is False:
        raise ValueError('Source corpus has unresolved qualification findings')
    for name in ('train.episodes.jsonl', 'train.native.jsonl'):
        if sha(source / name) != manifest['files'][name]:
            raise ValueError('Training source changed')
    # Keep native rows and compact metadata; no expanded Python copies of every episode.
    pools, originals = defaultdict(list), []
    with (source / 'train.episodes.jsonl').open() as ef, (source / 'train.native.jsonl').open() as nf:
        for eline, nline in zip(ef, nf, strict=True):
            e, n = loads(eline), loads(nline)
            if not in_scope(e):
                raise ValueError('Out-of-scope task in training corpus')
            if e['split'] != 'train' or e['text'] != n['input']:
                raise ValueError('Unpaired or non-training source')
            i = len(originals)
            originals.append((e['id'], nline, bucket(e), e['source'], e['language']))
            pools[bucket(e)].append(i)
    counts = {k: len(v) for k, v in pools.items()}
    target = exposure_counts(counts, weights)
    rng = random.Random(20261016)
    schedule = list(range(len(originals)))
    for key, indices in pools.items():
        strata = defaultdict(list)
        for i in indices:
            _, _, _, source_name, language = originals[i]
            strata[source_name, language].append(i)
        names = sorted(strata)
        rng.shuffle(names)
        for values in strata.values():
            rng.shuffle(values)
        for j in range(target[key] - len(indices)):
            values = strata[names[j % len(names)]]
            schedule.append(values[(j // len(names)) % len(values)])
    rng.shuffle(schedule)
    if set(schedule) != set(range(len(originals))):
        raise ValueError('Full-corpus coverage lost')
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'train.native.jsonl').open('w') as nf, (root / 'train.identities.jsonl').open('w') as ids:
        for i in schedule:
            ident, native, key, source_name, language = originals[i]
            nf.write(native)
            ids.write(dumps({'id': ident, 'bucket': key}) + '\n')
    dm = loads(Path(dev_source, 'manifest.json').read_text())
    for name in ('dev.native.jsonl', 'dev.episodes.jsonl'):
        if sha(Path(dev_source, name)) != dm['files'][name]:
            raise ValueError('Dev source changed')
        Path(root, name).write_bytes(Path(dev_source, name).read_bytes())
    report = {'status': 'complete', 'source': str(source), 'source_manifest_sha256': sha(source / 'manifest.json'),
              'unique_training_examples': len(originals), 'exposures_per_epoch': len(schedule), 'weights': weights,
              'unique_by_bucket': counts, 'exposures_by_bucket': target, 'seed': 20261016,
              'dev_source': str(dev_source), 'dev_manifest_sha256': sha(Path(dev_source, 'manifest.json')),
              'qualification': 'Every source training row included at least once per epoch. Repetitions balance general tasks; '
              'they are exposures, not additional independent examples. No dedicated product replay. Dev only; no final split data.',
              'files': {p.name: sha(p) for p in root.glob('*.jsonl')}}
    (root / 'manifest.json').write_text(dumps(report, indent=2))
    print(dumps(report, indent=2), flush=True)


def score_general_dev(episodes, predictions, policy):
    buckets, classes = defaultdict(Counter), defaultdict(Counter)
    for episode, raw in zip(episodes, predictions, strict=True):
        key = bucket(episode.model_dump())
        tasks = episode.tasks if key != 'classification' else [t for t in episode.tasks if t.kind == 'classification']
        subset = episode.model_copy(update={'tasks': tasks, 'gold': {t.id: episode.gold[t.id] for t in tasks}})
        buckets[key].update(score_episode(subset, raw))
        for task in episode.tasks:
            if task.kind == 'classification':
                subset = episode.model_copy(update={'tasks': [task], 'gold': {task.id: episode.gold[task.id]}})
                name = f'{episode.family}:{task.id}:{episode.text_origin}/{episode.label_origin}'
                classes[name].update(score_episode(subset, raw))
    scores = {'buckets': {key: summarize(value) for key, value in buckets.items()},
              'classification_tasks': {key: summarize(value)['request_exact'] for key, value in classes.items()}}
    gate = quality_gate(scores, policy)
    values = [v['score'] for v in gate['checks'].values()]
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError('Incomplete generalist Dev measurement')
    scores['macro_f1'] = sum(scores['buckets'][k]['field_f1'] for k in policy['required_buckets']) / len(policy['required_buckets'])
    return scores, gate, min(values)


def dev_callback(path, policy_path, output, step):
    """Select the weakest required Dev score, not training loss or product performance."""
    episodes = list(read_episodes(path))
    if any(e.split != 'dev' or not in_scope(e.model_dump()) for e in episodes):
        raise ValueError('Checkpoint selection only accepts Dev')
    policy = loads(Path(policy_path).read_text())
    root = Path(output) / 'generalist-dev'
    root.mkdir(parents=True, exist_ok=False)

    def measure(model, _dataset):
        import torch

        predictions = []
        with torch.no_grad():
            for episode in episodes:
                schema = compile_schema(episode.tasks, .5)
                raw = model.batch_extract([episode.text], [schema], threshold=.5,
                                          include_spans=True, include_confidence=True)[0]
                predictions.append(raw)
        scores, gate, minimum = score_general_dev(episodes, predictions, policy)
        current_step = step()
        report = {'step': current_step, 'scores': scores, 'gate': gate,
                  'selection_metric': minimum, 'dev_sha256': sha(path), 'policy_sha256': sha(policy_path),
                  'qualification': 'Small frozen balanced Dev for checkpoint selection; full Dev evaluation still required.'}
        (root / f'step-{current_step}.json').write_text(dumps(report, indent=2))
        with (root / f'step-{current_step}.predictions.jsonl').open('w') as stream:
            for episode, raw in zip(episodes, predictions, strict=True):
                stream.write(dumps({'id': episode.id, 'prediction': raw}) + '\n')
        return {'eval_general_min_score': minimum, 'eval_general_macro_f1': scores['macro_f1'],
                'eval_general_gate_passed': float(gate['passed'])}
    return measure


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/generalist-extraction-scoped')
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', default='configs/generalist-mixture.json')
    parser.add_argument('--dev-source', default='data/generalist-scoped-pilot')
    args = parser.parse_args()
    prepare(args.source, args.output, args.weights, args.dev_source)
