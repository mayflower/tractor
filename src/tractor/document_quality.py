"""Domain/output-form extraction gates; missing real evidence fails explicitly."""

import math
from collections import Counter, defaultdict

from .episode_metrics import prediction_rows, score_episode, score_task, summarize
from .extraction_scope import in_scope


def evaluate(episodes, predictions, policy, *, score_fn=score_episode, task_score_fn=None):
    groups = defaultdict(Counter)
    if len(predictions) != len(episodes) or {e.id for e in episodes} != set(predictions):
        raise ValueError('Predictions must cover every episode exactly once')
    for e in episodes:
        if not in_scope(e.model_dump()):
            raise ValueError('Out-of-scope evaluation episode')
        if e.family not in policy['domains'] or e.artifact not in policy['shapes']:
            raise ValueError('Unmapped domain or output form')
        raw = predictions[e.id]
        counts = score_fn(e, raw.get('prediction', {}), raw.get('error'))
        origin = f'{e.text_origin}/{e.label_origin}'
        groups['domain', e.family, origin].update(counts)
        groups['shape', e.artifact, origin].update(counts)
        for task in e.tasks:
            if task.kind == 'records':
                prediction = {} if raw.get('error') else raw.get('prediction', {})
                groups['task', 'records', origin].update(
                    task_score_fn(e, task, prediction) if task_score_fn else
                    score_task(task, e.gold[task.id], prediction_rows(task, prediction)))
    scores = {':'.join(k): summarize(v) for k, v in groups.items()}
    checks = {}
    def add(key, metric, value):
        checks[f'{key}:{metric}'] = {'score': value, 'passed': value is not None
                                    and math.isfinite(value) and policy['minimum_exclusive'] < value <= 1}
    for axis, names in [('domain', policy['domains']), ('shape', policy['shapes'])]:
        for name in names:
            for origin in policy['required_origins']:
                key = ':'.join((axis, name, origin))
                score = scores.get(key, {})
                add(key, 'field_f1', score.get('field_f1') if score.get('reference_complete', True) else None)
                if axis == 'shape' and name == 'repeated_records':
                    records = scores.get(f'task:records:{origin}', {}).get('counts', {})
                    gold, predicted = records.get('gold_records', 0), records.get('predicted_records', 0)
                    p = records.get('exact_records', 0) / predicted if predicted else 0.0 if gold else None
                    r = records.get('exact_records', 0) / gold if gold else 0.0 if predicted else None
                    value = None if p is None or r is None else 2*p*r/(p+r) if p+r else 0.0
                    add(key, 'record_f1', value)
    return {'passed': all(v['passed'] for v in checks.values()), 'checks': checks,
            'scores': scores, 'qualification': policy['qualification']}


def main():
    import argparse
    from pathlib import Path

    from .coverage_assembly import sha
    from .generalist_data import read_episodes
    from .jsonio import dumps, loads

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--policy', default='configs/extraction-quality.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    episodes = list(read_episodes(args.episodes))
    predictions = {}
    for line in Path(args.predictions).read_text().splitlines():
        row = loads(line)
        if row['id'] in predictions:
            raise ValueError('Duplicate prediction ID')
        predictions[row['id']] = row
    result = evaluate(episodes, predictions, loads(Path(args.policy).read_text()))
    result['inputs'] = {name: {'path': path, 'sha256': sha(Path(path))}
                        for name, path in [('episodes', args.episodes), ('predictions', args.predictions), ('policy', args.policy)]}
    with Path(args.output).open('x') as stream:
        stream.write(dumps(result, indent=2) + '\n')
    print(dumps({'passed': result['passed'], 'output': args.output}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
