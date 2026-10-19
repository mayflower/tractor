"""Resumable independent semantic review of public-source windows, under shared budget."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .compare import parse_provider, provider_request
from .coverage_assembly import sha
from .document_critic import load_key
from .jsonio import dumps, loads
from .teacher import reserve, settle_success


def review_batch(rows, root, env='~/.env'):
    load_key(env)
    prompt = Path('configs/public-source-review.txt').read_text()
    payload = {'format': 'Return JSON only.', 'examples': [{
        'id': row['id'], 'text': row['text'], 'tasks': row['tasks'],
        'gold': {key: [{field: {'status': answer['status'], 'values': [s['text'] for s in answer['items']]}
                        for field, answer in record.items()} for record in gold['records']]
                 for key, gold in row['gold'].items()}} for row in rows]}
    url, headers, body = provider_request('openai', payload, prompt, 4096)
    body['reasoning'] = {'effort': 'low'}
    body['text'] = {'format': {'type': 'json_object'}}
    key = hashlib.sha256(dumps(body, sort_keys=True).encode()).hexdigest()
    path = Path(root) / 'reviews' / f'{key}.json'
    if path.exists():
        return loads(path.read_text())
    if path.with_suffix('.failed').exists():
        return {'results': [{'id': r['id'], 'passed': False, 'issues': ['cached_provider_failure']} for r in rows]}
    bound = len(dumps(body).encode()) + 4096
    limits = loads(Path('configs/generalist-budget.json').read_text())
    reservation = reserve('reports/training/teacher-budget.json', limits, bound, 4096, short_luna=True)
    try:
        with httpx.Client(timeout=180) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
        raw = loads(response.text)
        if raw.get('usage'):
            settle_success('reports/training/teacher-budget.json', reservation, bound, 4096, raw['usage'])
        result = parse_provider('openai', raw)
        answers = result['results']
        if len(answers) != len(rows) or {r['id'] for r in rows} != {r['id'] for r in answers}:
            raise ValueError('Review IDs differ from input')
        if any(not isinstance(r['passed'], bool) or not isinstance(r['issues'], list) or r['passed'] == bool(r['issues']) for r in answers):
            raise ValueError('Invalid review verdict')
        result.update(usage=raw.get('usage'), request_sha256=key, model=body['model'])
        tmp = path.with_suffix('.tmp')
        tmp.write_text(dumps(result, indent=2))
        tmp.replace(path)
        return result
    except Exception as exc:
        path.with_suffix('.failed').write_text(dumps({'error': type(exc).__name__}))
        return {'results': [{'id': r['id'], 'passed': False, 'issues': [type(exc).__name__]} for r in rows]}


def audit_sample(source, output, workers=16, env='~/.env'):
    from collections import Counter, defaultdict
    root = Path(output)
    (root / 'reviews').mkdir(parents=True, exist_ok=True)
    rows = [loads(line) for line in Path(source).open()]
    batches = [rows[i:i+8] for i in range(0, len(rows), 8)]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for report in pool.map(lambda batch: review_batch(batch, root, env), batches):
            results.extend(report['results'])
    verdicts = {r['id']: r for r in results}
    counts = defaultdict(Counter)
    for row in rows:
        name = row['source'] if row['source'] != 'swde' else ':'.join(row['source_group'].split(':')[:2])
        counts[name]['reviewed'] += 1
        counts[name]['passed'] += verdicts[row['id']]['passed']
    report = {'source_sha256': sha(Path(source)), 'reviewer': 'GPT-6 Luna independent machine review',
              'counts': dict(counts), 'results': results, 'training_ready': False,
              'qualification': 'Adapter sample review only. No human certification and no automatic bulk qualification.'}
    (root / 'sample-report.json').write_text(dumps(report, indent=2))
    print(dumps(report['counts']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--env', default='~/.env')
    audit_sample(**vars(p.parse_args()))
