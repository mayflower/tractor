"""Independent-provider Silver review of synthetic documents; durable shared budget."""

import argparse
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .compare import parse_provider, provider_request
from .jsonio import dumps, loads
from .teacher import reserve, settle_success


def load_key(path):
    if os.environ.get('OPENAI_API_KEY'):
        return
    for line in Path(path).expanduser().read_text().splitlines():
        m = re.fullmatch(r'\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*?)\s*', line)
        if m:
            value = m[1]
            if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ['OPENAI_API_KEY'] = value
            return
    raise ValueError('OPENAI_API_KEY missing')


def review(record, root, env='~/.env', ledger='reports/training/teacher-budget.json',
           prompt_path='configs/document-critic.txt'):
    load_key(env)
    prompt = Path(prompt_path).read_text()
    episode = record['episode']
    gold = {task: {'complete': value['complete'], 'records': [
        {field: {'status': answer['status'], 'items': [{'text': span['text'],
            'context': episode['text'][max(0, span['start']-30):span['end']+30]} for span in answer['items']]}
         for field, answer in row.items()} for row in value['records']]}
        for task, value in episode['gold'].items()}
    payload = {'id': episode['id'], 'text': episode['text'], 'tasks': episode['tasks'], 'gold': gold,
               'offsets': 'All original character offsets have already been mechanically verified; do not count characters.',
               'domain': episode['family'], 'genre': record['plan']['document_genre'],
               'response_format': 'Return the requested JSON object only.'}
    url, headers, body = provider_request('openai', payload, prompt, 4096)
    body['reasoning'] = {'effort': 'low'}
    body['text'] = {'format': {'type': 'json_object'}}
    key = hashlib.sha256(dumps(body, sort_keys=True).encode()).hexdigest()
    directory = Path(root) / 'independent_reviews'
    directory.mkdir(exist_ok=True)
    path = directory / (key + '.json')
    if path.exists():
        return loads(path.read_text())
    if path.with_suffix('.failed').exists():
        return {'id': episode['id'], 'passed': False, 'issues': ['previous_provider_failure'], 'request_sha256': key}
    bound = len(dumps(body).encode()) + 4096
    limits = loads(Path('configs/generalist-budget.json').read_text())
    reservation = reserve(ledger, limits, bound, 4096, short_luna=True)
    try:
        with httpx.Client(timeout=120) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
        raw = loads(response.text)
        if raw.get('usage'):
            settle_success(ledger, reservation, bound, 4096, raw['usage'])
        result = parse_provider('openai', raw)
        if result.get('id') != episode['id'] or not isinstance(result.get('issues'), list):
            raise ValueError('Invalid independent review')
        passed = result.get('passed') is True and result['issues'] == []
        record = {'id': episode['id'], 'passed': passed, 'issues': result['issues'],
                  'request_sha256': key, 'usage': raw.get('usage'), 'model': body['model']}
        path.write_text(dumps(record, indent=2))
        return record
    except Exception as exc:
        path.with_suffix('.failed').write_text(dumps({'id': episode['id'], 'error': type(exc).__name__}))
        return {'id': episode['id'], 'passed': False, 'issues': [type(exc).__name__], 'request_sha256': key}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--env', default='~/.env')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    rows = [loads(p.read_text()) for p in sorted((Path(args.root) / 'reviewed').glob('*.json'))]
    rows = [r for r in rows if r['accepted_by_teacher']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda r: review(r, args.root, args.env), rows))
    print(dumps({'reviewed': len(results), 'accepted': sum(r['passed'] for r in results)}))


if __name__ == '__main__':
    main()
