"""Realistic authored-document pipeline: facts -> cached renderer -> independent review -> native validation.

Nothing is imported into training solely because a renderer returned valid JSON.
"""

import argparse
import hashlib
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .compare import provider_request
from .coverage_assembly import sha
from .document_scenarios import brief
from .episodes import Episode, audit_episodes
from .extraction_scope import in_scope
from .jsonio import dumps, loads
from .teacher import parse_teacher, reserve, settle_success

SPLITS = ('train', 'dev', 'calibration', 'test')
STYLES = {
    'train': ['concise professional vocabulary', 'formal but readable wording',
              'practical operational vocabulary', 'neutral factual tone',
              'appropriate technical terminology', 'plain language'],
    'dev': ['natural regional vocabulary', 'short sentences with understated headings'],
    'calibration': ['recipient-appropriate terminology', 'compact prose with conventional structure'],
    'test': ['precise contemporary professional wording', 'matter-of-fact language with natural abbreviations'],
}
GENRES = {'invoices': 'supplier invoice', 'contracts': 'commercial lease agreement excerpt',
          'forms': 'completed administrative application form',
          'banking': 'completed loan application with an account statement excerpt if transactions are supplied',
          'communication': 'customer email about an order or service appointment',
          'contacts': 'company website contact and profile page', 'insurance': 'completed insurance claim notification',
          'clinical': 'clinical discharge letter with medication list if medications are supplied',
          'careers': 'candidate CV', 'products': 'supplier catalogue product entry',
          'metadata': 'business letter', 'financial_reports': 'annual report excerpt',
          'research': 'scientific measurement report excerpt', 'news': 'short business news article',
          'logistics': 'packing list and consignment note', 'operations': 'completed maintenance inspection report'}

MARKER = re.compile(r'<v(\d+)>(.*?)</v\1>', re.DOTALL)
OPTIONAL = {
    'invoices': {'iban', 'due', 'code'}, 'contracts': {'end', 'notice'},
    'forms': {'birth_date', 'submitted', 'amount'}, 'banking': {'employer', 'income', 'identity'},
    'communication': {'contract', 'callback', 'email', 'address', 'appointment'},
    'contacts': {'employees', 'phone', 'industry'}, 'insurance': {'deductible', 'coverage'},
    'clinical': {'birth_date', 'followup'}, 'careers': {'salary', 'phone', 'degree'},
    'products': {'certification', 'weight'}, 'metadata': {'case', 'number'},
    'financial_reports': {'auditor', 'published'}, 'research': {'sample_size', 'date'},
    'news': {'ticker'}, 'logistics': {'container'}, 'operations': {'limit', 'time'},
}


def load_env(path):
    """Read only required provider variables; never execute or print environment files."""
    for line in Path(path).expanduser().read_text().splitlines():
        match = re.fullmatch(r'\s*(?:export\s+)?(ANTHROPIC_API_KEY|ANTHROPIC_WORKSPACE_ID)\s*=\s*(.*?)\s*', line)
        if match:
            value = match[2]
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(match[1], value)


def plan_item(spec, split, index, seed, shape=None, record_count=None):
    rng = random.Random(f'{seed}:{split}:{spec["domain"]}:{index}')
    language = 'de' if (index // 10 + index) % 2 == 0 else 'en'
    shape = shape or ('single_object' if index % 10 < 6 else 'repeated_records' if index % 10 < 9 else 'entity_list')
    facts, absent, tasks, row_specs = [], [], [], {}

    def add_task(ident, definitions, count, kind, anchor=None):
        chosen = list(definitions)
        if spec['domain'] == 'news' and shape == 'repeated_records' and kind == 'entities':
            chosen = [k for k in chosen if k not in {'person', 'amount', 'location'}]
        # Native feasibility starts small; all domain fields rotate across cases.
        if kind == 'entities':
            rng.shuffle(chosen)
            chosen = chosen[:(3 if shape == 'repeated_records' else min(6, len(chosen)))]
        task = {'id': ident, 'kind': kind, 'description': spec['document'],
                'fields': {k: definitions[k]['description'] for k in chosen}}
        if anchor:
            task['anchor'] = anchor
        tasks.append(task)
        row_specs[ident] = []
        for row_index in range(count):
            row = {}
            for key in chosen:
                path = f'{ident}.{row_index}.{key}'
                optional = key in OPTIONAL[spec['domain']]
                if key != anchor and optional and rng.random() < .12:
                    row[key] = None
                    absent.append({'path': path, 'description': definitions[key]['description']})
                    continue
                value = definitions[key]['examples'][0 if language == 'de' else -1]
                # Distinct natural-looking references, not opaque scenario IDs in document text.
                if key in {'reference', 'number', 'order', 'customer', 'contract', 'policy', 'claim', 'statement', 'shipment'} and any(c.isdigit() for c in value):
                    value = re.sub(r'\d{3,}', lambda m: str(rng.randrange(10 ** (len(m[0])-1), 10 ** len(m[0]))), value)
                if count > 1:
                    # Different item variants instead of invented row IDs. All facts are rendered explicitly.
                    options = definitions[key]['examples']
                    if key == anchor:
                        value = options[0 if language == 'de' else -1]
                        value += (' – Ausführung ' if language == 'de' else ' – variant ') + chr(65 + row_index)
                    elif key in {'quantity', 'packages'}:
                        value = f'{3 + row_index * 2} ' + ('Stück' if language == 'de' else 'units')
                marker = len(facts)
                row[key] = marker
                facts.append({'marker': f'v{marker}', 'path': path, 'description': definitions[key]['description'], 'value': value})
            row_specs[ident].append(row)
    add_task('document', spec['header'], 1, 'entities')
    # A narrow extraction query must not force an implausibly incomplete document.
    # Non-queried header facts supply context; they are never added to the gold schema.
    for key, definition in spec['header'].items():
        if key not in tasks[0]['fields']:
            facts.append({'marker': f'v{len(facts)}', 'path': f'context.{key}',
                          'description': definition['description'],
                          'value': definition['examples'][0 if language == 'de' else -1]})
    if spec['domain'] == 'invoices':
        facts.append({'marker': f'v{len(facts)}', 'path': 'context.tax_rate',
                      'description': 'VAT percentage consistent with the fictional supplier country and goods, from 0 to 25 percent',
                      'value': '19 %'})
    if shape == 'repeated_records' or spec['domain'] == 'invoices':
        count = record_count if record_count is not None else ((1, 2, 3, 4, 6, 8)[index % 6] if shape == 'repeated_records' else 2 + index % 2)
        if not 0 <= count <= 12:
            raise ValueError('Record count must be between zero and twelve')
        add_task('records', spec['record'], count, 'records', spec['anchor'])
        if shape != 'repeated_records':
            tasks.pop()
            row_specs.pop('records')
    ident = f'document:{seed}:{split}:{spec["domain"]}:{index}'
    return {'id': ident, 'split': split, 'text_origin': 'synthetic', 'domain': spec['domain'],
            'language': language, 'shape': shape, 'document_genre': GENRES[spec['domain']],
            'voice': STYLES[split][index % len(STYLES[split])],
            'editorial_variation': 'Use the customary structure of this document genre, not a cover note or an explanation of fields.',
            'authorship_brief': brief(spec['domain'], index, split),
            'facts': facts, 'absent_fields': absent, 'tasks': tasks, 'rows': row_specs,
            'schema_sha256': hashlib.sha256(dumps(spec, sort_keys=True).encode()).hexdigest()}


def prepare(output, train_count=30000, heldout_per_domain=200, config='configs/document-build.json'):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    config = loads(Path(config).read_text())
    counts = Counter()
    with (root / 'plan.jsonl').open('x') as stream:
        for domain, weight in config['domains'].items():
            spec = loads(Path('schemas/extraction', domain + '.json').read_text())
            for split in SPLITS:
                n = train_count * weight // 100 if split == 'train' else heldout_per_domain
                for index in range(n):
                    item = plan_item(spec, split, index, config['seed'])
                    stream.write(dumps(item) + '\n')
                    counts[f'{split}:{domain}'] += 1
    manifest = {'status': 'planned_not_rendered', 'counts': dict(counts), 'config': config,
                'files': {'plan.jsonl': sha(root / 'plan.jsonl')},
                'qualification': 'No training texts yet. Unique scenario IDs do not prove document diversity. '
                'Renderer, independent semantic/realism review, native validation and human sample audit required.'}
    (root / 'manifest.json').write_text(dumps(manifest, indent=2))
    return manifest


def response_key(item, prompt, max_tokens):
    return hashlib.sha256(dumps({'model': 'claude-haiku-5-5', 'input': item, 'prompt': prompt,
                                 'max_tokens': max_tokens}, sort_keys=True).encode()).hexdigest()


def request(item, prompt, directory, max_tokens, ledger, limits):
    directory.mkdir(parents=True, exist_ok=True)
    key = response_key(item, prompt, max_tokens)
    path = directory / (key + '.json')
    if path.exists():
        return loads(path.read_text())['result']
    if path.with_suffix('.failed').exists():
        raise ValueError('Previously failed request; inspect before explicit retry')
    url, headers, body = provider_request('anthropic', item, prompt, max_tokens)
    input_bound = len(dumps(body).encode()) + 4096
    reservation = reserve(ledger, limits, input_bound, max_tokens, short_haiku=True)
    try:
        with httpx.Client(timeout=120) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
        raw = loads(response.text)
        if raw.get('usage'):
            settle_success(ledger, reservation, input_bound, max_tokens, raw['usage'])
        result = parse_teacher(raw)
        if result.get('id') != item['id']:
            raise ValueError('Response ID mismatch')
        tmp = path.with_suffix('.tmp')
        tmp.write_text(dumps({'result': result, 'usage': raw.get('usage'), 'request_sha256': key,
                              'model': body['model'], 'status': 'unverified_silver'}, indent=2))
        tmp.replace(path)
        return result
    except Exception as exc:
        path.with_suffix('.failed').write_text(dumps({'id': item['id'], 'error_type': type(exc).__name__}))
        raise


def apply_facts(item, proposal):
    """Permit diverse fictional values, never changes to schema/identity/split/absence."""
    if proposal.get('id') != item['id']:
        raise ValueError('Fact scenario ID mismatch')
    facts = proposal.get('facts', [])
    if len(facts) != len(item['facts']):
        raise ValueError('Incomplete fact proposal')
    values = {f['marker']: f['value'] for f in facts}
    if len(values) != len(facts) or set(values) != {f['marker'] for f in item['facts']}:
        raise ValueError('Changed fact identities')
    if any(not isinstance(v, str) or not v.strip() or '<v' in v or len(v) > 300 for v in values.values()):
        raise ValueError('Invalid proposed value')
    result = dict(item)
    result['facts'] = [dict(f, value=values[f['marker']]) for f in item['facts']]
    return result


def reconstruct(item, result):
    if result['id'] != item['id']:
        raise ValueError('Wrong scenario ID')
    marked = result['marked_text']
    plain, cursor, spans = '', 0, {}
    for m in MARKER.finditer(marked):
        i = int(m[1])
        if i in spans or i >= len(item['facts']) or m[2] != item['facts'][i]['value']:
            raise ValueError('Changed, duplicated or unknown fact marker')
        plain += marked[cursor:m.start()]
        start = len(plain)
        plain += m[2]
        spans[i] = {'status': 'present', 'items': [{'start': start, 'end': len(plain), 'text': m[2]}]}
        cursor = m.end()
    plain += marked[cursor:]
    if len(spans) != len(item['facts']) or re.search(r'</?v\d+', plain):
        raise ValueError('Missing or malformed markers')
    if len(plain.split()) < 40 or plain.lstrip().startswith(('{', '[')):
        raise ValueError('Not a substantial document text')
    gold = {task: {'complete': True, 'records': [
        {k: {'status': 'absent'} if i is None else spans[i] for k, i in row.items()} for row in rows]}
        for task, rows in item['rows'].items()}
    # Repeated literal mentions of the same entity are complete supervision, not
    # first-occurrence-only labels. Ambiguous values shared by roles stay untouched.
    value_counts = Counter(f['value'].casefold() for f in item['facts'])
    identity_fields = {'supplier', 'customer', 'company', 'party_a', 'party_b', 'applicant', 'patient',
                       'clinician', 'insured', 'contact', 'name', 'sender', 'recipient', 'author', 'auditor',
                       'reference', 'number', 'policy', 'claim', 'account', 'statement', 'order', 'contract',
                       'shipment', 'email', 'phone', 'callback', 'address', 'website', 'title', 'date', 'report_date'}
    for task in item['tasks']:
        if task['kind'] != 'entities':
            continue
        for key, marker in item['rows'][task['id']][0].items():
            if marker is None or key not in identity_fields:
                continue
            value = item['facts'][marker]['value']
            if value_counts[value.casefold()] != 1:
                continue
            matches = list(re.finditer(r'(?<!\w)' + re.escape(value) + r'(?!\w)', plain, re.IGNORECASE))
            if matches:
                gold[task['id']]['records'][0][key]['items'] = [{'start': m.start(), 'end': m.end(), 'text': m[0]} for m in matches]
    episode = Episode.model_validate({'id': item['id'], 'source_group': item['id'], 'derivation_group': item['id'],
        'source': 'tractor_authored_documents', 'original_split': item['split'], 'split': item['split'],
        'family': item['domain'], 'language': item['language'], 'artifact': item['shape'],
        'annotation': 'silver', 'text_origin': 'synthetic', 'label_origin': 'teacher',
        'text': plain, 'tasks': item['tasks'], 'gold': gold})
    if not in_scope(episode.model_dump()):
        raise ValueError('Out-of-scope task')
    return episode


def restore_unique_markers(item, result):
    """Recover omitted annotation tags only; never alter the authored document text.

    A unique literal occurrence is not proof of its role. Semantic review remains
    mandatory after this operation; ambiguous or absent literals stay rejected.
    """
    marked = result['marked_text']
    existing = list(MARKER.finditer(marked))
    present = {f'v{m[1]}' for m in existing}
    insertions = []
    for fact in item['facts']:
        if fact['marker'] in present:
            continue
        matches = [m for m in re.finditer(re.escape(fact['value']), marked)
                   if not any(m.start() < e.end() and m.end() > e.start() for e in existing)]
        if len(matches) != 1:
            continue
        m = matches[0]
        if any(m.start() < end and m.end() > start for start, end, _ in insertions):
            raise ValueError('Overlapping unmarked fact candidates')
        insertions.append((m.start(), m.end(), fact['marker']))
    for start, end, marker in sorted(insertions, reverse=True):
        marked = marked[:start] + f'<{marker}>' + marked[start:end] + f'</{marker}>' + marked[end:]
    return dict(result, marked_text=marked), [marker for _, _, marker in insertions]


def approved(review):
    return (all(review.get(k) is True for k in ('supported', 'correct_roles', 'complete', 'natural_document', 'no_reasoning_required'))
            and review.get('issues') == [])


def check_scale_gate(root, pending):
    """A small successful API response sample does not authorize bulk dataset scaling."""
    reviewed = list((root / 'reviewed').glob('*.json'))
    if len(reviewed) + pending <= 256:
        return
    path = root / 'scale-audit.json'
    if not path.exists():
        raise ValueError('Scaling beyond 256 candidates requires a passing realism sample audit')
    audit = loads(path.read_text())
    if (audit.get('passed') is not True or audit.get('unresolved_findings') != []
            or audit.get('recipe_sha256') != sha(root / 'recipe.json')
            or not audit.get('reviewer')):
        raise ValueError('Realism scale audit is failed, incomplete or belongs to another recipe')
    reviewed_rows = {r['id']: r for p in reviewed if (r := loads(p.read_text()))['accepted_by_teacher']}
    ids = audit.get('reviewed_ids', [])
    if not ids or any(i not in reviewed_rows for i in ids):
        raise ValueError('Scale audit must identify the inspected accepted documents')
    samples = [reviewed_rows[i]['episode'] for i in ids]
    if ({e['family'] for e in samples} != set(loads((root / 'manifest.json').read_text())['config']['domains'])
            or {e['artifact'] for e in samples} != {'single_object', 'repeated_records', 'entity_list'}
            or {e['language'] for e in samples} != {'de', 'en'}):
        raise ValueError('Scale sample does not cover every domain, output form and language')


def render(output, limit=32, env=None, budget='configs/generalist-budget.json', ledger='reports/training/teacher-budget.json', workers=4, wanted_cells=None):
    if limit < 1:
        raise ValueError('Candidate limit must be positive')
    if env:
        load_env(env)
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError('ANTHROPIC_API_KEY missing')
    root = Path(output)
    manifest = loads((root / 'manifest.json').read_text())
    if sha(root / 'plan.jsonl') != manifest['files']['plan.jsonl']:
        raise ValueError('Plan changed')
    prompts = {k: Path(f'configs/document-{k}.txt').read_text() for k in ('facts', 'render', 'review')}
    recipe = {'prompts': prompts, 'code_sha256': sha(Path(__file__)),
              'facts_code_sha256': sha(Path(__file__).with_name('document_facts.py')),
              'scenario_code_sha256': sha(Path(__file__).with_name('document_scenarios.py')),
              'plan_sha256': manifest['files']['plan.jsonl']}
    recipe_path = root / 'recipe.json'
    if recipe_path.exists() and loads(recipe_path.read_text()) != recipe:
        raise ValueError('Authoring recipe changed; use a new output directory')
    if not recipe_path.exists():
        recipe_path.write_text(dumps(recipe, indent=2))
    limits = loads(Path(budget).read_text())
    # Round robin by domain/split for a useful first sample, rather than one-domain prefix.
    pools = {}
    for line in (root / 'plan.jsonl').open():
        item = loads(line)
        pools.setdefault((item['split'], item['domain']), []).append(item)
    order = [0, 7, 9, 1, 6, 2, 8, 3, 4, 5]
    indices = sorted(range(max(map(len, pools.values()))), key=lambda i: (i // 10, order.index(i % 10)))
    schedule = [pool[i] for i in indices for pool in pools.values() if i < len(pool)]

    def process(item):
        dest = root / 'reviewed' / (hashlib.sha256(item['id'].encode()).hexdigest() + '.json')
        if dest.exists():
            return
        seed_item = item
        fact_request = dict(item, facts=[{k: v for k, v in f.items() if k != 'value'} for f in item['facts']])
        proposed = request(fact_request, prompts['facts'], root / 'facts', 4096, ledger, limits)
        item = apply_facts(item, proposed)
        from .document_facts import coherent_invoice
        item = coherent_invoice(item)
        result = request(item, prompts['render'], root / 'renders', 2048, ledger, limits)
        try:
            result, restored = restore_unique_markers(item, result)
            episode = reconstruct(item, result)
            payload = {'id': item['id'], 'document': result['marked_text'], 'facts': item['facts'],
                       'absent_fields': item['absent_fields'], 'tasks': item['tasks'], 'genre': item['document_genre']}
            review = request(payload, prompts['review'], root / 'reviews', 1024, ledger, limits)
            record = {'id': item['id'], 'accepted_by_teacher': approved(review), 'review': review,
                      'episode': episode.model_dump(), 'plan': item, 'seed_plan': seed_item,
                      'restored_annotation_markers': restored,
                      'render_request_sha256': response_key(item, prompts['render'], 2048),
                      'review_request_sha256': response_key(payload, prompts['review'], 1024)}
        except (ValueError, KeyError) as exc:
            record = {'id': item['id'], 'accepted_by_teacher': False, 'error': str(exc)}
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(dumps(record, indent=2))
        print(dumps({'id': item['id'], 'accepted_by_teacher': record['accepted_by_teacher']}), flush=True)

    pending = [item for item in schedule if (wanted_cells is None or (item['split'], item['domain'], item['shape'], item['language']) in wanted_cells)
               and not (root / 'reviewed' / (hashlib.sha256(item['id'].encode()).hexdigest() + '.json')).exists()][:limit]
    check_scale_gate(root, len(pending))
    def safe_process(item):
        try:
            process(item)
        except Exception as exc:
            if 'Teacher budget exhausted' in str(exc):
                raise
            dest = root / 'reviewed' / (hashlib.sha256(item['id'].encode()).hexdigest() + '.json')
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(dumps({'id': item['id'], 'accepted_by_teacher': False,
                                  'error': type(exc).__name__, 'stage': 'provider_or_render_failure'}))
            print(dumps({'id': item['id'], 'accepted_by_teacher': False, 'error_type': type(exc).__name__}), flush=True)

    if not 1 <= workers <= 32:
        raise ValueError('Document workers must be between 1 and 32')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(safe_process, pending))
    return {'processed': len(pending), 'qualification': 'Teacher-reviewed candidates, not native-validated or realism-certified training data.'}


def validate(output, model='models/tractor', validation_output=None):
    import torch

    from .generalist_data import validate_native
    from .model import Extractor

    torch.set_num_threads(2)
    root = Path(output)
    extractor = Extractor(model, 'cpu')
    rows, rejections = [], []
    for path in sorted((root / 'reviewed').glob('*.json')):
        row = loads(path.read_text())
        if not row['accepted_by_teacher']:
            rejections.append({'id': row['id'], 'stage': 'semantic_realism_review', 'reason': row.get('review', row.get('error'))})
            continue
        episode = Episode.model_validate(row['episode'])
        try:
            native, tokens = validate_native(episode, extractor)
        except Exception as exc:
            rejections.append({'id': episode.id, 'stage': 'native', 'reason': type(exc).__name__ + ':' + str(exc)[:200]})
            continue
        rows.append((episode, native, tokens))
    audit = audit_episodes([e for e, _, _ in rows])
    dest = Path(validation_output) if validation_output else root / 'validated'
    dest.mkdir(exist_ok=False)
    for split in SPLITS:
        with (dest / f'{split}.episodes.jsonl').open('w') as ef, (dest / f'{split}.native.jsonl').open('w') as nf:
            for e, n, _ in rows:
                if e.split == split:
                    ef.write(dumps(e.model_dump()) + '\n')
                    nf.write(dumps(n) + '\n')
    (dest / 'rejections.json').write_text(dumps(rejections, indent=2))
    report = {'status': 'sample_native_validated_pending_human_realism_audit', 'training_ready': False,
              'counts': dict(Counter(e.split for e, _, _ in rows)),
              'coverage': dict(Counter(f'{e.split}:{e.family}:{e.artifact}:{e.language}' for e, _, _ in rows)),
              'audit': audit, 'rejected': len(rejections), 'reviewed': len(rows) + len(rejections),
              'max_tokens': max((t for _, _, t in rows), default=0), 'plan_sha256': sha(root / 'plan.jsonl'),
              'files': {p.name: sha(p) for p in dest.glob('*.jsonl')},
              'qualification': 'No heldout inference. Semantic and realism review uses a separate Teacher call, '
              'not an independent human. Never claim 30000 accepted from a small sample.'}
    (dest / 'manifest.json').write_text(dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', required=True)
    p.add_argument('--train-count', type=int, default=30000)
    p.add_argument('--heldout-per-domain', type=int, default=200)
    p = sub.add_parser('render')
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=32)
    p.add_argument('--env')
    p.add_argument('--workers', type=int, default=4)
    p = sub.add_parser('validate')
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='models/tractor')
    p.add_argument('--validation-output', help='New immutable output directory for a later validation snapshot')
    args = vars(parser.parse_args())
    command = args.pop('command')
    print(dumps(globals()[command](**args), indent=2))


if __name__ == '__main__':
    main()
