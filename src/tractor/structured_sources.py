"""Conservative text-to-JSON adapters: exact spans, projected fields, explicit Silver provenance."""

import hashlib
import re
from collections import Counter
from decimal import Decimal

from .episodes import Episode, group_split, text_key
from .jsonio import loads

CV_FIELDS = {
    'current_title': 'Current or most recent job title explicitly stated in the resume',
    'previous_titles': 'Up to three most recent previous job titles explicitly stated in the resume',
    'current_company': 'Current or most recent employer explicitly named in the resume',
    'previous_companies': 'Up to three most recent previous employers explicitly named in the resume',
    'core_skills': 'Up to three primary skills explicitly named in the resume',
    'secondary_skills': 'Up to three secondary skills explicitly named in the resume',
    'tools': 'Tools or technologies explicitly named in the resume',
    'location': 'Geographic location explicitly stated in the resume',
}
EXCLUDED = {'summary', 'seniority', 'primary_domain', 'industries', 'leadership_experience',
            'years_experience', 'key_achievements'}
ANCHORS = ('name', 'item', 'description', 'id', 'title', 'company', 'product')


def evidence(text, value):
    """Reject ambiguous occurrences and subword/partial-number accidental matches."""
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        raise ValueError('non_extractive_value')
    if not isinstance(value, (str, int, Decimal)):
        raise ValueError('unsupported_value_type')
    literal = str(value)
    if not literal or not literal.strip():
        raise ValueError('empty_value')
    pattern = (r'(?<!\w)' if literal[0].isalnum() else '') + re.escape(literal)
    pattern += r'(?!\w)' if literal[-1].isalnum() else ''
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError('missing_or_nonunique_evidence')
    match = matches[0]
    if not isinstance(value, str):
        # Never match 20 inside 20.5, 1,020 or a signed/scientific numeric expression.
        before, after = text[:match.start()], text[match.end():]
        if re.search(r'[\d.,+\-]$', before) or re.match(r'[.,]\d|[eE][+\-]?\d', after):
            raise ValueError('partial_numeric_evidence')
    return {'start': match.start(), 'end': match.end(), 'text': match.group()}


def answer(text, value, description):
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError('unverified_empty_annotation')
    spans = [evidence(text, item) for item in values]
    if any(len(s['text']) >= 3 and s['text'] in description for s in spans):
        raise ValueError('answer_in_schema')
    if len({(s['start'], s['end']) for s in spans}) != len(spans):
        raise ValueError('duplicate_target')
    return {'status': 'present', 'items': spans}


def parse_text_json(row):
    match = re.fullmatch(r'### Input: (.*?)\n### Schema: (.*?)\n### Output: (.*)', row['text'], re.S)
    if not match:
        raise ValueError('unsupported_text_json_format')
    text, schema, output = match.groups()
    schema, output = loads(schema), loads(output)
    if not isinstance(schema, dict) or not isinstance(output, dict):
        raise ValueError('root_must_be_object')
    return text, schema, output


def parse_resume(row):
    messages = re.findall(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>', row['text'], re.S)
    if [role for role, _ in messages] != ['system', 'user', 'assistant']:
        raise ValueError('unsupported_resume_conversation')
    text = messages[1][1]
    if not text.startswith('Resume:\n'):
        raise ValueError('missing_resume_prefix')
    text = text[len('Resume:\n'):]
    completion = messages[2][1].strip()
    # Accept only an empty reasoning wrapper; never treat model reasoning as source evidence.
    completion = re.sub(r'^<think>\s*</think>\s*', '', completion)
    output = loads(completion)
    if not isinstance(output, dict):
        raise ValueError('root_must_be_object')
    return text, {key: 'string' for key in CV_FIELDS}, output


def adapt(row, source, ident):
    text, schema, output = parse_text_json(row) if source == 'text_json' else parse_resume(row)
    counters, fields, values, tasks, gold = Counter(), {}, {}, [], {}

    def scalar(path, value):
        name = path[-1]
        description = CV_FIELDS[name] if source == 'resume_json' else ' / '.join(path).replace('_', ' ')
        try:
            target = answer(text, value, description)
        except ValueError as exc:
            counters['field_rejected:' + str(exc)] += 1
            return
        key = 'f' + str(len(fields))
        fields[key], values[key] = description, target
        counters['fields_accepted'] += 1

    def walk(spec, result, path=()):
        if not isinstance(spec, dict) or not isinstance(result, dict):
            counters['unsupported_object'] += 1
            return
        for name, value in result.items():
            if name not in spec or name in EXCLUDED or (source == 'resume_json' and name not in CV_FIELDS):
                counters['field_excluded_by_contract'] += 1
                continue
            if value is None or value == []:
                counters['unverified_null_or_empty_omitted'] += 1
                continue
            if isinstance(value, dict):
                walk(spec[name], value, (*path, name))
            elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
                records(spec[name], value, (*path, name))
            else:
                scalar((*path, name), value)

    def records(spec, rows, path):
        if not isinstance(spec, list) or len(spec) != 1 or not isinstance(spec[0], dict) or not rows:
            counters['record_rejected:unsupported_schema'] += 1
            return
        spec = spec[0]
        if any(not isinstance(row, dict) or set(row) != set(spec) for row in rows):
            counters['record_rejected:incomplete_or_mixed_rows'] += 1
            return
        anchor = next((key for key in ANCHORS if key in spec), None)
        retained, answers = {}, [{} for _ in rows]
        for name in spec:
            if name in EXCLUDED:
                continue
            desc = ' / '.join((*path, name)).replace('_', ' ')
            try:
                targets = [answer(text, row[name], desc) for row in rows]
            except ValueError as exc:
                counters['record_field_rejected:' + str(exc)] += 1
                continue
            retained[name] = desc
            for record, target in zip(answers, targets, strict=True):
                record[name] = target
        if anchor not in retained or len(retained) < 2 or any(len(r[anchor]['items']) != 1 for r in answers):
            counters['record_rejected:no_supported_anchor_or_fields'] += 1
            return
        anchors = [r[anchor]['items'][0]['start'] for r in answers]
        if len(anchors) != len(set(anchors)):
            counters['record_rejected:shared_anchor'] += 1
            return
        # Keep only representable identifiers, without guessing original nested objects.
        mapping = {key: 'r' + str(i) for i, key in enumerate(retained)}
        task_id = 'records' + str(len(tasks))
        tasks.append({'id': task_id, 'kind': 'records', 'description': ' / '.join(path),
                      'anchor': mapping[anchor], 'fields': {mapping[k]: v for k, v in retained.items()}})
        gold[task_id] = {'complete': True, 'records': [{mapping[k]: v for k, v in r.items()} for r in answers]}
        counters['record_tasks_accepted'] += 1

    walk(schema, output)
    if fields:
        tasks.insert(0, {'id': 'document', 'kind': 'entities', 'description': 'Explicit document fields', 'fields': fields})
        gold['document'] = {'complete': True, 'records': [values]}
    if not tasks:
        return None, counters
    # Group identical normalized source text across both supplements, not by row index.
    group = 'structured_document:' + text_key(text)
    episode = Episode.model_validate({
        'id': source + ':' + str(ident), 'source': source, 'source_group': group,
        'derivation_group': group, 'original_split': 'train', 'split': group_split(group),
        'family': 'structured_records' if any(t['kind'] == 'records' for t in tasks) else 'document_fields',
        'language': 'en', 'artifact': 'resume' if source == 'resume_json' else 'text_json_document',
        'annotation': 'silver', 'text_origin': 'real' if source == 'resume_json' else 'unknown',
        'label_origin': 'teacher' if source == 'resume_json' else 'unknown',
        'text': text, 'tasks': tasks, 'gold': gold,
    })
    return episode, counters


def fingerprint(episode):
    from .jsonio import dumps
    return hashlib.sha256(dumps({'text': episode.text, 'tasks': [t.model_dump() for t in episode.tasks],
        'gold': {k: v.model_dump() for k, v in episode.gold.items()}}, sort_keys=True).encode()).hexdigest()


def group_near_documents(episodes):
    """Group lexical variants before assigning new splits; never inspect predictions."""
    from collections import defaultdict

    from .public_extraction import word_shingles

    parent, all_shingles, index = {}, {}, defaultdict(set)

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for e in episodes:
        key = text_key(e.text)
        if key in parent:
            continue
        parent[key] = key
        sh = word_shingles(e.text)
        candidates = set()
        for item in sorted(sh, key=lambda x: hashlib.sha256(x.encode()).digest())[:8]:
            candidates.update(index.get(item, ()))
        for other in candidates:
            previous = all_shingles[other]
            if len(sh & previous) / min(len(sh), len(previous)) >= .9:
                a, b = find(key), find(other)
                parent[max(a, b)] = min(a, b)
        all_shingles[key] = sh
        for item in sh:
            index[item].add(key)
    grouped = []
    for e in episodes:
        group = 'structured_document:' + find(text_key(e.text))
        grouped.append(e.model_copy(update={'source_group': group, 'derivation_group': group,
                                           'split': group_split(group)}))
    return grouped


def integrate(config, output, model):
    import shutil
    from itertools import chain
    from pathlib import Path

    import pyarrow.parquet as pq
    import torch

    from .episodes import audit_episodes
    from .generalist_data import read_episodes, validate_native
    from .jsonio import dumps
    from .model import Extractor
    from .public_extraction import ProtectedIndex, checksum
    from .types import TractorError

    root, base = Path(output), Path(config['base'])
    root.mkdir(parents=True, exist_ok=False)
    base_manifest = loads((base / 'manifest.json').read_text())
    for name, sha in base_manifest['files'].items():
        if checksum(base / name) != sha:
            raise ValueError('Base corpus hash mismatch: ' + name)
    protected = ProtectedIndex()
    paths = [str(base / (s + '.episodes.jsonl')) for s in ('train', 'dev')] + config['protected_episodes']
    print('Index protected source texts', flush=True)
    for path in paths:
        for e in read_episodes(path):
            protected.add(e)
    report, episodes, seen = {}, [], set()
    with (root / 'rejections.jsonl').open('x') as rejected, (root / 'projection-audit.jsonl').open('x') as projected:
        for spec in config['sources']:
            source, path = spec['source'], Path(spec['path'])
            if checksum(path) != spec['sha256']:
                raise ValueError('Source hash mismatch')
            counts = Counter()
            if source == 'text_json':
                rows = path.read_bytes().splitlines()
            else:
                rows = (r for b in pq.ParquetFile(path).iter_batches(batch_size=128) for r in b.to_pylist())
            for ident, row in enumerate(rows):
                counts['raw_rows'] += 1
                try:
                    if isinstance(row, bytes):
                        row = loads(row.decode('utf-8'))
                    e, dropped = adapt(row, source, ident)
                    counts.update(dropped)
                    projected.write(dumps({'source': source, 'row': ident, 'counts': dict(dropped)}) + '\n')
                    if e is None:
                        raise ValueError('no_supported_extractive_fields')
                    reason = protected.overlaps(e)
                    if reason:
                        raise ValueError(reason)
                    key = fingerprint(e)
                    if key in seen:
                        raise ValueError('duplicate_example')
                    seen.add(key)
                    episodes.append(e)
                    counts['structurally_accepted'] += 1
                except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                    reason = type(exc).__name__ + ':' + str(exc)[:150]
                    counts['rejected:' + reason] += 1
                    rejected.write(dumps({'source': source, 'row': ident, 'reason': reason}) + '\n')
            report[source] = dict(counts)
            print('ADAPTED', source, counts['raw_rows'], counts['structurally_accepted'], flush=True)
    del protected
    episodes = group_near_documents(episodes)
    # Model validation is CPU-only; this is target reconstruction, not model evaluation.
    torch.set_num_threads(1)
    extractor = Extractor(model, 'cpu')
    accepted, validation = [], Counter()
    with (root / 'native-rejections.jsonl').open('x') as rejected:
        for e in episodes:
            try:
                native, tokens = validate_native(e, extractor)
                accepted.append((e, native))
                validation[e.source + ':accepted'] += 1
                validation['joint_tokens'] += tokens
                if len(accepted) % 500 == 0:
                    print('NATIVE', len(accepted), flush=True)
            except (ValueError, RuntimeError, TractorError) as exc:
                reason = type(exc).__name__ + ':' + str(exc)[:150]
                validation[e.source + ':rejected:' + reason] += 1
                rejected.write(dumps({'id': e.id, 'reason': reason}) + '\n')
    supplement = root / 'supplement'
    supplement.mkdir()
    for split in ('train', 'dev', 'calibration', 'test'):
        rows = [(e, n) for e, n in accepted if e.split == split]
        (supplement / (split + '.episodes.jsonl')).write_text(''.join(dumps(e.model_dump()) + '\n' for e, _ in rows))
        (supplement / (split + '.native.jsonl')).write_text(''.join(dumps(n) + '\n' for _, n in rows))
    for split in ('train', 'dev'):
        for kind in ('episodes', 'native'):
            name = split + '.' + kind + '.jsonl'
            shutil.copyfile(base / name, root / name)
            with (root / name).open('ab') as target, (supplement / name).open('rb') as source:
                shutil.copyfileobj(source, target)
    result = {
        'status': 'complete', 'config': config, 'base_manifest_sha256': checksum(base / 'manifest.json'),
        'base_train_examples': base_manifest['train_examples'],
        'train_examples': base_manifest['train_examples'] + sum(e.split == 'train' for e, _ in accepted),
        'dev_examples': base_manifest['dev_examples'] + sum(e.split == 'dev' for e, _ in accepted),
        'supplement_sources_by_split': {s: dict(Counter(e.source for e, _ in accepted if e.split == s))
                                        for s in ('train', 'dev', 'calibration', 'test')},
        'supplement_train_origins': dict(Counter(e.text_origin + '/' + e.label_origin for e, _ in accepted if e.split == 'train')),
        'supplement_train_families': dict(Counter(e.family for e, _ in accepted if e.split == 'train')),
        'supplement_unique_texts': len({text_key(e.text) for e, _ in accepted}),
        'supplement_source_groups': len({e.source_group for e, _ in accepted}),
        'source_import': report, 'native_validation': dict(validation),
        'native_validation_model_sha256': checksum(Path(model) / 'model.safetensors'),
        'audit': audit_episodes(chain(read_episodes(root / 'train.episodes.jsonl'), read_episodes(root / 'dev.episodes.jsonl'),
                                     (e for e, _ in accepted if e.split in {'calibration', 'test'}))),
        'qualification': 'Projected exact-span fields only. Silver annotations; text-json text and label origin unknown, resume labels LLM-produced. Nulls, missing fields and failed evidence are omitted from schema, never negatives. Arrays of objects preserve supported record pairing; no recursive-schema support claimed. Long source texts rejected, not gold-cropped. No model training or upload.',
        'files': {str(p.relative_to(root)): checksum(p) for p in chain(root.glob('*.jsonl'), supplement.glob('*.jsonl'))},
    }
    (root / 'manifest.json').write_text(dumps(result, indent=2))
    print(dumps({k: result[k] for k in ('train_examples', 'dev_examples', 'supplement_sources_by_split', 'supplement_train_families')}, indent=2), flush=True)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/structured-supplement.json')
    parser.add_argument('--output', default='data/generalist-extraction-integrated')
    parser.add_argument('--model', default='models/tractor')
    args = parser.parse_args()
    integrate(loads(Path(args.config).read_text()), args.output, args.model)


if __name__ == '__main__':
    main()
