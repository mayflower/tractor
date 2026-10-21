"""Pinned public document adapters; preserve text evidence, groups and source splits."""

import argparse
import gzip
import lzma
import re
from collections import Counter, defaultdict
from pathlib import Path

from .coverage_assembly import sha
from .episodes import Episode, audit_episodes, text_key
from .jsonio import dumps, loads
from .source_contracts import complete_token_boundary, description, split_lookup


def answer(text, start, end):
    if not 0 <= start < end <= len(text):
        raise ValueError('Invalid evidence offset')
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end-1].isspace():
        end -= 1
    if start == end:
        raise ValueError('Whitespace-only evidence')
    return {'status': 'present', 'items': [{'start': start, 'end': end, 'text': text[start:end]}]}


def envelope(source, ident, split, domain, language, text, tasks, gold):
    group = f'{source}:{ident}'
    return Episode.model_validate({'id': group, 'source_group': group, 'derivation_group': group,
        'source': source, 'original_split': split, 'split': split, 'family': domain,
        'language': language, 'artifact': 'repeated_records' if any(t['kind'] == 'records' for t in tasks) else 'single_object',
        'annotation': 'human', 'text_origin': 'real', 'label_origin': 'human',
        'text': text, 'tasks': tasks, 'gold': gold})


def vrdu(row, split, domain):
    text = row['ocr']['text']
    fields, groups = {}, defaultdict(list)

    def value(cell):
        literal, _box, offsets = cell
        if len(offsets) != 1:
            raise ValueError('Discontinuous VRDU value requires explicit projection')
        start, end = offsets[0]
        if text[start:end] != literal:
            raise ValueError('VRDU annotation does not match source text')
        if not complete_token_boundary(text, start, end):
            raise ValueError('VRDU annotation cuts through a printed token')
        return answer(text, start, end)
    for key, values in row['annotations']:
        if isinstance(key, list):
            for cells in values:
                groups[tuple(key)].append({k: value(c) for k, c in zip(key, cells, strict=True)})
        else:
            spans = [s for v in values for s in value(v)['items']]
            fields[key] = {'status': 'present', 'items': spans} if spans else {'status': 'absent'}
    tasks, gold = [], {}
    if fields:
        tasks.append({'id': 'header', 'kind': 'entities', 'description': 'Explicit document header fields',
                      'fields': {k: description('vrdu', k) for k in fields}})
        gold['header'] = {'complete': True, 'records': [fields]}
    # One table task, not competing tasks distinguished only by annotation
    # completeness. Query the common annotated columns for every source row;
    # never turn missing source annotations into fabricated negative targets.
    records = [record for values in groups.values() for record in values]
    if records:
        keys = sorted(set.intersection(*(set(record) for record in records)))
        if not keys:
            raise ValueError('VRDU rows have no common fully annotated fields')
        anchor = 'program_desc' if 'program_desc' in keys else keys[0]
        tasks.append({'id': 'items', 'kind': 'records', 'description': 'All annotated advertising line items',
                      'fields': {k: description('vrdu', k) for k in keys}, 'anchor': anchor})
        gold['items'] = {'complete': True, 'records': [{k: record[k] for k in keys} for record in records]}
    return envelope('vrdu', row['filename'], split, domain, 'en', text, tasks, gold)


def kleister(input_line, expected_line, source, split):
    cols = input_line.rstrip('\n').split('\t')
    text = re.sub(r'\\([nft\\])', lambda m: {'n': '\n', 'f': '\f', 't': '\t', '\\': '\\'}[m[1]], cols[-1])
    targets = defaultdict(list)
    for pair in expected_line.split():
        k, v = pair.split('=', 1)
        targets[k].append(v.replace('_', ' '))
    fields, gold, excluded = {}, {}, {}
    for key in cols[1].split():
        values = targets.get(key, [])
        if not values:
            excluded[key] = 'source_absence_not_verified_locally'
            continue
        spans = []
        for value in values:
            # Whitespace/case tolerant alignment returns ORIGINAL text, never normalized labels.
            pattern = r'(?<!\w)' + r'\s+'.join(re.escape(x) for x in value.split()) + r'(?!\w)'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if not matches:
                excluded[key] = 'normalized_or_unmatched_gold'
                break
            candidates = [{'start': m.start(), 'end': m.end(), 'text': m[0]} for m in matches]
            if source == 'kleister-nda' and key == 'jurisdiction':
                from .document_role_scope import explicit_local_role
                candidates = [s for s in candidates if explicit_local_role(text, key, s, source)]
                if not candidates:
                    excluded[key] = 'unsupported_local_field_role'
                    break
            spans.extend(candidates)
        if key in excluded:
            continue
        fields[key] = ('Jurisdiction explicitly named for governing law or courts'
                       if source == 'kleister-nda' and key == 'jurisdiction' else key.replace('_', ' '))
        unique = {(s['start'], s['end']): s for s in spans}
        gold[key] = {'status': 'present', 'items': list(unique.values())} if unique else {'status': 'absent'}
    if not fields or not any(v['status'] == 'present' for v in gold.values()):
        raise ValueError('No raw evidence remains after normalized-value exclusion')
    e = envelope(source, cols[0], split, 'contracts' if source == 'kleister-nda' else 'financial_reports',
                 'en', text, [{'id': 'document', 'kind': 'entities', 'description': 'Explicit source values only', 'fields': fields}],
                 {'document': {'complete': True, 'records': [gold]}})
    return e, excluded


def windows(episode, max_chars=700):
    """Gold-independent contiguous windows; reject entire document if a record/span crosses a boundary."""
    text = episode.text
    bounds, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind('\n', start + max_chars // 2, end)
            if cut < 0:
                cut = text.rfind(' ', start + max_chars // 2, end)
            if cut >= 0:
                end = cut + 1
        bounds.append((start, end))
        start = end
    def owner(span):
        return next((i for i, (a, b) in enumerate(bounds) if a <= span.start and span.end <= b), None)
    for task in episode.tasks:
        for row in episode.gold[task.id].records:
            owners = {owner(s) for a in row.values() for s in a.items}
            if None in owners or (task.kind == 'records' and len(owners) > 1):
                raise ValueError('Document cannot be windowed without crossing annotated span or record')
    result = []
    for i, (a, b) in enumerate(bounds):
        tasks, gold = [], {}
        for task in episode.tasks:
            rows = []
            for row in episode.gold[task.id].records:
                projected = {}
                for key, value in row.items():
                    items = [dict(start=s.start-a, end=s.end-a, text=s.text) for s in value.items if owner(s) == i]
                    projected[key] = {'status': 'present', 'items': items} if items else {'status': 'absent'}
                if task.kind == 'entities' or any(v['status'] == 'present' for v in projected.values()):
                    rows.append(projected)
            if rows:
                tasks.append(task.model_dump())
                gold[task.id] = {'complete': True, 'records': rows}
        if not tasks:
            continue
        data = episode.model_dump()
        data.update(id=f'{episode.id}:window:{a}:{b}', text=text[a:b], tasks=tasks, gold=gold)
        result.append(Episode.model_validate(data))
    return result


def build(output, source_root='data/document-sources', limit=200):
    """Create audited candidates; native validation/release eligibility remain separate gates."""
    root, sources = Path(output), Path(source_root)
    root.mkdir(parents=True, exist_ok=False)
    rejected, excluded, episodes, inputs = [], [], [], {}
    def record(source, ident, fn):
        try:
            result = fn()
            if isinstance(result, tuple):
                e, projection = result
                if projection:
                    excluded.append({'id': e.id, 'fields': projection})
            else:
                e = result
            episodes.append(e)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            rejected.append({'source': source, 'id': ident, 'reason': str(exc)[:200]})
    for source in ('kleister-nda', 'kleister-charity'):
        for original, split in [('train', 'train'), ('dev-0', 'dev')]:
            ip, gp = sources/source/original/'in.tsv.xz', sources/source/original/'expected.tsv'
            inputs[str(ip)], inputs[str(gp)] = sha(ip), sha(gp)
            with lzma.open(ip, 'rt') as fi, gp.open() as fg:
                for i, (il, gl) in enumerate(zip(fi, fg, strict=True)):
                    if i >= limit:
                        break
                    record(source, str(i), lambda il=il, gl=gl, source=source, split=split: kleister(il, gl, source, split))
    for group, domain in [('registration-form', 'forms'), ('ad-buy-form', 'invoices')]:
        registry = next(s for s in loads(Path('configs/document-sources.json').read_text()) if s['source'] == 'vrdu')
        split_file = sources / 'vrdu' / registry['active_splits'][group]
        if sha(split_file) != registry['files'][registry['active_splits'][group]]:
            raise ValueError('Pinned VRDU split changed')
        lookup = split_lookup(dict(registry, path=str(sources/'vrdu')), group)
        for name in registry['files']:
            if name.startswith(group+'/few_shot-splits/'):
                inputs[str(sources/'vrdu'/name)] = sha(sources/'vrdu'/name)
        path = sources/'vrdu'/group/'main/dataset.jsonl.gz'
        inputs[str(path)], inputs[str(split_file)] = sha(path), sha(split_file)
        counts = Counter()
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = loads(line)
                split = lookup.get(row['filename'])
                if split is None or counts[split] >= limit:
                    continue
                counts[split] += 1
                record('vrdu', row['filename'], lambda row=row, split=split, domain=domain: vrdu(row, split, domain))
    # Never split the same normalized document across sources/splits.
    by_text = defaultdict(set)
    for e in episodes:
        by_text[text_key(e.text)].add(e.split)
    conflicts = {key for key, splits in by_text.items() if len(splits) > 1}
    conflicted = [e for e in episodes if text_key(e.text) in conflicts]
    rejected.extend({'source': e.source, 'id': e.id, 'reason': 'cross_split_duplicate_text_all_copies_quarantined'} for e in conflicted)
    episodes = [e for e in episodes if text_key(e.text) not in conflicts]
    audit = audit_episodes(episodes)
    with (root/'documents.episodes.jsonl').open('w') as stream:
        for e in episodes:
            stream.write(dumps(e.model_dump())+'\n')
    windowed, window_rejections = [], []
    for e in episodes:
        try:
            windowed.extend(windows(e))
        except ValueError as exc:
            window_rejections.append({'id': e.id, 'reason': str(exc)})
    with (root/'windows.episodes.jsonl').open('w') as stream:
        for e in windowed:
            stream.write(dumps(e.model_dump())+'\n')
    for name, value in [('adapter-rejections', rejected), ('field-projections', excluded), ('window-rejections', window_rejections)]:
        (root/(name+'.json')).write_text(dumps(value, indent=2))
    manifest = {'status': 'adapter_audit_not_training_ready', 'audit': audit,
                'documents': len(episodes), 'windows': len(windowed),
                'counts': dict(Counter(f'{e.source}:{e.split}' for e in episodes)),
                'inputs': inputs, 'adapter_rejected': len(rejected), 'projected_documents': len(excluded),
                'window_rejected_documents': len(window_rejections),
                'files': {p.name: sha(p) for p in root.glob('*.jsonl')},
                'qualification': 'Source subsets, not complete import. Native validation and source terms review required. '
                'Windows do not count as independent real documents. No model inference.'}
    (root/'manifest.json').write_text(dumps(manifest, indent=2))
    return manifest


def validate(output, model='models/tractor'):
    import torch

    from .generalist_data import read_episodes, validate_native
    from .model import Extractor

    torch.set_num_threads(2)
    root = Path(output)
    manifest = loads((root / 'manifest.json').read_text())
    if sha(root / 'windows.episodes.jsonl') != manifest['files']['windows.episodes.jsonl']:
        raise ValueError('Source windows changed')
    extractor = Extractor(model, 'cpu')
    groups = defaultdict(list)
    for e in read_episodes(root / 'windows.episodes.jsonl'):
        groups[e.source_group].append(e)
    accepted, rejected = [], []
    for group, episodes in groups.items():
        rows = []
        try:
            for e in episodes:
                native, tokens = validate_native(e, extractor)
                rows.append((e, native, tokens))
        except Exception as exc:
            rejected.append({'group': group, 'reason': type(exc).__name__+':'+str(exc)[:160]})
            continue
        accepted.extend(rows)
    dest = root / 'validated'
    dest.mkdir(exist_ok=False)
    for split in ('train', 'dev', 'calibration', 'test'):
        with (dest / f'{split}.episodes.jsonl').open('w') as ef, (dest / f'{split}.native.jsonl').open('w') as nf:
            for e, n, _ in accepted:
                if e.split == split:
                    ef.write(dumps(e.model_dump())+'\n')
                    nf.write(dumps(n)+'\n')
    (dest / 'rejections.json').write_text(dumps(rejected, indent=2))
    result = {'status': 'native_validated_source_terms_and_human_audit_pending', 'training_ready': False,
              'unique_documents': len({e.source_group for e, _, _ in accepted}),
              'counts': dict(Counter(e.split for e, _, _ in accepted)),
              'source_documents': dict(Counter(k.split(':')[0] for k in {e.source_group for e, _, _ in accepted})),
              'rejected_documents': len(rejected), 'source_manifest_sha256': sha(root / 'manifest.json'),
              'files': {p.name: sha(p) for p in dest.glob('*.jsonl')},
              'qualification': 'Native targets checked on CPU; no inference. Partial document failures reject all its windows. '
              'Counts are windows; only unique_documents count toward public source quantity. '
              'No release eligibility asserted for sources with unresolved data terms.'}
    (dest / 'manifest.json').write_text(dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-root', default='data/document-sources')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--model', default='models/tractor')
    args = parser.parse_args()
    result = validate(args.output, args.model) if args.validate else build(args.output, args.source_root, args.limit)
    print(dumps(result, indent=2))
