"""Rebuild public receipt/form adapters with frozen split membership and explicit contracts."""
import argparse
import gzip
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .coverage_assembly import sha
from .document_campaign import atomic_json
from .document_duplicates import Index
from .document_sources import cord, vrdu
from .episodes import Episode, text_key
from .generalist_data import validate_native
from .jsonio import dumps, loads
from .public_source_review import review_batch
from .source_contracts import split_lookup


def safe_windows(episode, width=1000, overlap=250):
    """Text-only overlapping boundaries; drop ambiguous cut windows, not whole documents."""
    text = episode.text
    start = 0
    while start < len(text):
        end = min(start+width, len(text))
        if end < len(text):
            cut = text.rfind('\n', start+width//2, end)
            if cut < 0:
                cut = text.rfind(' ', start+width//2, end)
            if cut >= 0:
                end = cut+1
        tasks, gold, invalid = [], {}, False
        for task in episode.tasks:
            records = []
            for record in episode.gold[task.id].records:
                spans = [s for answer in record.values() for s in answer.items]
                touched = [s for s in spans if s.start < end and s.end > start]
                inside = [s for s in spans if start <= s.start and s.end <= end]
                if len(touched) != len(inside) or (task.kind == 'records' and touched and len(inside) != len(spans)):
                    invalid = True
                    break
                if task.kind == 'records' and not inside:
                    continue
                row = {}
                for field, answer in record.items():
                    values = [dict(start=s.start-start, end=s.end-start, text=s.text)
                              for s in answer.items if start <= s.start and s.end <= end]
                    row[field] = {'status': 'present' if values else 'absent', 'items': values}
                records.append(row)
            if invalid:
                break
            if records:
                tasks.append(task.model_dump())
                gold[task.id] = {'complete': True, 'records': records}
        if not invalid and tasks and any(a['items'] for g in gold.values() for r in g['records'] for a in r.values()):
            data = episode.model_dump()
            data.update(id=f'{episode.id}:window:{start}:{end}', text=text[start:end], tasks=tasks, gold=gold,
                        artifact='repeated_records' if any(t['kind'] == 'records' for t in tasks) else 'single_object')
            yield Episode.model_validate(data)
        if end == len(text):
            break
        start = max(start+1, end-overlap)
        # Do not start in the middle of a word.
        while start < end and not text[start-1].isspace():
            start += 1


def candidates(root):
    import pyarrow.parquet as pq
    specs = loads(Path('configs/document-sources.json').read_text())
    rejected = []
    for spec in specs:
        if spec['source'] not in {'cord', 'vrdu'}:
            continue
        for name, digest in spec['files'].items():
            if sha(Path(spec['path'], name)) != digest:
                raise ValueError('Pinned source changed')
        if spec['source'] == 'cord':
            for original, split in [('train', 'train'), ('validation', 'dev'), ('test', 'test')]:
                for path in sorted(Path(spec['path'], 'data').glob(original+'*.parquet')):
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['ground_truth']):
                        for row in batch.to_pylist():
                            data = loads(row['ground_truth'])
                            ident = f'{original}:{data["meta"]["image_id"]}'
                            try:
                                yield from safe_windows(cord(data, split, ident))
                            except ValueError as exc:
                                rejected.append({'source': 'cord', 'id': ident, 'reason': str(exc)})
        else:
            for group, domain in [('registration-form', 'forms'), ('ad-buy-form', 'invoices')]:
                lookup = split_lookup(spec, group)
                with gzip.open(Path(spec['path'], group, 'main/dataset.jsonl.gz'), 'rt') as stream:
                    for line in stream:
                        row = loads(line)
                        if row['filename'] not in lookup:
                            continue
                        try:
                            yield from safe_windows(vrdu(row, lookup[row['filename']], domain))
                        except ValueError as exc:
                            rejected.append({'source': 'vrdu', 'id': row['filename'], 'reason': str(exc)})
    atomic_json(root/'adapter-rejections.json', rejected)


def build(output, base='data/real-extraction-dedup', workers=16, env='~/.env'):
    import torch

    from .model import Extractor
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'reviews').mkdir()
    torch.set_num_threads(2)
    extractor = Extractor('models/base', 'cpu')
    rows, rejects = [], []
    for e in candidates(root):
        try:
            n, _ = validate_native(e, extractor)
            rows.append((e.model_dump(), n))
        except Exception as exc:
            rejects.append({'id': e.id, 'reason': type(exc).__name__+':'+str(exc)[:180]})
    del extractor
    atomic_json(root/'native-rejections.json', rejects)
    print(dumps({'stage': 'native_validated', 'candidates': len(rows), 'rejected': len(rejects)}), flush=True)
    accepted = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = [rows[i:i+8] for i in range(0, len(rows), 8)]
        for batch, result in zip(batches, pool.map(lambda b: review_batch([r[0] for r in b], root, env), batches), strict=True):
            verdicts = {r['id']: r for r in result['results']}
            accepted.extend((e, n) for e, n in batch if verdicts[e['id']]['passed'])
            atomic_json(root/'status.json', {'stage': 'reviewing', 'accepted': len(accepted), 'candidates': len(rows)})
    index, blocked, findings = Index(), set(), []
    for split in ('test', 'calibration', 'dev', 'train'):
        for e, _ in accepted:
            if e['split'] == split:
                matches = index.check(e, cross_only=True)
                findings.extend(matches)
                if matches:
                    blocked.add(e['source_group'])
    atomic_json(root/'duplicate-audit.json', {'findings': findings, 'excluded_groups': sorted(blocked),
                                           'policy': 'Public literal similarity only; masked boilerplate cannot quarantine real documents.'})
    grouped = defaultdict(list)
    for e, n in accepted:
        if e['source_group'] not in blocked:
            grouped[e['split']].append((e, n))
    base = Path(base)
    manifest = loads((base/'manifest.json').read_text())
    counts, groups, seen = Counter(), {}, {}
    for split in ('test', 'calibration', 'dev', 'train'):
        for kind in ('episodes', 'native'):
            name = f'{split}.{kind}.jsonl'
            if sha(base/name) != manifest['files'][name]:
                raise ValueError('Base changed')
        with (root/f'{split}.episodes.jsonl').open('w') as ef, (root/f'{split}.native.jsonl').open('w') as nf:
            def write(e, n, split=split, ef=ef, nf=nf):
                key = text_key(e['text'])
                if key in seen and seen[key] != split:
                    raise ValueError('Cross-source split overlap; quarantine whole source group before training')
                seen[key] = split
                prior = groups.get(e['source_group'])
                if prior is not None and prior[0] != split:
                    raise ValueError('Source group moved between splits')
                groups[e['source_group']] = (split, e['source'], e['family'])
                ef.write(dumps(e)+'\n')
                nf.write(dumps(n)+'\n')
                counts[split] += 1
            for el, nl in zip((base/f'{split}.episodes.jsonl').open(), (base/f'{split}.native.jsonl').open(), strict=True):
                e, n = loads(el), loads(nl)
                if e['source'] not in {'cord', 'vrdu'}:
                    write(e, n)
            for e, n in grouped[split]:
                write(e, n)
    report = {'training_ready': False, 'status': 'reviewed_native_validated_pending_direct_sample_and_mixture',
              'counts': dict(counts), 'unique_real_train': sum(v[0] == 'train' for v in groups.values()),
              'source_groups': dict(Counter(':'.join(v) for v in groups.values())),
              'base_manifest_sha256': sha(base/'manifest.json'), 'registry_sha256': sha(Path('configs/document-sources.json')),
              'files': {p.name: sha(p) for p in root.glob('*.jsonl')},
              'qualification': 'Rebuilt public CORD/VRDU literal task contracts. Every new window independently reviewed and native validated. Frozen old source split memberships preserved. No claim of full domain or heldout evaluation coverage.'}
    atomic_json(root/'manifest.json', report)
    atomic_json(root/'status.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base', default='data/real-extraction-dedup')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--env', default='~/.env')
    build(**vars(parser.parse_args()))
