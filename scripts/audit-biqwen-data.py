"""Read-only text-overlap and request-schema audit of the pinned extraction data."""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from tractor.biqwen import schema_for
from tractor.generalist_data import read_episodes
from tractor.jsonio import dumps, loads
from tractor.pipeline_evaluation import typed_requests_for
from tractor.source_policy import excluded_source


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def signature(schema):
    """All field properties used in upstream text encoding, excluding opaque IDs."""
    if schema.get('choices'):
        raise ValueError('Decision schemas are outside this extraction audit')
    tables = {t['id']: t for t in schema['tables']}
    return [
        (f['description'], f.get('kind', 'span'), f.get('multiple', False),
         f.get('resolve_references', False),
         tables[f['table_id']]['description'] if f.get('table_id') else None)
        for f in schema['fields']
    ]


def json_rows(path, hashes):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            checksum.update(line)
            row = loads(line)
            if excluded_source(row) or excluded_source(row.get('metadata', {})):
                raise ValueError(f'Excluded source in {path}')
            yield row
    hashes[str(path)] = checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', default='data/biqwen-source-audit/data')
    parser.add_argument('--tractor', default='data/full-training-de-en-native')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    dev = list(read_episodes(Path(args.tractor) / 'dev.episodes.jsonl'))
    if not dev or any(r.split != 'dev' or excluded_source(r) for r in dev):
        raise ValueError('Expected permitted Dev episodes only')
    texts = defaultdict(list)
    for row in dev:
        texts[text_hash(row.text)].append(row)
    native_texts = defaultdict(list)
    summary, matches, differences = {}, [], []
    for path in sorted(Path(args.native).glob('*.jsonl')):
        counts, sources, languages, overlap, families = (Counter() for _ in range(5))
        for row in json_rows(path, hashes):
            counts['rows'] += 1
            counts['record_examples'] += bool(row['schema']['tables'])
            sources[row['metadata']['source']] += 1
            languages[row['metadata']['language']] += 1
            families[row['metadata']['source'] + ':' + row['metadata']['family']] += 1
            digest = text_hash(row['text'])
            native_texts[digest].append((path.stem, row['id']))
            for current in texts.get(digest, []):
                overlap[current.source] += 1
                matches.append({'native_split': path.stem, 'native_id': row['id'],
                                'tractor_id': current.id, 'text_sha256': digest})
                if path.stem != 'validation':
                    continue  # Only metadata, never inference or selection on heldout.
                requests = typed_requests_for(current)
                counts['matched_dev'] += 1
                if len(requests) == 1:
                    actual = schema_for(next(iter(requests.values())))
                    same = signature(actual) == signature(row['schema'])
                    counts['identical_schema' if same else 'different_schema'] += 1
                    if not same:
                        differences.append({'id': current.id, 'native_id': row['id'],
                                            'native': row['schema'], 'api': actual})
                else:
                    counts['multiple_tasks'] += 1
        summary[path.stem] = {'counts': dict(counts), 'sources': dict(sources),
                             'languages': dict(languages), 'source_families': dict(families),
                             'dev_text_overlap_by_source': dict(overlap)}
    train_sources, train_families, train_overlap = (Counter() for _ in range(3))
    matched_native = defaultdict(set)
    train_total = 0
    for row in json_rows(Path(args.tractor) / 'train.episodes.jsonl', hashes):
        train_total += 1
        train_sources[row['source']] += 1
        train_families[row['source'] + ':' + row['family']] += 1
        for split, native_id in native_texts.get(text_hash(row['text']), []):
            train_overlap[split] += 1
            matched_native[split].add(native_id)
    matched_dev = len({r['tractor_id'] for r in matches})
    report = {
        'native_dataset_revision': '53188d619b99b361c8a8c72b0d4df524d879f385',
        'native_splits': summary,
        'tractor_train': {'episodes': train_total, 'sources': dict(train_sources),
                          'source_families': dict(train_families),
                          'text_overlap_native_splits': dict(train_overlap),
                          'matched_unique_native_examples': {k: len(v) for k, v in matched_native.items()}},
        'dev_episodes': len(dev), 'dev_matching_texts': matched_dev,
        'dev_no_native_text_match': len(dev) - matched_dev,
        'scope': 'Exact UTF-8 text hashes only; no near-duplicate or combined decision-data overlap claim.',
        'input_sha256': hashes,
    }
    with (Path(args.tractor) / 'dev.episodes.jsonl').open('rb') as stream:
        hashes[str(Path(args.tractor) / 'dev.episodes.jsonl')] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (out / 'summary.json').write_text(dumps(report, indent=2) + '\n')
    for name, rows in [('dev-matches', matches), ('schema-differences', differences)]:
        (out / f'{name}.jsonl').write_text(''.join(dumps(row) + '\n' for row in rows))
    print(dumps({'train': train_total, 'dev': len(dev), 'matched_dev': matched_dev, 'output': str(out)}))


if __name__ == '__main__':
    main()
