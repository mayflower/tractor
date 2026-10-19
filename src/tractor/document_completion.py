"""Continue completed data jobs into immutable coverage and split/near-duplicate audits.

This watcher performs no release promotion. It leaves concrete audit findings,
including source-count shortfalls, instead of marking unfinished data ready.
"""
import argparse
import fcntl
import time
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .document_campaign import atomic_json
from .document_duplicates import audit
from .extraction_scope import in_scope
from .jsonio import dumps, loads


def coverage(source, output):
    root = Path(source)
    manifest = loads((root / 'manifest.json').read_text())
    groups, texts, ids, errors = {}, {}, set(), Counter()
    counts, domains, shapes, languages, origins, fields, record_counts, lengths = (Counter() for _ in range(8))
    samples = {}
    for split in ('train', 'dev', 'calibration', 'test'):
        path = root / f'{split}.episodes.jsonl'
        if sha(path) != manifest['files'][path.name]:
            raise ValueError('Completed corpus changed')
        for line in path.open():
            row = loads(line)
            if row['split'] != split or not in_scope(row):
                errors['wrong_split_or_scope'] += 1
            if row['id'] in ids:
                errors['duplicate_id'] += 1
            ids.add(row['id'])
            for group in (row['source_group'], row['derivation_group']):
                if group in groups and groups[group] != split:
                    errors['cross_split_group'] += 1
                groups[group] = split
            text = ' '.join(row['text'].casefold().split())
            if text in texts and texts[text] != split:
                errors['cross_split_text'] += 1
            texts[text] = split
            counts[split] += 1
            domains[f'{split}:{row["family"]}'] += 1
            shapes[f'{split}:{row["artifact"]}'] += 1
            languages[f'{split}:{row["language"]}'] += 1
            origins[f'{split}:{row["text_origin"]}/{row["label_origin"]}'] += 1
            field_count = sum(len(t['fields']) for t in row['tasks'])
            fields['3-8' if 3 <= field_count <= 8 else '9-20' if 9 <= field_count <= 20 else '21-40' if 21 <= field_count <= 40 else 'other'] += 1
            for task in row['tasks']:
                if task['kind'] == 'records':
                    record_counts[str(len(row['gold'][task['id']]['records']))] += 1
            lengths[str(len(row['text']) // 500 * 500)] += 1
            bucket = (row['family'], row['artifact'], row['language'], row['text_origin'], row['label_origin'])
            # Earliest lexical identifier is deterministic and independent of scores.
            if bucket not in samples or row['id'] < samples[bucket]['id']:
                samples[bucket] = row
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'direct-inspection-sample.jsonl').open('x') as stream:
        for row in samples.values():
            stream.write(dumps(row) + '\n')
    report = {'source_manifest_sha256': sha(root / 'manifest.json'), 'counts': dict(counts), 'domains': dict(domains),
              'shapes': dict(shapes), 'languages': dict(languages), 'origins': dict(origins), 'schema_field_bands': dict(fields),
              'record_cardinalities': dict(record_counts), 'text_length_bands': dict(lengths), 'errors': dict(errors),
              'training_ready': False, 'qualification': 'Measured coverage, not an assertion of full reqs.md coverage. '
              'Missing wide-schema/long-table coverage stays visible; direct final inspection and duplicate findings must be resolved.'}
    atomic_json(dest / 'coverage.json', report)
    return report


def watch(synthetic, real, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = {'synthetic': Path(synthetic), 'real': Path(real)}
        while pending:
            for name, source in list(pending.items()):
                if not (source / 'manifest.json').exists():
                    continue
                dest = root / name
                if not (dest / 'coverage.json').exists():
                    coverage(source, dest)
                atomic_json(root / 'status.json', {'status': 'auditing', 'active': name, 'pending': list(pending)})
                if not (dest / 'near-duplicates.json').exists():
                    audit(source, dest / 'near-duplicates.json', cross_only=name == 'real')
                pending.pop(name)
                atomic_json(root / 'status.json', {'status': 'waiting_for_inputs' if pending else 'audits_complete',
                                                  'pending': list(pending), 'training_ready': False,
                                                  'qualification': 'No automatic waiver of final sample, source-count or duplicate findings.'})
            if pending:
                time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', default='data/documents-final-campaign/dataset')
    p.add_argument('--real', default='data/real-extraction-reviewed')
    p.add_argument('--output', default='reports/extraction-build/completion')
    watch(**vars(p.parse_args()))
