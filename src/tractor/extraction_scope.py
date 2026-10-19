"""Remove out-of-scope tasks from active extraction datasets, preserving paired rows."""

import argparse
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .jsonio import dumps, loads

EXCLUDED_FAMILIES = frozenset({'question_answering', 'multiple_answers', 'classification_intent',
                               'classification_routing', 'classification_urgency', 'classification_sentiment', 'relations'})
EXCLUDED_SOURCES = frozenset({'squad_v2', 'germanquad', 'multispan', 'adversarialqa', 'adversarial_qa', 'massive'})


def in_scope(row):
    from .document_text_scope import table_record_reason

    return (row['family'] not in EXCLUDED_FAMILIES and not row['family'].startswith('classification_')
            and row['source'] not in EXCLUDED_SOURCES
            and table_record_reason(row) is None
            and all(t['kind'] in {'entities', 'records'} for t in row.get('tasks', [])))


def prepare(source, output):
    source, output = Path(source), Path(output)
    manifest = loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete source corpus')
    for name, digest in manifest['files'].items():
        if sha(source / name) != digest:
            raise ValueError(f'Source hash mismatch: {name}')
    output.mkdir(parents=True, exist_ok=False)
    counts, removed, families, origins = Counter(), Counter(), Counter(), Counter()
    for split in ('train', 'dev', 'calibration', 'test'):
        with (source / f'{split}.episodes.jsonl').open() as ef, (source / f'{split}.native.jsonl').open() as nf, \
             (output / f'{split}.episodes.jsonl').open('w') as eo, (output / f'{split}.native.jsonl').open('w') as no:
            for eline, nline in zip(ef, nf, strict=True):
                e, n = loads(eline), loads(nline)
                if e['split'] != split or e['text'] != n['input']:
                    raise ValueError('Unpaired or wrong-split source')
                if not in_scope(e):
                    removed[f'{split}:{e["source"]}'] += 1
                    continue
                eo.write(eline)
                no.write(nline)
                counts[split] += 1
                families[f'{split}:{e["family"]}'] += 1
                origins[f'{split}:{e["text_origin"]}/{e["label_origin"]}'] += 1
    report = {'status': 'complete', 'source': str(source), 'source_manifest_sha256': sha(source / 'manifest.json'),
              'counts': dict(counts), 'removed': dict(removed), 'families': dict(families), 'origins': dict(origins),
              'excluded_families': sorted(EXCLUDED_FAMILIES), 'excluded_sources': sorted(EXCLUDED_SOURCES),
              'qualification': 'QA, relations and all classification are outside the product objective. Excluded by source/task semantics, '
              'not prediction quality. All splits filtered identically; retained rows unchanged. '
              'Existing model weights still reflect prior out-of-scope training. No inference or release qualification.',
              'files': {p.name: sha(p) for p in output.glob('*.jsonl')}}
    (output / 'manifest.json').write_text(dumps(report, indent=2))
    print(dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
