"""Build a reproducible general-task mixture without a product-specific training quota."""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from .coverage_assembly import sha
from .extraction_scope import in_scope
from .jsonio import dumps, loads


def bucket(row):
    family = row['family']
    if family.startswith('classification_'):
        return 'classification'
    return {'named_entities': 'entities', 'document_entities': 'entities',
            'question_answering': 'questions', 'multiple_answers': 'questions',
            'structured_records': 'records'}.get(family, family)


def balanced(rows, count, rng):
    """Round robin independently shuffled source/family/language strata, without replacement."""
    strata = defaultdict(list)
    for row in rows:
        e = row[0]
        strata[(e['source'], e['family'], e['language'])].append(row)
    for values in strata.values():
        rng.shuffle(values)
    keys = sorted(strata)
    rng.shuffle(keys)
    result = []
    while len(result) < count:
        progressed = False
        for key in keys:
            if strata[key] and len(result) < count:
                result.append(strata[key].pop())
                progressed = True
        if not progressed:
            raise ValueError('Insufficient unique examples for mixture quota')
    return result


def prepare(source, config_path, output):
    source, root = Path(source), Path(output)
    config = loads(Path(config_path).read_text())
    if sum(config['weights'].values()) != 100 or config['train_examples'] % 100:
        raise ValueError('Exact mixture quotas require weights summing to 100 and whole hundreds')
    manifest = loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Source corpus is incomplete')
    pools = {}
    for split in ('train', 'dev'):
        pools[split] = defaultdict(list)
        for kind in ('episodes', 'native'):
            name = f'{split}.{kind}.jsonl'
            if sha(source / name) != manifest['files'][name]:
                raise ValueError('Source hash mismatch')
        with (source / f'{split}.episodes.jsonl').open() as ef, (source / f'{split}.native.jsonl').open() as nf:
            for eline, nline in zip(ef, nf, strict=True):
                row = loads(eline)
                if row['split'] != split or row['text'] != loads(nline)['input']:
                    raise ValueError('Unpaired or wrong-split source row')
                if not in_scope(row):
                    raise ValueError('Out-of-scope task in mixture source; run extraction_scope first')
                key = bucket(row)
                if key not in config['weights']:
                    raise ValueError(f'Unmapped family: {row["family"]}')
                pools[split][key].append((row, eline, nline))
    root.mkdir(parents=True, exist_ok=False)
    counts, origins, families = Counter(), Counter(), Counter()
    for split in ('train', 'dev'):
        selected = []
        rng = random.Random(f'{config["seed"]}:{split}')
        for key, weight in config['weights'].items():
            count = config['train_examples'] * weight // 100 if split == 'train' else config['dev_per_bucket']
            candidates = pools[split][key]
            rows = balanced(candidates, count, rng)
            selected.extend(rows)
            counts[f'{split}:{key}'] += len(rows)
        rng.shuffle(selected)
        with (root / f'{split}.episodes.jsonl').open('w') as ef, (root / f'{split}.native.jsonl').open('w') as nf:
            for row, eline, nline in selected:
                ef.write(eline)
                nf.write(nline)
                origins[f'{split}:{row["text_origin"]}/{row["label_origin"]}'] += 1
                families[f'{split}:{row["family"]}'] += 1
    report = {'status': 'complete', 'config': config, 'counts': dict(counts), 'origins': dict(origins),
              'families': dict(families), 'source': str(source), 'source_manifest_sha256': sha(source / 'manifest.json'),
              'generator_sha256': sha(__file__), 'files': {p.name: sha(p) for p in root.glob('*.jsonl')}}
    (root / 'manifest.json').write_text(dumps(report, indent=2))
    print(dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', default='configs/generalist-mixture.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.source, args.config, args.output)
