"""Assemble only real extraction sources; quarantine split overlap and report original groups."""

import argparse
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .episodes import Episode, native_episode, text_key
from .extraction_scope import in_scope
from .jsonio import dumps, loads

SPLITS = ('test', 'calibration', 'dev', 'train')


def projected(row, historical=False):
    if row['text_origin'] != 'real' or not in_scope(row):
        return None
    if historical and row['source'] not in {'few_nerd', 'docred'}:
        return None
    # Prioritize explicit descriptive fields; these weak short category fields are
    # not needed to establish web attribute coverage and are easily ambiguous.
    if row['source'] == 'swde':
        for task in row['tasks']:
            remove = {'mpaa_rating', 'genre', 'type'} & task['fields'].keys()
            for key in remove:
                task['fields'].pop(key)
                for values in row['gold'][task['id']]['records']:
                    values.pop(key)
    positive = any(answer['items'] for gold in row['gold'].values() for values in gold['records'] for answer in values.values())
    # Preserve genuine, fully annotated source-level NER negatives. Do not count
    # arbitrary empty windows of long documents as additional source examples.
    source_negative = row['source'] in {'few_nerd', 'docred'} and all(
        gold['complete'] and all(answer['status'] == 'absent' for values in gold['records'] for answer in values.values())
        for gold in row['gold'].values())
    if not positive and not source_negative:
        return None
    return row


def rows(sources, split):
    for source in sources:
        path = Path(source) / f'{split}.episodes.jsonl'
        historical = Path(source).name == 'extraction-strict-base'
        for line in path.open():
            row = projected(loads(line), historical)
            if row is not None:
                yield row


def build(output, sources):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for source in sources:
        manifest = loads(Path(source, 'manifest.json').read_text())
        for split in SPLITS:
            name = f'{split}.episodes.jsonl'
            if sha(Path(source, name)) != manifest['files'][name]:
                raise ValueError('Source hash mismatch')
        inputs[source] = sha(Path(source, 'manifest.json'))
    # Split priority is heldout-first. Any overlapping positive train fragment
    # quarantines the entire original training page, not just the matching window.
    seen, blocked = {}, set()
    for split in SPLITS:
        for row in rows(sources, split):
            key = text_key(row['text'])
            if key in seen and seen[key][0] != split:
                blocked.add(row['source_group'])
            else:
                seen.setdefault(key, (split, row['source_group']))
    counts, origins, domains, groups, negatives = Counter(), Counter(), Counter(), {}, Counter()
    seen = set()
    dropped = Counter()
    sample = {}
    with (root / 'excluded-groups.json').open('w') as f:
        f.write(dumps(sorted(blocked), indent=2))
    for split in SPLITS:
        with (root / f'{split}.episodes.jsonl').open('w') as ef, (root / f'{split}.native.jsonl').open('w') as nf:
            for row in rows(sources, split):
                if row['source_group'] in blocked:
                    dropped['cross_split_group'] += 1
                    continue
                key = text_key(row['text'])
                if key in seen:
                    dropped['duplicate_positive_window'] += 1
                    continue
                seen.add(key)
                episode = Episode.model_validate(row)
                native, skipped = native_episode(episode)
                if skipped or native is None:
                    raise ValueError('Incomplete projected supervision')
                ef.write(dumps(row)+'\n')
                nf.write(dumps(native)+'\n')
                counts[split] += 1
                origins[f'{split}:{row["source"]}:{row["label_origin"]}'] += 1
                domains[f'{split}:{row["family"]}'] += 1
                groups[row['source_group']] = (split, row['source'], row['label_origin'])
                if not any(a['items'] for g in row['gold'].values() for r in g['records'] for a in r.values()):
                    negatives[f'{split}:{row["source"]}'] += 1
                if row['source'] not in {'few_nerd', 'docred'}:
                    category = row['source'] if row['source'] != 'swde' else ':'.join(row['source_group'].split(':')[:2])
                    examples = sample.setdefault(category, {})
                    if len(examples) < 50:
                        examples.setdefault(row['source_group'], row)
    with (root/'adapter-audit-sample.jsonl').open('w') as stream:
        for examples in sample.values():
            for row in examples.values():
                stream.write(dumps(row)+'\n')
    group_counts = Counter(f'{split}:{source}:{origin}' for split, source, origin in groups.values())
    real_train = sum(n for key, n in group_counts.items() if key.startswith('train:'))
    report = {'status': 'real_pool_assembled_pending_semantic_sample_audit', 'training_ready': False,
              'counts': dict(counts), 'unique_source_examples': dict(group_counts), 'unique_real_train': real_train,
              'minimum_real_train': 200000, 'minimum_real_train_met': real_train >= 200000,
              'source_level_negatives': dict(negatives),
              'origins_by_window': dict(origins), 'domains_by_window': dict(domains), 'dropped': dict(dropped),
              'inputs': inputs, 'files': {p.name: sha(p) for p in root.glob('*.jsonl')},
              'qualification': 'Real human text with human or rule-derived labels, reported separately. '
              'Document adapters contribute positive windows; fully annotated original NER negatives are retained and counted separately. '
              'Full-page raw source validations retain document negative windows separately. '
              'Native source validation preceded field projection. No schematic synthesis, QA, classes or relations. '
              'Counts distinguish independent source examples from windows. Few-NERD source units are annotated sentences, not recoverable original documents. '
              'Effective training mixture still needs balancing; no claim that all domains have independent human evaluation.'}
    (root/'manifest.json').write_text(dumps(report, indent=2))
    print(dumps({k: report[k] for k in ['status','counts','unique_real_train','minimum_real_train_met','dropped']}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--sources', nargs='+', required=True)
    build(**vars(p.parse_args()))
