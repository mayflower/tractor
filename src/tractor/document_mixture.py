"""Balanced document exposures with bounded real-source replay and a separate NER pool."""
import argparse
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from .coverage_assembly import sha
from .document_campaign import quotas
from .extraction_scope import in_scope
from .jsonio import dumps, loads


def verified(root):
    root = Path(root)
    manifest = loads((root / 'manifest.json').read_text())
    if manifest.get('unique_real_train') is not None and manifest.get('local_role_audit') is not True:
        raise ValueError('Public corpus requires the current local-role audit')
    if manifest.get('unique_real_train') is not None and manifest.get('role_policy_sha256') != sha(Path(__file__).with_name('document_role_scope.py')):
        raise ValueError('Public role policy changed; requalify corpus')
    if manifest.get('training_ready') is not True:
        raise ValueError(f'Final corpus audit has not qualified {root}')
    for split in ('train', 'dev'):
        for kind in ('episodes', 'native'):
            name = f'{split}.{kind}.jsonl'
            if sha(root / name) != manifest['files'][name]:
                raise ValueError('Qualified input changed')
    return manifest


def pairs(root, split):
    with Path(root, f'{split}.episodes.jsonl').open() as ef, Path(root, f'{split}.native.jsonl').open() as nf:
        for eline, nline in zip(ef, nf, strict=True):
            e = loads(eline)
            if e['split'] != split or e['text'] != loads(nline)['input'] or not in_scope(e):
                raise ValueError('Unpaired, wrong-split or out-of-scope source')
            yield e, eline, nline


def key(row):
    if row['source'] in {'few_nerd', 'docred'}:
        return ('generic_entities', 'entity_list')
    shape = 'repeated_records' if any(t['kind'] == 'records' for t in row['tasks']) else (
        'entity_list' if row['artifact'] == 'entity_list' else 'single_object')
    return row['family'], shape


def real_representatives(rows, seed):
    real_groups = defaultdict(dict)
    representatives = {}
    # Choose one representative window per independent source group using a
    # seeded hash, never according to where a gold answer happens to occur.
    for e in rows:
        k = key(e)
        rank = hashlib.sha256(f'{seed}:{e["id"]}'.encode()).digest()
        # Choose a task-bearing record window over a header-only window from
        # the same document, then use a seeded rank within that task stratum.
        grouped = any(t['kind'] == 'records' and len(t.get('fields', {})) > 1 for t in e['tasks'])
        priority = (0 if grouped else 1 if k[1] == 'repeated_records' else 2, rank)
        previous = representatives.get(e['source_group'])
        if previous is None or priority < previous[0]:
            representatives[e['source_group']] = (priority, k, e['id'])
    for group, (priority, k, ident) in representatives.items():
        real_groups[k][group] = (priority[1], ident)
    return real_groups


def prepare(synthetic, real, output, exposures=60000, seed=20261024, minimum_real_record_sources=0, max_synthetic_repeats=3):
    if max_synthetic_repeats < 1:
        raise ValueError('Synthetic repeat cap must be positive')
    sm = verified(synthetic)
    rm = verified(real)
    if rm['unique_real_train'] < 200000:
        raise ValueError('The qualified public source pool is below 200000 independent examples')
    config = loads(Path('configs/document-build.json').read_text())
    targets = Counter()
    for (split, domain, shape, _language), n in quotas(config, exposures).items():
        if split == 'train':
            targets[domain, shape] += n
    # All accepted synthetic training scenarios appear at least once. Public NER
    # is an available pool, not a reason to expand an epoch by millions of repeats.
    originals, pools = [], defaultdict(list)
    for e, eline, nline in pairs(synthetic, 'train'):
        pools[key(e)].append(len(originals))
        originals.append((e, eline, nline))
    expected = sm.get('accepted_by_split', sm.get('counts', {})).get('train')
    if not originals or len(originals) != expected:
        raise ValueError('Qualified synthetic scenario count differs from its manifest')
    if any(len(pools[k]) > n for k, n in targets.items()):
        raise ValueError('Exposure budget cannot include all synthetic scenarios')
    real_groups = real_representatives((e for e, _, _ in pairs(real, 'train')), seed)
    chosen, real_by_key = set(), Counter()
    for k, groups in real_groups.items():
        if k == ('generic_entities', 'entity_list'):
            limit = sum(n-len(pools[dk]) for dk, n in targets.items() if dk[1] == 'entity_list')
        elif k in targets and k[1] != 'entity_list':
            limit = min(targets[k] // 2, targets[k] - len(pools[k]))
        else:
            continue
        picks = sorted(groups.values())[:limit]
        chosen.update(ident for _, ident in picks)
        real_by_key[k] += len(picks)
    selected_real = []
    for e, eline, nline in pairs(real, 'train'):
        if e['id'] in chosen:
            selected_real.append((e, eline, nline))
    record_source_count = sum(any(t['kind'] == 'records' and len(t['fields']) > 1 for t in e['tasks'])
                              for e, _, _ in selected_real)
    if record_source_count < minimum_real_record_sources:
        raise ValueError(f'Only {record_source_count} independent real record sources; require {minimum_real_record_sources}')
    rng = random.Random(seed)
    real_replay = defaultdict(list)
    for row in selected_real:
        real_replay[key(row[0])].append(row)
    replay_counts = Counter()
    shortfalls = Counter()
    schedule = list(originals) + selected_real
    ner_left = real_by_key['generic_entities', 'entity_list']
    for k, n in targets.items():
        missing = n - len(pools[k]) - real_by_key[k]
        if k[1] == 'entity_list':
            supplied = min(missing, ner_left)
            ner_left -= supplied
            missing -= supplied
        # Up to half of a supported document cell may use public examples,
        # capped at eight exposures per independent source including its first.
        candidates = real_replay[k]
        extra = min(missing, max(0, n//2-real_by_key[k]), 7*len(candidates)) if k[1] != 'entity_list' else 0
        rng.shuffle(candidates)
        schedule.extend(candidates[i % len(candidates)] for i in range(extra))
        replay_counts[k] = extra
        missing -= extra
        allowed = min(missing, (max_synthetic_repeats-1)*len(pools[k]))
        shortfalls[':'.join(k)] = missing - allowed
        missing = allowed
        indices = list(pools[k])
        rng.shuffle(indices)
        schedule.extend(originals[indices[i % len(indices)]] for i in range(missing))
    if len(schedule) + sum(shortfalls.values()) != exposures:
        raise ValueError('Exposure accounting mismatch')
    rng.shuffle(schedule)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    with (root / 'train.native.jsonl').open('w') as nf, (root / 'train.identities.jsonl').open('w') as ids:
        for e, _eline, nline in schedule:
            nf.write(nline)
            identity = {k: e[k] for k in ['id', 'source_group', 'source', 'family', 'artifact', 'language', 'text_origin', 'label_origin']}
            ids.write(dumps(identity) + '\n')
            counts[':'.join((e['text_origin'], e['source'], *key(e)))] += 1
    # Only declared document-domain Dev rows enter checkpoint selection. Generic
    # NER is not silently assigned to a business domain to manufacture coverage.
    dev_count = Counter()
    dev_groups = defaultdict(dict)
    for e, _, _ in pairs(real, 'dev'):
        if e['family'] not in config['domains'] or e['artifact'] not in config['shapes']:
            continue
        group = e['family'], e['text_origin'], e['label_origin']
        rank = hashlib.sha256(f'{seed}:{e["id"]}'.encode()).digest()
        prior = dev_groups[group].get(e['source_group'])
        if prior is None or rank < prior[0]:
            dev_groups[group][e['source_group']] = (rank, e['id'])
    dev_ids = {ident for groups in dev_groups.values() for _, ident in sorted(groups.values())[:200]}
    with (root / 'dev.episodes.jsonl').open('w') as ef, (root / 'dev.native.jsonl').open('w') as nf:
        for source in (synthetic, real):
            for e, eline, nline in pairs(source, 'dev'):
                if source == real and e['id'] not in dev_ids:
                    continue
                if e['family'] in config['domains'] and e['artifact'] in config['shapes']:
                    ef.write(eline)
                    nf.write(nline)
                    dev_count[f'{e["family"]}:{e["text_origin"]}/{e["label_origin"]}'] += 1
    selection_cells = defaultdict(list)
    for e, eline, nline in pairs(root, 'dev'):
        selection_cells[e['family'], e['artifact'], e['language'], e['text_origin'], e['label_origin']].append((e['id'], eline, nline))
    with (root / 'dev.selection.episodes.jsonl').open('w') as ef, (root / 'dev.selection.native.jsonl').open('w') as nf:
        for cell_rows in selection_cells.values():
            cell_rows.sort(key=lambda row: hashlib.sha256(f'{seed}:{row[0]}'.encode()).digest())
            limit = 6 if loads(cell_rows[0][1])['text_origin'] == 'synthetic' else 200
            for _, eline, nline in cell_rows[:limit]:
                ef.write(eline)
                nf.write(nline)
    report = {'status': 'complete', 'training_ready': True, 'seed': seed, 'exposures_per_epoch': len(schedule), 'requested_exposures': exposures,
              'max_synthetic_repeats': max_synthetic_repeats,
              'unfilled_exposure_targets': {k: v for k, v in shortfalls.items() if v},
              'synthetic_unique_train': len(originals), 'real_unique_exposed': len(selected_real),
              'available_real_source_examples': rm['unique_real_train'],
              'real_record_source_count': record_source_count, 'minimum_real_record_sources': minimum_real_record_sources,
              'real_replay_exposures': {':'.join(k): v for k, v in replay_counts.items() if v}, 'exposures': dict(counts), 'dev_counts': dict(dev_count),
              'real_dev_selection': 'At most 200 independent groups per observed domain/origin; one seeded representative window each. Full source Dev remains available separately.',
              'checkpoint_dev_selection': 'Six synthetic cases per domain/form/language cell; all prepared real Dev cells up to 200 each. Fixed before training.',
              'inputs': {'synthetic': {'path': synthetic, 'manifest_sha256': sha(Path(synthetic, 'manifest.json'))},
                         'real': {'path': real, 'manifest_sha256': sha(Path(real, 'manifest.json'))}},
              'files': {p.name: sha(p) for p in root.glob('*.jsonl')},
              'qualification': '60/30/10 are upper exposure targets, not forced proportions. Epochs shrink when diversity cannot support the synthetic replay cap; deficits remain explicit. All qualified synthetic scenarios retained. '
              'Existing generic NER fills only remaining entity-list capacity and is reported as generic_entities, not a fabricated business domain. '
              'One representative window per real source group, preferring actual records; public replay capped at eight exposures per source. Missing real domain coverage is filled by synthetic replay, not mislabeled sources. '
              'No calibration/test data copied into the training directory. This is data qualification, not model-release qualification.'}
    (root / 'manifest.json').write_text(dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', required=True)
    p.add_argument('--real', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--exposures', type=int, default=60000)
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--max-synthetic-repeats', type=int, default=3)
    p.add_argument('--minimum-real-record-sources', type=int, default=0)
    prepare(**vars(p.parse_args()))
