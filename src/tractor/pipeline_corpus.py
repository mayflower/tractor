"""Build a bounded role/record repair mix with explicit final-value annotations."""
import argparse
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from .coverage_assembly import sha
from .document_mixture import pairs, verified
from .document_text_scope import restrict
from .episodes import Episode, audit_episodes, native_episode, text_key
from .generalist_data import read_episodes, validate_native
from .jsonio import dumps, loads
from .pipeline_contract_data import adapt


def build(mixture, output, supplement=None, model='models/tractor'):
    import torch

    from .model import Extractor
    mixture, root = Path(mixture), Path(output)
    manifest = loads((mixture/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(mixture/name) != digest:
            raise ValueError('Original mixture changed')
    inputs = [Path(v['path']) for v in manifest['inputs'].values()]
    for source in inputs:
        verified(source)
    identities = [loads(line) for line in (mixture/'train.identities.jsonl').open()]
    wanted = {e['id'] for e in identities}
    originals, natives, charity = {}, {}, defaultdict(list)
    for source in inputs:
        for row, _, native in pairs(source, 'train'):
            if row['id'] in wanted or row['source'] == 'kleister-charity':
                originals[row['id']] = Episode.model_validate(row)
                natives[row['id']] = loads(native)
            if row['source'] == 'kleister-charity':
                charity[row['source_group']].append(row['id'])
    if not wanted <= originals.keys():
        raise ValueError('Missing original training identities')
    # Pilot repair mix: all distinct prose records and metadata, broad retention.
    focused, retention = [], []
    for ident in sorted(wanted):
        e = originals[ident]
        (focused if e.family == 'metadata' or e.artifact == 'repeated_records' else retention).append(ident)
    rng = random.Random(2026101902)
    rng.shuffle(retention)
    schedule = focused + retention[:8000]
    counts = Counter(originals[i].source_group for i in schedule)
    for group, candidates in sorted(charity.items()):
        for ident in sorted(candidates, key=lambda x: hashlib.sha256(x.encode()).digest()):
            if counts[group] >= 3:
                break
            if ident not in schedule:
                schedule.append(ident)
                counts[group] += 1
    supplement_ids = set()
    if supplement:
        for e in read_episodes(supplement):
            if e.split != 'train' or e.id in originals:
                raise ValueError('Supplement identity or split violation')
            row, removed = restrict(e.model_dump())
            if row is None or removed:
                continue
            originals[e.id] = e
            schedule.append(e.id)
            supplement_ids.add(e.id)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    extractor = Extractor(model, 'cpu')
    adapted, native_rows, rejected, gaps = {}, {}, [], []
    for ident in dict.fromkeys(schedule):
        original = originals[ident]
        e, missing = adapt(original)
        gaps.extend(missing)
        try:
            if e.tasks != original.tasks or any(m['reason'] in {'source_label_boundary_repaired', 'source_title_boundary_repaired'} for m in missing) or ident in supplement_ids or ident not in natives:
                native, _ = validate_native(e, extractor)
            else:
                native = natives[ident]
        except Exception as exc:
            rejected.append({'id': ident, 'reason': type(exc).__name__, 'message': str(exc)[:200]})
            continue
        adapted[ident], native_rows[ident] = e, native
    schedule = [i for i in schedule if i in adapted]
    rng.shuffle(schedule)
    dev = []
    # Keep the exact same Dev identities for comparable before/after training.
    for e in read_episodes(mixture/'dev.selection.episodes.jsonl'):
        new, missing = adapt(e)
        gaps.extend(missing)
        dev.append(new)
    audit = audit_episodes([*adapted.values(), *dev])
    # Original qualified corpora already have frozen cross-split provenance. New
    # supplement must also be checked against ALL heldout texts/groups, not just Dev selection.
    forbidden_groups, forbidden_texts = set(), set()
    from .document_duplicates import Index
    index = Index()
    for ident in supplement_ids & adapted.keys():
        index.check(adapted[ident].model_dump())
    duplicate_supplement = set()
    for source in inputs:
        for split in ('dev', 'calibration', 'test'):
            path = source/f'{split}.episodes.jsonl'
            if not path.exists():
                continue
            if sha(path) != loads((source/'manifest.json').read_text())['files'][path.name]:
                raise ValueError('Heldout source changed')
            for line in path.open():
                row = loads(line)
                forbidden_groups.update((row['source_group'], row['derivation_group']))
                forbidden_texts.add(text_key(row['text']))
                # Only compare supplement with heldout; avoid a huge retained index.
                if supplement_ids:
                    for match in index.check(row, cross_only=True, add=False):
                        if match['prior_id'] in supplement_ids:
                            duplicate_supplement.add(match['prior_id'])
    if any(e.source_group in forbidden_groups or e.derivation_group in forbidden_groups or text_key(e.text) in forbidden_texts
           for ident, e in adapted.items() if ident not in supplement_ids):
        raise ValueError('Original training source overlaps heldout')
    for ident in supplement_ids & adapted.keys():
        e = adapted[ident]
        if ident in duplicate_supplement or e.source_group in forbidden_groups or text_key(e.text) in forbidden_texts:
            rejected.append({'id': ident, 'reason': 'heldout_overlap'})
    excluded = {r['id'] for r in rejected}
    schedule = [i for i in schedule if i not in excluded]
    for split, rows in [('train', [adapted[i] for i in schedule]), ('dev', dev)]:
        with (root/f'{split}.episodes.jsonl').open('w') as ef, (root/f'{split}.native.jsonl').open('w') as nf:
            for e in rows:
                native = native_rows[e.id] if split == 'train' else native_episode(e)[0]
                ef.write(dumps(e.model_dump())+'\n')
                nf.write(dumps(native)+'\n')
    (root/'annotation-gaps.json').write_text(dumps(gaps, indent=2))
    (root/'rejections.json').write_text(dumps(rejected, indent=2))
    report = {'training_ready': True, 'scope': 'Explicit prose extraction plus programmatic normalization',
              'train_examples': len(schedule), 'unique_train_examples': len(set(schedule)), 'dev_examples': len(dev),
              'training_by_source': dict(Counter(adapted[i].source for i in schedule)),
              'training_by_domain': dict(Counter(adapted[i].family for i in schedule)),
              'training_by_shape': dict(Counter(adapted[i].artifact for i in schedule)),
              'source_groups': len({adapted[i].source_group for i in schedule}),
              'supplement_accepted': len(set(schedule) & supplement_ids), 'rejected': len(rejected),
              'max_exposures_per_example': max(Counter(schedule).values()),
              'annotation_gaps': sum(g['reason'] not in {'source_label_boundary_repaired', 'source_title_boundary_repaired'} for g in gaps),
              'source_boundary_repairs': sum(g['reason'] in {'source_label_boundary_repaired', 'source_title_boundary_repaired'} for g in gaps), 'audit': audit,
              'source_manifest_sha256': sha(mixture/'manifest.json'),
              'contract_code_sha256': sha(Path(__file__).with_name('pipeline_contract_data.py')),
              'qualification': 'Bounded repair pilot; all records/metadata plus general retention. Full 200k public reservoir remains unchanged. Heldout split membership frozen. Missing final annotations do not count as correct. No promotion.',
              'files': {p.name: sha(p) for p in root.glob('*.jsonl')}}
    (root/'manifest.json').write_text(dumps(report, indent=2))
    print(dumps(report), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mixture', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--supplement')
    p.add_argument('--model', default='models/tractor')
    build(**vars(p.parse_args()))
