"""Remove complete source groups flagged by a reproducible cross-split duplicate audit."""
import argparse
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .jsonio import dumps, loads


def quarantine(source, audit_path, output):
    source, output = Path(source), Path(output)
    manifest = loads((source / 'manifest.json').read_text())
    audit = loads(Path(audit_path).read_text())
    blocked = set(audit['cross_split_groups_to_quarantine'])
    for name, digest in audit['inputs'].items():
        if sha(source / name) != digest:
            raise ValueError('Audited source changed')
    output.mkdir(parents=True, exist_ok=False)
    counts, removed, groups, negatives = Counter(), Counter(), {}, Counter()
    for split in ('train', 'dev', 'calibration', 'test'):
        ep, np = source / f'{split}.episodes.jsonl', source / f'{split}.native.jsonl'
        if sha(np) != manifest['files'][np.name]:
            raise ValueError('Native source changed')
        with ep.open() as ef, np.open() as nf, (output / ep.name).open('w') as eo, (output / np.name).open('w') as no:
            for eline, nline in zip(ef, nf, strict=True):
                row = loads(eline)
                if row['text'] != loads(nline)['input'] or row['split'] != split:
                    raise ValueError('Unpaired source')
                if row['source_group'] in blocked:
                    removed[split] += 1
                    continue
                eo.write(eline)
                no.write(nline)
                counts[split] += 1
                groups[row['source_group']] = (split, row['source'], row['label_origin'])
                if not any(a['items'] for g in row['gold'].values() for r in g['records'] for a in r.values()):
                    negatives[f'{split}:{row["source"]}'] += 1
    real_train = sum(v[0] == 'train' for v in groups.values())
    report = dict(manifest, status='semantic_and_duplicate_filter_complete_pending_final_sample', training_ready=False,
                  counts=dict(counts), unique_source_examples=dict(Counter(':'.join(v) for v in groups.values())),
                  unique_real_train=real_train, minimum_real_train_met=real_train >= 200000,
                  source_level_negatives=dict(negatives), quarantine={'audit_sha256': sha(Path(audit_path)),
                  'source_manifest_sha256': sha(source / 'manifest.json'), 'removed_windows': dict(removed),
                  'excluded_source_groups': sorted(blocked)}, files={p.name: sha(p) for p in output.glob('*.jsonl')})
    report['qualification'] += ' All source groups flagged by the cross-split near-duplicate audit were removed in full. Final direct sample inspection remains pending.'
    (output / 'manifest.json').write_text(dumps(report, indent=2))
    print(dumps({'unique_real_train': real_train, 'counts': dict(counts), 'removed_windows': dict(removed)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--audit-path', required=True)
    p.add_argument('--output', required=True)
    quarantine(**vars(p.parse_args()))
