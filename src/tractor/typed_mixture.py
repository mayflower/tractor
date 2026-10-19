"""Attach migrated source pools to an unchanged exposure schedule and native data."""
import argparse
from pathlib import Path

from .coverage_assembly import sha
from .jsonio import dumps, loads


def verify(root):
    root = Path(root)
    manifest = loads((root / 'manifest.json').read_text())
    if not manifest.get('type_migration'):
        raise ValueError('Input has not completed type migration')
    for name, digest in manifest['files'].items():
        if sha(root / name) != digest:
            raise ValueError(f'Migrated input changed: {root / name}')
    return manifest


def attach(mixture, synthetic, real):
    root = Path(mixture)
    manifest = verify(root)
    identities = [loads(line) for line in (root / 'train.identities.jsonl').open()]
    wanted = {e['id'] for e in identities}
    rows, inputs = {}, {}
    for kind, path in [('synthetic', synthetic), ('real', real)]:
        source = Path(path)
        verify(source)
        inputs[kind] = {'path': str(source), 'manifest_sha256': sha(source / 'manifest.json')}
        for line in (source / 'train.episodes.jsonl').open():
            row = loads(line)
            if row['id'] in wanted:
                if row['split'] != 'train' or row['id'] in rows:
                    raise ValueError('Training identity collision or wrong split')
                rows[row['id']] = line
    if rows.keys() != wanted:
        raise ValueError('Training identities missing from migrated source pools')
    target = root / 'train.episodes.jsonl'
    # Complete validation before publishing the assembled file or manifest.
    with (root / 'train.native.jsonl').open() as native:
        for identity, line in zip(identities, native, strict=True):
            row = loads(rows[identity['id']])
            if row['text'] != loads(line)['input'] or row['source_group'] != identity['source_group']:
                raise ValueError('Exposure schedule and migrated training text disagree')
    with target.open('x') as stream:
        for identity in identities:
            stream.write(rows[identity['id']])
    manifest['inputs'] = inputs
    manifest['typed_training_schedule'] = {
        'exposures': len(identities), 'unique_episodes': len(wanted),
        'native_sha256': sha(root / 'train.native.jsonl'),
        'identities_sha256': sha(root / 'train.identities.jsonl'),
        'code_sha256': sha(Path(__file__)),
        'qualification': 'Exact original exposure order, multiplicities, source groups and native supervision. '
                         'Training episode view now carries independent final targets and typed contracts.',
    }
    manifest['files'][target.name] = sha(target)
    manifest['normalization_ready'] = all(loads((Path(p) / 'manifest.json').read_text())['normalization_ready']
                                          for p in (synthetic, real))
    (root / 'manifest.json').write_text(dumps(manifest, indent=2) + '\n')
    return manifest['typed_training_schedule']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixture', required=True)
    parser.add_argument('--synthetic', required=True)
    parser.add_argument('--real', required=True)
    print(dumps(attach(**vars(parser.parse_args()))), flush=True)
