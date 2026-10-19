"""Restore explicitly frozen provenance without changing requests or supervision."""
import argparse
import shutil
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .episodes import Episode
from .jsonio import dumps, loads

ORIGIN_KEYS = ('text_origin', 'label_origin')


def origins(row):
    if all(key in row for key in ORIGIN_KEYS):
        return {key: row[key] for key in ORIGIN_KEYS}
    e = Episode.model_validate(row)
    return {key: getattr(e, key) for key in ORIGIN_KEYS}


def restore(row, reference):
    if any(row[key] != reference[key] for key in ('id', 'source', 'source_group', 'split', 'text')):
        raise ValueError('Reference identity/text/split mismatch')
    result = {**row, **origins(reference)}
    # Public requests read text/tasks/processing/locale/source, all unchanged.
    if {k: v for k, v in row.items() if k not in ORIGIN_KEYS} != {
            k: v for k, v in result.items() if k not in ORIGIN_KEYS}:
        raise ValueError('Provenance restoration changed non-origin fields')
    return result


def repair(source, reference, output, training_references=()):
    source, reference, output = Path(source), Path(reference), Path(output)
    manifest = loads((source / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(source / name) != digest:
            raise ValueError('Frozen migration input changed')
    training = {}
    if training_references:
        wanted = {loads(line)['id'] for line in (source / 'train.identities.jsonl').open()}
        for path in training_references:
            for line in Path(path, 'train.episodes.jsonl').open():
                row = loads(line)
                if row['id'] in wanted:
                    if row['id'] in training:
                        raise ValueError('Duplicate training identity across reference pools')
                    training[row['id']] = row
        if training.keys() != wanted:
            raise ValueError('Missing reference training identities')
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for path in sorted(source.glob('*.episodes.jsonl')):
        counts, origin_counts = Counter(), Counter()
        refpath = reference / path.name
        if path.name == 'train.episodes.jsonl' and training:
            references = (training[loads(line)['id']] for line in (source / 'train.identities.jsonl').open())
        else:
            references = (loads(line) for line in refpath.open())
        with path.open() as stream, (output / path.name).open('w') as target:
            for line, ref in zip(stream, references, strict=True):
                row = loads(line)
                fixed = restore(row, ref)
                counts['episodes'] += 1
                counts['reconstructed_origin_was_wrong'] += origins(row) != origins(fixed)
                counts['omitted_or_wrong_explicit_origin'] += any(row.get(k) != fixed[k] for k in ORIGIN_KEYS)
                origin_counts[f"{fixed['text_origin']}/{fixed['label_origin']}"] += 1
                target.write(dumps(fixed) + '\n')
        reports[path.name] = {'counts': dict(counts), 'origins': dict(origin_counts),
                             'source_sha256': sha(path), 'output_sha256': sha(output / path.name)}
    for name in manifest['files']:
        if name not in reports:
            shutil.copyfile(source / name, output / name)
    report = {'source': str(source), 'reference': str(reference),
              'source_manifest_sha256': sha(source / 'manifest.json'),
              'training_references': {str(p): sha(Path(p, 'train.episodes.jsonl')) for p in training_references},
              'reference_hashes': {p.name: sha(p) for p in reference.glob('*.episodes.jsonl')},
              'files': reports, 'code_sha256': sha(Path(__file__)),
              'qualification': 'Only text_origin and label_origin restored from frozen reference identities. '
                               'All public request inputs, annotations, native targets and splits unchanged.'}
    (output / 'provenance-restoration.json').write_text(dumps(report, indent=2) + '\n')
    manifest['provenance_restoration'] = report
    manifest['files'] = {name: sha(output / name) for name in manifest['files']}
    manifest['files']['provenance-restoration.json'] = sha(output / 'provenance-restoration.json')
    (output / 'manifest.json').write_text(dumps(manifest, indent=2) + '\n')
    print(dumps({'output': str(output), 'files': reports}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-references', nargs='*', default=[])
    repair(**vars(parser.parse_args()))
