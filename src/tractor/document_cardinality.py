"""Apply declared scalar row-cell contracts without changing text, gold or split membership."""
import argparse
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .document_campaign import atomic_json
from .episodes import Episode, native_episode
from .jsonio import dumps, loads


def scalar_contract(row):
    # Authored domain schemas declare record cells as strings. VRDU source rows
    # also contain one explicitly annotated span per queried column. Other sources can
    # contain multi-span subitems and deliberately retains its list contract.
    if row['source'] in {'tractor_authored_documents', 'vrdu'}:
        for task in row['tasks']:
            if task['kind'] != 'records':
                continue
            task['cardinalities'] = {key: 'required_one' if key == task['anchor'] else 'optional_one'
                                     for key in task['fields']}
            for record in row['gold'][task['id']]['records']:
                if any(len(answer['items']) > 1 for answer in record.values()):
                    raise ValueError('Source violates declared scalar cell contract; do not discard gold values')
    return row


def build(source, output):
    source, root = Path(source), Path(output)
    manifest = loads((source/'manifest.json').read_text())
    root.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    for split in ('train', 'dev', 'calibration', 'test'):
        name = f'{split}.episodes.jsonl'
        native_name = f'{split}.native.jsonl'
        if sha(source/name) != manifest['files'][name] or sha(source/native_name) != manifest['files'][native_name]:
            raise ValueError('Source hash changed')
        with (source/name).open() as src, (source/native_name).open() as native_src, (root/name).open('w') as ep, (root/native_name).open('w') as nf:
            for line, native_line in zip(src, native_src, strict=True):
                original = loads(line)
                if original['source'] not in {'tractor_authored_documents', 'vrdu'}:
                    if original['text'] != loads(native_line)['input']:
                        raise ValueError('Source pair changed')
                    ep.write(line)
                    nf.write(native_line)
                    counts[split] += 1
                    continue
                row = scalar_contract(original)
                e = Episode.model_validate(row)
                native, skipped = native_episode(e)
                if skipped or native is None:
                    raise ValueError('Lost supervision under scalar contract')
                ep.write(dumps(e.model_dump())+'\n')
                nf.write(dumps(native)+'\n')
                counts[split] += 1
    report = dict(manifest, training_ready=False, status='scalar_contract_pending_native_sample',
                  cardinality_source_manifest_sha256=sha(source/'manifest.json'),
                  cardinality_policy='Authored string cells and VRDU common columns use scalar optional_one, exclusive spans. No gold/text/split changes.',
                  counts=dict(counts), files={p.name: sha(p) for p in root.glob('*.jsonl')})
    atomic_json(root/'manifest.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    build(**vars(parser.parse_args()))
