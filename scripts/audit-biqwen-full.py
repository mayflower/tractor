"""Independently verify full v1/v2 inventory and unchanged source evidence."""

import argparse
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from tractor.coverage_assembly import sha
from tractor.jsonio import dumps, loads


def audit(previous, current, output):
    old_root, new_root = Path(previous), Path(current)
    old_manifest = loads((old_root / 'manifest.json').read_text())
    manifest = loads((new_root / 'manifest.json').read_text())
    for root, frozen in [(old_root, old_manifest), (new_root, manifest)]:
        for name, digest in frozen['files'].items():
            if sha(root / name) != digest:
                raise ValueError(f'File binding changed: {root / name}')
    counts, languages, sources, forms, domains = (Counter() for _ in range(5))
    for split in ['train', 'dev']:
        documents = set()
        with (old_root / f'{split}.native.jsonl').open() as old_stream, \
                (new_root / f'{split}.native.jsonl').open() as new_stream, \
                (new_root / f'{split}.identities.jsonl').open() as identities:
            for old_line, new_line, identity_line in zip_longest(old_stream, new_stream, identities):
                if None in (old_line, new_line, identity_line):
                    raise ValueError('Request inventory or identity alignment changed')
                old, new, identity = loads(old_line), loads(new_line), loads(identity_line)
                if new['identity'] != identity or any(new[k] != old[k] for k in ['id', 'text', 'evidence']):
                    raise ValueError('Original text/evidence/identity changed')
                if any(identity[k] != v for k, v in old['identity'].items()):
                    raise ValueError('Source identity changed')
                if identity['split'] != split or identity['language'] not in {'de', 'en'}:
                    raise ValueError('Wrong split or language')
                if old['schema']['tables'] != new['schema']['tables']:
                    raise ValueError('Record schema changed')
                for a, b in zip(old['schema']['fields'], new['schema']['fields'], strict=True):
                    if {k: v for k, v in a.items() if k != 'description'} != {
                            k: v for k, v in b.items() if k != 'description'}:
                        raise ValueError('Field bindings changed')
                    if not b['description'].startswith(a['description'] + '\nType: '):
                        raise ValueError('Field description not preserved')
                counts[split + '_requests'] += 1
                if split == 'train':
                    forms[identity['form']] += 1
                    domains[identity['domain']] += 1
                    if identity['id'] not in documents:
                        languages[identity['language']] += 1
                        sources[identity['source']] += 1
                documents.add(identity['id'])
        counts[split + '_documents'] = len(documents)
        if counts[split + '_requests'] != manifest['native_requests'][split]:
            raise ValueError('Native request count differs from manifest')
        if len(documents) != manifest['counts'][split + '_accepted_episodes']:
            raise ValueError('Document count differs from manifest')
    if dict(languages) != manifest['languages'] or dict(sources) != manifest['sources']:
        raise ValueError('Language/source count differs')
    if dict(forms) != manifest['training_by_task']:
        raise ValueError('Output form count differs')
    if sha(old_root / 'rejections.jsonl') != sha(new_root / 'rejections.jsonl'):
        raise ValueError('Exclusions changed; a separate source-based review is required')
    result = {'status': 'passed', 'counts': dict(counts), 'languages': dict(languages),
              'sources': dict(sources), 'request_forms': dict(forms), 'request_domains': dict(domains),
              'all_original_text_and_evidence_unchanged': True, 'identities_aligned': True,
              'all_exclusions_unchanged': True, 'manifest_sha256': sha(new_root / 'manifest.json'),
              'scope': 'Complete technical inventory; not exhaustive human semantic review.'}
    Path(output).write_text(dumps(result, indent=2) + '\n')
    print(dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', default='data/biqwen-full-de-en')
    parser.add_argument('--current', default='data/biqwen-full-de-en-v2')
    parser.add_argument('--output', required=True)
    audit(**vars(parser.parse_args()))
