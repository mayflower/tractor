"""Literal SWDE web attributes: gold-independent visible-text windows, original page counts."""

import argparse
import hashlib
import html
import multiprocessing
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .coverage_assembly import sha
from .document_sources import envelope, windows
from .episodes import text_key
from .jsonio import dumps, loads

DOMAINS = {'auto': 'products', 'camera': 'products', 'book': 'research', 'job': 'careers',
           'movie': 'metadata', 'nbaplayer': 'contacts', 'restaurant': 'contacts', 'university': 'contacts'}


def visible_text(raw):
    from bs4 import BeautifulSoup, Comment

    soup = BeautifulSoup(raw, 'html.parser')
    for tag in soup(['script', 'style', 'noscript', 'head', 'template']):
        tag.decompose()
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        comment.extract()
    return '\n'.join(re.sub(r'\s+', ' ', text).strip() for text in soup.stripped_strings if text.strip())


def groundtruth(root, vertical):
    result = defaultdict(lambda: defaultdict(dict))
    for path in sorted((Path(root) / vertical).glob('*.txt')):
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        domain, site, field = lines[0].split('\t')
        if domain != vertical:
            raise ValueError('Wrong SWDE groundtruth vertical')
        for line in lines[2:]:
            cols = line.split('\t')
            ident, count = cols[:2]
            values = [html.unescape(x).strip() for x in cols[2:] if x != '<NULL>']
            if len(values) != int(count):
                raise ValueError('SWDE annotation count mismatch')
            result[site][ident][field] = values
    return {site: dict(rows) for site, rows in result.items()}


def adapt(raw, values, vertical, site, ident, split):
    text = visible_text(raw)
    fields, gold, excluded = {}, {}, {}
    for key, targets in values.items():
        if not targets:
            excluded[key] = 'source_absence_not_verified_locally'
            continue
        spans = {}
        for value in targets:
            # A book ISBN inside a shop's KB-<isbn> stock code is not an ISBN mention.
            # Likewise a model identifier must not match the prefix of another model.
            pattern = r'(?<![\w-])' + r'\s+'.join(re.escape(x) for x in value.split()) + r'(?![\w-])'
            matches = list(re.finditer(pattern, text)) if value else []
            if not matches:
                excluded[key] = 'not_literal_in_visible_text'
                break
            for m in matches:
                spans[m.start(), m.end()] = {'start': m.start(), 'end': m.end(), 'text': m[0]}
        if key in excluded:
            continue
        fields[key] = f'{vertical} {key.replace("_", " ")} explicitly written on the page'
        gold[key] = {'status': 'present', 'items': list(spans.values())} if spans else {'status': 'absent'}
    if len(fields) < 2 or not any(v['status'] == 'present' for v in gold.values()):
        raise ValueError('Fewer than two literal source fields')
    episode = envelope('swde', f'{vertical}:{site}:{ident}', split, DOMAINS[vertical], 'en', text,
                       [{'id': 'document', 'kind': 'entities', 'description': 'Explicit web page attributes', 'fields': fields}],
                       {'document': {'complete': True, 'records': [gold]}})
    # Source annotations are hand-written extraction rules, not per-page human annotations.
    return episode.model_copy(update={'annotation': 'silver', 'label_origin': 'deterministic'}), excluded


def init_worker(model):
    import torch

    from .model import Extractor

    torch.set_num_threads(1)
    global EXTRACTOR
    EXTRACTOR = Extractor(model, 'cpu')


def process(item):
    from .generalist_data import validate_native

    path, values, vertical, site, ident, split = item
    group = f'swde:{vertical}:{site}:{ident}'
    try:
        episode, excluded = adapt(Path(path).read_text(encoding='utf-8-sig'), values, vertical, site, ident, split)
        candidates = windows(episode, max_chars=800)
        selected = []
        for chunk in candidates:
            positive = any(a.items for row in chunk.gold['document'].records for a in row.values())
            # Bound training negatives without moving boundaries toward gold evidence.
            # Heldout retains every window for honest full-page pipeline evaluation.
            if split == 'train' and not positive and int(hashlib.sha256(chunk.id.encode()).hexdigest()[:8], 16) % 10:
                continue
            native, tokens = validate_native(chunk, EXTRACTOR)
            selected.append((chunk.model_dump(), native, tokens))
        if not selected:
            raise ValueError('No windows retained')
        return {'group': group, 'split': split, 'rows': selected, 'excluded_fields': excluded,
                'original_text_sha256': text_key(episode.text), 'source_windows': len(candidates)}
    except Exception as exc:
        return {'group': group, 'split': split, 'error': type(exc).__name__ + ': ' + str(exc)[:240]}


def build(output, sources='data/document-sources/swde', limit=0, workers=6, model='models/tractor', verticals=None):
    import py7zr

    root, sources = Path(output), Path(sources)
    adapter_hash = sha(Path(__file__))
    root.mkdir(parents=True, exist_ok=False)
    inventory = loads((sources / 'inventory.json').read_text())
    for item in inventory['files'].values():
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('SWDE source changed')
    gtroot = sources / 'groundtruth'
    if not gtroot.exists():
        with py7zr.SevenZipFile(sources / 'groundtruth.7z') as archive:
            archive.extractall(gtroot)
    gtroot = gtroot / 'groundtruth'
    counts, documents, rejected, projected = Counter(), Counter(), Counter(), Counter()
    seen_documents = {}
    split_sites = {}
    handles = {s: ((root / f'{s}.episodes.jsonl').open('w'), (root / f'{s}.native.jsonl').open('w'))
               for s in ('train', 'dev', 'calibration', 'test')}
    with (root / 'provenance.jsonl').open('w') as provenance, (root / 'rejections.jsonl').open('w') as rejects:
        with multiprocessing.get_context('spawn').Pool(workers, initializer=init_worker, initargs=(model,)) as pool:
            for vertical in verticals or DOMAINS:
                truth = groundtruth(gtroot, vertical)
                sites = sorted(truth, key=lambda s: hashlib.sha256((vertical + s).encode()).digest())
                split_sites[vertical] = {site: 'train' if i < 8 else 'dev' if i == 8 else 'calibration/test'
                                         for i, site in enumerate(sites)}
                directory = sources / f'{vertical}-pages'
                existed = directory.exists()
                if not existed:
                    with py7zr.SevenZipFile(sources / f'{vertical}.7z') as archive:
                        archive.extractall(directory)
                jobs = []
                for path in sorted(directory.rglob('*.htm')):
                    site = path.parent.name.split('-', 1)[1].split('(')[0]
                    split = split_sites[vertical][site]
                    if split == 'calibration/test':
                        split = 'calibration' if int(hashlib.sha256(path.stem.encode()).hexdigest()[:8], 16) % 2 else 'test'
                    if path.stem in truth[site]:
                        jobs.append((str(path), truth[site][path.stem], vertical, site, path.stem, split))
                if limit:
                    jobs = jobs[:limit]
                for result in pool.imap(process, jobs, chunksize=8):
                    split = result['split']
                    if 'error' not in result and result['original_text_sha256'] in seen_documents:
                        result['error'] = 'duplicate_original_page_excluded'
                    if 'error' in result:
                        rejected[vertical] += 1
                        rejects.write(dumps({k: v for k, v in result.items() if k != 'rows'}) + '\n')
                        continue
                    seen_documents[result['original_text_sha256']] = result['group']
                    ef, nf = handles[split]
                    for e, native, _ in result.pop('rows'):
                        ef.write(dumps(e) + '\n')
                        nf.write(dumps(native) + '\n')
                        counts[split] += 1
                    documents[f'{vertical}:{split}'] += 1
                    projected[vertical] += bool(result['excluded_fields'])
                    provenance.write(dumps(result) + '\n')
                print(dumps({'vertical': vertical, 'documents': dict(documents), 'rejected': dict(rejected)}), flush=True)
                if not existed:
                    shutil.rmtree(directory)  # Only the temporary extraction created above, never source archives.
    for pair in handles.values():
        for handle in pair:
            handle.close()
    report = {'status': 'native_validated_pending_adapter_audit', 'training_ready': False,
              'documents': dict(documents), 'unique_documents': sum(documents.values()),
              'unique_train_documents': sum(v for k, v in documents.items() if k.endswith(':train')),
              'windows': dict(counts), 'rejected': dict(rejected), 'projected_documents': dict(projected),
              'source_inventory_sha256': sha(sources / 'inventory.json'), 'site_splits': split_sites,
              'adapter_sha256': adapter_hash,
              'files': {p.name: sha(p) for p in root.glob('*.jsonl')},
              'qualification': 'Real HTML text, rule-derived silver labels. Literal-only field projection. '
              'Training uses positive windows plus 10% hashed empty windows; heldout keeps all windows. '
              'Window boundaries never depend on labels. Unique page counts are not window counts. '
              'Archive includes Ms-RL software license; third-party page redistribution remains unresolved.'}
    (root / 'manifest.json').write_text(dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--sources', default='data/document-sources/swde')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--model', default='models/tractor')
    p.add_argument('--verticals', nargs='+', choices=DOMAINS)
    print(dumps(build(**vars(p.parse_args())), indent=2))
