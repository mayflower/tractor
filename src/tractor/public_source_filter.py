"""Review every document window and retain only independently accepted originals.

NER annotations are preserved as a separate existing source pool. Review never
rewrites source text, values, offsets or origin labels. Rejected candidates are
not part of the qualified count.
"""
import argparse
import fcntl
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .coverage_assembly import sha
from .document_campaign import atomic_json
from .jsonio import dumps, loads
from .public_source_review import review_batch


def run(source, output, workers=32, env='~/.env'):
    source, root = Path(source), Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'reviews').mkdir(exist_ok=True)
    with (root / 'filter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = loads((source / 'manifest.json').read_text())
        recipe = {'source_manifest_sha256': sha(source / 'manifest.json'),
                  'prompt_sha256': sha(Path('configs/public-source-review.txt')),
                  'reviewer_code_sha256': sha(Path(__file__).with_name('public_source_review.py')),
                  'filter_code_sha256': sha(Path(__file__))}
        recipe_path = root / 'recipe.json'
        if recipe_path.exists() and loads(recipe_path.read_text()) != recipe:
            raise ValueError('Review recipe changed; use a new output directory')
        atomic_json(recipe_path, recipe)
        if (root / 'manifest.json').exists():
            raise ValueError('Completed output already exists')
        counts, reviewed, rejected, groups, negatives = Counter(), Counter(), Counter(), {}, Counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for split in ('test', 'calibration', 'dev', 'train'):
                ep, np = source / f'{split}.episodes.jsonl', source / f'{split}.native.jsonl'
                if sha(ep) != manifest['files'][ep.name] or sha(np) != manifest['files'][np.name]:
                    raise ValueError('Source changed')
                with ep.open() as ef, np.open() as nf, (root / ep.name).open('w') as eo, (root / np.name).open('w') as no, \
                     (root / f'{split}.rejections.jsonl').open('w') as reject:
                    pending = []

                    def write(row, eline, nline):
                        eo.write(eline)
                        no.write(nline)
                        counts[split] += 1
                        groups[row['source_group']] = (split, row['source'], row['label_origin'])
                        if not any(a['items'] for g in row['gold'].values() for r in g['records'] for a in r.values()):
                            negatives[f'{split}:{row["source"]}'] += 1

                    def flush():
                        batches = [pending[i:i+8] for i in range(0, len(pending), 8)]
                        reports = pool.map(lambda batch: review_batch([r[0] for r in batch], root, env), batches)
                        for batch, report in zip(batches, reports, strict=True):
                            verdicts = {r['id']: r for r in report['results']}
                            for row, eline, nline in batch:
                                key = f'{split}:{row["source"]}'
                                reviewed[key] += 1
                                verdict = verdicts[row['id']]
                                if verdict['passed']:
                                    write(row, eline, nline)
                                else:
                                    rejected[key] += 1
                                    reject.write(dumps(verdict) + '\n')
                        eo.flush()
                        no.flush()
                        status = {'status': 'reviewing', 'counts': dict(counts), 'reviewed': dict(reviewed),
                                  'rejected': dict(rejected), 'unique_real_train_so_far': sum(v[0] == 'train' for v in groups.values()),
                                  'budget': loads(Path('reports/training/teacher-budget.json').read_text())}
                        atomic_json(root / 'status.json', status)
                        print(dumps(status), flush=True)
                        pending.clear()

                    for eline, nline in zip(ef, nf, strict=True):
                        row, native = loads(eline), loads(nline)
                        if row['text'] != native['input'] or row['split'] != split:
                            raise ValueError('Unpaired source')
                        if row['source'] in {'few_nerd', 'docred'}:
                            write(row, eline, nline)
                        else:
                            pending.append((row, eline, nline))
                        if len(pending) >= workers * 8 * 4:
                            flush()
                    if pending:
                        flush()
        group_counts = Counter(':'.join(value) for value in groups.values())
        real_train = sum(v[0] == 'train' for v in groups.values())
        report = {'status': 'semantic_filter_complete_pending_final_audit', 'training_ready': False,
                  'counts': dict(counts), 'unique_source_examples': dict(group_counts), 'unique_real_train': real_train,
                  'minimum_real_train_met': real_train >= 200000, 'source_level_negatives': dict(negatives),
                  'reviewed': dict(reviewed), 'rejected': dict(rejected), 'recipe': recipe,
                  'files': {p.name: sha(p) for p in root.glob('*.jsonl')},
                  'qualification': 'Every retained document window accepted by independent GPT-6 Luna role review. '
                  'Existing Few-NERD/DocRED source annotations retained separately without bulk reannotation. '
                  'Machine review does not certify human quality. Native source validation and exact split/group dedup preceded review. '
                  'No text, value or offset was rewritten by the reviewer. Final near-duplicate and sample audit remains required.'}
        atomic_json(root / 'manifest.json', report)
        atomic_json(root / 'status.json', report)
        return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--env', default='~/.env')
    run(**vars(p.parse_args()))
