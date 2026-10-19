"""Resume a qualified authoring campaign until accepted cell quotas are filled.

The durable Teacher ledger is shared with all earlier runs. Rejections do not count;
all final episode/native pairs are written from checked candidates, never raw renders.
"""

import argparse
import fcntl
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .coverage_assembly import sha
from .document_build import SPLITS, check_scale_gate, render
from .document_critic import review as independent_review
from .episodes import Episode, audit_episodes, text_key
from .jsonio import dumps, loads


def quotas(config, train_count=30000, heldout_per_domain=200):
    target = {}
    for split in SPLITS:
        for domain, weight in config['domains'].items():
            count = train_count * weight // 100 if split == 'train' else heldout_per_domain
            for shape, percent in config['shapes'].items():
                for language in config['languages']:
                    target[split, domain, shape, language] = count * percent // 200
    return target


def cell(episode):
    return episode['split'], episode['family'], episode['artifact'], episode['language']


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(dumps(value, indent=2))
    tmp.replace(path)


def run(output, env=None, workers=16, batch=512, train_count=30000, heldout_per_domain=200, model='models/tractor'):
    import torch

    from .generalist_data import validate_native
    from .model import Extractor

    root = Path(output)
    with (root / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_scale_gate(root, 257)
        scale_audit = loads((root / 'scale-audit.json').read_text())
        excluded_ids = set(scale_audit.get('excluded_ids', []))
        config = loads((root / 'manifest.json').read_text())['config']
        target = quotas(config, train_count, heldout_per_domain)
        checked = root / 'native_checked'
        checked.mkdir(exist_ok=True)
        torch.set_num_threads(2)
        extractor = Extractor(model, 'cpu')
        model_hash = sha(Path(model) / 'model.safetensors')
        accepted, counts, seen = [], Counter(), set()
        inspected = set()
        rejected = Counter()

        def scan():
            nonlocal accepted
            pending_paths = [p for p in sorted((root / 'reviewed').glob('*.json')) if p.name not in inspected]
            rows = {p.name: loads(p.read_text()) for p in pending_paths}
            review_rows = [r for r in rows.values() if r['accepted_by_teacher']]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                independent = {r['id']: r for r in pool.map(lambda r: independent_review(r, root, env or '~/.env'), review_rows)}
            for path in pending_paths:
                if path.name in inspected:
                    continue
                inspected.add(path.name)
                result = rows[path.name]
                if result['id'] in excluded_ids:
                    rejected['direct_sample_audit'] += 1
                    continue
                if not result['accepted_by_teacher']:
                    rejected['teacher_or_render'] += 1
                    continue
                if not independent[result['id']]['passed']:
                    rejected['independent_review'] += 1
                    continue
                episode = result['episode']
                key = cell(episode)
                if key not in target:
                    raise ValueError('Candidate outside declared quota cells')
                dest = checked / path.name
                if dest.exists():
                    validated = loads(dest.read_text())
                    if validated['reviewed_sha256'] != sha(path) or validated['model_sha256'] != model_hash:
                        raise ValueError('Native validation cache provenance mismatch')
                else:
                    try:
                        native, tokens = validate_native(Episode.model_validate(episode), extractor)
                        validated = {'id': episode['id'], 'passed': True, 'episode': episode, 'native': native, 'tokens': tokens}
                    except Exception as exc:
                        validated = {'id': episode['id'], 'passed': False, 'reason': type(exc).__name__ + ': ' + str(exc)[:240]}
                    validated.update(reviewed_sha256=sha(path), model_sha256=model_hash)
                    atomic_json(dest, validated)
                if not validated['passed']:
                    rejected['native'] += 1
                    continue
                text_hash = text_key(episode['text'])
                if text_hash in seen:
                    rejected['duplicate_text'] += 1
                    continue
                if counts[key] >= target[key]:
                    rejected['quota_surplus'] += 1
                    continue
                seen.add(text_hash)
                accepted.append(dest)
                counts[key] += 1

        def status(state):
            report = {'status': state, 'accepted': sum(counts.values()), 'target': sum(target.values()),
                      'accepted_by_split': {s: sum(n for k, n in counts.items() if k[0] == s) for s in SPLITS},
                      'cells': {':'.join(k): {'accepted': counts[k], 'target': n} for k, n in target.items()},
                      'rejected': dict(rejected), 'reviewed': len(inspected), 'recipe_sha256': sha(root / 'recipe.json'),
                      'budget': loads(Path('reports/training/teacher-budget.json').read_text())}
            atomic_json(root / 'campaign-status.json', report)
            print(dumps({k: v for k, v in report.items() if k != 'cells'}), flush=True)
            return report

        scan()
        while True:
            missing = {key for key, n in target.items() if counts[key] < n}
            status('running' if missing else 'accepted_quotas_complete')
            if not missing:
                break
            result = render(root, limit=batch, env=env, workers=workers, wanted_cells=missing)
            if result['processed'] == 0:
                status('planned_candidates_exhausted')
                from .document_refill import refill

                refill(root)
                continue
            scan()
        destination = root / 'dataset'
        destination.mkdir(exist_ok=False)
        episodes = []
        with (destination / 'selection.jsonl').open('w') as selection:
            streams = {s: ((destination / f'{s}.episodes.jsonl').open('w'), (destination / f'{s}.native.jsonl').open('w')) for s in SPLITS}
            for path in accepted:
                result = loads(path.read_text())
                episode = result['episode']
                episodes.append(Episode.model_validate(episode))
                ef, nf = streams[episode['split']]
                ef.write(dumps(episode) + '\n')
                nf.write(dumps(result['native']) + '\n')
                selection.write(dumps({'id': episode['id'], 'validation_sha256': sha(path)}) + '\n')
            for pair in streams.values():
                for stream in pair:
                    stream.close()
        audit = audit_episodes(episodes)
        report = status('complete_pending_corpus_audit')
        report.update(training_ready=False, audit=audit,
                      files={p.name: sha(p) for p in destination.glob('*.jsonl')},
                      qualification='Accepted native-validated Silver synthetic documents; independent final sample, near-duplicate and mixture audit required before training. No model quality claim.')
        atomic_json(destination / 'manifest.json', report)
        return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--env')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--train-count', type=int, default=30000)
    p.add_argument('--heldout-per-domain', type=int, default=200)
    p.add_argument('--model', default='models/tractor')
    run(**vars(p.parse_args()))
