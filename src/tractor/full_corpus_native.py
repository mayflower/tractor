"""Validate and compile the entire assembled corpus without sampling."""

import argparse
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

from .api_training import examples, validate
from .coverage_assembly import sha
from .episodes import Episode
from .jsonio import dumps, loads
from .types import TractorError

_extractor = None
_collator = None


def initialize(model):
    import torch
    from gliner2.training.trainer import ExtractorCollator

    from .model import Extractor
    torch.set_num_threads(1)
    global _extractor, _collator
    _extractor = Extractor(model, 'cpu')
    _collator = ExtractorCollator(
        _extractor.model.processor, is_training=False, build_targets=True,
        architecture='boundary', max_gold_per_query=64,
    )


def compile_batch(lines):
    results = []
    for line in lines:
        episode = Episode.model_validate(loads(line))
        compiled, rejected = examples(episode)
        accepted = []
        for task, request, native in compiled:
            try:
                tokens = validate(request, native, _extractor, _collator)
            except TractorError as exc:
                if exc.code != 'token_limit':
                    raise
                rejected.append({'task': task, 'reason': exc.code})
                continue
            accepted.append((task, native, tokens, request.many))
        results.append((line, accepted, rejected))
    return results


def build(source, output, model, workers=8):
    source, root = Path(source), Path(output)
    manifest = loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'assembled' or not manifest['inputs']:
        raise ValueError('Expected an assembled full corpus')
    for name in ['train.episodes.jsonl', 'dev.episodes.jsonl']:
        if sha(source / name) != manifest['files'][name]:
            raise ValueError('Assembled input changed')
    root.mkdir(parents=True, exist_ok=False)
    counts, reasons, languages, sources, shapes = (Counter() for _ in range(5))
    seen, max_tokens = set(), 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize, initargs=(model,),
    ) as pool, (root / 'rejections.jsonl').open('w') as rejects:
        for split in ['train', 'dev']:
            with (source / f'{split}.episodes.jsonl').open() as inp, \
                    (root / f'{split}.native.jsonl').open('w') as nf, \
                    (root / f'{split}.identities.jsonl').open('w') as ids, \
                    (root / f'{split}.episodes.jsonl').open('w') as episodes:
                # Bounded submission keeps the whole corpus out of IPC memory.
                chunks = batched(inp, 64)
                for window in batched(chunks, workers * 2):
                    for result in pool.map(compile_batch, window):
                        for line, accepted, skipped in result:
                            row = loads(line)
                            if row['split'] != split:
                                raise ValueError('Split changed during compilation')
                            counts[split + '_input_episodes'] += 1
                            counts[split + '_input_tasks'] += len(row['tasks'])
                            counts[split + '_accepted_tasks'] += len(accepted)
                            counts[split + '_rejected_tasks'] += len(skipped)
                            if len(accepted) + len(skipped) != len(row['tasks']):
                                raise ValueError('Unaccounted task')
                            for rejection in skipped:
                                rejects.write(dumps({'id': row['id'], 'split': split,
                                                     'language': row['language'], **rejection}) + '\n')
                                reasons[split + ':' + rejection['reason']] += 1
                            if split == 'dev':
                                episodes.write(line)  # All Dev requests remain scored.
                            elif accepted:
                                tasks = {a[0] for a in accepted}
                                selected = dict(row, tasks=[t for t in row['tasks'] if t['id'] in tasks],
                                                gold={k: v for k, v in row['gold'].items() if k in tasks},
                                                processing={k: v for k, v in row['processing'].items() if k in tasks})
                                episodes.write(dumps(selected) + '\n')
                                seen.add(row['id'])
                                languages[row['language']] += 1
                                sources[row['source']] += 1
                            else:
                                counts['train_excluded_episodes'] += 1
                            for task, native, tokens, many in accepted:
                                nf.write(dumps(native) + '\n')
                                ids.write(dumps({'id': row['id'], 'task': task, 'split': split,
                                                 'source_group': row['source_group'],
                                                 'language': row['language']}) + '\n')
                                if split == 'train':
                                    max_tokens = max(max_tokens, tokens)
                                    shapes['records' if many else 'entities'] += 1
                    progress = {'status': 'compiling', 'counts': dict(counts)}
                    (root / 'progress.json').write_text(dumps(progress, indent=2) + '\n')
                    print(dumps(progress), flush=True)
    if counts['train_input_episodes'] != manifest['unique_train_episodes']:
        raise ValueError('Full corpus not consumed')
    if len(seen) + counts['train_excluded_episodes'] != counts['train_input_episodes']:
        raise ValueError('Unaccounted training episodes')
    if sha(root / 'dev.episodes.jsonl') != sha(source / 'dev.episodes.jsonl'):
        raise ValueError('Dev changed')
    report = {
        'status': 'complete', 'training_ready': True, 'full_corpus': True,
        'source': str(source), 'source_manifest_sha256': sha(source / 'manifest.json'),
        'counts': dict(counts), 'unique_train_episodes': len(seen),
        'native_requests': {s: counts[s + '_accepted_tasks'] for s in ['train', 'dev']},
        'languages': dict(languages), 'sources': dict(sources), 'training_by_task': dict(shapes),
        'rejections_by_reason': dict(reasons), 'max_training_tokens': max_tokens,
        'sampling': False, 'quotas': False, 'model_inference_performed': False,
        'semantic_audit': 'Existing corpus qualification retained; complete technical target validation, not exhaustive human semantic review.',
        'files': {p.name: sha(p) for p in root.iterdir() if p.is_file()},
    }
    (root / 'manifest.json').write_text(dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'output', 'model']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=8)
    print(dumps(build(**vars(parser.parse_args())), indent=2))
