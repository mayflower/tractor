"""Full DE/EN corpus compiler for the production BiQwen request format."""

import argparse
import hashlib
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

from .api_training import supervised_requests
from .biqwen import load_runtime, schema_for, verify_bundle
from .coverage_assembly import sha
from .episodes import Episode, text_key
from .extraction_scope import in_scope
from .jsonio import dumps, loads
from .model import field_ids
from .source_policy import excluded_source

VERSION = 'biqwen-api-multiple-evidence-v1'
_runtime = None
_tokenizer = None
_limits = None


class UnsuitableEvidence(ValueError):
    pass


def native_example(request, records):
    ids = field_ids(request)
    fields = {fid: set() for fid in ids.values()}
    output = []
    owners, anchors = {}, set()
    if not request.anchor and len(records) != 1:
        raise UnsuitableEvidence('non_record_target_count')
    for index, row in enumerate(records):
        if set(row) != set(ids):
            raise ValueError('Target fields differ from the public request')
        spans = {}
        for key, values in row.items():
            pairs = set()
            for value in values:
                a, b = value['start'], value['end']
                if not 0 <= a < b <= len(request.text) or request.text[a:b] != value['text']:
                    raise ValueError('Original target offsets/text changed')
                pairs.add((a, b))
            spans[ids[key]] = sorted(pairs)
            fields[ids[key]].update(pairs)
            if request.anchor:
                for pair in pairs:
                    owner = (ids[key], pair)
                    if owner in owners and owners[owner] != index:
                        raise UnsuitableEvidence('shared_record_occurrence')
                    owners[owner] = index
        if request.anchor:
            anchor = spans.pop(ids[request.anchor])
            if len(anchor) != 1:
                raise UnsuitableEvidence('record_requires_one_anchor_occurrence')
            if tuple(anchor[0]) in anchors:
                raise UnsuitableEvidence('duplicate_record_anchor')
            anchors.add(tuple(anchor[0]))
            output.append({'table_id': 'record', 'anchor': anchor[0], 'fields': spans})
    return {'format': VERSION, 'text': request.text, 'schema': schema_for(request),
            'evidence': {'fields': {k: sorted(v) for k, v in fields.items()}, 'records': output}}


def encode_example(example, encoding, tokenizer, limits):
    """Build exact token targets, preserving all occurrences and record bindings."""
    if example.get('format') != VERSION or example['schema'].get('choices'):
        raise ValueError('Expected extraction-only public API supervision')
    try:
        encoded = encoding.encode({'text': example['text'], 'schema': example['schema']}, tokenizer,
                                  limits['max_length'], schema_prefix=limits['schema_prefix'])
    except encoding.CapacityError as exc:
        raise UnsuitableEvidence('token_limit') from exc
    index = encoding.build_offset_index(encoded['offsets'])

    def pair(chars):
        try:
            result = encoding.char_span_to_tokens(example['text'], encoded['offsets'], *chars, index=index)
        except ValueError as exc:
            raise UnsuitableEvidence('non_token_aligned_span') from exc
        if result[1] - result[0] > limits['max_span_tokens']:
            raise UnsuitableEvidence('span_token_limit')
        if tuple(encoding.token_span_to_chars(example['text'], encoded['offsets'], result)) != tuple(chars):
            raise ValueError('Token conversion changed source evidence')
        return tuple(result)

    evidence = example['evidence']
    if len(evidence['records']) > limits['max_records']:
        raise UnsuitableEvidence('record_capacity')
    if set(evidence['fields']) != {f['id'] for f in example['schema']['fields']}:
        raise ValueError('Evidence and schema fields differ')
    links = {}
    for fid, values in evidence['fields'].items():
        if len(values) > limits['max_gold_spans'] or len(values) >= limits['candidate_top_k']:
            raise UnsuitableEvidence('field_evidence_capacity')
        links[fid] = [{'source': pair(v), 'value': pair(v), 'relation': 'direct'} for v in values]
    records = []
    for row in evidence['records']:
        fields = {fid: [pair(v) for v in values] for fid, values in row['fields'].items()}
        records.append({'table_id': row['table_id'], 'anchor': pair(row['anchor']), 'fields': fields})
    encoded['gold'] = {'links': links, 'records': records, 'choices': {}, 'choice_targets': {}}
    return encoded


def initialize(model):
    from transformers import AutoTokenizer

    global _runtime, _tokenizer, _limits
    path = Path(model).resolve()
    _, _, _runtime = load_runtime(path)
    _tokenizer = AutoTokenizer.from_pretrained(path / 'model/tokenizer', local_files_only=True)
    _limits = loads((path / 'model/extractor_config.json').read_text())['config']


def compile_batch(lines):
    result = []
    for line in lines:
        raw = loads(line)
        if excluded_source(raw) or not in_scope(raw) or raw['language'] not in {'de', 'en'}:
            raise ValueError('Unclean source in the full corpus')
        episode = Episode.model_validate(raw)
        requests, rejected = supervised_requests(episode)
        accepted = []
        for task, request, records in requests:
            try:
                example = native_example(request, records)
                encoded = encode_example(example, _runtime, _tokenizer, _limits)
            except UnsuitableEvidence as exc:
                rejected.append({'task': task, 'reason': str(exc)})
                continue
            identity = {'id': episode.id, 'task': task, 'split': episode.split,
                        'source_group': episode.source_group, 'source': episode.source,
                        'language': episode.language}
            example['id'] = hashlib.sha256(dumps([episode.id, task]).encode()).hexdigest()
            example['identity'] = identity
            example['tokens'] = len(encoded['input_ids'])
            accepted.append(example)
        result.append((raw, accepted, rejected))
    return result


def verify_source(source, selection):
    """Recheck the full selection and every heldout ID/group/text before compiling."""
    manifest = loads((source / 'manifest.json').read_text())
    if manifest.get('status') != 'assembled' or not manifest.get('inputs'):
        raise ValueError('Expected the assembled full corpus, never a sampled derivative')
    for name in ['train.episodes.jsonl', 'dev.episodes.jsonl']:
        if sha(source / name) != manifest['files'][name]:
            raise ValueError('Full source changed')
    config = loads(selection.read_text())
    if set(config) != {'languages', 'sources', 'heldout', 'dev'} or set(config['languages']) != {'de', 'en'}:
        raise ValueError('Sampling, quotas or missing language in source selection')
    if {s['path']: s['sha256'] for s in config['sources']} != {
            k: v['sha256'] for k, v in manifest['inputs'].items()}:
        raise ValueError('Full source selection differs from assembled inputs')
    forbidden_ids, groups, texts = set(), set(), set()
    for spec in config['heldout']:
        path = Path(spec['path'])
        if sha(path) != spec['sha256']:
            raise ValueError(f'Heldout artifact changed: {path}')
        for line in path.open():
            row = loads(line)
            if row['split'] not in {'dev', 'test', 'calibration'} or excluded_source(row):
                raise ValueError('Invalid heldout registry')
            forbidden_ids.add(row['id'])
            groups.update((row['source_group'], row['derivation_group']))
            texts.add(text_key(row['text']))
    count, seen = 0, set()
    for line in (source / 'train.episodes.jsonl').open():
        row = loads(line)
        if (row['split'] != 'train' or row['id'] in forbidden_ids or row['id'] in seen
                or groups & {row['source_group'], row['derivation_group']} or text_key(row['text']) in texts):
            raise ValueError('Split leakage or duplicate training identity')
        seen.add(row['id'])
        count += 1
    if count != manifest['unique_train_episodes']:
        raise ValueError('Full training inventory differs')
    return manifest


def build(source, output, model, selection='configs/full-training-data.json', workers=4):
    source, root, model = Path(source), Path(output), Path(model).resolve()
    manifest = verify_source(source, Path(selection))
    lock = verify_bundle(model)
    root.mkdir(parents=True, exist_ok=False)
    counts, reasons, languages, sources, shapes = (Counter() for _ in range(5))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(str(model),)) as pool, \
            (root / 'rejections.jsonl').open('w') as rejects:
        for split in ['train', 'dev']:
            with (source / f'{split}.episodes.jsonl').open() as inp, \
                    (root / f'{split}.native.jsonl').open('w') as native, \
                    (root / f'{split}.identities.jsonl').open('w') as identities:
                for window in batched(batched(inp, 64), workers * 2):
                    for batch in pool.map(compile_batch, window):
                        for row, accepted, skipped in batch:
                            if row['split'] != split or len(accepted) + len(skipped) != len(row['tasks']):
                                raise ValueError('Split changed or unaccounted tasks')
                            counts[split + '_input_episodes'] += 1
                            counts[split + '_input_tasks'] += len(row['tasks'])
                            counts[split + '_accepted_tasks'] += len(accepted)
                            counts[split + '_rejected_tasks'] += len(skipped)
                            counts[split + '_accepted_episodes'] += bool(accepted)
                            for reject in skipped:
                                rejects.write(dumps({'id': row['id'], 'split': split,
                                                     'language': row['language'], 'source': row['source'], **reject}) + '\n')
                                reasons[split + ':' + reject['reason']] += 1
                            for example in accepted:
                                native.write(dumps(example) + '\n')
                                identities.write(dumps(example['identity']) + '\n')
                                if split == 'train':
                                    shapes['records' if example['schema']['tables'] else 'entities'] += 1
                            if split == 'train' and accepted:
                                languages[row['language']] += 1
                                sources[row['source']] += 1
                    (root / 'progress.json').write_text(dumps({'status': 'compiling', 'counts': dict(counts)}, indent=2))
                    if counts[split + '_input_episodes'] % 10240 < workers * 128:
                        print(dumps({'split': split, 'counts': dict(counts)}), flush=True)
    if counts['train_input_episodes'] != manifest['unique_train_episodes']:
        raise ValueError('Incomplete full corpus consumption')
    (root / 'dev.episodes.jsonl').write_bytes((source / 'dev.episodes.jsonl').read_bytes())
    report = {'status': 'complete', 'training_ready': True, 'full_corpus': True, 'format': VERSION,
              'source': str(source), 'source_manifest_sha256': sha(source / 'manifest.json'),
              'model_revision': lock['revision'], 'selection_sha256': sha(Path(selection)),
              'counts': dict(counts), 'unique_train_episodes': counts['train_accepted_episodes'],
              'native_requests': {s: counts[s + '_accepted_tasks'] for s in ['train', 'dev']},
              'languages': dict(languages), 'sources': dict(sources), 'training_by_task': dict(shapes),
              'rejections_by_reason': dict(reasons), 'sampling': False, 'quotas': False,
              'model_inference_performed': False, 'training_started': False,
              'semantic_audit': 'Existing source qualification retained; exhaustive technical checks are not human semantic review.',
              'compiler_sha256': {p: sha(Path(p)) for p in ['src/tractor/biqwen_training_data.py',
                 'src/tractor/api_training.py', 'src/tractor/biqwen.py', 'src/tractor/pipeline_evaluation.py']},
              'files': {p.name: sha(p) for p in root.glob('*.jsonl')}}
    (root / 'manifest.json').write_text(dumps(report, indent=2) + '\n')
    print(dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/full-training-de-en')
    parser.add_argument('--output', default='data/biqwen-full-de-en')
    parser.add_argument('--model', default='models/biqwen-extract-decide-4b')
    parser.add_argument('--selection', default='configs/full-training-data.json')
    parser.add_argument('--workers', type=int, default=4)
    build(**vars(parser.parse_args()))
