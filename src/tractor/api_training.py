"""Compile source-span supervision with exactly the public API's native schemas."""
import argparse
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .generalist_data import read_episodes
from .jsonio import dumps, loads
from .model import native_example
from .pipeline_evaluation import typed_requests_for


def examples(episode):
    """One training request per API task; final annotations never supervise spans."""
    requests = typed_requests_for(episode)
    accepted, rejected = [], []
    for task in episode.tasks:
        gold = episode.gold[task.id]
        if not gold.complete or any(answer.status not in {'present', 'absent'}
                                    for row in gold.records for answer in row.values()):
            rejected.append({'task': task.id, 'reason': 'incomplete_span_annotation'})
            continue
        # The same fragment cannot be a street and a city in the same address.
        # Weak source string matching sometimes creates this contradictory gold.
        # Quarantine supervision; do not relabel it as an absent/negative field.
        rules = episode.processing.get(task.id, {})
        conflict = False
        for row in gold.records:
            components = {}
            for key, answer in row.items():
                rule = rules.get(key)
                if rule and rule.kind == 'address' and rule.component in {'street', 'city'}:
                    for span in answer.items:
                        identity = (rule.group, span.start, span.end)
                        if identity in components and components[identity] != rule.component:
                            conflict = True
                        components[identity] = rule.component
        if conflict:
            rejected.append({'task': task.id, 'reason': 'contradictory_address_component_labels'})
            continue
        records = [{key: [span.model_dump() for span in
                    (answer.items or (answer.answer_alternatives[0] if answer.answer_alternatives else []))]
                    for key, answer in row.items()} for row in gold.records]
        request = requests[task.id]
        native = native_example({'request': request.model_dump(), 'gold': {'records': records}})
        accepted.append((task.id, request, native))
    return accepted, rejected


def validate(request, native, extractor, collator):
    """Recover the actual native targets, including original offsets and records."""
    from .model import schema_for
    tokens = extractor.preflight(request, schema_for(request))
    batch = collator([(native['input'], native['output'])])
    targets = native['output']['json_structures'] if request.anchor else [{'entities': native['output']['entities']}]
    expected = {}
    for row in targets:
        for task, fields in row.items():
            for field, spans in fields.items():
                expected.setdefault((task, field), set()).update((s['start'], s['end']) for s in spans)
    actual = {}
    for query in batch.query_layouts[0].queries:
        if not query.extractive:
            continue
        qi = query.query_id
        pairs = batch.targets.mention_pairs[0, qi][batch.targets.mention_mask[0, qi]]
        key = (query.task_name if query.task_type != 'entities' else 'entities', query.role_name)
        actual[key] = {(batch.start_mappings[0][int(pair[0])], batch.end_mappings[0][int(pair[1]) - 1]) for pair in pairs}
    if actual != expected:
        raise ValueError('Public native training lost source-span targets')
    if request.anchor:
        record_count = sum(any(fields.values()) for row in targets for fields in row.values())
        if batch.targets.record_targets is None or int(batch.targets.record_targets[2].sum()) != record_count:
            raise ValueError('Public native record target count changed')
        validate_record_bindings(targets, batch)
    return tokens


def validate_record_bindings(targets, batch):
    """Check each packed record, not just the union of its field spans."""
    def signature(fields):
        return tuple(sorted((field, tuple(sorted(set(spans)))) for field, spans in fields.items()))

    wanted = Counter(signature({field: [(s['start'], s['end']) for s in spans]
                                for field, spans in fields.items()})
                     for row in targets for fields in row.values() if any(fields.values()))
    packed_spans, packed_mask, record_mask = batch.targets.record_targets[:3]
    recovered = Counter()
    for group, spec in enumerate(batch.record_specs[0].values()):
        for record in range(record_mask.shape[2]):
            if not record_mask[0, group, record]:
                continue
            fields = {}
            for index, field in enumerate(spec.fields):
                pairs = packed_spans[0, group, record, index][packed_mask[0, group, record, index]]
                fields[field.name] = [(batch.start_mappings[0][int(start)],
                                       batch.end_mappings[0][int(end)-1]) for start, end in pairs]
            recovered[signature(fields)] += 1
    if recovered != wanted:
        raise ValueError('Public native training changed record field bindings')


def build(source, dev, output, model):
    import torch
    from gliner2.training.trainer import ExtractorCollator

    from .model import Extractor
    source, dev, root = Path(source), Path(dev), Path(output)
    manifest = loads((source / 'manifest.json').read_text())
    if manifest.get('training_ready') is False:
        raise ValueError('Source is not training ready')
    train_path = source / 'train.episodes.jsonl'
    if sha(train_path) != manifest['files'][train_path.name]:
        raise ValueError('Frozen source changed')
    dev_rows = list(read_episodes(dev))
    if not dev_rows or any(e.split != 'dev' for e in dev_rows):
        raise ValueError('Only frozen Dev selects checkpoints')
    forbidden_groups = {g for e in dev_rows for g in (e.source_group, e.derivation_group)}
    torch.set_num_threads(2)
    extractor = Extractor(model, 'cpu')
    collator = ExtractorCollator(extractor.model.processor, is_training=False, build_targets=True,
                                architecture='boundary', max_gold_per_query=64)
    root.mkdir(parents=True, exist_ok=False)
    rejected, counts, shapes, sources, token_counts = [], Counter(), Counter(), Counter(), Counter()
    seen, cached = set(), {}
    for split, rows in [('train', read_episodes(train_path)), ('dev', iter(dev_rows))]:
        with (root / f'{split}.native.jsonl').open('w') as nf, (root / f'{split}.identities.jsonl').open('w') as ids:
            for e in rows:
                if e.split != split or split == 'train' and forbidden_groups & {e.source_group, e.derivation_group}:
                    raise ValueError('Split or source-group leakage')
                if e.id not in cached:
                    compiled, skipped = examples(e)
                    rejected.extend({'id': e.id, 'split': split, **r} for r in skipped)
                    accepted = []
                    for task_id, request, native in compiled:
                        try:
                            tokens = validate(request, native, extractor, collator)
                        except Exception as exc:
                            # Only capacity exclusions are expected. Target loss
                            # or other implementation failures abort the build.
                            from .types import TractorError
                            if not isinstance(exc, TractorError) or exc.code != 'token_limit':
                                raise
                            rejected.append({'id': e.id, 'task': task_id, 'split': split, 'reason': 'token_limit'})
                            continue
                        accepted.append((task_id, native, tokens, request.many))
                    cached[e.id] = accepted
                for task_id, native, tokens, many in cached[e.id]:
                    nf.write(dumps(native) + '\n')
                    ids.write(dumps({'id': e.id, 'task': task_id, 'source_group': e.source_group, 'split': split}) + '\n')
                    counts[split] += 1
                    if split == 'train':
                        seen.add(e.id)
                        shapes['records' if many else 'entities'] += 1
                        sources[e.source] += 1
                        token_counts[str(tokens)] += 1
    (root / 'dev.episodes.jsonl').write_bytes(dev.read_bytes())
    (root / 'rejections.json').write_text(dumps(rejected, indent=2) + '\n')
    report = {'status': 'complete', 'training_ready': True, 'source': str(source),
              'source_manifest_sha256': sha(source / 'manifest.json'), 'dev_sha256': sha(dev),
              'native_requests': dict(counts), 'unique_train_episodes': len(seen), 'training_by_task': dict(shapes),
              'training_by_source': dict(sources), 'max_training_tokens': max(map(int, token_counts)),
              'rejections_by_reason': dict(Counter(r['reason'] for r in rejected)),
              'code_sha256': sha(Path(__file__)),
              'qualification': 'Exact production API schemas, one native example per request; original spans and records. '
                               'Original source split qualification retained. All Dev episodes remain in pipeline scoring, including capacity failures.',
              'files': {p.name: sha(p) for p in root.iterdir() if p.is_file()}}
    (root / 'manifest.json').write_text(dumps(report, indent=2) + '\n')
    print(dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'dev', 'output', 'model'):
        parser.add_argument('--' + arg, required=True)
    build(**vars(parser.parse_args()))
