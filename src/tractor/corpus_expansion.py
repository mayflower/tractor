"""Full local source inventory and bounded, reproducible native pilot materialization."""

import argparse
import hashlib
import random
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from .episodes import Episode, audit_episodes, text_key
from .generalist_data import read_episodes, validate_native
from .jsonio import dumps, loads
from .source_import import docred, gretel, guidex, multispan, qa


def example_key(episode):
    payload = {'text': episode.text, 'tasks': [t.model_dump() for t in episode.tasks],
               'gold': {k: v.model_dump() for k, v in episode.gold.items()}}
    return hashlib.sha256(dumps(payload, sort_keys=True).encode()).hexdigest()


def product_episode(row):
    """Replay complete synthetic spans, independently of downstream scalar API issues.

    Multiple annotated values are positive extraction targets even when the API
    must return ambiguous_value. Invalid units/numbers likewise retain evidence;
    normalization is not an annotation-completeness signal.
    """
    if row.get('origin') != 'synthetic':
        raise ValueError('Product replay requires complete synthetic gold records')
    from .model import description, field_ids
    from .types import ExtractRequest

    request = ExtractRequest.model_validate(row['request'])
    ids = field_ids(request)
    fields = {ids[k]: description(v) for k, v in request.fields.items()}
    anchor = ids[request.anchor] if request.anchor else None
    records = [{ids[k]: {'status': 'present' if spans else 'absent', 'items': spans}
                for k, spans in record.items()} for record in row['gold']['records']]
    if not anchor and len(records) != 1:
        raise ValueError('Document replay requires one gold record')
    task = {'id': 'record', 'kind': 'records' if anchor else 'entities',
            'description': 'Product measurements', 'fields': fields, 'anchor': anchor}
    return Episode.model_validate({
        'id': 'product_replay:' + row['id'], 'source': 'product_replay',
        'source_group': 'product_replay:' + row['source_group'],
        'derivation_group': 'product_replay:' + row['source_group'],
        'original_split': 'train', 'split': 'train', 'family': 'product_measurements',
        'language': row.get('metadata', {}).get('language', 'unknown'), 'artifact': row['scenario'],
        'annotation': 'synthetic_gold', 'text': request.text, 'tasks': [task],
        'gold': {'record': {'complete': True, 'records': records}},
    })


def source_rows(path):
    if Path(path).suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    else:
        data = loads(Path(path).read_text())
        yield from data['data'] if isinstance(data, dict) else data


def import_all(config, root):
    root.mkdir(parents=True, exist_ok=False)
    report, seen_ids, text_splits, conflicts = {}, set(), {}, set()
    with ExitStack() as stack:
        streams = {s: stack.enter_context((root / f'{s}.episodes.jsonl').open('w'))
                   for s in ('train', 'dev', 'calibration', 'test')}

        def emit(episode, counts):
            if episode.id in seen_ids:
                counts['duplicate_id'] += 1
                return
            seen_ids.add(episode.id)
            key = text_key(episode.text)
            previous = text_splits.setdefault(key, episode.split)
            if previous != episode.split:
                conflicts.add(key)
            streams[episode.split].write(dumps(episode.model_dump()) + '\n')
            counts['accepted'] += 1
            counts['split:' + episode.split] += 1

        for source, path in config['sources'].items():
            counts = Counter()
            for row in source_rows(path):
                counts['read'] += 1
                try:
                    episode = qa(row, source) if source in ('germanquad', 'squad_v2') else {
                        'gretel': gretel, 'guidex': guidex, 'multispan': multispan, 'docred': docred}[source](row)
                    emit(episode, counts)
                except (ValueError, KeyError, IndexError, SyntaxError) as exc:
                    counts['rejected:' + type(exc).__name__ + ':' + str(exc).splitlines()[0][:150]] += 1
            with Path(path).open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            report[source] = {'counts': dict(counts), 'sha256': checksum, 'file': path}
            print(source, dict(counts), flush=True)
        counts = Counter()
        for path in config['synthetic_files']:
            for episode in read_episodes(path):
                if episode.annotation == 'synthetic_gold' and episode.source in {
                    'tractor_fact_generator', 'tractor_contrast_generator'
                }:
                    counts['read'] += 1
                    emit(episode, counts)
        report['synthetic'] = {'counts': dict(counts), 'files': config['synthetic_files']}
        counts = Counter()
        with Path(config['product_train']).open() as stream:
            for line in stream:
                counts['read'] += 1
                try:
                    emit(product_episode(loads(line)), counts)
                except ValueError as exc:
                    counts['rejected:' + str(exc).splitlines()[0][:150]] += 1
        report['product_replay'] = {'counts': dict(counts), 'file': config['product_train']}
    (root / 'conflicting-text-hashes.json').write_text(dumps(sorted(conflicts)))
    (root / 'manifest.json').write_text(dumps({'sources': report, 'cross_split_text_conflicts': len(conflicts),
        'native_validation': 'Not yet performed on the full parsed pool; performed on every selected pilot row',
        'unsupported': 'DocRED relations; recursive GuideX structures; overlong contexts are not cropped'}, indent=2))


def source_bucket(episode):
    if episode.source == 'product_replay':
        return episode.source
    if episode.text_origin == 'synthetic' and episode.label_origin == 'deterministic':
        return 'synthetic'
    return episode.source


def build_pilot(config, pool, output, model):
    from .model import Extractor

    for split in ('train', 'dev'):
        targets = config[split + '_targets']
        if any(n < 0 for n in targets.values()) or sum(targets.values()) > config[split + '_size']:
            raise ValueError('Source targets must be nonnegative and not exceed the split size')
    output.mkdir(parents=True, exist_ok=False)
    blocked_path = pool / 'near-duplicate-blocked-groups.json'
    blocked_groups = set(loads(blocked_path.read_text())) if blocked_path.exists() else set()
    extractor = Extractor(model, 'cpu')
    conflicts = set(loads((pool / 'conflicting-text-hashes.json').read_text()))
    selected, report = [], {}
    for split in ('train', 'dev'):
        by_source = defaultdict(list)
        counts = Counter()
        for episode in read_episodes(pool / f'{split}.episodes.jsonl'):
            if episode.source_group in blocked_groups:
                counts['near_duplicate_group_quarantine'] += 1
            elif text_key(episode.text) in conflicts:
                counts['cross_split_text_quarantine'] += 1
            else:
                by_source[source_bucket(episode)].append(episode)
        rng = random.Random(config['seed'])
        for rows in by_source.values():
            rng.shuffle(rows)
            # Interleave source groups: large articles must not consume the prefix.
            groups = defaultdict(list)
            for e in rows:
                groups[e.source_group].append(e)
            rows[:] = [e for i in range(max(map(len, groups.values())))
                       for group in groups.values() for e in group[i:i+1]]
        positions, accepted, documents, groups = Counter(), Counter(), Counter(), Counter()
        wanted = config[split + '_targets']
        total = config[split + '_size']
        native_rows, episodes = [], []

        def take(source, by_source=by_source, positions=positions, documents=documents,
                 groups=groups, counts=counts, episodes=episodes, native_rows=native_rows,
                 accepted=accepted, split=split):
            rows = by_source[source]
            while positions[source] < len(rows):
                e = rows[positions[source]]
                positions[source] += 1
                key = text_key(e.text)
                # Original replay layout groups contain many different source documents.
                group_cap = 100000 if source == 'product_replay' else config['max_per_source_group']
                if documents[key] >= config['max_per_document'] or groups[e.source_group] >= group_cap:
                    counts['sampling_cap:' + source] += 1
                    continue
                try:
                    native, tokens = validate_native(e, extractor)
                except Exception as exc:
                    from .types import TractorError
                    if not isinstance(exc, (ValueError, RuntimeError, TractorError)):
                        raise
                    counts['native_rejected:' + source + ':' + type(exc).__name__ + ':' + str(exc)[:100]] += 1
                    continue
                episodes.append(e)
                native_rows.append(native)
                documents[key] += 1
                groups[e.source_group] += 1
                accepted[source] += 1
                counts['joint_tokens'] += tokens
                if len(episodes) % 1000 == 0:
                    print(split, len(episodes), dict(accepted), flush=True)
                return True
            return False

        for source, count in wanted.items():
            for _ in range(count):
                if not take(source):
                    break
        while len(episodes) < total:
            previous = len(episodes)
            for source in config['sources']:
                if len(episodes) == total:
                    break
                take(source)
            if len(episodes) == previous:
                raise ValueError(f'Insufficient distinct supported data: {split} {len(episodes)}/{total}')
        order = list(range(len(episodes)))
        rng.shuffle(order)
        with (output / f'{split}.episodes.jsonl').open('w') as ef, (output / f'{split}.native.jsonl').open('w') as nf:
            for i in order:
                ef.write(dumps(episodes[i].model_dump()) + '\n')
                nf.write(dumps(native_rows[i]) + '\n')
        selected.extend(episodes)
        report[split] = {'episodes': len(episodes), 'unique_documents': len(documents),
                         'source_groups': len(groups), 'sources': dict(accepted), 'counts': dict(counts),
                         'families': dict(Counter(e.family for e in episodes)),
                         'languages': dict(Counter(e.language for e in episodes)),
                         'origins': dict(Counter(e.text_origin + '/' + e.label_origin for e in episodes)),
                         'task_kinds': dict(Counter(t.kind for e in episodes for t in e.tasks))}
    report['audit'] = audit_episodes(selected)
    report['config'] = config
    report['near_duplicate_blocked_groups'] = sorted(blocked_groups)
    report['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.jsonl')}
    report['limits'] = 'Parsed full-source pool is not all native-validated. Only selected rows have collator proof. No model predictions, training or final-test analysis used for selection.'
    (output / 'manifest.json').write_text(dumps(report, indent=2))
    print(dumps({s: {k: v for k, v in report[s].items() if k != 'counts'} for s in ('train', 'dev')}, indent=2))


def screen_pool(pool):
    """Find overlap across the entire real-text pool before sampling replacements."""
    from .episodes import audit_similarity

    seen, rows = set(), []
    for split in ('test', 'calibration', 'dev', 'train'):
        for episode in read_episodes(pool / f'{split}.episodes.jsonl'):
            key = (text_key(episode.text), episode.split, episode.source_group)
            if episode.text_origin == 'real' and key not in seen:
                seen.add(key)
                rows.append(episode)
    report = audit_similarity(rows)
    metadata = {e.id: e for e in rows}
    block_file = pool / 'near-duplicate-blocked-groups.json'
    blocked = set(loads(block_file.read_text())) if block_file.exists() else set()
    reserved = set()
    priority = {'test': 3, 'calibration': 2, 'dev': 1, 'train': 0}
    for match in report['matches']:
        lower = min((metadata[match['a']], metadata[match['b']]), key=lambda e: priority[e.split])
        (blocked if lower.split in ('train', 'dev') else reserved).add(lower.source_group)
    report['blocked_pilot_groups'] = sorted(blocked)
    report['reserved_only_groups'] = sorted(reserved)
    report['distinct_real_documents_and_groups'] = len(rows)
    block_file.write_text(dumps(sorted(blocked)))
    (pool / 'reserved-overlap-groups.json').write_text(dumps(sorted(reserved)))
    (pool / 'overlap-screen.json').write_text(dumps(report, indent=2))
    print(dumps({'matches': len(report['matches']), 'blocked_pilot_groups': len(blocked),
                 'reserved_only_groups': len(reserved)}, indent=2))


def audit_pilot(pool, output):
    """Check selected data against reserved groups without inspecting model outcomes."""
    from .episodes import audit_similarity

    documents, groups, counts = defaultdict(set), defaultdict(set), Counter()
    heldout = []
    reserved_path = pool / 'reserved-overlap-groups.json'
    reserved_groups = set(loads(reserved_path.read_text())) if reserved_path.exists() else set()
    for split in ('train', 'dev', 'calibration', 'test'):
        for episode in read_episodes(pool / f'{split}.episodes.jsonl'):
            counts[episode.source] += 1
            documents[episode.source].add(text_key(episode.text))
            groups[episode.source].add(episode.source_group)
            if split in ('calibration', 'test') and episode.source_group not in reserved_groups:
                heldout.append(episode)
    selected = [e for split in ('train', 'dev')
                for e in read_episodes(output / f'{split}.episodes.jsonl')]
    all_rows = selected + heldout
    keys = [example_key(e) for e in selected]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate task/gold examples in the pilot')
    report = {'duplicate_task_gold_examples': 0, 'source_inventory': {s: {'parsed_episodes': counts[s],
        'distinct_texts': len(documents[s]), 'source_groups': len(groups[s])} for s in counts},
        'exact_split_audit': audit_episodes(all_rows), 'near_duplicates': audit_similarity(all_rows)}
    metadata = {e.id: e for e in all_rows}
    blocked, reserved = set(), 0
    priority = {'test': 3, 'calibration': 2, 'dev': 1, 'train': 0}
    for match in report['near_duplicates']['matches']:
        lower = min((metadata[match['a']], metadata[match['b']]), key=lambda e: priority[e.split])
        if lower.split in ('train', 'dev'):
            blocked.add(lower.source_group)
        else:
            reserved += 1
    report['blocked_pilot_groups'] = sorted(blocked)
    report['reserved_only_matches'] = reserved
    report['quarantined_reserved_groups'] = sorted(reserved_groups)
    (output / 'audit.json').write_text(dumps(report, indent=2))
    block_file = pool / 'near-duplicate-blocked-groups.json'
    previous = set(loads(block_file.read_text())) if block_file.exists() else set()
    block_file.write_text(dumps(sorted(previous | blocked)))
    if blocked:
        raise ValueError(f'{len(blocked)} pilot groups overlap other splits; rebuild into a new output and audit again')
    print(dumps({'exact_split_audit': report['exact_split_audit'],
                 'pilot_near_matches': 0, 'reserved_only_matches': reserved}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['import', 'screen', 'build', 'audit'])
    parser.add_argument('--config', default='configs/pilot-corpus.json')
    parser.add_argument('--pool', default='data/generalist-expanded')
    parser.add_argument('--output', default='data/generalist-pilot-30k')
    parser.add_argument('--model', default='models/tractor')
    args = parser.parse_args()
    config = loads(Path(args.config).read_text())
    if args.action == 'import':
        import_all(config, Path(args.pool))
    elif args.action == 'screen':
        screen_pool(Path(args.pool))
    elif args.action == 'build':
        build_pilot(config, Path(args.pool), Path(args.output), args.model)
    else:
        audit_pilot(Path(args.pool), Path(args.output))


if __name__ == '__main__':
    main()
