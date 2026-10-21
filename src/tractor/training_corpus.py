"""Assemble every eligible episode from explicitly frozen training sources."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .document_text_scope import restrict
from .episodes import text_key
from .jsonio import dumps, loads


def fingerprint(row):
    return hashlib.sha256(dumps(row, sort_keys=True).encode()).hexdigest()


def verify_training_binding(data, binding):
    """Fail before loading a model if the requested full dataset was substituted."""
    if Path(data).resolve() != Path(binding['path']).resolve():
        raise ValueError('Training data differs from the explicitly bound full dataset')
    root = Path(data)
    if sha(root / 'manifest.json') != binding['manifest_sha256']:
        raise ValueError('Full training dataset manifest changed')
    manifest = loads((root / 'manifest.json').read_text())
    if not manifest.get('full_corpus') or not manifest.get('training_ready'):
        raise ValueError('Full training dataset has not passed native validation')
    if manifest['unique_train_episodes'] != binding['unique_train_episodes']:
        raise ValueError('Full training episode count changed')
    if manifest['native_requests']['train'] != binding['native_train_requests']:
        raise ValueError('Full training request count changed')
    for name in ('train.native.jsonl', 'train.identities.jsonl', 'dev.episodes.jsonl',
                 'dev.native.jsonl'):
        if sha(root / name) != manifest['files'][name]:
            raise ValueError(f'Full training artifact changed: {name}')
    return manifest


def assemble(config_path, output):
    config_path, root = Path(config_path), Path(output)
    config = loads(config_path.read_text())
    if set(config) != {'languages', 'sources', 'heldout', 'dev'}:
        raise ValueError('Full corpus configuration cannot contain sampling or quota options')
    if set(config['languages']) != {'de', 'en'}:
        raise ValueError('The full training corpus must include German and English')
    root.mkdir(parents=True, exist_ok=False)
    forbidden_ids, forbidden_groups, forbidden_texts = set(), set(), set()
    inputs = {}
    for spec in config['heldout']:
        path = Path(spec['path'])
        if sha(path) != spec['sha256']:
            raise ValueError(f'Heldout file changed: {path}')
        for line in path.open():
            row = loads(line)
            if row['split'] not in {'dev', 'calibration', 'test'}:
                raise ValueError('Unexpected training row in heldout registry')
            forbidden_ids.add(row['id'])
            forbidden_groups.update((row['source_group'], row['derivation_group']))
            forbidden_texts.add(text_key(row['text']))
    languages, sources, families, origins = Counter(), Counter(), Counter(), Counter()
    seen, totals, task_changes = {}, Counter(), Counter()
    train = root / 'train.episodes.jsonl'
    with train.open('w') as out, (root / 'selection.jsonl').open('w') as audit:
        for spec in config['sources']:
            path = Path(spec['path'])
            if sha(path) != spec['sha256']:
                raise ValueError(f'Training source changed: {path}')
            counts = Counter()
            for number, line in enumerate(path.open(), 1):
                row = loads(line)
                event = {'source_file': str(path), 'line': number, 'id': row['id']}
                counts['input_rows'] += 1
                if row['split'] != 'train':
                    raise ValueError(f'Non-training row in training source: {row["id"]}')
                reason = None
                if row['language'] not in config['languages']:
                    reason = 'language:' + row['language']
                elif row['id'] in forbidden_ids:
                    reason = 'heldout_episode_id'
                elif forbidden_groups & {row['source_group'], row['derivation_group']}:
                    reason = 'heldout_source_or_derivation_group'
                elif text_key(row['text']) in forbidden_texts:
                    reason = 'heldout_text'
                if reason:
                    event.update(action='excluded', reason=reason)
                    counts['excluded:' + reason] += 1
                else:
                    original_tasks = {t['id'] for t in row['tasks']}
                    row, removed = restrict(row)
                    for change in removed:
                        task_changes[change['reason']] += 1
                    if removed:
                        event['removed_tasks'] = removed
                    if row is None:
                        event.update(action='excluded', reason='no_in_scope_tasks')
                        counts['excluded:no_in_scope_tasks'] += 1
                    elif any(t['kind'] not in {'entities', 'records'} for t in row['tasks']):
                        raise ValueError(f'Unsupported task survived scope filter: {row["id"]}')
                    elif row['id'] in seen:
                        previous = seen[row['id']]
                        if (previous['text_sha256'] != hashlib.sha256(row['text'].encode()).hexdigest()
                                or previous['tasks'] != {t['id'] for t in row['tasks']}
                                or previous['language'] != row['language']):
                            raise ValueError(f'Conflicting identity/text/task inventory: {row["id"]}')
                        changed = previous['fingerprint'] != fingerprint(row)
                        event.update(action='duplicate', retained_from=previous['source'],
                                     differing_annotation_version=changed)
                        counts['duplicate_ids'] += 1
                        counts['different_versions'] += changed
                    else:
                        seen[row['id']] = {
                            'source': str(path), 'fingerprint': fingerprint(row),
                            'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                            'tasks': {t['id'] for t in row['tasks']}, 'language': row['language'],
                        }
                        out.write(dumps(row) + '\n')
                        event.update(action='included', language=row['language'],
                                     retained_tasks=len(original_tasks) - len(removed))
                        counts['included'] += 1
                        languages[row['language']] += 1
                        sources[row['source']] += 1
                        families[row['family']] += 1
                        origins[row['text_origin'] + '/' + row['label_origin']] += 1
                audit.write(dumps(event) + '\n')
            accounted = counts['included'] + counts['duplicate_ids'] + sum(
                v for k, v in counts.items() if k.startswith('excluded:'))
            if accounted != counts['input_rows']:
                raise ValueError('Unaccounted training source rows')
            inputs[str(path)] = {'sha256': spec['sha256'], 'counts': dict(counts)}
            totals.update(counts)
            print(dumps({'source': str(path), 'counts': dict(counts)}), flush=True)
    dev = Path(config['dev'])
    if str(dev) not in {s['path'] for s in config['heldout']}:
        raise ValueError('Dev must be in the heldout registry')
    from .source_policy import excluded_source
    excluded_dev = 0
    with dev.open() as source, (root / 'dev.episodes.jsonl').open('w') as target:
        for line in source:
            if excluded_source(loads(line)):
                excluded_dev += 1
            else:
                target.write(line)
    manifest = {
        'status': 'assembled', 'training_ready': False, 'native_training_ready': False,
        'selection': 'all eligible de/en training IDs, one exposure each; no sampling or quotas',
        'annotation_precedence': 'Earlier configured source wins for an existing ID; differences audited.',
        'semantic_audit': 'Existing corrections retained; no claim of a new exhaustive semantic review.',
        'counts': dict(totals), 'unique_train_episodes': len(seen),
        'languages': dict(languages), 'sources': dict(sources),
        'families': dict(families), 'origins': dict(origins),
        'removed_tasks': dict(task_changes), 'inputs': inputs,
        'heldout': config['heldout'], 'selection_config_sha256': sha(config_path),
        'excluded_dev_source_rows': excluded_dev,
        'files': {p.name: sha(p) for p in root.iterdir() if p.is_file()},
    }
    (root / 'manifest.json').write_text(dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(dumps(assemble(args.config, args.output), indent=2))
