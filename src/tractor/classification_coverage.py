"""Versioned public and synthetic classification coverage, with native target checks."""

import argparse
import multiprocessing
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .episodes import Episode, audit_episodes, text_key
from .generalist_data import read_episodes
from .jsonio import dumps, loads
from .public_extraction import ProtectedIndex, check_row, checksum, init_validator

SPLITS = ('train', 'dev', 'calibration', 'test')
DOMAINS = {
    'social': 'Social network messages and notifications', 'transport': 'Travel and transport',
    'calendar': 'Calendar appointments', 'play': 'Play music, radio, games or other media',
    'news': 'News reports', 'datetime': 'Time and date', 'recommendation': 'Recommendations for places or events',
    'email': 'Email and contacts', 'iot': 'Control smart home devices', 'general': 'Greetings, jokes and general chat',
    'audio': 'Audio volume control', 'lists': 'Manage lists', 'qa': 'General factual questions',
    'cooking': 'Cooking and recipes', 'takeaway': 'Takeaway food orders',
    'music': 'Music information, settings and preferences', 'alarm': 'Alarm clock', 'weather': 'Weather forecasts',
}
INTENTS = {
    'datetime_query': 'Ask the time or date', 'iot_hue_lightchange': 'Change light color',
    'transport_ticket': 'Book transport tickets', 'takeaway_query': 'Ask about takeaway food',
    'qa_stock': 'Ask stock market information', 'general_greet': 'Greet the assistant',
    'recommendation_events': 'Recommend events', 'music_dislikeness': 'Express dislike for music',
    'iot_wemo_off': 'Switch a smart plug off', 'cooking_recipe': 'Request a recipe',
    'qa_currency': 'Ask about currency conversion', 'transport_traffic': 'Ask traffic conditions',
    'general_quirky': 'General conversational request or remark', 'weather_query': 'Ask about weather',
    'audio_volume_up': 'Increase audio volume', 'email_addcontact': 'Add an email contact',
    'takeaway_order': 'Order takeaway food', 'email_querycontact': 'Ask about an email contact',
    'iot_hue_lightup': 'Increase light brightness', 'recommendation_locations': 'Recommend places',
    'play_audiobook': 'Play an audiobook', 'lists_createoradd': 'Create a list or add an item',
    'news_query': 'Ask for news', 'alarm_query': 'Ask about existing alarms',
    'iot_wemo_on': 'Switch a smart plug on', 'general_joke': 'Request a joke',
    'qa_definition': 'Ask for a definition', 'social_query': 'Read social network updates',
    'music_settings': 'Change music playback settings', 'audio_volume_other': 'Other audio volume adjustment',
    'calendar_remove': 'Remove a calendar event', 'iot_hue_lightdim': 'Dim lights',
    'calendar_query': 'Ask about calendar events', 'email_sendemail': 'Send an email',
    'iot_cleaning': 'Start a cleaning device', 'audio_volume_down': 'Decrease audio volume',
    'play_radio': 'Play radio', 'cooking_query': 'Ask cooking advice',
    'datetime_convert': 'Convert between time zones', 'qa_maths': 'Ask a mathematical question',
    'iot_hue_lightoff': 'Turn lights off', 'iot_hue_lighton': 'Turn lights on',
    'transport_query': 'Ask transport information', 'music_likeness': 'Express liking for music',
    'email_query': 'Ask about email messages', 'play_music': 'Play music',
    'audio_volume_mute': 'Mute audio', 'social_post': 'Post on a social network',
    'alarm_set': 'Set an alarm', 'qa_factoid': 'Ask a factual knowledge question',
    'calendar_set': 'Create a calendar event', 'play_game': 'Play a game',
    'alarm_remove': 'Remove an alarm', 'lists_remove': 'Remove a list or list item',
    'transport_taxi': 'Request a taxi', 'recommendation_movies': 'Recommend movies',
    'iot_coffee': 'Control a coffee machine', 'music_query': 'Ask about music',
    'play_podcasts': 'Play a podcast', 'lists_query': 'Ask about list contents',
}


def class_task(task_id, description, labels, label, rng, opaque=False):
    keys = list(labels)
    rng.shuffle(keys)
    mapping = {key: f'choice_{i}' if opaque else key for i, key in enumerate(keys)}
    task = {'id': task_id, 'kind': 'classification', 'description': description,
            'labels': {mapping[key]: labels[key] for key in keys}}
    gold = {'complete': True, 'records': [{'label': {'status': 'present', 'label': mapping[label]}}]}
    return task, gold


def synthetic_episode(config, axis, split, index):
    spec, seed = config['axes'][axis], config['seed']
    # Related contrasting cases share identity, language, names and reference, but have different labels.
    group = f'classification:{seed}:{split}:{axis}:{index // 2}'
    rng = random.Random(group)
    language = ('de', 'en')[(index // 4) % 2]
    label = spec['classes'][index % len(spec['classes'])]
    name = rng.choice(['Alex Weber', 'Sam Fischer', 'Robin Meyer', 'Kim Bauer', 'Jules Hart', 'Mika Roth'])
    ref = f'{rng.choice(["AX", "BR", "CT", "DV"])}-{rng.randrange(100000, 999999)}'
    site = rng.choice(['Berlin', 'Hamburg', 'Bremen', 'Leipzig', 'Hannover', 'Dresden'])
    context = (f'Kontakt: {name}. Referenz: {ref}. Standort: {site}.' if language == 'de'
               else f'Contact: {name}. Reference: {ref}. Location: {site}.')
    evidence = label['train' if split == 'train' else 'heldout'][language]
    wrappers = {
        'train': ('Aktuelle Nachricht: {e}\n{c}', 'Current message: {e}\n{c}'),
        'dev': ('{c}\nWeitergeleitete Mitteilung:\n{e}', '{c}\nForwarded note:\n{e}'),
        'calibration': ('Notiz aus dem Posteingang\n{e}\nZuordnung: {c}', 'Inbox note\n{e}\nAssigned details: {c}'),
        'test': ('{c}\nWortlaut der Rückmeldung: „{e}“', '{c}\nResponse as received: “{e}”'),
    }
    text = wrappers[split][language == 'en'].format(e=evidence, c=context)
    task, gold = class_task('category', spec['description'],
                            {row['id']: row['description'] for row in spec['classes']}, label['id'],
                            random.Random(group + str(index)), opaque=(index // 8) % 2 == 1)
    tasks, targets = [task], {'category': gold}
    mixed = (index // 2) % 3 == 0
    if mixed:
        tasks.append({'id': 'details', 'kind': 'entities', 'description': 'Contact details from the message',
                      'fields': {'contact': 'Named contact person', 'reference': 'Reference identifier',
                                 'email': 'Explicit email address of the contact'}})
        record = {'email': {'status': 'absent'}}
        for key, value in (('contact', name), ('reference', ref)):
            start = text.index(value)
            record[key] = {'status': 'present', 'items': [{'start': start, 'end': start + len(value), 'text': value}]}
        targets['details'] = {'complete': True, 'records': [record]}
    return Episode.model_validate({
        'id': f'{group}:{index % 2}', 'source_group': group, 'derivation_group': group,
        'source': 'tractor_classification_facts', 'original_split': split, 'split': split,
        'family': f'classification_{axis}', 'language': language, 'artifact': 'message',
        'annotation': 'synthetic_gold', 'text_origin': 'synthetic', 'label_origin': 'deterministic',
        'text': text, 'tasks': tasks, 'gold': targets,
    })


def massive_episode(row, scenarios, intents, seed):
    group = f'massive:utterance:{row["id"]}'
    rng = random.Random(f'{seed}:{group}')
    original = row['partition']
    if original not in {'train', 'dev', 'test'}:
        raise ValueError('Unknown MASSIVE partition')
    split = original if original != 'test' else ('calibration' if rng.random() < .5 else 'test')
    intent, domain = intents[row['intent']], scenarios[row['scenario']]
    matching = rng.random() < .5
    alternatives = [key for key in intents if key != intent and key.split('_')[0] == intent.split('_')[0]]
    if not alternatives or rng.random() < .5:
        alternatives = [key for key in intents if key != intent]
    candidate = intent if matching else rng.choice(alternatives)
    t1, g1 = class_task('domain', 'Which area does this user request concern?', DOMAINS, domain,
                        rng, opaque=int(row['id']) % 2 == 1)
    t2, g2 = class_task('intent_match', f'Does the user request this intent: {INTENTS[candidate]}?',
                        {'yes': 'The requested intent matches', 'no': 'The requested intent does not match'},
                        'yes' if matching else 'no', rng)
    return Episode.model_validate({
        'id': f'{group}:{row["locale"]}', 'source_group': group, 'derivation_group': group,
        'source': 'massive', 'original_split': original, 'split': split, 'family': 'classification_intent',
        'language': row['locale'].split('-')[0], 'artifact': 'assistant_request', 'annotation': 'human',
        'text_origin': 'real', 'label_origin': 'human', 'text': row['utt'],
        'tasks': [t1, t2], 'gold': {'domain': g1, 'intent_match': g2},
    })


def public_annotation_conflicts(rows):
    texts = defaultdict(list)
    for row in rows:
        texts[(row['locale'], text_key(row['utt']))].append(row)
    return {f'massive:utterance:{row["id"]}' for group in texts.values()
            if len({(r['scenario'], r['intent']) for r in group}) > 1 for row in group}


def public_rows(config, conflicts=None):
    import pyarrow.parquet as pq

    root = Path(config['source']['root'])
    raw, converted = [], []
    for locale in config['source']['locales']:
        for partition in ('train', 'validation', 'test'):
            table = pq.read_table(root / locale / partition / '0000.parquet')
            features = loads(table.schema.metadata[b'huggingface'])['info']['features']
            scenarios, intents = features['scenario']['names'], features['intent']['names']
            if set(scenarios) != set(DOMAINS) or set(intents) != set(INTENTS):
                raise ValueError('Unexpected source label ontology')
            for row in table.to_pylist():
                raw.append(row)
                converted.append(massive_episode(row, scenarios, intents, config['seed']))
    bad = public_annotation_conflicts(raw)
    if conflicts is not None:
        conflicts.update({group: 'conflicting_public_annotation' for group in bad})
    yield from (e for e in converted if e.source_group not in bad)


def split_conflicts(episodes):
    """Drop whole parallel groups if any normalized text occurs across reserved splits."""
    texts = defaultdict(list)
    for e in episodes:
        texts[text_key(e.text)].append(e)
    return {e.source_group for rows in texts.values() if len({r.split for r in rows}) > 1 for e in rows}


def build(config_path, output, model='models/tractor', workers=4):
    config = loads(Path(config_path).read_text())
    if config.get('retired'):
        raise ValueError('Classification is outside the extraction scope')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    from .extraction_scope import in_scope

    if config.get('source') or any(not in_scope({'family': f'classification_{axis}', 'source': 'synthetic'})
                                   for axis in config['axes']):
        raise ValueError('Out-of-scope classification configuration')
    source_files = {}
    synthetic = [synthetic_episode(config, axis, split, i) for split in SPLITS for axis in config['axes']
                 for i in range(config['train_per_axis'] if split == 'train' else config['evaluation_per_axis'])]
    rejected_groups = {}
    public = []  # Public assistant-intent classification is outside the objective.
    conflicts = split_conflicts(public)
    rejected_groups.update({group: 'cross_split_normalized_text' for group in conflicts})
    public = [e for e in public if e.source_group not in conflicts]
    existing = ProtectedIndex()
    for path in config['protected_episodes']:
        for e in read_episodes(path):
            existing.add(e)
    # New heldouts must also avoid old training, without modifying frozen old files.
    old_train = ProtectedIndex()
    for e in read_episodes('data/generalist-extraction-combined/train.episodes.jsonl'):
        old_train.add(e)
    for e in public + synthetic:
        reason = existing.overlaps(e) or (old_train.overlaps(e) if e.split != 'train' else None)
        if reason:
            rejected_groups[e.source_group] = reason
    candidates = [e for e in public + synthetic if e.source_group not in rejected_groups]
    reserved = ProtectedIndex()
    for e in candidates:
        if e.split != 'train':
            reserved.add(e)
    for e in candidates:
        if e.split == 'train' and (reason := reserved.overlaps(e)):
            rejected_groups[e.source_group] = reason
    candidates = [e for e in candidates if e.source_group not in rejected_groups]
    audit_episodes(candidates)
    print(dumps({'candidates': len(candidates), 'excluded_groups': len(rejected_groups)}), flush=True)
    validated, native_rejections = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_validator, initargs=(model,)) as pool:
        for line, native, length, error in pool.map(check_row, (dumps(e.model_dump()) for e in candidates), chunksize=32):
            e = loads(line)
            if error:
                native_rejections.append({'id': e['id'], 'error': error})
                rejected_groups[e['source_group']] = error
            else:
                validated.append((e, line, native, length))
    counts, families, origins, kinds, languages, labels = (Counter() for _ in range(6))
    for split in SPLITS:
        with (root / f'{split}.episodes.jsonl').open('w') as ef, (root / f'{split}.native.jsonl').open('w') as nf:
            for e, line, native, _length in validated:
                if e['split'] != split or e['source_group'] in rejected_groups:
                    continue
                ef.write(line + '\n')
                nf.write(native + '\n')
                counts[split] += 1
                families[f'{split}:{e["family"]}'] += 1
                origins[f'{split}:{e["text_origin"]}/{e["label_origin"]}'] += 1
                languages[f'{split}:{e["language"]}'] += 1
                for task in e['tasks']:
                    kinds[f'{split}:{task["kind"]}'] += 1
                    if task['kind'] == 'classification':
                        value = e['gold'][task['id']]['records'][0]['label']['label']
                        labels[f'{split}:{e["family"]}:{task["id"]}:{task["labels"][value]}'] += 1
    (root / 'excluded-groups.json').write_text(dumps(rejected_groups, indent=2))
    (root / 'native-rejections.json').write_text(dumps(native_rejections, indent=2))
    accepted = [Episode.model_validate(e) for e, _, _, _ in validated if e['source_group'] not in rejected_groups]
    report = {'status': 'validated', 'config': config, 'counts': dict(counts), 'families': dict(families),
              'origins': dict(origins), 'languages': dict(languages), 'task_kinds': dict(kinds), 'labels': dict(labels),
              'audit': audit_episodes(accepted), 'excluded_groups': len(rejected_groups),
              'native_rejections': len(native_rejections), 'source_files': source_files,
              'generator_sha256': checksum(__file__), 'validation_source_sha256': checksum(Path(__file__).with_name('generalist_data.py')),
              'model_sha256': checksum(Path(model) / 'model.safetensors'),
              'protected_file_hashes': {p: checksum(p) for p in config['protected_episodes']},
              'old_training_sha256': checksum('data/generalist-extraction-combined/train.episodes.jsonl'),
              'qualification': 'Only approval status, document kind and request type classification. '
              'Public classification is separate from the extraction-only minimum. Parallel utterance IDs stay grouped. '
              'Synthetic templates and shared heldout evidence wording do not prove unseen-domain generalization. '
              'Single-label tasks only; mixed examples combine classification with span extraction. No model quality claim.',
              'files': {p.name: checksum(p) for p in root.glob('*.jsonl')}}
    if not all(counts[s] for s in SPLITS) or any(not families[f'train:classification_{a}'] for a in config['axes']):
        raise ValueError('Classification coverage was emptied by filtering')
    (root / 'manifest.json').write_text(dumps(report, indent=2))
    print(dumps({'counts': report['counts'], 'origins': report['origins'], 'native_rejections': len(native_rejections)}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/classification-coverage.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='models/tractor')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    build(args.config, args.output, args.model, args.workers)
