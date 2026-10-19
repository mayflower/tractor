"""Fact-grounded extraction coverage pilot, with split-owned rendering families."""

import argparse
import hashlib
import itertools
import multiprocessing
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .episodes import Episode, audit_episodes, schema_variant, text_key
from .jsonio import dumps, loads
from .public_extraction import ProtectedIndex, check_row, checksum, init_validator

SPLITS = ('train', 'dev', 'calibration', 'test')
FAMILIES = ('records', 'relations', 'events', 'document_fields', 'role_contrasts', 'cross_section')
# Translation equivalents live in one domain, not in separate source collections.
DOMAINS = [
    ('invoice', ('Position', 'Artikel', 'Menge', 'Einzelpreis'), ('Line', 'Item', 'Quantity', 'Unit price'),
     ('Kabel', 'Adapter', 'Sensor', 'Filter'), ('cable', 'adapter', 'sensor', 'filter'), 'EUR'),
    ('purchase_order', ('Position', 'Material', 'Bestellmenge', 'Budget'),
     ('Line', 'Material', 'Ordered quantity', 'Budget'),
     ('Schraube', 'Dichtung', 'Ventil', 'Pumpe'), ('screw', 'seal', 'valve', 'pump'), 'EUR'),
    ('resume', ('Eintrag', 'Arbeitgeber', 'Wochenstunden', 'Beginn'),
     ('Entry', 'Employer', 'Weekly hours', 'Start'),
     ('Nordlicht', 'Bergwald', 'Seeblick', 'Feldweg'), ('Cedar', 'Willow', 'Maple', 'Birch'), 'date'),
    ('contract', ('Klausel', 'Leistung', 'Laufzeit', 'Vergütung'),
     ('Clause', 'Service', 'Duration', 'Fee'),
     ('Wartung', 'Beratung', 'Reinigung', 'Prüfung'), ('maintenance', 'consulting', 'cleaning', 'inspection'), 'EUR'),
    ('support_ticket', ('Ticket', 'Komponente', 'Priorität', 'Termin'),
     ('Ticket', 'Component', 'Priority', 'Due date'),
     ('Drucker', 'Router', 'Monitor', 'Scanner'), ('printer', 'router', 'monitor', 'scanner'), 'date'),
    ('log', ('Vorgang', 'Dienst', 'Versuche', 'Zeitpunkt'),
     ('Operation', 'Service', 'Attempts', 'Timestamp'),
     ('Archiv', 'Suche', 'Import', 'Export'), ('archive', 'search', 'import', 'export'), 'date'),
    ('meeting_minutes', ('Punkt', 'Aufgabe', 'Aufwand', 'Frist'),
     ('Point', 'Action', 'Effort', 'Deadline'),
     ('Abnahme', 'Migration', 'Inventur', 'Schulung'), ('acceptance', 'migration', 'inventory', 'training'), 'date'),
    ('shipping_form', ('Sendung', 'Ware', 'Pakete', 'Ankunft'),
     ('Shipment', 'Goods', 'Packages', 'Arrival'),
     ('Papier', 'Glas', 'Textilien', 'Werkzeug'), ('paper', 'glass', 'textiles', 'tools'), 'date'),
    # Entire domain schemas reserved for evaluation.
    ('insurance_form', ('Fall', 'Objekt', 'Anzahl', 'Schadenbetrag'),
     ('Claim', 'Object', 'Count', 'Claim amount'),
     ('Fenster', 'Dach', 'Zaun', 'Tür'), ('window', 'roof', 'fence', 'door'), 'EUR'),
    ('research_note', ('Probe', 'Material', 'Messungen', 'Termin'),
     ('Sample', 'Material', 'Measurements', 'Date'),
     ('Kupfer', 'Eisen', 'Zink', 'Silber'), ('copper', 'iron', 'zinc', 'silver'), 'date'),
]
RELATIONS = [
    ('employment', 'Person and their explicitly stated current employer',
     ['{head} arbeitet bei {tail}.', '{tail} beschäftigt {head}.',
      'Im Personalverzeichnis steht: {head}, beschäftigt durch {tail}.',
      'Auf Nachfrage bestätigt {head}: Mein Arbeitgeber ist {tail}.'],
     ['{head} works for {tail}.', '{tail} employs {head}.',
      'The staff directory lists {head} as an employee of {tail}.',
      'When asked, {head} confirmed: My employer is {tail}.']),
    ('supplier', 'Supplier and the organization it supplies to',
     ['{head} beliefert {tail}.', '{tail} bezieht Waren von {head}.',
      'Lieferant für {tail}: {head}.', 'Für {tail} übernimmt {head} die Warenversorgung.'],
     ['{head} supplies {tail}.', '{tail} receives goods from {head}.',
      'Supplier to {tail}: {head}.', 'For {tail}, goods are provided by {head}.']),
    ('ownership', 'Owner and the asset explicitly owned by that owner',
     ['{head} besitzt {tail}.', '{tail} gehört {head}.',
      'Eigentümer von {tail}: {head}.', 'Als Eigentum von {head} ist {tail} registriert.'],
     ['{head} owns {tail}.', '{tail} belongs to {head}.',
      'Owner of {tail}: {head}.', 'Registered as the property of {head}: {tail}.']),
    ('assignment', 'Responsible person and the ticket assigned to that person',
     ['{head} bearbeitet {tail}.', '{tail} ist {head} zugewiesen.',
      'Zuständig für {tail}: {head}.', 'Die Bearbeitung von {tail} übernimmt {head}.'],
     ['{head} handles {tail}.', '{tail} is assigned to {head}.',
      'Responsible for {tail}: {head}.', 'Handling of {tail} is undertaken by {head}.']),
]


class Document:
    """Track occurrences at rendering time; never rediscover a repeated value with find()."""

    def __init__(self):
        self.text = ''

    def add(self, value):
        self.text += value

    def value(self, value):
        start = len(self.text)
        self.add(value)
        return {'status': 'present', 'items': [{'start': start, 'end': len(self.text), 'text': value}]}

    def render(self, pattern, values):
        import re
        result = {}
        cursor = 0
        for match in re.finditer(r'\{(\w+)\}', pattern):
            self.add(pattern[cursor:match.start()])
            key = match[1]
            result[key] = self.value(values[key])
            cursor = match.end()
        self.add(pattern[cursor:])
        return result


def allocation(total, weights):
    if total % 100:
        raise ValueError('Counts must be multiples of 100 for exact quotas')
    if sum(weights.values()) != 100:
        raise ValueError('Weights must sum to 100')
    return [key for key, weight in weights.items() for _ in range(total * weight // 100)]


def expanded_contrast(doc, rng, de, split, index, variant):
    """Additional scalar roles for the scale pool; literal facts, never normalized values."""
    role = (index // 2) % 5
    a, b = rng.sample(range(10, 900), 2)
    date_a = f"{rng.randint(2020, 2035)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    date_b = f"{rng.randint(2020, 2035)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    cases = [
        ('Current confirmed appointment date, excluding tentative and superseded dates', date_a, date_b,
         'vorläufiger Termin', 'tentative date', 'bestätigter Termin', 'confirmed date'),
        ('Current delivery address, excluding the former address',
         f'Birkenweg {a}, Bonn' if de else f'{a} Birch Road, Bristol',
         f'Seestrasse {b}, Kiel' if de else f'{b} Lake Street, York',
         'frühere Lieferadresse', 'former delivery address', 'aktuelle Lieferadresse', 'current delivery address'),
        ('Explicitly reported shipped quantity including its unit, excluding the ordered quantity',
         f'{a} Stück' if de else f'{a} items', f'{b} Stück' if de else f'{b} items',
         'Bestellmenge', 'ordered quantity', 'gelieferte Menge', 'shipped quantity'),
        ('Explicitly reported payment amount including currency, excluding the invoice amount',
         f'{a} EUR', f'{b} EUR', 'Rechnungsbetrag', 'invoice amount', 'Zahlungsbetrag', 'payment amount'),
        ('Current contact email address, excluding the withdrawn former address',
         f'old{a}@example.org', f'new{b}@example.org', 'frühere E-Mail-Adresse', 'former email address',
         'aktuelle Kontaktadresse', 'current contact email'),
    ]
    description, old, new, old_de, old_en, new_de, new_en = cases[role]
    prior, current = (old_de, new_de) if de else (old_en, new_en)
    patterns = {'train': f'{prior}: {{old}}.\n',
                'dev': f'{"Im bisherigen Eintrag steht" if de else "The earlier entry states"}: {prior} {{old}}.\n',
                'calibration': f'{{old}} ({prior})\n',
                'test': f'{"Zum Vergleich" if de else "For comparison"} — {prior} = {{old}}\n'}
    doc.render(patterns[split], {'old': old})
    if variant == 1:
        doc.add(f'Keine Angabe für {current}. Der frühere Eintrag ist hierfür nicht gültig.\n' if de else
                f'No value is provided for {current}. The earlier entry is not valid for that field.\n')
        answer = {'status': 'absent'}
    else:
        doc.add(f'Verbindliche Angabe für {current}: ' if de else f'Confirmed value for {current}: ')
        answer = doc.value(new)
        doc.add('.\n')
    return {'value': description}, [{'value': answer}], ['explicit_absence' if variant else 'role_selection']


def make_episode(split, index, family, language, seed=731, variant=0):
    rng = random.Random(f'{seed}:{split}:{index}')
    split_id = SPLITS.index(split)
    de = language == 'de' or (language == 'mixed' and index % 2 == 0)
    domain = DOMAINS[rng.randrange(8 if split == 'train' else len(DOMAINS))]
    artifact, labels_de, labels_en, values_de, values_en, value_type = domain
    labels = labels_de if de else labels_en
    names = values_de if de else values_en
    scenario = f'coverage:{seed}:{split}:{index}'
    doc = Document()
    # Shared factual identity across minimal-pair variants; split assigned before rendering.
    doc.add(f'{"Referenz" if de else "Reference"}: {split_id + 1}{index:06d}\n')
    if language == 'mixed':
        doc.add('Please use the German entries below.\n' if de else 'Bitte die englischen Angaben unten verwenden.\n')
    rows = []
    difficulty = []
    fields = {'anchor': labels[0], 'item': labels[1], 'amount': labels[2], 'value': labels[3]}
    task = {'id': 'extract', 'kind': 'records', 'description': 'Extract the listed entries and keep their fields together',
            'fields': fields, 'anchor': 'anchor'}
    n = rng.randint(2, 4)
    if family in {'records', 'cross_section'}:
        pending = []
        for j in range(n):
            values = {'anchor': f'R{rng.randrange(100, 999)}-{j}', 'item': names[j],
                      'amount': str(rng.randint(1, 24)),
                      'value': (f'{rng.randint(25, 900)} EUR' if value_type == 'EUR'
                                else f'{rng.randint(2023, 2027)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}')}
            if family == 'cross_section':
                pattern = (f'{labels[0]}: {{anchor}}; {labels[1]}: {{item}}.\n' if split_id == 0 else
                           f'{{anchor}} — {labels[1]} {{item}}.\n' if split_id == 1 else
                           f'{labels[1]} {{item}} ({{anchor}}).\n' if split_id == 2 else
                           f'{{item}} / {labels[0]} {{anchor}}\n')
                row = doc.render(pattern, values)
                pending.append((row, values))
            else:
                keys = list(values)
                if split_id == 0:
                    rng.shuffle(keys)
                    pattern = '; '.join(fields[k] + ': {' + k + '}' for k in keys) + '.\n'
                elif split_id == 1:
                    pattern = f'{{anchor}} betrifft {{item}}; {labels[2]} {{amount}}, {labels[3]} {{value}}.\n' if de else (
                        f'{{anchor}} concerns {{item}}; {labels[2]} {{amount}}, {labels[3]} {{value}}.\n')
                elif split_id == 2:
                    pattern = ' | '.join(fields[k] + ' = {' + k + '}' for k in reversed(keys)) + '\n'
                else:
                    pattern = f'{{item}} ({{anchor}})\n  {labels[3]}: {{value}}\n  {labels[2]}: {{amount}}\n'
                omitted = None
                if index >= 10000 and (index + j) % 4 == 0:
                    omitted = ('amount', 'value')[((index + j) // 4) % 2]
                    pattern = '; '.join(fields[k] + ': {' + k + '}' for k in keys if k != omitted) + '.\n'
                    difficulty.append('missing_record_field')
                rendered = doc.render(pattern, values)
                if omitted:
                    rendered[omitted] = {'status': 'absent'}
                rows.append(rendered)
        if pending:
            doc.add('\nDetails nach Referenz\n' if de else '\nDetails by reference\n')
            rng.shuffle(pending)
            for row, values in pending:
                doc.add(values['anchor'] + ': ')
                row.update(doc.render(f'{labels[3]} {{value}}; {labels[2]} {{amount}}.\n', values))
                rows.append(row)
            difficulty.append('cross_section_reference_join_within_window')
    elif family == 'relations':
        rel, description, de_patterns, en_patterns = RELATIONS[index % len(RELATIONS)]
        if index >= 10000:
            artifact = rel + '_note'
        task = {'id': rel, 'kind': 'relations', 'description': description,
                'fields': {'head': 'Source entity in the described relationship', 'tail': 'Target entity'}}
        for _j in range(n):
            head = f'{rng.choice(["Anna", "Mika", "Robin", "Jules", "Kim"])} {rng.choice(["Weber", "Klein", "Roth", "Bauer"])}'
            tail = (f'Ticket {rng.randrange(1000, 9999)}' if rel == 'assignment' else
                    f'Anlage {rng.randrange(1000, 9999)}' if rel == 'ownership' else
                    f'{rng.choice(["Cedar", "Delta", "Nordlicht", "Lyra"])} {rng.randrange(10, 99)}')
            rows.append(doc.render((de_patterns if de else en_patterns)[split_id] + '\n',
                                   {'head': head, 'tail': tail}))
        doc.add('Die Nachbarfirma hat keine Verbindung zu diesen Vorgängen.\n' if de else
                'The neighboring company has no connection with these matters.\n')
    elif family == 'events':
        event = ['delivery', 'payment', 'cancellation', 'appointment', 'incident'][index % 5]
        fields = {'event': 'Event reference', 'actor': 'Initiator or responsible party',
                  'recipient': 'Recipient or affected party', 'date': 'Date of this event',
                  'object': 'Object of this event'}
        task.update(fields=fields, anchor='event', description=f'Extract {event} events with their participants')
        verbs = {'delivery': ('liefert an', 'delivers to'), 'payment': ('zahlt an', 'pays'),
                 'cancellation': ('kündigt gegenüber', 'cancels with'),
                 'appointment': ('trifft', 'meets'), 'incident': ('meldet an', 'reports to')}
        for j in range(n):
            values = {'event': f'E{rng.randrange(100, 999)}-{j}',
                      'actor': f'Alex {rng.choice(["Roth", "Weber", "Klein"])}',
                      'recipient': f'{rng.choice(["Cedar", "Lyra", "Delta"])} {j + 1}',
                      'date': f'2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}',
                      'object': f'{"Vorgang" if de else "case"} {rng.randrange(1000, 9999)}'}
            verb = verbs[event][0 if de else 1]
            patterns = [f'{{event}}: {{actor}} {verb} {{recipient}}; {{object}}; {{date}}.\n',
                        f'{{date}} — {{event}}\n{{actor}} {verb} {{recipient}}. {"Bezug" if de else "Regarding"}: {{object}}.\n',
                        f'{{object}} / {{event}}: {{date}}, {{actor}} {verb} {{recipient}}.\n',
                        f'{{actor}} {verb} {{recipient}}\n{{event}} — {{object}} — {{date}}\n']
            rows.append(doc.render(patterns[split_id], values))
        artifact = {'delivery': 'shipping_notice', 'payment': 'payment_notice', 'cancellation': 'contract_email',
                    'appointment': 'calendar_message', 'incident': 'incident_report'}[event]
    else:
        task = {'id': 'selection', 'kind': 'entities', 'description': 'Extract only the requested document fields',
                'fields': {}}
        if family == 'document_fields':
            if index >= 10000:
                artifact = 'contact_note'
            fields = {'contact': 'Named contact persons', 'reference': 'External customer reference',
                      'email': 'Contact email addresses explicitly given in the document'}
            task['fields'] = fields
            contacts = []
            for _j in range(n):
                name = f'{rng.choice(["Anna", "Robin", "Jules", "Kim"])} {rng.choice(["Weber", "Roth", "Bauer"])}'
                patterns = ['Kontakt: {name}.\n' if de else 'Contact: {name}.\n',
                            'Bitte wenden Sie sich an {name}.\n' if de else 'Please contact {name}.\n',
                            'Ansprechperson = {name}\n' if de else 'Contact person = {name}\n',
                            'Rückfragen beantwortet {name}.\n' if de else 'Questions are answered by {name}.\n']
                contacts.extend(doc.render(patterns[split_id], {'name': name})['name']['items'])
            row = {'contact': {'status': 'present', 'items': contacts}}
            doc.add('Kundenreferenz: ' if de else 'Customer reference: ')
            row['reference'] = doc.value(f'K{rng.randrange(10000, 99999)}')
            doc.add('\n')
            if index % 2:
                doc.add('E-Mail: ')
                row['email'] = doc.value(f'contact{index}@example.org')
                doc.add('\n')
            else:
                row['email'] = {'status': 'absent'}
                difficulty.append('missing_optional_field')
            rows = [row]
        else:
            if index >= 10000:
                fields, rows, flags = expanded_contrast(doc, rng, de, split, index, variant)
                task['fields'] = fields
                difficulty.extend(flags)
            else:
                # Explicit state transition, not a guessed policy for unresolved contradictions.
                fields = {'value': 'The explicitly confirmed current appointment date; exclude superseded or withdrawn dates'}
                task['fields'] = fields
                first, second = rng.sample(range(1, 28), 2)
                old, new = f'2026-11-{first:02d}', f'2026-11-{second:02d}'
                patterns = [('Vorschlag: {old}.\n', 'Proposal: {old}.\n'),
                            ('Zuerst war {old} vorgesehen.\n', 'Initially {old} was planned.\n'),
                            ('Vorläufiger Termin = {old}\n', 'Tentative date = {old}\n'),
                            ('Planungsstand vom Montag: {old}.\n', 'Monday planning status: {old}.\n')]
                doc.render(patterns[split_id][0 if de else 1], {'old': old})
                absent = variant == 1
                if absent:
                    doc.add('Der Vorschlag wurde zurückgezogen. Es gibt keinen bestätigten Termin.\n' if de else
                            'The proposal was withdrawn. There is no confirmed date.\n')
                    answer = {'status': 'absent'}
                    difficulty.append('explicit_absence')
                else:
                    doc.add('Der Vorschlag ist ersetzt. Verbindlich bestätigt: ' if de else
                            'The proposal is superseded. Confirmed date: ')
                    answer = doc.value(new)
                    doc.add('.\n')
                    difficulty.append('correction')
                rows = [{'value': answer}]
            artifact = 'email_thread'
    # An irrelevant footer is a distractor, not an instruction to the extractor.
    if index % 3 == 0:
        doc.add('\nArchivvermerk: Entwürfe sind nicht verbindlich.\n' if de else
                '\nArchive note: drafts are not binding.\n')
        difficulty.append('irrelevant_footer')
    episode = Episode.model_validate({
        'id': f'{scenario}:{variant}', 'source_group': scenario, 'derivation_group': scenario,
        'source': 'tractor_coverage_facts', 'original_split': split, 'split': split,
        'family': family, 'language': language, 'artifact': artifact,
        'annotation': 'synthetic_gold', 'text_origin': 'synthetic', 'label_origin': 'deterministic',
        'text': doc.text, 'tasks': [task], 'gold': {task['id']: {'complete': True, 'records': rows}},
    })
    # Rename fields without changing meaning; relation head/tail are fixed by the native contract.
    if index % 2:
        episode = schema_variant(episode, seed + index)
    metadata = {'id': episode.id, 'scenario': scenario, 'split': split, 'family': family,
                'language': language, 'domain': artifact, 'difficulties': difficulty,
                'render_family': f'{family}:{split}', 'heldout_domain': domain[0] in {'insurance_form', 'research_note'}
                and family in {'records', 'cross_section'}, 'minimal_pair': family == 'role_contrasts'}
    return episode, metadata


def generate_split(split, total, config, offset=0):
    # Allocate scenarios in pairs for contrasts, so both states stay in the same split.
    families = allocation(total, config['family_weights'])
    pair_languages = allocation(total, config['language_weights'])[::2]
    random.Random(f"{config['seed']}:{split}:languages").shuffle(pair_languages)
    languages = [language for language in pair_languages for _ in range(2)]
    index = 0
    while index < total:
        family = families[index]
        variants = 2 if family == 'role_contrasts' and index + 1 < total and families[index + 1] == family else 1
        for variant in range(variants):
            yield make_episode(split, index + offset, family, languages[index], config['seed'], variant)
        index += variants


def training_blocks(config, blocks, protected=None, replacements=None):
    """Keep the frozen pilot, reject additional bodies differing only by reference ID."""
    seen = set()
    retry_index = 1000000
    for block in range(blocks):
        rows = generate_split('train', config['pilot_train'], config, offset=block * config['pilot_train'])
        for _, group in itertools.groupby(rows, key=lambda row: row[0].source_group):
            group = list(group)
            for _attempt in range(10000):
                bodies = [' '.join(e.text.split('\n', 1)[1].split()).casefold() for e, _ in group]
                overlaps = [(e.id, reason) for e, _ in group
                            if protected and (reason := protected.overlaps(e))]
                if overlaps and block == 0:
                    raise ValueError('Frozen pilot overlaps protected evaluation')
                if overlaps and replacements is not None:
                    replacements.extend({'id': ident, 'reason': reason} for ident, reason in overlaps)
                if not overlaps and (block == 0 or not any(body in seen for body in bodies)):
                    seen.update(bodies)
                    yield from group
                    break
                episode = group[0][0]
                group = [make_episode('train', retry_index, episode.family, episode.language,
                                      config['seed'], variant) for variant in range(len(group))]
                retry_index += 1
            else:
                raise ValueError('Unique-body space exhausted; broaden generator before scaling')


def require_scale_gate(report, config):
    import math

    if report.get('status') != 'complete' or not report.get('scale_allowed'):
        raise ValueError('Matched Dev pilot has not passed the scale gate')
    selected = report.get('selected')
    if selected not in {'S20', 'S40'}:
        raise ValueError('Scale requires a selected synthetic arm')
    base, candidate = report['scores']['S0'], report['scores'][selected]
    keys = ('synthetic_macro_f1', 'public_macro_f1', 'product_exact_request')
    if any(not math.isfinite(float(row[key])) for row in (base, candidate) for key in keys):
        raise ValueError('Scale gate requires finite metrics')
    gates = config['scale_requires']
    if (candidate['synthetic_macro_f1'] < base['synthetic_macro_f1'] + gates['synthetic_macro_f1_improvement']
            or candidate['public_macro_f1'] < base['public_macro_f1'] - gates['public_macro_f1_max_drop']
            or candidate['product_exact_request'] < gates['product_exact_request_floor']):
        raise ValueError('Scale metrics do not satisfy the configured gates')


def build(config_path, output, model, workers=4, scale_report=None):
    config = loads(Path(config_path).read_text())
    if config.get('retired'):
        raise ValueError('Schematic generator retired; use document_build')
    blocks = 1
    if config['pilot_train'] > 10000:
        raise ValueError('Pilot generation is capped at 10000; use the scale gate')
    if scale_report:
        gate = loads(Path(scale_report).read_text())
        require_scale_gate(gate, config)
        if config['pilot_train'] != 10000 or config['scale_train'] != 100000:
            raise ValueError('Scale profile requires ten fixed 10000-example blocks')
        blocks = 10
        pilot = Path(gate['pilot_data'])
        manifest_bytes = (pilot / 'manifest.json').read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != gate['pilot_manifest_sha256']:
            raise ValueError('Scale gate refers to a different pilot dataset')
        pilot_config = loads(manifest_bytes)['config']
        for key in ('seed', 'family_weights', 'language_weights', 'evaluation_per_split'):
            if config[key] != pilot_config[key]:
                raise ValueError('Scale configuration changed after the pilot')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    counts, tokens, rejected = Counter(), Counter(), Counter()
    all_episodes = []
    metadata = []
    replacements = []
    protected = None
    if scale_report:
        from .generalist_data import read_episodes

        protected = ProtectedIndex()
        for path in config.get('protected_episodes', []):
            for episode in read_episodes(path):
                protected.add(episode)
        for split in SPLITS:
            if split != 'train':
                for episode, _ in generate_split(split, config['evaluation_per_split'], config):
                    protected.add(episode)
    for split in SPLITS:
        total = config['pilot_train'] if split == 'train' else config['evaluation_per_split']
        rows = (training_blocks(config, blocks, protected, replacements) if split == 'train'
                else generate_split(split, total, config))
        for episode, meta in rows:
            all_episodes.append(episode)
            metadata.append(meta)
    audit = audit_episodes(all_episodes)
    if scale_report:
        for split in SPLITS:
            rows = [e for e in all_episodes if e.split == split]
            if split == 'train':
                rows = rows[:config['pilot_train']]
            original = (pilot / f'{split}.episodes.jsonl').read_bytes()
            if (''.join(dumps(e.model_dump()) + '\n' for e in rows)).encode() != original:
                raise ValueError('Scale must preserve every pilot/evaluation example exactly')
        root.joinpath('replaced-overlaps.json').write_text(dumps(replacements, indent=2))
        overlaps = [{'id': e.id, 'reason': reason} for e in all_episodes
                    if e.split == 'train' and (reason := protected.overlaps(e))]
        root.joinpath('scale-overlaps.json').write_text(dumps(overlaps, indent=2))
        if overlaps:
            raise ValueError('Expanded training overlaps reserved evaluation; do not train')
    # Unique documents across splits; minimal pairs may share fact identity, not split.
    unique = Counter(e.split for e in {text_key(e.text): e for e in all_episodes}.values())
    root.joinpath('metadata.jsonl').write_text(''.join(dumps(m) + '\n' for m in metadata))
    handles = {(split, kind): root.joinpath(f'{split}.{kind}.jsonl').open('w')
               for split in SPLITS for kind in ('episodes', 'native')}
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_validator, initargs=(model,)) as pool:
            lines = (dumps(e.model_dump()) for e in all_episodes)
            with root.joinpath('rejections.jsonl').open('w') as reject:
                for line, native, length, error in pool.map(check_row, lines, chunksize=32):
                    e = loads(line)
                    if error:
                        rejected[error] += 1
                        reject.write(dumps({'id': e['id'], 'error': error}) + '\n')
                        continue
                    split = e['split']
                    handles[split, 'episodes'].write(line + '\n')
                    handles[split, 'native'].write(native + '\n')
                    counts[f'{split}:{e["family"]}'] += 1
                    tokens[split] += length
    finally:
        for handle in handles.values():
            handle.close()
    report = {'status': 'validated' if not rejected else 'rejected_rows_require_review',
              'config': config, 'audit': audit, 'unique_texts_by_split': dict(unique),
              'unique_train_bodies_without_reference': len({
                  ' '.join(e.text.split('\n', 1)[1].split()).casefold() for e in all_episodes if e.split == 'train'}),
              'accepted': dict(counts), 'joint_tokens': dict(tokens),
              'languages': dict(Counter(f'{e.split}:{e.language}' for e in all_episodes)),
              'difficulty_examples': dict(Counter(m['split'] for m in metadata if m['difficulties'])),
              'render_families': sorted({m['render_family'] for m in metadata}), 'rejected': dict(rejected),
              'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'qualification': config['qualification'],
              'scale_status': 'passed_matched_dev_pilot' if scale_report else 'blocked_pending_matched_dev_pilot',
              'scale_report_sha256': hashlib.sha256(Path(scale_report).read_bytes()).hexdigest() if scale_report else None,
              'external_cost_usd': 0,
              'native_validation_model_sha256': checksum(Path(model) / 'model.safetensors'),
              'protected_file_hashes': {path: checksum(path) for path in config.get('protected_episodes', [])},
              'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.jsonl')}}
    root.joinpath('manifest.json').write_text(dumps(report, indent=2))
    print(dumps(report, indent=2), flush=True)
    if rejected:
        raise ValueError('Native validation rejected generated targets; inspect rejections before use')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/synthetic-coverage.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='models/tractor')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--scale-report')
    args = parser.parse_args()
    build(args.config, args.output, args.model, args.workers, args.scale_report)


if __name__ == '__main__':
    main()
