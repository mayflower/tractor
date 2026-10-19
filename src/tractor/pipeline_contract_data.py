"""Versioned corpus field contracts. No prediction-dependent rules or gold-based selection."""
import re
from copy import deepcopy
from datetime import datetime
from decimal import Decimal

from .episodes import Episode

GB = {'address__post_town': 'city', 'address__postcode': 'postcode', 'address__street_line': 'street'}
SOURCE_LABELS = {
    'phone': ['Phone:', 'Telephone:', 'Phone Number:', ':'],
    'cuisine': ['Cuisine:', ':'],
    'publisher': ['Publisher:', ':'],
    'publication_date': ['Pub. Date:', 'Publication Date:', ':'],
}
DATE_FIELDS = {'date', 'birth_date', 'start', 'end', 'submitted', 'due', 'published', 'followup', 'report_date'}
NUMBER_FIELDS = {'employees', 'sample_size'}
MONEY_FIELDS = {'net', 'tax', 'gross', 'unit_price', 'line_total'}


def oracle(text, kind, *, date_order=None, allow_partial=False):
    """Independent annotation conversion with intentionally narrower accepted grammar.

    Never imports/calls a production normalizer. Failures retain literal annotation
    and are reported as unsupported contract coverage, not guessed or deleted.
    """
    if kind == 'date':
        months = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']
        literal = text.strip()
        for n, month in enumerate(months, 1):
            literal = re.sub(r'(?i)\b'+month+r'\b', f'{n:02d}', literal)
        for fmt in ('%Y-%m-%d', '%d. %m %Y', '%d %m %Y', '%B %d, %Y', '%B %d %Y', '%d %B %Y', '%b %d, %Y', '%b %d %Y', '%d %b %Y', '%m %d, %Y', '%m %d %Y'):
            try:
                return datetime.strptime(literal, fmt).date().isoformat()
            except ValueError:
                continue
        if date_order:
            for separator in ('.', '/', '-'):
                fmt = separator.join({'D': '%d', 'M': '%m', 'Y': '%Y'}[part] for part in date_order)
                try:
                    return datetime.strptime(text.strip(), fmt).date().isoformat()
                except ValueError:
                    continue
        if allow_partial:
            for fmt in ('%B %Y', '%b %Y', '%m %Y', '%Y-%m', '%Y'):
                try:
                    parsed = datetime.strptime(literal, fmt)
                    return parsed.strftime('%Y' if fmt == '%Y' else '%Y-%m')
                except ValueError:
                    continue
        raise ValueError('independent_date_annotation_unavailable')
    if kind == 'number':
        if not re.fullmatch(r'[+-]?\d+', text.strip()):
            raise ValueError('independent_number_annotation_unavailable')
        return str(int(text.strip()))
    if kind == 'money':
        # Authored invoice facts explicitly define EUR with two decimal places.
        m = re.fullmatch(r'([+-]?\d{1,3}(?:[.,]\d{3})*|[+-]?\d+)([.,])(\d{2}) EUR', text.strip())
        if not m:
            raise ValueError('independent_money_annotation_unavailable')
        group = ',' if m[2] == '.' else '.'
        if m[2] in m[1]:
            raise ValueError('invalid_money_annotation')
        val = Decimal(m[1].replace(group, '')+'.'+m[3])
        return format(val, 'f').rstrip('0').rstrip('.')
    raise ValueError('Unknown annotation conversion')


def adapt(episode):
    row = deepcopy(episode.model_dump())
    contracts = row['processing']
    audit = []
    for task in row['tasks']:
        rules = contracts.setdefault(task['id'], {})
        records = row['gold'][task['id']]['records']
        if row['source'] == 'swde':
            for key, labels in SOURCE_LABELS.items():
                if key not in task['fields']:
                    continue
                rules[key] = {'kind': 'literal', 'prefixes': labels}
                # Correct the original annotated cell boundary using only an
                # explicit source label. Never search for a preferred answer.
                for record in records:
                    a = record[key]
                    for spans in [a['items'], *a['answer_alternatives']]:
                        for span in spans:
                            original = dict(span)
                            leading = len(span['text']) - len(span['text'].lstrip())
                            literal = span['text'].lstrip()
                            for label in labels:
                                if literal.casefold().startswith(label.casefold()):
                                    offset = leading + len(label)
                                    rest = original['text'][offset:]
                                    offset += len(rest) - len(rest.lstrip())
                                    value = original['text'][offset:].rstrip()
                                    if value:
                                        span.update(start=original['start']+offset,
                                                    end=original['start']+offset+len(value), text=value)
                                        audit.append({'id': row['id'], 'task': task['id'], 'field': key,
                                                      'reason': 'source_label_boundary_repaired',
                                                      'before': original, 'after': dict(span)})
                                    break
        if row['source'] == 'swde' and row['source_group'].startswith('swde:movie:'):
            if 'title' in task['fields']:
                task['fields']['title'] = 'Complete displayed film title, including the release year in parentheses when it is printed as part of the title'
                rules['title'] = {'kind': 'literal', 'include_adjacent_year': True}
                # Source-boundary annotation repair, independent of predictions
                # and the production resolver. Preserve an explicit before/after
                # audit and align pre-existing literal final annotations as well.
                for record in records:
                    answer = record['title']
                    replacements = {}
                    for variant in [answer['items'], *answer['answer_alternatives']]:
                        for span in variant:
                            if re.search(r'\([0-9]{4}\)$', span['text'].rstrip()):
                                continue
                            following = row['text'][span['end']:]
                            match = re.match(r'[ \t]*\(([0-9]{4})\)', following)
                            if not match or int(match[1]) == 0:
                                continue
                            before = dict(span)
                            span['end'] += match.end()
                            span['text'] = row['text'][span['start']:span['end']]
                            replacements[before['text']] = span['text']
                            audit.append({'id': row['id'], 'task': task['id'], 'field': 'title',
                                'reason': 'source_title_boundary_repaired', 'before': before, 'after': dict(span)})
                    if replacements and not answer['final_annotation_missing']:
                        variants = answer['answer_alternatives'] or [answer['items']]
                        targets = [list(dict.fromkeys(s['text'] for s in spans)) for spans in variants]
                        answer['final_values'] = targets[0] if len(targets) == 1 else None
                        answer['final_alternatives'] = targets if len(targets) > 1 else []
            if 'director' in task['fields']:
                task['fields']['director'] = 'All explicitly named film directors, one person per value'
                rules['director'] = {'kind': 'list', 'separator': 'newline'}
        if row['source'] in {'kleister-charity', 'tractor_authored_gb_offices'}:
            for key, component in GB.items():
                if key in task['fields']:
                    rules[key] = {'kind': 'address', 'country': 'GB', 'component': component, 'group': 'registered_office'}
            if 'charity_name' in task['fields']:
                task['fields']['charity_name'] = 'Full legal name of the reporting charity; exclude page numbers and names of partners or other organisations'
            if 'charity_number' in task['fields']:
                task['fields']['charity_number'] = 'Registered charity number of the reporting charity, not its company number or another organisation identifier'
        # Preserve source spans as native supervision, annotate end values separately.
        for key, rule in rules.items():
            for record in records:
                answer = record[key]
                if answer['status'] != 'present':
                    continue
                # Existing independent annotations (including explicitly missing
                # ones) are authoritative. Reapplying an adapter must not replace
                # a normalized date/number with its literal source evidence.
                if (answer['final_values'] is not None or answer['final_alternatives']
                        or answer['final_annotation_missing']):
                    continue
                variants = answer['answer_alternatives'] or [answer['items']]
                targets = []
                for variant in variants:
                    values = [s['text'] for s in variant]
                    if rule.get('component') == 'postcode':
                        values = [''.join(v.split()).upper() for v in values]
                        values = [v[:-3]+' '+v[-3:] for v in values]
                    targets.append(list(dict.fromkeys(values)))
                answer['final_values'] = targets[0] if len(targets) == 1 else None
                answer['final_alternatives'] = targets if len(targets) > 1 else []
        # New normalized views have a declared contract before answers are read.
        # Missing independent targets stay missing; the request NEVER changes
        # based on an answer. Such fields cannot pass release qualification.
        if row['source'] == 'tractor_authored_documents':
            for key in task['fields']:
                if key == task['anchor'] or key in rules:
                    continue
                kind = 'date' if key in DATE_FIELDS else 'number' if key in NUMBER_FIELDS else (
                    'money' if row['family'] == 'invoices' and key in MONEY_FIELDS else None)
                if kind is None:
                    continue
                rules[key] = {'kind': kind, **({'currency': 'EUR'} if kind == 'money' else {}),
                              **({'date_order': 'DMY' if row['language'] == 'de' else None, 'allow_partial': True} if kind == 'date' else {})}
                annotated = []
                try:
                    for record in records:
                        a = record[key]
                        options = a['answer_alternatives'] or [a['items']]
                        annotated.append([list(dict.fromkeys(oracle(s['text'], kind, date_order=rules[key].get('date_order'), allow_partial=rules[key].get('allow_partial', False)) for s in spans)) for spans in options])
                except ValueError as exc:
                    audit.append({'id': row['id'], 'task': task['id'], 'field': key, 'kind': kind, 'reason': str(exc)})
                    for record in records:
                        if record[key]['status'] == 'present':
                            record[key]['final_annotation_missing'] = True
                    continue
                for record, targets in zip(records, annotated, strict=True):
                    record[key]['final_values'] = targets[0] if len(targets) == 1 else None
                    record[key]['final_alternatives'] = targets if len(targets) > 1 else []
    return Episode.model_validate(row), audit
