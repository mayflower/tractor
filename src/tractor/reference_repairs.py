"""Offline source-contract repairs and explicit, source-checked adjudications.

Never imported by inference. References do not choose runtime normalization.
"""
import hashlib
import json
import re

from .episodes import Episode

CASE_PREFIXES = ['File No.', 'Our ref:', 'FILE REF:', 'Ref:', 'Az.']
FILM_TITLE_DESCRIPTION = (
    'Complete displayed film title, optionally including the release year in parentheses '
    'when it is printed as part of the title'
)


def acceptance_context(episode, task_id, record, field):
    """Bind an offline decision to its source, schema and annotated record."""
    task = next(t for t in episode.tasks if t.id == task_id)
    answer = episode.gold[task_id].records[record][field]
    rule = episode.processing.get(task_id, {}).get(field)
    payload = {
        'id': episode.id, 'source_text': episode.text,
        'task': task.model_dump(), 'record': record, 'field': field,
        'processing': rule.model_dump() if rule else None,
        'evidence': [s.model_dump() for s in answer.items],
        'alternative_evidence': [[s.model_dump() for s in group]
                                 for group in answer.answer_alternatives],
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def apply_accepted_answers(episode, reviews):
    """Add explicitly reviewed complete answer sets, never fuzzy equivalence.

    Keeping whole sets matters: accepting ``[Asian, Asian/Sushi]`` must not
    accept ``[Asian]`` alone, nor merge two different people. Existing references,
    source spans, issue expectations and inference requests are preserved.
    """
    row = episode.model_dump()
    changes = []
    for review in reviews:
        if review['id'] != episode.id:
            continue
        if review['conclusion'] != 'semantic_equivalent':
            raise ValueError('Only approved semantic equivalents may be accepted')
        task, record, field = review['task'], review['record'], review['field']
        if acceptance_context(episode, task, record, field) != review['context_sha256']:
            raise ValueError('Reviewed source or field contract changed')
        answer = row['gold'][task]['records'][record][field]
        if (answer['status'] != 'present' or answer['final_annotation_missing']
                or answer['expected_issues'] or not answer['reference_complete']):
            raise ValueError('Acceptance requires a complete positive reference')
        approved = review['approved_alternatives']
        if not approved or any(
                not isinstance(group, list) or not group
                or any(not isinstance(v, str) or not v.strip() for v in group)
                for group in approved):
            raise ValueError('Approved alternatives must be nonempty answer sets')
        original = answer['final_alternatives'] or (
            [answer['final_values']] if answer['final_values'] is not None else
            [[s['text'] for s in group] for group in (
                answer['answer_alternatives'] or [answer['items']])])
        alternatives = [list(group) for group in original]
        seen = {frozenset(group) for group in alternatives}
        for group in approved:
            if frozenset(group) not in seen:
                alternatives.append(list(dict.fromkeys(group)))
                seen.add(frozenset(group))
        if len(alternatives) != len(original):
            answer.update(final_values=None, final_alternatives=alternatives)
            changes.append({'review': review['audit_id'], 'task': task,
                            'record': record, 'field': field,
                            'added_alternatives': len(alternatives) - len(original)})
    return Episode.model_validate(row), changes


def source_contracts(row):
    """Prediction-independent corrections, applicable before compiling any split."""
    changes = []
    for task in row['tasks']:
        rules = row['processing'].setdefault(task['id'], {})
        # OpenTable's source attribute is the restaurant's address. Its adjacent
        # annotated lines are components of that one attribute, not independent
        # addresses. This source rule does not parse gold with the runtime parser.
        if (row['source'] == 'swde' and row['source_group'].startswith('swde:restaurant:opentable:')
                and 'address' in task['fields']
                and rules.get('address', {}).get('kind') == 'address'
                and rules['address'].get('component') in (None, 'full')):
            for record in row['gold'][task['id']]['records']:
                answer = record['address']
                spans = sorted(answer['items'], key=lambda s: s['start'])
                if (len(spans) > 1 and not answer['answer_alternatives']
                        and not answer['final_alternatives']
                        and len({s['text'] for s in spans}) == len(spans)
                        and all(b['start'] >= a['end'] and re.fullmatch(
                            r'[ \t]*\r?\n[ \t]*', row['text'][a['end']:b['start']])
                            for a, b in zip(spans, spans[1:], strict=False))):
                    start, end = spans[0]['start'], spans[-1]['end']
                    value = row['text'][start:end]
                    final = ', '.join(value.splitlines())
                    answer.update(items=[{'start': start, 'end': end, 'text': value}], final_values=[final])
                    changes.append({'task': task['id'], 'field': 'address',
                                    'reason': 'source_address_lines_are_one_attribute',
                                    'before': spans, 'after': final})
        for key in task['fields']:
            prefixes = None
            if (row['source'] == 'tractor_authored_documents'
                    and row['family'] == 'metadata' and key == 'case'
                    and rules.get(key, {}).get('kind') == 'identifier'):
                prefixes = CASE_PREFIXES
            if (row['source'] == 'tractor_authored_documents'
                    and row['family'] == 'products' and key == 'size'
                    and rules.get(key, {}).get('kind', 'literal') == 'literal'):
                prefixes = ['Size ']
            if prefixes:
                rules.setdefault(key, {'kind': 'literal'})['prefixes'] = prefixes
                for record in row['gold'][task['id']]['records']:
                    answer = record[key]
                    replacements = {}
                    for spans in [answer['items'], *answer['answer_alternatives']]:
                        for span in spans:
                            for prefix in prefixes:
                                if span['text'].casefold().startswith(prefix.casefold()):
                                    old = span['text']
                                    value = old[len(prefix):].strip()
                                    if not value:
                                        continue
                                    start = span['start'] + old.index(value)
                                    span.update(start=start, end=start+len(value), text=value)
                                    replacements[old] = value
                                    break
                    for targets in [answer['final_values'], *answer['final_alternatives']]:
                        if targets is not None:
                            targets[:] = [replacements.get(v, v) for v in targets]
                    if replacements:
                        changes.append({'task': task['id'], 'field': key, 'reason': 'label_boundary',
                                        'replacements': replacements})
        if row['source'] == 'swde' and row['source_group'].startswith('swde:movie:') and 'title' in task['fields']:
            task['fields']['title'] = FILM_TITLE_DESCRIPTION
            rules.setdefault('title', {'kind': 'literal'}).pop('include_adjacent_year', None)
            for record in row['gold'][task['id']]['records']:
                answer = record['title']
                # Printed year is optional only for an unambiguous title here.
                # Do not conflate distinct dated films sharing one bare title.
                titles = (answer['final_values'] if answer['final_values'] is not None else
                          [v for group in answer['final_alternatives'] for v in group]
                          if answer['final_alternatives'] else [s['text'] for s in answer['items']])
                if (answer['final_values'] is None and not answer['final_alternatives']
                        and not answer['final_annotation_missing']):
                    answer['final_values'] = list(dict.fromkeys(titles))
                for title in dict.fromkeys(titles):
                    bare = re.sub(r'\s*\([0-9]{4}\)$', '', title).strip()
                    pattern = r'(?<!\w)' + re.escape(bare) + r'(?:[ \t]*\([0-9]{4}\))?(?!\w)'
                    variants = list(dict.fromkeys(m[0] for m in re.finditer(pattern, row['text'], re.I)))
                    years = {m[1] for v in variants if (m := re.search(r'\(([0-9]{4})\)$', v))}
                    if len(years) > 1:
                        continue
                    variants = list(dict.fromkeys([title, bare, *variants]))
                    if len(variants) > 1 and not any(set(variants) & set(g) for g in answer['value_equivalences']):
                        answer['value_equivalences'].append(variants)
    return changes


def apply_reviews(episode, decisions):
    """Apply frozen human/contextual review, fail closed if its source changed."""
    row = episode.model_dump()
    audit = source_contracts(row)
    for review in decisions:
        if review['id'] != row['id']:
            continue
        if hashlib.sha256(row['text'].encode()).hexdigest() != review['source_sha256']:
            raise ValueError('Reviewed source changed')
        answer = row['gold'][review['task']]['records'][review['record']][review['field']]
        category = review['category']
        if review['conclusion'] == 'semantic_false_negative' or category == 'place_granularity':
            variants = list(dict.fromkeys(
                [v for group in review['raw_expected'] for v in group] + review['raw_actual']))
            if variants not in answer['value_equivalences']:
                answer['value_equivalences'].append(variants)
        elif category in {'people_merged_in_gold', 'dangling_separator_in_gold', 'address_line_partition'}:
            if category == 'address_line_partition':
                # This specific source review established one contiguous address.
                # Independent source evidence, never the production parser as gold.
                spans = sorted(answer['items'], key=lambda s: s['start'])
                start, end = spans[0]['start'], spans[-1]['end']
                if any(row['text'][a['end']:b['start']].strip() for a, b in zip(spans, spans[1:], strict=False)):
                    raise ValueError('Reviewed address is not contiguous')
                targets = [row['text'][start:end]]
                final = [', '.join(targets[0].splitlines())]
            else:
                targets = review['raw_actual']
                final = targets
            spans = []
            for value in targets:
                # Restrict repaired boundaries to the original annotated region.
                for old in answer['items']:
                    offset = old['text'].find(value)
                    if offset >= 0:
                        start = old['start'] + offset
                        break
                else:
                    if category != 'address_line_partition':
                        raise ValueError('Repair leaves original evidence')
                spans.append({'start': start, 'end': start+len(value), 'text': value})
            answer.update(items=spans, answer_alternatives=[], final_values=final, final_alternatives=[])
        elif review['conclusion'] not in {'annotation_error', 'contract_overspecification'}:
            continue
        audit.append({'task': review['task'], 'field': review['field'], 'record': review['record'],
                      'reason': category})
    return Episode.model_validate(row), audit
