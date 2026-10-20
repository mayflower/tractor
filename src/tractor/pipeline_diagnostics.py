"""Final-value diagnostics per declared field and explicit record, with evidence counts."""
from collections import Counter, defaultdict

from .episode_metrics import summarize
from .pipeline_evaluation import expected, field_observation, task_counts, value_key


def diagnostics(episodes, predictions, minimum_support=30, threshold=.7):
    fields = defaultdict(Counter)
    records = defaultdict(Counter)
    examples = defaultdict(list)
    unresolved_examples = defaultdict(list)
    for e in episodes:
        item = predictions[e.id]
        response = {} if item.get('error') else item.get('prediction', {})
        origin = f'{e.text_origin}/{e.label_origin}'
        for task in e.tasks:
            targets = expected(e, task)
            actual = response.get('data', {}).get(task.id, {} if task.kind == 'entities' else [])
            actual = [actual] if task.kind == 'entities' else actual
            counts, matched = task_counts(e, task, response, with_matches=True)
            matches = dict(matched)
            if task.kind == 'records':
                record = records[f'{e.family}:{origin}']
                record.update(counts)
                record['unmatched_gold_records'] += len(targets) - len(matched)
                record['unmatched_predicted_records'] += len(actual) - len(matched)
                record['matched_records_with_wrong_fields'] += len(matched) - counts['exact_records']
            for index, row in enumerate(targets):
                found_row = actual[matches[index]] if index in matches else {}
                for key, (status, alternatives) in row.items():
                    field_key = f'{e.family}:{task.id}:{key}:{origin}'
                    c = fields[field_key]
                    if status in {'ambiguous', 'unknown_annotation'}:
                        c['unscored_fields'] += 1
                        continue
                    observed = field_observation(e, task, index, key, found_row, response,
                                                 matches.get(index), (status, alternatives))
                    found, gold, exact = observed['found'], observed['gold'], observed['exact']
                    c.update(observed['counts'])
                    if observed['unresolved_reference']:
                        if len(unresolved_examples[field_key]) < 3:
                            unresolved_examples[field_key].append({
                                'id': e.id, 'record': index, 'reference': sorted(gold),
                                'actual': sorted(found), 'diagnosis': 'incomplete_reference_requires_review'})
                        continue
                    if not exact:
                        kind = ('issue_mismatch' if observed['values_correct'] else
                                'missing_value' if not found else 'extra_value' if not gold else 'wrong_or_incomplete_value')
                        c[kind] += 1
                        if len(examples[field_key]) < 3:
                            examples[field_key].append({'id': e.id, 'record': index, 'expected': sorted(gold),
                                                       'actual': sorted(found), 'diagnosis': kind,
                                                       'expected_issues': sorted(observed['expected_issues']),
                                                       'actual_issues': sorted(observed['actual_issues'])})
            for index in set(range(len(actual))) - set(matches.values()):
                for key in task.fields:
                    found = {value_key(v) for v in actual[index].get(key, [])}
                    c = fields[f'{e.family}:{task.id}:{key}:{origin}']
                    c.update(field_fp=int(bool(found)), fp=len(found), extra_record_values=len(found))
    scores = {k: summarize(c) for k, c in sorted(fields.items())}
    checks = {}
    for key, result in scores.items():
        support = result['counts'].get('answerable', 0)
        value = result['field_f1']
        incomplete = result['counts'].get('incomplete_reference_fields', 0) > 0
        checks[key] = {'score': None if incomplete else value, 'positive_support': support,
                       'reference_complete': not incomplete,
                       'status': 'insufficient_evidence' if support < minimum_support or value is None or result['counts'].get('unscored_fields', 0) > 0 else
                       'insufficient_evidence' if incomplete else 'pass' if value > threshold else 'fail'}
    rejection_checks = {}
    for key, result in scores.items():
        support = result['counts'].get('expected_rejections', 0)
        if support:
            value = result['counts'].get('correct_rejections', 0) / support
            rejection_checks[key] = {'score': value, 'support': support,
                'status': 'insufficient_evidence' if support < minimum_support else 'pass' if value > threshold else 'fail'}
    return {'rejection_checks': rejection_checks, 'field_scores': scores, 'field_checks': checks, 'field_error_examples': dict(examples),
            'unresolved_reference_examples': dict(unresolved_examples),
            'record_diagnostics': {k: dict(v) for k, v in records.items()},
            'field_support_policy': {'minimum_positive_support': minimum_support,
                                     'threshold_exclusive': threshold,
                                     'insufficient_evidence_is_not_success': True}}
