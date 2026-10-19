"""Offline error attribution; annotated evidence never enters runtime inference.

The gold-evidence probe tests processing contracts, not model quality. Categories
are observable symptoms, not proof of a unique causal stage.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .episodes import Episode
from .jsonio import loads
from .pipeline_evaluation import (
    expected,
    field_observation,
    infer,
    structural_issue_errors,
    task_counts,
)


def rows(response, section, task):
    value = response.get(section, {}).get(task.id, {} if task.kind == 'entities' else [])
    return [value] if task.kind == 'entities' else value


def gold_probe(episode):
    """Use the first independently supplied span alternative, never best-of scoring."""
    class AnnotatedEvidence:
        def __init__(self):
            self.position = 0

        def candidates(self, request):
            task = episode.tasks[self.position]
            self.position += 1
            return [{key: [s.model_dump() for s in (a.items or (a.answer_alternatives[0]
                           if a.answer_alternatives else []))] for key, a in row.items()}
                    for row in episode.gold[task.id].records]
    return infer(AnnotatedEvidence(), episode)


def attribute(episodes, predictions):
    totals = Counter()
    fields = defaultdict(Counter)
    examples = defaultdict(list)
    anchors = Counter()
    for e in episodes:
        if e.split != 'dev':
            raise ValueError('Attribution is Dev-only; calibration/test remain held out')
        item = predictions[e.id]
        response = {} if item.get('error') else item.get('prediction', {})
        probe = gold_probe(e)
        totals['episodes'] += 1
        totals['structural_issue_errors'] += len(structural_issue_errors(e, response))
        for task in e.tasks:
            target = expected(e, task)
            actual = rows(response, 'data', task)
            details = rows(response, 'details', task)
            _, pairs = task_counts(e, task, response, with_matches=True)
            matches = dict(pairs)
            _, probe_pairs = task_counts(e, task, probe, with_matches=True)
            probe_matches = dict(probe_pairs)
            probe_rows = rows(probe, 'data', task)
            if task.kind == 'records':
                anchors['gold_records'] += len(target)
                anchors['predicted_records'] += len(actual)
                anchors['unmatched_gold_records'] += len(target) - len(matches)
                anchors['unmatched_predicted_records'] += len(actual) - len(matches)
                identities = [tuple(row.get(task.anchor, [])) for row in actual]
                anchors['repeated_predicted_anchor_values'] += len(identities) - len(set(identities))
            for i, row in enumerate(target):
                j = matches.get(i)
                found_row = actual[j] if j is not None else {}
                evidence = details[j] if j is not None and j < len(details) else {}
                for key, (status, alternatives) in row.items():
                    group = f'{e.family}:{task.id}:{key}:{e.text_origin}/{e.label_origin}'
                    c = fields[group]
                    if status in {'unknown_annotation', 'ambiguous'}:
                        c['unscored_annotation'] += 1
                        continue
                    observation = field_observation(e, task, i, key, found_row, response, j, (status, alternatives))
                    found = observation['found']
                    if observation['exact']:
                        c['correct'] += 1
                        continue
                    p = probe_matches.get(i)
                    probe_observation = field_observation(e, task, i, key,
                        probe_rows[p] if p is not None else {}, probe, p, (status, alternatives))
                    probe_values = probe_observation['found']
                    c['gold_evidence_probe_pass' if probe_observation['exact'] else 'gold_evidence_probe_fail'] += 1
                    other = any(field_observation(e, task, i, key, candidate, response, n,
                                                  (status, alternatives))['exact']
                                and candidate.get(key) for n, candidate in enumerate(actual) if n != j)
                    if item.get('error'):
                        category = 'request_error'
                    elif task.kind == 'records' and j is None:
                        category = 'unmatched_anchor'
                    elif other:
                        category = 'correct_value_in_other_record'
                    elif observation['values_correct'] and not observation['issues_correct']:
                        category = 'issue_mismatch'
                    elif any(issue['task'] == task.id and issue['field'] == key and issue['index'] == j
                             for issue in response.get('issues', [])):
                        category = 'processing_issue'
                    elif not found and not evidence.get(key):
                        category = 'missing_value_and_evidence'
                    elif not found:
                        category = 'evidence_without_final_value'
                    else:
                        category = 'value_or_boundary_mismatch'
                    c[category] += 1
                    totals[category] += 1
                    if len(examples[group]) < 3:
                        examples[group].append({'id': e.id, 'record': i, 'category': category,
                            'expected_alternatives': [sorted(a) for a in alternatives], 'actual': sorted(found),
                            'evidence': evidence.get(key, []), 'gold_evidence_probe': sorted(probe_values),
                            'field_description': task.fields[key],
                            'expected_issues': sorted(observation['expected_issues']),
                            'actual_issues': sorted(observation['actual_issues']),
                            'gold_probe_issues': sorted(probe_observation['actual_issues'])})
            totals['extra_predicted_records'] += len(actual) - len(matches)
    return {'qualification': 'Offline Dev symptom attribution, NOT production scores or causal proof. '
            'Gold-evidence probe uses first annotated alternative; never used for runtime selection. '
            'Saved details can omit failed evidence in older predictions. Repeated anchor values are not automatically duplicates.',
            'counts': dict(totals), 'record_anchor_audit': dict(anchors),
            'fields': {k: dict(v) for k, v in sorted(fields.items())}, 'examples': dict(examples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    episodes = [Episode.model_validate_json(line) for line in args.episodes.read_text().splitlines() if line]
    items = [loads(line) for line in args.predictions.read_text().splitlines() if line]
    predictions = {item['id']: item for item in items}
    if len(predictions) != len(items) or set(predictions) != {e.id for e in episodes}:
        raise ValueError('Predictions must cover exactly the unique evaluation IDs')
    report = attribute(episodes, predictions)
    report['inputs'] = {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for key, path in [('episodes', args.episodes), ('predictions', args.predictions)]}
    report['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('pipeline_attribution.py', 'document_pipeline.py', 'field_normalization.py',
                     'pipeline_evaluation.py', 'field_contracts.py', 'contact_normalization.py',
                     'temporal_normalization.py', 'scalar_normalization.py', 'identifier_normalization.py',
                     'normalize.py', 'address_normalization.py', 'service.py', 'types.py', 'episodes.py')}
    report['counting_policy'] = 'One symptom per mismatched annotated gold field; extra predicted records '
    report['counting_policy'] += 'counted separately. Gold-evidence probe counts concern only mismatched fields.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'counts': report['counts'], 'anchors': report['record_anchor_audit']}))


if __name__ == '__main__':
    main()
