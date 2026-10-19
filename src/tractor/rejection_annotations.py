"""Annotate defined rejection outcomes independently of production inference.

Uses frozen corpus evidence and the independent annotation converter; never calls
the model, production resolver or normalizers. Unknown grammars remain unknown.
"""
import argparse
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .corpus_annotations import annotate
from .coverage_assembly import sha
from .episodes import Episode
from .jsonio import dumps, loads
from .pipeline_evaluation import typed_requests_for

REASONS = {
    'invalid_iban_checksum_annotation': 'invalid_iban',
    'invalid_phone_annotation': 'invalid_phone',
    'phone_annotation_requires_country': 'missing_phone_country',
    'ambiguous_numeric_annotation': 'ambiguous_number_format',
}


def outcome(value, rule):
    try:
        return annotate(value, rule), set()
    except (ValueError, ArithmeticError) as exc:
        code = REASONS.get(str(exc))
        if code:
            return [], {code}
        if (rule['kind'] == 'date' and not rule.get('date_order')
                and re.fullmatch(r'\d{1,2}([./-])\d{1,2}\1\d{4}', value.strip())):
            sep = re.search(r'[./-]', value)[0]
            dates = set()
            for fmt in (f'%d{sep}%m{sep}%Y', f'%m{sep}%d{sep}%Y'):
                try:
                    dates.add(datetime.strptime(value.strip(), fmt).date())
                except ValueError:
                    pass
            if len(dates) == 2:
                return [], {'ambiguous_or_unsupported_date'}
        raise


def adapt(episode):
    row = episode.model_dump()
    audit = []
    for task in row['tasks']:
        for index, record in enumerate(row['gold'][task['id']]['records']):
            for key, answer in record.items():
                if not answer['final_annotation_missing']:
                    continue
                rule = row['processing'].get(task['id'], {}).get(key)
                if rule is None or answer['status'] != 'present':
                    continue
                rule = {**rule, 'locale': rule.get('locale') or row['locale']}
                variants = answer['answer_alternatives'] or [answer['items']]
                results = []
                try:
                    for spans in variants:
                        values, issues = [], set()
                        for span in spans:
                            found, codes = outcome(span['text'], rule)
                            values.extend(found)
                            issues.update(codes)
                        if not values and rule['cardinality'] in {'required_one', 'one_or_more'}:
                            issues.add('missing_required_value')
                        results.append((list(dict.fromkeys(values)), issues))
                except (ValueError, ArithmeticError):
                    continue
                # Differing outcome sets across source alternatives need explicit
                # joint alternatives, not permission to mix unrelated values/issues.
                if not results or not results[0][1] or any(codes != results[0][1] for _, codes in results):
                    continue
                targets = [values for values, _ in results]
                answer.update(final_annotation_missing=False, expected_issues=sorted(results[0][1]),
                              final_values=targets[0] if len(targets) == 1 else None,
                              final_alternatives=targets if len(targets) > 1 else [])
                audit.append({'id': row['id'], 'task': task['id'], 'record': index, 'field': key,
                              'expected_issues': answer['expected_issues'], 'kind': rule['kind']})
    result = Episode.model_validate(row)
    if typed_requests_for(result) != typed_requests_for(episode):
        raise ValueError('Outcome annotation changed the public request')
    return result, audit


def build(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts, codes = Counter(), Counter()
    with source.open() as stream, (output / 'dev.episodes.jsonl').open('w') as target, (output / 'annotations.jsonl').open('w') as audit:
        for line in stream:
            e = Episode.model_validate(loads(line))
            if e.split != 'dev':
                raise ValueError('This initial outcome audit is Dev-only')
            result, events = adapt(e)
            serialized = dumps(result.model_dump())
            if Episode.model_validate(loads(serialized)) != result:
                raise ValueError('Serialized outcome changed episode provenance')
            target.write(serialized + '\n')
            counts['episodes'] += 1
            counts['annotated_outcomes'] += len(events)
            counts['remaining_missing_annotations'] += sum(a.final_annotation_missing for g in result.gold.values()
                                                          for r in g.records for a in r.values())
            for event in events:
                codes.update(event['expected_issues'])
                audit.write(dumps(event) + '\n')
    report = {'counts': dict(counts), 'issue_codes': dict(codes), 'source': str(source),
              'source_sha256': sha(source), 'files': {p.name: sha(p) for p in output.glob('*.jsonl')},
              'code_sha256': {name: sha(Path(__file__).with_name(name)) for name in
                             ('rejection_annotations.py', 'corpus_annotations.py', 'episodes.py')},
              'qualification': 'Independent expected API outcomes. No predictions consulted; no text, span, '
                               'schema or split changed. Unknown annotation grammars stay unscored.'}
    (output / 'manifest.json').write_text(dumps(report, indent=2) + '\n')
    print(dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    build(**vars(parser.parse_args()))
