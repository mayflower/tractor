"""Migrate qualified corpus views without changing source supervision or splits."""
import argparse
import hashlib
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .corpus_annotations import annotate
from .coverage_assembly import sha
from .episodes import Episode
from .field_contracts import Processing
from .jsonio import dumps, loads
from .pipeline_evaluation import typed_requests_for
from .typed_corpus_policy import VERSION, choose


def evidence(row):
    """Fingerprint everything except processing and separate final annotations."""
    original = deepcopy(row)
    original.pop('processing', None)
    for gold in original['gold'].values():
        for record in gold['records']:
            for answer in record.values():
                for key in ('final_values', 'final_alternatives', 'final_annotation_missing'):
                    answer.pop(key, None)
    return hashlib.sha256(dumps(original, sort_keys=True).encode()).hexdigest()


def migrate(episode):
    row = episode.model_dump()
    before = evidence(row)
    # Type policy cannot read gold, split membership or model predictions.
    schema = {k: row[k] for k in ('source', 'source_group', 'family', 'language', 'text', 'locale')}
    events = []
    for task in row['tasks']:
        rules = row['processing'].setdefault(task['id'], {})
        for key in task['fields']:
            old = rules.get(key)
            new, debt = choose(schema, task, key, old)
            new = Processing.model_validate(new).model_dump()
            rules[key] = new
            context = {'id': row['id'], 'source': row['source'], 'family': row['family'],
                       'task': task['id'], 'field': key, 'kind': new['kind']}
            if debt:
                events.append({**context, 'event': 'contract_gap', 'reason': debt})
            changed = old is None or old != new
            if changed:
                events.append({**context, 'event': 'contract_assigned',
                               'before': old, 'after': new})
            for index, record in enumerate(row['gold'][task['id']]['records']):
                answer = record[key]
                if answer['status'] != 'present':
                    continue
                if not changed and not answer['final_annotation_missing'] and (
                        answer['final_values'] is not None or answer['final_alternatives']):
                    continue
                variants = answer['answer_alternatives'] or [answer['items']]
                try:
                    targets = [list(dict.fromkeys(v for span in spans for v in annotate(span['text'], new)))
                               for spans in variants]
                except (ValueError, ArithmeticError) as exc:
                    answer.update(final_values=None, final_alternatives=[], final_annotation_missing=True)
                    events.append({**context, 'record': index, 'event': 'annotation_gap',
                                   'reason': str(exc)[:160]})
                    continue
                was_missing = answer['final_annotation_missing']
                answer.update(final_values=targets[0] if len(targets) == 1 else None,
                              final_alternatives=targets if len(targets) > 1 else [],
                              final_annotation_missing=False)
                if was_missing:
                    events.append({**context, 'record': index, 'event': 'annotation_recovered'})
    result = Episode.model_validate(row)
    typed_requests_for(result)  # Actual public discriminated schema, not a private approximation.
    if evidence(result.model_dump()) != before:
        raise ValueError('Migration changed evidence, schema, identities or split')
    return result, events


def migrate_file(source, output, audit):
    counts, types, by_origin, reasons = Counter(), Counter(), Counter(), Counter()
    evidence_before, evidence_after = hashlib.sha256(), hashlib.sha256()
    with Path(source).open() as stream, Path(output).open('w') as target:
        for line in stream:
            original = Episode.model_validate(loads(line))
            result, events = migrate(original)
            evidence_before.update(evidence(original.model_dump()).encode())
            evidence_after.update(evidence(result.model_dump()).encode())
            counts['episodes'] += 1
            for task in result.tasks:
                for key, rule in result.processing[task.id].items():
                    types['text' if rule.kind == 'literal' else rule.kind] += 1
                    by_origin[f'{result.text_origin}:{rule.kind}'] += 1
                    for record in result.gold[task.id].records:
                        counts['present_answers'] += record[key].status == 'present'
                        counts['missing_final_annotations'] += record[key].final_annotation_missing
            counts['previous_missing_final_annotations'] += sum(
                a.final_annotation_missing for g in original.gold.values() for r in g.records for a in r.values())
            for event in events:
                counts[event['event']] += 1
                if 'reason' in event:
                    reasons[event['event'] + ':' + event['reason']] += 1
                audit.write(dumps(event) + '\n')
            # Origin defaults depend on annotation/source. Dropping an explicitly
            # synthetic/deterministic value would reconstruct different provenance.
            serialized = dumps(result.model_dump())
            if evidence(Episode.model_validate(loads(serialized)).model_dump()) != evidence(result.model_dump()):
                raise ValueError('Serialized episode changed evidence or provenance')
            target.write(serialized + '\n')
            if counts['episodes'] % 10000 == 0:
                print(dumps({'file': str(source), 'episodes': counts['episodes']}), flush=True)
    if evidence_before.digest() != evidence_after.digest():
        raise ValueError('Evidence fingerprint mismatch')
    return {'counts': dict(counts), 'field_types': dict(types), 'by_origin': dict(by_origin),
            'gaps': dict(reasons), 'evidence_sha256': evidence_after.hexdigest(),
            'source_sha256': sha(source), 'output_sha256': sha(output)}


def build(source, output):
    source, output = Path(source), Path(output)
    source_manifest = source / 'manifest.json'
    manifest = loads(source_manifest.read_text()) if source_manifest.exists() else {'files': {}}
    if manifest.get('training_ready') is False:
        raise ValueError('Unqualified source cannot become a typed training corpus')
    code = {name: sha(Path(__file__).with_name(name)) for name in
            ('typed_corpus.py', 'typed_corpus_policy.py', 'corpus_annotations.py', 'pipeline_contract_data.py')}
    # Verify frozen inputs, including heldout files, before any output writes.
    for name, digest in manifest['files'].items():
        if sha(source / name) != digest:
            raise ValueError(f'Input hash mismatch: {source / name}')
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    with (output / 'migration-audit.jsonl').open('w') as audit:
        for path in sorted(source.glob('*.episodes.jsonl')):
            summaries[path.name] = migrate_file(path, output / path.name, audit)
        for name in manifest['files']:
            if name not in summaries:
                shutil.copyfile(source / name, output / name)
    report = {
        'policy': VERSION, 'source': str(source),
        'source_manifest_sha256': sha(source_manifest) if source_manifest.exists() else None,
        'code_sha256': code,
        'splits': summaries,
        'qualification': 'Evidence, identities, schemas, split membership and native supervision preserved. '
                         'Final annotations use an independent converter. No model evaluation or promotion. '
                         'Missing contracts/targets remain gaps, not successes.',
    }
    if any(sha(Path(__file__).with_name(name)) != digest for name, digest in code.items()):
        raise ValueError('Migration code changed during build; rerun with frozen code')
    (output / 'migration-report.json').write_text(dumps(report, indent=2) + '\n')
    manifest.update(type_migration=report, normalization_ready=all(
        not s['counts'].get('contract_gap') and not s['counts'].get('missing_final_annotations')
        for s in summaries.values()))
    manifest['files'] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    (output / 'manifest.json').write_text(dumps(manifest, indent=2) + '\n')
    print(dumps({'output': str(output), 'splits': {k: v['counts'] for k, v in summaries.items()}}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    build(**vars(parser.parse_args()))
