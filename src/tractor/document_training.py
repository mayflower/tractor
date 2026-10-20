"""Strict-domain Dev checkpoint selection; release gates remain separate and complete."""
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .generalist_data import read_episodes
from .inference_context import EVALUATION_PRECISION, evaluation_precision
from .jsonio import dumps, loads
from .pipeline_evaluation import EVALUATION_CONTRACT, evaluate_pipeline, infer


def selection_score(report):
    # Rank measured groups without letting one permanently-zero group freeze
    # checkpoint selection. Release gates remain unchanged and independent.
    measured = {key: value['score'] for key, value in report['checks'].items()
                if value['score'] is not None}
    for key, score in report.get('scores', {}).items():
        if (key.startswith(('domain:', 'shape:')) and score.get('field_f1') is not None
                and score.get('reference_complete', True)):
            measured.setdefault(key + ':field_f1', score['field_f1'])
    for key, check in report.get('field_checks', {}).items():
        if check['status'] != 'insufficient_evidence':
            measured['field:' + key] = check['score']
    if not measured or any(not 0 <= value <= 1 for value in measured.values()):
        raise ValueError('No finite, valid measured Dev scores')
    values = [float(value) for value in measured.values()]
    return .9 * min(values) + .1 * sum(values) / len(values)


def regression_guard(report, baseline):
    """Bound field regressions and preserve domain and rejection outcomes on identical Dev."""
    failures = {}
    if report.get('evaluation_contract') != baseline.get('evaluation_contract'):
        return {'passed': False, 'failures': {'evaluation_contract': 'Re-score both models with the same evaluation contract'}}
    for section, tolerance in [('field_checks', .02), ('rejection_checks', 0.)]:
        for key, before in baseline.get(section, {}).items():
            if before['status'] == 'insufficient_evidence':
                continue
            after = report.get(section, {}).get(key, {})
            if (after.get('score') is None or after.get('status') == 'insufficient_evidence'
                    or float(after['score']) + tolerance + 1e-12 < float(before['score'])):
                failures[section + ':' + key] = {'before': before['score'], 'after': after.get('score'),
                                                'allowed_drop': tolerance}
    for key, before in baseline['scores'].items():
        if not key.startswith(('domain:', 'shape:')):
            continue
        if not before.get('reference_complete', True):
            continue
        if not report['scores'].get(key, {}).get('reference_complete', True):
            failures[key] = {'reason': 'reference coverage changed'}
            continue
        after = report['scores'].get(key, {}).get('field_f1')
        if after is None or float(after) + .005 + 1e-12 < float(before['field_f1']):
            failures[key] = {'before': before['field_f1'], 'after': after, 'allowed_drop': .005}
    # Include sparse rejection groups in the total; a group below the field
    # support floor must not silently lose a previously correct rejection.
    def correct_rejections(value):
        return sum(float(check['score']) * check['support'] for check in value.get('rejection_checks', {}).values()
                   if check['score'] is not None)
    if correct_rejections(report) + 1e-9 < correct_rejections(baseline):
        failures['total_correct_rejections'] = {'before': correct_rejections(baseline),
                                                'after': correct_rejections(report), 'allowed_drop': 0}
    return {'passed': not failures, 'failures': failures}


def dev_callback(path, policy_path, output, step, threshold=.5, record_threshold=None, regression_baseline=None):
    for value in (threshold, record_threshold):
        if value is not None and (isinstance(value, bool) or not 0 < float(value) <= 1):
            raise ValueError('Provisional Dev threshold must be in (0, 1]')
    episodes = list(read_episodes(path))
    if not episodes or any(e.split != 'dev' for e in episodes):
        raise ValueError('Checkpoint selection requires nonempty Dev only')
    policy = loads(Path(policy_path).read_text())
    baseline = loads(Path(regression_baseline).read_text()) if regression_baseline else None
    if baseline and (baseline.get('input_sha256') != sha(Path(path))
                     or baseline.get('policy_sha256') != sha(Path(policy_path))
                     or float(baseline.get('threshold', -1)) != threshold
                     or (float(baseline['record_threshold']) if baseline.get('record_threshold') is not None else None) != record_threshold
                     or baseline.get('evaluation_precision') != EVALUATION_PRECISION
                     or baseline.get('evaluation_contract') != EVALUATION_CONTRACT):
        raise ValueError('Regression baseline must use identical Dev, policy, thresholds and precision')
    coverage = Counter((e.family, e.artifact, f'{e.text_origin}/{e.label_origin}') for e in episodes)
    for domain in policy['domains']:
        if not any(k[0] == domain for k in coverage):
            raise ValueError(f'Missing Dev domain: {domain}')
    for shape in policy['shapes']:
        if not any(k[1] == shape for k in coverage):
            raise ValueError(f'Missing Dev output form: {shape}')
    root = Path(output) / 'extraction-dev'
    root.mkdir(parents=True, exist_ok=False)
    hashes = {'dev_sha256': sha(Path(path)), 'policy_sha256': sha(Path(policy_path)),
              'pipeline_sources': {name: sha(Path(__file__).with_name(name)) for name in
                                   ('document_pipeline.py', 'field_contracts.py', 'field_normalization.py',
                                    'pipeline_evaluation.py', 'pipeline_diagnostics.py', 'normalize.py', 'model.py',
                                    'address_normalization.py', 'contact_normalization.py', 'scalar_normalization.py',
                                    'temporal_normalization.py', 'identifier_normalization.py', 'service.py', 'types.py',
                                    'record_contracts.py', 'evaluation_reference.py', 'episodes.py',
                                    'episode_metrics.py', 'document_quality.py')},
              'threshold': threshold, 'record_threshold': record_threshold,
              'evaluation_precision': EVALUATION_PRECISION,
              'precision_source_sha256': sha(Path(__file__).with_name('inference_context.py'))}

    def measure(model, _dataset):
        from .model import Extractor
        backend = Extractor.from_native(model)
        backend.native_threshold = threshold
        backend.native_record_threshold = record_threshold
        predictions = {}
        try:
            with evaluation_precision():
                for episode in episodes:
                    try:
                        raw = infer(backend, episode)
                        predictions[episode.id] = {'prediction': raw}
                    except Exception as exc:
                        predictions[episode.id] = {'error': type(exc).__name__}
        finally:
            backend.borrowed_hook.remove()
        report = evaluate_pipeline(episodes, predictions, policy)
        minimum = selection_score(report)
        if baseline:
            report['regression_guard'] = regression_guard(report, baseline)
            report['unconstrained_selection_metric'] = minimum
            report['regression_baseline_sha256'] = sha(Path(regression_baseline))
            if not report['regression_guard']['passed']:
                minimum -= 1.0  # Every eligible score is in [0, 1].
        current = step()
        report.update(hashes, step=current, selection_metric=minimum,
                      missing_evidence=[k for k, v in report['checks'].items() if v['score'] is None],
                      selection_qualification='0.9 * minimum + 0.1 * macro mean of unique measured Dev gates, including Silver. '
                      'Unavailable real evidence remains explicitly missing and blocks release qualification. '
                      'Recorded thresholds are provisional; calibration and test are not used.')
        (root / f'step-{current}.json').write_text(dumps(report, indent=2))
        with (root / f'step-{current}.predictions.jsonl').open('w') as stream:
            for ident, raw in predictions.items():
                stream.write(dumps({'id': ident, **raw}) + '\n')
        return {'eval_pipeline_selection_score': minimum, 'eval_extraction_release_gate': float(report['passed'])}
    return measure
