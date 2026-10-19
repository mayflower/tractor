"""Strict-domain Dev checkpoint selection; release gates remain separate and complete."""
from collections import Counter
from pathlib import Path

from .coverage_assembly import sha
from .generalist_data import read_episodes
from .jsonio import dumps, loads
from .pipeline_evaluation import evaluate_pipeline, infer


def selection_score(report):
    # Rank measured groups without letting one permanently-zero group freeze
    # checkpoint selection. Release gates remain unchanged and independent.
    measured = {key: value['score'] for key, value in report['checks'].items()
                if value['score'] is not None}
    for key, score in report.get('scores', {}).items():
        if key.startswith(('domain:', 'shape:')) and score.get('field_f1') is not None:
            measured.setdefault(key + ':field_f1', score['field_f1'])
    for key, check in report.get('field_checks', {}).items():
        if check['status'] != 'insufficient_evidence':
            measured['field:' + key] = check['score']
    if not measured or any(not 0 <= value <= 1 for value in measured.values()):
        raise ValueError('No finite, valid measured Dev scores')
    values = list(measured.values())
    return .9 * min(values) + .1 * sum(values) / len(values)


def dev_callback(path, policy_path, output, step, threshold=.5, record_threshold=None):
    for value in (threshold, record_threshold):
        if value is not None and (isinstance(value, bool) or not 0 < float(value) <= 1):
            raise ValueError('Provisional Dev threshold must be in (0, 1]')
    episodes = list(read_episodes(path))
    if not episodes or any(e.split != 'dev' for e in episodes):
        raise ValueError('Checkpoint selection requires nonempty Dev only')
    policy = loads(Path(policy_path).read_text())
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
                                    'temporal_normalization.py', 'identifier_normalization.py', 'service.py', 'types.py')},
              'threshold': threshold, 'record_threshold': record_threshold}

    def measure(model, _dataset):
        import torch

        from .model import Extractor
        backend = Extractor.from_native(model)
        backend.native_threshold = threshold
        backend.native_record_threshold = record_threshold
        predictions = {}
        try:
            with torch.no_grad():
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
