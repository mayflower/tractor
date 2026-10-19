"""Evaluate a frozen checkpoint on Dev only, retaining predictions and strict release gates."""
import argparse
from pathlib import Path

from .coverage_assembly import sha
from .document_training import selection_score
from .generalist_data import read_episodes
from .jsonio import dumps, loads
from .pipeline_evaluation import evaluate_pipeline, infer


def run(model, episodes, output, device='cuda', policy='configs/extraction-quality.json', threshold=None, record_threshold=None):
    for value in (threshold, record_threshold):
        if value is not None and (isinstance(value, bool) or not 0 < float(value) <= 1):
            raise ValueError('Provisional Dev threshold must be in (0, 1]')
    import torch

    from .model import Extractor
    rows = list(read_episodes(episodes))
    if not rows or any(e.split != 'dev' for e in rows):
        raise ValueError('This comparison evaluator accepts Dev only')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    source_names = ('document_pipeline.py', 'field_contracts.py', 'field_normalization.py',
                    'pipeline_evaluation.py', 'pipeline_diagnostics.py', 'normalize.py', 'model.py', 'types.py', 'service.py',
                    'contact_normalization.py', 'temporal_normalization.py', 'scalar_normalization.py',
                    'identifier_normalization.py', 'address_normalization.py', 'episodes.py', 'episode_metrics.py', 'document_eval.py')
    source_hashes = {name: sha(Path(__file__).with_name(name)) for name in source_names}
    # Keep the exact implementation reviewable alongside the frozen predictions.
    snapshot = root / 'pipeline-source'
    snapshot.mkdir()
    for name in source_names:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    native = Extractor(model, device)
    if threshold is not None:
        # Dev sensitivity experiment only; never modify a bundle's calibrated policy.
        native.native_threshold = float(threshold)
    if record_threshold is not None:
        native.native_record_threshold = float(record_threshold)
    if threshold is not None or record_threshold is not None:
        native.use_native_policy = True
    predictions = {}
    with torch.inference_mode(), (root/'predictions.jsonl').open('w') as stream:
        for i, e in enumerate(rows):
            try:
                raw = infer(native, e)
                row = {'prediction': raw}
            except Exception as exc:
                row = {'error': type(exc).__name__, 'message': str(exc)[:400]}
            predictions[e.id] = row
            stream.write(dumps({'id': e.id, **row})+'\n')
            if (i+1) % 100 == 0:
                print(dumps({'completed': i+1, 'total': len(rows)}), flush=True)
    report = evaluate_pipeline(rows, predictions, loads(Path(policy).read_text()))
    if any(sha(Path(__file__).with_name(name)) != digest for name, digest in source_hashes.items()):
        raise ValueError('Pipeline implementation changed during evaluation')
    report.update(selection_metric=selection_score(report), model=str(model),
                  input_sha256=sha(Path(episodes)), weights_sha256=sha(Path(model)/'model.safetensors'),
                  threshold=native.native_threshold,
                  threshold_override=threshold is not None,
                  record_threshold=native.native_record_threshold,
                  record_threshold_override=record_threshold is not None,
                  use_native_policy=native.use_native_policy, bundle_field_thresholds=native.thresholds,
                  scope='Production pipeline Dev selection; not final calibration or test.',
                  policy_sha256=sha(Path(policy)),
                  normalization_artifacts={str(path): sha(path) for path in
                      (Path('flake.lock'), Path('flake.nix'), Path('uv.lock'), Path('configs/libpostal-data.sha256'))
                      if path.is_file()},
                  pipeline_sources=source_hashes)
    (root/'metrics.json').write_text(dumps(report, indent=2))
    print(dumps({'selection_metric': report['selection_metric'], 'passed': report['passed']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--policy', default='configs/extraction-quality.json')
    parser.add_argument('--threshold', type=float, help='Provisional Dev-only sensitivity test; bundle unchanged')
    parser.add_argument('--record-threshold', type=float, help='Provisional threshold for records only; entity fields keep --threshold')
    run(**vars(parser.parse_args()))
