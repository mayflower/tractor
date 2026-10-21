"""Dev-only diagnostic of native schemas; final values use Tractor normalization.

Native schemas are NOT an alternate public API. No native calibration/test rows
are evaluated, and no labels are passed to the model.
"""

import argparse
from collections import deque
from pathlib import Path
from types import MethodType

from tractor.biqwen import BiQwenExtractor, decode_multiple
from tractor.generalist_data import read_episodes
from tractor.inference_context import evaluation_precision
from tractor.jsonio import dumps, loads
from tractor.pipeline_evaluation import evaluate_pipeline, infer, typed_requests_for
from tractor.source_policy import excluded_source
from tractor.types import TractorError


class NativeEvidence:
    def __init__(self, episode, raw, upstream):
        self.tasks = deque(enumerate(episode.tasks))
        self.raw, self.upstream = raw, upstream

    def candidates(self, request):
        index, task = self.tasks.popleft()
        if list(task.fields) != list(request.fields):
            raise ValueError('Native task fields differ from the public request')
        if self.upstream:
            capacity = self.raw['capacity']
            if capacity['record_overflow'] or capacity['anchor_candidates_exhausted']:
                raise TractorError(422, 'output_capacity', 'Upstream reported output capacity')
        table = f't{index}'
        rows = self.raw['tables'][table] if request.anchor else [{'fields': self.raw['fields']}]
        records = []
        for row in rows:
            record = {}
            for name in request.fields:
                if name == request.anchor:
                    pairs = [row['anchor']]
                else:
                    pairs = row['fields'].get(f'{table}.{name}', [])
                    if self.upstream:
                        values = [pairs] if isinstance(pairs, dict) else pairs
                        pairs = [v['source'] for v in values if v['source'] is not None]
                record[name] = [{'text': request.text[a:b], 'start': a, 'end': b} for a, b in sorted(pairs)]
            records.append(record)
        return records


def validate_contract(episode, native):
    if native['metadata']['split'] != 'validation' or episode.split != 'dev':
        raise ValueError('Only native validation and Tractor Dev are permitted')
    if excluded_source(episode) or excluded_source(native['metadata']):
        raise ValueError('Excluded source')
    if native['text'] != episode.text or native['schema']['choices']:
        raise ValueError('Text differs or schema contains decisions')
    expected = []
    for i, task in enumerate(episode.tasks):
        expected.extend(f't{i}.@{key}' if key == task.anchor else f't{i}.{key}' for key in task.fields)
    fields = native['schema']['fields']
    if [f['id'] for f in fields] != expected or any(f.get('resolve_references') for f in fields):
        raise ValueError('Native field mapping differs or uses reference resolution')
    typed_requests_for(episode)  # Validate the production contract before inference.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/biqwen-extract-decide-4b')
    parser.add_argument('--episodes', default='data/full-training-de-en-native/dev.episodes.jsonl')
    parser.add_argument('--native-validation', default='data/biqwen-source-audit/data/validation.jsonl')
    parser.add_argument('--baseline', default='reports/cord-removal-20261021/biqwen/predictions.jsonl')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    episodes = {r.id: r for r in read_episodes(args.episodes)}
    selected = []
    for line in Path(args.native_validation).open():
        native = loads(line)
        matches = [episodes[key] for key in native['metadata']['tractor_ids'] if key in episodes]
        if len(matches) != 1:
            raise ValueError('Expected exactly one Dev episode per native validation row')
        validate_contract(matches[0], native)
        selected.append((matches[0], native))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    backend = BiQwenExtractor(args.model, 'cuda', 'bf16')
    captured = []
    upstream_decode = type(backend.model)._decode_example

    def both(model, *inputs):
        captured.append(upstream_decode(model, *inputs))
        return decode_multiple(model, *inputs)

    backend.model._decode_example = MethodType(both, backend.model)
    predictions = {'native_upstream': {}, 'native_multiple': {}}
    with evaluation_precision(), (root / 'predictions.jsonl').open('w') as stream:
        for index, (episode, native) in enumerate(selected):
            captured.clear()
            try:
                encoded = backend.encoding.encode(
                    {'text': episode.text, 'schema': native['schema']}, backend.model.tokenizer,
                    backend.input_limit, schema_prefix=backend.model.config.schema_prefix,
                )
                raw = backend.model(backend.encoding.collate([encoded], backend.model.tokenizer.pad_token_id))['predictions'][0]
                for label, result, upstream in [('native_multiple', raw, False), ('native_upstream', captured.pop(), True)]:
                    try:
                        value = {'prediction': infer(NativeEvidence(episode, result, upstream), episode)}
                    except Exception as exc:
                        value = {'error': type(exc).__name__, 'message': str(exc)[:400]}
                    predictions[label][episode.id] = value
            except Exception as exc:
                for label in predictions:
                    predictions[label][episode.id] = {'error': type(exc).__name__, 'message': str(exc)[:400]}
            stream.write(dumps({'id': episode.id, **{k: v[episode.id] for k, v in predictions.items()}}) + '\n')
            stream.flush()
            if (index + 1) % 50 == 0:
                print(dumps({'completed': index + 1, 'total': len(selected)}), flush=True)
    baseline = {r['id']: {k: v for k, v in r.items() if k != 'id'}
                for r in map(loads, Path(args.baseline).open())}
    predictions['api_multiple'] = {row.id: baseline[row.id] for row, _ in selected}
    policy = loads(Path('configs/extraction-quality.json').read_text())
    report = {key: evaluate_pipeline([r for r, _ in selected], values, policy) for key, values in predictions.items()}
    report['scope'] = 'All native validation rows matched to Tractor Dev; native schema is diagnostic only; same final normalization.'
    report['documents'] = len(selected)
    report['model_revision'] = backend.model_revision
    report['weights_hashes'] = backend.weights_hashes
    (root / 'metrics.json').write_text(dumps(report, indent=2) + '\n')
    print(dumps({'completed': len(selected), 'finished': True}), flush=True)


if __name__ == '__main__':
    main()
