"""Dev-only decoder ablation: both decoders see identical neural states in one pass."""

import argparse
from collections import deque
from pathlib import Path
from types import MethodType

from tractor.biqwen import BiQwenExtractor, decode_multiple
from tractor.generalist_data import read_episodes
from tractor.inference_context import evaluation_precision
from tractor.jsonio import dumps, loads
from tractor.model import field_ids
from tractor.pipeline_evaluation import evaluate_pipeline, infer
from tractor.types import TractorError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/biqwen-extract-decide-4b')
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = list(read_episodes(args.episodes))
    if not rows or any(row.split != 'dev' for row in rows):
        raise ValueError('Only Dev is permitted')
    rows = [row for row in rows if any(t.kind == 'records' for t in row.tasks)]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    backend = BiQwenExtractor(args.model, 'cuda', 'bf16')
    upstream_decode = type(backend.model)._decode_example
    captured = deque()

    def both(model, *args):
        captured.append(upstream_decode(model, *args))
        return decode_multiple(model, *args)

    backend.model._decode_example = MethodType(both, backend.model)

    class Upstream:
        def candidates(self, request):
            raw = captured.popleft()
            capacity = raw['capacity']
            if capacity['record_overflow'] or capacity['anchor_candidates_exhausted']:
                raise TractorError(422, 'output_capacity', 'Upstream reported output capacity')
            ids = field_ids(request)
            rows = raw['tables']['record'] if request.anchor else [{'fields': raw['fields']}]
            records = []
            for row in rows:
                record = {}
                for name, fid in ids.items():
                    if name == request.anchor:
                        pairs = [row['anchor']]
                    else:
                        values = row['fields'].get(fid, [])
                        if isinstance(values, dict):
                            values = [values]
                        pairs = [v['source'] for v in values if v['source'] is not None]
                    record[name] = [{'text': request.text[a:b], 'start': a, 'end': b} for a, b in pairs]
                records.append(record)
            return records

    predictions = {'upstream': {}, 'multiple': {}}
    with evaluation_precision(), (root / 'predictions.jsonl').open('w') as stream:
        for index, row in enumerate(rows):
            captured.clear()
            for label, model in [('multiple', backend), ('upstream', Upstream())]:
                try:
                    value = {'prediction': infer(model, row)}
                except Exception as exc:
                    value = {'error': type(exc).__name__, 'message': str(exc)[:400]}
                predictions[label][row.id] = value
            stream.write(dumps({'id': row.id, **{k: v[row.id] for k, v in predictions.items()}}) + '\n')
            stream.flush()
            if (index + 1) % 50 == 0:
                print(dumps({'completed': index + 1, 'total': len(rows)}), flush=True)
    policy = loads(Path('configs/extraction-quality.json').read_text())
    report = {k: evaluate_pipeline(rows, v, policy) for k, v in predictions.items()}
    report['scope'] = 'All Dev episodes containing records; identical neural states, unchanged weights; final pipeline values and issues.'
    report['documents'] = len(rows)
    (root / 'metrics.json').write_text(dumps(report, indent=2))
    print(dumps({'documents': len(rows), 'completed': True}), flush=True)


if __name__ == '__main__':
    main()
