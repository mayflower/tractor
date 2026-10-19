"""Measure actual schema/record target capacity with explicit non-training fixtures."""
from pathlib import Path

import torch

from tractor.document_sources import envelope
from tractor.generalist_data import validate_native
from tractor.jsonio import dumps
from tractor.model import Extractor

torch.set_num_threads(2)
extractor = Extractor('models/base', 'cpu')
results = []
for kind, counts in [('fields', [3, 8, 9, 20, 21, 40]), ('records', [1, 2, 5, 6, 20, 21])]:
    for count in counts:
        text = ''
        values = []
        for i in range(count):
            prefix = f'Field {i+1}: ' if kind == 'fields' else 'Item: '
            value = f'VALUE{i+1:03d}'
            start = len(text) + len(prefix)
            values.append({'status': 'present', 'items': [{'text': value, 'start': start, 'end': start + len(value)}]})
            text += prefix + value + '\n'
        fields = {f'field{i+1}': f'Literal text value of field {i+1}' for i in range(count)} if kind == 'fields' else {'item': 'Explicit item code'}
        task = {'id': 'document', 'kind': 'entities' if kind == 'fields' else 'records',
                'description': 'Capacity fixture, not a training document', 'fields': fields}
        if kind == 'records':
            task['anchor'] = 'item'
        gold = [dict(zip(fields, values, strict=True))] if kind == 'fields' else [{'item': value} for value in values]
        episode = envelope('capacity_fixture', f'{kind}:{count}', 'train', 'operations', 'en', text,
                           [task], {'document': {'complete': True, 'records': gold}})
        try:
            _, tokens = validate_native(episode, extractor)
            results.append({'axis': kind, 'count': count, 'native_targets_supported': True, 'joint_tokens': tokens})
        except Exception as exc:
            results.append({'axis': kind, 'count': count, 'native_targets_supported': False,
                            'reason': type(exc).__name__ + ': ' + str(exc)[:240]})
report = {'base': 'models/base', 'joint_token_limit': extractor.input_limit, 'cases': results,
          'qualification': 'Artificial capacity fixtures only, excluded from datasets. Checks actual tokenizer/collator targets, '
          'not learned extraction accuracy, realistic long-document support or latency.'}
path = Path('reports/extraction-build/native-capacity.json')
with path.open('x') as stream:
    stream.write(dumps(report, indent=2))
print(dumps(report, indent=2))
