"""Real 4B forward/backward feasibility checks, without optimizer or latency tests."""

import argparse
import gc
import hashlib
from pathlib import Path

from .biqwen import TYPED_SCHEMA, BiQwenExtractor
from .biqwen_architecture import ARCHITECTURE
from .biqwen_training import TARGETS, configure_training_model
from .biqwen_training_data import encode_example, native_example
from .coverage_assembly import sha
from .jsonio import dumps, loads
from .types import ExtractRequest


def parameter_hashes(model):
    """Includes frozen backbone and trainable parameters; no optimizer is created."""
    import torch

    return {name: hashlib.sha256(p.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}


def stress_examples(backend):
    """Technical fixtures only, never written into a training or evaluation split."""
    def example(text, fields, targets, anchor=None):
        request = ExtractRequest(text=text, fields=fields, anchor=anchor, many=bool(anchor))
        records = []
        for row in targets:
            records.append({name: [{'start': text.index(value), 'end': text.index(value) + len(value),
                                    'text': value} for value in values] for name, values in row.items()})
        return native_example(request, records)

    quantity = {'width': {'type': 'quantity', 'description': 'Width / Breite', 'unit': 'mm'}}
    yield 'short_de', example('Die Breite beträgt 12 cm.', quantity, [{'width': ['12 cm']}])
    yield 'short_en', example('The width is 12 cm.', quantity, [{'width': ['12 cm']}])
    fields = {'name': {'type': 'person', 'description': 'Contact person'},
              'phone': {'type': 'phone', 'description': 'Phone of this contact', 'country': 'DE'}}
    text = ('Anna Müller ist unter +49 30 123456 erreichbar. '
            'Für Bill Smith ist +49 40 987654 angegeben; alternativ +49 40 987655.')
    yield 'records_de', example(text, fields, [
        {'name': ['Anna Müller'], 'phone': ['+49 30 123456']},
        {'name': ['Bill Smith'], 'phone': ['+49 40 987654', '+49 40 987655']}], 'name')
    names = ['Anna Adler', 'Bill Berg', 'Clara Conrad', 'David Dorn', 'Eva Engel',
             'Frank Fuchs', 'Greta Graf', 'Hans Hahn', 'Ida Ilse', 'Jan Jung',
             'Klara Kern', 'Lars Lang', 'Mara Maas', 'Nils Neu', 'Olga Ort', 'Paul Peters']
    records = [{'name': [name], 'phone': [f'+49 30 1234{i:04d}']} for i, name in enumerate(names)]
    text = ' '.join(f"{r['name'][0]} can be reached at {r['phone'][0]}." for r in records)
    yield 'records_16_en', example(text, fields, records, 'name')
    values = [f'REF-{i:04d}' for i in range(47)]
    yield 'evidence_47_en', example('The stated references are ' + ', '.join(values) + '.',
        {'ref': {'type': 'identifier', 'description': 'Explicit reference', 'cardinality': 'unique_list'}},
        [{'ref': values}])

    for label, fields, targets in [
        ('context_4096', quantity, {'width': ['12 cm']}),
        ('fields_32_context_4096', {f'f{i}': {'type': 'text', 'description': f'Explicit reference {i}'}
                                  for i in range(32)}, {f'f{i}': ['12 cm'] for i in range(32)}),
    ]:
        # Each added ASCII word has a single token with this pinned tokenizer.
        # Binary search checks actual joint schema/document lengths, not chars.
        lo, hi = 0, backend.input_limit
        while lo < hi:
            mid = (lo + hi + 1) // 2
            raw = example('Width: 12 cm.' + ' text' * mid, fields, [targets])
            try:
                backend.encoding.encode({'text': raw['text'], 'schema': raw['schema']},
                                        backend.model.tokenizer, backend.input_limit,
                                        schema_prefix=backend.model.config.schema_prefix)
                lo = mid
            except backend.encoding.CapacityError:
                hi = mid - 1
        yield label, example('Width: 12 cm.' + ' text' * lo, fields, [targets])


def run(config, output, batch_size=None):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    batch_size = batch_size or config['batch_size']
    if batch_size < 1:
        raise ValueError('Positive microbatch required')
    report = {'status': 'running', 'optimizer_updates': 0, 'latency_measured': False,
              'batch_size': batch_size, 'architecture': ARCHITECTURE, 'schema_version': TYPED_SCHEMA,
              'config': config, 'cases': [], 'code_sha256': {
                  str(p): sha(p) for p in sorted(Path(__file__).parent.glob('biqwen*.py'))}}

    def persist():
        (root / 'report.json').write_text(dumps(report, indent=2) + '\n')

    persist()
    try:
        torch.manual_seed(config['seed'])
        backend = BiQwenExtractor(config['model'], 'cuda', 'bf16')
        backend.model.backbone = get_peft_model(backend.model.backbone, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=float(config['lora_dropout']), target_modules=TARGETS, bias='none'))
        model = configure_training_model(backend, config['loss_weights'])
        if config['gradient_checkpointing']:
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.backbone.enable_input_require_grads()
        model.backbone.config.use_cache = False
        model.train()
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        report['weights_sha256'] = backend.weights_hashes
        report['device'] = torch.cuda.get_device_name()
        before = parameter_hashes(model)
        limits = loads(Path(config['model'], 'model/extractor_config.json').read_text())['config']
        for label, raw in stress_examples(backend):
            model.zero_grad(set_to_none=True)
            model.training_observations = []
            encoded = encode_example(raw, backend.encoding, model.tokenizer, limits)
            if '4096' in label and len(encoded['input_ids']) != backend.input_limit:
                raise ValueError('Stress case does not reach the supported input limit')
            torch.cuda.reset_peak_memory_stats()
            result = model(backend.encoding.collate([encoded] * batch_size, model.tokenizer.pad_token_id), decode=False)
            result['loss'].backward()
            nonzero = []
            for name, p in model.named_parameters():
                if not p.requires_grad and p.grad is not None:
                    raise ValueError(f'Frozen parameter received gradient: {name}')
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError(f'Non-finite gradient: {name}')
                    if torch.count_nonzero(p.grad):
                        nonzero.append(name)
            if not any('lora_' in n for n in nonzero) or not any(n.startswith('boundaries.') for n in nonzero):
                raise ValueError('Backbone adapters and boundary heads must receive gradients')
            if raw['schema']['tables'] and not any(n.startswith('assignment.interaction.') for n in nonzero):
                raise ValueError('Joint record scorer received no gradient')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                  float(config['max_grad_norm']), error_if_nonfinite=True)
            report['cases'].append({'case': label, 'tokens': len(encoded['input_ids']),
                                    'loss': float(result['loss'].detach()), 'gradient_norm': float(norm),
                                    'nonzero_gradient_parameters': nonzero,
                                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            persist()
            print(dumps({'case': label, 'passed': True, 'tokens': len(encoded['input_ids']),
                         'peak_allocated_bytes': torch.cuda.max_memory_allocated()}), flush=True)
            del result
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        report['all_parameters_unchanged'] = before == parameter_hashes(model)
        if not report['all_parameters_unchanged']:
            raise ValueError('Preflight changed weights')
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__, message=str(exc))
        raise
    finally:
        persist()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/biqwen-full.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    run(loads(Path(args.config).read_text()), args.output, args.batch_size)
