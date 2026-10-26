from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from tractor.biqwen import BiQwenExtractor, decode_multiple, load_runtime
from tractor.biqwen_attention import BIDIRECTIONAL, CAUSAL, attention_contract, install_attention
from tractor.biqwen_training import (
    BIDIRECTIONAL_RECIPE,
    RECIPE,
    IndexedJSONL,
    configure_training_model,
    epoch_order,
    load_checkpoint,
    multiple_losses,
    record_coverage,
    save_checkpoint,
    update_batches,
    validate_config,
)
from tractor.biqwen_training_data import UnsuitableEvidence, encode_example, native_example
from tractor.jsonio import dumps, loads
from tractor.types import ExtractRequest


def test_every_epoch_covers_tail_and_resume_exactly_once():
    order = epoch_order(19, 42, 0)
    updates = list(update_batches(order, 2, 4))
    assert [sum(map(len, u)) for u in updates] == [8, 8, 3]
    assert Counter(i for u in updates for b in u for i in b) == Counter(range(19))
    coverage = bytearray(19)
    record_coverage(coverage, [i for b in updates[0] for i in b])
    for update in update_batches(order, 2, 4, cursor=8):
        record_coverage(coverage, [i for b in update for i in b])
    assert all(v == 1 for v in coverage)
    with pytest.raises(ValueError, match='repeated'):
        record_coverage(coverage, [order[0]])
    assert epoch_order(19, 42, 1) != order


@pytest.mark.parametrize(('key', 'value'), [('max_steps', 4000), ('sampling', True),
                                          ('quotas', True), ('validation_limit', 256),
                                          ('early_stopping', True)])
def test_shortcuts_cannot_be_called_full_training(key, value):
    config = loads(Path('configs/biqwen-full.json').read_text())
    validate_config(config)
    config[key] = value
    with pytest.raises(ValueError, match='forbids'):
        validate_config(config)


def record_request():
    return ExtractRequest(text='Alpha: 10 cm, 12 cm. Beta: 20 cm.', many=True, anchor='name', fields={
        'name': {'type': 'text', 'description': 'Product'},
        'width': {'type': 'quantity', 'description': 'Width', 'unit': 'mm'},
    })


def targets(request):
    return [
        {'name': [{'start': 0, 'end': 5, 'text': 'Alpha'}],
         'width': [{'start': 7, 'end': 12, 'text': '10 cm'}, {'start': 14, 'end': 19, 'text': '12 cm'}]},
        {'name': [{'start': 21, 'end': 25, 'text': 'Beta'}],
         'width': [{'start': 27, 'end': 32, 'text': '20 cm'}]},
    ]


def test_training_preserves_competing_values_and_exclusive_occurrences():
    request = record_request()
    records = targets(request)
    example = native_example(request, records)
    assert example['schema']['fields'][1]['description'].startswith('Width\nType: quantity;')
    assert example['evidence']['records'][0]['fields']['f1'] == [(7, 12), (14, 19)]
    records[1]['width'].append(records[0]['width'][0])
    with pytest.raises(UnsuitableEvidence, match='shared_record_occurrence'):
        native_example(request, records)


def test_assignment_loss_trains_both_positives_and_rejects_wrong_owner():
    logits = torch.tensor([0., -1., -1., 1.], requires_grad=True)
    source = torch.tensor([0., 0., 0.], requires_grad=True)
    fields = [{'id': 'a', 'kind': 'anchor'}, {'id': 'v', 'kind': 'span'}]
    pools = {
        'a': {'pairs': [(0, 1)], 'gold_source': {(0, 1)}, 'index': {(0, 1): 0},
              'states': torch.zeros(1, 2), 'source_logits': source[:1]},
        'v': {'pairs': [(1, 2), (2, 3), (3, 4)], 'gold_source': {(1, 2), (2, 3)},
              'index': {(1, 2): 0, (2, 3): 1, (3, 4): 2}, 'source_logits': source},
    }
    meta = {'schema': {'fields': fields, 'choices': [], 'tables': [
        {'id': 'record', 'anchor_field_id': 'a', 'field_ids': ['v']}]},
        'gold': {'records': [{'table_id': 'record', 'anchor': (0, 1),
                              'fields': {'v': [(1, 2), (2, 3)]}}]}}
    model = SimpleNamespace(device='cpu', _boundary_loss=lambda *args: source.sum() * 0,
                            _assignment_logits=lambda *args: logits)
    totals = {k: [] for k in ['boundary', 'source', 'assignment', 'choice', 'reference']}
    multiple_losses(model, meta, pools, torch.zeros(2, 2), None, None, 0, None, totals, Counter())
    assert len(totals['assignment']) == 1
    totals['assignment'][0].backward()
    assert logits.grad[1] < 0 and logits.grad[2] < 0 and logits.grad[3] > 0
    assert not totals['choice'] and not totals['reference']


def test_indexed_corpus_keeps_utf8_offsets_and_all_rows(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text(''.join(dumps({'text': t}) + '\n' for t in ['Müller', '東京', 'end']))
    dataset = IndexedJSONL(path)
    assert len(dataset) == 3
    assert [dataset[i]['text'] for i in [2, 0, 1]] == ['end', 'Müller', '東京']
    dataset.close()


@pytest.mark.parametrize('attention_mode', [CAUSAL, BIDIRECTIONAL])
def test_cpu_training_checkpoint_roundtrip_and_multi_value_gradient(tmp_path, monkeypatch, attention_mode):
    # Real pinned extraction code and tiny random Qwen layers: software test only.
    path = Path('models/biqwen-extract-decide-4b').resolve()
    if not (path / 'tractor_qwen/model.py').exists():
        pytest.skip('Optional pinned BiQwen runtime is not downloaded')
    from peft import LoraConfig, TaskType, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    from tractor import biqwen_training

    bootstrap, runtime, encoding = load_runtime(path)
    bootstrap.configure_gdn('torch')
    request = record_request()
    raw = native_example(request, targets(request))
    tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator([request.text, dumps(raw['schema'])],
                                 trainers.WordLevelTrainer(special_tokens=['[PAD]', '[UNK]']))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]')
    limits = {'base_model': 'tiny-random-qwen3.5', 'revision': None, 'head_dim': 32,
              'adapter_heads': 4, 'dropout': 0., 'max_length': 512, 'schema_prefix': True,
              'max_span_tokens': 48, 'candidate_top_k': 48, 'max_records': 16, 'max_gold_spans': 64}
    tiny = runtime.build_tiny_model(tokenizer, config=runtime.ExtractorConfig(**limits))
    base = BiQwenExtractor.__new__(BiQwenExtractor)
    base.model, base.encoding, base.input_limit = tiny, encoding, 512
    base.model_revision, base.weights_hashes = 'fixture', {'base': 'fixture-hash'}
    def fresh():
        result = BiQwenExtractor.__new__(BiQwenExtractor)
        result.__dict__.update(base.__dict__)
        result.model = deepcopy(base.model)
        return result

    backend = fresh()
    backend.model.backbone = get_peft_model(backend.model.backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, target_modules=['q_proj', 'v_proj'], r=4, lora_alpha=8))
    model = configure_training_model(backend)
    install_attention(model.backbone, attention_mode)
    model._decode_example = MethodType(decode_multiple, model)
    model.train()
    encoded = encode_example(raw, encoding, tokenizer, limits)
    assert len(encoded['gold']['records'][0]['fields']['f1']) == 2
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    output = model(encoding.collate([encoded], tokenizer.pad_token_id), decode=False)
    assert torch.isfinite(output['loss']) and output['diagnostics']['backbone_forward_passes'] == 1
    output['loss'].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.assignment.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for n, p in model.named_parameters() if 'lora_' in n)
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith('choice_classifier.'))
    optimizer.step()
    # Observing proposals must leave production decoding unchanged and never
    # pass the training gold into inference.
    from tractor.biqwen_diagnostics import EvidenceTrace
    model.eval()
    expected = backend.candidates(request)
    with EvidenceTrace(backend) as trace:
        assert trace.candidates(request) == expected
        assert next(iter(trace.requests.values()))['fields']
    assert backend.candidates(request) == expected
    state = {'step': 1, 'epoch': 0, 'cursor': 1, 'coverage': bytearray([1, 0]), 'best_metric': None}
    checkpoint_config = {'model': str(path), 'recipe': RECIPE}
    if attention_mode == BIDIRECTIONAL:
        checkpoint_config.update(recipe=BIDIRECTIONAL_RECIPE, attention_mode=attention_mode)
    pointer = save_checkpoint(tmp_path, backend, optimizer, state, {'fixture': True}, checkpoint_config)
    checkpoint = tmp_path / pointer['checkpoint']
    saved = torch.load(checkpoint / 'training-state.pt', map_location='cpu', weights_only=True)
    assert saved['coverage'] == state['coverage'] and saved['cursor'] == 1
    monkeypatch.setattr(biqwen_training, 'BiQwenExtractor', lambda *args: fresh())
    restored = load_checkpoint(checkpoint, trainable=True)
    assert attention_contract(restored.model.backbone) == attention_contract(model.backbone)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name]), name
    restored.model.eval()
    assert restored.candidates(request) == expected
    (checkpoint / 'heads.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='artifact changed'):
        load_checkpoint(checkpoint)

    # Exercise the actual epoch runner: interruption after four of five requests,
    # exact resume of the remainder, then a complete second epoch. These are tiny
    # random CPU fixtures, never a shortened run of the actual training corpus.
    from tractor.document_sources import envelope

    config = loads(Path('configs/biqwen-full.json').read_text())
    if attention_mode == BIDIRECTIONAL:
        config.update(recipe=BIDIRECTIONAL_RECIPE, attention_mode=attention_mode)
    work = tmp_path / 'runner'
    (work / 'configs').mkdir(parents=True)
    (work / 'data').mkdir()
    (work / 'base/model').mkdir(parents=True)
    (work / 'base/model/extractor_config.json').write_text(dumps({'config': limits}))
    (work / 'data/train.native.jsonl').write_text(''.join(dumps(raw) + '\n' for _ in range(5)))
    (work / 'data/train.identities.jsonl').write_text(''.join(dumps({'id': i}) + '\n' for i in range(5)))
    dev = envelope('tractor_authored_documents', 'dev-fixture', 'dev', 'products', 'en', 'Ada',
                   [{'id': 'document', 'kind': 'entities', 'description': 'Name', 'fields': {'name': 'Name'}}],
                   {'document': {'complete': True, 'records': [{'name': {'status': 'present',
                     'items': [{'start': 0, 'end': 3, 'text': 'Ada'}]}}]}})
    (work / 'data/dev.episodes.jsonl').write_text(dumps(dev.model_dump()) + '\n')
    (work / 'policy.json').write_text('{}')
    (work / 'baseline.json').write_text('{}')
    config.update(model='base', dataset={'path': 'data'}, output_dir='run', num_epochs=2,
                  batch_size=2, gradient_accumulation=2, checkpoint_updates=1,
                  gradient_checkpointing=False, policy='policy.json', baseline='baseline.json')
    (work / 'configs/run.json').write_text(dumps(config))
    (work / 'configs/active-training.json').write_text(dumps({
        'backend': 'biqwen', 'binding': config['dataset'], 'training_config': 'configs/run.json'}))
    plan = {'native_requests_per_epoch': 5, 'full_dev_episodes': 1, 'total_optimizer_updates': 4,
            'dataset_files': {}, 'source_sha256': {}}
    monkeypatch.setattr(biqwen_training, 'prepare_plan', lambda _: plan)
    evaluations = []

    def measure(backend, rows, policy, output):
        evaluations.append(len(rows))
        return {'fixture_score': .8}

    monkeypatch.setattr(biqwen_training, 'evaluate_dev', measure)
    monkeypatch.setattr(biqwen_training, 'regression_guard', lambda *args: {'passed': True, 'failures': {}})
    monkeypatch.setattr(biqwen_training, 'selection_score', lambda report: report['fixture_score'])
    calls = []

    def interrupt_once(*args):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError('simulated interruption')
        return encode_example(*args)

    monkeypatch.setattr(biqwen_training, 'encode_example', interrupt_once)
    monkeypatch.chdir(work)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        biqwen_training.train(config)
    assert not loads(Path('run/status.json').read_text())['training_complete']
    assert not Path('run/best.json').exists()  # A partial epoch cannot become best.
    biqwen_training.train(config, resume=True)
    status = loads(Path('run/status.json').read_text())
    assert status['training_complete'] and status['step'] == 4 and status['epoch'] == 2
    assert evaluations == [1, 1]
    for epoch in [1, 2]:
        coverage = loads(Path(f'run/epoch-{epoch}-coverage.json').read_text())
        assert coverage['seen_requests'] == 5 and coverage['all_requests_seen_exactly_once']
        diagnostics = loads(Path(f'run/epoch-{epoch}-diagnostics.json').read_text())
        assert diagnostics['all']['requests'] == 5
