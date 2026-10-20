from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tractor.api_training import validate_record_bindings
from tractor.document_training import regression_guard, selection_score
from tractor.inference_context import evaluation_precision
from tractor.jsonio import dumps, loads
from tractor.training_adapter import (
    configure_complete_batches,
    configure_decoder_only,
    configure_frozen_encoder,
    configure_record_loss,
)


def test_evaluation_precision_restores_trainer_settings_on_failure():
    initial = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        with pytest.raises(RuntimeError), evaluation_precision():
            assert torch.get_float32_matmul_precision() == 'highest'
            assert not torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32
            assert torch.is_inference_mode_enabled()
            raise RuntimeError('failed evaluation')
        assert torch.get_float32_matmul_precision() == 'high'
        assert torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.is_inference_mode_enabled()
    finally:
        torch.set_float32_matmul_precision(initial[0])
        torch.backends.cuda.matmul.allow_tf32 = initial[1]
        torch.backends.cudnn.allow_tf32 = initial[2]


def test_selection_score_survives_decimal_json_roundtrip():
    report = {'checks': {'field': {'score': .71}}, 'scores': {}, 'field_checks': {}}
    assert selection_score(report) == selection_score(loads(dumps(report)))


@dataclass(frozen=True)
class Settings:
    record_loss_weight: float = 1.
    record_field_threshold: float = .5


def test_record_weight_is_persisted_without_changing_inference_thresholds():
    model = SimpleNamespace(boundary_settings=Settings(), config=SimpleNamespace(
        boundary_head={'record_loss_weight': 1., 'record_field_threshold': .5}))
    configure_record_loss(model, 192)
    assert model.boundary_settings.record_loss_weight == 192.
    assert model.config.boundary_head == {'record_loss_weight': 192., 'record_field_threshold': .5}
    for invalid in (True, 0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            configure_record_loss(model, invalid)


def test_native_record_validation_detects_swaps_with_unchanged_unions():
    fields = [SimpleNamespace(name='f0'), SimpleNamespace(name='f1')]
    packed = torch.tensor([[[[[[0, 1]], [[1, 2]]], [[[2, 3]], [[3, 4]]]]]])
    batch = SimpleNamespace(
        targets=SimpleNamespace(record_targets=(packed, torch.ones(packed.shape[:-1], dtype=torch.bool),
                                                torch.ones((1, 1, 2), dtype=torch.bool))),
        record_specs=({0: SimpleNamespace(fields=fields)},),
        start_mappings=[[0, 10, 20, 30]], end_mappings=[[3, 13, 23, 33]])
    targets = [{'record': {'f0': [{'start': 0, 'end': 3}], 'f1': [{'start': 10, 'end': 13}]}},
               {'record': {'f0': [{'start': 20, 'end': 23}], 'f1': [{'start': 30, 'end': 33}]}}]
    validate_record_bindings(targets, batch)
    packed[0, 0, :, 1] = packed[0, 0, [1, 0], 1].clone()
    with pytest.raises(ValueError, match='field bindings'):
        validate_record_bindings(targets, batch)


def test_decoder_only_optimizer_preserves_encoder_and_disables_frozen_dropout():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.9))
            self.record_decoder = torch.nn.Linear(3, 1)

        def forward(self, value):
            return self.record_decoder(self.encoder(value))

    model = Model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    handle = configure_decoder_only(model)
    try:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model.train()
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        assert model.training and model.record_decoder.training
        assert not model.encoder.training
        for name, param in model.named_parameters():
            assert (not torch.equal(param, before[name])) == name.startswith('record_decoder.')
    finally:
        handle.remove()


def test_regression_guard_rejects_field_drop_even_when_group_average_improves():
    baseline = {'field_checks': {'reference': {'score': .62, 'status': 'fail'}},
                'rejection_checks': {'code': {'score': 1., 'support': 2, 'status': 'insufficient_evidence'}},
                'scores': {'domain:metadata': {'field_f1': .91}}}
    candidate = loads(dumps(baseline))
    assert regression_guard(candidate, baseline)['passed']
    candidate['scores']['domain:metadata']['field_f1'] = .95
    candidate['field_checks']['reference']['score'] = .42
    assert 'field_checks:reference' in regression_guard(candidate, baseline)['failures']
    candidate['field_checks']['reference']['score'] = .62
    candidate['rejection_checks']['code']['score'] = .5
    assert 'total_correct_rejections' in regression_guard(candidate, baseline)['failures']


def test_frozen_encoder_updates_both_heads_without_encoder_dropout_or_weight_changes():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.9))
            self.boundary_head = torch.nn.Linear(3, 3)
            self.record_decoder = torch.nn.Linear(3, 1)

        def forward(self, value):
            return self.record_decoder(self.boundary_head(self.encoder(value)))

    model = Model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    handle = configure_frozen_encoder(model)
    try:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model.train()
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        assert model.training and model.boundary_head.training and model.record_decoder.training
        assert not model.encoder.training
        for name, param in model.named_parameters():
            assert (not torch.equal(param, before[name])) == (not name.startswith('encoder.'))
            if name.startswith('encoder.'):
                assert param.grad is None
    finally:
        handle.remove()


def test_complete_epochs_keep_examples_in_partial_final_batch():
    from types import SimpleNamespace

    from torch.utils.data import DataLoader
    trainer = SimpleNamespace(
        is_distributed=False,
        _create_dataloader=lambda dataset, batch_size, **kw: DataLoader(
            dataset, batch_size=batch_size, shuffle=kw['shuffle'], drop_last=kw['is_training']),
    )
    evidence = configure_complete_batches(trainer)
    loader = trainer._create_dataloader(list(range(19)), 8, shuffle=False)
    batches = list(loader)
    assert [len(b) for b in batches] == [8, 8, 3]
    assert [int(n) for b in batches for n in b] == list(range(19))
    assert evidence['training_loader_sizes'] == [{'examples': 19, 'batches': 3}]


def test_learning_evidence_accepts_updated_head_when_first_parameter_is_inactive():
    from tractor.train import LearningEvidence

    model = torch.nn.Module()
    model.record_decoder = torch.nn.ParameterDict({
        'a_optional': torch.nn.Parameter(torch.ones(128)),
        'b_active': torch.nn.Parameter(torch.ones(128)),
    })
    evidence = LearningEvidence(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
    model.record_decoder['b_active'].sum().backward()
    optimizer.step()
    report = evidence.report()['record_decoder']
    assert report['learning_demonstrated']
    assert not report['parameters']['record_decoder.a_optional']['changed']
    assert report['parameters']['record_decoder.b_active']['changed']
    for handle in evidence.handles:
        handle.remove()


def test_learning_evidence_requires_gradient_and_update_in_same_parameter():
    from tractor.train import LearningEvidence

    model = torch.nn.Module()
    model.record_decoder = torch.nn.ParameterDict({
        'a_changed': torch.nn.Parameter(torch.ones(128)),
        'b_gradient': torch.nn.Parameter(torch.ones(128)),
    })
    evidence = LearningEvidence(model)
    model.record_decoder['b_gradient'].sum().backward()
    with torch.no_grad():
        model.record_decoder['a_changed'].add_(.01)
    report = evidence.report()['record_decoder']
    assert report['changed'] and report['nonzero_finite_gradient']
    assert not report['learning_demonstrated']
    for handle in evidence.handles:
        handle.remove()


@pytest.mark.parametrize('options, message', [
    ({'max_steps': 600, 'tractor_max_seconds': None}, 'complete epochs'),
    ({'early_stopping': True, 'tractor_max_seconds': None}, 'complete epochs'),
    ({'max_steps': -1}, 'wall-clock'),
])
def test_full_dataset_cannot_silently_become_a_bounded_pilot(tmp_path, monkeypatch, options, message):
    from tractor.train import train
    monkeypatch.setattr('tractor.training_corpus.verify_training_binding', lambda *args: {})
    path = tmp_path / 'config.json'
    path.write_text(dumps({'tractor_dataset': {'path': str(tmp_path)}, **options}))
    with pytest.raises(ValueError, match=message):
        train('no-model-should-be-loaded', tmp_path, path)
