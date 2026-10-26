from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tractor.biqwen_acceleration import select_gdn


def test_kernel_selection_rejects_invalid_or_cpu_fla():
    backend = SimpleNamespace(device='cpu')
    with pytest.raises(ValueError, match='torch or fla'):
        select_gdn(backend, 'automatic')
    with pytest.raises(ValueError, match='CUDA'):
        select_gdn(backend, 'fla')


def test_torch_fla_torch_restores_real_cpu_reference(monkeypatch):
    pytest.importorskip('fla')
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    from tractor import biqwen_acceleration
    from tractor.biqwen import load_runtime

    path = Path('models/biqwen-extract-decide-4b').resolve()
    if not (path / 'bootstrap.py').exists():
        pytest.skip('Optional pinned runtime not present')
    for name in ['causal_conv1d_fn', 'causal_conv1d_update', 'torch_recurrent_gated_delta_rule',
                 'torch_chunk_gated_delta_rule']:
        monkeypatch.setattr(qwen, name, getattr(qwen, name))  # restore after test
    monkeypatch.setattr(biqwen_acceleration, '_BOOTSTRAPS', {})
    bootstrap, _, _ = load_runtime(path)
    bootstrap.configure_gdn('torch')
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, layer_types=['linear_attention'],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, use_cache=False,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': .5, 'mrope_section': [2,1,1]})
    backbone = Qwen3_5TextModel(config).eval()
    backend = SimpleNamespace(path=path, device='cuda', model=SimpleNamespace(backbone=backbone))
    select_gdn(backend, 'torch')
    with torch.no_grad():
        before = backbone(torch.tensor([[1, 2, 3]])).last_hidden_state
    # Binding is checked without executing a CUDA kernel on the CPU fixture.
    report = select_gdn(backend, 'fla')
    assert report['expected'] == 'fla'
    assert qwen.torch_chunk_gated_delta_rule.__tractor_backend__ == 'fla'
    select_gdn(backend, 'torch')
    with torch.no_grad():
        after = backbone(torch.tensor([[1, 2, 3]])).last_hidden_state
    assert torch.equal(before, after)
    assert backend.gdn_backend == 'torch'
