from copy import deepcopy

import pytest
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from tractor.biqwen_attention import BIDIRECTIONAL, CAUSAL, attention_contract, install_attention


def tiny(implementation='sdpa'):
    torch.manual_seed(19)
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=['linear_attention'] * 3 + ['full_attention'],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        pad_token_id=0, use_cache=False, attn_implementation=implementation,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': .5, 'mrope_section': [2, 1, 1]},
    )
    return Qwen3_5TextModel(config).eval()


@pytest.mark.parametrize('implementation', ['sdpa', 'eager'])
def test_future_context_reaches_backbone_without_changing_weights_or_other_instances(implementation):
    causal = tiny(implementation)
    model = deepcopy(causal)
    install_attention(model, BIDIRECTIONAL)
    install_attention(model, BIDIRECTIONAL)  # idempotent, no stacked wrappers
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in causal.state_dict().items())
    assert attention_contract(causal) == {'mode': CAUSAL}
    assert attention_contract(model) == {'mode': BIDIRECTIONAL,
        'bidirectional_full_attention_layers': 1, 'causal_deltanet_layers': 3, 'cache': False}
    a, b = torch.tensor([[2, 3, 4, 5]]), torch.tensor([[2, 3, 4, 6]])
    with torch.inference_mode():
        assert torch.equal(causal(a).last_hidden_state[:, 0], causal(b).last_hidden_state[:, 0])
        assert not torch.allclose(model(a).last_hidden_state[:, 0], model(b).last_hidden_state[:, 0])
        assert model(a).past_key_values is None
    # A loss at the first position can train representations of later tokens.
    embeds = model.embed_tokens(a).detach().requires_grad_()
    model(inputs_embeds=embeds).last_hidden_state[0, 0, 0].backward()
    assert torch.isfinite(embeds.grad).all() and torch.count_nonzero(embeds.grad[0, -1])
    with pytest.raises(ValueError, match='silently restore'):
        install_attention(model, CAUSAL)


@pytest.mark.parametrize('implementation', ['sdpa', 'eager'])
@pytest.mark.parametrize('padding', ['left', 'right'])
def test_padding_and_other_examples_cannot_leak_into_valid_tokens(implementation, padding):
    model = tiny(implementation)
    install_attention(model, BIDIRECTIONAL)
    ids = torch.tensor([[2, 3, 4, 0, 0], [7, 8, 9, 10, 11]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    if padding == 'left':
        ids[0], mask[0] = ids[0].roll(2), mask[0].roll(2)
    changed = ids.clone()
    changed[0, ~mask[0].bool()] = 17
    changed[1] = 21
    with torch.inference_mode():
        first = model(ids, attention_mask=mask).last_hidden_state
        second = model(changed, attention_mask=mask).last_hidden_state
    assert torch.allclose(first[0, mask[0].bool()], second[0, mask[0].bool()], atol=1e-6)


def test_gradient_checkpointing_and_cache_rejection():
    model = tiny().train()
    install_attention(model, BIDIRECTIONAL)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model(torch.tensor([[2, 3, 4]])).last_hidden_state[0, 0, 0].backward()
    assert model.layers[-1].self_attn.q_proj.weight.grad is not None
    for args in [{'use_cache': True}, {'past_key_values': object()}, {'is_causal': True}]:
        with pytest.raises(ValueError):
            model(torch.tensor([[2, 3]]), **args)
    for mask in [torch.zeros(1, 2), torch.ones(1, 1, 2, 2), {'full_attention': None}, torch.full((1, 2), .5)]:
        with pytest.raises(ValueError, match='mask'):
            model(torch.tensor([[2, 3]]), attention_mask=mask)


def test_unknown_modes_and_backbones_fail():
    with pytest.raises(ValueError, match='Unsupported'):
        install_attention(tiny(), 'automatic')
    with pytest.raises(ValueError, match='Qwen3.5'):
        install_attention(type('Other', (), {'config': type('Config', (), {})()})(), BIDIRECTIONAL)
