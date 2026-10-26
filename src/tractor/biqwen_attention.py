"""Instance-local, versioned bidirectional full attention for Qwen3.5 text."""

from types import MethodType

CAUSAL = 'qwen3.5-causal-v1'
BIDIRECTIONAL = 'qwen3.5-hybrid-bidirectional-v1'


def text_backbone(backbone):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    model = backbone.get_base_model() if hasattr(backbone, 'get_base_model') else backbone
    if not isinstance(model, Qwen3_5TextModel):
        raise ValueError('Bidirectional extraction requires the Qwen3.5 text backbone')
    return model


def bidirectional_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                          past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    from transformers.masking_utils import (
        create_bidirectional_mask,
        create_recurrent_attention_mask,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    if past_key_values is not None or use_cache:
        raise ValueError('Bidirectional extraction forbids incremental caches')
    if kwargs.get('is_causal') is not None:
        raise ValueError('The checkpoint fixes attention direction; callers cannot override it')
    if attention_mask is not None and (not hasattr(attention_mask, 'ndim') or attention_mask.ndim != 2):
        raise ValueError('Bidirectional extraction accepts only a 2D padding mask')
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError('Specify exactly one of input_ids or inputs_embeds')
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if attention_mask is not None:
        if tuple(attention_mask.shape) != tuple(inputs_embeds.shape[:2]):
            raise ValueError('Padding mask must match the complete input')
        if not ((attention_mask == 0) | (attention_mask == 1)).all() or not attention_mask.bool().any(dim=1).all():
            raise ValueError('Padding mask must be binary with at least one token per example')
    mask_args = {'config': self.config, 'inputs_embeds': inputs_embeds,
                 'attention_mask': attention_mask, 'past_key_values': None}
    masks = {'full_attention': create_bidirectional_mask(**mask_args),
             'linear_attention': create_recurrent_attention_mask(**mask_args)}
    # Use the library's original implementation. No global monkeypatch, copied
    # decoder loop, second backbone pass or changed recurrent-layer algorithm.
    return Qwen3_5TextModel.forward(self, inputs_embeds=inputs_embeds, attention_mask=masks,
                                   position_ids=position_ids, past_key_values=None, use_cache=False, **kwargs)


def install_attention(backbone, mode):
    if mode not in {CAUSAL, BIDIRECTIONAL}:
        raise ValueError('Unsupported BiQwen attention contract')
    current = getattr(backbone.config, 'tractor_attention_mode', CAUSAL)
    if mode == CAUSAL:
        if current != CAUSAL:
            raise ValueError('Cannot silently restore causal attention on an adapted backbone')
        return
    model = text_backbone(backbone)
    if current not in {CAUSAL, BIDIRECTIONAL}:
        raise ValueError('Unknown existing attention contract')
    if model.config._attn_implementation not in {'eager', 'sdpa'}:
        raise ValueError('Bidirectional extraction currently supports eager and SDPA attention')
    kinds = model.config.layer_types
    if (len(kinds) != len(model.layers) or 'full_attention' not in kinds
            or set(kinds) - {'full_attention', 'linear_attention'}):
        raise ValueError('Unsupported Qwen3.5 layer layout')
    for kind, layer in zip(kinds, model.layers, strict=True):
        if layer.block_type != kind:
            raise ValueError('Backbone layers disagree with their configuration')
    for kind, layer in zip(kinds, model.layers, strict=True):
        if kind == 'full_attention':
            layer.self_attn.is_causal = False
    model.config.use_cache = False
    model.config.tractor_attention_mode = BIDIRECTIONAL
    model.forward = MethodType(bidirectional_forward, model)


def attention_contract(backbone):
    mode = getattr(backbone.config, 'tractor_attention_mode', CAUSAL)
    if mode == CAUSAL:
        return {'mode': CAUSAL}
    model = text_backbone(backbone)
    if (mode != BIDIRECTIONAL or getattr(model.forward, '__func__', None) is not bidirectional_forward
            or any(layer.self_attn.is_causal for layer in model.layers if layer.block_type == 'full_attention')):
        raise ValueError('Backbone attention marker and implementation disagree')
    return {'mode': mode, 'bidirectional_full_attention_layers': model.config.layer_types.count('full_attention'),
            'causal_deltanet_layers': model.config.layer_types.count('linear_attention'),
            'cache': False}
