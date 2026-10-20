"""Keep field descriptions present during native training, as during inference."""

import math
from dataclasses import replace

from gliner2.processor import SamplingConfig, SchemaTransformer


class DescribedProcessor(SchemaTransformer):
    def _transform_schema(self, *args, **kwargs):
        if kwargs.get("label_descriptions"):
            kwargs["example_mode"] = "descriptions"
        return super()._transform_schema(*args, **kwargs)


def configure_processor(processor):
    # No architecture or optimizer changes. Upstream randomly omits descriptions;
    # f0/f1 have no stable semantics in this product, so omission would corrupt labels.
    processor.__class__ = DescribedProcessor
    processor.sampling_config = SamplingConfig(
        remove_json_structure_prob=0,
        remove_json_field_prob=0,
        synthetic_entity_label_prob=0,
        remove_relations_prob=0,
        remove_classification_label_prob=0,
        synthetic_label_prob=0,
    )
    return processor


def configure_record_loss(model, weight):
    """Use the native record objective; persist its explicit training weight."""
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('Record loss weight must be finite and positive')
    model.boundary_settings = replace(model.boundary_settings, record_loss_weight=float(weight))
    model.config.boundary_head = dict(model.config.boundary_head, record_loss_weight=float(weight))


def configure_decoder_only(model):
    """Freeze every parameter except the native record decoder, before optimizer creation."""
    names = [name for name, _ in model.named_parameters() if name.startswith('record_decoder.')]
    if not names:
        raise ValueError('Decoder-only training requires a native record_decoder')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('record_decoder.'))
    # Keep encoder dropout disabled even when the trainer calls model.train().
    # The root and boundary head remain in training mode for native gold proposal
    # injection. eval() on the boundary head would silently disable that path.
    def keep_frozen_modules_in_eval(module, _inputs):
        module.encoder.eval()
    handle = model.register_forward_pre_hook(keep_frozen_modules_in_eval)
    keep_frozen_modules_in_eval(model, None)
    return handle
