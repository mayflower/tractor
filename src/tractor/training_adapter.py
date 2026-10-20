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


def configure_frozen_encoder(model):
    """Train extraction heads jointly while keeping every encoder weight fixed."""
    if not any(name.startswith('encoder.') for name, _ in model.named_parameters()):
        raise ValueError('Frozen-encoder training requires an encoder')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith('encoder.'))

    def keep_encoder_in_eval(module, _inputs):
        module.encoder.eval()

    handle = model.register_forward_pre_hook(keep_encoder_in_eval)
    keep_encoder_in_eval(model, None)
    return handle


def configure_complete_batches(trainer):
    """Retain the final partial batch instead of dropping source examples."""
    from torch.utils.data import DataLoader

    if trainer.is_distributed:
        raise ValueError('Complete-epoch adapter currently requires single-device training')
    original = trainer._create_dataloader
    evidence = {'drop_last': False, 'training_loader_sizes': []}

    def create(dataset, batch_size, shuffle=True, is_training=True):
        loader = original(dataset, batch_size, shuffle=shuffle, is_training=is_training)
        if not is_training:
            return loader
        # Preserve the original sampler (including length grouping) and collator.
        complete = DataLoader(
            dataset, batch_size=loader.batch_size, sampler=loader.sampler,
            drop_last=False, num_workers=loader.num_workers, collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory, timeout=loader.timeout,
            worker_init_fn=loader.worker_init_fn, generator=loader.generator,
            prefetch_factor=loader.prefetch_factor, persistent_workers=loader.persistent_workers,
            multiprocessing_context=loader.multiprocessing_context,
        )
        evidence['training_loader_sizes'].append({'examples': len(dataset), 'batches': len(complete)})
        return complete

    trainer._create_dataloader = create
    return evidence
