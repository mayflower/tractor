"""Keep field descriptions present during native training, as during inference."""

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
