"""Per-example offline reuse of threshold-independent native neural outputs.

The encoder/schema/head inputs are identical across threshold settings. Native
candidate filtering and record decoding still execute for every setting. This
context is never used by serving or benchmarking.
"""

from contextlib import contextmanager


@contextmanager
def reuse_neural_outputs(extractor):
    replacements = []
    for owner, name in [
        (extractor.model, "_encode_core"),
        (extractor.model.boundary_head, "forward"),
        (extractor.model.record_decoder, "forward_groups_dense"),
    ]:
        original = getattr(owner, name)
        cache = []

        def cached(*args, _original=original, _cache=cache, **kwargs):
            if not _cache:
                _cache.append(_original(*args, **kwargs))
            return _cache[0]

        setattr(owner, name, cached)
        replacements.append((owner, name, original))
    try:
        yield
    finally:
        for owner, name, original in replacements:
            setattr(owner, name, original)
