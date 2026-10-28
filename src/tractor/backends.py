"""Offline backend selection by local bundle format, independent of public API."""

from pathlib import Path


def load_extractor(path, device="cpu", precision="fp32"):
    from .inference_bundle import BIQWEN_MARKERS, MANIFEST

    if Path(path, MANIFEST).is_file():
        from .inference_bundle import load_inference_bundle

        return load_inference_bundle(path, device, precision)
    if Path(path, "checkpoint.json").is_file():
        from .biqwen_training import load_checkpoint

        return load_checkpoint(path, device, precision)
    if Path(path, "model/extractor_config.json").is_file():
        from .biqwen import BiQwenExtractor

        return BiQwenExtractor(path, device, precision)
    if any(Path(path, marker).exists() for marker in BIQWEN_MARKERS):
        # A defective BiQwen layout must fail, not silently load as GLiNER.
        raise ValueError(f"Incomplete BiQwen bundle: {path}")
    from .model import Extractor

    return Extractor(path, device, precision)
