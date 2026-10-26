"""Offline backend selection by local bundle format, independent of public API."""

from pathlib import Path


def load_extractor(path, device="cpu", precision="fp32"):
    if Path(path, "checkpoint.json").is_file():
        from .biqwen_training import load_checkpoint

        return load_checkpoint(path, device, precision)
    if Path(path, "model/extractor_config.json").is_file():
        from .biqwen import BiQwenExtractor

        return BiQwenExtractor(path, device, precision)
    from .model import Extractor

    return Extractor(path, device, precision)
