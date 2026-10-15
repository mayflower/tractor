"""Check native imports and tensor execution without downloading model weights."""

import importlib.metadata
import sys

import babel
import fastapi
import pint
import sentencepiece
import simplejson
import torch
from gliner2 import AutoExtractor
from gliner2.training import ExtractorTrainer, TrainingConfig

print(f"Python: {sys.version.split()[0]}")
for name in ("gliner2", "torch", "transformers", "fastapi", "pint"):
    print(f"{name}: {importlib.metadata.version(name)}")
assert all((babel, fastapi, pint, sentencepiece, simplejson))
assert all((AutoExtractor, ExtractorTrainer, TrainingConfig))
for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    tensor = torch.ones((16, 16), device=device)
    assert (tensor @ tensor).sum().item() == 4096
    print(f"{device}: tensor operation OK")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}; CUDA runtime: {torch.version.cuda}")
else:
    print("CUDA unavailable; CPU environment only.")
print("Native imports OK. Model inference and training have not been tested.")
