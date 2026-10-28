"""Optional presence/answerability heads on BiQwen field representations.

The heads read the field query and the source logits that one backbone pass
already produced. They add no schema input, so the existing schema version and
attention stay identical. Untrained heads are never released: a module is
`released` only after strict loading of a matching saved artifact. Development
initialization exists solely for an explicitly chosen training path.
"""

import hashlib
from pathlib import Path

import torch
from torch import nn

HEADS_VERSION = "tractor-field-uncertainty-heads-v1"
POOLED = 3  # max, mean of top-4 and log-count of positive source logits


def pooled_source(source_logits):
    logits = source_logits.float()
    if logits.numel() == 0:
        return torch.tensor([-20.0, -20.0, 0.0], device=logits.device)
    top = logits.topk(min(4, logits.numel())).values
    return torch.stack((logits.max(), top.mean(), torch.log1p((logits > 0).sum().float())))


class FieldUncertaintyHeads(nn.Module):
    """Two logits per field: source presence and typed answerability."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.body = nn.Sequential(nn.Linear(dim + POOLED, hidden), nn.GELU())
        self.presence = nn.Linear(hidden, 1)
        self.answerability = nn.Linear(hidden, 1)
        self.released = False

    def features(self, query, source_logits):
        pooled = pooled_source(source_logits).to(dtype=query.dtype)
        return torch.cat((self.norm(query), pooled), dim=-1)

    def forward(self, query, source_logits):
        hidden = self.body(self.features(query, source_logits))
        return {"presence_logit": self.presence(hidden).squeeze(-1),
                "answerability_logit": self.answerability(hidden).squeeze(-1)}


def initialize_for_training(dim, seed):
    """Development-only random initialization; never released by itself."""
    generator = torch.Generator().manual_seed(seed)
    heads = FieldUncertaintyHeads(dim)
    with torch.no_grad():
        for parameter in heads.parameters():
            if parameter.dim() > 1:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return heads


def describe(heads):
    return {"version": HEADS_VERSION, "dim": heads.dim,
            "parameters": {k: list(v.shape) for k, v in heads.state_dict().items()}}


def save_heads(heads, directory, binding, provenance):
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "uncertainty-heads.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in heads.state_dict().items()}, str(path))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {**describe(heads), "file": path.name, "sha256": digest, "binding": binding, "provenance": provenance}


def load_heads(directory, description, binding):
    """Strict: version, binding, hash, every parameter name and shape must match."""
    from safetensors.torch import load_file

    if description.get("version") != HEADS_VERSION:
        raise ValueError("Unsupported uncertainty head version")
    if description.get("binding") != binding:
        raise ValueError("Uncertainty heads belong to another base, adapter or schema")
    path = Path(directory) / description["file"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != description["sha256"]:
            raise ValueError("Uncertainty heads changed")
    tensors = load_file(str(path))
    heads = FieldUncertaintyHeads(description["dim"])
    expected = describe(heads)["parameters"]
    if expected != description["parameters"] or {k: list(v.shape) for k, v in tensors.items()} != expected:
        raise ValueError("Uncertainty head parameters differ from the declared architecture")
    heads.load_state_dict(tensors, strict=True)
    heads.eval()
    return heads


def attach(model, heads):
    """Attach verified heads; they only emit signals once explicitly released.

    Not registered as a submodule: extraction-head checkpoints, their strict
    key check and the existing state dict stay exactly as before.
    """
    object.__setattr__(model, "uncertainty_heads", heads)
    return model
