"""Versioned Tractor heads layered on the immutable merged BiQwen base."""

import math
from types import MethodType

import torch
from torch import nn

from .biqwen import TYPED_SCHEMA

ARCHITECTURE = 'biqwen-joint-record-v2'
LEGACY_ARCHITECTURE = 'biqwen-additive-record-v1'


class JointFieldAssignment(nn.Module):
    """Preserve old projections and learn a nonseparable anchor/field residual."""

    def __init__(self, previous):
        super().__init__()
        self.anchor_proj = previous.anchor_proj
        self.field_proj = previous.field_proj
        d = self.anchor_proj.out_features
        self.interaction = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.interaction.to(device=self.anchor_proj.weight.device, dtype=self.anchor_proj.weight.dtype)
        # Exact legacy initialization; the output projection gets gradients on
        # the first update, opening the upstream residual on subsequent updates.
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)

    def condition(self, anchor, field):
        a, f = self.anchor_proj(anchor), self.field_proj(field)
        return a + f + self.interaction(torch.cat((a, f, a * f), dim=-1))


def assignment_logits(self, anchor, query, pool):
    condition = self.assignment.condition(anchor, query)
    scale = math.sqrt(self.config.head_dim)
    null = (condition * self.absent_embedding).sum().view(1) / scale
    scores = (pool['states'] * condition).sum(-1) / scale + pool['source_logits']
    return torch.cat((null, scores))


def install_architecture(backend, version=ARCHITECTURE):
    if version != ARCHITECTURE:
        raise ValueError('Unsupported BiQwen architecture')
    model = backend.model
    current = getattr(model, 'tractor_architecture', LEGACY_ARCHITECTURE)
    if current == ARCHITECTURE:
        if not isinstance(model.assignment, JointFieldAssignment):
            raise ValueError('Architecture marker and head disagree')
    elif current == LEGACY_ARCHITECTURE:
        model.assignment = JointFieldAssignment(model.assignment)
        model._assignment_logits = MethodType(assignment_logits, model)
    else:
        raise ValueError('Cannot replace an unknown architecture')
    model.tractor_architecture = ARCHITECTURE
    backend.schema_version = TYPED_SCHEMA
    return model
