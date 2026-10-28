"""Field-conditioned distribution head over explicit numeric grids (prototype, disabled).

One small head scores every grid point of every estimate field from the field
representation of the existing pass; there is no backbone call per point,
quantile or field. Masses for the in-range grid and two explicit tail states
come from one softmax, so all statistics derive from the same distribution.
No release without trained, verified weights and matching calibration.
"""

import torch
from torch import nn

HEAD_VERSION = "tractor-numeric-distribution-v1"
BELOW, ABOVE = 0, 1  # tail state indices before the grid points


class GridDistributionHead(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.field = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.point = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.tails = nn.Linear(hidden, 2)
        self.released = False

    def forward(self, fields, coordinates, mask):
        """fields [B, D]; coordinates/mask [B, P] (padded). Returns logits [B, 2 + P]."""
        context = self.field(fields)
        points = self.point(coordinates.unsqueeze(-1).to(context.dtype))
        logits = torch.einsum("bh,bph->bp", context, points) / context.shape[-1] ** 0.5
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.cat((self.tails(context), logits), dim=-1)


def masses(logits):
    return torch.softmax(logits.float(), dim=-1)


def normalized_coordinates(values):
    """Map one field's grid coordinates to [0, 1] for the point encoder."""
    low, high = min(values), max(values)
    return [(v - low) / (high - low) for v in values]


def pad(grids):
    """Different grid lengths in one batch; mask marks real points."""
    width = max(len(g) for g in grids)
    coordinates = torch.zeros(len(grids), width)
    mask = torch.zeros(len(grids), width, dtype=torch.bool)
    for row, grid in enumerate(grids):
        coordinates[row, :len(grid)] = torch.tensor(normalized_coordinates([float(v) for v in grid]))
        mask[row, :len(grid)] = True
    return coordinates, mask


def target_distribution(grid, value, width):
    """Two-hot target between neighboring points; outside bounds goes to a tail state.

    The target value never enters the inference path.
    """
    target = torch.zeros(2 + width)
    values = [float(v) for v in grid]
    value = float(value)
    if value < values[0]:
        target[BELOW] = 1
    elif value > values[-1]:
        target[ABOVE] = 1
    else:
        for index in range(len(values) - 1):
            low, high = values[index], values[index + 1]
            if low <= value <= high:
                share = 0.0 if high == low else (value - low) / (high - low)
                target[2 + index] = 1 - share
                target[2 + index + 1] += share
                break
        else:
            target[2] = 1
    return target


def loss(logits, targets, known):
    """Cross-entropy against (two-hot) targets; unknown targets are masked."""
    if not known.any():
        return logits.sum() * 0
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    per_row = -(targets * log_probabilities).sum(-1)
    return per_row[known].mean()


def statistics(grid, row_masses, quantiles=False):
    """Tail masses plus statistics conditional on the in-range grid."""
    from .numeric_grid import summary

    values = row_masses.tolist()
    in_range = values[2:2 + len(grid)]
    result = summary(grid, in_range, quantiles=quantiles)
    result.update(p_below=values[BELOW], p_above=values[ABOVE])
    return result
