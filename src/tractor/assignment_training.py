"""Bounded positive weighting for the pinned native natural-record objective.

Training only: retain upstream matching, scalar, missing-field and object losses.
The extra term is exactly the positive part of BCE for matched list fields.
No inference schema, normalization, decoder rule or threshold is changed.
"""
import math
from collections import Counter

import torch
from torch.nn import functional as F


def matched_assignments(output, gold, record_mask):
    if ((output.modes != 0) & output.group_mask).any():
        raise ValueError('Assignment repair supports explicitly anchored natural records only')
    count = output.assign_logits.shape[-1] - 1
    if gold.shape[-1] < count:
        gold = F.pad(gold, (0, count - gold.shape[-1]))
    gold = gold[..., :count].to(output.assign_logits.device)
    if count == 0:
        return None
    anchor = gold.gather(3, output.anchor_fields.clamp_min(0)[..., None, None, None].expand(
        *gold.shape[:3], 1, count)).squeeze(3)
    keep = record_mask.to(anchor.device) & output.group_mask[..., None] & anchor.any(-1)
    b, r, g = torch.where(keep)
    if not b.numel():
        return None
    instance = anchor[b, r, g].long().argmax(-1)
    valid = instance < output.assign_logits.shape[2]
    b, r, g, instance = b[valid], r[valid], g[valid], instance[valid]
    valid = output.instance_mask[b, r, instance]
    b, r, g, instance = b[valid], r[valid], g[valid], instance[valid]
    logits = output.assign_logits[b, r, instance, :, 1:]
    target = gold[b, r, g]
    lists = output.field_mask[b, r] & ~output.scalar_fields[b, r]
    membership = output.field_membership[b, r]
    fields = output.field_mask[b, r].sum(-1).clamp_min(1)
    return logits, target, lists, membership, fields


def positive_assignment_term(matched):
    logits, target, lists, _membership, fields = matched
    # masked_fill avoids multiplying infinities from padded non-candidates by 0.
    terms = F.softplus(-logits).masked_fill(~(target & lists[..., None]), 0.)
    return (terms.mean(-1).sum(-1) / fields).mean()


class AssignmentLoss:
    """Install a reversible process-local wrapper around native dense record loss."""
    def __init__(self, positive_weight=1., diagnostic_batches=128):
        if (isinstance(positive_weight, bool) or not isinstance(positive_weight, (int, float))
                or not math.isfinite(positive_weight) or not 1 <= positive_weight <= 8):
            raise ValueError('Assignment positive weight must be finite and between 1 and 8')
        from gliner2.models.boundary import records
        if getattr(records.compute_dense_batch_loss, '_tractor_assignment_loss', False):
            raise ValueError('An assignment loss adapter is already installed')
        self.weight = float(positive_weight)
        self.batches = diagnostic_batches
        self.stats = Counter()
        self.original = records.compute_dense_batch_loss
        self.module = records

        def wrapped(output, gold_indicator, record_mask):
            result = self.original(output, gold_indicator, record_mask)
            matched = matched_assignments(output, gold_indicator, record_mask)
            if matched is None or not matched[0].shape[0]:
                return result
            if torch.is_grad_enabled() and self.stats['batches'] < self.batches:
                self.observe(matched)
            if self.weight != 1:
                result = dict(result, field_loss=result['field_loss']
                              + (self.weight - 1) * positive_assignment_term(matched))
            return result
        wrapped._tractor_assignment_loss = True
        self.wrapped = wrapped
        records.compute_dense_batch_loss = wrapped

    def observe(self, matched):
        logits, target, lists, membership, _fields = matched
        valid = lists[..., None] & membership
        positive, negative = valid & target, valid & ~target
        with torch.no_grad():
            probability = logits.detach().float().sigmoid()
            values = torch.stack((positive.sum(), negative.sum(),
                                  ((1 - probability) * positive).sum(), (probability * negative).sum())).cpu().tolist()
        self.stats.update(dict(zip(('positive_pairs', 'negative_pairs', 'positive_gradient_mass',
                                    'negative_gradient_mass'), values, strict=True)))
        self.stats['batches'] += 1

    def evidence(self):
        return {'positive_weight': self.weight, 'diagnostics': dict(self.stats),
                'qualification': 'Counts exclude padding and scalar anchors. Gradient masses are unweighted BCE logit derivatives before native candidate/field/record reduction; diagnostic only.'}

    def remove(self):
        if self.module.compute_dense_batch_loss is not self.wrapped:
            raise RuntimeError('Native record loss changed while adapter was installed')
        self.module.compute_dense_batch_loss = self.original
