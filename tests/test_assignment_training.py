from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from tractor.assignment_training import (
    AssignmentLoss,
    matched_assignments,
    positive_assignment_term,
)


def example():
    # Gold record order is deliberately opposite to anchor candidate order.
    logits = torch.zeros(1, 1, 2, 2, 3, requires_grad=True)
    output = SimpleNamespace(assign_logits=logits, modes=torch.zeros(1, 1, dtype=torch.long),
        object_logits=torch.zeros(1, 1, 2), pool_spans=torch.tensor([[[0, 1], [2, 3]]]),
        group_mask=torch.ones(1, 1, dtype=torch.bool), anchor_fields=torch.zeros(1, 1, dtype=torch.long),
        instance_mask=torch.ones(1, 1, 2, dtype=torch.bool), field_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        scalar_fields=torch.tensor([[[True, False]]]), field_membership=torch.ones(1, 1, 2, 2, dtype=torch.bool))
    gold = torch.tensor([[[[[False, True], [True, False]], [[True, False], [False, True]]]]])
    return output, gold, torch.ones(1, 1, 2, dtype=torch.bool)


def test_pinned_native_loss_is_unchanged_at_one_and_weights_only_positive_assignments():
    from gliner2.models.boundary import records

    output, gold, mask = example()
    original = records.compute_dense_batch_loss(output, gold, mask)
    adapter = AssignmentLoss(1)
    try:
        actual = records.compute_dense_batch_loss(output, gold, mask)
        assert torch.equal(actual['field_loss'], original['field_loss'])
        assert torch.equal(actual['object_loss'], original['object_loss'])
    finally:
        adapter.remove()
    adapter = AssignmentLoss(3)
    try:
        actual = records.compute_dense_batch_loss(output, gold, mask)
        expected = original['field_loss'] + 2 * positive_assignment_term(matched_assignments(output, gold, mask))
        torch.testing.assert_close(actual['field_loss'], expected)
        assert torch.equal(actual['object_loss'], original['object_loss'])
        assert adapter.evidence()['diagnostics']['positive_pairs'] == 2
    finally:
        adapter.remove()


def test_weighted_term_matches_pytorch_bce_values_and_gradients():
    output, gold, mask = example()
    matched = matched_assignments(output, gold, mask)
    logits, targets, lists, _, fields = matched
    ordinary = (F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none').mean(-1)
                .masked_fill(~lists, 0).sum(-1) / fields).mean()
    expected = (F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=torch.tensor(3.), reduction='none')
                .mean(-1).masked_fill(~lists, 0).sum(-1) / fields).mean()
    actual = ordinary + 2 * positive_assignment_term(matched)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected, output.assign_logits, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, output.assign_logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[0, 0, 1, 1, 1] < 0  # record 0 attaches to anchor 1
    assert actual_grad[0, 0, 0, 1, 2] < 0
    assert actual_grad[..., 0, :].count_nonzero() == 0  # scalar anchor untouched
    assert actual_grad[..., 0].count_nonzero() == 0  # NULL column untouched


def test_absent_fields_still_have_negative_supervision_and_padding_is_ignored():
    output, gold, mask = example()
    gold[..., 1, :] = False
    output.field_membership[..., -1] = False
    matched = matched_assignments(output, gold, mask)
    assert positive_assignment_term(matched).item() == 0
    adapter = AssignmentLoss(1)
    try:
        adapter.observe(matched)
        assert adapter.stats['positive_pairs'] == 0
        assert adapter.stats['negative_pairs'] == 2
    finally:
        adapter.remove()


def test_adapter_restores_native_loss_and_rejects_unsupported_mode():
    from gliner2.models.boundary import records
    original = records.compute_dense_batch_loss
    adapter = AssignmentLoss(3)
    try:
        with pytest.raises(ValueError, match='already installed'):
            AssignmentLoss(3)
        output, gold, mask = example()
        output.modes.fill_(1)
        with pytest.raises(ValueError, match='natural records'):
            matched_assignments(output, gold, mask)
    finally:
        adapter.remove()
    assert records.compute_dense_batch_loss is original
    for weight in (0, True, 9, float('nan')):
        with pytest.raises(ValueError):
            AssignmentLoss(weight)
