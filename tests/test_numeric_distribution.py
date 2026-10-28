"""Estimate prototype: exact grids, one distribution, monotone quantiles (software only)."""
from datetime import date
from decimal import Decimal

import pytest

from tractor.clairic_types import parse_request
from tractor.numeric_grid import GridError, dates, grid_for, linear, log_points, quantile, summary


def test_decimal_grids_have_no_float_drift_and_reject_unaligned_bounds():
    assert linear("-0.3", "0.3", "0.1") == [Decimal(v) for v in ("-0.3", "-0.2", "-0.1", "0.0", "0.1", "0.2", "0.3")]
    with pytest.raises(GridError, match="not on the step"):
        linear(0, 1, "0.3")
    with pytest.raises(GridError, match="exceeds"):
        linear(0, 10 ** 9, 1)


def test_log_scale_needs_a_positive_domain_and_keeps_exact_endpoints():
    points, coordinates = log_points("0.01", "100", 5)
    assert points[0] == Decimal("0.01") and points[-1] == Decimal("100") and points[2] == Decimal("1")
    assert coordinates == sorted(coordinates)
    for bounds in (("0", "1"), ("-1", "1"), ("5", "1")):
        with pytest.raises(GridError):
            log_points(*bounds, 5)
    spec = parse_request({"model": "clairic", "state": "x", "fields": {"v": {
        "type": "estimate", "min": 1, "max": 10, "scale": "log", "step": 1}}}).fields["v"]
    with pytest.raises(GridError, match="log-scale"):
        grid_for(spec)


def test_date_grids_follow_explicit_calendar_rules():
    assert dates(date(2028, 1, 31), date(2028, 4, 30), "month") == [
        date(2028, 1, 31), date(2028, 2, 29), date(2028, 3, 31), date(2028, 4, 30)]
    assert dates(date(2028, 2, 29), date(2030, 2, 28), "year") == [
        date(2028, 2, 29), date(2029, 2, 28), date(2030, 2, 28)]
    assert len(dates(date(2026, 1, 1), date(2026, 1, 31), 10)) == 4
    with pytest.raises(GridError):
        dates(date(2026, 1, 1), date(2026, 1, 30), 7)


def test_quantiles_come_from_one_cdf_and_are_monotone():
    points = linear(0, 10, 1)
    masses = [0, 1, 2, 4, 2, 1, 0, 0, 0, 0, 0]
    stats = summary(points, masses, quantiles=True)
    assert stats["quantiles"] == sorted(stats["quantiles"]) and len(stats["quantiles"]) == 99
    assert stats["p10"] <= stats["median"] <= stats["p90"]
    assert stats["median"] == quantile(points, masses, 0.5) == 3
    assert stats["conditioning"] == "non_null_in_range"
    assert "mean" not in summary(dates(date(2026, 1, 1), date(2026, 1, 3), 1), [1, 1, 1])


def test_head_masks_padding_tails_and_trains_on_tiny_fixtures():
    torch = pytest.importorskip("torch")
    from tractor.numeric_distribution import (
        GridDistributionHead,
        loss,
        masses,
        pad,
        statistics,
        target_distribution,
    )

    torch.manual_seed(0)
    grids = [linear(0, 4, 1), linear(-1, 1, "0.5"), linear(10, 12, 1)]
    coordinates, mask = pad(grids)
    head = GridDistributionHead(8)
    assert not head.released
    fields = torch.randn(3, 8)
    logits = head(fields, coordinates, mask)
    probabilities = masses(logits)
    assert torch.allclose(probabilities.sum(-1), torch.ones(3))
    assert probabilities[2, 2 + 3:].sum() < 1e-6  # Padding gets no mass.
    targets = torch.stack([target_distribution(g, v, mask.shape[1])
                           for g, v in zip(grids, [Decimal("2.5"), Decimal("5"), Decimal("11")], strict=True)])
    assert targets[0, 2 + 2] == targets[0, 2 + 3] == 0.5 and targets[1, 1] == 1
    known = torch.tensor([True, True, False])
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    first = None
    for _ in range(60):
        value = loss(head(fields, coordinates, mask), targets, known)
        first = first if first is not None else float(value.detach())
        optimizer.zero_grad()
        value.backward()
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        optimizer.step()
    assert float(value.detach()) < first
    stats = statistics(grids[0], masses(head(fields, coordinates, mask))[0])
    assert 0 <= stats["p_below"] <= 1 and stats["p10"] <= stats["median"] <= stats["p90"]
