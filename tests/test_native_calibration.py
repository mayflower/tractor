import pytest

from tractor.calibration import calibration_error, fit_score_mapping


def test_mapping_reduces_known_overconfidence_without_claiming_generalization():
    observations = [(.99, int(i % 2 == 0)) for i in range(100)]
    mapping = fit_score_mapping(observations)
    assert mapping is not None and mapping["scale"] >= 0
    assert calibration_error(observations, mapping)["brier"] < calibration_error(observations)["brier"]


def test_no_probability_claim_from_one_outcome_or_empty_predictions():
    assert fit_score_mapping([(.99, 1)] * 100) is None
    assert calibration_error([]) == {"emitted_fields": 0, "brier": None, "ece": None}


@pytest.mark.parametrize("score", [0., 1.])
def test_extreme_scores_are_finite(score):
    result = calibration_error([(score, 1)], {"scale": 20., "bias": -20.})
    assert 0 <= result["brier"] <= 1
