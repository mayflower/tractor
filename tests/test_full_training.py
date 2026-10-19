import pytest

from tractor.full_training import exposure_counts, score_general_dev


def test_full_schedule_retains_majority_and_balances_minorities():
    result = exposure_counts({'entities': 123785, 'relations': 20001}, {'entities': 50, 'relations': 50})
    assert result == {'entities': 124000, 'relations': 124000}
    assert sum(result.values()) % 32 == 0


def test_full_schedule_rejects_missing_buckets():
    with pytest.raises(ValueError):
        exposure_counts({'entities': 100}, {'entities': 50, 'relations': 50})


def test_checkpoint_measurement_rejects_missing_required_scores():
    policy = {'minimum_exclusive': .7, 'required_buckets': ['entities'],
              'required_classification_tasks': ['routing'], 'bucket_metric': 'field_f1',
              'qualification': 'all required'}
    with pytest.raises(ValueError, match='Incomplete'):
        score_general_dev([], [], policy)
