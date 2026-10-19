import pytest

from tractor.generalist_quality import quality_gate

POLICY = {'minimum_exclusive': .7, 'bucket_metric': 'field_f1',
          'required_buckets': ['relations', 'records'], 'required_classification_tasks': ['routing'],
          'qualification': 'Every area must pass'}


def scores(value):
    return {'buckets': {'relations': {'field_f1': value}, 'records': {'field_f1': .99}},
            'classification_tasks': {'routing': .95}, 'macro_f1': .99, 'product_exact_requests': 1.0}


@pytest.mark.parametrize('value', [.7, .69, None, float('nan'), float('inf'), True, 1.01])
def test_high_average_and_product_score_cannot_hide_failed_area(value):
    assert not quality_gate(scores(value), POLICY)['passed']


def test_every_area_must_be_present_and_above_threshold():
    assert quality_gate(scores(.70001), POLICY)['passed']
    s = scores(.8)
    del s['classification_tasks']['routing']
    assert not quality_gate(s, POLICY)['passed']
    del s['buckets']['relations']
    assert len(quality_gate(s, POLICY)['failed_areas']) == 2


def test_repository_json_decimal_values_are_supported():
    from tractor.jsonio import loads

    policy = POLICY | {'minimum_exclusive': loads('0.70')}
    assert quality_gate(scores(loads('0.71')), policy)['passed']
    assert not quality_gate(scores(loads('0.70')), policy)['passed']
