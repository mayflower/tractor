import pytest

from tractor.jsonio import loads
from tractor.teacher import reserve


def test_short_price_profile_is_bounded_and_uses_same_durable_budget(tmp_path):
    limits = {'teacher_max_requests': 10, 'teacher_max_input_tokens': 1000000,
              'teacher_max_output_tokens': 1000000, 'max_external_cost_usd': 500}
    path = tmp_path / 'ledger.json'
    first = reserve(path, limits, 1000, 1000)
    second = reserve(path, limits, 1000, 1000, short_haiku=True)
    assert second['usd_reserved'] == pytest.approx(first['usd_reserved'] + .0006)
    with pytest.raises(ValueError, match='bounded'):
        reserve(path, limits, 100001, 1, short_haiku=True)
    assert loads(path.read_text())['requests'] == 2


def test_success_usage_releases_only_unused_reservation_once(tmp_path):
    from tractor.jsonio import loads
    from tractor.teacher import reserve, settle_success

    path = tmp_path / 'ledger.json'
    limits = {'max_external_cost_usd': 500, 'teacher_max_requests': 100,
              'teacher_max_input_tokens': 1000000, 'teacher_max_output_tokens': 1000000}
    reservation = reserve(path, limits, 20000, 4096, short_luna=True)
    usage = {'input_tokens': 2000, 'output_tokens': 300}
    result = settle_success(path, reservation, 20000, 4096, usage)
    assert abs(result['usd_reserved'] - .00035) < 1e-10
    assert result['requests'] == 1
    again = settle_success(path, reservation, 20000, 4096, usage)
    assert again == result
    assert float(loads(path.read_text())['usd_reserved']) == result['usd_reserved']


def test_claim_written_before_credit_is_conservative_after_interruption(tmp_path):
    from tractor.jsonio import dumps
    from tractor.teacher import reserve, settle_success

    path = tmp_path / 'ledger.json'
    limits = {'max_external_cost_usd': 500, 'teacher_max_requests': 100,
              'teacher_max_input_tokens': 1000000, 'teacher_max_output_tokens': 1000000}
    state = reserve(path, limits, 20000, 4096, short_luna=True)
    claims = path.with_suffix('.settled')
    claims.mkdir()
    (claims / '1.json').write_text(dumps({'simulated': 'claim persisted but ledger not credited'}))
    assert settle_success(path, state, 20000, 4096, {'input_tokens': 20, 'output_tokens': 3}) == state
