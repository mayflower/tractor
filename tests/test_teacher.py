import pytest

from tractor.teacher import reserve


def test_budget_cannot_be_overspent_or_reset(tmp_path):
    limits = {"teacher_max_requests": 1, "teacher_max_input_tokens": 1000,
              "teacher_max_output_tokens": 1000, "max_external_cost_usd": 1}
    path = tmp_path / "budget.json"
    first = reserve(path, limits, 200, 200)
    assert first["requests"] == 1
    with pytest.raises(ValueError, match="budget exhausted"):
        reserve(path, limits, 200, 200)
    assert '"requests": 1' in path.read_text()


def test_money_limit_is_checked_before_request(tmp_path):
    limits = {"teacher_max_requests": 100, "teacher_max_input_tokens": 1000000,
              "teacher_max_output_tokens": 1000000, "max_external_cost_usd": .00001}
    with pytest.raises(ValueError, match="usd_reserved"):
        reserve(tmp_path / "budget.json", limits, 200, 200)


def test_teacher_json_fences_do_not_accept_prose_or_truncation():
    from tractor.teacher import parse_teacher

    raw = {'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': '```json\n{"ok": true}\n```'}]}
    assert parse_teacher(raw) == {'ok': True}
    raw['content'][0]['text'] = 'Some prose {"ok": true}'
    with pytest.raises(ValueError):
        parse_teacher(raw)
    raw['stop_reason'] = 'max_tokens'
    with pytest.raises(ValueError, match='Incomplete'):
        parse_teacher(raw)
