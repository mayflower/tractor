from decimal import Decimal

import pytest

from tractor.compare import exact, parse_provider, score_response, select_rows
from tractor.jsonio import dumps


def test_exact_numbers_and_failures():
    assert exact(Decimal("0.1"), Decimal("0.10"))
    assert not exact(True, 1)
    assert not exact("1", 1)
    assert not exact([{"x": 1}, {"x": 2}], [{"x": 2}, {"x": 1}])
    gold = {"data": {"x": None}, "issues": [{"field": "x", "code": "not_found"}]}
    assert score_response(gold, gold) == {"exact": True, "data_exact": True}
    assert not score_response({"data": {"x": None}, "issues": []}, gold)["exact"]
    with pytest.raises(ValueError):
        score_response({**gold, "extra": 1}, gold)


def test_provider_refusals_truncation_and_duplicate_keys():
    with pytest.raises(ValueError):
        parse_provider("anthropic", {"stop_reason": "max_tokens", "content": []})
    with pytest.raises(ValueError):
        parse_provider("openai", {"status": "incomplete"})
    with pytest.raises(ValueError):
        parse_provider("anthropic", {"stop_reason": "end_turn", "content": [
            {"type": "text", "text": '{"data": 1, "data": 2}'}]})
    assert parse_provider("openai", {"status": "completed", "output": [
        {"type": "reasoning"}, {"type": "message", "content": [
            {"type": "output_text", "text": '{"data":{"x":0.1},"issues":[]}'}]}]
    })["data"]["x"] == Decimal("0.1")


def test_selection_frozen_and_stratified(tmp_path):
    rows = [{"id": str(i), "scenario": str(i % 2), "gold": i} for i in range(10)]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(dumps(r) for r in rows))
    first = select_rows(path, 4)
    path.write_text("\n".join(dumps({**r, "gold": -1}) for r in reversed(rows)))
    assert [r["id"] for r in select_rows(path, 4)] == [r["id"] for r in first]
    assert [r["scenario"] for r in first] == ["0", "1", "0", "1"]
    with pytest.raises(ValueError):
        select_rows(path, 11)


def test_http_run_counts_failures_without_leaking_gold(tmp_path, monkeypatch):
    import httpx

    from tractor import compare
    from tractor.jsonio import loads

    rows = [{"id": str(i), "request": {"text": "width 1 cm", "fields": {
        "width": {"type": "quantity", "description": "width", "unit": "mm"}}},
        "gold": {"response": {"data": {"width": 10}, "issues": []}}} for i in range(3)]
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(dumps(row) for row in rows))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return JSON")
    seen = []

    def respond(request):
        payload = loads(request.content)
        assert "gold" not in payload
        assert "10" not in payload["input"]
        assert payload["model"] == "gpt-6-luna"
        seen.append(payload)
        if len(seen) == 1:
            return httpx.Response(429, headers={"x-request-id": "limited"})
        text = 'not json' if len(seen) == 2 else '{"data":{"width":10},"issues":[]}'
        return httpx.Response(200, json={"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": text}]}]})

    client_type = httpx.Client
    monkeypatch.setattr(compare.httpx, "Client", lambda **kw: client_type(
        transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    output = tmp_path / "run"
    args = ["--provider", "openai", "--data", str(data), "--limit", "3",
            "--prompt", str(prompt), "--output", str(output)]
    compare.main(args)
    summary = loads((output / "summary.json").read_text())
    assert summary["counts"] == {"requests": 3, "exact": 1, "data_exact": 1, "errors": 2}
    assert len(seen) == 3  # No retries, failures remain in denominator.
    assert "test-secret" not in (output / "responses.jsonl").read_text()
    with pytest.raises(FileExistsError):
        compare.main(args)


def test_anthropic_workspace_header(monkeypatch):
    from tractor.compare import provider_request

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "workspace")
    _, headers, body = provider_request("anthropic", {"text": "x"}, "prompt", 100)
    assert headers["anthropic-workspace-id"] == "workspace"
    assert body["thinking"] == {"type": "disabled"}
    assert "workspace" not in dumps(body)
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID")
    assert "anthropic-workspace-id" not in provider_request(
        "anthropic", {"text": "x"}, "prompt", 100)[1]
