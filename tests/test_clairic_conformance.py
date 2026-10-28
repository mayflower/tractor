"""Official Clairic client against the local profile on a loopback port.

Conformance levels are reported separately:
A  raw schema, route and SDK transport (sync and async, errors propagate);
B  the official uncertainty invariants (p_null/confidence on every answer).
The subset satisfies A. B is `unsupported` without real released uncertainty
capabilities; it is never weakened to pass. With fixture probabilities B only
checks projection invariants, not calibration or model quality.
Requires the `clairic-conformance` extra; otherwise these tests are not executed.
"""
import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
from clairic_backend import DescriptionBackend
from test_clairic_http import FixtureUncertainty

from tractor.server import create_app

extraction_models = pytest.importorskip(
    "langchain_cloudflare.extraction_models",
    reason="not executed: install the clairic-conformance extra (pinned official client)")
sys.path.insert(0, str(Path(__file__).parent / "fixtures/clairic_official"))
from helpers import ExtractionQualityError, assert_extraction_result  # noqa: E402

STATE = ("Invoice INV-2291 from Northwind Supply, issued 2026-09-02, net 30. "
         "4 pallets at $250.00 each, shipping $80.00. Contact: Dana Ruiz, Priya Shah.")
# Official fixture state, restricted to its source-only fields.
REQUEST = {"state": STATE, "fields": {"invoice_number": {"type": "text"},
                                      "contacts": {"type": "list", "items": "text"},
                                      "po_number": {"type": "text", "instructions":
                                                    "Extract the explicitly stated purchase order number."}},
           "options": {"return_state_text": True}}
EXPECTED = {"invoice_number": "INV-2291", "contacts": ["Dana Ruiz", "Priya Shah"], "po_number": None}
ANSWERS = {"invoice number": ["INV-2291"], "contacts": ["Dana Ruiz", "Priya Shah"]}


@pytest.fixture
def loopback_only(monkeypatch):
    """Fail any connection that is not to this machine's loopback interface."""
    original = socket.socket.connect

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError(f"Unexpected external connection to {address[0]}")
        return original(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, address: connect(self, address) or 0)


def serve(**options):
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(api="clairic", **options), host="127.0.0.1",
                                           port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "Local server did not start"
        time.sleep(0.02)
    return server, thread, port


@pytest.fixture
def local_model(loopback_only, request):
    options = getattr(request, "param", {})
    backend = DescriptionBackend(ANSWERS)
    server, thread, port = serve(extractor=backend, **options)
    model = extraction_models.CloudflareWorkersAIExtractionModel(
        account_id="local", api_token="local", api_base_url=f"http://127.0.0.1:{port}/client/v4/accounts")
    yield model, backend
    server.should_exit = True
    thread.join(10)


def test_level_a_sdk_transport_sync_and_async(local_model):
    model, backend = local_model
    with pytest.raises(AssertionError, match="external"):
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)  # The guard is active.
    assert model._inference_url.endswith("/client/v4/accounts/local/ai/run/@cf/cloudflare/clairic")
    result = model.extract(**REQUEST)
    assert result == asyncio.run(model.aextract(**REQUEST))
    assert {k: a["value"] for k, a in result["answers"].items()} == EXPECTED
    assert result["state_text"] == STATE and result["truncated"] is False
    assert result["usage"]["input_tokens"] > 0 and result["usage"]["output_tokens"] == 0
    assert len(backend.inference_calls) == 2
    # The official SDK sends model "clairic" and the full typed request.
    sent = backend.inference_calls[0]
    assert sent.text == STATE and len(sent.fields) == 3


def test_level_a_errors_propagate_through_the_sdk(local_model):
    import httpx
    import requests

    model, backend = local_model
    unsupported = {"state": STATE, "fields": {"days_to_pay": {"type": "estimate", "min": 0, "max": 120}}}
    with pytest.raises(requests.HTTPError) as error:
        model.extract(**unsupported)
    assert error.value.response.status_code == 422
    assert error.value.response.json()["errors"][0]["code"] == "tractor.capability_unavailable"
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(model.aextract(**unsupported))
    assert backend.inference_calls == []


def test_level_b_uncertainty_invariants_are_unsupported_in_the_subset(local_model):
    model, _ = local_model
    result = model.extract(**REQUEST)
    payload = {"model": "clairic", **REQUEST}
    # The official helper requires confidence/p_null; the subset omits them.
    with pytest.raises((KeyError, AssertionError)) as failure:
        assert_extraction_result(result, payload, EXPECTED)
    assert not isinstance(failure.value, ExtractionQualityError)
    pytest.skip("unsupported: level B requires released typed_answerability and final_outcome_confidence")


@pytest.mark.parametrize("local_model", [{"uncertainty": FixtureUncertainty(0.01, 0.9)}], indirect=True)
def test_level_b_projection_invariants_with_fixture_capabilities(local_model):
    model, _ = local_model
    # With p_null 0.01 the null decision keeps values; po_number would need a value
    # and therefore must fail loudly rather than return a nonconforming null.
    request = {**REQUEST, "fields": {k: v for k, v in REQUEST["fields"].items() if k != "po_number"}}
    expected = {k: v for k, v in EXPECTED.items() if k != "po_number"}
    result = model.extract(**request)
    assert_extraction_result(result, {"model": "clairic", **request}, expected)
