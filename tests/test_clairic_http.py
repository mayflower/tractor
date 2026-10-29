"""Local Clairic REST profile: route, envelope, projection, errors and lifecycle."""
import asyncio
import logging
import threading
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from clairic_backend import ROUTE, DescriptionBackend
from fastapi.testclient import TestClient

from tractor.capabilities import Capability
from tractor.clairic_compiler import Policy
from tractor.jsonio import loads
from tractor.server import create_app
from tractor.types import TractorError
from tractor.uncertainty import FieldUncertainty, UncertaintyProvider

INVOICE = loads(Path(__file__).parents[1].joinpath("examples/clairic/invoice.json").read_text())


def body(fields, state="Invoice number: INV-2291. Total amount: 1080 USD.", **extra):
    return {"model": "clairic", "state": state, "fields": fields, **extra}


def client(backend, **options):
    return TestClient(create_app(extractor=backend, api="clairic", **options))


def test_documented_request_returns_envelope_with_exact_evidence():
    backend = DescriptionBackend({"invoice number": ["INV-2291"], "total amount": ["1080 USD"]})
    with client(backend) as http:
        response = http.post(ROUTE, json=INVOICE)
    assert response.status_code == 200
    envelope = response.json()
    assert set(envelope) == {"result", "success", "errors", "messages"}
    assert envelope["success"] is True and envelope["errors"] == [] and envelope["messages"] == []
    result = envelope["result"]
    state = INVOICE["state"]
    assert result["model"] == "clairic" and result["truncated"] is False
    assert result["state_text"] == state
    number = result["answers"]["invoice_number"]
    assert number == {"type": "text", "value": "INV-2291", "span": [16, 24]}
    assert state[slice(*number["span"])] == number["value"]
    # The normalized number keeps its source evidence; equality is not claimed.
    assert result["answers"]["total"] == {"type": "number", "value": 1080, "span": [40, 48]}
    # Missing probabilities are omitted, not null and not placeholders.
    assert not {"p_null", "confidence", "guess"} & {k for a in result["answers"].values() for k in a}
    assert result["usage"] == {"input_tokens": backend.last_tokens, "output_tokens": 0, "passes": 1}
    assert len(backend.inference_calls) == 1


def test_profiles_are_exclusive_per_process():
    backend = DescriptionBackend({})
    with client(backend) as http:
        assert http.post("/v1/extract", json={}).status_code == 404
        assert http.get("/readyz").json() == {"status": "ready"}
        assert http.post(ROUTE.replace("clairic", "clef"), json=INVOICE).status_code == 404
    with TestClient(create_app(extractor=backend)) as http:
        assert http.post(ROUTE, json=INVOICE).status_code == 404
    with pytest.raises(ValueError):
        create_app(extractor=backend, api="openai")


def test_lists_keep_typed_values_and_every_occurrence_span():
    state = "Contact: Dana Ruiz, Priya Shah. Signed: Dana Ruiz."
    backend = DescriptionBackend({"contacts": ["Dana Ruiz", "Priya Shah"]})
    with client(backend) as http:
        answer = http.post(ROUTE, json=body({"contacts": {"type": "list", "items": "text"}}, state)).json()
    answer = answer["result"]["answers"]["contacts"]
    assert answer["value"] == ["Dana Ruiz", "Priya Shah"]
    assert answer["spans"] == [[9, 18], [20, 30], [40, 49]]
    assert all(state[s:e] in answer["value"] for s, e in answer["spans"])
    empty = DescriptionBackend({})
    with client(empty) as http:
        result = http.post(ROUTE, json=body({"contacts": {"type": "list", "items": "text"}}, state)).json()
    assert result["result"]["answers"]["contacts"] == {"type": "list", "value": [], "spans": []}


def test_not_found_scalar_is_a_plain_null():
    with client(DescriptionBackend({})) as http:
        result = http.post(ROUTE, json=body({"po_number": {"type": "text"}})).json()["result"]
    assert result["answers"]["po_number"] == {"type": "text", "value": None}


@pytest.mark.parametrize("fields,answers,code", [
    ({"w": {"type": "number", "unit": "mm"}}, {"w": ["12 cm", "13 cm"]}, "ambiguous_value"),
    ({"w": {"type": "number", "unit": "mm"}}, {"w": ["5 cm"]}, "non_scalar_quantity"),
    ({"total": {"type": "number", "unit": "EUR"}}, {"total": ["1080 USD"]}, "currency_mismatch"),
    ({"n": {"type": "number", "integer": True}}, {"n": ["12.5"]}, "precision_exceeded"),
    ({"d": {"type": "duration", "unit": "seconds"}}, {"d": ["2 months"]}, "calendar_duration"),
    ({"d": {"type": "date"}}, {"d": ["03/04/2026"]}, "ambiguous_or_unsupported_date"),
])
def test_invalid_or_ambiguous_present_evidence_is_an_explicit_error(fields, answers, code):
    state = "W: 12 cm 13 cm 5 cm; 1080 USD; n=12.5; 2 months; 03/04/2026; ca. 5 cm"
    with client(DescriptionBackend(answers)) as http:
        response = http.post(ROUTE, json=body(fields, state))
    assert response.status_code == 422
    envelope = response.json()
    assert envelope["success"] is False and envelope["result"] is None
    [error] = envelope["errors"]
    assert error["code"] == "tractor.unresolved_field" and error["field"] == next(iter(fields))
    assert code in [d["code"] for d in error["details"]]


@pytest.mark.parametrize("spec,source,value", [
    ({"type": "number", "decimals": 2}, "12.50", Decimal("12.5")),
    ({"type": "number", "integer": True}, "1,080", 1080),
    ({"type": "duration", "unit": "hours"}, "90 minutes", Decimal("1.5")),
    ({"type": "duration"}, "2 days", 172800),
    ({"type": "duration", "unit": "years"}, "18 months", Decimal("1.5")),
    ({"type": "date"}, "2026-10-02", "2026-10-02"),
    ({"type": "list", "items": "number"}, "7", [7]),
])
def test_typed_values_follow_the_documented_projection(spec, source, value):
    with client(DescriptionBackend({"v": [source]}), policy=Policy(locale="en-US")) as http:
        response = http.post(ROUTE, json=body({"v": spec}, f"Value: {source}."))
    assert response.status_code == 200, response.json()
    answer = loads(response.content)["result"]["answers"]["v"]
    assert answer["value"] == value


def test_unsupported_requests_never_reach_inference():
    backend = DescriptionBackend({"x": ["INV"]})
    with client(backend) as http:
        for fields, extra in [
            ({"x": {"type": "estimate", "min": 0, "max": 1}}, {}),
            ({"x": {"type": "forecast", "min": 0, "max": 1}}, {}),
            ({"x": {"type": "text", "null_threshold": 0.5}}, {}),
            ({"x": {"type": "text"}}, {"images": ["data:image/png;base64,AA=="]}),
            ({"x": {"type": "number", "instructions": "Sum of the explicitly stated a and the explicitly stated b"}}, {}),
            ({"x": {"type": "text"}, "y": {"type": "text", "instructions": "Total incl. shipping"}}, {}),
        ]:
            response = http.post(ROUTE, json=body(fields, **extra))
            assert response.status_code == 422
            assert response.json()["errors"][0]["code"] in {
                "tractor.capability_unavailable", "tractor.unsupported_operation"}
        assert http.post(ROUTE, content=b"{").status_code == 400
        assert http.post(ROUTE, json=body({"x": {"type": "text"}}, model="other")).status_code == 400
        assert http.post(ROUTE.replace("local", "not local"), json=INVOICE).status_code == 400
    assert backend.inference_calls == []


def test_capacity_errors_are_errors_not_empty_results():
    class TooLong(DescriptionBackend):
        def candidates_with_signals(self, request):
            if request.text.startswith("Atlas"):
                return super().candidates_with_signals(request)
            raise TractorError(422, "token_limit", "Input exceeds 8192 tokens")

    with client(TooLong({}), body_limit=512) as http:
        assert http.post(ROUTE, json=body({"x": {"type": "text"}})).json()["errors"][0]["code"] == \
            "tractor.capacity_exceeded"
        big = http.post(ROUTE, json=body({"x": {"type": "text"}}, "x" * 600))
        assert big.status_code == 413 and big.json()["errors"][0]["code"] == "tractor.capacity_exceeded"
        assert http.get("/readyz").status_code == 200


class FixtureUncertainty(UncertaintyProvider):
    """Test-only probabilities; the binding marks them as a fixture."""

    def __init__(self, p_null, confidence=None):
        binding = {"fixture": True}
        self.capabilities = [Capability(name, architecture=True, weights="fixture", provenance="fixture",
                                        calibration="fixture", released=True, binding=binding)
                             for name in ("typed_answerability", "final_outcome_confidence")]
        self.p_null, self.confidence = p_null, confidence

    def score(self, plan, trace):
        return {f.name: FieldUncertainty(self.p_null, self.confidence) for f in plan.fields}


@pytest.mark.parametrize("p_null,threshold,value,guess", [
    (0.2, 0.5, "INV-2291", "INV-2291"),
    (0.5, 0.5, "INV-2291", "INV-2291"),  # Equality does not exceed the threshold.
    (0.8, 0.5, None, "INV-2291"),
    (0.3, 0.25, None, "INV-2291"),
    (0.01, 0.01, "INV-2291", "INV-2291"),
])
def test_null_threshold_with_a_real_capability(p_null, threshold, value, guess):
    backend = DescriptionBackend({"invoice number": ["INV-2291"]})
    with client(backend, uncertainty=FixtureUncertainty(p_null, 0.7)) as http:
        result = http.post(ROUTE, json=body({"invoice_number": {"type": "text", "null_threshold": threshold}}))
    answer = loads(result.content)["result"]["answers"]["invoice_number"]
    assert answer["value"] == value and answer["guess"] == guess
    assert answer["p_null"] == Decimal(str(p_null)) and answer["confidence"] == Decimal("0.7")


def test_low_p_null_without_a_value_withholds_probabilities_for_that_field_only():
    backend = DescriptionBackend({"invoice number": ["INV-2291"]})
    fields = {"po_number": {"type": "text"}, "invoice_number": {"type": "text"}}
    with client(backend, uncertainty=FixtureUncertainty(0.1, 0.7)) as http:
        response = http.post(ROUTE, json=body(fields))
        assert http.get("/readyz").status_code == 200
    envelope = loads(response.content)
    assert response.status_code == 200 and envelope["success"]
    assert envelope["result"]["answers"]["po_number"] == {"type": "text", "value": None}
    assert envelope["result"]["answers"]["invoice_number"]["p_null"] == Decimal("0.1")
    assert envelope["messages"] == [{"code": "tractor.uncertainty_withheld", "field": "po_number",
                                     "message": "p_null is at or below the threshold but no value exists"}]
    with client(DescriptionBackend({}), uncertainty=FixtureUncertainty(0.9)) as http:
        answer = http.post(ROUTE, json=body({"po_number": {"type": "text"}})).json()["result"]["answers"]
    assert answer["po_number"] == {"type": "text", "value": None, "p_null": 0.9}


def test_a_null_decision_drops_the_source_span():
    backend = DescriptionBackend({"invoice number": ["INV-2291"]})
    with client(backend, uncertainty=FixtureUncertainty(0.8)) as http:
        answer = http.post(ROUTE, json=body({"invoice_number": {"type": "text"}})).json()["result"]["answers"]
    assert answer["invoice_number"] == {"type": "text", "value": None, "p_null": 0.8, "guess": "INV-2291"}


def test_compute_runs_only_when_enabled_and_uses_one_pass():
    state = "Subtotal: 1000.00 USD. Shipping: 80.00 USD. Issued 2026-01-31."
    backend = DescriptionBackend({"subtotal": ["1000.00 USD"], "shipping": ["80.00 USD"],
                                  "issue date": ["2026-01-31"]})
    fields = {"total": {"type": "number", "unit": "USD", "decimals": 2,
                        "instructions": "Sum of the explicitly stated subtotal and the explicitly stated shipping"},
              "due": {"type": "date", "instructions": "The explicitly stated issue date plus 1 month"},
              "printed": {"type": "number", "unit": "USD",
                          "instructions": "Extract the explicitly stated subtotal."}}
    with client(backend, policy=Policy(compute=True)) as http:
        result = loads(http.post(ROUTE, json=body(fields, state)).content)["result"]
    assert result["answers"]["total"] == {"type": "number", "value": 1080}
    assert result["answers"]["due"] == {"type": "date", "value": "2026-02-28"}
    assert result["answers"]["printed"]["value"] == 1000 and "span" in result["answers"]["printed"]
    assert len(backend.inference_calls) == 1


class Blocking(DescriptionBackend):
    def __init__(self):
        super().__init__({"invoice number": ["INV"]})
        self.entered, self.release = threading.Event(), threading.Event()

    def candidates_with_signals(self, request):
        if request.text == "block":
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError("Test deadline")
        if request.text == "fail":
            raise RuntimeError("private source text must not be logged")
        return super().candidates_with_signals(request)


def lifecycle(check, **options):
    async def run():
        backend = Blocking()
        app = create_app(extractor=backend, api="clairic", **options)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
                await check(backend, http)
    asyncio.run(run())


def text_body(text, **extra):
    return body({"invoice_number": {"type": "text"}}, text, **extra)


def test_disconnect_holds_the_lock_and_failures_mark_unready(caplog):
    async def check(backend, http):
        first = asyncio.create_task(http.post(ROUTE, json=text_body("block")))
        assert await asyncio.to_thread(backend.entered.wait, 2)
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        second = asyncio.create_task(http.post(ROUTE, json=text_body("second")))
        await asyncio.sleep(0.05)
        assert "second" not in [r.text for r in backend.calls]
        third = await http.post(ROUTE, json=text_body("third"))
        assert third.status_code == 503 and third.json()["errors"][0]["code"] == "tractor.overloaded"
        backend.release.set()
        assert (await second).status_code == 200
        with caplog.at_level(logging.DEBUG):
            failed = await http.post(ROUTE, json=text_body("fail"))
        assert failed.status_code == 500 and "private source" not in caplog.text
        assert (await http.get("/readyz")).status_code == 503
        assert (await http.post(ROUTE, json=text_body("after"))).json()["errors"][0]["code"] == "tractor.not_ready"

    lifecycle(check, wait_timeout=2)


def test_wait_timeout_and_reject_if_busy():
    async def check(backend, http):
        first = asyncio.create_task(http.post(ROUTE, json=text_body("block")))
        assert await asyncio.to_thread(backend.entered.wait, 2)
        busy = await http.post(ROUTE, json=text_body("busy", options={"rejectIfBusy": True}))
        assert busy.status_code == 503
        waited = await http.post(ROUTE, json=text_body("second"))
        assert waited.status_code == 503
        assert {"busy", "second"}.isdisjoint(r.text for r in backend.calls)
        backend.release.set()
        assert (await first).status_code == 200

    lifecycle(check, wait_timeout=0.05)
