"""Clairic protocol parsing and capability contracts; no model, no ML imports."""
import subprocess
import sys
from datetime import date
from decimal import Decimal

import pytest

from tractor.capabilities import (
    CAPABILITIES,
    Capability,
    CapabilityUnavailable,
    InvalidInput,
    ModelCapabilities,
    extractor_capabilities,
)
from tractor.clairic_types import FIELD_TYPES, parse_request
from tractor.jsonio import loads
from tractor.types import ExtractRequest


def request(fields, **extra):
    return {"model": "clairic", "state": "Invoice INV-1.", "fields": fields, **extra}


def test_all_nine_types_are_representable_and_names_are_exact():
    fields = {
        "_private": {"type": "text"},
        "total.amount": {"type": "number", "unit": "USD", "decimals": 2},
        "issue-date": {"type": "date"},
        "t": {"type": "time"},
        "dt": {"type": "datetime"},
        "wait": {"type": "duration", "unit": "days", "integer": True},
        "names": {"type": "list", "items": "text"},
        "days_to_pay": {"type": "estimate", "min": 0, "max": 120, "step": 1},
        "sales": {"type": "forecast", "min": 0, "max": 10, "horizon": 4, "interval": "month"},
    }
    parsed = parse_request(request(fields))
    assert list(parsed.fields) == list(fields)
    assert {f.type for f in parsed.fields.values()} == set(FIELD_TYPES)
    assert parsed.model == "clairic"
    assert parse_request({**request(fields), "model": "@cf/cloudflare/clairic"}).model == "@cf/cloudflare/clairic"


@pytest.mark.parametrize("payload", [
    {"state": "x", "fields": {"a": {"type": "text"}}},  # model is required despite its default
    request({"a": {"type": "text"}}, model="gpt"),
    request({}),
    request({"1bad": {"type": "text"}}),
    request({"a b": {"type": "text"}}),
    request({"a" * 65: {"type": "text"}}),
    request({"a": {}}),
    request({"a": {"type": "boolean"}}),
    request({"a": {"type": "text", "unit": "USD"}}),
    request({"a": {"type": "text", "instructions": ""}}),
    request({"a": {"type": "text", "instructions": None}}),
    request({"a": {"type": "number", "decimals": 7}}),
    request({"a": {"type": "number", "decimals": -1}}),
    request({"a": {"type": "number", "decimals": True}}),
    request({"a": {"type": "number", "integer": True, "decimals": 2}}),
    request({"a": {"type": "number", "integer": "yes"}}),
    request({"a": {"type": "date", "integer": True}}),
    request({"a": {"type": "list"}}),
    request({"a": {"type": "list", "items": "duration"}}),
    request({"a": {"type": "list", "items": "text", "decimals": 1}}),
    request({"a": {"type": "list", "items": "text", "null_threshold": 0.5}}),
    request({"a": {"type": "duration", "unit": "fortnights"}}),
    request({"a": {"type": "text", "null_threshold": 0}}),
    request({"a": {"type": "text", "null_threshold": 1.5}}),
    request({"a": {"type": "text", "null_threshold": True}}),
    request({"a": {"type": "estimate", "min": 0}}),
    request({"a": {"type": "estimate", "min": 5, "max": 1}}),
    request({"a": {"type": "estimate", "min": 0, "max": "2026-01-01"}}),
    request({"a": {"type": "estimate", "min": 0, "max": 10, "step": 0}}),
    request({"a": {"type": "estimate", "min": 0, "max": 10, "step": "month"}}),
    request({"a": {"type": "estimate", "min": 0, "max": 10, "scale": "log"}}),
    request({"a": {"type": "estimate", "min": "2026-01-01", "max": "2026-02-30"}}),
    request({"a": {"type": "forecast", "min": 0, "max": 1, "horizon": 65}}),
    request({"a": {"type": "forecast", "min": 0, "max": 1, "interval": "decade"}}),
    request({"a": {"type": "text"}}, state=""),
    request({"a": {"type": "text"}}, state="   "),
    request({"a": {"type": "text"}}, reference_date="19.10.2026"),
    request({"a": {"type": "text"}}, options={"unknown": True}),
    request({"a": {"type": "text"}}, options={"return_state_text": "yes"}),
    request({"a": {"type": "text"}}, images=["https://example.com/x.png"]),
    request({"a": {"type": "text"}}, extra_key=1),
])
def test_invalid_requests_fail_before_compilation(payload):
    with pytest.raises(InvalidInput):
        parse_request(payload)


@pytest.mark.parametrize("raw", [
    b'{"model": "clairic", "state": "x", "fields": {"a": {"type": "text", "null_threshold": NaN}}}',
    b'{"model": "clairic", "state": "x", "fields": {"a": {"type": "text", "null_threshold": Infinity}}}',
    b'{"model": "clairic", "state": "x", "fields": {"a": {"type": "text"}, "a": {"type": "date"}}}',
])
def test_ambiguous_json_never_passes(raw):
    with pytest.raises(ValueError):
        parse_request(loads(raw))


def test_missing_and_explicit_null_threshold_are_distinct():
    absent = parse_request(request({"a": {"type": "text"}})).fields["a"]
    explicit = parse_request(request({"a": {"type": "text", "null_threshold": 0.5}})).fields["a"]
    assert absent.threshold == explicit.threshold == Decimal("0.5")
    assert "null_threshold" not in absent.explicit and "null_threshold" in explicit.explicit
    assert absent.null_threshold is None and explicit.null_threshold == Decimal("0.5")


def test_optional_inputs_and_transport_option():
    parsed = parse_request(request({"a": {"type": "text"}}, reference_date="2026-10-19",
                                   options={"return_state_text": True, "rejectIfBusy": True},
                                   images=[{"content_type": "image/png", "base64": "AA=="}]))
    assert parsed.reference_date == date(2026, 10, 19)
    assert parsed.options.return_state_text and parsed.options.reject_if_busy
    assert not parsed.options.include_quantiles
    # An empty state is valid only together with images.
    assert parse_request(request({"a": {"type": "text"}}, state="", images=["data:image/png;base64,AA=="]))


def test_capabilities_need_every_evidence_level():
    assert not Capability("source_presence", architecture=True, weights="sha", provenance="run",
                          released=True).executable
    assert Capability("source_presence", architecture=True, weights="sha", provenance="run",
                      calibration="cal", released=True).executable
    # A release flag alone never passes a learned-capability gate.
    flagged = Capability("typed_answerability", released=True)
    assert not flagged.executable and "weights" in flagged.missing()
    assert Capability("deterministic_compute", architecture=True, released=True).executable
    with pytest.raises(ValueError):
        Capability("calibrated")


def test_legacy_extractor_declares_only_demonstrated_capabilities():
    class Legacy:
        backend, model_revision, schema_version, precision = "biqwen", "sha", "biqwen-types-v2", "fp32"

    capabilities = extractor_capabilities(Legacy())
    assert {c["name"] for c in capabilities.describe() if c["executable"]} == {
        "source_spans", "typed_extraction", "typed_lists"}
    for name in ("source_presence", "typed_answerability", "final_outcome_confidence",
                 "numeric_distribution", "forecast", "images", "deterministic_compute"):
        with pytest.raises(CapabilityUnavailable):
            capabilities.require([name])
    assert extractor_capabilities(Legacy(), compute=True).executable("deterministic_compute")
    assert len(ModelCapabilities().describe()) == len(CAPABILITIES)


def test_native_v1_contract_is_unchanged():
    with pytest.raises(ValueError):
        ExtractRequest(text="x", fields={"_private": {"type": "text", "description": "d"}})
    assert ExtractRequest(text="x", fields={"a": {"type": "text", "description": "d"}})


def test_protocol_imports_no_ml_libraries():
    code = ("import sys, tractor, tractor.clairic_types, tractor.capabilities, tractor.evidence, "
            "tractor.uncertainty, tractor.clairic_compute\n"
            "assert not {'torch', 'transformers', 'gliner2'} & set(sys.modules), sys.modules.keys()")
    subprocess.run([sys.executable, "-c", code], check=True)
