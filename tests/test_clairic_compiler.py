"""Clairic requests compile to explicit plans on existing Tractor contracts."""
import pytest

from tractor.capabilities import (
    Capability,
    CapabilityUnavailable,
    CapacityExceeded,
    UnsupportedOperation,
    extractor_capabilities,
)
from tractor.clairic_compiler import Policy, classify, compile_request
from tractor.clairic_types import parse_request
from tractor.types import Date, Money, Number, Quantity, Text


class Legacy:
    backend, model_revision = "test", "sha"


def plan(fields, policy=None, capabilities=None, **extra):
    policy = policy or Policy()
    request = parse_request({"model": "clairic", "state": "Invoice INV-1 total 5 USD.", "fields": fields, **extra})
    return compile_request(request, capabilities or extractor_capabilities(Legacy(), compute=policy.compute), policy)


def test_external_names_map_to_collision_free_internal_ids_with_semantics():
    compiled = plan({"_x": {"type": "text"}, "x": {"type": "text"},
                     "invoice.number": {"type": "text",
                                        "instructions": "Extract the explicitly stated invoice number."}})
    assert [f.name for f in compiled.fields] == ["_x", "x", "invoice.number"]
    assert [f.internal_id for f in compiled.fields] == ["f0", "f1", "f2"]
    assert compiled.internal_names == {"f0": "_x", "f1": "x", "f2": "invoice.number"}
    fields = compiled.extract_request.fields
    # The semantic part of the instruction reaches the model as the description.
    assert fields["f2"].description == "invoice number" and fields["f0"].description == "x"
    assert compiled.extract_request.text == "Invoice INV-1 total 5 USD."


def test_types_compile_to_existing_contracts_without_false_equivalences():
    compiled = plan({
        "a": {"type": "number", "unit": "USD"}, "b": {"type": "number", "unit": "mm"},
        "c": {"type": "number", "unit": "pallets"}, "d": {"type": "number"},
        "e": {"type": "list", "items": "text"}, "f": {"type": "list", "items": "date"},
        "g": {"type": "date"}, "h": {"type": "list", "items": "number", "decimals": 2},
    })
    fields = compiled.extract_request.fields
    assert isinstance(fields["f0"], Money) and fields["f0"].currency == "USD"
    assert isinstance(fields["f1"], Quantity) and fields["f1"].unit == "mm"
    # An unknown unit is an output label only; no conversion is claimed.
    assert type(fields["f2"]) is Number and type(fields["f3"]) is Number
    # A Clairic list is typed multiple extraction, not Tractor's separator list.
    assert type(fields["f4"]) is Text and fields["f4"].cardinality == "unique_list"
    assert type(fields["f5"]) is Date and fields["f5"].cardinality == "unique_list"
    assert type(fields["f7"]) is Number and fields["f7"].cardinality == "unique_list"
    assert fields["f6"].date_order is None and compiled.extract_request.locale is None
    assert "typed_lists" in compiled.capabilities


def test_locale_and_date_order_come_only_from_the_server_policy():
    compiled = plan({"g": {"type": "date"}, "n": {"type": "number"}},
                    Policy(locale="de-DE", date_order="DMY"))
    assert compiled.extract_request.fields["f0"].date_order == "DMY"
    assert compiled.extract_request.locale == "de-DE"


@pytest.mark.parametrize("instructions,expected", [
    (None, "supported_source_only"),
    ("Extract the explicitly stated total amount.", "supported_source_only"),
    ("Extract the explicitly stated invoice number. Return null if absent.", "supported_source_only"),
    ("Copy the literally written company name exactly as written.", "supported_source_only"),
    ("Extrahiere die ausdrücklich genannte Rechnungsnummer.", "supported_source_only"),
    ("Sum of the explicitly stated subtotal and the explicitly stated shipping", "supported_compute"),
    ("The explicitly stated issue date plus 30 days.", "supported_compute"),
    # Counterexamples: derivations dressed as labels, and free text, are unknown.
    ("Amount due including shipping", "unknown"),
    ("Issue date plus the net payment days", "unknown"),
    ("Purchase order number", "unknown"),
    ("Extract the explicitly stated issue date plus 30 days.", "unknown"),
    ("Extract the explicitly stated total including shipping.", "unknown"),
    ("Extract the explicitly stated expected delivery date.", "unknown"),
    ("Extrahiere die ausdrücklich genannte Summe.", "unknown"),
])
def test_instruction_forms(instructions, expected):
    spec = parse_request({"model": "clairic", "state": "x", "fields": {"value": {
        "type": "number", **({"instructions": instructions} if instructions else {})}}}).fields["value"]
    assert classify(spec)[0] == expected


def test_extract_versus_compute_counterexamples():
    extracted = plan({"total": {"type": "number", "unit": "USD",
                                "instructions": "Extract the explicitly stated total amount."}})
    assert extracted.fields[0].operation == "supported_source_only"
    instruction = "Sum of the explicitly stated subtotal and the explicitly stated shipping"
    with pytest.raises(CapabilityUnavailable) as error:
        plan({"total": {"type": "number", "unit": "USD", "instructions": instruction}})
    assert error.value.capability == "deterministic_compute"
    computed = plan({"total": {"type": "number", "unit": "USD", "instructions": instruction}}, Policy(compute=True))
    [field] = computed.fields
    assert field.operation == "supported_compute" and field.compute.operator == "sum"
    assert [o.label for o in field.compute.operands] == ["subtotal", "shipping"]
    assert all(isinstance(f, Money) for f in computed.extract_request.fields.values())
    due = plan({"due": {"type": "date", "instructions": "The explicitly stated issue date plus 30 days"}},
               Policy(compute=True)).fields[0].compute
    assert due.operator == "add_days" and due.constant == 30


@pytest.mark.parametrize("fields,extra,error,capability", [
    ({"d": {"type": "estimate", "min": 0, "max": 10}}, {}, CapabilityUnavailable, "numeric_distribution"),
    ({"d": {"type": "forecast", "min": 0, "max": 10}}, {}, CapabilityUnavailable, "forecast"),
    ({"d": {"type": "text", "null_threshold": 0.5}}, {}, CapabilityUnavailable, "typed_answerability"),
    ({"d": {"type": "text"}}, {"images": ["data:image/png;base64,AA=="]}, CapabilityUnavailable, "images"),
    ({"d": {"type": "date"}}, {"reference_date": "2026-10-19"}, UnsupportedOperation, None),
    ({"d": {"type": "text", "instructions": "Amount due including shipping"}}, {}, UnsupportedOperation, None),
])
def test_unsupported_operations_fail_at_compile_time(fields, extra, error, capability):
    with pytest.raises(error) as raised:
        plan(fields, **extra)
    assert raised.value.capability == capability


def test_estimate_with_capability_still_needs_a_wired_runtime():
    released = Capability("numeric_distribution", architecture=True, weights="w", provenance="p",
                          calibration="c", released=True)
    capabilities = extractor_capabilities(Legacy())
    capabilities.add(released)
    with pytest.raises(UnsupportedOperation, match="not wired"):
        plan({"d": {"type": "estimate", "min": 0, "max": 10}}, capabilities=capabilities)


def test_mixed_requests_are_rejected_as_a_whole_with_every_reason():
    with pytest.raises(UnsupportedOperation) as error:
        plan({"ok": {"type": "text"}, "bad": {"type": "text", "instructions": "Total incl. shipping"},
              "est": {"type": "estimate", "min": 0, "max": 1}})
    assert [d["field"] for d in error.value.details] == ["bad", "est"]


def test_reference_date_without_temporal_fields_and_include_quantiles_without_estimates():
    assert plan({"t": {"type": "text"}}, reference_date="2026-10-19",
                options={"include_quantiles": True}).fields


def test_internal_capacity_is_explicit():
    with pytest.raises(CapacityExceeded):
        plan({f"f{i}": {"type": "text"} for i in range(33)})
