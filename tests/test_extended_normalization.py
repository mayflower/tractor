"""Expanded types: independently specified outputs and ambiguity regressions."""
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from tractor import (
    Date,
    Datetime,
    Duration,
    Iban,
    Interval,
    Money,
    Number,
    Percent,
    Quantity,
    Time,
    Url,
)
from tractor.field_normalization import Unresolved, normalize_value
from tractor.jsonio import loads
from tractor.normalize import compatible_policy, normalize
from tractor.server import create_app
from tractor.service import extract
from tractor.types import ExtractRequest, FieldSpec

CASES = [
    (Date("Date", allow_partial=True), "03/2019", "2019-03"),
    (Date("Date", allow_partial=True), "03.2019", "2019-03"),
    (Date("Date", date_order="DMY"), "14.10.2026, 07:45 Uhr", "2026-10-14"),
    (Datetime("Timestamp", date_order="DMY"), "14.10.2026, 07:45 Uhr", "2026-10-14T07:45"),
    (Datetime("Timestamp"), "October 14, 2026 2:35 PM", "2026-10-14T14:35"),
    (Datetime("Timestamp"), "2026-10-14T07:45:12.125Z", "2026-10-14T07:45:12.125000+00:00"),
    (Datetime("Timestamp", date_order="DMY"), "14.10.2026,07:45", "2026-10-14T07:45"),
    (Time("Time"), "7:45 Uhr", "07:45"),
    (Time("Time"), "12:00 AM", "00:00"),
    (Time("Time"), "12:00 PM", "12:00"),
    (Time("Time"), "14:35:08 UTC", "14:35:08+00:00"),
    (Number("Count"), "n = 186", Decimal(186)),
    (Number("Count"), "128 participants", Decimal(128)),
    (Number("Count"), "142 Mitarbeitende", Decimal(142)),
    (Number("Count"), "n = 48 pouch cells", Decimal(48)),
    (Number("Count"), "n = 312 Sensoren aus 24 Standorten", Decimal(312)),
    (Number("Count"), "n = 412 half-hourly observations across 17 measurement days", Decimal(412)),
    (Number("Count"), "n = 60 specimens (12 formulations × 5 replicates)", Decimal(60)),
    (Number("Count"), "n = 48 (20 controls)", Decimal(48)),
    (Number("Count"), "n = 84 adults (42 per arm)", Decimal(84)),
    (Number("Count"), "n = 36 specimens (12 per batch, three batches)", Decimal(36)),
    (Number("Count"), "148 soil cores", Decimal(148)),
    (Number("Count"), "18 packages", Decimal(18)),
    (Number("Count", locale="en-US"), "1,234 employees", Decimal(1234)),
    (Money("Amount", currency="EUR"), "12EUR", Decimal(12)),
    (Money("Amount", currency="EUR"), "EUR12", Decimal(12)),
    (Money("Amount", currency="EUR", locale="de-DE"), "240 Millionen Euro", Decimal(240000000)),
    (Money("Amount", currency="USD", locale="en-US"), "MSRP\n$23,075", Decimal(23075)),
    (Money("Amount", currency="GBP", locale="en-GB"), "£3,850 per month", Decimal(3850)),
    (Money("Amount", currency="GBP", locale="en-GB"), "£58,000 per annum", Decimal(58000)),
    (Money("Amount", currency="EUR"), "125 EUR per inspection", Decimal(125)),
    (Money("Amount", currency="USD", locale="en-US"), "1.5 billion US dollars", Decimal(1500000000)),
    (Money("Amount", currency="EUR", locale="de-DE"), "1,5 Billion Euro", Decimal(1500000000000)),
    (Percent("Percent", locale="de-DE"), "4,2 %", Decimal("4.2")),
    (Percent("Percent"), "125 percent", Decimal(125)),
    (Iban("Account"), "GB29 NWBK 6016 1331 9268 19", "GB29NWBK60161331926819"),
    (Iban("Account"), "IBAN: DE44 5001 0517 5407 3249 31", "DE44500105175407324931"),
    (Url("Website"), "HTTPS://EXAMPLE.ORG/SomePath?q=Hi#Part", "https://example.org/SomePath?q=Hi#Part"),
    (Url("Website"), "www.EXAMPLE.ORG/SomePath", "www.example.org/SomePath"),
    (Url("Website"), "https://bücher.de/Ä", "https://xn--bcher-kva.de/Ä"),
    (Url("Website"), "https://EXAMPLE.ORG/?#", "https://example.org/?#"),
    (Url("Website"), "https://[2001:db8::1]:443/a", "https://[2001:db8::1]:443/a"),
    (Duration("Duration"), "3 calendar months", "P3M"),
    (Duration("Duration"), "90 days", "P90D"),
    (Duration("Duration"), "2 Stunden und 30 Minuten", "PT2H30M"),
    (Duration("Duration"), "P1Y2M", "P1Y2M"),
    (Duration("Duration"), "1.5 hours", "PT1.5H"),
    (Interval("Period", date_order="DMY"), "01.09.2026 – 30.09.2026", "2026-09-01/2026-09-30"),
    (Interval("Period"), "1 August 2026 to 31 August 2026", "2026-08-01/2026-08-31"),
    (Interval("Period"), "2026-08-01/2026-08-31", "2026-08-01/2026-08-31"),
    (Interval("Hours"), "09:00–11:00", "09:00/11:00"),
    (Interval("Period"), "2026-08-01T09:00Z/2026-08-01T11:00Z",
     "2026-08-01T09:00+00:00/2026-08-01T11:00+00:00"),
]


@pytest.mark.parametrize("field,source,expected", CASES)
def test_explicit_values(field, source, expected):
    assert normalize_value(source, field.processing_rule()) == [expected]


@pytest.mark.parametrize("field,source", [
    (Date("Date"), "03/2019"),
    (Date("Date", allow_partial=True), "13/2019"),
    (Date("Date", allow_partial=True), "03/0000"),
    (Datetime("Timestamp"), "03/04/2026 10:00"),
    (Datetime("Timestamp"), "tomorrow 10:00"),
    (Datetime("Timestamp"), "March 2026 10:00"),
    (Datetime("Timestamp"), "2026-02-30T10:00"),
    (Time("Time"), "14:00 CET"),
    (Time("Time"), "24:00"),
    (Time("Time"), "14:00+00:99"),
    (Time("Time"), "14:00 PM"),
    (Number("Count"), "12 participants and 3 controls"),
    (Number("Count"), "approximately 128 participants"),
    (Number("Count"), "12 × 5"),
    (Number("Count"), "48 (20 controls)"),
    (Number("Count"), "n = 48 or 50 participants"),
    (Number("Count"), "n = 48 participants (n = 20 controls)"),
    (Number("Count"), "n = 48 participants plus 5 controls"),
    (Number("Count"), "n = 48 participants (estimated)"),
    (Number("Count"), "n = 48 participants (at least 20 controls)"),
    (Number("Count"), "n = 48 + 5"),
    (Number("Count"), "-5 employees"),
    (Number("Count"), "1.5 employees"),
    (Money("Amount", currency="EUR"), "240 EUR or 250 EUR"),
    (Money("Amount", currency="EUR"), "1 EUR 2"),
    (Money("Amount", currency="EUR"), "1 billion EUR"),
    (Money("Amount", currency="EUR"), "about 240 EUR"),
    (Money("Amount", currency="EUR"), "240 EUR plus VAT"),
    (Money("Amount", currency="EUR"), "240 pounds"),
    (Money("Amount", currency="EUR"), "1e100 million EUR"),
    (Percent("Percent"), "4.2"),
    (Percent("Percent"), "4-6%"),
    (Percent("Percent"), "10 basis points"),
    (Iban("Account"), "GB29 NWBK 6016 1331 9269 19"),
    (Iban("Account"), "GB29NWBK60161331926819 / DE44500105175407324931"),
    (Url("Website"), "javascript:alert(1)"),
    (Url("Website"), "https://example.org/a b"),
    (Url("Website"), "https://example.org:99999"),
    (Url("Website"), "https://example.org/%ZZ"),
    (Url("Website"), "contact@example.org"),
    (Url("Website"), "example.org.attacker\n.evil"),
    (Duration("Duration"), "about 3 months"),
    (Duration("Duration"), "3 months or 90 days"),
    (Duration("Duration"), "2 months 3 months"),
    (Duration("Duration"), "P"),
    (Duration("Duration"), "P1YT"),
    (Duration("Duration"), "P1.5Y2M"),
    (Duration("Duration"), "P2W1D"),
    (Interval("Period"), "fiscal year ended 31 March 2025"),
    (Interval("Period"), "2026-10-19/today"),
    (Interval("Period"), "2026-10-19/2026-09-19"),
    (Interval("Hours"), "23:00–01:00"),
    (Interval("Period"), "2026-10-19T09:00Z/2026-10-19T11:00"),
])
def test_ambiguity_and_unstated_facts_are_not_repaired(field, source):
    with pytest.raises(Unresolved):
        normalize_value(source, field.processing_rule())


def test_rate_suffix_does_not_supply_a_number_locale():
    with pytest.raises(Unresolved) as error:
        normalize_value("£58,000 per annum", Money("Salary", currency="GBP").processing_rule())
    assert str(error.value) == "ambiguous_number_format"


@pytest.mark.parametrize('text,value,issue', [
    ('<td>1,4 kg</td>', Decimal('1.4'), None),
    ('<b>1,4 kg</b>', Decimal('1.4'), None),
    ('<td>>1,4 kg</td>', None, 'non_scalar_quantity'),
    ('> 1,4 kg', None, 'non_scalar_quantity'),
    ('at least <b>1,4 kg</b>', None, 'non_scalar_quantity'),
])
def test_quantity_markup_boundary_is_not_a_comparison(text, value, issue):
    start = text.index('1,4 kg')
    result = normalize(text, {'text': '1,4 kg', 'start': start, 'end': start + 6}, 'kg', 'de-DE')
    assert result.value == value and result.issue == issue


@pytest.mark.parametrize("source,target,expected", [
    ("5 nm", "m", "0.000000005"), ("31.4 nm", "um", "0.0314"),
    ("812 MPa", "Pa", "812000000"), ("500 µg", "mg", "0.5"),
    ("20 °C", "K", "293.15"), ("20 °C", "degF", "68"), ("32 °F", "degC", "0"),
])
def test_additional_units_independent_targets(source, target, expected):
    result = normalize(source, {"text": source, "start": 0, "end": len(source)}, target, "en-US")
    assert result.issue is None
    assert abs(result.value - Decimal(expected)) < Decimal("1e-30")


def test_bundle_accepts_only_compatible_additions():
    policy = loads((Path(__file__).parent / "fixtures/normalization-policy.json").read_text())
    assert compatible_policy(policy)
    policy["aliases"]["cm"] = "m"
    assert not compatible_policy(policy)


def test_all_new_types_share_public_http_processing():
    selected = {}
    for field, source, expected in CASES:
        selected.setdefault(field.type, (field, source, expected))
    selected = list(selected.values())
    text = " | ".join(source for _, source, _ in selected)
    fields = {f"field{i}": field for i, (field, _, _) in enumerate(selected)}
    row = {}
    position = 0
    for i, (_, source, _) in enumerate(selected):
        start = text.index(source, position)
        row[f"field{i}"] = [{"text": source, "start": start, "end": start + len(source)}]
        position = start + len(source)

    class Backend:
        def candidates(self, request):
            if request.text == text:
                return [row]
            return [] if request.anchor else [{key: [] for key in request.fields}]

    request = ExtractRequest(text=text, fields=fields)
    result = extract(Backend(), request)
    assert result["data"] == {f"field{i}": value for i, (_, _, value) in enumerate(selected)}
    assert result["issues"] == []
    with TestClient(create_app(extractor=Backend())) as client:
        response = client.post("/v1/extract", json=request.model_dump())
        assert response.status_code == 200
        assert loads(response.content) == result
        schema = client.get("/openapi.json").json()
        assert [path for path, methods in schema["paths"].items() if "post" in methods] == ["/v1/extract"]
    from tractor.document_sources import envelope
    from tractor.pipeline_evaluation import infer, task_counts
    episode = envelope("example", "extended-types", "dev", "operations", "en", text,
        [{"id": "document", "kind": "entities", "description": "Explicit values",
          "fields": {key: field.description for key, field in fields.items()}}],
        {"document": {"complete": True, "records": [{
            key: {"status": "present", "items": row[key], "final_values": [str(selected[i][2])]}
            for i, key in enumerate(fields)}]}})
    episode.processing = {"document": {key: field.processing_rule() for key, field in fields.items()}}
    prediction = infer(Backend(), episode)
    assert task_counts(episode, episode.tasks[0], prediction)["exact_fields"] == len(fields)
    assert all("_" not in name for name in TypeAdapter(FieldSpec).json_schema()["discriminator"]["mapping"])


def test_quantity_schema_accepts_new_units():
    assert Quantity("Temperature", unit="degC").processing_rule().unit == "degC"
