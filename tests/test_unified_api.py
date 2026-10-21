"""Public type selection and HTTP/SDK/evaluation parity, without model fixtures."""
from decimal import Decimal

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tractor import Client, Date, Identifier, Money, Quantity, Text
from tractor.document_sources import envelope
from tractor.pipeline_evaluation import infer, typed_requests_for
from tractor.server import create_app
from tractor.service import extract
from tractor.types import ExtractRequest


def span(text, value, start=0):
    offset = text.index(value, start)
    return {"text": value, "start": offset, "end": offset + len(value)}


class Evidence:
    def __init__(self, text, rows):
        self.text, self.rows = text, rows

    def candidates(self, request):
        if request.text != self.text:  # Server warmup.
            return [] if request.anchor else [{k: [] for k in request.fields}]
        assert isinstance(request, ExtractRequest)
        return self.rows


def test_types_select_normalization_and_scalar_deduplication():
    text = "24.10.2026; EUR 1.234,50; 12 cm; 120 mm; AB 0042"
    request = ExtractRequest(text=text, locale="de-DE", fields={
        "day": Date("Date", date_order="DMY"),
        "amount": Money("Amount", currency="EUR"),
        "length": Quantity("Length", unit="mm"),
        "id": Identifier("Identifier", remove_spaces=True),
        "literal": Text("Literal date"),
    })
    backend = Evidence(text, [{
        "day": [span(text, "24.10.2026")], "amount": [span(text, "EUR 1.234,50")],
        "length": [span(text, "12 cm"), span(text, "120 mm")],
        "id": [span(text, "AB 0042")], "literal": [span(text, "24.10.2026")],
    }])
    result = extract(backend, request)
    assert result["data"] == {"day": "2026-10-24", "amount": Decimal("1234.50"),
                              "length": Decimal("120"), "id": "AB0042", "literal": "24.10.2026"}
    assert result["issues"] == []
    assert set(result) == {"data", "issues"}
    assert len(extract(backend, request, _diagnostics=True)["details"]["length"]["evidence"]) == 2
    with TestClient(create_app(extractor=backend)) as http:
        assert http.post("/v1/extract", json=request.model_dump()).json()["data"]["day"] == "2026-10-24"
        with Client() as sdk:
            sdk._http.close()
            sdk._http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(
                lambda req: httpx.Response(200, content=http.post("/v1/extract", content=req.content).content)))
            assert sdk.extract(text, request.fields, locale="de-DE").data == result["data"]
        paths = http.get("/openapi.json").json()["paths"]
        assert [path for path, value in paths.items() if "post" in value] == ["/v1/extract"]
        assert http.post("/v1/extract-document", json={}).status_code == 404
    assert not hasattr(Client, "extract_document")


@pytest.mark.parametrize("field", [
    {"description": "Date"},  # type is mandatory on the wire
    {"type": "unknown", "description": "Unknown"},
    {"type": "money", "description": "Amount"},  # missing currency
    {"type": "text", "description": "Text", "date_order": "DMY"},
    {"type": "date", "description": "Date", "unit": "mm"},
])
def test_incomplete_or_incompatible_field_contracts_fail(field):
    with pytest.raises(ValidationError):
        ExtractRequest(text="anything", fields={"value": field})


def test_old_separate_processing_contract_rejected():
    with pytest.raises(ValidationError):
        ExtractRequest(text="x", fields={"value": Text("Text")}, processing={})


def test_distinct_normalized_values_remain_ambiguous():
    text = "12 cm; 13 cm"
    request = ExtractRequest(text=text, fields={"length": Quantity("Length", unit="mm")})
    result = extract(Evidence(text, [{"length": [span(text, "12 cm"), span(text, "13 cm")]}]), request)
    assert result == {"data": {"length": [120, 130]}, "issues": [{"field": "length", "code": "ambiguous_value"}]}


def test_evaluation_and_http_share_record_service_and_preserve_records():
    text = "A 24.10.2026 B 25.10.2026"
    rows = [{"name": [span(text, "A")], "day": [span(text, "24.10.2026")]},
            {"name": [span(text, "B")], "day": [span(text, "25.10.2026")]}]
    gold = [{"name": {"status": "present", "items": row["name"]},
             "day": {"status": "present", "items": row["day"]}} for row in rows]
    episode = envelope("example", "parity", "dev", "metadata", "de", text,
                       [{"id": "rows", "kind": "records", "description": "Records",
                         "fields": {"name": "Name", "day": "Date"}, "anchor": "name"}],
                       {"rows": {"complete": True, "records": gold}})
    from tractor.field_contracts import Processing
    episode.processing = {"rows": {"day": Processing(kind="date", date_order="DMY")}}
    request = typed_requests_for(episode)["rows"]
    backend = Evidence(text, rows)
    expected = [{"name": ["A"], "day": ["2026-10-24"]}, {"name": ["B"], "day": ["2026-10-25"]}]
    assert infer(backend, episode)["data"]["rows"] == expected
    with TestClient(create_app(extractor=backend)) as http:
        assert http.post("/v1/extract", json=request.model_dump()).json()["data"] == expected
    # The expected annotations are not consulted when choosing types or normalizing.
    episode.gold = {}
    assert infer(backend, episode)["data"]["rows"] == expected
