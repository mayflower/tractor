from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tractor.document_pipeline import DocumentRequest, address_parts, resolve_document
from tractor.document_sources import envelope
from tractor.pipeline_evaluation import episode_counts
from tractor.server import create_app


def span(text, value, start=0):
    position = text.index(value, start)
    return {"text": value, "start": position, "end": position + len(value)}


def request(text):
    return DocumentRequest(
        text=text,
        tasks=[
            {
                "id": "office",
                "kind": "entities",
                "description": "Registered office",
                "fields": {
                    "street": "Registered office street",
                    "city": "Registered office town",
                    "postcode": "Registered office postcode",
                    "number": "Charity number",
                },
            }
        ],
        processing={
            "office": {
                key: {
                    "kind": "address",
                    "country": "GB",
                    "component": key,
                    "group": "registered_office",
                }
                for key in ("street", "city", "postcode")
            }
        },
        include_details=True,
    )


def test_address_components_use_only_same_group_evidence():
    text = "12 Cock Lane, London EC1A 9BU. Other office: York YO1 7HD."
    raw = {
        "entities": {
            "street": [span(text, "12 Cock Lane, London EC1A 9BU")],
            "city": [span(text, "London EC1A 9BU")],
            "postcode": [],
            "number": [],
        }
    }
    result = resolve_document(request(text), raw)
    assert result["data"]["office"] == {
        "street": ["12 Cock Lane"],
        "city": ["London"],
        "postcode": ["EC1A 9BU"],
        "number": [],
    }
    assert not result["issues"]
    assert result["details"]["office"]["postcode"][0]["text"] == "12 Cock Lane, London EC1A 9BU"
    assert resolve_document(request(text), {"entities": {}})["data"]["office"]["postcode"] == []


def test_conflicting_addresses_are_not_silently_selected():
    text = "London EC1A 9BU; York YO1 7HD"
    raw = {"entities": {"city": [span(text, "London EC1A 9BU"), span(text, "York YO1 7HD")]}}
    result = resolve_document(request(text), raw)
    assert result["data"]["office"]["postcode"] == []
    assert any(i["code"] == "ambiguous_address" for i in result["issues"])


def test_invalid_spans_cannot_be_used_for_recovery():
    result = resolve_document(
        request("London EC1A 9BU"),
        {"entities": {"city": [{"text": "London EC1A 9BU", "start": 99, "end": 113}]}},
    )
    assert result["data"]["office"]["postcode"] == []
    assert result["issues"][0]["code"] == "invalid_span"


def test_quantity_uses_production_decimal_normalization():
    text = "Length 12 cm"
    req = DocumentRequest(
        text=text,
        tasks=[
            {
                "id": "size",
                "kind": "entities",
                "description": "Size",
                "fields": {"length": "Length"},
            }
        ],
        processing={"size": {"length": {"kind": "quantity", "unit": "mm"}}},
    )
    result = resolve_document(req, {"entities": {"length": [span(text, "12 cm")]}})
    assert result["data"]["size"]["length"] == [Decimal("120")]


def test_component_country_must_be_explicit():
    data = request("London").model_dump()
    data["processing"]["office"]["city"]["country"] = None
    with pytest.raises(ValidationError):
        DocumentRequest.model_validate(data)


def test_address_groups_do_not_cross():
    req = DocumentRequest(
        text="London EC1A 9BU",
        tasks=[
            {
                "id": "x",
                "kind": "entities",
                "description": "Addresses",
                "fields": {
                    "billing_city": "Billing city",
                    "shipping_postcode": "Shipping postcode",
                },
            }
        ],
        processing={
            "x": {
                "billing_city": {
                    "kind": "address",
                    "country": "GB",
                    "component": "city",
                    "group": "billing",
                },
                "shipping_postcode": {
                    "kind": "address",
                    "country": "GB",
                    "component": "postcode",
                    "group": "shipping",
                },
            }
        },
    )
    result = resolve_document(req, {"entities": {"billing_city": [span(req.text, req.text)]}})
    assert result["data"]["x"]["shipping_postcode"] == []


def test_partial_or_multi_postcode_address_does_not_invent_city():
    assert "city" not in address_parts("12 Cock Lane EC1A 9BU", "street")
    assert address_parts("London EC1A 9BU York YO1 7HD", "city") == {}


def test_http_executes_same_processing_path():
    from tractor.service import extract
    from tractor.types import Address, ExtractRequest

    text = "London EC1A 9BU"

    class Backend:
        def candidates(self, req):
            if req.text != text:
                return [] if req.anchor else [{k: [] for k in req.fields}]
            return [{"city": [span(text, text)], "postcode": []}]

    req = ExtractRequest(text=text, fields={
        "city": Address("Registered office city", country="GB", component="city", group="office"),
        "postcode": Address("Registered office postcode", country="GB", component="postcode", group="office"),
    })
    with TestClient(create_app(extractor=Backend())) as client:
        response = client.post("/v1/extract", json=req.model_dump())
        assert response.status_code == 200
        assert response.json() == extract(Backend(), req)
        assert response.json()["data"] == {"city": "London", "postcode": "EC1A 9BU"}
        schema = client.get("/openapi.json").json()
        assert [p for p, v in schema["paths"].items() if "post" in v] == ["/v1/extract"]
        assert client.post("/v1/extract-document", json={}).status_code == 404


def test_final_value_scoring_does_not_require_duplicate_mentions():
    text = "London London"
    e = envelope(
        "sample",
        "x",
        "dev",
        "metadata",
        "en",
        text,
        [
            {
                "id": "document",
                "kind": "entities",
                "description": "Places",
                "fields": {"city": "City"},
            }
        ],
        {
            "document": {
                "complete": True,
                "records": [
                    {
                        "city": {
                            "status": "present",
                            "items": [span(text, "London"), span(text, "London", 1)],
                        }
                    }
                ],
            }
        },
    )
    counts = episode_counts(e, {"data": {"document": {"city": ["London"]}}, "issues": []})
    assert counts["exact_requests"] == 1 and counts["field_tp"] == 1
    assert counts["fn"] == 0


@pytest.mark.parametrize('text,selected,expected', [
    ('Sphere (1998)', 'Sphere', 'Sphere (1998)'),
    ('Sphere (1998)', 'Sphere (1998)', 'Sphere (1998)'),
    ('Sphere\n(1998)', 'Sphere', 'Sphere'),
    ('Sphere was released (1998)', 'Sphere', 'Sphere'),
    ('Sphere (0000)', 'Sphere', 'Sphere'),
    ('Sphere (1998–1999)', 'Sphere', 'Sphere'),
])
def test_explicit_adjacent_year_contract_preserves_source(text, selected, expected):
    req = DocumentRequest(text=text, tasks=[{'id': 'doc', 'kind': 'entities',
        'description': 'Displayed heading', 'fields': {'title': 'Printed title including attached year'}}],
        processing={'doc': {'title': {'kind': 'literal', 'include_adjacent_year': True}}}, include_details=True)
    raw = {'entities': {'title': [span(text, selected)]}}
    result = resolve_document(req, raw)
    assert result['data']['doc']['title'] == [expected]
    assert all(text[s['start']:s['end']] == s['text'] for s in result['details']['doc']['title'])
    req.processing = {}
    assert resolve_document(req, raw)['data']['doc']['title'] == [selected]
