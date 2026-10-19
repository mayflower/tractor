"""Conservative contact normalization and public pipeline behavior."""
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tractor import Address, Email, Organization, Person, Phone, Place, Postcode
from tractor.document_sources import envelope
from tractor.field_normalization import Unresolved, normalize_value
from tractor.pipeline_evaluation import episode_counts, infer
from tractor.server import create_app
from tractor.service import extract
from tractor.types import ExtractRequest


@pytest.mark.parametrize(("field", "source", "expected"), [
    (Person("Person"), "  Jose\u0301\u00a0 van  Dijk ", "José van Dijk"),
    (Person("Person"), "Dr. Anne-Marie O’Neill", "Dr. Anne-Marie O’Neill"),
    (Organization("Company"), "  ACME\nGmbH & Co. KG ", "ACME GmbH & Co. KG"),
    (Place("Town"), " Frankfurt   am Main ", "Frankfurt am Main"),
    (Phone("Phone", country="DE"), "030 901820", "+4930901820"),
    (Phone("Phone"), "+44 20 8366 1177", "+442083661177"),
    (Phone("Phone", country="GB"), "020 8366 1177 ext. 42", "+442083661177;ext=42"),
    (Email("Email"), "Person@EXAMPLE.ORG", "Person@example.org"),
    (Postcode("ZIP", country="DE"), "01067", "01067"),
    (Postcode("ZIP", country="DE"), "04107 Leipzig", "04107"),
    (Postcode("ZIP", country="DE"), "01067 Dresden, Straße", "01067"),
    (Postcode("ZIP", country="GB"), "London EC1A 9BU", "EC1A 9BU"),
    (Postcode("ZIP", country="GB"), "ec1a9bu", "EC1A 9BU"),
    (Postcode("ZIP", country="US"), "021391234", "02139-1234"),
    (Postcode("ZIP", country="CA"), "k1a0b1", "K1A 0B1"),
    (Postcode("ZIP", country="NL"), "1012ab", "1012 AB"),
    (Address("Address"), "Lindenweg 14\n04107 Leipzig", "Lindenweg 14, 04107 Leipzig"),
    (Address("Address"), "Lindenweg 14,   04107 Leipzig", "Lindenweg 14, 04107 Leipzig"),
])
def test_normalizes_explicit_source_and_is_idempotent(field, source, expected):
    assert normalize_value(source, field.processing_rule()) == [expected]
    assert normalize_value(expected, field.processing_rule()) == [expected]


@pytest.mark.parametrize(("field", "source", "code"), [
    (Phone("Phone"), "030 901820", "missing_phone_country"),
    (Phone("Phone", country="DE"), "123", "invalid_phone"),
    (Phone("Phone"), "+44 20 8366 1177 and call tomorrow", "invalid_phone"),
    (Email("Email"), "person@@example.org", "invalid_email"),
    (Email("Email"), "person@example.org other@example.org", "invalid_email"),
    (Postcode("ZIP", country="DE"), "1067", "invalid_postal_code"),
    (Postcode("ZIP", country="DE"), "01067 Dresden / 04107 Leipzig", "ambiguous_postal_code"),
    (Postcode("ZIP", country="DE"), "ID01067", "invalid_postal_code"),
    (Postcode("ZIP", country="DE"), "01067-1234", "invalid_postal_code"),
    (Address("Address"), ",,,", "empty_value"),
    (Postcode("ZIP", country="GB"), "London", "invalid_postal_code"),
    (Postcode("ZIP", country="ZZ"), "12345", "unsupported_postal_country"),
    (Person("Person"), "12345", "invalid_name"),
    (Place("Town"), "Lei\u200bpzig", "invalid_text_character"),
])
def test_invalid_or_ambiguous_input_is_not_guessed(field, source, code):
    with pytest.raises(Unresolved, match=code):
        normalize_value(source, field.processing_rule())


def span(text, source):
    start = text.index(source)
    return {"start": start, "end": start + len(source), "text": source}


class Backend:
    def __init__(self, text, rows):
        self.text, self.rows = text, rows

    def candidates(self, request):
        if request.text == self.text:
            return self.rows
        return [] if request.anchor else [{k: [] for k in request.fields}]


@pytest.mark.parametrize(("field", "values", "expected", "ambiguous"), [
    (Place("Town"), ["LEIPZIG", "Leipzig"], "LEIPZIG", False),
    (Place("Town"), ["Frankfurt", "Frankfurt am Main"], None, True),
    (Person("Person"), ["Anne Müller", "Anne Muller"], None, True),
    (Organization("Company"), ["ACME GmbH", "ACME AG"], None, True),
    (Phone("Phone", country="GB"), ["020 8366 1177", "+44 20 8366 1177"], "+442083661177", False),
    (Phone("Phone", country="GB"), ["020 8366 1177 ext. 1", "020 8366 1177 ext. 2"], None, True),
    (Email("Email"), ["Person@EXAMPLE.ORG", "Person@example.org"], "Person@example.org", False),
    (Email("Email"), ["Person@example.org", "person@example.org"], None, True),
    (Address("Address"), ["Lindenweg 14\n04107 Leipzig", "Lindenweg 14, 04107 Leipzig"],
     "Lindenweg 14, 04107 Leipzig", False),
])
def test_only_declared_equivalences_deduplicate(field, values, expected, ambiguous):
    text = " | ".join(values)
    backend = Backend(text, [{"value": [span(text, value) for value in values]}])
    request = ExtractRequest(text=text, fields={"value": field})
    response = extract(backend, request)
    assert response["data"] == {"value": expected}
    assert response["issues"] == ([{"field": "value", "code": "ambiguous_value"}] if ambiguous else [])
    assert set(response) == {"data", "issues"}


def test_http_normalizes_all_contact_types_and_rejects_evidence_option():
    text = "José van Dijk | ACME GmbH | Leipzig | 030 901820 | Person@EXAMPLE.ORG | 01067 | Lindenweg 14\n04107 Leipzig"
    fields = {"person": Person("Person"), "company": Organization("Company"),
              "city": Place("City"), "phone": Phone("Phone", country="DE"), "email": Email("Email"),
              "zip": Postcode("Postal code", country="DE"), "address": Address("Address")}
    values = text.split(" | ")
    request = ExtractRequest(text=text, fields=fields)
    backend = Backend(text, [{k: [span(text, v)] for k, v in zip(fields, values, strict=True)}])
    with TestClient(create_app(extractor=backend)) as client:
        response = client.post("/v1/extract", json=request.model_dump())
        assert response.status_code == 200
        assert response.json() == {"data": {
            "person": "José van Dijk", "company": "ACME GmbH", "city": "Leipzig",
            "phone": "+4930901820", "email": "Person@example.org", "zip": "01067",
            "address": "Lindenweg 14, 04107 Leipzig"}, "issues": []}
        assert client.post("/v1/extract", json={**request.model_dump(), "include_details": True}).status_code == 422
        schema = client.get("/openapi.json").json()["paths"]["/v1/extract"]["post"]
        assert "include_details" not in str(schema)
        assert "details" not in schema["responses"]["200"]["content"]["application/json"]["schema"]["properties"]


def test_evaluation_compares_same_typed_equivalence_without_gold_guided_normalization():
    text = "LEIPZIG"
    episode = envelope("example", "city", "dev", "contacts", "de", text,
                       [{"id": "document", "kind": "entities", "description": "Contact",
                         "fields": {"city": "City"}}],
                       {"document": {"complete": True, "records": [{
                           "city": {"status": "present", "items": [span(text, text)],
                                    "final_values": ["Leipzig"]}}]}})
    from tractor.field_contracts import Processing
    episode.processing = {"document": {"city": Processing(kind="place")}}
    backend = Backend(text, [{"city": [span(text, text)]}])
    prediction = infer(backend, episode)
    assert prediction["data"]["document"]["city"] == ["LEIPZIG"]
    from tractor.pipeline_evaluation import task_counts
    assert task_counts(episode, episode.tasks[0], prediction)["exact_fields"] == 1
    changed_gold = deepcopy(episode)
    changed_gold.gold["document"].records[0]["city"].final_values = ["Berlin"]
    assert infer(backend, changed_gold) == prediction
    assert task_counts(changed_gold, changed_gold.tasks[0], prediction)["exact_fields"] == 0
    assert episode_counts(episode, prediction)  # Full evaluator accepts the result.


def test_name_anchor_normalizes_without_merging_distinct_records():
    text = " Anne   Müller | Anne Müller "
    request = ExtractRequest(text=text, fields={"person": Person("Person")},
                             anchor="person", many=True)
    backend = Backend(text, [{"person": [span(text, "Anne   Müller")]},
                             {"person": [span(text, "Anne Müller")]}])
    assert extract(backend, request)["data"] == [{"person": "Anne Müller"}, {"person": "Anne Müller"}]


def test_missing_postal_country_rejected():
    with pytest.raises(ValidationError):
        Postcode("Postal code")
