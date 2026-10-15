from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tractor import Quantity, Text
from tractor.jsonio import dumps, loads
from tractor.server import create_app
from tractor.service import resolve
from tractor.types import ExtractRequest


class EmptyExtractor:
    def candidates(self, request):
        return [] if request.anchor else [{k: [] for k in request.fields}]


@pytest.fixture
def client():
    with TestClient(create_app(extractor=EmptyExtractor())) as client:
        yield client


@pytest.mark.parametrize(
    "row",
    [loads(x) for x in Path("tests/fixtures/http-errors.jsonl").read_text().splitlines()],
)
def test_http_errors(client, row):
    response = client.post("/v1/extract", content=row.get("raw_body") or dumps(row["request"]))
    assert response.status_code == row["expected_status"]
    assert "error" in response.json()


@pytest.mark.parametrize(
    "row",
    [loads(x) for x in Path("tests/fixtures/api-cases.jsonl").read_text().splitlines()],
    ids=lambda r: r["id"],
)
def test_gold_span_resolution(row):
    result = resolve(ExtractRequest.model_validate(row["request"]), row["gold"]["records"])
    assert result["data"] == row["gold"]["response"]["data"]
    assert result["issues"] == row["gold"]["response"]["issues"]


def test_json_decimal_duplicates():
    assert dumps({"v": Decimal("16.09344")}) == '{"v": 16.09344}'
    with pytest.raises(ValueError):
        loads('{"v":1,"v":2}')


def test_body_limit(client):
    assert client.post("/v1/extract", content=b"x" * (128 * 1024 + 1)).status_code == 413


def test_fields():
    assert Text("Produkt").type == "text"
    assert Quantity("Breite", unit="mm").unit == "mm"
