"""Run against a ready local API: uv run --no-sync python examples/manual/python_client.py."""

import os
from decimal import Decimal

from tractor import Client, Quantity, Text
from tractor.jsonio import dumps

fields = {"product": Text("Produktbezeichnung"), "width_mm": Quantity("Breite", unit="mm")}
with Client(os.environ.get("TRACTOR_URL", "http://127.0.0.1:8000"), timeout=10) as client:
    records = client.extract(
        "Alpha ist 10 cm breit. Beta ist 2 inches breit.", fields,
        anchor="product", many=True, include_details=True,
    )
    assert records.data == [
        {"product": "Alpha", "width_mm": 100},
        {"product": "Beta", "width_mm": Decimal("50.8")},
    ]
    assert not records.issues
    print(dumps({"data": records.data, "issues": records.issues}))
    text = "📦 Atlas ist 12,5 cm breit."
    evidence = client.extract(text, fields, locale="de-DE", include_details=True)
    source = evidence.details["width_mm"]
    assert text[source["start"]:source["end"]] == source["text"] == "12,5 cm"
    assert evidence.data["width_mm"] == 125
    print(dumps({"data": evidence.data, "details": evidence.details}))
    missing = client.extract(
        "Modell CM125 hat keine Maßangaben.", {"width_mm": fields["width_mm"]},
    )
    assert missing.data["width_mm"] is None
    assert missing.issues == [{"field": "width_mm", "code": "not_found"}]
    print(dumps({"data": missing.data, "issues": missing.issues}))
