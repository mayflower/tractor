"""Offline fresh-process bundle checks, executed inside an isolated container."""

import urllib.request
from decimal import Decimal

from tractor import Client, Quantity, Text
from tractor.jsonio import dumps

with Client("http://127.0.0.1:8000", timeout=30) as client:
    example = client.extract(
        "Der Rover-X fährt 10 miles pro Ladung und ist 12,5 cm breit.",
        {
            "product": Text("Produktbezeichnung"),
            "range_km": Quantity("Reichweite pro Ladung", unit="km"),
            "width_mm": Quantity("Breite", unit="mm"),
        },
        locale="de-DE",
    )
    records = client.extract(
        "Alpha: Breite 10 cm. Beta: Breite 12 cm.",
        {"product": Text("Produktbezeichnung"), "width_mm": Quantity("Breite", unit="mm")},
        many=True,
        anchor="product",
    )
    assert all(v is None or isinstance(v, (str, int, Decimal)) for v in example.data.values())
    assert example.data == {"product": "Rover-X", "range_km": Decimal("16.09344"), "width_mm": 125}
    assert example.issues == []
    assert records.data == [
        {"product": "Alpha", "width_mm": 100},
        {"product": "Beta", "width_mm": 120},
    ]
    assert records.issues == []
    for result in [example, records]:
        print(
            dumps(
                {"data": result.data, "issues": result.issues},
            )
        )
assert urllib.request.urlopen("http://127.0.0.1:8000/readyz").status == 200
print("Offline bundle loaded in a fresh process; real HTTP/SDK requests completed.")
