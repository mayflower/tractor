from tractor import Client, Quantity, Text

fields = {
    "product": Text("Produktbezeichnung"),
    "range_km": Quantity("Reichweite pro Ladung", unit="km"),
    "width_mm": Quantity("Breite", unit="mm"),
}
with Client() as client:
    result = client.extract(
        "Der Rover-X fährt 10 miles pro Ladung und ist 12,5 cm breit.",
        fields,
        locale="de-DE",
        include_details=True,
    )
    print(result)
    records = client.extract(
        "Alpha: Breite 10 cm. Beta: Breite 12 cm.",
        {"product": Text("Produktbezeichnung"), "width_mm": Quantity("Breite", unit="mm")},
        many=True,
        anchor="product",
    )
    print(records)
    uncertain = client.extract("Breite 1,234 cm.", {"width_mm": Quantity("Breite", unit="mm")})
    for issue in uncertain.issues:
        print(issue)
