"""Versioned reliability corpus: independent factors, orthogonal sampling, held-out layouts."""

import hashlib
import random
from collections import Counter
from copy import deepcopy
from decimal import Decimal, localcontext
from pathlib import Path

from .data import DIMENSIONS, render, validate_rows
from .jsonio import dumps

VERSION = "synthetic"
# Each complete layout belongs to one split. Languages are stratified before splitting.
LAYOUTS = {
    "de": [
        "Produkt {product}\n{facts}",
        "Zum Artikel {product} wurden folgende Daten notiert: {facts}",
        "[ {product} ]\nTechnische Angaben: {facts}",
        "Bitte prüfen Sie {product}. Die Spezifikation lautet: {facts}",
        "Modellbezeichnung: {product}\nMesswerte: {facts}",
        "Für unsere Bestellung von {product} gilt: {facts}",
        "Prüfling {product}:\n{facts}\nPrüfung abgeschlossen.",
        "Eintrag im Warenverzeichnis — {product}\n{facts}",
        "Wir haben {product} vermessen. Ergebnis: {facts}",
        "Liefergegenstand {product}\n{facts}\nFreigabe erteilt.",
        "Kundennotiz zu {product}: {facts}",
        "Die Dokumentation nennt {product} mit diesen Eigenschaften: {facts}",
        "Messkarte / Bezeichnung {product} / {facts}",
        "Auszug aus der Inventarliste:\nArtikel {product}\n{facts}",
        "Anfrage beantwortet: Bei {product} sind die Daten wie folgt: {facts}",
        "Baugruppe {product} wurde erfasst; {facts}",
        "Protokollanfang\nObjekt: {product}\n{facts}\nProtokollende",
        "Als Variante ist {product} aufgeführt. Dazu gehört: {facts}",
    ],
    "en": [
        "Product {product}\n{facts}",
        "The following specifications apply to item {product}: {facts}",
        "[ {product} ]\nTechnical details: {facts}",
        "Please inspect {product}. Its specification reads: {facts}",
        "Model designation: {product}\nMeasurements: {facts}",
        "Our order for {product} specifies: {facts}",
        "Test specimen {product}:\n{facts}\nInspection complete.",
        "Catalogue entry — {product}\n{facts}",
        "We measured {product}. Results: {facts}",
        "Shipment item {product}\n{facts}\nApproved for delivery.",
        "Customer note regarding {product}: {facts}",
        "The documentation lists {product} with these properties: {facts}",
        "Measurement card / designation {product} / {facts}",
        "Excerpt from inventory:\nItem {product}\n{facts}",
        "Enquiry resolved: the details for {product} are as follows: {facts}",
        "Assembly {product} was recorded; {facts}",
        "Start of record\nObject: {product}\n{facts}\nEnd of record",
        "The listed variant is {product}. Associated specifications: {facts}",
    ],
}
# Fact syntax is also held out; variants never migrate between splits.
FACTS = {
    "train": ["{label}: {quantity}", "{label} = {quantity}", "{quantity} ({label})"],
    "dev": ["{label} beträgt {quantity}", "{label} measures {quantity}"],
    "test": ["{label} → {quantity}", "{label} [{quantity}]"],
}
NAMES = {
    "train": ["Atlas", "Rover", "Sigma", "Komet", "Cedar", "Nimbus", "Orion", "Mistral"],
    "dev": ["Vega", "Birke", "Delta", "Quartz"],
    "test": ["Lyra", "Erle", "Helix", "Topaz"],
}
# Same-dimension roles are intentional semantic distractors, with independent target units.
ROLES = [
    [("Breite", "width"), ("Länge", "length"), ("Höhe", "height"), ("Tiefe", "depth")],
    [
        ("Nettogewicht", "net weight"),
        ("Bruttogewicht", "gross weight"),
        ("Verpackungsgewicht", "packaging weight"),
    ],
    [("Laufzeit", "run time"), ("Ladezeit", "charge time"), ("Wartezeit", "waiting time")],
    [
        ("Tankvolumen", "tank volume"),
        ("Behältervolumen", "container volume"),
        ("Verpackungsvolumen", "packaging volume"),
    ],
    [("Nutzfläche", "usable area"), ("Grundfläche", "base area"), ("Deckfläche", "top area")],
    [
        ("Fahrgeschwindigkeit", "travel speed"),
        ("Rückfahrgeschwindigkeit", "reverse speed"),
        ("Transportgeschwindigkeit", "transport speed"),
    ],
]
SOURCES = [list(d[4]) for d in DIMENSIONS]
SOURCES[0] += [
    ("ft", "0.3048"),
    ("yd", "0.9144"),
    ("nmi", "1852"),
    ("Zoll", "0.0254"),
    ("centimeters", "0.01"),
]
SOURCES[1] += [("mg", "0.000001"), ("pounds", "0.45359237")]
SOURCES[2] += [("Sekunden", "1"), ("minutes", "60")]
SOURCES[3] += [("litres", "1"), ("m³", "1000")]
TARGETS = [list(d[5]) for d in DIMENSIONS]
TARGETS[0] += [
    ("in", "0.0254"),
    ("ft", "0.3048"),
    ("yd", "0.9144"),
    ("mile", "1609.344"),
    ("nmi", "1852"),
]
TARGETS[1] += [("lb", "0.45359237")]
# Explicit mixture: half ordinary valid values, the rest hard semantics and abstention.
SCENARIOS = (
    ["scalar"] * 18
    + ["missing_role", "missing_record_value", "correction", "same_name", "adjective"] * 3
    + [
        "zero",
        "negative",
        "scientific",
        "grouped",
        "space_grouped",
        "unicode_minus",
        "no_locale",
        "attached_unit",
        "mixed_dimensions",
        "wide_schema",
        "many_records",
        "cross_language",
        "text_only",
        "no_records",
        "competing",
        "repeated_equal",
        "qualified",
        "range",
        "tolerance",
        "dimensions",
        "missing_unit",
        "unsupported_unit",
        "ambiguous_unit",
        "incompatible_dimension",
        "fraction",
        "ambiguous_number",
        "invalid_number",
        "overflow",
    ]
)


def converted(value, source_factor, target_factor):
    """Independent SI arithmetic; never call production normalization to create gold."""
    with localcontext() as ctx:
        ctx.prec = 50
        result = value * Decimal(source_factor) / Decimal(target_factor)
        ctx.prec = 34
        return +result


def quantity(rng, dimension, locale, scenario):
    unit, factor = rng.choice(SOURCES[dimension])
    value = Decimal(rng.randrange(1, 999999)) / 100
    decimal = "," if locale == "de-DE" else "."
    if scenario in {"negative", "unicode_minus"}:
        value = -value
    if scenario == "zero":
        value = Decimal(0)
    if scenario == "no_locale":
        value = Decimal(rng.randrange(1, 9999))
    literal = str(value).replace(".", decimal)
    if scenario == "scientific":
        literal += "e-2"
        value /= 100
    if scenario in {"grouped", "space_grouped"}:
        value = Decimal(rng.randrange(100000, 999999)) / 100
        literal = f"{value:,.2f}"
        if locale == "de-DE":
            literal = literal.translate(str.maketrans(",.", ".,"))
        if scenario == "space_grouped":
            literal = literal.replace(
                "." if locale == "de-DE" else ",", rng.choice([" ", "\u00a0", "\u202f"])
            )
    if scenario == "unicode_minus":
        literal = literal.replace("-", "−")
    token = literal + ("" if scenario == "attached_unit" else " ") + unit
    issue = None
    if scenario == "qualified":
        token = rng.choice(["bis zu ", "mindestens ", "ca. ", "up to ", "about ", "≤ "]) + token
        issue = "non_scalar_quantity"
    elif scenario in {"range", "tolerance", "dimensions"}:
        join = {"range": "–", "tolerance": " ± ", "dimensions": " × "}[scenario]
        token = literal + join + "2 " + unit
        issue = "non_scalar_quantity"
    elif scenario == "missing_unit":
        token, issue = literal, "missing_unit"
    elif scenario == "unsupported_unit":
        token, issue = literal + " " + rng.choice(["angstrom", "rad", "psi", "oz"]), "unsupported_unit"
    elif scenario == "ambiguous_unit":
        token, issue = literal + " " + rng.choice(["gal", "Pfund"]), "ambiguous_unit"
    elif scenario == "incompatible_dimension":
        token, issue = literal + (" kg" if dimension != 1 else " cm"), "incompatible_dimension"
    elif scenario == "fraction":
        token, issue = "1/2 " + unit, "unsupported_number_format"
    elif scenario == "ambiguous_number":
        token, issue = "1,234 " + unit, "ambiguous_number_format"
    elif scenario == "invalid_number":
        token, issue = "1,2,3 " + unit, "invalid_number_format"
    elif scenario == "overflow":
        token, issue = "1e101 " + unit, "numeric_out_of_range"
    return token, value, factor, issue


def make_row(rng, split, index, language, family, scenario):
    en = language == "en"
    locale = rng.choice(["en-US", "en-GB"]) if en else "de-DE"
    if scenario in {"ambiguous_number", "no_locale"}:
        locale = None
    dimension = rng.randrange(6)
    if scenario in {"adjective", "missing_role"}:
        dimension = 0
    roles = rng.sample(range(len(ROLES[dimension])), rng.randint(1, 3))
    if scenario in {"adjective", "missing_role"}:
        roles = [0, 1]  # width requested, length present in the hard negative.
    anchored = rng.random() < 0.55
    if scenario in {"same_name", "missing_record_value", "no_records"}:
        anchored = True
    many = anchored and (
        rng.random() < 0.85 or scenario in {"same_name", "missing_record_value", "no_records"}
    )
    count = rng.randint(1, 3) if many else 1
    if scenario in {"same_name", "missing_record_value"}:
        count = rng.randint(2, 3)
    if scenario == "many_records":
        anchored = many = True
        count, roles = rng.randint(4, 6), [rng.choice(roles)]
    if scenario == "wide_schema":
        count = 1
    anchor = "p" + str(rng.randrange(1000))
    fields = {
        anchor: {
            "type": "text",
            "description": "Product designation" if en else "Produktbezeichnung",
        }
    }
    specs = []
    mixed = rng.sample(range(6), len(roles))
    choices = [
        (mixed[i] if scenario == "mixed_dimensions" else dimension, role)
        for i, role in enumerate(roles)
    ]
    if scenario == "wide_schema":
        choices = rng.sample(
            [(d, r) for d in range(6) for r in range(len(ROLES[d]))], rng.randint(4, 7)
        )
    for dim, role in choices:
        r = role % len(ROLES[dim])
        label = ROLES[dim][r][int(en)]
        key = "f" + str(rng.randrange(1000000))
        while key in fields:
            key = "f" + str(rng.randrange(1000000))
        target, target_factor = rng.choice(TARGETS[dim])
        description = rng.choice(
            [
                label,
                ("Specified " if en else "Angegebene ") + label,
                ("Product " if en else "Produkt: ") + label,
            ]
        )
        if scenario == "cross_language":
            description = ROLES[dim][r][int(not en)]
        fields[key] = {"type": "quantity", "description": description, "unit": target}
        specs.append((key, label, dim, target_factor))
    if scenario == "text_only":
        fields = {anchor: fields[anchor]}
    elif not anchored and rng.random() < 0.5:
        del fields[anchor]
    keys = list(fields)
    rng.shuffle(keys)
    fields = {key: fields[key] for key in keys}
    text = rng.choice(["", "🔧 ", "Übersicht\n" if not en else "Overview\n", "📦\n"])
    records, outputs, issues = [], [], []
    product = None
    for ri in range(count):
        if ri == 0 or scenario != "same_name":
            product = rng.choice(NAMES[split]) + "-" + str(rng.randrange(10000000))
        pieces, local_spans, values, codes = [], {}, {anchor: product}, {}
        for si, (key, label, dim, target_factor) in enumerate(specs):
            special = scenario if si == 0 else ("no_locale" if locale is None else "scalar")
            token, value, factor, issue = quantity(rng, dim, locale, special)
            absent = (scenario == "missing_role" and si == 0) or (
                scenario == "missing_record_value" and ri == count - 1
            )
            if absent:
                # No width label at all is a strong negative against borrowing length.
                if scenario == "missing_role":
                    local_spans[key], values[key], codes[key] = [], None, "not_found"
                    continue
                token = "not specified" if en else "nicht angegeben"
            prefix = (
                ("not 1 cm, but " if en else "nicht 1 cm, sondern ")
                if scenario == "correction" and si == 0
                else ""
            )
            pattern = rng.choice(FACTS[split])
            if split == "dev":
                pattern = FACTS[split][int(en)]
            if scenario == "adjective" and si == 0:
                pattern = "{quantity} wide" if en else "{quantity} breit"
            chunk, spans = render(pattern, {"label": label, "quantity": prefix + token})
            if prefix:
                spans["quantity"]["start"] += len(prefix)
                spans["quantity"]["text"] = token
            found = [] if absent else [spans["quantity"]]
            competing_value = None
            if scenario in {"competing", "repeated_equal"} and si == 0:
                if scenario == "competing":
                    other_unit, other_factor = rng.choice(SOURCES[dim])
                    other = "2 " + other_unit
                else:
                    other = token
                extra, more = render(
                    "; {label}: {quantity}", {"label": label, "quantity": other}, len(chunk)
                )
                chunk += extra
                found.append(more["quantity"])
                if scenario == "competing":
                    issue = "ambiguous_value"
                    competing_value = converted(Decimal(2), other_factor, target_factor)
            pieces.append((key, chunk, found))
            values[key] = None if issue or absent else converted(value, factor, target_factor)
            if competing_value is not None:
                # Both candidates stay in the final data, in source order, next to the issue.
                values[key] = [converted(value, factor, target_factor), competing_value]
            if issue or absent:
                codes[key] = "not_found" if absent else issue
        rng.shuffle(pieces)
        facts = ""
        for key, chunk, spans in pieces:
            if facts:
                facts += rng.choice(["; ", "\n", " | "])
            local_spans[key] = [
                dict(s, start=s["start"] + len(facts), end=s["end"] + len(facts)) for s in spans
            ]
            facts += chunk
        # Irrelevant identifiers and prices must not become quantities.
        if rng.random() < 0.3:
            facts += f"; {'Order' if en else 'Auftrag'} #{rng.randrange(100000, 999999)}, EUR {rng.randrange(10, 999)}"
        if ri:
            text += "\n\n"
        chunk, spans = render(
            LAYOUTS[language][family], {"product": product, "facts": facts}, len(text)
        )
        record = {anchor: [spans["product"]]}
        offset = spans["facts"]["start"]
        record.update(
            {
                key: [dict(s, start=s["start"] + offset, end=s["end"] + offset) for s in ss]
                for key, ss in local_spans.items()
            }
        )
        text += chunk
        records.append({key: record[key] for key in fields})
        outputs.append({key: values[key] for key in fields})
        for key in fields:
            if key in codes:
                issues.append({"field": key, "code": codes[key]} | ({"index": ri} if many else {}))
    if scenario == "no_records":
        text = (
            "No products listed." if en else "Keine Produkte aufgeführt."
        ) + f" Ref. {rng.randrange(10**12)}"
        records, outputs, issues = [], [], [{"field": None, "code": "not_found"}]
    return {
        "id": f"{VERSION}-{split}-{index:07d}",
        "source_group": f"{VERSION}-{language}-layout-{family:02d}",
        "origin": "synthetic",
        "scenario": scenario,
        "metadata": {
            "version": VERSION,
            "language": language,
            "dimension": dimension,
            "variant": "original",
        },
        "request": {
            "text": text,
            "fields": fields,
            "locale": locale,
            "anchor": anchor if anchored else None,
            "many": many,
        },
        "gold": {
            "records": records,
            "response": {"data": outputs if many else outputs[0], "issues": issues},
        },
    }


def counterfactual(row, index):
    """Same source with permuted/renamed fields: associations must follow descriptions."""
    row = deepcopy(row)
    row["id"] = row["id"].rsplit("-", 1)[0] + f"-{index:07d}"
    if (index // 10) % 2 and any(
        field["type"] == "quantity" for field in row["request"]["fields"].values()
    ):
        row["metadata"]["variant"] = "target_unit"
        data = row["gold"]["response"]["data"]
        for key, field in row["request"]["fields"].items():
            if field["type"] != "quantity":
                continue
            options = next(items for items in TARGETS if field["unit"] in dict(items))
            old_factor = dict(options)[field["unit"]]
            unit, factor = options[
                (next(i for i, item in enumerate(options) if item[0] == field["unit"]) + 1)
                % len(options)
            ]
            field["unit"] = unit
            for record in data if isinstance(data, list) else [data]:
                if record[key] is not None:
                    record[key] = converted(record[key], old_factor, factor)
        return row
    mapping = {key: f"slot_{i}" for i, key in enumerate(reversed(row["request"]["fields"]))}
    row["metadata"]["variant"] = "schema_permutation"
    row["request"]["fields"] = {mapping[k]: row["request"]["fields"][k] for k in mapping}
    if row["request"]["anchor"]:
        row["request"]["anchor"] = mapping[row["request"]["anchor"]]
    row["gold"]["records"] = [{mapping[k]: r[k] for k in mapping} for r in row["gold"]["records"]]
    data = row["gold"]["response"]["data"]
    result = [
        {mapping[k]: r[k] for k in mapping} for r in (data if isinstance(data, list) else [data])
    ]
    row["gold"]["response"]["data"] = result if isinstance(data, list) else result[0]
    for issue in row["gold"]["response"]["issues"]:
        if issue["field"] is not None:
            issue["field"] = mapping[issue["field"]]
    order = list(mapping.values())
    row["gold"]["response"]["issues"].sort(
        key=lambda i: (i.get("index", 0), order.index(i["field"]) if i["field"] else -1)
    )
    return row


def generate(output, seed=20261015, sizes=(120000, 10000, 20000)):
    if len(sizes) != 3 or any(n <= 0 or n % 10 for n in sizes):
        raise ValueError("Split sizes must be positive multiples of ten")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Corpus output must be empty; never overwrite a frozen corpus")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    partition = {split: {} for split in ("train", "dev", "test")}
    ids = list(range(18))
    rng.shuffle(ids)
    for language in LAYOUTS:
        for split, group in zip(partition, (ids[:12], ids[12:15], ids[15:]), strict=True):
            partition[split][language] = group
    summary = {
        "version": VERSION,
        "seed": seed,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "layout_split": partition,
        "splits": {},
        "test_policy": "Frozen evaluation only; never select thresholds or checkpoints on test. Prior authored fixtures are regression data, not fresh holdout.",
    }
    for split, size in zip(partition, sizes, strict=True):
        counts, digest = Counter(), hashlib.sha256()
        with (output / f"{split}.jsonl").open("w") as file:
            index = 0
            while index < size:
                language = rng.choice(["de", "en"])
                row = make_row(
                    rng,
                    split,
                    index,
                    language,
                    rng.choice(partition[split][language]),
                    rng.choice(SCENARIOS),
                )
                # Exactly 10% schema invariance companions; group stays in one split.
                pair = index % 10 == 8
                if pair:
                    row["metadata"]["pair_id"] = row["id"]
                rows = [row, counterfactual(row, index + 1)] if pair else [row]
                validate_rows(rows)
                for item in rows:
                    encoded = dumps(item) + "\n"
                    file.write(encoded)
                    digest.update(encoded.encode())
                    counts[item["scenario"]] += 1
                index += len(rows)
        summary["splits"][split] = {
            "samples": size,
            "sha256": digest.hexdigest(),
            "scenarios": dict(sorted(counts.items())),
        }
    (output / "summary.json").write_text(dumps(summary, indent=2) + "\n")
    return summary
