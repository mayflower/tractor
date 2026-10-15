"""Deterministic synthetic facts, family-level splits and native export."""

import hashlib
import random
import string
import unicodedata
from decimal import Decimal, localcontext
from pathlib import Path

from .jsonio import dumps, loads
from .types import ExtractRequest

# Whole families, including their variants, belong to one split.
TEMPLATES = [
    "{product}: {label} {quantity}; {label2} {quantity2}.",
    "Das Produkt {product} hat folgende Angaben: {label} = {quantity}, {label2} = {quantity2}.",
    "Datenblatt für {product}\n- {label}: {quantity}\n- {label2}: {quantity2}",
    "Hallo, für {product} bestätigen wir {quantity} ({label}) und {quantity2} ({label2}).",
    "Messprotokoll: {product}\nMessgröße {label}: {quantity}\nMessgröße {label2}: {quantity2}",
    "Artikel | {label} | {label2}\n{product} | {quantity} | {quantity2}",
    "Bei {product} liegt die Angabe für {label} bei {quantity}; für {label2} bei {quantity2}.",
    "{label}: {quantity}. {label2}: {quantity2}. Diese Werte gehören zu {product}.",
    "Technische Übersicht ({product}):\n{quantity} — {label}\n{quantity2} — {label2}",
    "Angebot: {product}. Spezifikation {label} / {label2}: {quantity} / {quantity2}.",
    "Wir liefern {product} mit {label} {quantity} sowie {label2} {quantity2}.",
    "Die Prüfung von {product} ergab: {label} {quantity}, {label2} {quantity2}.",
    "Item {product}: {label} is {quantity}; {label2} is {quantity2}.",
    "Please record {product} with {quantity} for {label} and {quantity2} for {label2}.",
    "{product} specifications\n{label} = {quantity}\n{label2} = {quantity2}",
    "For {product}, the stated {label} equals {quantity}, whereas {label2} equals {quantity2}.",
    "Order confirmation for {product}: {label} {quantity}; {label2} {quantity2}.",
    "Lab notes [{product}]: {quantity} ({label}); {quantity2} ({label2}).",
    "Im Lager wurde {product} erfasst: {quantity} für {label}, {quantity2} für {label2}.",
    "Katalogeintrag {product}\nEigenschaft {label}: {quantity}\nEigenschaft {label2}: {quantity2}",
    "Rückmeldung zur Anfrage über {product}: {label} beträgt {quantity}. {label2} beträgt {quantity2}.",
    "Zusammenfassung — {product} ({label}: {quantity}; {label2}: {quantity2}).",
    "Notiz: Für {product} sind {label} mit {quantity} und {label2} mit {quantity2} dokumentiert.",
    "Im Vergleich wird {product} geführt. Seine Werte: {label} {quantity}, {label2} {quantity2}.",
    "Aktualisierte Spezifikation\nProdukt: {product}\n{label}: {quantity}\n{label2}: {quantity2}",
    "Zur Freigabe von {product} gelten diese Angaben: {quantity} [{label}] und {quantity2} [{label2}].",
    "Das Etikett an {product} nennt {label} {quantity}. Darunter steht {label2} {quantity2}.",
    "Review of {product}: measured {label} {quantity}; recorded {label2} {quantity2}.",
    "Specification card\nName: {product}\n{label}: {quantity}\n{label2}: {quantity2}",
    "Customer enquiry — {product}, supplied with {label}: {quantity}, {label2}: {quantity2}.",
    "Versandinformation für {product}. Aufgeführt sind {label} ({quantity}) und {label2} ({quantity2}).",
    "Unter der Bezeichnung {product} stehen die Werte {quantity} für {label} und {quantity2} für {label2}.",
    "Inventar\n{product}\n{label} → {quantity}\n{label2} → {quantity2}",
    "Engineering record: {label} = {quantity}; {label2} = {quantity2}; product = {product}.",
    "We checked {product}. Its {label} was reported as {quantity}, its {label2} as {quantity2}.",
    "Release notes for {product}\n* {label}: {quantity}\n* {label2}: {quantity2}",
]
# Independent SI factors, deliberately not Pint conversions.
DIMENSIONS = [
    (
        "Breite",
        "width",
        "Höhe",
        "height",
        [("cm", "0.01"), ("mm", "0.001"), ("m", "1"), ("inches", "0.0254"), ("miles", "1609.344")],
        [("mm", "0.001"), ("cm", "0.01"), ("m", "1"), ("km", "1000")],
    ),
    (
        "Gewicht",
        "weight",
        "Verpackungsgewicht",
        "package weight",
        [("kg", "1"), ("g", "0.001"), ("lb", "0.45359237")],
        [("kg", "1"), ("g", "0.001"), ("mg", "0.000001")],
    ),
    (
        "Dauer",
        "duration",
        "Ladezeit",
        "charge time",
        [("hours", "3600"), ("min", "60"), ("s", "1"), ("ms", "0.001")],
        [("h", "3600"), ("min", "60"), ("s", "1"), ("ms", "0.001")],
    ),
    (
        "Volumen",
        "volume",
        "Verpackungsvolumen",
        "package volume",
        [("L", "1"), ("mL", "0.001"), ("m3", "1000")],
        [("L", "1"), ("mL", "0.001"), ("m3", "1000")],
    ),
    (
        "Fläche",
        "area",
        "Grundfläche",
        "base area",
        [("cm²", "0.0001"), ("m²", "1"), ("mm²", "0.000001")],
        [("m2", "1"), ("cm2", "0.0001"), ("mm2", "0.000001")],
    ),
    (
        "Geschwindigkeit",
        "speed",
        "Maximalgeschwindigkeit",
        "maximum speed",
        [
            ("km/h", "0.27777777777777777777777777777777777777777777777778"),
            ("m/s", "1"),
            ("mph", "0.44704"),
        ],
        [
            ("m/s", "1"),
            ("km/h", "0.27777777777777777777777777777777777777777777777778"),
            ("mph", "0.44704"),
        ],
    ),
]


def render(template, values, offset=0):
    text, spans = "", {}
    for literal, key, _, _ in string.Formatter().parse(template):
        text += literal
        if key:
            start = len(text) + offset
            text += values[key]
            spans[key] = {"text": values[key], "start": start, "end": len(text) + offset}
    return text, spans


def read_rows(path):
    return [loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(dumps(row) + "\n")


def generate_row(rng, family, index, split):
    english = family in {12, 13, 14, 15, 16, 17, 27, 28, 29, 33, 34, 35}
    locale = "en-US" if english else "de-DE"
    dim = rng.choice(DIMENSIONS)
    label, label2 = (dim[1], dim[3]) if english else (dim[0], dim[2])
    target, target_factor = rng.choice(dim[5])
    scenario = index % 16
    anchored = index % 3 != 0
    count = 2 if anchored and index % 2 == 0 else 1
    names = rng.sample(["product", "item", "objekt", "name"], 1) + rng.sample(
        ["amount", "wert", "x", "a", "measure"], 2
    )
    product_key, qkey, qkey2 = names
    fields = {
        product_key: {
            "type": "text",
            "description": "Product name" if english else "Produktbezeichnung",
        },
        qkey: {"type": "quantity", "description": label, "unit": target},
        qkey2: {"type": "quantity", "description": label2, "unit": target},
    }
    if not anchored and scenario % 3 == 0 and scenario != 15:
        fields.pop(product_key)
    if scenario % 4 == 1:
        fields.pop(qkey2)
    if not anchored and scenario == 15:
        fields = {product_key: fields[product_key]}
    keys = list(fields)
    rng.shuffle(keys)
    fields = {k: fields[k] for k in keys}
    text, records, outputs, issues = "", [], [], []
    previous_product = None
    for record_index in range(count):
        unit, factor = rng.choice(dim[4])
        a, b = Decimal(rng.randint(1, 9900)) / 10, Decimal(rng.randint(1, 9900)) / 10
        if scenario == 10:
            a = -a
        number = str(a).replace(".", "," if locale == "de-DE" else ".")
        second = str(b).replace(".", "," if locale == "de-DE" else ".")
        quantity, quantity2 = f"{number} {unit}", f"{second} {unit}"
        issue, absent = None, False
        if scenario == 1:
            quantity = f"about {quantity}" if english else f"bis zu {quantity}"
            issue = "non_scalar_quantity"
        elif scenario == 2:
            quantity = f"{number}–{second} {unit}"
            issue = "non_scalar_quantity"
        elif scenario == 3:
            quantity = number
            issue = "missing_unit"
        elif scenario == 4:
            quantity = "unknown" if english else "unbekannt"
            absent, issue = True, "not_found"
        elif scenario == 5:
            quantity = f"{number} °C"
            issue = "unsupported_unit"
        elif scenario == 6:
            quantity = f"{number} gal"
            issue = "ambiguous_unit"
        elif scenario == 7:
            quantity = f"{number} nm"
            issue = "unsupported_unit"
        elif scenario == 8:
            quantity = f"{number} {unit} ± 1 {unit}"
            issue = "non_scalar_quantity"
        elif scenario == 9:
            quantity = f"1e101 {unit}"
            issue = "numeric_out_of_range"
        elif scenario == 11:
            quantity = f"1/2 {unit}"
            issue = "unsupported_number_format"
        elif scenario == 12:
            quantity = f"1,234 {unit}"
            locale = None
            issue = "ambiguous_number_format"
            quantity2 = f"{int(b)} {unit}"
            b = Decimal(int(b))
        elif scenario == 13:
            quantity = f"{number}e1 {unit}"
            a *= 10
        correction_prefix = ""
        if scenario == 14:
            correction_prefix = f"not 1 {unit}, but " if english else f"nicht 1 {unit}, sondern "
        product = f"{rng.choice(['Atlas', 'Rover', 'Nova', 'Sigma'])}-{split[0]}{index}"
        if record_index and scenario % 2 == 0:
            product = previous_product
        previous_product = product
        if text:
            text += "\n\n"
        chunk, spans = render(
            TEMPLATES[family],
            {
                "product": product,
                "label": label,
                "quantity": correction_prefix + quantity,
                "label2": label2,
                "quantity2": quantity2,
            },
            len(text),
        )
        text += chunk
        if correction_prefix:
            spans["quantity"]["start"] += len(correction_prefix)
            spans["quantity"]["text"] = quantity
        competing = None
        if scenario == 0 and not anchored:
            extra, extra_spans = render(
                " Weitere Angabe für {label}: {quantity}.",
                {"label": label, "quantity": f"{second} {unit}"},
                len(text),
            )
            text += extra
            competing = extra_spans["quantity"]
            issue = "ambiguous_value"
        record = {
            product_key: [spans["product"]],
            qkey: [] if absent else [spans["quantity"]],
            qkey2: [spans["quantity2"]],
        }
        if competing:
            record[qkey].append(competing)
        records.append({k: record[k] for k in fields})
        with localcontext() as context:
            context.prec = 50
            v1 = a * Decimal(factor) / Decimal(target_factor)
            v2 = b * Decimal(factor) / Decimal(target_factor)
            context.prec = 34
            v1, v2 = +v1, +v2
        data = {product_key: product, qkey: None if issue else v1, qkey2: v2}
        outputs.append({k: data[k] for k in fields})
        if issue:
            entry = {"field": qkey, "code": issue}
            if anchored:
                entry["index"] = record_index
            issues.append(entry)
    request = {
        "text": text,
        "fields": fields,
        "locale": locale,
        "many": anchored,
        "anchor": product_key if anchored else None,
    }
    if scenario == 15 and anchored:
        request["text"] = (
            "No products were found." if english else "Keine Produkte vorhanden."
        ) + f" Dokument {split}-{index}."
        records, outputs, issues = [], [], [{"field": None, "code": "not_found"}]
    # Preserve request-field order for issue ordering.
    issues.sort(key=lambda v: (v.get("index", 0), keys.index(v["field"]) if v["field"] else -1))
    return {
        "id": f"{split}-{index:06d}",
        "source_group": f"family-{family:02d}",
        "origin": "synthetic",
        "scenario": scenario,
        "request": request,
        "gold": {
            "records": records,
            "response": {"data": outputs if anchored else outputs[0], "issues": issues},
        },
    }


def generate(output, seed=42, sizes=(12000, 1500, 1500)):
    if len(sizes) != 3 or any(size <= 0 for size in sizes):
        raise ValueError("All three split sizes must be positive")
    rng = random.Random(seed)
    families = list(range(len(TEMPLATES)))
    rng.shuffle(families)
    splits = {"train": families[:24], "dev": families[24:30], "test": families[30:]}
    summary = {"seed": seed, "families": splits, "counts": {}}
    for (split, ids), size in zip(splits.items(), sizes, strict=True):
        rows = [generate_row(rng, ids[i % len(ids)], i, split) for i in range(size)]
        write_rows(Path(output) / f"{split}.jsonl", rows)
        summary["counts"][split] = len(rows)
    Path(output, "summary.json").write_text(dumps(summary, indent=2))
    return summary


def validate_rows(rows):
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample ID {row['id']}")
        ids.add(row["id"])
        request = ExtractRequest.model_validate(row["request"])
        if row["origin"] not in {"synthetic", "authored", "real"} or not row["source_group"]:
            raise ValueError(f"Invalid provenance: {row['id']}")
        if not request.anchor and len(row["gold"]["records"]) != 1:
            raise ValueError("Document sample must have one span mapping")
        for record in row["gold"]["records"]:
            if set(record) != set(request.fields):
                raise ValueError("Gold fields differ from request")
            for spans in record.values():
                for span in spans:
                    from .service import valid_span

                    if not valid_span(request.text, span):
                        raise ValueError(f"Invalid span: {row['id']}")
            if request.anchor and len(record[request.anchor]) != 1:
                raise ValueError("Gold anchor must identify one occurrence")


def validate_dataset(root, extractor):
    from gliner2.training.trainer import ExtractorCollator

    from .model import native_example, schema_for
    from .training_adapter import configure_processor

    configure_processor(extractor.model.processor)
    collator = ExtractorCollator(
        extractor.model.processor, is_training=True, architecture="boundary", max_gold_per_query=64
    )
    groups, hashes, summary = set(), set(), {}
    for split in ("train", "dev", "test"):
        rows = read_rows(Path(root) / f"{split}.jsonl")
        validate_rows(rows)
        split_groups = {row["source_group"] for row in rows}
        split_hashes = {
            hashlib.sha256(
                " ".join(
                    unicodedata.normalize("NFKC", row["request"]["text"]).casefold().split()
                ).encode()
            ).hexdigest()
            for row in rows
        }
        if groups & split_groups or hashes & split_hashes:
            raise ValueError("Split leakage detected")
        groups |= split_groups
        hashes |= split_hashes
        native = []
        max_tokens = 0
        for row in rows:
            request = ExtractRequest.model_validate(row["request"])
            max_tokens = max(max_tokens, extractor.preflight(request, schema_for(request)))
            n = native_example(row)
            try:
                batch = collator([(n["input"], n["output"])])
                if len(batch) != 1:
                    raise ValueError("Collator discarded sample")
            except Exception as exc:
                raise ValueError(f"Native collation failed for {row['id']}") from exc
            native.append(n)
        write_rows(Path(root) / f"{split}.native.jsonl", native)
        summary[split] = {
            "samples": len(rows),
            "families": len(split_groups),
            "max_tokens": max_tokens,
        }
    Path(root, "validation.json").write_text(dumps(summary, indent=2))
    return summary
