"""Streaming corpus audit: independent gold consistency, leakage, exact native targets."""

import hashlib
import unicodedata
from collections import Counter
from decimal import Decimal
from pathlib import Path

from .data import validate_rows
from .jsonio import dumps, loads
from .service import resolve
from .types import ExtractRequest


def check_oracle(row):
    actual = resolve(ExtractRequest.model_validate(row["request"]), row["gold"]["records"])
    expected = row["gold"]["response"]
    if actual["issues"] != expected["issues"]:
        raise ValueError(
            f"Issue oracle mismatch: {row['id']}: {actual['issues']} != {expected['issues']}"
        )
    a = actual["data"] if isinstance(actual["data"], list) else [actual["data"]]
    e = expected["data"] if isinstance(expected["data"], list) else [expected["data"]]
    if len(a) != len(e):
        raise ValueError(f"Record oracle mismatch: {row['id']}")
    for left, right in zip(a, e, strict=True):
        if left.keys() != right.keys():
            raise ValueError(f"Field oracle mismatch: {row['id']}")
        for key, value in right.items():
            if isinstance(value, (Decimal, int)):
                matches = left[key] is not None and abs(left[key] - value) <= max(
                    abs(value), Decimal(1)
                ) * Decimal("1e-32")
            else:
                matches = left[key] == value
            if not matches:
                raise ValueError(
                    f"Value oracle mismatch: {row['id']} {key}: {left[key]} != {value}"
                )


def check_native(row, request, collator):
    from .model import field_ids, native_example

    native = native_example(row)
    batch = collator([(native["input"], native["output"])])
    if len(batch) != 1:
        raise ValueError(f"Native collator dropped {row['id']}")
    inverse = {v: k for k, v in field_ids(request).items()}

    def chars(pair):
        start, end = map(int, pair)
        return batch.start_mappings[0][start], batch.end_mappings[0][end - 1]

    if request.anchor:
        spans, mask, records, query_ids, *_ = batch.targets.record_targets
        actual = []
        for ri in range(int(records.sum())):
            record = {}
            for fi, qi in enumerate(query_ids[0, 0]):
                name = inverse[batch.query_layouts[0].query(int(qi)).role_name]
                record[name] = [chars(pair) for pair in spans[0, 0, ri, fi][mask[0, 0, ri, fi]]]
            actual.append(record)
    else:
        actual = [{}]
        for query in batch.query_layouts[0].queries:
            qi = query.query_id
            pairs = batch.targets.mention_pairs[0, qi][batch.targets.mention_mask[0, qi]]
            actual[0][inverse[query.role_name]] = [chars(pair) for pair in pairs]
    expected = [
        {name: [(s["start"], s["end"]) for s in values] for name, values in record.items()}
        for record in row["gold"]["records"]
    ]
    if actual != expected:
        raise ValueError(
            f"Native original-offset targets differ: {row['id']}: {actual} != {expected}"
        )
    return native


def audit(root, model=None):
    root = Path(root)
    manifest = loads((root / "summary.json").read_text())
    extractor = collator = None
    if model:
        from gliner2.training.trainer import ExtractorCollator

        from .model import Extractor
        from .training_adapter import configure_processor

        extractor = Extractor(model, "cpu")
        configure_processor(extractor.model.processor)
        collator = ExtractorCollator(
            extractor.model.processor,
            is_training=True,
            architecture="boundary",
            max_gold_per_query=64,
        )
    seen_groups, seen_text, seen_ids, seen_pairs = set(), set(), set(), set()
    report = {"version": manifest["version"], "native_targets_checked": bool(model), "splits": {}}
    for split in ("train", "dev", "test"):
        groups, texts, pairs, local_duplicates = set(), {}, set(), 0
        counts, languages, modes, fields, records, scenarios, dimensions, locales = (
            Counter() for _ in range(8)
        )
        digest, count, max_tokens = hashlib.sha256(), 0, 0
        destination = root / f"{split}.native.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        output = temporary.open("w") if model else None
        try:
            with (root / f"{split}.jsonl").open("rb") as source:
                for line in source:
                    digest.update(line)
                    row = loads(line)
                    validate_rows([row])
                    if row["id"] in seen_ids:
                        raise ValueError("Duplicate ID")
                    seen_ids.add(row["id"])
                    group = row["source_group"]
                    text_hash = hashlib.sha256(
                        " ".join(
                            unicodedata.normalize("NFKC", row["request"]["text"]).casefold().split()
                        ).encode()
                    ).hexdigest()
                    pair = row["metadata"].get("pair_id")
                    if (
                        group in seen_groups
                        or text_hash in seen_text
                        or (pair and pair in seen_pairs)
                    ):
                        raise ValueError(f"Cross-split leakage: {row['id']}")
                    if text_hash in texts:
                        if not pair or texts[text_hash] != pair:
                            raise ValueError(f"Unplanned duplicate source: {row['id']}")
                        local_duplicates += 1
                    texts[text_hash] = pair
                    groups.add(group)
                    if pair:
                        pairs.add(pair)
                    check_oracle(row)
                    request = ExtractRequest.model_validate(row["request"])
                    if extractor:
                        from .model import schema_for

                        max_tokens = max(
                            max_tokens, extractor.preflight(request, schema_for(request))
                        )
                        native = check_native(row, request, collator)
                        output.write(dumps(native) + "\n")
                    counts.update(i["code"] for i in row["gold"]["response"]["issues"])
                    languages[row["metadata"]["language"]] += 1
                    modes[
                        "many" if request.many else "anchored_one" if request.anchor else "document"
                    ] += 1
                    fields[str(len(request.fields))] += 1
                    records[str(len(row["gold"]["records"]))] += 1
                    scenarios[row["scenario"]] += 1
                    dimensions[str(row["metadata"]["dimension"])] += 1
                    locales[str(request.locale)] += 1
                    count += 1
                    if count % 10000 == 0:
                        print(f"Audited {split}: {count}", flush=True)
            expected = manifest["splits"][split]
            if digest.hexdigest() != expected["sha256"] or count != expected["samples"]:
                raise ValueError(f"Frozen corpus manifest mismatch: {split}")
        finally:
            if output:
                output.close()
        if model:
            temporary.replace(destination)
        seen_groups |= groups
        seen_text |= texts.keys()
        seen_pairs |= pairs
        report["splits"][split] = {
            "samples": count,
            "sha256": digest.hexdigest(),
            "source_groups": len(groups),
            "unique_texts": len(texts),
            "invariance_pairs": len(pairs),
            "intentional_duplicate_texts": local_duplicates,
            "max_joint_tokens": max_tokens if model else None,
            "languages": dict(languages),
            "modes": dict(modes),
            "fields": dict(fields),
            "records": dict(records),
            "issues": dict(counts),
            "scenarios": dict(scenarios),
            "dimensions": dict(dimensions),
            "locales": dict(locales),
        }
    report["cross_split_duplicate_texts"] = 0
    report["cross_split_source_groups"] = 0
    (root / "audit.json").write_text(dumps(report, indent=2) + "\n")
    return report
