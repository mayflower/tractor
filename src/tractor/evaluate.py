"""Offset-aware end-to-end metrics, native threshold calibration and bundle export."""

import itertools
import shutil
from collections import Counter
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path

from .calibration import reuse_neural_outputs
from .data import read_rows
from .jsonio import dumps, loads
from .model import Extractor
from .service import resolve
from .types import ExtractRequest, Quantity


def equal(a, b):
    if isinstance(a, (int, float, Decimal)) and isinstance(b, (int, float, Decimal)):
        a, b = Decimal(str(a)), Decimal(str(b))
        return abs(a - b) <= max(abs(b), Decimal(1)) * Decimal("1e-32")
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def record_key(record, request):
    if not request.anchor:
        return "document"
    spans = record[request.anchor]
    return (spans[0]["start"], spans[0]["end"]) if spans else None


def score(row, candidates, result):
    request = ExtractRequest.model_validate(row["request"])
    gold = row["gold"]
    c = Counter(requests=1)
    response = {"data": result["data"], "issues": result["issues"]}
    c["exact_requests"] = int(equal(response, gold["response"]))
    gold_spans = set()
    predicted_spans = set()
    for rows, destination in [(gold["records"], gold_spans), (candidates, predicted_spans)]:
        for record in rows:
            key = record_key(record, request)
            for field, spans in record.items():
                for span in spans:
                    destination.add((str(key), field, span["start"], span["end"]))
    c["span_tp"] = len(gold_spans & predicted_spans)
    c["span_fp"] = len(predicted_spans - gold_spans)
    c["span_fn"] = len(gold_spans - predicted_spans)
    # Match record identity by source anchor offsets, never by name or list index.
    expected_data = gold["response"]["data"]
    predicted_data = result["data"]
    if request.many:
        expected = {
            record_key(r, request): d for r, d in zip(gold["records"], expected_data, strict=True)
        }
        ordered = sorted(candidates, key=lambda r: r[request.anchor][0]["start"])
        predicted = {
            record_key(r, request): d for r, d in zip(ordered, predicted_data, strict=True)
        }
    else:
        expected = {"document": expected_data}
        predicted = {"document": predicted_data}
    c["gold_records"] = len(expected)
    c["predicted_records"] = len(predicted)
    c["matched_records"] = len(expected.keys() & predicted.keys())
    for key in expected.keys() | predicted.keys():
        actual = predicted.get(key, {})
        wanted = expected.get(key, {})
        if key in expected and key in predicted:
            c["exact_records"] += int(equal(actual, wanted))
        for field in request.fields:
            a, b = actual.get(field), wanted.get(field)
            c["fields"] += 1
            if b is not None:
                c["answerable"] += 1
                c["covered"] += int(a is not None)
                if a is not None and equal(a, b):
                    c["tp"] += 1
                else:
                    c["fn"] += 1
                    c["fp"] += int(a is not None)
            elif a is not None:
                c["fp"] += 1
            elif key in expected:
                c["correct_abstentions"] += 1
            if key in expected and b is None:
                c["unanswerable"] += 1
            if a is not None:
                c["answered"] += 1
                if isinstance(request.fields[field], Quantity):
                    c["quantity_answers"] += 1
                    c["wrong_quantity_answers"] += int(b is None or not equal(a, b))
    return c


def metrics(counts):
    c = counts

    def ratio(a, b):
        return c[a] / c[b] if c[b] else 0.0

    f1 = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if 2 * c["tp"] + c["fp"] + c["fn"] else 0.0
    return {
        "counts": dict(c),
        "field_f1": f1,
        "precision": ratio("tp", "answered"),
        "coverage": ratio("covered", "answerable"),
        "recall": ratio("tp", "answerable"),
        "exact_request_rate": ratio("exact_requests", "requests"),
        "exact_record_rate": ratio("exact_records", "gold_records"),
        "record_match_recall": ratio("matched_records", "gold_records"),
        "abstention_accuracy": ratio("correct_abstentions", "unanswerable"),
        "false_normalization_rate": ratio("wrong_quantity_answers", "quantity_answers"),
        "span_f1": 2 * c["span_tp"] / (2 * c["span_tp"] + c["span_fp"] + c["span_fn"])
        if c["span_tp"] + c["span_fp"] + c["span_fn"]
        else 0.0,
    }


def evaluate(args):
    if args.calibrate and args.split != "dev":
        raise ValueError("Calibration is permitted only on dev")
    rows = read_rows(
        "tests/fixtures/api-cases.jsonl"
        if args.split == "authored"
        else Path(args.data) / f"{args.split}.jsonl"
    )
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(args.model)
    models = (
        sorted(root.glob("checkpoint-epoch-*"))
        if not root.joinpath("config.json").exists()
        else [root]
    )
    if not models:
        raise ValueError("No checkpoint models found")
    selection = loads(Path(args.selection).read_text()) if args.selection else None
    all_summaries = []
    for model in models:
        x = Extractor(model, args.device, args.precision)
        grid = (
            [
                dict(zip(("text", "quantity", "anchor"), values, strict=True))
                for values in itertools.product((0.25, 0.5, 0.75), repeat=3)
            ]
            if args.calibrate
            else [selection["thresholds"] if selection else x.thresholds]
        )
        counts = [Counter() for _ in grid]
        grouped = [{} for _ in grid]
        gold_normalizer = Counter()
        raw_path = destination / f"{model.name}-predictions.jsonl"
        with raw_path.open("w") as raw:
            for index, row in enumerate(rows):
                request = ExtractRequest.model_validate(row["request"])
                # Shared encoder states only within this offline evaluation sample.
                hidden = x.encode(request) if args.calibrate else None
                normal = resolve(request, row["gold"]["records"])
                gold_normalizer["requests"] += 1
                gold_normalizer["exact_requests"] += int(
                    equal(
                        {"data": normal["data"], "issues": normal["issues"]},
                        row["gold"]["response"],
                    )
                )
                with reuse_neural_outputs(x) if args.calibrate else nullcontext():
                    for gi, thresholds in enumerate(grid):
                        x.thresholds = thresholds
                        candidates = x.candidates(request, hidden_states=hidden)
                        result = resolve(request, candidates)
                        c = score(row, candidates, result)
                        counts[gi].update(c)
                        tags = [
                            row["origin"],
                            "records" if request.anchor else "document",
                            str(request.locale),
                            f"scenario-{row.get('scenario', 'authored')}",
                        ]
                        for tag in tags:
                            grouped[gi].setdefault(tag, Counter()).update(c)
                        raw.write(
                            dumps(
                                {
                                    "id": row["id"],
                                    "thresholds": thresholds,
                                    "candidates": candidates,
                                    "result": result,
                                }
                            )
                            + "\n"
                        )
                if (index + 1) % 100 == 0:
                    print(f"{model.name}: {index + 1}/{len(rows)}", flush=True)
        for gi, thresholds in enumerate(grid):
            summary = {
                "model": str(model.resolve()),
                "thresholds": thresholds,
                "precision_mode": args.precision,
                "split": args.split,
                "metrics": metrics(counts[gi]),
                "groups": {k: metrics(v) for k, v in grouped[gi].items()},
                "normalizer_on_gold": dict(gold_normalizer),
                "predictions": str(raw_path),
            }
            all_summaries.append(summary)
        del x
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = max(
        all_summaries,
        key=lambda r: (
            r["metrics"]["field_f1"],
            r["metrics"]["exact_request_rate"],
            r["metrics"]["precision"],
        ),
    )
    Path(destination, "summary.json").write_text(dumps(all_summaries, indent=2))
    if args.calibrate:
        Path(destination, "selection.json").write_text(dumps(best, indent=2))
    print(dumps(best, indent=2))
    return best


def export(run, selection_path, output):
    selection = loads(Path(selection_path).read_text())
    source = Path(selection["model"]).resolve()
    run = Path(run).resolve()
    if source != run and run not in source.parents:
        raise ValueError("Selected checkpoint does not belong to the supplied run")
    evidence = run / "training-evidence.json"
    if not evidence.exists():
        raise ValueError("Missing completed training evidence")
    report = loads(evidence.read_text())
    if report["result"]["total_steps"] <= 0:
        raise ValueError("No optimizer steps recorded")
    target = Path(output)
    if target.exists():
        raise ValueError("Export destination already exists")
    shutil.copytree(source, target)
    from .model import BASE_MODEL, COMMIT, REVISION
    from .normalize import ALIASES, PINT_UNITS

    Path(target, "normalization.json").write_text(
        dumps(
            {
                "contract": "tractor-v1",
                "precision": 50,
                "output_significant_digits": 34,
                "rounding": "ROUND_HALF_EVEN",
                "aliases": ALIASES,
                "units": PINT_UNITS,
            },
            indent=2,
        )
    )
    Path(target, "thresholds.json").write_text(dumps(selection["thresholds"], indent=2))
    Path(target, "source-lock.json").write_text(
        dumps(
            {"base_model": BASE_MODEL, "base_model_revision": REVISION, "gliner_commit": COMMIT},
            indent=2,
        )
    )
    shutil.copy2(selection_path, target / "evaluation.json")
    shutil.copy2(evidence, target / "training-evidence.json")
    Path(target, "MODEL_CARD.md").write_text(
        "# tractor quantity extractor\n\n"
        "Full fine-tuning of the pinned multilingual GLiNER boundary model.\n"
        "Training data: synthetic DE/EN templates with source spans and independent arithmetic gold.\n"
        "Evaluation is synthetic and authored regression only, not real-world validation.\n"
        "Field descriptions, locale and explicit units are required for reliable interpretation.\n"
        "See evaluation.json and training-evidence.json for measured results and provenance.\n"
        "License of base model: Apache-2.0. No JEV protocol compatibility claim.\n"
    )
