"""Offset-aware end-to-end metrics, native threshold calibration and bundle export."""

import hashlib
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
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float, Decimal)) and isinstance(b, (int, float, Decimal)):
        a, b = Decimal(str(a)), Decimal(str(b))
        return abs(a - b) <= max(abs(b), Decimal(1)) * Decimal("1e-32")
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def evaluate_episodes(args):
    """Native multi-task evaluation; never silently reuse product metrics or cached predictions."""
    from .episode_metrics import score_episode, summarize
    from .generalist_data import read_episodes

    if args.calibrate:
        raise ValueError("Native episode calibration is separate from checkpoint evaluation")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    extractor = Extractor(args.model, args.device, args.precision)
    totals, families, task_kinds, classification_tasks = Counter(), {}, {}, {}
    with destination.joinpath("predictions.jsonl").open("x") as stream:
        for episode in read_episodes(args.episodes):
            if episode.split != args.split:
                raise ValueError("Episode split does not match requested evaluation split")
            raw, error = {}, None
            try:
                raw = extractor.extract_tasks(episode.text, episode.tasks)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            counts = score_episode(episode, raw, error)
            totals.update(counts)
            families.setdefault(episode.family, Counter()).update(counts)
            for kind in {task.kind for task in episode.tasks}:
                tasks = [task for task in episode.tasks if task.kind == kind]
                subset = episode.model_copy(update={'tasks': tasks, 'gold': {t.id: episode.gold[t.id] for t in tasks}})
                task_kinds.setdefault(kind, Counter()).update(score_episode(subset, raw, error))
            for task in episode.tasks:
                if task.kind == 'classification':
                    subset = episode.model_copy(update={'tasks': [task], 'gold': {task.id: episode.gold[task.id]}})
                    key = f'{episode.family}:{task.id}:{episode.text_origin}/{episode.label_origin}'
                    classification_tasks.setdefault(key, Counter()).update(score_episode(subset, raw, error))
            stream.write(dumps({"id": episode.id, "source_group": episode.source_group,
                                "family": episode.family, "prediction": raw, "error": error,
                                "counts": dict(counts)}) + "\n")
            stream.flush()
    report = {"model_sha256": extractor.model_revision, "split": args.split,
              "level": "native", "overall": summarize(totals),
              "families": {key: summarize(value) for key, value in families.items()},
              "task_kinds": {key: summarize(value) for key, value in task_kinds.items()},
              "classification_tasks": {key: summarize(value) for key, value in classification_tasks.items()}}
    f1s = [value["field_f1"] for value in report["families"].values()
           if value["field_f1"] is not None]
    report["macro_field_f1"] = sum(f1s) / len(f1s) if f1s else None
    destination.joinpath("summary.json").write_text(dumps(report, indent=2))
    print(dumps(report, indent=2))


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
    if args.calibrate and args.split not in {"dev", "calibration"}:
        raise ValueError("Threshold tuning requires dev or a separate calibration split")
    if args.calibrate and args.split == "calibration" and not Path(args.model, "config.json").is_file():
        raise ValueError("Final calibration requires one frozen model, not checkpoint ranking")
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
    groups = report.get("required_learning_groups", ["encoder", "boundary_head", "record_decoder"])
    if not groups or any(not report.get("parameters", {}).get(key, {}).get("changed")
                         or not report.get("parameters", {}).get(key, {}).get("nonzero_finite_gradient")
                         for key in groups):
        raise ValueError("Required learning evidence is incomplete")
    target = Path(output)
    if target.exists():
        raise ValueError("Export destination already exists")
    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                "encoder_config/config.json"]
    if any(not source.joinpath(name).is_file() for name in required):
        raise ValueError("Checkpoint lacks required offline inference files")
    target.mkdir(parents=True)
    for name in required:
        target.joinpath(name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target / name)
    from .model import COMMIT
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
    with source.joinpath("model.safetensors").open("rb") as stream:
        weights_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    source_lock = {"weights_sha256": weights_sha256, "gliner_commit": COMMIT,
                   "parent_weights_sha256": report.get("initial_weights_sha256")}
    Path(target, "source-lock.json").write_text(dumps(source_lock, indent=2))
    if selection.get("native_policy"):
        policy = loads(Path(selection["native_policy"]).read_text())
        if policy.get("model_sha256") != weights_sha256 or policy.get("gliner_commit") != COMMIT:
            raise ValueError("Native calibration policy belongs to another model")
        Path(target, "native-policy.json").write_text(dumps(policy, indent=2))
    # Keep internal paths, optimizer state and per-example predictions outside the bundle.
    measured = {key: selection[key] for key in ("metrics", "split", "precision_mode") if key in selection}
    measured["weights_sha256"] = weights_sha256
    Path(target, "metrics.json").write_text(dumps(measured, indent=2))
    Path(target, "README.md").write_text(
        "# tractor · System One Extraction Model\n\n"
        "Local training candidate for evidence-based extraction.\n\n"
        "The HTTP contract supports Text and Quantity fields. Native tasks use the local schema interface.\n"
        "See metrics.json for available measured results and their split. Development results are not final test results.\n"
        "This export does not promote a release or establish general-purpose reliability.\n"
        "Source and data licensing must be resolved before publication.\n"
    )
