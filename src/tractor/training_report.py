"""Paired source-cluster uncertainty and fixed-rule pilot selection."""

import random
from collections import Counter, defaultdict
from pathlib import Path

from .episode_metrics import summarize
from .jsonio import dumps, loads


def clustered_difference(before_path, after_path, repeats=500, seed=42):
    before = {r["id"]: r for r in map(loads, Path(before_path).read_text().splitlines())}
    after = {r["id"]: r for r in map(loads, Path(after_path).read_text().splitlines())}
    if before.keys() != after.keys():
        raise ValueError("Paired evaluation requires identical episode IDs")
    groups = defaultdict(list)
    for key in before:
        if before[key]["source_group"] != after[key]["source_group"]:
            raise ValueError("Source group changed between comparisons")
        groups[before[key]["source_group"]].append(key)
    names = sorted(groups)
    if len(names) < 2:
        return {"source_groups": len(names), "interval": None}
    # This interval concerns request exactness only. Aggregate those sufficient
    # statistics per source group once, preserving the original resampling order.
    group_counts = {}
    for group, keys in groups.items():
        group_counts[group] = tuple(
            sum(rows[key]["counts"].get(metric, 0) for key in keys)
            for rows in (before, after) for metric in ("requests", "exact_requests"))
    if any(values[0] <= 0 or values[2] <= 0 for values in group_counts.values()):
        raise ValueError("Each paired source group needs a positive request denominator")
    rng, differences = random.Random(seed), []
    for _ in range(repeats):
        before_n = before_exact = after_n = after_exact = 0
        for group in rng.choices(names, k=len(names)):
            bn, be, an, ae = group_counts[group]
            before_n += bn
            before_exact += be
            after_n += an
            after_exact += ae
        differences.append(after_exact / after_n - before_exact / before_n)
    differences.sort()
    return {"source_groups": len(names), "repeats": repeats, "seed": seed,
            "metric": "paired request exact difference", "confidence": 0.95,
            "interval": [differences[int(repeats * .025)], differences[int(repeats * .975)]]}


def pilot_report(root):
    root = Path(root)
    rows = {}
    arms = ["B0", "B1", "P0", "P1", "P2", "P3"]
    for extra in ("P4", "P5"):
        if root.joinpath(extra, "native", "summary.json").exists():
            arms.append(extra)
    for arm in arms:
        native = loads(root.joinpath(arm, "native", "summary.json").read_text())
        product = loads(root.joinpath(arm, "product", "summary.json").read_text())[0]
        rows[arm] = {"native": native, "product": product["metrics"]}
        if arm != "B1":
            rows[arm]["difference_from_release"] = clustered_difference(
                root / "B1/native/predictions.jsonl", root / arm / "native/predictions.jsonl")
    baseline = rows["B1"]
    design_path = root / "design.json"
    design = loads(design_path.read_text()) if design_path.exists() else {}
    regression_metrics = design.get("product_regression_metrics", ["field_f1", "exact_record_rate"])
    minimum_precision = design.get("minimum_product_precision")
    minimum_metrics = design.get("minimum_product_metrics", {})
    if any(not 0 <= float(value) <= 1 for value in minimum_metrics.values()):
        raise ValueError("Minimum product metrics must be between 0 and 1")
    eligible = [arm for arm in arms if arm.startswith("P")
                and all(float(rows[arm]["product"][key]) >= float(baseline["product"][key]) - .01
                        for key in regression_metrics)
                and all(float(rows[arm]["product"][key]) >= float(value)
                        for key, value in minimum_metrics.items())
                and (minimum_precision is None or
                     float(rows[arm]["product"]["precision"]) >= float(minimum_precision))
                and rows[arm]["native"]["macro_field_f1"] > baseline["native"]["macro_field_f1"]]
    winner = max(eligible, key=lambda a: rows[a]["native"]["macro_field_f1"]) if eligible else "B1"
    result = {"arms": rows, "selected": winner, "eligible": eligible,
              "product_regression_metrics": regression_metrics,
              "minimum_product_precision": minimum_precision,
              "minimum_product_metrics": minimum_metrics,
              "qualification": "Single-seed development pilot; no final test or promotion"}
    root.joinpath("summary.json").write_text(dumps(result, indent=2))
    lines = ["# Local pilot comparison", "", "Single-seed development results. No release promotion.", "",
             "| Arm | Native macro F1 | Native request exact | Product field F1 | Product exact records | Product exact requests |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, row in rows.items():
        n, p = row["native"], row["product"]
        lines.append(f"| {arm} | {n['macro_field_f1']:.4f} | {n['overall']['request_exact']:.4f} | "
                     f"{p['field_f1']:.4f} | {p['exact_record_rate']:.4f} | {p['exact_request_rate']:.4f} |")
    lines += ["", f"Selected: {winner}. Eligible candidates: {', '.join(eligible) or 'none'}.", ""]
    root.joinpath("comparison.md").write_text("\n".join(lines))
    return result


def rescore_predictions(episodes_path, directory):
    """Recompute metrics from stored raw outputs without another model invocation."""
    from .episode_metrics import score_episode
    from .generalist_data import read_episodes

    directory = Path(directory)
    episodes = {e.id: e for e in read_episodes(episodes_path)}
    path = directory / "predictions.jsonl"
    rows = [loads(line) for line in path.read_text().splitlines()]
    if {r["id"] for r in rows} != episodes.keys() or len(rows) != len(episodes):
        raise ValueError("Prediction IDs do not match episode file")
    totals, families, origins = Counter(), defaultdict(Counter), defaultdict(Counter)
    sources, languages, artifacts = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        episode = episodes[row["id"]]
        counts = score_episode(episode, row["prediction"], row["error"])
        row["counts"] = dict(counts)
        totals.update(counts)
        families[episode.family].update(counts)
        origins[episode.text_origin + "/" + episode.label_origin].update(counts)
        sources[episode.source].update(counts)
        languages[episode.language].update(counts)
        artifacts[episode.artifact].update(counts)
    summary_path = directory / "summary.json"
    summary = loads(summary_path.read_text())
    summary.update(overall=summarize(totals), families={k: summarize(v) for k,v in families.items()},
                   origins={k: summarize(v) for k,v in origins.items()},
                   sources={k: summarize(v) for k,v in sources.items()},
                   languages={k: summarize(v) for k,v in languages.items()},
                   artifacts={k: summarize(v) for k,v in artifacts.items()})
    f1s = [v["field_f1"] for v in summary["families"].values() if v["field_f1"] is not None]
    summary["macro_field_f1"] = sum(f1s)/len(f1s) if f1s else None
    path.write_text("".join(dumps(row) + "\n" for row in rows))
    summary_path.write_text(dumps(summary, indent=2))
    return summary
