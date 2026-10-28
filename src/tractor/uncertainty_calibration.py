"""Calibration and evaluation of versioned uncertainty events; no ML imports.

Input: one already selected model state and its stored full production results
on the separate calibration split. Partitions (fit / select / validate) are
fixed by group hash before fitting; test is never used here. p_null is fitted
and the abstention policy fixed first; only then is final-outcome correctness
defined on the answers that policy actually returns, and calibrated.
"""

import hashlib
import math
from collections import defaultdict
from decimal import Decimal

from .uncertainty import EVENT_VERSION, PROJECTION_POLICY
from .uncertainty_artifacts import FORMAT, Platt

MIN_EXAMPLES = 200
MIN_PER_CLASS = 20
BINS = 10


def partition(group, fractions):
    """Deterministic group-separated partition; whole groups stay together."""
    point = int(hashlib.sha256(group.encode()).hexdigest()[:12], 16) / float(16 ** 12)
    total = 0.0
    for name, share in fractions.items():
        total += share
        if point < total:
            return name
    return list(fractions)[-1]


def split_rows(rows, fractions):
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("Partition fractions must sum to 1")
    if any(row["split"] != "calibration" for row in rows):
        raise ValueError("Calibration uses only the calibration split")
    parts = defaultdict(list)
    for row in rows:
        parts[partition(row["group"], fractions)].append(row)
    return dict(parts)


def fit_platt(scores, labels, l2=1e-3, iterations=100):
    """Newton-Raphson logistic fit of one score; deterministic and dependency free."""
    a, b = 1.0, 0.0
    for _ in range(iterations):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(scores, labels, strict=True):
            p = Platt(a, b)(x)
            w = max(p * (1 - p), 1e-12)
            ga += (p - y) * x
            gb += p - y
            haa += w * x * x
            hab += w * x
            hbb += w
        ga += l2 * a
        haa += l2
        hbb += 1e-12
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-10:
            break
    return {"a": a, "b": b}


def wilson(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def metrics(probabilities, labels, groups=None):
    """Brier, log loss, reliability bins and ECE (count-weighted |accuracy - confidence|)."""
    n = len(labels)
    if n == 0:
        return {"count": 0, "status": "insufficient_evidence"}
    eps = 1e-12
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)) / n
    log_loss = -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                    for p, y in zip(probabilities, labels, strict=True)) / n
    bins = []
    for index in range(BINS):
        members = [(p, y) for p, y in zip(probabilities, labels, strict=True)
                   if index / BINS <= p < (index + 1) / BINS or (index == BINS - 1 and p == 1)]
        if members:
            positives = sum(y for _, y in members)
            bins.append({"bin": [index / BINS, (index + 1) / BINS], "count": len(members),
                         "mean_probability": sum(p for p, _ in members) / len(members),
                         "observed": positives / len(members), "interval": wilson(positives, len(members))})
    ece = sum(b["count"] * abs(b["observed"] - b["mean_probability"]) for b in bins) / n
    positives = sum(labels)
    status = ("insufficient_evidence" if n < MIN_EXAMPLES or min(positives, n - positives) < MIN_PER_CLASS
              else "measured")
    result = {"count": n, "positives": positives, "brier": brier, "log_loss": log_loss, "ece": ece,
              "ece_definition": f"{BINS} equal-width bins, count-weighted |observed - mean probability|",
              "bins": bins, "status": status}
    if groups is not None:
        result["groups"] = len(set(groups))
    return result


def risk_coverage(confidences, correct):
    """Error risk among answered fields as coverage decreases by confidence."""
    order = sorted(zip(confidences, correct, strict=True), key=lambda item: -item[0])
    curve, errors = [], 0
    for answered, (confidence, ok) in enumerate(order, start=1):
        errors += 1 - ok
        curve.append({"coverage": answered / len(order), "risk": errors / answered, "threshold": confidence})
    return curve


def projected_correct(row, p_null, threshold):
    """Correctness of the answer the fixed policy returns: value or abstention."""
    abstain = Decimal(str(p_null)) > Decimal(str(threshold))
    if abstain:
        return int(row["reference_absent"])
    return int(row["value_correct"])


def calibrate(rows, fractions, threshold_scope):
    """Fit p_null, fix the policy, then fit confidence on the policy's actual outcomes.

    Each row: group, split, answerability_logit, answerable (0/1/None),
    outcome_score, value_correct (0/1), reference_absent (0/1).
    """
    parts = split_rows(rows, fractions)
    fit = [r for r in parts.get("fit", []) if r["answerable"] is not None]
    answerability = fit_platt([r["answerability_logit"] for r in fit], [r["answerable"] for r in fit])
    calibrator = Platt(**answerability)

    def p_null(row):
        return 1 - calibrator(row["answerability_logit"])

    report = {"typed_answerability": {}, "final_outcome_correctness": {}}
    for name in ("select", "validate"):
        known = [r for r in parts.get(name, []) if r["answerable"] is not None]
        report["typed_answerability"][name] = metrics(
            [1 - p_null(r) for r in known], [r["answerable"] for r in known], [r["group"] for r in known])
    confidence = {}
    for threshold in threshold_scope:
        targets = [projected_correct(r, p_null(r), threshold) for r in parts.get("fit", [])]
        params = fit_platt([r["outcome_score"] for r in parts.get("fit", [])], targets)
        confidence[str(threshold)] = params
        scorer = Platt(**params)
        for name in ("select", "validate"):
            members = parts.get(name, [])
            correct = [projected_correct(r, p_null(r), threshold) for r in members]
            probabilities = [scorer(r["outcome_score"]) for r in members]
            entry = metrics(probabilities, correct, [r["group"] for r in members])
            entry["risk_coverage"] = risk_coverage(probabilities, correct)
            report["final_outcome_correctness"].setdefault(name, {})[str(threshold)] = entry
    return {"answerability": answerability, "confidence": confidence,
            "counts": {k: len(v) for k, v in parts.items()}, "report": report}


def artifact(result, binding, provenance, heads, outcome_scorer, threshold_scope, *, fixture):
    """A versioned artifact. Fixtures are marked and can never be released."""
    from .jsonio import dumps

    def digest(value):
        return hashlib.sha256(dumps(value, sort_keys=True).encode()).hexdigest()

    if len(threshold_scope) != 1:
        raise ValueError("One confidence calibration per released threshold scope")
    threshold = str(threshold_scope[0])
    calibration = {
        "typed_answerability": {"method": "platt", "params": result["answerability"],
                                "sha256": digest(result["answerability"])},
        "final_outcome_correctness": {"method": "platt", "params": result["confidence"][threshold],
                                      "sha256": digest(result["confidence"][threshold])},
    }
    return {"format": FORMAT, "fixture": fixture, "released": False, "event_version": EVENT_VERSION,
            "projection_policy": PROJECTION_POLICY, "binding": binding, "provenance": provenance,
            "heads": heads, "outcome_scorer": outcome_scorer, "threshold_scope": threshold_scope,
            "calibration": calibration, "counts": result["counts"], "report": result["report"]}


def release(artifact, criteria):
    """Release only with explicit owner-approved criteria that the report meets."""
    if artifact["fixture"]:
        raise ValueError("Fixture artifacts cannot be released")
    if criteria.get("status") != "approved":
        raise ValueError("Release criteria are proposed, not approved")
    report = artifact["report"]
    validate = [report["typed_answerability"]["validate"],
                *report["final_outcome_correctness"]["validate"].values()]
    if any(entry["status"] != "measured" for entry in validate):
        raise ValueError("insufficient_evidence for at least one event")
    if any(entry["ece"] > criteria["max_ece"] for entry in validate):
        raise ValueError("Calibration error exceeds the approved criterion")
    return {**artifact, "released": True, "release_criteria": criteria}
