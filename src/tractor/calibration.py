"""Per-example offline reuse of threshold-independent native neural outputs.

The encoder/schema/head inputs are identical across threshold settings. Native
candidate filtering and record decoding still execute for every setting. This
context is never used by serving or benchmarking.
"""

from contextlib import contextmanager


def field_observations(episode, raw):
    """Emitted complete-field correctness, including one-to-one record assignment."""
    from .episode_metrics import compare_field, prediction_rows, score_task

    observations = []
    for task in episode.tasks:
        gold = episode.gold[task.id]
        predicted = prediction_rows(task, raw)
        _, matches = score_task(task, gold, predicted, with_matches=True)
        assignments = {j: i for i, j in matches}
        if task.kind == "entities":
            rows = [raw.get("entities", {})]
        elif task.kind == "classification":
            rows = [{"label": raw.get(task.id)}]
        elif task.kind == "relations":
            rows = raw.get("relation_extraction", {}).get(task.id, [])
        else:
            rows = raw.get(task.id, [])
        for j, row in enumerate(rows):
            expected = gold.records[assignments[j]] if j in assignments else {}
            for key in predicted[j]:
                value = row.get(key)
                values = value if isinstance(value, list) else [value] if value else []
                if not values:
                    continue
                answer = expected.get(key)
                if answer and answer.status in {"unknown_annotation", "ambiguous"}:
                    continue
                if not all(isinstance(v, dict) and "confidence" in v for v in values):
                    continue
                confidence = min(float(v["confidence"]) for v in values)
                correct = False
                if answer:
                    _, fp, fn = compare_field(answer, predicted[j][key])
                    correct = fp == fn == 0 and answer.status == "present"
                observations.append((confidence, int(correct)))
    return observations


def fit_score_mapping(observations):
    """Small monotone logistic calibration; not a claim of universal probability accuracy."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    if len(observations) < 20 or len({y for _, y in observations}) < 2:
        return None
    scores = np.clip([s for s, _ in observations], 1e-6, 1 - 1e-6)
    x, y = np.log(scores / (1 - scores)), np.array([y for _, y in observations])

    def loss(parameters):
        probabilities = np.clip(expit(parameters[0] * x + parameters[1]), 1e-9, 1 - 1e-9)
        return float(-(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)).mean())

    result = minimize(loss, [1., 0.], bounds=[(0., 20.), (-20., 20.)])
    return {"scale": float(result.x[0]), "bias": float(result.x[1])} if result.success else None


def calibration_error(observations, mapping=None):
    import math

    if not observations:
        return {"emitted_fields": 0, "brier": None, "ece": None}
    mapped = []
    for score, label in observations:
        score = min(max(score, 1e-6), 1 - 1e-6)
        if mapping:
            logit = mapping["scale"] * math.log(score / (1 - score)) + mapping["bias"]
            score = 1 / (1 + math.exp(-max(-60, min(60, logit))))
        mapped.append((score, label))
    ece = 0.
    for index in range(10):
        values = [(s, y) for s, y in mapped if min(int(s * 10), 9) == index]
        if values:
            ece += abs(sum(s for s, _ in values) - sum(y for _, y in values)) / len(mapped)
    return {"emitted_fields": len(mapped),
            "brier": sum((s - y) ** 2 for s, y in mapped) / len(mapped), "ece": ece}


def calibrate_tasks(model, path, output, device="cpu"):
    """Fit on one source-group half; choose a raw native threshold on the other half."""
    import hashlib
    from collections import Counter
    from pathlib import Path

    from .episode_metrics import score_episode, summarize
    from .generalist_data import read_episodes
    from .jsonio import dumps
    from .model import COMMIT, Extractor

    episodes = list(read_episodes(path))
    if not episodes or any(e.split != "calibration" for e in episodes):
        raise ValueError("Native calibration requires a nonempty separate calibration split")
    if any(not gold.complete or any(answer.status in {"ambiguous", "unknown_annotation"}
                                   for row in gold.records for answer in row.values())
           for episode in episodes for gold in episode.gold.values()):
        raise ValueError("Native calibration needs complete, unambiguous annotations")
    fit, selection = [], []
    for episode in episodes:
        bit = int(hashlib.sha256(episode.source_group.encode()).hexdigest()[:8], 16) % 2
        (fit if bit else selection).append(episode)
    if min(len({e.source_group for e in rows}) for rows in (fit, selection)) < 10:
        raise ValueError("At least ten independent source groups per calibration half are required")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    extractor = Extractor(model, device)
    fitted, fit_errors = [], 0
    for episode in fit:
        try:
            raw = extractor.extract_tasks(episode.text, episode.tasks, threshold=.25)
            fitted.extend(field_observations(episode, raw))
        except Exception:
            fit_errors += 1
    mapping = fit_score_mapping(fitted)
    policies = []
    with destination.joinpath("predictions.jsonl").open("w") as stream:
        for threshold in (.25, .5, .75):
            counts, observations = Counter(), []
            for episode in selection:
                raw, error = {}, None
                try:
                    raw = extractor.extract_tasks(episode.text, episode.tasks, threshold=threshold)
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                counts.update(score_episode(episode, raw, error))
                if not error:
                    observations.extend(field_observations(episode, raw))
                stream.write(dumps({"id": episode.id, "source_group": episode.source_group,
                                    "threshold": threshold, "prediction": raw, "error": error}) + "\n")
            policies.append({"threshold": threshold, "metrics": summarize(counts),
                             "raw_score_error": calibration_error(observations),
                             "mapped_score_error": calibration_error(observations, mapping) if mapping else None})
    normal = max(policies, key=lambda p: p["metrics"]["field_f1"] or 0)
    if not any(p["metrics"]["counts"].get("answered", 0) for p in policies):
        destination.joinpath("failure.json").write_text(dumps({"policies": policies}, indent=2))
        raise ValueError("No emitted fields: refusing to write a native policy")
    candidates = [p for p in policies if (p["metrics"]["field_precision"] or 0) >= .99
                  and p["metrics"]["counts"].get("answered", 0) >= 20]
    conservative = max(candidates, key=lambda p: p["metrics"]["answerable_coverage"] or 0) if candidates else None
    report = {"model_sha256": extractor.model_revision, "fit_source_groups": len({e.source_group for e in fit}),
              "selection_source_groups": len({e.source_group for e in selection}), "fit_errors": fit_errors,
              "score_mapping": mapping, "policies": policies, "normal": normal, "conservative": conservative,
              "limitations": "Conditional emitted-field score diagnostic; point-estimate precision target, not an SLA. "
              "Scores remain raw in inference. Single-label classification may not abstain under the native threshold. "
              "No final test is executed here."}
    destination.joinpath("report.json").write_text(dumps(report, indent=2))
    policy = {"format": "tractor-native-threshold-v1", "model_sha256": extractor.model_revision,
              "gliner_commit": COMMIT, "threshold": normal["threshold"]}
    destination.joinpath("native-policy.json").write_text(dumps(policy, indent=2))
    return report


@contextmanager
def reuse_neural_outputs(extractor):
    replacements = []
    for owner, name in [
        (extractor.model, "_encode_core"),
        (extractor.model.boundary_head, "forward"),
        (extractor.model.record_decoder, "forward_groups_dense"),
    ]:
        original = getattr(owner, name)
        cache = []

        def cached(*args, _original=original, _cache=cache, **kwargs):
            if not _cache:
                _cache.append(_original(*args, **kwargs))
            return _cache[0]

        setattr(owner, name, cached)
        replacements.append((owner, name, original))
    try:
        yield
    finally:
        for owner, name, original in replacements:
            setattr(owner, name, original)
