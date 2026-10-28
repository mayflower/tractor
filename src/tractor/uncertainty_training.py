"""Labels, masked losses and a reproducible preflight for the uncertainty heads.

This module prepares training; it starts no real BiQwen run. Unknown labels
are masked per event and never counted as negatives. A missing gold value is
not an absence label, and an empty prediction never proves source absence.
Train fits heads (and only explicitly configured adapters), Dev selects,
calibration is fitted later on its own split, and test stays untouched.
"""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from .jsonio import dumps, loads
from .source_policy import excluded_source
from .uncertainty import EVENT_VERSION, Event

RECIPE = "tractor-uncertainty-heads-v1"
SPLITS = ("train", "dev", "calibration", "test")
SCALAR = {"optional_one", "required_one"}
CONFIG_KEYS = {"recipe", "base_bundle", "dataset", "outcome_predictions", "output", "heads_lr",
               "weight_decay", "epochs", "seed", "train_adapter", "loss_weights", "selection",
               "calibration", "resources", "resume", "status"}


def field_targets(answer, cardinality):
    """Per-event targets for one annotated field: 1, 0 or None (masked)."""
    status = answer.status
    if status == "unknown_annotation":
        return {Event.SOURCE_PRESENCE: None, Event.TYPED_ANSWERABILITY: None}
    if status == "absent":
        # Absence is only known from an exhaustive reference.
        known = answer.reference_complete
        return {Event.SOURCE_PRESENCE: 0 if known else None, Event.TYPED_ANSWERABILITY: 0 if known else None}
    presence = 1
    if status == "ambiguous":
        return {Event.SOURCE_PRESENCE: presence,
                Event.TYPED_ANSWERABILITY: 0 if cardinality in SCALAR else None}
    if answer.expected_issues:
        return {Event.SOURCE_PRESENCE: presence, Event.TYPED_ANSWERABILITY: 0}
    if answer.final_annotation_missing or (answer.final_values is None and not answer.final_alternatives):
        return {Event.SOURCE_PRESENCE: presence, Event.TYPED_ANSWERABILITY: None}
    answers = [answer.final_values] if answer.final_values is not None else answer.final_alternatives
    if cardinality in SCALAR and all(len(set(values)) > 1 for values in answers):
        # Several distinct values cannot be one compatible scalar answer.
        return {Event.SOURCE_PRESENCE: presence, Event.TYPED_ANSWERABILITY: 0}
    return {Event.SOURCE_PRESENCE: presence, Event.TYPED_ANSWERABILITY: 1}


def outcome_target(observation):
    """final_outcome_correctness from the shared production comparator, or None."""
    return int(observation["exact"]) if observation["known"] else None


def provenance(episode):
    return {"source": episode.source, "text_origin": episode.text_origin,
            "label_origin": episode.label_origin, "annotation": episode.annotation}


def label_rows(episodes):
    """One row per (episode, task, record, field); duplicates are not independent."""
    rows = []
    for episode in episodes:
        for task in episode.tasks:
            gold = episode.gold.get(task.id)
            if gold is None:
                continue
            rules = episode.processing.get(task.id, {})
            for index, record in enumerate(gold.records):
                for key, answer in record.items():
                    rule = rules.get(key)
                    cardinality = rule.cardinality if rule else task.cardinalities.get(key, "unique_list")
                    targets = field_targets(answer, cardinality)
                    if not gold.complete and answer.status == "absent":
                        targets = dict.fromkeys(targets)
                    rows.append({"episode": episode.id, "group": episode.source_group,
                                 "derivation_group": episode.derivation_group, "split": episode.split,
                                 "task": task.id, "record": index, "field": key,
                                 "kind": rule.kind if rule else "literal", "cardinality": cardinality,
                                 "language": episode.language, "provenance": provenance(episode),
                                 "excluded": excluded_source(episode),
                                 "targets": {e.value: t for e, t in targets.items()}})
    return rows


def audit_rows(rows):
    """Coverage per event/type/split/provenance plus hard failures."""
    errors = []
    if any(row["excluded"] for row in rows):
        errors.append("CORD-derived rows are present")
    if any(row["split"] not in SPLITS for row in rows):
        errors.append("Unknown split")
    groups = defaultdict(set)
    for row in rows:
        groups[row["group"]].add(row["split"])
        groups["derivation:" + row["derivation_group"]].add(row["split"])
    overlapping = sorted(g for g, splits in groups.items() if len(splits) > 1)
    if overlapping:
        errors.append(f"Source groups cross splits: {overlapping[:5]}")
    keys = Counter((r["episode"], r["task"], r["record"], r["field"]) for r in rows)
    if any(count > 1 for count in keys.values()):
        errors.append("Duplicate field rows")
    coverage = defaultdict(Counter)
    for row in rows:
        for event, target in row["targets"].items():
            label = "unknown" if target is None else str(target)
            coverage[f"{row['split']}/{event}"][label] += 1
            coverage[f"{row['split']}/{event}/{row['kind']}"][label] += 1
            coverage[f"{row['split']}/{event}/{row['provenance']['text_origin']}"][label] += 1
    return {"errors": errors, "rows": len(rows), "event_version": EVENT_VERSION,
            "coverage": {k: dict(v) for k, v in sorted(coverage.items())}}


def fold_of(group, folds):
    return int(hashlib.sha256(group.encode()).hexdigest(), 16) % folds


def validate_outcome_rows(rows, folds):
    """Correctness labels must come from a predictor that never trained on that group."""
    for row in rows:
        if "outcome" not in row:
            continue
        if row.get("predictor_excluded_fold") != fold_of(row["group"], folds):
            raise ValueError(f"Outcome label of {row['episode']} is not out-of-fold")
    return True


def masked_bce(logits, targets):
    """Binary cross-entropy over known targets only; NaN marks unknown."""
    import torch
    from torch.nn import functional as F

    mask = ~torch.isnan(targets)
    if not mask.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def head_loss(outputs, targets, weights):
    import torch

    loss = 0
    for event, key in ((Event.SOURCE_PRESENCE.value, "presence_logit"),
                       (Event.TYPED_ANSWERABILITY.value, "answerability_logit")):
        target = torch.tensor([float("nan") if t is None else float(t) for t in targets[event]])
        loss = loss + float(weights[event]) * masked_bce(outputs[key], target)
    return loss


def validate_config(config):
    missing = CONFIG_KEYS - set(config)
    unknown = set(config) - CONFIG_KEYS
    if missing or unknown:
        raise ValueError(f"Recipe keys differ: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if config["recipe"] != RECIPE:
        raise ValueError("Unsupported uncertainty recipe")
    if config["status"] != "prepared_not_started":
        raise ValueError("Preflight only checks prepared, not started recipes")
    if config["train_adapter"] and config["selection"].get("regression_gates") != "full_pipeline":
        raise ValueError("Adapter training requires the full extraction/rejection/record regression gates")
    if config["selection"].get("split") != "dev" or config["calibration"].get("split") != "calibration":
        raise ValueError("Dev selects; calibration is fitted only on the calibration split")
    if Path(config["output"]).resolve() == Path("configs/active-training.json").resolve():
        raise ValueError("The active training configuration is not an output")
    return True


def preflight(config, rows, *, dim=32, seed=0):
    """Software checks with a tiny model; never loads the 4B weights."""
    import tempfile

    import torch

    from .uncertainty_heads import initialize_for_training, load_heads, save_heads

    validate_config(config)
    audit = audit_rows(rows)
    if audit["errors"]:
        raise ValueError("; ".join(audit["errors"]))
    train = [r for r in rows if r["split"] == "train"]
    if not train:
        raise ValueError("No training rows")
    torch.manual_seed(seed)
    heads = initialize_for_training(dim, seed)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=float(config["heads_lr"]),
                                  weight_decay=float(config["weight_decay"]))
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimized != {id(p) for p in heads.parameters()}:
        raise ValueError("Optimizer does not cover every new head parameter")
    batch = train[:8]
    queries, logits = torch.randn(len(batch), dim), [torch.randn(5) for _ in batch]
    outputs = [heads(q, s) for q, s in zip(queries, logits, strict=True)]
    stacked = {k: torch.stack([o[k] for o in outputs]) for k in outputs[0]}
    targets = {e: [r["targets"][e] for r in batch] for e in (Event.SOURCE_PRESENCE.value,
                                                             Event.TYPED_ANSWERABILITY.value)}
    loss = head_loss(stacked, targets, config["loss_weights"])
    optimizer.zero_grad()
    loss.backward()
    grads = {name: p.grad for name, p in heads.named_parameters()}
    if any(g is None or not torch.isfinite(g).all() for g in grads.values()):
        raise ValueError("Missing or non-finite head gradients")
    optimizer.step()
    with tempfile.TemporaryDirectory() as directory:
        binding = {"fixture": True}
        description = save_heads(heads, Path(directory) / "heads", binding, {"run": "preflight"})
        restored = load_heads(Path(directory) / "heads", description, binding)
        for name, value in heads.state_dict().items():
            if not torch.equal(value, restored.state_dict()[name]):
                raise ValueError(f"Head roundtrip changed {name}")
        state = {"optimizer": optimizer.state_dict(), "step": 1}
        torch.save(state, Path(directory) / "state.pt")
        resumed = torch.optim.AdamW(restored.parameters(), lr=float(config["heads_lr"]))
        resumed.load_state_dict(torch.load(Path(directory) / "state.pt", weights_only=True)["optimizer"])
    return {"recipe": RECIPE, "loss": float(loss.detach()), "parameters": sum(p.numel() for p in heads.parameters()),
            "audit": audit, "checks": ["config", "splits", "coverage", "forward_backward", "optimizer",
                                       "save_load", "resume"],
            "scope": "Software preflight with a tiny model; no quality or calibration evidence"}


def main(argv=None):
    import argparse

    from .generalist_data import read_episodes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = loads(Path(args.config).read_text())
    rows = label_rows(e for path in config["dataset"] for e in read_episodes(path))
    print(dumps(preflight(config, rows), indent=2))


if __name__ == "__main__":
    main()
