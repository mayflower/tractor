"""Uncertainty events, heads, preflight, calibration and artifact binding (CPU fixtures).

Tiny random tensors check software behavior only; nothing here is evidence of
model quality or calibration of real BiQwen outputs.
"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tractor.capabilities import IncompatibleArtifact
from tractor.document_sources import envelope
from tractor.episodes import Answer
from tractor.jsonio import loads
from tractor.uncertainty import Event, NullDecisionInconsistent, null_decision
from tractor.uncertainty_calibration import (
    artifact,
    calibrate,
    fit_platt,
    metrics,
    release,
    split_rows,
)
from tractor.uncertainty_training import audit_rows, field_targets, label_rows, validate_config

P, A = Event.SOURCE_PRESENCE, Event.TYPED_ANSWERABILITY


@pytest.mark.parametrize("p_null,threshold,value,expected", [
    (0.5, "0.5", "x", ("x", "x")), (0.51, "0.5", "x", (None, "x")), (0.49, "0.5", "x", ("x", "x")),
    (1.0, "1", "x", ("x", "x")), (0.9, "0.5", None, (None, None)),
])
def test_null_decision_boundaries(p_null, threshold, value, expected):
    assert null_decision(value, p_null, threshold) == expected


def test_null_decision_never_invents_a_value():
    with pytest.raises(NullDecisionInconsistent):
        null_decision(None, 0.2, "0.5")
    with pytest.raises(ValueError):
        null_decision("x", None, "0.5")


def span(text, value):
    start = text.index(value)
    return {"start": start, "end": start + len(value), "text": value}


@pytest.mark.parametrize("answer,cardinality,targets", [
    ({"status": "present", "items": [span("A 1", "1")], "final_values": ["1"]}, "optional_one", (1, 1)),
    ({"status": "present", "items": [span("A 1", "1")], "final_values": ["1"],
      "expected_issues": ["invalid_number_format"]}, "optional_one", (1, 0)),
    ({"status": "present", "items": [span("A 1", "1")], "final_values": ["1", "2"]}, "optional_one", (1, 0)),
    ({"status": "present", "items": [span("A 1", "1")], "final_values": ["1", "2"]}, "unique_list", (1, 1)),
    ({"status": "present", "items": [span("A 1", "1")], "final_annotation_missing": True}, "optional_one", (1, None)),
    ({"status": "absent"}, "optional_one", (0, 0)),
    ({"status": "absent", "reference_complete": False}, "optional_one", (None, None)),
    ({"status": "ambiguous", "items": [span("A 1", "1")]}, "optional_one", (1, 0)),
    ({"status": "unknown_annotation"}, "optional_one", (None, None)),
])
def test_label_states_map_to_masked_event_targets(answer, cardinality, targets):
    result = field_targets(Answer.model_validate(answer), cardinality)
    assert (result[P], result[A]) == targets


def episode(split, group, source="synthetic", status="present"):
    text = "Invoice INV-1"
    item = {"status": status, "items": [span(text, "INV-1")]} if status == "present" else {"status": status}
    result = envelope(source, f"{group}-{split}", split, "metadata", "en", text,
                      [{"id": "t", "kind": "entities", "description": "Fields", "fields": {"n": "Number"}}],
                      {"t": {"complete": True, "records": [{"n": item}]}})
    return result.model_copy(update={"source_group": group, "derivation_group": group})


def test_label_rows_audit_splits_groups_and_cord():
    rows = label_rows([episode("train", "g1"), episode("dev", "g2", status="absent")])
    assert [r["targets"]["source_presence"] for r in rows] == [1, 0]
    assert audit_rows(rows)["errors"] == []
    assert audit_rows(label_rows([episode("train", "g1"), episode("dev", "g1")]))["errors"]
    assert "CORD" in audit_rows(label_rows([episode("train", "g3", source="cord-v2")]))["errors"][0]


def config():
    return loads(Path(__file__).parents[1].joinpath("configs/uncertainty-heads-dev.json").read_text())


def test_recipe_is_prepared_and_guards_its_contract():
    assert validate_config(config())
    for change in ({"status": "running"}, {"train_adapter": True, "selection": {"split": "dev"}},
                   {"calibration": {"split": "test"}}):
        with pytest.raises(ValueError):
            validate_config({**config(), **change})


def test_masked_loss_ignores_unknown_labels():
    torch = pytest.importorskip("torch")
    from tractor.uncertainty_training import masked_bce

    logits = torch.tensor([2.0, -1.0, 5.0], requires_grad=True)
    loss = masked_bce(logits, torch.tensor([1.0, float("nan"), 0.0]))
    loss.backward()
    assert logits.grad[1] == 0 and logits.grad[0] != 0
    assert masked_bce(logits, torch.tensor([float("nan")] * 3)) == 0


def test_preflight_runs_forward_backward_save_load_and_resume():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from tractor.uncertainty_training import preflight

    rows = label_rows([episode("train", f"g{i}", status="present" if i % 2 else "absent") for i in range(6)])
    report = preflight(config(), rows, dim=16)
    assert report["checks"][-1] == "resume" and "no quality" in report["scope"]


def test_heads_round_trip_strictly_and_untrained_heads_are_never_released(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from tractor.uncertainty_heads import initialize_for_training, load_heads, save_heads

    heads = initialize_for_training(16, 1)
    assert not heads.released
    description = save_heads(heads, tmp_path / "h", {"base": "rev1"}, {"run": "fixture"})
    restored = load_heads(tmp_path / "h", description, {"base": "rev1"})
    query, logits = torch.randn(16), torch.randn(7)
    assert torch.equal(heads(query, logits)["answerability_logit"], restored(query, logits)["answerability_logit"])
    assert not restored.released
    with pytest.raises(ValueError, match="another base"):
        load_heads(tmp_path / "h", description, {"base": "rev2"})
    with pytest.raises(ValueError, match="parameters differ"):
        load_heads(tmp_path / "h", {**description, "dim": 8}, {"base": "rev1"})
    (tmp_path / "h/uncertainty-heads.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_heads(tmp_path / "h", description, {"base": "rev1"})


def test_released_heads_emit_signals_in_the_same_decode_without_changing_extraction(monkeypatch):
    torch = pytest.importorskip("torch")
    from tractor.biqwen import decode_multiple, schema_for
    from tractor.types import ExtractRequest
    from tractor.uncertainty_heads import attach, initialize_for_training

    monkeypatch.setitem(sys.modules, "tractor._biqwen_runtime.encoding",
                        SimpleNamespace(token_span_to_chars=lambda text, offsets, pair: pair))
    req = ExtractRequest(text="Alpha 12 cm", fields={"n": {"type": "text", "description": "Name"}})
    pools = {"f0": {"pairs": [(0, 5), (6, 11)], "source_logits": torch.tensor([3., -3.])}}
    model = SimpleNamespace(config=SimpleNamespace(source_threshold=.5, candidate_top_k=48),
                            _temperature=lambda _: 1., trace_signals=True)
    meta = {"text": req.text, "offsets": [], "schema": schema_for(req)}
    baseline = decode_multiple(model, meta, pools, torch.zeros(1, 16), None, None)
    heads = initialize_for_training(16, 0)
    attach(model, heads)
    unreleased = decode_multiple(model, meta, pools, torch.zeros(1, 16), None, None)
    assert unreleased == baseline and "field_signals" not in unreleased
    heads.released = True
    released = decode_multiple(model, meta, pools, torch.zeros(1, 16), None, None)
    assert released["fields"] == baseline["fields"] and released["signals"] == baseline["signals"]
    assert set(released["field_signals"][0]) == {"field", "presence_logit", "answerability_logit"}


def synthetic(n, seed, slope=2.0, shift=0.0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        x = rng.uniform(-4, 4)
        answerable = int(rng.random() < 1 / (1 + 2.718281828 ** -(slope * x + shift)))
        score = rng.uniform(-3, 3)
        rows.append({"group": f"g{i}", "split": "calibration", "answerability_logit": x, "answerable": answerable,
                     "outcome_score": score, "value_correct": int(rng.random() < 1 / (1 + 2.718281828 ** -score)),
                     "reference_absent": 1 - answerable})
    return rows


def test_platt_recovers_known_parameters_and_detects_miscalibration():
    rows = synthetic(4000, 3)
    params = fit_platt([r["answerability_logit"] for r in rows], [r["answerable"] for r in rows])
    assert params["a"] == pytest.approx(2.0, abs=0.25) and params["b"] == pytest.approx(0.0, abs=0.2)
    overconfident = metrics([0.99 if r["answerable"] or i % 2 else 0.01 for i, r in enumerate(rows)],
                            [r["answerable"] for r in rows])
    calibrated = metrics([1 / (1 + 2.718281828 ** -(2 * r["answerability_logit"])) for r in rows],
                         [r["answerable"] for r in rows])
    assert overconfident["ece"] > 0.15 > 0.05 > calibrated["ece"]
    assert metrics([0.5] * 30, [1] * 15 + [0] * 15)["status"] == "insufficient_evidence"


def test_calibration_uses_only_its_split_and_fixtures_never_release():
    with pytest.raises(ValueError, match="calibration split"):
        split_rows([{"group": "g", "split": "test"}], {"fit": 1.0})
    result = calibrate(synthetic(3000, 5), {"fit": 0.6, "select": 0.2, "validate": 0.2}, [0.5])
    assert set(result["counts"]) == {"fit", "select", "validate"}
    made = artifact(result, {"fixture": True}, {"run": "fixture"}, {"sha256": "x", "file": "h"},
                    {"weights": [0] * 6, "bias": 0}, [0.5], fixture=True)
    with pytest.raises(ValueError, match="Fixture"):
        release(made, {"status": "approved", "max_ece": 1})
    with pytest.raises(ValueError, match="proposed"):
        release({**made, "fixture": False}, {"status": "proposed", "max_ece": 1})


def manifest():
    return {"files": {"checkpoint/heads.safetensors": "h1", "uncertainty/uncertainty-heads.safetensors": "u1"},
            "base": {"revision": "rev1"}, "checkpoint": {"manifest_sha256": "c1"}, "backend": "decoder",
            "contract": {"schema_version": "biqwen-types-v2", "attention_mode": "hybrid"}}


def released_artifact(**changes):
    from tractor.uncertainty_artifacts import FORMAT, binding_for

    params = {"a": 1.0, "b": 0.0, "sha256": "p"}
    entry = {"method": "platt", "params": {"a": 1.0, "b": 0.0}, "sha256": params["sha256"]}
    made = {"format": FORMAT, "fixture": False, "released": True, "binding": binding_for(manifest(), "fp32"),
            "heads": {"file": "uncertainty/uncertainty-heads.safetensors", "sha256": "u1"},
            "provenance": {"run": "r1"}, "threshold_scope": [0.5],
            "outcome_scorer": {"weights": [0.0] * 6, "bias": 2.0},
            "calibration": {"typed_answerability": entry, "final_outcome_correctness": entry}}
    return {**made, **changes}


def test_artifacts_bind_to_the_exact_model_state():
    from tractor.uncertainty_artifacts import ReleasedUncertainty, verify_binding

    assert verify_binding(released_artifact(), manifest(), "fp32")
    other = manifest()
    other["files"]["checkpoint/heads.safetensors"] = "h2"
    for broken, target, precision in [(released_artifact(), other, "fp32"),
                                      (released_artifact(), manifest(), "bf16"),
                                      (released_artifact(fixture=True), manifest(), "fp32"),
                                      (released_artifact(format="v0"), manifest(), "fp32")]:
        with pytest.raises(IncompatibleArtifact):
            ReleasedUncertainty(broken, target, precision)
    provider = ReleasedUncertainty(released_artifact(), manifest(), "fp32")
    assert all(c.executable for c in provider.capabilities)
    assert not any(c.executable for c in ReleasedUncertainty(released_artifact(released=False),
                                                             manifest(), "fp32").capabilities)


def scored(artifact_changes=(), signals=None, **spec):
    from decimal import Decimal

    from tractor.uncertainty_artifacts import ReleasedUncertainty

    provider = ReleasedUncertainty(released_artifact(**dict(artifact_changes)), manifest(), "fp32")
    threshold = spec.pop("null_threshold", None)
    field_spec = SimpleNamespace(type=spec.pop("type", "text"), explicit=frozenset(spec.pop("explicit", ())),
                                 threshold=Decimal(str(threshold)) if threshold is not None else Decimal("0.5"))
    field = SimpleNamespace(name="x", internal_id="f0", operation="supported_source_only", spec=field_spec)
    evidence = SimpleNamespace(candidates=(), values=(), rejected=(), issues=(),
                               field_signals={"answerability_logit": 2.0} if signals is None else signals)
    return provider.score(SimpleNamespace(fields=[field]), SimpleNamespace(field=lambda _: evidence))


def test_released_provider_scores_only_with_real_signals():
    from tractor.uncertainty import FieldUncertainty

    [(name, score)] = scored().items()
    assert name == "x" and score.p_null == pytest.approx(1 - 1 / (1 + 2.718281828 ** -2.0))
    assert score.confidence is not None
    # A missing answerability signal is never replaced by a placeholder.
    assert scored(signals={}) == {"x": FieldUncertainty(None, None)}


def test_unreleased_artifacts_produce_no_public_probabilities():
    assert scored([("released", False)]) == {}


def test_threshold_scope_applies_only_to_requested_null_decisions():
    from tractor.capabilities import CapabilityUnavailable

    scope = [("threshold_scope", [0.3])]
    assert scored(scope) == {}  # Default threshold outside the scope: nothing claimed, no error.
    assert set(scored(scope, type="list")) == {"x"}  # Lists never use a null decision.
    assert set(scored(scope, null_threshold=0.3, explicit=["null_threshold"])) == {"x"}
    with pytest.raises(CapabilityUnavailable):
        scored(scope, null_threshold=0.5, explicit=["null_threshold"])
