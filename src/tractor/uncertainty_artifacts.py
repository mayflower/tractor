"""Uncertainty artifact binding and the released provider; no ML imports.

An artifact is usable only for the exact model state it was fitted on: weight
hashes, merged base, schema, decoder, normalization/projection/compiler
versions, precision and threshold scope. Fixture artifacts can never be
released. A mismatch disables the affected capability; it never degrades to
uncalibrated scores.
"""

import math
from decimal import Decimal

from .capabilities import Capability, CapabilityUnavailable, IncompatibleArtifact
from .uncertainty import (
    EVENT_VERSION,
    PROJECTION_POLICY,
    Event,
    FieldUncertainty,
    UncertaintyProvider,
)

FORMAT = "tractor-uncertainty-artifact-v1"
FEATURES_VERSION = "tractor-outcome-features-v1"
EVENT_CAPABILITY = {Event.TYPED_ANSWERABILITY.value: "typed_answerability",
                    Event.SOURCE_PRESENCE.value: "source_presence",
                    Event.FINAL_OUTCOME_CORRECTNESS.value: "final_outcome_confidence"}


def current_versions():
    from .clairic_compiler import COMPILER_VERSION
    from .clairic_results import PROJECTION_VERSION

    return {"compiler": COMPILER_VERSION, "projection": PROJECTION_VERSION,
            "projection_policy": PROJECTION_POLICY, "events": EVENT_VERSION, "features": FEATURES_VERSION}


def binding_for(manifest, precision):
    """The exact model state an artifact applies to, from a verified bundle manifest.

    The artifact's own heads under uncertainty/ are bound separately by their
    hash, so recording them in a bundle does not change the bound state.
    """
    weights = {k: v for k, v in manifest["files"].items()
               if not k.startswith("uncertainty/") and (k.startswith("checkpoint/") or k.endswith(".safetensors"))}
    return {"base_revision": manifest["base"]["revision"], "weights": weights,
            "checkpoint_manifest_sha256": manifest["checkpoint"]["manifest_sha256"],
            "schema_version": manifest["contract"]["schema_version"],
            "attention_mode": manifest["contract"]["attention_mode"],
            "decoder": manifest["backend"], "precision": precision, **current_versions()}


def verify_binding(artifact, manifest, precision=None):
    if artifact.get("format") != FORMAT:
        raise IncompatibleArtifact("Unsupported uncertainty artifact format")
    if artifact.get("fixture", True):
        raise IncompatibleArtifact("Fixture artifacts cannot be released")
    expected = binding_for(manifest, precision or artifact["binding"].get("precision"))
    actual = artifact["binding"]
    differing = sorted(k for k in expected if actual.get(k) != expected[k])
    if differing:
        raise IncompatibleArtifact(f"Uncertainty artifact is bound to another model state: {differing}")
    heads = artifact.get("heads")
    if heads is not None and manifest["files"].get(heads["file"]) != heads["sha256"]:
        raise IncompatibleArtifact("Uncertainty heads are missing from the bundle or changed")
    return True


class Platt:
    """p = sigmoid(a * x + b), fitted on a separate calibration split."""

    def __init__(self, a, b):
        self.a, self.b = float(a), float(b)

    def __call__(self, x):
        z = self.a * x + self.b
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def outcome_features(evidence, field_signals):
    """Inference-time features of one projected field; no gold, no second backbone call.

    None when the answerability signal is absent: a missing signal is never
    replaced by a placeholder value.
    """
    if "answerability_logit" not in field_signals:
        return None
    scores = [o.signals["source_score"] for o in evidence.candidates if "source_score" in o.signals]
    return [max(scores, default=0.0), float(len(evidence.candidates)), float(len(evidence.values)),
            float(len(evidence.rejected)), float(bool([c for c in evidence.issues if c != "not_found"])),
            float(field_signals["answerability_logit"])]


class OutcomeScorer:
    """Logistic baseline over documented features; scores the whole field decision."""

    def __init__(self, weights, bias):
        self.weights, self.bias = [float(w) for w in weights], float(bias)

    def __call__(self, features):
        if len(features) != len(self.weights):
            raise ValueError("Feature definition differs from the scorer")
        return sum(w * x for w, x in zip(self.weights, features, strict=True)) + self.bias


class ReleasedUncertainty(UncertaintyProvider):
    """Probabilities from verified artifacts only. Built after binding verification."""

    def __init__(self, artifact, manifest, precision):
        verify_binding(artifact, manifest, precision)
        self.artifact = artifact
        self.released = bool(artifact.get("released"))
        calibration = artifact["calibration"]
        self.scope = [Decimal(str(t)) for t in artifact["threshold_scope"]]
        self.p_null = Platt(**calibration[Event.TYPED_ANSWERABILITY.value]["params"]) \
            if Event.TYPED_ANSWERABILITY.value in calibration else None
        self.confidence = None
        if Event.FINAL_OUTCOME_CORRECTNESS.value in calibration:
            scorer = artifact["outcome_scorer"]
            self.scorer = OutcomeScorer(scorer["weights"], scorer["bias"])
            self.confidence = Platt(**calibration[Event.FINAL_OUTCOME_CORRECTNESS.value]["params"])
        self.capabilities = [
            Capability(EVENT_CAPABILITY[event], architecture=True, weights=artifact["heads"]["sha256"],
                       provenance=artifact["provenance"]["run"], calibration=entry["sha256"],
                       released=artifact.get("released", False), binding=artifact["binding"])
            for event, entry in calibration.items()]

    def score(self, plan, trace):
        if not self.released:
            return {}  # An unreleased artifact never produces public probabilities.
        scores = {}
        for field in plan.fields:
            if field.operation != "supported_source_only":
                continue  # Compute paths need their own event and artifacts.
            if field.spec.type != "list" and field.spec.threshold not in self.scope:
                if "null_threshold" in field.spec.explicit:
                    raise CapabilityUnavailable("null_threshold is outside the calibrated scope",
                                                field=field.name, capability="typed_answerability")
                continue  # The default threshold was not calibrated: claim nothing for this field.
            evidence = trace.field(field.internal_id)
            signals = dict(evidence.field_signals)
            p_null = None
            if self.p_null is not None and "answerability_logit" in signals:
                # p_null is the probability of NOT typed_answerability.
                p_null = 1 - self.p_null(signals["answerability_logit"])
            confidence = None
            features = outcome_features(evidence, signals)
            if self.confidence is not None and features is not None:
                confidence = self.confidence(self.scorer(features))
            scores[field.name] = FieldUncertainty(p_null, confidence)
        return scores


def add_to_bundle(bundle, artifact, heads_file):
    """Record a calibrated artifact and its heads in an inference bundle manifest.

    Everything is checked before the bundle changes; on any later failure the
    heads and the original manifest are restored, so a retry starts clean.
    """
    import shutil
    from pathlib import Path

    from .inference_bundle import MANIFEST, sha, verify_inference_bundle
    from .jsonio import dumps, loads

    bundle = Path(bundle)
    original = (bundle / MANIFEST).read_text()
    manifest = loads(original)
    if manifest.get("uncertainty") is not None:
        raise ValueError("Bundle already declares uncertainty artifacts")
    if sha(Path(heads_file)) != artifact["heads"]["sha256"]:
        raise ValueError("Heads file differs from the artifact")
    target = bundle / "uncertainty" / Path(heads_file).name
    name = target.relative_to(bundle).as_posix()
    manifest["files"][name] = artifact["heads"]["sha256"]
    manifest["uncertainty"] = {**artifact, "heads": {**artifact["heads"], "file": name}}
    manifest["capabilities"] = sorted({*manifest["capabilities"],
                                       *(EVENT_CAPABILITY[e] for e in artifact["calibration"])})
    verify_binding(manifest["uncertainty"], manifest)
    if target.parent.exists():
        raise FileExistsError(f"Refusing to overwrite {target.parent}")
    try:
        target.parent.mkdir()
        shutil.copyfile(heads_file, target)
        (bundle / MANIFEST).write_text(dumps(manifest, indent=2, sort_keys=True) + "\n")
        return verify_inference_bundle(bundle)
    except BaseException:
        (bundle / MANIFEST).write_text(original)
        shutil.rmtree(target.parent, ignore_errors=True)
        raise


def load_released(bundle, extractor, precision):
    """Provider for a bundle's verified artifact, with its heads attached; or None."""
    from pathlib import Path

    from .inference_bundle import MANIFEST
    from .jsonio import loads

    manifest = loads((Path(bundle) / MANIFEST).read_text())
    artifact = manifest.get("uncertainty")
    if artifact is None:
        return None
    provider = ReleasedUncertainty(artifact, manifest, precision)
    from .uncertainty_heads import attach, load_heads

    heads = load_heads(bundle, artifact["heads"], artifact["binding"])
    heads.released = bool(artifact.get("released"))
    attach(extractor.model, heads.to(next(extractor.model.parameters()).device))
    return provider
