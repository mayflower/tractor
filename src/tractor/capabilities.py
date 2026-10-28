"""Declared model capabilities, their evidence levels and local API errors; no ML imports.

A capability is executable only when every evidence level its contract needs is
present. Parsing a request type never creates a capability, and a feature flag
alone never passes a gate that requires trained weights or calibration.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

CAPABILITIES = (
    "source_spans",
    "typed_extraction",
    "typed_lists",
    "source_presence",
    "typed_answerability",
    "final_outcome_confidence",
    "deterministic_compute",
    "numeric_distribution",
    "forecast",
    "images",
)
# Which evidence a capability needs before it may run. Deterministic software
# paths need no weights; learned probabilities need weights AND calibration.
REQUIREMENTS = MappingProxyType({
    "source_spans": ("architecture", "weights"),
    "typed_extraction": ("architecture", "weights"),
    "typed_lists": ("architecture", "weights"),
    "source_presence": ("architecture", "weights", "provenance", "calibration"),
    "typed_answerability": ("architecture", "weights", "provenance", "calibration"),
    "final_outcome_confidence": ("architecture", "weights", "provenance", "calibration"),
    "deterministic_compute": ("architecture",),
    "numeric_distribution": ("architecture", "weights", "provenance", "calibration"),
    "forecast": ("architecture", "weights", "provenance", "calibration"),
    "images": ("architecture", "weights", "provenance"),
})


@dataclass(frozen=True)
class Capability:
    """Separate evidence levels; there is no single calibrated flag.

    `binding` names the exact artifacts (weight hashes, schema, decoder,
    normalization/projection versions, precision, policy) the evidence is for.
    """

    name: str
    architecture: bool = False
    weights: str | None = None
    provenance: str | None = None
    calibration: str | None = None
    released: bool = False
    binding: dict = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        if self.name not in CAPABILITIES:
            raise ValueError(f"Unknown capability: {self.name}")

    def missing(self):
        levels = {"architecture": self.architecture, "weights": self.weights,
                  "provenance": self.provenance, "calibration": self.calibration}
        gaps = [level for level in REQUIREMENTS[self.name] if not levels[level]]
        return gaps + ([] if self.released else ["release"])

    @property
    def executable(self):
        return not self.missing()

    def describe(self):
        return {"name": self.name, "executable": self.executable, "missing": self.missing(),
                "architecture": self.architecture, "weights": self.weights,
                "provenance": self.provenance, "calibration": self.calibration,
                "released": self.released, "binding": self.binding, "note": self.note}


class ModelCapabilities:
    """The capabilities one loaded model instance actually declares."""

    def __init__(self, capabilities=()):
        self._items = {name: Capability(name) for name in CAPABILITIES}
        for capability in capabilities:
            self._items[capability.name] = capability

    def __getitem__(self, name):
        return self._items[name]

    def executable(self, name):
        return self._items[name].executable

    def add(self, capability):
        self._items[capability.name] = capability

    def require(self, names, field=None):
        for name in names:
            if not self._items[name].executable:
                missing = ", ".join(self._items[name].missing())
                raise CapabilityUnavailable(
                    f"Capability {name} is not available ({missing})", field=field, capability=name)

    def describe(self):
        return [self._items[name].describe() for name in CAPABILITIES]


def extractor_capabilities(extractor, *, compute=False):
    """Describe only what an existing extraction backend demonstrably provides.

    Legacy and current BiQwen/GLiNER bundles provide source spans and typed
    extraction through the production normalizers. They carry no presence,
    answerability, confidence or distribution weights; those stay absent.
    """
    binding = {"backend": getattr(extractor, "backend", type(extractor).__name__),
               "model_revision": getattr(extractor, "model_revision", None),
               "schema_version": getattr(extractor, "schema_version", None),
               "precision": getattr(extractor, "precision", None)}
    weights = binding["model_revision"] or "unversioned-test-backend"
    items = [Capability(name, architecture=True, weights=weights, released=True, binding=binding)
             for name in ("source_spans", "typed_extraction", "typed_lists")]
    if compute:
        items.append(Capability("deterministic_compute", architecture=True, released=True,
                                note="Deterministic Decimal arithmetic over extracted operands"))
    declared = getattr(extractor, "capabilities", None)
    if declared is not None:
        items.extend(declared)
    return ModelCapabilities(items)


class ClairicError(Exception):
    """Local compatibility error; codes are local strings, never Cloudflare numbers."""

    status = 400
    code = "tractor.invalid_request"
    keeps_ready = True  # A request-level failure says nothing about the model's health.

    def __init__(self, message, *, field=None, capability=None, details=None):
        super().__init__(message)
        self.field = field
        self.capability = capability
        self.details = details or []

    def describe(self):
        error = {"code": self.code, "message": str(self)}
        if self.field is not None:
            error["field"] = self.field
        if self.capability is not None:
            error["capability"] = self.capability
        if self.details:
            error["details"] = self.details
        return error


class InvalidInput(ClairicError):
    status = 400
    code = "tractor.invalid_request"


class UnsupportedOperation(ClairicError):
    status = 422
    code = "tractor.unsupported_operation"


class CapabilityUnavailable(ClairicError):
    status = 422
    code = "tractor.capability_unavailable"


class CapacityExceeded(ClairicError):
    status = 413
    code = "tractor.capacity_exceeded"


class IncompatibleArtifact(ClairicError):
    status = 503
    code = "tractor.incompatible_artifact"


class ProcessingFailed(ClairicError):
    """Present evidence was invalid or ambiguous; never reported as an empty answer."""

    status = 422
    code = "tractor.unresolved_field"
