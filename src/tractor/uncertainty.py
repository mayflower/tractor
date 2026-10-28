"""Versioned uncertainty events and the null-decision rule; no ML imports.

Three distinct events, never interchangeable:

source_presence           The requested information is explicitly present in the
                          source. Present but invalid evidence is still present.
typed_answerability       Under the field's type, cardinality and processing rules
                          a permissible answer is determinable from the source.
final_outcome_correctness The complete projected field answer equals the
                          independent reference (a correct abstention counts).

p_null is P(not typed_answerability) under PROJECTION_POLICY: a compatible scalar
needs exactly one value, so a native ambiguity list is not answerable here.
confidence is P(final_outcome_correctness) of the answer actually returned at the
applied threshold. confidence is never 1 - p_null.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

EVENT_VERSION = "tractor-uncertainty-events-v1"
PROJECTION_POLICY = "clairic-scalar-single-value-v1"


class Event(StrEnum):
    SOURCE_PRESENCE = "source_presence"
    TYPED_ANSWERABILITY = "typed_answerability"
    FINAL_OUTCOME_CORRECTNESS = "final_outcome_correctness"


class Label(StrEnum):
    """Offline label states. Unknown is masked, never treated as negative."""

    PRESENT = "present"
    ABSENT = "absent"
    INVALID_PRESENT = "invalid_present"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


# Event targets per label state; None masks the example for that event.
TARGETS = {
    Label.PRESENT: {Event.SOURCE_PRESENCE: 1, Event.TYPED_ANSWERABILITY: 1},
    Label.ABSENT: {Event.SOURCE_PRESENCE: 0, Event.TYPED_ANSWERABILITY: 0},
    Label.INVALID_PRESENT: {Event.SOURCE_PRESENCE: 1, Event.TYPED_ANSWERABILITY: 0},
    Label.AMBIGUOUS: {Event.SOURCE_PRESENCE: 1, Event.TYPED_ANSWERABILITY: 0},
    Label.UNKNOWN: {Event.SOURCE_PRESENCE: None, Event.TYPED_ANSWERABILITY: None},
}


@dataclass(frozen=True)
class FieldUncertainty:
    """Probabilities for one field; None means the capability did not produce one."""

    p_null: float | None = None
    confidence: float | None = None


class NullDecisionInconsistent(Exception):
    """p_null says answer, but no valid value exists to return."""


def null_decision(value, p_null, threshold):
    """(value, guess) under: value is null exactly when p_null > threshold.

    Equality does not exceed the threshold. With p_null <= threshold a real
    value must exist; we never invent one or return a nonconforming null.
    """
    if p_null is None:
        raise ValueError("A null decision requires an actual p_null")
    if Decimal(str(p_null)) > Decimal(threshold):
        return None, value
    if value is None:
        raise NullDecisionInconsistent("p_null is at or below the threshold but no value exists")
    return value, value


class UncertaintyProvider:
    """Interface for released uncertainty artifacts.

    `capabilities` declares executable capabilities with their exact binding;
    `score(plan, trace)` returns {field name: FieldUncertainty}. A provider is
    attached only after its artifacts verified against the loaded model.
    """

    capabilities = ()

    def score(self, plan, trace):
        raise NotImplementedError
