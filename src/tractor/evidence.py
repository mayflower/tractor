"""Request-scoped typed evidence from source occurrence to final value; no ML imports.

source occurrence -> model signals -> selection/assignment -> normalization or
failure -> final value(s) -> public projection. Offsets are half-open Python
string indices (Unicode code points) into the unmodified input text. Missing
signals stay missing; parsing success never turns into a confidence of 1.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

SIGNAL_NAMES = ("raw_source_logit", "source_score", "source_temperature", "assignment_margin")


@dataclass(frozen=True)
class Occurrence:
    start: int
    end: int
    text: str
    # Only signals the backend actually produced; never substituted defaults.
    signals: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    @property
    def span(self):
        return [self.start, self.end]


@dataclass(frozen=True)
class ValueEvidence:
    """One normalized value with every occurrence that produced it."""

    value: object
    occurrences: tuple


@dataclass(frozen=True)
class Rejection:
    occurrence: Occurrence | None
    code: str


@dataclass(frozen=True)
class FieldEvidence:
    field: str
    index: int
    candidates: tuple
    values: tuple
    rejected: tuple
    issues: tuple
    # Field-level outputs of optional heads from the same pass, keyed by name.
    field_signals: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def occurrences_for(self, value):
        for item in self.values:
            if item.value == value:
                return item.occurrences
        return ()


@dataclass(frozen=True)
class Trace:
    """Everything one request produced. Never shared between requests."""

    fields: MappingProxyType
    input_tokens: int | None
    backbone_calls: int | None
    timings: MappingProxyType

    def field(self, name, index=0):
        return self.fields[(index, name)]


class EvidenceCollector:
    """Receives processing events from the production pipeline for one request."""

    def __init__(self):
        self.accepted_events = []
        self.rejected_events = []

    def accepted(self, task, index, key, value, span):
        self.accepted_events.append((task, index, key, value, dict(span)))

    def rejected(self, task, index, key, span, code):
        self.rejected_events.append((task, index, key, None if span is None else dict(span), code))

    def build(self, request, records, signals, result_issues, *, task="result", kinds=None,
              input_tokens=None, backbone_calls=None, timings=None):
        """Group accepted occurrences by the same type-aware equality as the final values."""
        from .contact_normalization import comparison_key

        kinds = kinds or {}
        anchors = [None] * len(records)
        if request.anchor:
            anchors = [(r[request.anchor][0]["start"], r[request.anchor][0]["end"]) for r in records]

        def occurrence(index, key, span):
            anchor = anchors[index] if index < len(anchors) else None
            found = signals.get((anchor, key, span["start"], span["end"]), {})
            return Occurrence(span["start"], span["end"], span["text"], MappingProxyType(dict(found)))

        fields = {}
        for index, record in enumerate(records or [dict.fromkeys(request.fields, [])]):
            for key in request.fields:
                grouped = {}
                for event_task, event_index, event_key, value, span in self.accepted_events:
                    if (event_task, event_index, event_key) != (task, index, key):
                        continue
                    group = grouped.setdefault(comparison_key(value, kinds.get(key)), [value, []])
                    item = occurrence(index, key, span)
                    if all((o.start, o.end) != (item.start, item.end) for o in group[1]):
                        group[1].append(item)
                rejected = tuple(
                    Rejection(None if span is None else occurrence(index, key, span), code)
                    for event_task, event_index, event_key, span, code in self.rejected_events
                    if (event_task, event_index, event_key) == (task, index, key))
                issues = tuple(i["code"] for i in result_issues
                               if i.get("field") == key and i.get("index", 0) == index)
                fields[(index, key)] = FieldEvidence(
                    field=key, index=index,
                    candidates=tuple(occurrence(index, key, s) for s in record.get(key, [])),
                    values=tuple(ValueEvidence(v, tuple(o)) for v, o in grouped.values()),
                    rejected=rejected, issues=issues,
                    # Field-level signals are per request field, keyed without an anchor.
                    field_signals=MappingProxyType(dict(signals.get((None, key, None, None), {}))))
        return Trace(fields=MappingProxyType(fields), input_tokens=input_tokens,
                     backbone_calls=backbone_calls, timings=MappingProxyType(dict(timings or {})))
