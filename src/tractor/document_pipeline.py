"""Production document pipeline: validated evidence -> deterministic final values."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .address_normalization import address_parts, joined_address_spans
from .episodes import Task
from .field_contracts import Processing
from .field_normalization import Unresolved, normalize_value


class DocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1, max_length=32)
    processing: dict[str, dict[str, Processing]] = Field(default_factory=dict)
    locale: Literal["de-DE", "en-US", "en-GB"] | None = None
    include_details: bool = False

    @model_validator(mode="after")
    def valid(self):
        tasks = {t.id: t for t in self.tasks}
        if len(tasks) != len(self.tasks) or not self.text.strip():
            raise ValueError("Nonempty text and unique task IDs required")
        if any(t.kind not in {"entities", "records"} for t in self.tasks):
            raise ValueError("Only literal entity and record extraction is supported")
        if sum(len(t.fields) for t in self.tasks) > 128:
            raise ValueError("Too many fields")
        entity_keys = [k for t in self.tasks if t.kind == "entities" for k in t.fields]
        if len(entity_keys) != len(set(entity_keys)):
            raise ValueError("Entity field IDs must be unique")
        for task, fields in self.processing.items():
            if task not in tasks or not fields.keys() <= tasks[task].fields.keys():
                raise ValueError("Processing must refer to declared fields")
        return self


def unique(values, kind=None):
    from .contact_normalization import comparison_key
    result = {}
    for value in values:
        result.setdefault(comparison_key(value, kind), value)
    return list(result.values())


def resolve_document(request, raw, collector=None):
    """No gold, source identity or hidden document context enters processing.

    An optional request-scoped collector observes accepted and rejected
    occurrences; it never changes the selected values or issues.
    """
    from .normalize import normalize
    from .service import valid_span

    data, details, issues = {}, {}, []
    for task in request.tasks:
        rows = [raw.get("entities", {})] if task.kind == "entities" else raw.get(task.id, [])
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError("Invalid native task result")
        results = []
        evidence = []
        rules = request.processing.get(task.id, {})
        for index, row in enumerate(rows):
            spans = {}
            invalid = set()
            for key in task.fields:
                values = row.get(key, [])
                values = [] if values is None else values if isinstance(values, list) else [values]
                if any(not isinstance(s, dict) or not valid_span(request.text, s) for s in values):
                    invalid.add(key)
                    values = []
                    if collector:
                        collector.rejected(task.id, index, key, None, "invalid_span")
                    issues.append(
                        {"task": task.id, "index": index, "field": key, "code": "invalid_span"}
                    )
                spans[key] = values
            if task.kind == "records" and (task.anchor in invalid or len(spans[task.anchor]) != 1):
                issues.append(
                    {
                        "task": task.id,
                        "index": index,
                        "field": task.anchor,
                        "code": "invalid_anchor",
                    }
                )
                continue
            result = {}
            sources = {}
            for key in task.fields:
                rule = rules.get(key)
                value = []
                # Preserve valid source evidence even when normalization fails.
                used = list(spans[key])
                if key in invalid:
                    result[key] = []
                    sources[key] = []
                    continue
                if rule and rule.kind == "address" and rule.component not in (None, "full"):
                    # Sharing is explicitly scoped to one address group AND one
                    # record. Never search the whole document for a missing value.
                    for other, other_rule in rules.items():
                        if other_rule.kind != "address" or (
                            other_rule.group,
                            other_rule.country,
                        ) != (rule.group, rule.country):
                            continue
                        for span in spans[other]:
                            component = address_parts(span["text"], other_rule.component, other_rule.country).get(
                                rule.component
                            )
                            if component is not None:
                                value.append(component)
                                used.append(span)
                                if collector:
                                    collector.accepted(task.id, index, key, component, span)
                    if len(unique(value, rule.kind if rule else None)) > 1:
                        value = []
                        issues.append(
                            {
                                "task": task.id,
                                "index": index,
                                "field": key,
                                "code": "ambiguous_address",
                            }
                        )
                else:
                    selected = (joined_address_spans(request.text, spans[key], rule.country)
                                if rule and rule.kind == "address" else spans[key])
                    for span in selected:
                        if rule and rule.include_adjacent_year and not re.search(r'\([0-9]{4}\)$', span['text'].rstrip()):
                            # Explicit displayed-title contract: extend only this
                            # validated evidence span into an attached year. Never
                            # search elsewhere in the document or cross a newline.
                            suffix = re.match(r'[ \t]*\(([0-9]{4})\)', request.text[span['end']:])
                            if suffix and int(suffix[1]) > 0:
                                end = span['end'] + suffix.end()
                                span = dict(span, end=end, text=request.text[span['start']:end])
                        if rule and rule.kind == "quantity":
                            parsed = normalize(request.text, span, rule.unit, rule.locale or request.locale)
                            if parsed.issue:
                                issues.append(
                                    {
                                        "task": task.id,
                                        "index": index,
                                        "field": key,
                                        "code": parsed.issue,
                                    }
                                )
                                if collector:
                                    collector.rejected(task.id, index, key, span, parsed.issue)
                                continue
                            produced = [parsed.value]
                        elif rule:
                            try:
                                produced = normalize_value(span["text"], rule, request.locale)
                            except Unresolved as exc:
                                issues.append({"task": task.id, "index": index, "field": key, "code": str(exc)})
                                if collector:
                                    collector.rejected(task.id, index, key, span, str(exc))
                                continue
                        else:
                            produced = [span["text"]]
                        value.extend(produced)
                        if collector:
                            for item in produced:
                                collector.accepted(task.id, index, key, item, span)
                        used.append(span)
                cardinality = rule.cardinality if rule else task.cardinalities.get(key, "unique_list")
                if cardinality in {"required_one", "optional_one"} and len(unique(value, rule.kind if rule else None)) > 1:
                    # Competing candidates stay in the result, in source order; the issue marks them unresolved.
                    issues.append({"task": task.id, "index": index, "field": key, "code": "ambiguous_value"})
                if cardinality in {"required_one", "one_or_more"} and not value:
                    issues.append({"task": task.id, "index": index, "field": key, "code": "missing_required_value"})
                result[key] = unique(value, rule.kind if rule else None)
                sources[key] = list(
                    {
                        (s["start"], s["end"]): {k: s[k] for k in ("text", "start", "end")}
                        for s in used
                    }.values()
                )
                if (
                    rule
                    and spans[key]
                    and not value
                    and not any(
                        i["task"] == task.id and i["index"] == index and i["field"] == key
                        for i in issues
                    )
                ):
                    issues.append(
                        {"task": task.id, "index": index, "field": key, "code": "unresolved_value"}
                    )
            results.append(result)
            evidence.append(sources)
        data[task.id] = results[0] if task.kind == "entities" else results
        details[task.id] = evidence[0] if task.kind == "entities" else evidence
    response = {"data": data, "issues": issues}
    if request.include_details:
        response["details"] = details
    return response
