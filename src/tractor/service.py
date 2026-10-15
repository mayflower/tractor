"""Deterministic response construction from native source candidates."""

import time

from .normalize import normalize
from .types import Quantity


def valid_span(text, span):
    start, end = span.get("start"), span.get("end")
    return (
        type(start) is int
        and type(end) is int
        and 0 <= start < end <= len(text)
        and text[start:end] == span.get("text")
    )


def resolve(request, records):
    issues = []
    empty = dict.fromkeys(request.fields)
    if request.anchor:
        if any(
            not r[request.anchor]
            or len(r[request.anchor]) != 1
            or not valid_span(request.text, r[request.anchor][0])
            for r in records
        ):
            raise RuntimeError("Native record returned invalid anchor")
        records = sorted(records, key=lambda r: r[request.anchor][0]["start"])
        if not records or (not request.many and len(records) != 1):
            result = {
                "data": [] if request.many else empty,
                "issues": [
                    {"field": None, "code": "not_found" if not records else "ambiguous_record"}
                ],
            }
            if request.include_details:
                result["details"] = [] if request.many else empty.copy()
            return result
    output, details = [], []
    for index, record in enumerate(records):
        values, sources = {}, {}
        for name, field in request.fields.items():
            spans = record[name]
            issue = None
            value = source = None
            if any(not valid_span(request.text, span) for span in spans):
                issue = "invalid_span"
            else:
                spans = list({(s["start"], s["end"]): s for s in spans}.values())
                # Keep maximal overlapping spans, preserving qualifiers and repeated occurrences.
                spans = [
                    s
                    for s in spans
                    if not any(
                        t["start"] <= s["start"] and s["end"] <= t["end"] and t != s for t in spans
                    )
                ]
                if not spans:
                    issue = "not_found"
                elif len(spans) > 1:
                    issue = "ambiguous_value"
                else:
                    span = spans[0]
                    source = {k: span[k] for k in ("text", "start", "end")}
                    if "confidence" in span:
                        source["extraction_score"] = span["confidence"]
                    if isinstance(field, Quantity):
                        normalized = normalize(request.text, span, field.unit, request.locale)
                        value, issue = normalized.value, normalized.issue
                        source.update(source_unit=normalized.source_unit, target_unit=field.unit)
                    else:
                        value = span["text"]
            if issue:
                entry = {"field": name, "code": issue}
                if request.many:
                    entry["index"] = index
                issues.append(entry)
            values[name], sources[name] = value, source
        output.append(values)
        details.append(sources)
    result = {"data": output if request.many else output[0], "issues": issues}
    if request.include_details:
        result["details"] = details if request.many else details[0]
    return result


def extract(extractor, request):
    candidates = extractor.candidates(request)
    started = time.perf_counter()
    result = resolve(request, candidates)
    if hasattr(extractor, "last_timings"):
        extractor.last_timings["normalize"] = (time.perf_counter() - started) * 1000
    return result
