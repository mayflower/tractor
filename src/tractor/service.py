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


def resolve(request, records, *, _diagnostics=False, _collector=None):
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
            if _diagnostics:
                result["details"] = [] if request.many else empty.copy()
            return result
    # One typed contract compiles to the private evidence-processing engine.
    from .document_pipeline import DocumentRequest, resolve_document
    from .episodes import Task

    task = Task(id="result", kind="records" if request.anchor else "entities",
                description="Requested fields", fields={k: f.description for k, f in request.fields.items()},
                anchor=request.anchor)
    processing = {k: f.processing_rule() for k, f in request.fields.items()}
    internal = DocumentRequest(text=request.text, tasks=[task], processing={"result": processing},
                               locale=request.locale, include_details=True)
    # Preserve the established maximal-overlap policy before normalization.
    cleaned = []
    for row in records:
        result = {}
        for key in request.fields:
            spans = row.get(key, [])
            if all(valid_span(request.text, span) for span in spans):
                spans = list({(s["start"], s["end"]): s for s in spans}.values())
                spans = [s for s in spans if not any(
                    t["start"] <= s["start"] and s["end"] <= t["end"] and t != s for t in spans)]
            result[key] = spans
        cleaned.append(result)
    raw = {"result": cleaned} if request.anchor else {"entities": cleaned[0] if cleaned else {}}
    resolved = resolve_document(internal, raw, _collector)
    rows = resolved["data"]["result"] if request.anchor else [resolved["data"]["result"]]
    evidence = resolved["details"]["result"] if request.anchor else [resolved["details"]["result"]]
    output, details = [], []
    for index, (row, evidence_row) in enumerate(zip(rows, evidence, strict=True)):
        values, sources = {}, {}
        for key, field in request.fields.items():
            multiple = field.cardinality in {"unique_list", "one_or_more"}
            found = row[key]
            # A scalar with several distinct candidates returns all of them with `ambiguous_value`.
            values[key] = found if multiple or len(found) > 1 else found[0] if found else None
            spans = evidence_row[key]
            source = spans if multiple else spans[0].copy() if spans else None
            if not multiple and len(spans) > 1:
                source["evidence"] = spans
            if isinstance(field, Quantity) and source and not multiple:
                parsed = normalize(request.text, spans[0], field.unit, field.locale or request.locale)
                source.update(source_unit=parsed.source_unit, target_unit=field.unit)
            sources[key] = source
            field_issues = [i for i in resolved["issues"] if i["index"] == index and i["field"] == key]
            if not found and not field_issues:
                field_issues = [{"code": "not_found"}]
            for issue in field_issues:
                entry = {"field": key, "code": issue["code"]}
                if request.many:
                    entry["index"] = index
                issues.append(entry)
        output.append(values)
        details.append(sources)
    result = {"data": output if request.many else output[0], "issues": issues}
    if _diagnostics:
        result["details"] = details if request.many else details[0]
    return result


def extract(extractor, request, *, _diagnostics=False):
    candidates = extractor.candidates(request)
    started = time.perf_counter()
    result = resolve(request, candidates, _diagnostics=_diagnostics)
    if hasattr(extractor, "last_timings"):
        extractor.last_timings["normalize"] = (time.perf_counter() - started) * 1000
    return result


def candidates_with_signals(extractor, request):
    """One extraction pass; backends without scores contribute no signals."""
    if hasattr(extractor, "candidates_with_signals"):
        return extractor.candidates_with_signals(request)
    return extractor.candidates(request), {}


def extract_traced(extractor, request):
    """The native result plus its request-scoped evidence trace.

    Same candidates, same resolution, same output as `extract`; the trace only
    observes. Token counts come from this request's own pass, read while the
    caller still holds the inference worker.
    """
    from .evidence import EvidenceCollector

    calls = getattr(extractor, "encoder_calls", None)
    records, signals = candidates_with_signals(extractor, request)
    passes = None if calls is None else extractor.encoder_calls - calls
    started = time.perf_counter()
    collector = EvidenceCollector()
    result = resolve(request, records, _collector=collector)
    timings = dict(getattr(extractor, "last_timings", {}))
    timings["normalize"] = (time.perf_counter() - started) * 1000
    if request.anchor:
        records = sorted(records, key=lambda r: r[request.anchor][0]["start"])
    kinds = {key: field.processing_rule().kind for key, field in request.fields.items()}
    tokens = getattr(extractor, "last_tokens", None)
    trace = collector.build(request, records, signals, result["issues"], kinds=kinds,
                            input_tokens=tokens if isinstance(tokens, int) and tokens > 0 else None,
                            backbone_calls=passes, timings=timings)
    return result, trace
