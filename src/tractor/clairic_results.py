"""Project native results and their evidence onto the Clairic model payload.

The projection never creates evidence or probabilities. Absent optional
probabilities are omitted (not null, not a placeholder). Normalization errors
and native scalar ambiguity become explicit compatibility errors; a plain
`value: null` in this subset only means that no extraction value was accepted
without a specific processing error. It is not a probabilistic null decision
and not proof that the source lacks the value.
"""

import re
from decimal import Context, Decimal, Inexact, localcontext

from .capabilities import ClairicError, ProcessingFailed
from .clairic_compute import evaluate, round_result
from .uncertainty import NullDecisionInconsistent, null_decision

PROJECTION_VERSION = "clairic-projection-v1"
SECONDS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800}
ISO_DURATION = re.compile(r"P(?:(?P<Y>[0-9.]+)Y)?(?:(?P<M>[0-9.]+)M)?(?:(?P<W>[0-9.]+)W)?(?:(?P<D>[0-9.]+)D)?"
                          r"(?:T(?:(?P<H>[0-9.]+)H)?(?:(?P<TM>[0-9.]+)M)?(?:(?P<S>[0-9.]+)S)?)?")


class ProjectionError(ClairicError):
    status = 500
    code = "tractor.projection_inconsistency"


class UsageUnavailable(ClairicError):
    status = 500
    code = "tractor.usage_unavailable"


def decimals_allowed(spec):
    return 0 if spec.integer else spec.decimals


def number(name, value, spec):
    """Stated values are never rounded; extra precision is an explicit error."""
    value = Decimal(value)
    allowed = decimals_allowed(spec)
    exponent = value.normalize().as_tuple().exponent
    places = max(0, -exponent) if isinstance(exponent, int) else 0
    if allowed is not None and places > allowed:
        raise ProcessingFailed(f"Stated value has more than {allowed} decimals", field=name,
                               details=[{"code": "precision_exceeded"}])
    return int(value) if value == value.to_integral_value() else value.normalize()


def duration(name, iso, spec):
    """ISO duration to a number in the requested unit, without calendar guesses."""
    match = ISO_DURATION.fullmatch(iso)
    if not match:
        raise ProcessingFailed("Unsupported duration form", field=name, details=[{"code": "invalid_duration"}])
    parts = {k: Decimal(v) for k, v in match.groupdict().items() if v is not None}
    unit = spec.unit or "seconds"
    calendar_parts = {k for k in parts if k in {"Y", "M"}}
    with localcontext(Context(prec=50, traps=[Inexact])):
        try:
            if unit in {"months", "years"}:
                if set(parts) - {"Y", "M"}:
                    raise ProcessingFailed("Fixed-length parts cannot be expressed in calendar units",
                                           field=name, details=[{"code": "mixed_calendar_duration"}])
                months = parts.get("Y", Decimal(0)) * 12 + parts.get("M", Decimal(0))
                result = months if unit == "months" else months / 12
            else:
                if calendar_parts:
                    raise ProcessingFailed("Months and years have no fixed length in seconds",
                                           field=name, details=[{"code": "calendar_duration"}])
                seconds = (parts.get("W", 0) * 604800 + parts.get("D", 0) * 86400 + parts.get("H", 0) * 3600
                           + parts.get("TM", 0) * 60 + parts.get("S", 0))
                result = Decimal(seconds) / SECONDS[unit]
        except Inexact:
            raise ProcessingFailed("Duration is not exactly representable in the requested unit",
                                   field=name, details=[{"code": "precision_exceeded"}]) from None
    return number(name, result, spec)


def scalar_value(name, kind, value, spec):
    if kind == "number":
        return number(name, value, spec)
    if kind == "duration":
        return duration(name, value, spec)
    return value


def first_occurrence(evidence, value):
    occurrences = evidence.occurrences_for(value)
    return min(occurrences, key=lambda o: (o.start, o.end)) if occurrences else None


def project_source_field(plan_field, result, trace, state):
    name, spec, key = plan_field.name, plan_field.spec, plan_field.internal_id
    evidence = trace.field(key)
    issues = [code for code in evidence.issues if code != "not_found"]
    if issues:
        raise ProcessingFailed("Present evidence could not be resolved to a valid answer", field=name,
                               details=[{"code": code} for code in issues])
    found = result["data"][key]
    if spec.type == "list":
        values = []
        for item in found:
            if spec.items == "text" and not evidence.occurrences_for(item):
                raise ProjectionError("List value lacks source evidence", field=name)
            values.append(scalar_value(name, spec.items, item, spec))
        spans = sorted({o.span[0]: o for v in found for o in evidence.occurrences_for(v)}.values(),
                       key=lambda o: (o.start, o.end))
        if spec.items == "text" and any(state[o.start:o.end] not in found for o in spans):
            raise ProjectionError("Text list span does not match a returned value", field=name)
        return {"type": "list", "value": values, "spans": [o.span for o in spans]}, found
    if isinstance(found, list):
        # Only reached without an ambiguity issue if the native contract changed.
        raise ProjectionError("Scalar result is a list", field=name)
    answer = {"type": spec.type, "value": None}
    if found is None:
        return answer, None
    occurrence = first_occurrence(evidence, found)
    if occurrence is None:
        raise ProjectionError("Value lacks source evidence", field=name)
    if spec.type == "text" and state[occurrence.start:occurrence.end] != found:
        # A transformed text would not be an exact source span.
        raise ProcessingFailed("Text value is not an exact source span", field=name,
                               details=[{"code": "text_not_source_span"}])
    answer["value"] = scalar_value(name, spec.type, found, spec)
    answer["span"] = occurrence.span
    return answer, answer["value"]


def project_compute_field(plan_field, result):
    operation, spec = plan_field.compute, plan_field.spec
    value = evaluate(plan_field.name, operation, result["data"], result["issues"])
    if spec.type == "number":
        value = round_result(plan_field.name, value, decimals_allowed(spec))
        value = number(plan_field.name, value, spec)
    return {"type": spec.type, "value": value}, value


def project(plan, result, trace, *, uncertainty=None, messages=None):
    """The Clairic payload; field-level notices are appended to `messages`."""
    request = plan.request
    messages = [] if messages is None else messages
    if trace.input_tokens is None:
        raise UsageUnavailable("The backend did not report its input token count")
    scores = uncertainty.score(plan, trace) if uncertainty is not None else {}
    answers = {}
    for plan_field in plan.fields:
        if plan_field.operation == "supported_compute":
            answer, value = project_compute_field(plan_field, result)
        else:
            answer, value = project_source_field(plan_field, result, trace, request.state)
        score = scores.get(plan_field.name)
        if score is not None and plan_field.spec.type != "list":
            if score.p_null is not None:
                try:
                    kept, guess = null_decision(value, score.p_null, plan_field.spec.threshold)
                except NullDecisionInconsistent as exc:
                    # Answerability and span selection disagree for this field only:
                    # keep its extraction result and claim no probability for it.
                    messages.append({"code": "tractor.uncertainty_withheld", "message": str(exc),
                                     "field": plan_field.name})
                    answers[plan_field.name] = answer
                    continue
                answer["p_null"] = score.p_null
                if guess is not None:
                    answer["guess"] = guess
                if kept is None:
                    answer["value"] = None
                    answer.pop("span", None)  # A null answer carries no source evidence.
        if score is not None and score.confidence is not None:
            answer["confidence"] = score.confidence
        answers[plan_field.name] = answer
    payload = {"model": "clairic", "answers": answers, "truncated": False,
               # One local extraction pass; no tokens are generated autoregressively.
               "usage": {"input_tokens": trace.input_tokens, "output_tokens": 0, "passes": 1}}
    if request.options.return_state_text:
        payload["state_text"] = request.state
    return payload
