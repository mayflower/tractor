"""Full production pipeline Dev contracts and final-value metrics; no span gate."""

from collections import Counter
from decimal import Decimal, InvalidOperation

from .document_pipeline import DocumentRequest

ADDRESS_COMPONENTS = {
    "address__post_town": "city",
    "address__postcode": "postcode",
    "address__street_line": "street",
}


def request_for(episode):
    # Dataset-level contract, independent of gold, predictions and checkpoint.
    # Kleister Charity is the explicit UK-address corpus. No country guessing.
    processing = {}
    if episode.source == "kleister-charity":
        for task in episode.tasks:
            fields = {
                key: {
                    "kind": "address",
                    "country": "GB",
                    "component": component,
                    "group": "registered_office",
                }
                for key, component in ADDRESS_COMPONENTS.items()
                if key in task.fields
            }
            if fields:
                processing[task.id] = fields
    for task, rules in episode.processing.items():
        processing.setdefault(task, {}).update(rules)
    return DocumentRequest(
        text=episode.text, tasks=episode.tasks, processing=processing, locale=episode.locale, include_details=True
    )


def typed_requests_for(episode):
    """Adapt corpus contracts to the sole public API, without reading gold."""
    from .types import ExtractRequest
    legacy = request_for(episode)
    requests = {}
    for task in episode.tasks:
        if task.kind not in {"entities", "records"}:
            raise ValueError("Only extraction tasks are supported")
        fields = {}
        for key, description in task.fields.items():
            rule = legacy.processing.get(task.id, {}).get(key)
            options = rule.model_dump(exclude_defaults=True) if rule else {}
            kind = options.pop("kind", "literal")
            fields[key] = dict(type="text" if kind == "literal" else kind, description=description,
                               **options)
            fields[key].setdefault("cardinality", rule.cardinality if rule else
                                   {"zero_or_more": "unique_list"}.get(
                                       task.cardinalities.get(key), task.cardinalities.get(key, "unique_list")))
        requests[task.id] = ExtractRequest(text=episode.text, fields=fields, locale=episode.locale,
                                          many=task.kind == "records", anchor=task.anchor)
    return requests


def infer(extractor, episode):
    from .service import extract
    response = {"data": {}, "details": {}, "issues": []}
    for task_id, request in typed_requests_for(episode).items():
        result = extract(extractor, request, _diagnostics=True)
        # Metrics store value sets, while the API exposes declared scalars/lists.
        rows = result["data"] if request.many else [result["data"]]
        rows = [{k: v if isinstance(v, list) else [] if v is None else [v]
                 for k, v in row.items()} for row in rows]
        response["data"][task_id] = rows if request.many else rows[0]
        detail_rows = result["details"] if request.many else [result["details"]]
        detail_rows = [{k: v if isinstance(v, list) else [] if v is None else v.get("evidence", [v])
                        for k, v in row.items()} for row in detail_rows]
        response["details"][task_id] = detail_rows if request.many else detail_rows[0]
        response["issues"].extend(dict(issue, task=task_id, index=issue.get("index", 0))
                                  for issue in result["issues"])
    return response


def expected(episode, task):
    # Expected components come from independently annotated component spans.
    # Never run the production address parser on gold to manufacture success.
    rows = []
    for row in episode.gold[task.id].records:
        values = {}
        for key, answer in row.items():
            if answer.final_annotation_missing:
                values[key] = ('unknown_annotation', [set()])
                continue
            if answer.final_values is not None or answer.final_alternatives:
                alternatives = answer.final_alternatives or [answer.final_values]
                values[key] = (answer.status, [set(v) for v in alternatives])
                continue
            variants = answer.answer_alternatives or [answer.items]
            alternatives = []
            for spans in variants:
                texts = [s.text for s in spans]
                if episode.source == "kleister-charity" and key == "address__postcode":
                    texts = ["".join(t.split()).upper() for t in texts]
                    texts = [t[:-3] + " " + t[-3:] for t in texts]
                alternatives.append(set(texts))
            values[key] = (answer.status, alternatives)
        rows.append(values)
    return rows


def value_key(value):
    # Numeric final annotations use canonical decimal strings, without rounding.
    if isinstance(value, Decimal):
        if value == 0:
            return '0'
        return format(value, 'f').rstrip('0').rstrip('.') if '.' in format(value, 'f') else format(value, 'f')
    return str(value)


def field_value_key(value, kind):
    """Final numeric annotations compare as numbers, never as source text.

    No locale guessing or production parsing of gold: final annotations already
    express the requested numeric value. Identifiers and text retain their zeros.
    """
    from .contact_normalization import comparison_key
    if kind in {'quantity', 'number', 'money', 'percent'} and isinstance(value, str):
        try:
            parsed = Decimal(value)
            if parsed.is_finite() and abs(parsed.as_tuple().exponent) <= 1000:
                value = parsed
        except InvalidOperation:
            pass
    return comparison_key(value_key(value), kind)



def field_observation(episode, task, record, key, actual, response, predicted_index, target=None):
    """One final API outcome comparator shared by gates, diagnostics and attribution."""
    status, alternatives = target if target is not None else expected(episode, task)[record][key]
    answer = episode.gold[task.id].records[record][key]
    rule = request_for(episode).processing.get(task.id, {}).get(key)
    kind = rule.kind if rule else None
    def canonical(v):
        return field_value_key(v, kind)
    found = {canonical(v) for v in actual.get(key, [])}
    choices = [{canonical(v) for v in values} for values in alternatives]
    gold = min(choices, key=lambda values: (len(values ^ found), -len(values & found)))
    codes = {issue['code'] for issue in response.get('issues', [])
             if issue.get('task') == task.id and issue.get('field') == key
             and issue.get('index', 0) == predicted_index} if predicted_index is not None else set()
    required = (rule.cardinality if rule else task.cardinalities.get(key)) in {'required_one', 'one_or_more'}
    expected_codes = set(answer.expected_issues) if answer.expected_issues is not None else (
        {'missing_required_value' if required else 'not_found'} if status == 'absent' else set())
    known = status not in {'unknown_annotation', 'ambiguous'}
    issues_correct = codes == expected_codes
    values_correct = found == gold
    exact = known and values_correct and issues_correct
    rejection = known and status == 'present' and bool(expected_codes) and not any(choices)
    positive = status == 'present' and not rejection
    counts = Counter()
    if not known:
        counts['unscored_fields'] = 1
    else:
        counts.update(fields=1, exact_fields=int(exact),
            tp=len(found & gold), fp=len(found-gold), fn=len(gold-found),
            field_tp=int(positive and exact), field_fp=int(bool(found) and not exact),
            field_fn=int(positive and not exact), answered=int(bool(found)),
            answerable=int(positive), covered=int(positive and bool(found)),
            absent=int(status == 'absent'), false_absent=int(status == 'absent' and bool(found)),
            issue_fields=1, issue_correct=int(issues_correct), issue_mismatches=int(not issues_correct),
            expected_rejections=int(rejection), correct_rejections=int(rejection and exact),
            rejected_with_value=int(rejection and bool(found)))
    return dict(counts=counts, known=known, exact=exact, values_correct=values_correct,
                issues_correct=issues_correct, rejection=rejection, found=found, gold=gold,
                expected_issues=expected_codes, actual_issues=codes)


def task_counts(episode, task, response, *, with_matches=False):
    from scipy.optimize import linear_sum_assignment

    target = expected(episode, task)
    rules = request_for(episode).processing.get(task.id, {})
    def key_for(value, field):
        rule = rules.get(field)
        return field_value_key(value, rule.kind if rule else None)
    value = response.get("data", {}).get(task.id, {} if task.kind == "entities" else [])
    predicted = [value] if task.kind == "entities" else value

    def pair(record, actual, predicted_index):
        counts = Counter()
        for key, target_field in target[record].items():
            counts.update(field_observation(episode, task, record, key, actual, response,
                                           predicted_index, target_field)['counts'])
        return counts

    costs = []
    for i, row in enumerate(target):
        line = []
        for j, actual in enumerate(predicted):
            c = pair(i, actual, j)
            cost = c["fp"] + c["fn"] + c["issue_mismatches"]
            if task.anchor and {key_for(v, task.anchor) for v in actual.get(task.anchor, [])} not in [
                {key_for(v, task.anchor) for v in alternative} for alternative in row[task.anchor][1]]:
                cost += 1000000
            line.append(cost)
        costs.append(line)
    matches = []
    if target and predicted:
        ii, jj = linear_sum_assignment(costs)
        matches = [(int(i), int(j)) for i, j in zip(ii, jj, strict=True) if costs[i][j] < 1000000]
    counts = Counter(gold_records=len(target), predicted_records=len(predicted))
    for i, j in matches:
        c = pair(i, predicted[j], j)
        counts.update(c)
        counts["exact_records"] += int(
            episode.gold[task.id].complete
            and not c["unscored_fields"]
            and not c["fp"]
            and not c["fn"]
            and not c["issue_mismatches"]
        )
    for i in set(range(len(target))) - {i for i, _ in matches}:
        counts.update(pair(i, {}, None))
    for j in set(range(len(predicted))) - {j for _, j in matches}:
        values = predicted[j]
        counts.update(
            fp=sum(len(set(v)) for v in values.values()),
            field_fp=sum(bool(v) for v in values.values()),
            fields=len(task.fields),
            answered=sum(bool(v) for v in values.values()),
            extra_records=1,
        )
    return (counts, matches) if with_matches else counts


def episode_counts(episode, response, error=None):
    counts = Counter(requests=1, errors=int(error is not None))
    if error:
        response = {}
    complete = True
    for task in episode.tasks:
        counts.update(task_counts(episode, task, response))
        complete &= episode.gold[task.id].complete and all(
            a.status not in {"ambiguous", "unknown_annotation"} and not a.final_annotation_missing
            for row in episode.gold[task.id].records
            for a in row.values()
        )
    counts["exact_requests"] = int(
        not error
        and complete
        and not counts["fp"]
        and not counts["fn"]
        and counts["gold_records"] == counts["predicted_records"]
        and not counts["issue_mismatches"]
        and not structural_issue_errors(episode, response)
    )
    return counts


def structural_issue_errors(episode, response):
    """Reject unscoped/out-of-range issues; recognize a correctly empty record list."""
    tasks = {task.id: task for task in episode.tasks}
    errors = []
    for issue in response.get('issues', []):
        task = tasks.get(issue.get('task'))
        if task is None:
            errors.append(issue)
            continue
        actual = response.get('data', {}).get(task.id, [] if task.kind == 'records' else {})
        if (task.kind == 'records' and not episode.gold[task.id].records and actual == []
                and issue.get('field') is None and issue.get('code') == 'not_found'):
            continue
        size = len(actual) if task.kind == 'records' else 1
        index = issue.get('index', 0)
        if issue.get('field') not in task.fields or type(index) is not int or not 0 <= index < size:
            errors.append(issue)
    return errors


def evaluate_pipeline(episodes, predictions, policy):
    from .document_quality import evaluate

    report = evaluate(
        episodes, predictions, policy, score_fn=episode_counts, task_score_fn=task_counts
    )
    from .pipeline_diagnostics import diagnostics
    report.update(diagnostics(episodes, predictions, policy.get('minimum_field_support', 30), policy['minimum_exclusive']))
    report['domain_shape_passed'] = report['passed']
    report['field_gates_passed'] = bool(report['field_checks']) and all(v['status'] == 'pass' for v in report['field_checks'].values())
    report['rejection_gates_passed'] = all(v['status'] == 'pass' for v in report['rejection_checks'].values())
    report['passed'] = report['passed'] and report['field_gates_passed'] and report['rejection_gates_passed']
    report["evaluation_contract"] = (
        "production-extraction-pipeline-v3: final typed values, exact field/record issue sets and distinct records; evidence retained separately"
    )
    report["qualification"] = (
        "Full production document pipeline. Explicit per-field contracts, independent final annotations, source evidence preserved. Legacy Kleister GB address adapter retained. Fields without processing remain literal unique values. Missing real evidence blocks release. No latency measurement."
    )
    return report
