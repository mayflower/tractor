"""Occurrence-aware task scoring with one-to-one record matching and explicit failures."""

from collections import Counter


def spans(value):
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [(v["start"], v["end"], v["text"]) for v in values]


def choices(answer):
    if answer.label is not None:
        return [[("label", answer.label)]]
    variants = answer.answer_alternatives or [answer.items]
    return [[(s.start, s.end, s.text) for s in variant] for variant in variants]


def compare_field(answer, predicted):
    alternatives = choices(answer)
    # OR interpretations of one answer; each interpretation retains multiplicity.
    def compare(expected):
        a, b = Counter(expected), Counter(predicted)
        tp = sum((a & b).values())
        return tp, sum((b - a).values()), sum((a - b).values())
    scored = [compare(option) for option in alternatives]
    return min(scored, key=lambda c: (c[1] + c[2], -c[0]))


def prediction_rows(task, raw):
    if task.kind == "classification":
        result = raw.get(task.id)
        label = result.get("label") if isinstance(result, dict) else result
        return [{"label": [] if label is None else [("label", label)]}]
    if task.kind == "entities":
        rows = [raw.get("entities", {})]
    elif task.kind == "relations":
        rows = raw.get("relation_extraction", {}).get(task.id, [])
    else:
        rows = raw.get(task.id, [])
    return [{key: spans(row.get(key)) for key in task.fields} for row in rows]


def score_task(task, gold, predicted, with_matches=False):
    from scipy.optimize import linear_sum_assignment

    def pair(expected, actual):
        count = Counter()
        for key, answer in expected.items():
            if answer.status in {"unknown_annotation", "ambiguous"}:
                count["unscored_fields"] += 1
                continue
            values = actual.get(key, [])
            tp, fp, fn = compare_field(answer, values)
            correct = fp == fn == 0
            present = answer.status == "present"
            count.update(tp=tp, fp=fp, fn=fn, fields=1, exact_fields=int(fp == fn == 0),
                         field_tp=int(present and correct),
                         field_fp=int(bool(values) and not correct),
                         field_fn=int(present and not correct),
                         answered=int(bool(values)), answerable=int(answer.status == "present"),
                         covered=int(answer.status == "present" and bool(values)),
                         absent=int(answer.status == "absent"),
                         false_absent=int(answer.status == "absent" and bool(values)))
        return count

    expected = gold.records
    costs = []
    for row in expected:
        line = []
        for actual in predicted:
            c = pair(row, actual)
            cost = c["fp"] + c["fn"]
            if task.anchor and not any(Counter(option) == Counter(actual.get(task.anchor, []))
                                       for option in choices(row[task.anchor])):
                cost += 1000000
            line.append(cost)
        costs.append(line)
    matches = []
    if expected and predicted:
        ii, jj = linear_sum_assignment(costs)
        matches = [(int(i), int(j)) for i, j in zip(ii, jj, strict=True)
                   if costs[i][j] < 1000000]
    counts = Counter(gold_records=len(expected), predicted_records=len(predicted))
    for i, j in matches:
        c = pair(expected[i], predicted[j])
        counts.update(c)
        counts["exact_records"] += int(gold.complete and not c["unscored_fields"]
                                       and c["fp"] == c["fn"] == 0)
    for i in set(range(len(expected))) - {i for i, _ in matches}:
        counts.update(pair(expected[i], {}))
    for j in set(range(len(predicted))) - {j for _, j in matches}:
        counts["fp"] += sum(len(value) for value in predicted[j].values())
        counts["field_fp"] += sum(bool(value) for value in predicted[j].values())
        counts["fields"] += len(predicted[j])
        counts["answered"] += sum(bool(value) for value in predicted[j].values())
        counts["extra_records"] += 1
    return (counts, matches) if with_matches else counts


def score_episode(episode, raw, error=None):
    counts = Counter(requests=1, errors=int(error is not None))
    if error:
        # Preserve the complete denominator and count every answer as missed.
        raw = {}
    full_annotation = True
    for task in episode.tasks:
        gold = episode.gold[task.id]
        full_annotation &= gold.complete and all(
            a.status not in {"unknown_annotation", "ambiguous"}
            for r in gold.records for a in r.values())
        counts.update(score_task(task, gold, prediction_rows(task, raw)))
    counts["exact_requests"] = int(not error and full_annotation and counts["fp"] == 0
                                   and counts["fn"] == 0
                                   and counts["gold_records"] == counts["predicted_records"])
    return counts


def summarize(counts):
    def divide(a, b):
        return a / b if b else None
    tp, fp, fn = counts["field_tp"], counts["field_fp"], counts["field_fn"]
    result = {
        "counts": dict(counts),
        "field_precision": divide(tp, tp + fp), "field_recall": divide(tp, tp + fn),
        "field_f1": divide(2 * tp, 2 * tp + fp + fn),
        "element_f1": divide(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
        "request_exact": divide(counts["exact_requests"], counts.get("scored_requests", counts["requests"])),
        "reference_complete": not bool(counts.get("incomplete_reference_fields", 0)),
        "record_precision": divide(counts["exact_records"], counts["predicted_records"]),
        "record_recall": divide(counts["exact_records"], counts["gold_records"]),
        "answer_rate": divide(counts["answered"], counts["fields"]),
        "answerable_coverage": divide(counts["covered"], counts["answerable"]),
        "false_answer_rate_absent": divide(counts["false_absent"], counts["absent"]),
        "rejection_accuracy": divide(counts["correct_rejections"], counts["expected_rejections"]),
        "issue_accuracy": divide(counts["issue_correct"], counts["issue_fields"]),
        "error_rate": divide(counts["errors"], counts["requests"]),
    }
    if not result['reference_complete']:
        # The reviewed subset changes with the outputs. Its fraction is neither
        # a population estimate nor a comparable model-quality score.
        metrics = ('field_precision', 'field_recall', 'field_f1', 'element_f1',
                   'request_exact', 'record_precision', 'record_recall')
        result['reviewed_subset_scores'] = {key: result[key] for key in metrics}
        result.update(dict.fromkeys(metrics))
    return result
