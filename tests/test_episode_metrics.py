from tractor.episode_metrics import score_episode, summarize
from tractor.episodes import Episode


def records_episode():
    def answer(start, text):
        return {"status": "present", "items": [{"start": start, "end": start + len(text), "text": text}]}
    return Episode.model_validate({
        "id": "records", "source_group": "one", "derivation_group": "one", "source": "authored",
        "original_split": "dev", "split": "dev", "family": "records", "language": "en",
        "artifact": "table", "annotation": "synthetic_gold", "text": "A X A Y",
        "tasks": [{"id": "r", "kind": "records", "description": "Pairs", "anchor": "name",
                   "fields": {"name": "Name", "value": "Value"}}],
        "gold": {"r": {"complete": True, "records": [
            {"name": answer(0, "A"), "value": answer(2, "X")},
            {"name": answer(4, "A"), "value": answer(6, "Y")},
        ]}},
    })


def raw(episode):
    return {"r": [{k: [s.model_dump() for s in a.items] for k, a in row.items()}
                  for row in episode.gold["r"].records]}


def test_wrong_record_values_cannot_match_independently():
    episode = records_episode()
    prediction = raw(episode)
    prediction["r"][0]["value"], prediction["r"][1]["value"] = (
        prediction["r"][1]["value"], prediction["r"][0]["value"])
    counts = score_episode(episode, prediction)
    assert counts["exact_records"] == 0
    assert counts["fp"] == counts["fn"] == 2


def test_duplicates_count_as_extra_predictions():
    episode = records_episode()
    prediction = raw(episode)
    prediction["r"].append(prediction["r"][0])
    counts = score_episode(episode, prediction)
    assert counts["fp"] == 2 and counts["extra_records"] == 1
    assert counts["exact_requests"] == 0


def test_errors_and_empty_predictions_cannot_win():
    episode = records_episode()
    counts = score_episode(episode, {}, error="timeout")
    result = summarize(counts)
    assert result["field_precision"] is None
    assert result["field_recall"] == 0
    assert result["request_exact"] == 0
    assert result["error_rate"] == 1


def test_occurrence_aware_matching_ignores_record_order():
    episode = records_episode()
    prediction = raw(episode)
    prediction["r"].reverse()
    assert score_episode(episode, prediction)["exact_requests"] == 1


def test_partial_list_is_not_a_correct_complete_field():
    from tractor.episode_metrics import score_task
    from tractor.episodes import Answer, Gold, Span, Task

    task = Task(id="names", kind="entities", description="All names", fields={"name": "All names"})
    gold = Gold(complete=True, records=[{"name": Answer(status="present", items=[
        Span(start=0, end=1, text="A"), Span(start=2, end=3, text="B")])}])
    counts = score_task(task, gold, [{"name": [(0, 1, "A")]}])
    assert counts["tp"] == 1
    assert counts["field_tp"] == 0
    assert counts["field_fp"] == counts["field_fn"] == 1
