"""Contract boundaries that protect supervision and evaluation integrity."""

import pytest

from tractor.episodes import Episode, audit_episodes, audit_similarity, schema_variant
from tractor.evaluate import equal


def example():
    return {
        "id": "one", "source_group": "doc", "derivation_group": "doc",
        "source": "authored", "original_split": "train", "split": "train",
        "family": "person", "language": "de", "artifact": "note", "annotation": "synthetic_gold",
        "text": "Änne arbeitet hier.",
        "tasks": [{"id": "people", "kind": "entities", "description": "People",
                   "fields": {"name": "Name einer Person"}}],
        "gold": {"people": {"complete": True, "records": [{"name": {
            "status": "present", "items": [{"start": 0, "end": 4, "text": "Änne"}]
        }}]}},
    }


def test_unicode_offsets_and_unknowns():
    row = example()
    assert Episode.model_validate(row).gold["people"].records[0]["name"].items[0].text == "Änne"
    row["gold"]["people"]["records"][0]["name"]["items"][0]["end"] = 5
    with pytest.raises(ValueError, match="original text"):
        Episode.model_validate(row)


def test_source_and_text_leakage():
    a = Episode.model_validate(example())
    b = a.model_copy(update={"id": "two", "split": "test"})
    with pytest.raises(ValueError, match="crosses splits"):
        audit_episodes([a, b])
    b = b.model_copy(update={"source_group": "other", "derivation_group": "other"})
    with pytest.raises(ValueError, match="source text"):
        audit_episodes([a, b])


def test_alternatives_are_not_items():
    row = example()
    answer = row["gold"]["people"]["records"][0]["name"]
    answer["answer_alternatives"] = [answer["items"]]
    with pytest.raises(ValueError, match="OR"):
        Episode.model_validate(row)


@pytest.mark.parametrize("a,b", [(False, 0), (True, 1), (None, ""), ([], None)])
def test_typed_equality(a, b):
    assert not equal(a, b)


def test_boolean_equality():
    assert equal(False, False)
    assert equal({"active": True}, {"active": True})


def test_ambiguous_evidence_is_preserved_without_positive_supervision():
    row = example()
    row["gold"]["people"]["records"][0]["name"]["status"] = "ambiguous"
    episode = Episode.model_validate(row)
    assert episode.gold["people"].records[0]["name"].items[0].text == "Änne"


def test_near_duplicate_real_text_is_detected_across_datasets():
    row = example()
    row.update(annotation="human", text="Änne " + " ".join(f"word{i}" for i in range(30)))
    a = Episode.model_validate(row)
    b = a.model_copy(update={"id": "other", "source_group": "other", "derivation_group": "other",
                             "split": "dev", "text": a.text.replace("word5", "word5,")})
    assert audit_similarity([a, b])["matches"]


def test_schema_variant_keeps_descriptions_offsets_and_source_groups():
    original = Episode.model_validate(example())
    variant = schema_variant(original, 42)
    assert variant.tasks[0].fields == {"f0": "Name einer Person"}
    assert variant.gold["people"].records[0]["f0"] == original.gold["people"].records[0]["name"]
    assert variant.source_group == original.source_group
    assert variant.split == original.split
    assert variant.text == original.text
    assert variant.id != original.id
