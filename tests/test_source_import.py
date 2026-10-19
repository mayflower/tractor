import pytest

from tractor.source_import import guidex, parse_guidex_labels, qa, unique_span


@pytest.mark.parametrize("value", ["__import__('os').system('false')", "[x() for x in []]",
                                  "[Record(value=open('file'))]", "[obj.Record(value='x')]"])
def test_guidex_rejects_executable_expressions(value):
    with pytest.raises(ValueError):
        parse_guidex_labels(value)


def test_guidex_literal_constructor():
    assert parse_guidex_labels('[Record(name="Anna")]') == [("Record", {"name": "Anna"})]


def test_repeated_occurrences_need_explicit_offsets():
    with pytest.raises(ValueError, match="ambiguous"):
        unique_span("Anna met Anna", "Anna")


def test_qa_annotations_are_alternatives_not_a_list():
    episode = qa({"id": "1", "title": "Article", "context": "Anna is here.",
                  "question": "Who is here?", "answers": {"text": ["Anna", "Anna"],
                                                           "answer_start": [0, 0]}}, "squad_v2")
    answer = episode.gold["qa"].records[0]["answer"]
    assert len(answer.answer_alternatives) == 2
    assert not answer.items


def test_guidex_rejects_answers_in_field_descriptions():
    row = {"unlabelled_sentence": "Anna works here.", "labels": '[Person(name="Anna")]',
           "text": 'class Person:\n    name: str\n    """The person Anna."""\n'}
    with pytest.raises(ValueError, match="answer_in_schema"):
        guidex(row)
