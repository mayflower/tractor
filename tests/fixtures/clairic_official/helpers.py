"""Vendored, unmodified assertion logic from the official Clairic client tests.

Source: cloudflare/langchain-cloudflare@85a0e443d618beac451335a3fcc423da211ae5c5
  libs/langchain-cloudflare/tests/extraction_helpers.py (assert_extraction_result,
  ExtractionQualityError) and examples/workers/src/system_one.py (group_list_spans).
License: MIT, Copyright (c) 2024 LangChain, Inc.; full text in LICENSE next to this file.
Only the fixture-specific imports were dropped; the assertions are unchanged.
"""

def group_list_spans(
    *, text: str, values: list[str], spans: list[list[int]]
) -> dict[str, list[list[int]]]:
    """Anchor every occurrence by exact substring, preserving value order.

    Native list values and occurrence spans need not have equal lengths or
    matching order. Reject spans outside the source or absent from the values,
    and values without any matching occurrence. Do not infer or repair names.
    """
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError("List values must be strings")
    if not isinstance(spans, list):
        raise ValueError("List spans must be a list")
    grouped: dict[str, list[list[int]]] = {value: [] for value in values}
    for span in spans:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(type(offset) is int for offset in span)
        ):
            raise ValueError("Each source span must contain two integer offsets")
        start, end = span
        if not 0 <= start < end <= len(text):
            raise ValueError("Source span is outside the input text")
        value = text[start:end]
        if value not in grouped:
            raise ValueError("Source span does not match any list value")
        grouped[value].append(span)
    if any(not occurrences for occurrences in grouped.values()):
        raise ValueError("List value has no matching source span")
    return grouped


class ExtractionQualityError(AssertionError):
    """A well-formed native answer disagrees with the fixed expected value."""


def assert_extraction_result(result: dict, payload: dict, expected: dict) -> None:
    """Require correct values, source evidence, uncertainty and native usage."""
    assert result["model"] == "clairic"
    assert result["truncated"] is False
    assert set(result["answers"]) == set(payload["fields"])
    assert result["usage"]["input_tokens"] > 0
    assert result["usage"]["output_tokens"] == 0
    if payload.get("images"):
        assert result["usage"]["image_tokens"] > 0
        if not payload["state"]:
            assert result["usage"]["state_tokens"] == 0
    if payload.get("options", {}).get("return_state_text"):
        assert result["state_text"] == payload["state"]
    mismatches = []
    for key, field in payload["fields"].items():
        answer = result["answers"][key]
        assert answer["type"] == field["type"]
        if answer["value"] != expected[key]:
            mismatches.append(f"{key}: expected {expected[key]!r}, got {answer}")
        assert 0 <= answer["confidence"] <= 1
        if field["type"] != "list":
            assert 0 <= answer["p_null"] <= 1
            threshold = field.get("null_threshold", 0.5)
            assert (answer["value"] is None) == (answer["p_null"] > threshold)
        if field["type"] == "list":
            try:
                group_list_spans(
                    text=payload["state"],
                    values=answer["value"],
                    spans=answer["spans"],
                )
            except ValueError as error:
                raise AssertionError(str(error)) from error
        elif answer["value"] is not None and field["type"] == "text":
            assert isinstance(answer["value"], str)
            if answer.get("source") == "image":
                assert payload.get("images"), "Image answers require image input"
                assert answer["span"] is None
                if "spell_confidence" in answer:
                    assert 0 <= answer["spell_confidence"] <= 1
            else:
                start, end = answer["span"]
                assert 0 <= start < end <= len(payload["state"])
                assert payload["state"][start:end] == answer["value"]
    if mismatches:
        raise ExtractionQualityError("; ".join(mismatches))
