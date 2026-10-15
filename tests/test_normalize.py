from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tractor.jsonio import loads
from tractor.normalize import normalize

CASES = [
    loads(line)
    for line in Path("tests/fixtures/normalization.jsonl").read_text().splitlines()
]


@pytest.mark.parametrize("row", CASES, ids=lambda row: row["id"])
def test_normalization(row):
    result = normalize(row["text"], row["span"], row["target_unit"], row["locale"])
    expected = row["expected"]
    assert result.issue == expected.get("issue")
    if "value_decimal" in expected:
        assert abs(result.value - Decimal(expected["value_decimal"])) <= Decimal(
            expected.get("absolute_tolerance", "0")
        )


@given(st.integers(-(10**12), 10**12))
def test_independent_cm_mm(value):
    text = f"{value} cm"
    result = normalize(text, {"text": text, "start": 0, "end": len(text)}, "mm", "en-US")
    assert result.value == Decimal(value) * 10


@pytest.mark.parametrize("prefix", ["ca. ", "ungefähr ", "at most ", "up to ", "bis zu "])
def test_adjacent_qualifier(prefix):
    text = prefix + "12 cm"
    assert (
        normalize(text, {"text": "12 cm", "start": len(prefix), "end": len(text)}, "mm").issue
        == "non_scalar_quantity"
    )


def test_repeated_source_and_invalid_offset():
    text = "12 cm, dann 12 cm"
    assert normalize(text, {"text": "12 cm", "start": 11, "end": 16}, "mm").issue == "invalid_span"


@given(
    st.integers(-(10**9), 10**9),
    st.sampled_from(["mm", "cm", "m", "kg", "g", "ms", "s", "mL", "L", "m2", "cm2", "m/s", "km/h"]),
)
def test_identity(value, unit):
    text = f"{value} {unit}"
    result = normalize(text, {"text": text, "start": 0, "end": len(text)}, unit, "en-US")
    assert result.value == value


@given(st.integers(-(10**9), 10**9))
def test_scale_composition_and_roundtrip(value):
    source = f"{value} cm"
    mm = normalize(source, {"text": source, "start": 0, "end": len(source)}, "mm", "en-US").value
    intermediate = f"{mm} mm"
    m = normalize(
        intermediate, {"text": intermediate, "start": 0, "end": len(intermediate)}, "m", "en-US"
    ).value
    assert m == Decimal(value) / 100
    intermediate = f"{m} m"
    original = normalize(
        intermediate, {"text": intermediate, "start": 0, "end": len(intermediate)}, "cm", "en-US"
    ).value
    assert original == value


def test_huge_exponent_is_a_domain_issue():
    text = "1e" + ("9" * 5000) + " cm"
    assert (
        normalize(text, {"text": text, "start": 0, "end": len(text)}, "mm").issue
        == "numeric_out_of_range"
    )


def test_next_line_negative_measurement_is_not_a_range():
    text = "Höhe = 10 cm\n-2 cm (Breite)"
    span = {"text": "10 cm", "start": 7, "end": 12}
    result = normalize(text, span, "mm", "de-DE")
    assert result.issue is None
    assert result.value == 100


def test_range_spanning_lines_remains_non_scalar_when_selected_whole():
    text = "10 cm\n-20 cm"
    result = normalize(text, {"text": text, "start": 0, "end": len(text)}, "mm", "de-DE")
    assert result.value is None
    assert result.issue == "unsupported_number_format"
