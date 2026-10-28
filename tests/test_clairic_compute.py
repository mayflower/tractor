"""Deterministic compute: exact Decimal arithmetic and explicit calendar rules."""
from datetime import date
from decimal import Decimal

import pytest

from tractor.capabilities import ProcessingFailed
from tractor.clairic_compute import Operand, Operation, add_months, evaluate, round_result

A, B = Operand("f0", "subtotal", "money"), Operand("f1", "shipping", "money")
DAY = Operand("f0", "issue date", "date")


@pytest.mark.parametrize("operator,a,b,expected", [
    ("sum", Decimal("0.1"), Decimal("0.2"), Decimal("0.3")),
    ("difference", Decimal("5"), Decimal("7.25"), Decimal("-2.25")),
    ("product", Decimal("4"), Decimal("250.00"), Decimal("1000.00")),
    ("sum", Decimal("0"), Decimal("-0"), Decimal("0")),
])
def test_exact_decimal_arithmetic(operator, a, b, expected):
    assert evaluate("x", Operation(operator, (A, B)), {"f0": a, "f1": b}, []) == expected


@pytest.mark.parametrize("value,decimals,expected", [
    (Decimal("2.345"), 2, Decimal("2.34")),  # Half-even, once, at the end.
    (Decimal("2.355"), 2, Decimal("2.36")),
    (Decimal("-1.005"), 2, Decimal("-1.00")),
    (Decimal("7"), None, Decimal("7")),
])
def test_rounding_rule(value, decimals, expected):
    assert round_result("x", value, decimals) == expected


@pytest.mark.parametrize("day,operator,constant,expected", [
    ("2026-01-31", "add_months", 1, "2026-02-28"),
    ("2028-01-31", "add_months", 1, "2028-02-29"),
    ("2026-11-30", "add_months", 3, "2027-02-28"),
    ("2028-02-29", "add_years", 1, "2029-02-28"),
    ("2026-12-25", "add_days", 10, "2027-01-04"),
    ("2028-02-28", "add_days", 1, "2028-02-29"),
    ("2026-09-02", "add_days", 30, "2026-10-02"),
    ("2026-12-28", "add_weeks", 1, "2027-01-04"),
])
def test_calendar_addition(day, operator, constant, expected):
    assert evaluate("x", Operation(operator, (DAY,), constant), {"f0": day}, []) == expected


def test_out_of_range_dates_are_errors():
    with pytest.raises(ProcessingFailed, match="calendar"):
        evaluate("x", Operation("add_years", (DAY,), 9000), {"f0": "2026-01-01"}, [])
    with pytest.raises(OverflowError):
        add_months(date(9999, 12, 1), 1)


@pytest.mark.parametrize("data,issues,code", [
    ({"f0": None, "f1": Decimal(1)}, [], "missing_operand"),
    ({"f0": [Decimal(1), Decimal(2)], "f1": Decimal(1)}, [{"field": "f0", "code": "ambiguous_value"}],
     "ambiguous_value"),
    ({"f0": [Decimal(1), Decimal(2)], "f1": Decimal(1)}, [], "ambiguous_operand"),
    ({"f0": None, "f1": Decimal(1)}, [{"field": "f0", "code": "currency_mismatch"}], "currency_mismatch"),
])
def test_missing_invalid_or_competing_operands_are_explicit(data, issues, code):
    with pytest.raises(ProcessingFailed) as error:
        evaluate("total", Operation("sum", (A, B)), data, issues)
    assert error.value.field == "total" and error.value.details[0]["code"] == code


@pytest.mark.parametrize("operator,operands,constant", [
    ("divide", (A, B), None),
    ("sum", (A,), None),
    ("add_days", (DAY,), None),
    ("add_days", (DAY, DAY), 1),
    ("sum", (A, B, A), None),
])
def test_only_bounded_known_operations_exist(operator, operands, constant):
    with pytest.raises(ValueError):
        Operation(operator, operands, constant)
