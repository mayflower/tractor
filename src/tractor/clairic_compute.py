"""Deterministic arithmetic over extracted, typed operands; no ML imports.

Only the operators below exist. Operands come from the same extraction and
normalization pass as every other field. Decimal arithmetic, explicit rounding
of the final result only, no exchange rates, no implicit clock and no fixed
seconds for calendar units. Computed values have no source span; the operand
evidence stays in the trace.
"""

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from .capabilities import ProcessingFailed

CONTEXT = Context(prec=50, rounding=ROUND_HALF_EVEN)
OPERATORS = ("sum", "difference", "product", "add_days", "add_weeks", "add_months", "add_years")
MAX_OPERANDS = 2


@dataclass(frozen=True)
class Operand:
    internal_id: str
    label: str
    kind: str  # "number", "money", "quantity" or "date"


@dataclass(frozen=True)
class Operation:
    """One bounded node: a fixed operator over at most two operands or a constant."""

    operator: str
    operands: tuple
    constant: int | None = None

    def __post_init__(self):
        if self.operator not in OPERATORS:
            raise ValueError(f"Unsupported operator: {self.operator}")
        if not 1 <= len(self.operands) <= MAX_OPERANDS:
            raise ValueError("Operations take one or two operands")
        if self.operator.startswith("add_") and (self.constant is None or len(self.operands) != 1):
            raise ValueError("Date addition takes one date operand and an integer constant")
        if not self.operator.startswith("add_") and len(self.operands) != 2:
            raise ValueError("Arithmetic operations take exactly two operands")


def operand_value(name, operand, data, issues):
    """Exactly one unambiguous, successfully normalized value, or a clear error."""
    codes = [i["code"] for i in issues if i.get("field") == operand.internal_id and i["code"] != "not_found"]
    if codes:
        raise ProcessingFailed(f"Operand {operand.label!r} could not be resolved", field=name,
                               details=[{"operand": operand.label, "code": c} for c in codes])
    value = data.get(operand.internal_id)
    if value is None:
        raise ProcessingFailed(f"Operand {operand.label!r} is not stated", field=name,
                               details=[{"operand": operand.label, "code": "missing_operand"}])
    if isinstance(value, list):
        raise ProcessingFailed(f"Operand {operand.label!r} has competing values", field=name,
                               details=[{"operand": operand.label, "code": "ambiguous_operand"}])
    return value


def add_months(day, months):
    """Calendar month arithmetic; a day past the target month's end clamps to its last day."""
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    if not 1 <= year <= 9999:
        raise OverflowError
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def evaluate(name, operation, data, issues):
    values = [operand_value(name, o, data, issues) for o in operation.operands]
    try:
        if operation.operator.startswith("add_"):
            day = date.fromisoformat(values[0])
            if operation.operator == "add_days":
                return (day + timedelta(days=operation.constant)).isoformat()
            if operation.operator == "add_weeks":
                return (day + timedelta(weeks=operation.constant)).isoformat()
            months = operation.constant * (12 if operation.operator == "add_years" else 1)
            return add_months(day, months).isoformat()
    except (OverflowError, ValueError):
        raise ProcessingFailed("Date result is outside the supported calendar", field=name,
                               details=[{"code": "date_out_of_range"}]) from None
    with localcontext(CONTEXT):
        a, b = (Decimal(v) for v in values)
        if operation.operator == "sum":
            return a + b
        if operation.operator == "difference":
            return a - b
        return a * b


def round_result(name, value, decimals):
    """Round once, at the end, half-even; the stated operands are never rounded."""
    if decimals is None:
        return value
    with localcontext(CONTEXT):
        try:
            return value.quantize(Decimal(1).scaleb(-decimals))
        except ArithmeticError:
            raise ProcessingFailed("Result cannot be represented", field=name,
                                   details=[{"code": "numeric_out_of_range"}]) from None
