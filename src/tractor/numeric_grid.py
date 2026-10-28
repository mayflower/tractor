"""Exact grids and discrete quantiles for the `estimate` prototype; no ML imports.

Linear grids use Decimal (no binary float drift). A max that is not on the
step grid is rejected, not rounded. Log scales use log-spaced model coordinates
over a positive domain, but the public value set is the explicit grid points;
a step in original units never silently becomes a log distance. Date grids use
day ordinals; month/year steps keep the start day and clamp to month ends.
"""

import calendar
import math
from datetime import date, timedelta
from decimal import Decimal

MAX_POINTS = 4096
GRID_VERSION = "tractor-numeric-grid-v1"


class GridError(ValueError):
    pass


def linear(minimum, maximum, step):
    minimum, maximum, step = Decimal(minimum), Decimal(maximum), Decimal(step)
    if step <= 0 or maximum <= minimum:
        raise GridError("Grid needs min < max and a positive step")
    count = (maximum - minimum) / step
    if count != count.to_integral_value():
        raise GridError("max is not on the step grid")
    if count + 1 > MAX_POINTS:
        raise GridError(f"Grid exceeds {MAX_POINTS} points")
    return [minimum + step * i for i in range(int(count) + 1)]


def log_points(minimum, maximum, count):
    """Positive domain only; returns grid points and their log model coordinates."""
    minimum, maximum = Decimal(minimum), Decimal(maximum)
    if minimum <= 0 or maximum <= minimum:
        raise GridError("A log scale requires 0 < min < max")
    if not 2 <= count <= MAX_POINTS:
        raise GridError("Unsupported log grid size")
    low, high = math.log(minimum), math.log(maximum)
    coordinates = [low + (high - low) * i / (count - 1) for i in range(count)]
    # Endpoints are exact; inner points are the documented 12-significant-digit rendering.
    points = [minimum] + [Decimal(f"{math.exp(c):.12g}") for c in coordinates[1:-1]] + [maximum]
    return points, coordinates


def add_months(day, months):
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def dates(minimum, maximum, step):
    if not isinstance(minimum, date) or not isinstance(maximum, date) or maximum <= minimum:
        raise GridError("Date grids need ISO date bounds with min < max")
    points, current = [], minimum
    while current <= maximum:
        points.append(current)
        if len(points) > MAX_POINTS:
            raise GridError(f"Grid exceeds {MAX_POINTS} points")
        if step in ("month", "year"):
            current = add_months(minimum, len(points) * (12 if step == "year" else 1))
        else:
            if Decimal(step) != Decimal(step).to_integral_value() or Decimal(step) <= 0:
                raise GridError("Date steps are whole positive days")
            current = current + timedelta(days=int(step))
    if points[-1] != maximum:
        raise GridError("max is not on the date grid")
    return points


def grid_for(spec):
    """Grid points for a parsed Clairic estimate FieldSpec."""
    if isinstance(spec.min, date):
        return dates(spec.min, spec.max, spec.step or 1)
    if spec.scale == "log":
        if spec.step is not None:
            raise GridError("A step in original units is not a log-scale spacing")
        return log_points(spec.min, spec.max, 256)[0]
    return linear(spec.min, spec.max, spec.step or 1)


def quantile(points, masses, q):
    """Smallest grid point whose cumulative mass reaches q; monotone by construction."""
    if not 0 < q < 1:
        raise ValueError("Quantile level must be in (0, 1)")
    total = sum(masses)
    if total <= 0:
        raise ValueError("Distribution has no in-range mass")
    cumulative = 0.0
    for point, mass in zip(points, masses, strict=True):
        cumulative += mass / total
        if cumulative >= q - 1e-12:
            return point
    return points[-1]


def summary(points, masses, *, quantiles=False):
    """Statistics conditional on non-null AND the allowed range.

    Tail masses alone determine neither an unconditional mean nor quantiles
    inside a tail, so those are not produced here.
    """
    total = sum(masses)
    numeric = not isinstance(points[0], date)
    result = {"median": quantile(points, masses, 0.5), "p10": quantile(points, masses, 0.1),
              "p90": quantile(points, masses, 0.9), "conditioning": "non_null_in_range"}
    if numeric:
        result["mean"] = sum(Decimal(str(m / total)) * p for p, m in zip(points, masses, strict=True))
    if quantiles:
        result["quantiles"] = [quantile(points, masses, k / 100) for k in range(1, 100)]
    return result
