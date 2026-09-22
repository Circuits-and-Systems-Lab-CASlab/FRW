"""Finite Dirichlet and Neumann boundary helpers for the FRW engine."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

import numpy as np

from bgs.models import Bounds, Conductor


AXES = ("x", "y", "z")


def axis_indices(neumann_axes: Iterable[str]) -> tuple[int, ...]:
    """Convert axis names into sorted coordinate indices."""

    names = tuple(neumann_axes)
    unknown = set(names).difference(AXES)
    if unknown:
        raise ValueError("Unknown Neumann axes: " + ", ".join(sorted(unknown)))
    return tuple(index for index, name in enumerate(AXES) if name in names)


def bounds_arrays(bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper window coordinates."""

    lower = np.array([bounds.xmin, bounds.ymin, bounds.zmin], dtype=float)
    upper = np.array([bounds.xmax, bounds.ymax, bounds.zmax], dtype=float)
    return lower, upper


def inside_window(point, window: Bounds, tolerance: float = 0.0) -> bool:
    """Return whether a point lies inside a finite window."""

    lower, upper = bounds_arrays(window)
    point = np.asarray(point, dtype=float)
    return bool(
        np.all(point >= lower - tolerance)
        and np.all(point <= upper + tolerance)
    )


def window_distance(point, window: Bounds, neumann_axes=()) -> float:
    """Return the distance to the nearest Dirichlet window face."""

    lower, upper = bounds_arrays(window)
    point = np.asarray(point, dtype=float)
    neumann = set(axis_indices(neumann_axes))
    dirichlet = [axis for axis in range(3) if axis not in neumann]

    if not dirichlet:
        return math.inf
    if not inside_window(point, window):
        return 0.0

    return float(
        min(
            distance
            for axis in dirichlet
            for distance in (
                point[axis] - lower[axis],
                upper[axis] - point[axis],
            )
        )
    )


def _reflection_count(value: float, lower: float, upper: float) -> int:
    if lower <= value <= upper:
        return 0
    length = upper - lower
    relative = value - lower
    if relative > length:
        return int(math.floor(relative / length))
    return int(math.ceil(-relative / length))


def reflect_point(point, window: Bounds, neumann_axes=()):
    """Absorb at Dirichlet faces and fold Neumann crossings into the window."""

    lower, upper = bounds_arrays(window)
    point = np.asarray(point, dtype=float)
    neumann = set(axis_indices(neumann_axes))

    for axis in range(3):
        if axis not in neumann and not lower[axis] <= point[axis] <= upper[axis]:
            return point, 0, True

    reflected = point.copy()
    reflections = 0
    for axis in neumann:
        length = upper[axis] - lower[axis]
        folded = np.mod(point[axis] - lower[axis], 2.0 * length)
        reflected[axis] = lower[axis] + (
            folded if folded <= length else 2.0 * length - folded
        )
        reflections += _reflection_count(
            float(point[axis]),
            float(lower[axis]),
            float(upper[axis]),
        )

    return reflected, reflections, False


def _interval_distance(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def _mirror_distance(
    value: float,
    interval_lower: float,
    interval_upper: float,
    window_lower: float,
    window_upper: float,
) -> float:
    """Return the distance to the nearest periodic mirror of an interval."""

    length = window_upper - window_lower
    x = value - window_lower
    a = interval_lower - window_lower
    b = interval_upper - window_lower
    best = math.inf

    for period in (-1, 0, 1):
        offset = 2.0 * period * length
        best = min(
            best,
            _interval_distance(x, a + offset, b + offset),
            _interval_distance(x, -b + offset, -a + offset),
        )

    return float(best)


def conductor_distance(
    point,
    conductor: Conductor,
    window: Bounds | None = None,
    neumann_axes=(),
) -> float:
    """Return the distance to a conductor and its Neumann mirror images."""

    if not neumann_axes:
        return conductor.distance_to_point(point)

    neumann = set(axis_indices(neumann_axes))
    if window is None:
        raise ValueError("Neumann axes require a finite window.")

    point = np.asarray(point, dtype=float)
    conductor_lower = conductor.min_bounds
    conductor_upper = conductor.max_bounds
    window_lower, window_upper = bounds_arrays(window)
    distances = []

    for axis in range(3):
        if axis in neumann:
            distance = _mirror_distance(
                float(point[axis]),
                float(conductor_lower[axis]),
                float(conductor_upper[axis]),
                float(window_lower[axis]),
                float(window_upper[axis]),
            )
        else:
            distance = _interval_distance(
                float(point[axis]),
                float(conductor_lower[axis]),
                float(conductor_upper[axis]),
            )
        distances.append(distance)

    return float(max(distances))


def validate_window(
    conductors: Sequence[Conductor],
    window: Bounds | None,
    neumann_axes=(),
    tolerance: float = 1.0e-12,
) -> None:
    """Validate the finite-window geometry and boundary selection."""

    neumann = set(axis_indices(neumann_axes))
    if window is None:
        if neumann:
            raise ValueError("Neumann axes require a finite window.")
        return

    lower, upper = bounds_arrays(window)
    for conductor in conductors:
        if not inside_window(conductor.min_bounds, window, tolerance) or not inside_window(
            conductor.max_bounds,
            window,
            tolerance,
        ):
            raise ValueError(
                f"Conductor {conductor.name!r} is not contained in the window."
            )

        for axis in range(3):
            if axis in neumann:
                continue
            if (
                abs(conductor.min_bounds[axis] - lower[axis]) <= tolerance
                or abs(conductor.max_bounds[axis] - upper[axis]) <= tolerance
            ):
                raise ValueError(
                    f"Conductor {conductor.name!r} touches a Dirichlet "
                    f"{AXES[axis]} wall. Enlarge the window or make that "
                    "axis Neumann."
                )
