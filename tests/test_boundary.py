import numpy as np
import pytest

from FRW.boundary import (
    conductor_distance,
    reflect_point,
    validate_window,
    window_distance,
)
from bgs.gaussian_surface import generate_bgs
from bgs.models import Bounds, Conductor


def _conductor(name, bounds):
    return Conductor(name, bounds, bounds.vertices())


def test_mixed_window_limits_only_dirichlet_axes() -> None:
    window = Bounds(0, 10, 0, 10, 0, 10)

    assert window_distance([2, 3, 9], window, ("z",)) == 2.0
    assert window_distance([2, 3, 9], window, ("x", "y", "z")) == np.inf


def test_neumann_crossing_reflects_and_dirichlet_crossing_absorbs() -> None:
    window = Bounds(0, 10, 0, 10, 0, 10)

    reflected, count, absorbed = reflect_point(
        [2, 3, 12],
        window,
        ("z",),
    )
    np.testing.assert_allclose(reflected, [2, 3, 8])
    assert count == 1
    assert absorbed is False

    _point, count, absorbed = reflect_point(
        [-1, 3, 12],
        window,
        ("z",),
    )
    assert count == 0
    assert absorbed is True


def test_reflected_conductor_distance_maps_back_to_real_conductor() -> None:
    window = Bounds(0, 10, 0, 10, 0, 10)
    conductor = _conductor("C1", Bounds(2, 4, 2, 4, 0, 2))

    assert conductor_distance(
        [3, 3, 8],
        conductor,
        window,
        ("z",),
    ) == conductor.distance_to_point([3, 3, 8])


def test_conductor_may_touch_neumann_but_not_dirichlet_wall() -> None:
    window = Bounds(0, 10, 0, 10, 0, 10)
    conductor = _conductor("C1", Bounds(2, 4, 2, 4, 0, 2))

    validate_window([conductor], window, ("z",))

    with pytest.raises(ValueError, match="Dirichlet z wall"):
        validate_window([conductor], window)


def test_bgs_omits_faces_that_coincide_with_neumann_walls() -> None:
    window = Bounds(-2, 4, -2, 4, 0, 1)
    conductor = _conductor("C1", Bounds(0, 1, 0, 1, 0, 1))

    result = generate_bgs(
        conductor,
        [conductor],
        max_distance=1.0,
        window=window,
        neumann_axes=("z",),
    )

    directions = {face.direction for face in result.gaussian_surface.faces}
    assert directions == {"NX", "PX", "NY", "PY"}
