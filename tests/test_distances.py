import numpy as np

from bgs.distances import block_distance, directional_distances, limiting_directions
from bgs.models import Bounds, Conductor


def test_c1_to_c2_positive_x_gap() -> None:
    c1 = Bounds(0, 1, 0, 1, 0, 1)
    c2 = Bounds(3, 4, 0, 1, 0, 1)

    distances = directional_distances(c1, c2)

    assert distances == {
        "PX": 2,
        "PY": -1,
        "PZ": -1,
        "NX": -4,
        "NY": -1,
        "NZ": -1,
    }
    assert block_distance(c1, c2) == 2
    assert limiting_directions(c1, c2) == ("PX",)


def test_conductor_chebyshev_distance() -> None:
    conductor = Conductor(
        "C1",
        Bounds(0, 1, 0, 1, 0, 1),
        face_vertices=np.empty((0, 3)),
    )

    assert conductor.distance_to_point([0.5, 0.5, 0.5]) == 0.0
    assert conductor.distance_to_point([3.0, -2.0, 0.5]) == 2.0
