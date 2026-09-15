import numpy as np

from FRW.visualization import extract_walk_trace


def test_extract_walk_trace_uses_the_saved_solver_path() -> None:
    walk = {
        "start_point": [1.0, 2.0, 3.0],
        "path_data": [
            {"center": [1.0, 2.0, 3.0], "a": 4.0},
            {"center": [2.0, 2.5, 3.0], "a": 1.0},
        ],
        "hit_conductor": "C2",
        "termination": "hit_conductor",
    }

    path_points, maximal_cubes, terminal = extract_walk_trace(walk)

    np.testing.assert_allclose(
        path_points,
        [[1.0, 2.0, 3.0], [2.0, 2.5, 3.0]],
    )
    assert [radius for _center, radius in maximal_cubes] == [2.0, 0.5]
    assert terminal == "C2"


def test_extract_walk_trace_keeps_a_start_point_without_hops() -> None:
    path_points, maximal_cubes, terminal = extract_walk_trace(
        {
            "start_point": [1.0, 2.0, 3.0],
            "path_data": [],
            "hit_conductor": None,
            "termination": "max_hops",
        }
    )

    np.testing.assert_allclose(path_points, [[1.0, 2.0, 3.0]])
    assert maximal_cubes == []
    assert terminal == "max_hops"
