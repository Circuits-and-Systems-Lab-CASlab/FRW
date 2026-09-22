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


def test_visualize_3d_walk_html(tmp_path) -> None:
    from FRW.visualization import visualize_3d_walk_html
    from bgs.models import Bounds, Conductor
    from bgs.vgss import build_all_net_vgss

    c1 = Conductor("C1", Bounds(0, 1, 0, 1, 0, 1), Bounds(0, 1, 0, 1, 0, 1).vertices())
    c2 = Conductor("C2", Bounds(2, 3, 0, 1, 0, 1), Bounds(2, 3, 0, 1, 0, 1).vertices())
    all_net_vgss = build_all_net_vgss([c1, c2])

    path_points = np.array([[0.5, 0.5, 1.2], [1.0, 0.5, 1.2], [2.0, 0.5, 0.5]])
    maximal_cubes = [([0.5, 0.5, 1.2], 0.2), ([1.0, 0.5, 1.2], 0.5)]
    terminal = "C2"

    html_file = tmp_path / "test_walk.html"
    out_path = visualize_3d_walk_html(
        [c1, c2],
        all_net_vgss[0],
        path_points,
        maximal_cubes,
        terminal,
        all_net_vgss=all_net_vgss,
        save_path=html_file,
    )

    assert out_path.exists()
    content = out_path.read_text()
    assert "plotly-2.35.2.min.js" in content
    assert "mesh3d" in content
    assert "Conductor: C1" in content
    assert "Conductor: C2" in content
    assert "Maximal Cubes" in content
    assert "Terminal: C2" in content

