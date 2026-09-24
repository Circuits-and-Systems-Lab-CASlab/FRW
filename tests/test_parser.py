from pathlib import Path

import numpy as np

from bgs.matlab_parser import parse_matlab_geometry
from geometry import Bounds
from input_formats import load_geometry, parse_matlab_text


def test_parser_preserves_ordered_scalar_and_symbolic_matrix_assignments() -> None:
    document = parse_matlab_text(
        """
        eps_air = 1.0;
        eps_d1 = 3.9;
        EPS_D1 = [eps_d1 eps_air; eps_d1 eps_air];
        """
    )

    assert [assignment.name for assignment in document.assignments] == [
        "eps_air",
        "eps_d1",
        "EPS_D1",
    ]
    matrix = document.assignments[2]
    assert matrix.rows == (("eps_d1", "eps_air"), ("eps_d1", "eps_air"))


def test_legacy_conductor_parser_api_is_preserved(tmp_path: Path) -> None:
    input_path = tmp_path / "legacy.m"
    input_path.write_text(
        """
        eps_r = 3;
        C1 = [
            0 0 0   1 0 0   1 0 1   0 0 1;
            0 1 0   1 1 0   1 1 1   0 1 1;
            0 0 0   0 1 0   0 1 1   0 0 1;
            1 0 0   1 1 0   1 1 1   1 0 1;
            0 0 0   1 0 0   1 1 0   0 1 0;
            0 0 1   1 0 1   1 1 1   0 1 1
        ];
        """,
        encoding="utf-8",
    )

    conductors = parse_matlab_geometry(input_path)

    assert [item.name for item in conductors] == ["C1"]
    assert conductors[0].bounds == Bounds(0, 1, 0, 1, 0, 1)
    assert conductors[0].relative_permittivity == 3


def test_standard_dielectric_input_is_interpreted_into_volumes() -> None:
    path = Path(__file__).parents[1] / "input" / "DIELECTRIC_INPUT_3CONDUCTORS.m"
    geometry = load_geometry(path)

    assert [item.name for item in geometry.conductors] == ["C1", "C2", "C3"]
    assert [item.name for item in geometry.dielectrics] == ["D1", "D2", "D3", "D4"]
    assert geometry.background_permittivity == 1.006

    d1, d2, d3, d4 = geometry.dielectrics
    assert d1.bounds == Bounds(1, 1.5, 0, 1, 0, 1)
    assert d2.bounds == Bounds(1.5, 2, 0, 1, 0, 1)
    assert d3.bounds == Bounds(3, 3.5, 0, 1, 0, 1)
    assert d4.bounds == Bounds(0, 4.5, 0, 1, 1, 2)
    assert [d.relative_permittivity for d in geometry.dielectrics] == [3.9, 4.2, 3.0, 7.5]

    assert geometry.boundary_permittivity_pairs["C1"].shape == (6, 2)
    assert geometry.boundary_permittivity_pairs["D1"].shape == (5, 2)
    assert geometry.boundary_permittivity_pairs["D2"].shape == (4, 2)
    assert geometry.boundary_permittivity_pairs["D4"].shape == (5, 2)
    np.testing.assert_allclose(
        geometry.boundary_permittivity_pairs["D1"][2],
        [3.9, 4.2],
    )
