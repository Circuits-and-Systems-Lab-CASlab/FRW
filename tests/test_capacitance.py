from __future__ import annotations

import pytest

from FRW.capacitance import (
    E0,
    L_UNIT,
    assemble_capacitance_matrix,
    calculate_capacitance_row,
)
from bgs.models import Bounds, Conductor
from bgs.topology import find_touching_nets


class _MasterVGSS:
    def __init__(self, net):
        self.net = net


def _conductor(name: str, bounds: Bounds) -> Conductor:
    return Conductor(
        name=name,
        bounds=bounds,
        face_vertices=bounds.vertices(),
    )


def test_capacitance_row_uses_all_walks_and_tracks_infinity() -> None:
    conductors = [
        _conductor("C1", Bounds(0, 1, 0, 1, 0, 1)),
        _conductor("C2", Bounds(3, 4, 0, 1, 0, 1)),
    ]
    nets = find_touching_nets(conductors)
    master_vgss = _MasterVGSS(nets[0])

    walks = [
        {
            "omega": 2.0,
            "termination": "hit_conductor",
            "hit_conductor": "C1",
        },
        {
            "omega": -1.0,
            "termination": "hit_conductor",
            "hit_conductor": "C2",
        },
        {
            "omega": -0.5,
            "termination": "escaped",
            "hit_conductor": "Infinity (Ground)",
        },
        {
            "omega": 9.0,
            "termination": "max_hops",
            "hit_conductor": None,
        },
    ]

    result = calculate_capacitance_row(
        walks,
        H=4.0,
        conductors=conductors,
        master_vgss=master_vgss,
    )

    # H / N = 1.0, proving that all four launched walks are in N.
    assert result["geometric_row"]["Net1"] == pytest.approx(2.0)
    assert result["geometric_row"]["Net2"] == pytest.approx(-1.0)
    assert result["geometric_infinity"] == pytest.approx(-0.5)

    assert result["capacitance_row_F"]["Net1"] == pytest.approx(
        2.0 * E0 * L_UNIT
    )
    assert result["unresolved_walks"] == 1
    assert result["unresolved_rate"] == pytest.approx(0.25)
    assert result["is_complete"] is False


def test_capacitance_accumulates_touching_blocks_by_net() -> None:
    conductors = [
        _conductor("C1", Bounds(0, 1, 0, 1, 0, 1)),
        _conductor("C2", Bounds(1, 2, 0, 1, 0, 1)),
        _conductor("C3", Bounds(4, 5, 0, 1, 0, 1)),
    ]
    nets = find_touching_nets(conductors)
    master_vgss = _MasterVGSS(nets[0])

    walks = [
        {
            "omega": 1.5,
            "termination": "hit_conductor",
            "hit_conductor": "C1",
        },
        {
            "omega": 2.5,
            "termination": "hit_conductor",
            "hit_conductor": "C2",
        },
    ]

    result = calculate_capacitance_row(
        walks,
        H=2.0,
        conductors=conductors,
        master_vgss=master_vgss,
    )

    assert result["raw_weight_sums"]["Net1"] == pytest.approx(4.0)
    assert result["raw_weight_sums"]["Net2"] == pytest.approx(0.0)


def test_assemble_capacitance_matrix_preserves_independent_rows() -> None:
    row_1 = {
        "master_net": "Net1",
        "net_order": ["Net1", "Net2"],
        "capacitance_row_F": {"Net1": 4.0, "Net2": -1.0},
        "infinity_capacitance_F": -3.0,
        "is_complete": True,
    }
    row_2 = {
        "master_net": "Net2",
        "net_order": ["Net1", "Net2"],
        "capacitance_row_F": {"Net1": -1.2, "Net2": 3.0},
        "infinity_capacitance_F": -1.8,
        "is_complete": True,
    }

    result = assemble_capacitance_matrix([row_1, row_2])

    assert result["matrix_F"] == [[4.0, -1.0], [-1.2, 3.0]]
    assert result["diagonal_F"] == [4.0, 3.0]
    assert result["relative_asymmetry"] == pytest.approx(0.05)
    assert result["all_rows_complete"] is True
