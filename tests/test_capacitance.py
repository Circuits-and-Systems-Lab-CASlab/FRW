from __future__ import annotations

import pytest

from FRW.capacitance import (
    CapacitanceRowAccumulator,
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


def test_capacitance_accumulates_dirichlet_window_hits() -> None:
    conductors = [
        _conductor("C1", Bounds(0, 1, 0, 1, 0, 1)),
    ]
    master_vgss = _MasterVGSS(find_touching_nets(conductors)[0])
    accumulator = CapacitanceRowAccumulator(conductors, master_vgss)

    accumulator.update(
        {
            "omega": 2.0,
            "termination": "hit_window",
            "hit_conductor": None,
            "wall_potential": 3.0,
        }
    )
    result = accumulator.finalize(H=4.0)

    assert result["window_hits"] == 1
    assert result["geometric_window"] == pytest.approx(8.0)
    assert result["window_capacitance_F"] == pytest.approx(8.0 * E0 * L_UNIT)
    assert result["window_potential_contribution_F"] == pytest.approx(
        24.0 * E0 * L_UNIT
    )


def test_stratified_accumulator_uses_complete_pairs_for_statistics() -> None:
    conductors = [
        _conductor("C1", Bounds(0, 1, 0, 1, 0, 1)),
        _conductor("C2", Bounds(3, 4, 0, 1, 0, 1)),
    ]
    nets = find_touching_nets(conductors)
    master_vgss = _MasterVGSS(nets[0])
    accumulator = CapacitanceRowAccumulator(
        conductors,
        master_vgss,
        first_transition_mode="stratified",
    )

    outcomes = [
        {
            "omega": 2.0,
            "termination": "hit_conductor",
            "hit_conductor": "C1",
            "first_transition_stratum": "inward",
        },
        {
            "omega": 4.0,
            "termination": "hit_conductor",
            "hit_conductor": "C1",
            "first_transition_stratum": "outward",
        },
        {
            "omega": 6.0,
            "termination": "hit_conductor",
            "hit_conductor": "C1",
            "first_transition_stratum": "inward",
        },
        {
            "omega": 2.0,
            "termination": "hit_conductor",
            "hit_conductor": "C1",
            "first_transition_stratum": "outward",
        },
    ]

    completed = [accumulator.update(outcome) for outcome in outcomes]
    result = accumulator.finalize(H=2.0)

    assert completed == [False, True, False, True]
    assert result["num_walks"] == 4
    assert result["statistical_sample_count"] == 2
    assert result["raw_weight_sums"]["Net1"] == pytest.approx(14.0)
    assert result["mean_weight_contribution"]["Net1"] == pytest.approx(3.5)
    assert result["geometric_row"]["Net1"] == pytest.approx(7.0)
    assert result["weight_standard_error"]["Net1"] == pytest.approx(0.5)
    assert result["geometric_standard_error"]["Net1"] == pytest.approx(1.0)
    assert result["standard_error_F"]["Net1"] == pytest.approx(E0 * L_UNIT)


def test_stratified_accumulator_rejects_an_incomplete_pair() -> None:
    conductors = [
        _conductor("C1", Bounds(0, 1, 0, 1, 0, 1)),
    ]
    nets = find_touching_nets(conductors)
    accumulator = CapacitanceRowAccumulator(
        conductors,
        _MasterVGSS(nets[0]),
        first_transition_mode="stratified",
    )
    accumulator.update(
        {
            "omega": 1.0,
            "termination": "hit_conductor",
            "hit_conductor": "C1",
            "first_transition_stratum": "inward",
        }
    )

    with pytest.raises(ValueError, match="incomplete"):
        accumulator.finalize(H=1.0)


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
    assert result["all_rows_converged"] is None

    row_1["convergence"] = {"enabled": True, "converged": True}
    row_2["convergence"] = {"enabled": True, "converged": False}
    adaptive_result = assemble_capacitance_matrix([row_1, row_2])
    assert adaptive_result["all_rows_converged"] is False


def test_unit_scales_and_formatting(tmp_path) -> None:
    from FRW.capacitance import (
        export_capacitance_csv,
        export_spice_subcircuit,
        format_capacitance_matrix,
        get_unit_scale,
    )

    scale, unit_str = get_unit_scale("pF")
    assert scale == 1.0e12
    assert unit_str == "pF"

    scale_f, unit_f = get_unit_scale("fF")
    assert scale_f == 1.0e15
    assert unit_f == "fF"

    scale_auto, unit_auto = get_unit_scale("auto", max_abs_val=1e-13)
    assert unit_auto == "fF"

    matrix_res = {
        "matrix_F": [[2e-12, -1e-12], [-1e-12, 3e-12]],
        "net_order": ["Net1", "Net2"],
        "relative_asymmetry": 0.0,
        "max_abs_asymmetry_F": 0.0,
        "all_rows_converged": True,
    }

    table_str = format_capacitance_matrix(matrix_res, unit="pF")
    assert "Capacitance Matrix (pF)" in table_str
    assert "Net1" in table_str
    assert "Net2" in table_str
    assert "2.0000" in table_str

    csv_file = tmp_path / "matrix.csv"
    export_capacitance_csv(matrix_res, csv_file, unit="pF")
    assert csv_file.exists()
    content = csv_file.read_text()
    assert "Net,Net1,Net2" in content
    assert "Net1,2.000000e+00,-1.000000e+00" in content

    subckt_file = tmp_path / "model.subckt"
    export_spice_subcircuit(matrix_res, subckt_file, subckt_name="my_model", unit="pF")
    assert subckt_file.exists()
    spice_content = subckt_file.read_text()
    assert ".subckt my_model Net1 Net2" in spice_content
    assert "C_Net1_Net2 Net1 Net2 1p" in spice_content
    assert "C_Net1_GND Net1 0 1p" in spice_content
    assert "C_Net2_GND Net2 0 2p" in spice_content
    assert ".ends my_model" in spice_content

