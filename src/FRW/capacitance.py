from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from bgs.topology import find_touching_nets

E0 = 8.8541878128e-12
L_UNIT = 1.0e0

INFINITY_LABEL = "Infinity"

def calculate_capacitance_row(walks: Sequence[Mapping], H: float, conductors, master_vgss, ) -> dict:
    """Convert one batch of raw FRW walks into one capacitance-matrix row.

    Homogeneous estimator:

        C_ij = E0 * L_UNIT * H / N * sum(omega_k * I[hit net j])

    """

    num_walks = len(walks)
    if num_walks == 0:
        raise ValueError("At least one walk is required to estimate capacitance.")
    if not np.isfinite(H) or H < 0.0:
        raise ValueError("H must be finite and non-negative.")

    nets = find_touching_nets(conductors)
    net_order = [net.name for net in nets]
    conductor_to_net = {
        conductor.name: net.name
        for net in nets
        for conductor in net.conductors
    }

    weight_sum = {net_name: 0.0 for net_name in net_order}
    infinity_weight_sum = 0.0
    unresolved_walks = 0

    for walk in walks:
        omega = float(walk["omega"])

        termination = walk.get("termination")
        hit_conductor = walk.get("hit_conductor")

        if termination == "hit_conductor":
            target_net = conductor_to_net[hit_conductor]
            weight_sum[target_net] += omega
        elif termination == "escaped":
            infinity_weight_sum += omega
        elif termination == "max_hops":
            unresolved_walks += 1

    geometric_factor = H / num_walks
    physical_factor = E0 * L_UNIT * geometric_factor

    capacitance_row = {
        net_name: physical_factor * weight_sum
        for net_name, weight_sum in weight_sum.items()
    }

    geometric_infinity = geometric_factor * infinity_weight_sum
    infinity_capacitance = physical_factor * infinity_weight_sum

    unresolved_rate = unresolved_walks / num_walks
    row_sum_finite = float(sum(capacitance_row.values()))
    row_sum_with_infinity = row_sum_finite + infinity_capacitance

    return {
        "master_net": master_vgss.net.name,
        "net_order": net_order,
        "num_walks": num_walks,
        "H": float(H),
        "E0": E0,
        "L_UNIT": L_UNIT,
        "raw_weight_sums":  weight_sum,
        "infinity_weight_sum": float(infinity_weight_sum),
        "capacitance_row_F": capacitance_row,
        "geometric_infinity": float(geometric_infinity),
        "infinity_capacitance_F": float(infinity_capacitance),
        "unresolved_walks": unresolved_walks,
        "unresolved_rate": float(unresolved_rate),
        "is_complete": unresolved_walks == 0,
        "row_sum_finite_F": row_sum_finite,
        "row_sum_with_infinity_F": float(row_sum_with_infinity),
    }


def assemble_capacitance_matrix(
    row_results: Sequence[Mapping],
    net_order: Sequence[str] | None = None,
) -> dict:
    """Assemble independently estimated rows into a full net capacitance matrix."""

    if not row_results:
        raise ValueError("At least one capacitance row is required.")

    if net_order is None:
        net_order = list(row_results[0]["net_order"])
    else:
        net_order = list(net_order)

    row_by_master = {row["master_net"]: row for row in row_results}

    matrix = np.zeros((len(net_order), len(net_order)), dtype=float)
    infinity_column = np.zeros(len(net_order), dtype=float)

    for row_index, master_net in enumerate(net_order):
        row_result = row_by_master[master_net]
        row_values = row_result["capacitance_row_F"]

        for column_index, target_net in enumerate(net_order):
            matrix[row_index, column_index] = float(row_values[target_net])

        infinity_column[row_index] = float(
            row_result["infinity_capacitance_F"]
        )

    difference = matrix - matrix.T
    max_abs_entry = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    max_abs_asymmetry = float(np.max(np.abs(difference))) if matrix.size else 0.0
    relative_asymmetry = (
        max_abs_asymmetry / max_abs_entry
        if max_abs_entry > 0.0
        else 0.0
    )

    return {
        "net_order": net_order,
        "matrix_F": matrix.tolist(),
        "infinity_column_F": infinity_column.tolist(),
        "diagonal_F": np.diag(matrix).tolist(),
        "max_abs_asymmetry_F": max_abs_asymmetry,
        "relative_asymmetry": float(relative_asymmetry),
        "all_rows_complete": all(bool(row["is_complete"]) for row in row_results),
    }


def print_capacitance_row(row_result: Mapping) -> None:
    master_net = row_result["master_net"]

    print("\n" + "-" * 30)
    print(f"Capacitance row for {master_net}:")
    for target_net in row_result["net_order"]:
        value = row_result["capacitance_row_F"][target_net]
        print(f"  C[{master_net}, {target_net}] = {value:.6e} F")

    print(
        "  C["
        f"{master_net}, {INFINITY_LABEL}] = "
        f"{row_result['infinity_capacitance_F']:.6e} F"
    )
    print(
        "  finite row sum = "
        f"{row_result['row_sum_finite_F']:.6e} F"
    )
    print(
        "  row sum including infinity = "
        f"{row_result['row_sum_with_infinity_F']:.6e} F"
    )
    print(
        "  unresolved walks = "
        f"{row_result['unresolved_walks']} "
        f"({row_result['unresolved_rate'] * 100.0:.3f}%)"
    )
    if not row_result["is_complete"]:
        print("  WARNING: this row contains unresolved max-hop walks.")


def print_capacitance_matrix(matrix_result: Mapping) -> None:
    matrix = np.asarray(matrix_result["matrix_F"], dtype=float)
    net_order = matrix_result["net_order"]

    print("\nCapacitance matrix (F):")
    print("Net order: " + ", ".join(net_order))
    print(np.array2string(matrix, precision=4, suppress_small=False))
    
    print(
        "Relative symmetry error: "
        f"{matrix_result['relative_asymmetry']:.3e}"
    )

