#!/usr/bin/env python3
"""Memory-safe diagnostic runner for the VGSS/FRW capacitance implementation.

Runs one master net at a time, computes compact statistical diagnostics, and
never writes individual walk paths to disk.  Peak memory is therefore roughly
one master-net batch rather than the full all-net run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bgs.matlab_parser import parse_matlab_geometry
from bgs.topology import find_touching_nets
from bgs.vgss import build_all_net_vgss
from FRW.capacitance import E0, L_UNIT, assemble_capacitance_matrix, calculate_capacitance_row
from FRW.metrics import calculate_frw_metrics
from FRW.random_walk import run_frw_geometry

INFINITY = "Infinity"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run compact FRW diagnostics without exporting full walk paths. "
            "One master net is held in memory at a time."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--walks", type=int, default=10_000)
    parser.add_argument("--scale-factor", type=float, default=1.25)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--max-hops", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "diagnostics",
    )
    parser.add_argument(
        "--label",
        default="frw_math_diagnostic",
        help="Base name for compact output files.",
    )
    return parser


def _target_for_walk(walk: dict[str, Any], conductor_to_net: dict[str, str]) -> str | None:
    termination = walk.get("termination")
    if termination == "hit_conductor":
        return conductor_to_net[walk["hit_conductor"]]
    if termination == "escaped":
        return INFINITY
    return None


def _coefficient_statistics(
    walks: list[dict[str, Any]],
    H: float,
    net_order: list[str],
    conductor_to_net: dict[str, str],
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    """Estimate every row coefficient and its conditional Monte Carlo SE.

    H is treated as fixed here.  The later across-seed test will include the
    additional variation from estimating H.
    """

    n = len(walks)
    physical_scale = E0 * L_UNIT * H
    labels = net_order + [INFINITY]

    contribution_arrays = {
        label: np.zeros(n, dtype=np.float64)
        for label in labels
    }
    omega_by_target: dict[str, list[float]] = {label: [] for label in labels}
    all_resolved_omega = np.zeros(n, dtype=np.float64)

    for index, walk in enumerate(walks):
        target = _target_for_walk(walk, conductor_to_net)
        if target is None:
            continue
        omega = float(walk["omega"])
        contribution_arrays[target][index] = omega
        omega_by_target[target].append(omega)
        all_resolved_omega[index] = omega

    stats: dict[str, dict[str, float | int]] = {}
    for label in labels:
        values = contribution_arrays[label]
        estimate = physical_scale * float(np.mean(values))
        if n > 1:
            se = abs(physical_scale) * float(np.std(values, ddof=1)) / math.sqrt(n)
        else:
            se = 0.0
        hit_omegas = np.asarray(omega_by_target[label], dtype=float)
        if hit_omegas.size:
            positive_fraction = float(np.mean(hit_omegas > 0.0))
            negative_fraction = float(np.mean(hit_omegas < 0.0))
            mean_hit_omega = float(np.mean(hit_omegas))
            std_hit_omega = float(np.std(hit_omegas, ddof=1)) if hit_omegas.size > 1 else 0.0
            min_hit_omega = float(np.min(hit_omegas))
            max_hit_omega = float(np.max(hit_omegas))
        else:
            positive_fraction = 0.0
            negative_fraction = 0.0
            mean_hit_omega = 0.0
            std_hit_omega = 0.0
            min_hit_omega = 0.0
            max_hit_omega = 0.0

        stats[label] = {
            "count": int(hit_omegas.size),
            "estimate_F": estimate,
            "standard_error_F_conditional_on_H": se,
            "ci95_half_width_F_conditional_on_H": 1.96 * se,
            "mean_omega_on_hits": mean_hit_omega,
            "std_omega_on_hits": std_hit_omega,
            "min_omega_on_hits": min_hit_omega,
            "max_omega_on_hits": max_hit_omega,
            "positive_omega_fraction_on_hits": positive_fraction,
            "negative_omega_fraction_on_hits": negative_fraction,
        }

    residual = physical_scale * float(np.mean(all_resolved_omega))
    if n > 1:
        residual_se = abs(physical_scale) * float(np.std(all_resolved_omega, ddof=1)) / math.sqrt(n)
    else:
        residual_se = 0.0
    residual_z = residual / residual_se if residual_se > 0.0 else 0.0

    return stats, {
        "row_sum_with_infinity_F_from_walk_contributions": residual,
        "row_sum_standard_error_F_conditional_on_H": residual_se,
        "row_sum_z_score_conditional_on_H": residual_z,
    }


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = _parser().parse_args()
    if args.walks <= 1:
        raise SystemExit("--walks must be greater than 1 for standard-error diagnostics.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    conductors = parse_matlab_geometry(args.input_file)
    nets = find_touching_nets(conductors)
    all_vgss = build_all_net_vgss(
        conductors,
        scale_factor=args.scale_factor,
        max_distance=args.max_distance,
    )

    net_order = [net.name for net in nets]
    conductor_to_net = {
        conductor.name: net.name
        for net in nets
        for conductor in net.conductors
    }

    seed_sequence = np.random.SeedSequence(args.seed)
    child_sequences = seed_sequence.spawn(len(all_vgss))

    print("Input:", args.input_file)
    print("Walks per master net:", args.walks)
    print("Scale factor:", args.scale_factor)
    print("Max distance:", args.max_distance)
    print("Max hops:", args.max_hops)
    print("Base seed:", args.seed)
    print("\nNet mapping:")
    for net in nets:
        print(f"  {net.name}: {', '.join(net.conductor_names)}")

    compact_rows: list[dict[str, Any]] = []
    capacitance_rows: list[dict[str, Any]] = []
    se_rows: list[list[float]] = []
    infinity_se: list[float] = []

    for index, master_vgss in enumerate(all_vgss):
        master_name = master_vgss.net.name
        child_seed = int(child_sequences[index].generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(child_sequences[index])

        print("\n" + "=" * 72)
        print(f"Running {master_name} with independent child seed {child_seed}")
        print("=" * 72)

        raw = run_frw_geometry(
            conductors,
            master_vgss,
            rng,
            num_walks=args.walks,
            max_hops=args.max_hops,
        )
        walks = raw["walks"]

        metrics = calculate_frw_metrics(walks, conductors, master_vgss)
        row = calculate_capacitance_row(
            walks,
            raw["H"],
            conductors,
            master_vgss,
        )
        coeff_stats, conservation_stats = _coefficient_statistics(
            walks,
            raw["H"],
            net_order,
            conductor_to_net,
        )

        direct_diag = float(row["capacitance_row_F"][master_name])
        offdiag_sum = sum(
            float(row["capacitance_row_F"][target])
            for target in net_order
            if target != master_name
        )
        signed_infinity = float(row["infinity_capacitance_F"])
        conservation_diag = -(offdiag_sum + signed_infinity)

        row_summary = {
            "master_net": master_name,
            "conductors": list(master_vgss.net.conductor_names),
            "child_seed": child_seed,
            "walks": args.walks,
            "Sg": float(raw["Sg"]),
            "mean_p": float(raw["mean_p"]),
            "H": float(raw["H"]),
            "sampling_stats": raw["sampling_stats"],
            "metrics": metrics,
            "capacitance": row,
            "coefficient_statistics": coeff_stats,
            "conservation": {
                **conservation_stats,
                "direct_diagonal_F": direct_diag,
                "diagonal_from_offdiagonals_and_infinity_F": conservation_diag,
                "direct_minus_conservation_diagonal_F": direct_diag - conservation_diag,
            },
        }
        compact_rows.append(_to_builtin(row_summary))
        capacitance_rows.append(row)
        se_rows.append([
            float(coeff_stats[target]["standard_error_F_conditional_on_H"])
            for target in net_order
        ])
        infinity_se.append(float(coeff_stats[INFINITY]["standard_error_F_conditional_on_H"]))

        print(f"H = {raw['H']:.10e}")
        print(f"VGSS acceptance rate = {raw['sampling_stats']['acceptance_rate']:.6f}")
        print("Termination counts =", metrics["termination_counts"])
        print("Capacitance row (F):")
        for target in net_order:
            estimate = row["capacitance_row_F"][target]
            se = coeff_stats[target]["standard_error_F_conditional_on_H"]
            print(f"  C[{master_name},{target}] = {estimate:+.10e}  SE~{se:.3e}")
        print(
            f"  C[{master_name},Infinity] = {signed_infinity:+.10e}  "
            f"SE~{coeff_stats[INFINITY]['standard_error_F_conditional_on_H']:.3e}"
        )
        print(
            "Row sum including infinity = "
            f"{conservation_stats['row_sum_with_infinity_F_from_walk_contributions']:+.10e} "
            f"(z~{conservation_stats['row_sum_z_score_conditional_on_H']:+.3f})"
        )
        print(
            "Direct diagonal - conservation diagonal = "
            f"{direct_diag - conservation_diag:+.10e} F"
        )

        # Release the path-heavy data before the next master net.
        del walks
        del raw

    matrix_result = assemble_capacitance_matrix(capacitance_rows, net_order)
    matrix = np.asarray(matrix_result["matrix_F"], dtype=float)
    se_matrix = np.asarray(se_rows, dtype=float)
    infinity_column = np.asarray(matrix_result["infinity_column_F"], dtype=float)
    infinity_se_array = np.asarray(infinity_se, dtype=float)

    reciprocity_rows: list[dict[str, Any]] = []
    max_abs_reciprocity_z = 0.0
    for i in range(len(net_order)):
        for j in range(i + 1, len(net_order)):
            difference = float(matrix[i, j] - matrix[j, i])
            se_difference = float(math.hypot(se_matrix[i, j], se_matrix[j, i]))
            z = difference / se_difference if se_difference > 0.0 else 0.0
            max_abs_reciprocity_z = max(max_abs_reciprocity_z, abs(z))
            reciprocity_rows.append(
                {
                    "pair": [net_order[i], net_order[j]],
                    "Cij_F": float(matrix[i, j]),
                    "Cji_F": float(matrix[j, i]),
                    "difference_F": difference,
                    "difference_standard_error_F_conditional_on_H": se_difference,
                    "z_score_conditional_on_H": z,
                }
            )

    compact_output = {
        "input_file": str(args.input_file),
        "parameters": {
            "walks_per_master_net": args.walks,
            "scale_factor": args.scale_factor,
            "max_distance": args.max_distance,
            "max_hops": args.max_hops,
            "base_seed": args.seed,
            "E0": E0,
            "L_UNIT": L_UNIT,
            "standard_error_note": (
                "Per-coefficient SEs condition on the sampled H value. "
                "A later multi-seed test is needed to include H-estimation variation."
            ),
        },
        "net_mapping": {
            net.name: list(net.conductor_names)
            for net in nets
        },
        "matrix": {
            **matrix_result,
            "standard_error_matrix_F_conditional_on_H": se_matrix.tolist(),
            "ci95_half_width_matrix_F_conditional_on_H": (1.96 * se_matrix).tolist(),
            "infinity_standard_error_F_conditional_on_H": infinity_se_array.tolist(),
            "reciprocity_diagnostics": reciprocity_rows,
            "max_abs_reciprocity_z_score_conditional_on_H": max_abs_reciprocity_z,
        },
        "rows": compact_rows,
    }

    json_path = args.output_dir / f"{args.label}_summary.json"
    matrix_path = args.output_dir / f"{args.label}_matrix.csv"
    se_path = args.output_dir / f"{args.label}_standard_errors.csv"
    row_path = args.output_dir / f"{args.label}_row_diagnostics.csv"

    json_path.write_text(json.dumps(_to_builtin(compact_output), indent=2), encoding="utf-8")

    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["master_net", *net_order, INFINITY])
        for i, master in enumerate(net_order):
            writer.writerow([master, *matrix[i, :].tolist(), float(infinity_column[i])])

    with se_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["master_net", *net_order, INFINITY])
        for i, master in enumerate(net_order):
            writer.writerow([master, *se_matrix[i, :].tolist(), float(infinity_se_array[i])])

    with row_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "master_net",
            "H",
            "Sg",
            "mean_p",
            "vgss_acceptance_rate",
            "hit_conductor_count",
            "escaped_count",
            "max_hops_count",
            "row_sum_with_infinity_F",
            "row_sum_SE_F_conditional_on_H",
            "row_sum_z_conditional_on_H",
            "direct_diagonal_F",
            "conservation_diagonal_F",
            "direct_minus_conservation_diagonal_F",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in compact_rows:
            term = item["metrics"]["termination_counts"]
            cons = item["conservation"]
            writer.writerow(
                {
                    "master_net": item["master_net"],
                    "H": item["H"],
                    "Sg": item["Sg"],
                    "mean_p": item["mean_p"],
                    "vgss_acceptance_rate": item["sampling_stats"]["acceptance_rate"],
                    "hit_conductor_count": term["hit_conductor"],
                    "escaped_count": term["escaped"],
                    "max_hops_count": term["max_hops"],
                    "row_sum_with_infinity_F": cons["row_sum_with_infinity_F_from_walk_contributions"],
                    "row_sum_SE_F_conditional_on_H": cons["row_sum_standard_error_F_conditional_on_H"],
                    "row_sum_z_conditional_on_H": cons["row_sum_z_score_conditional_on_H"],
                    "direct_diagonal_F": cons["direct_diagonal_F"],
                    "conservation_diagonal_F": cons["diagonal_from_offdiagonals_and_infinity_F"],
                    "direct_minus_conservation_diagonal_F": cons["direct_minus_conservation_diagonal_F"],
                }
            )

    print("\n" + "#" * 72)
    print("FINAL MATRIX (F)")
    print("Net order:", net_order)
    print(np.array2string(matrix, precision=10, suppress_small=False))
    print("\nCONDITIONAL STANDARD-ERROR MATRIX (F)")
    print(np.array2string(se_matrix, precision=4, suppress_small=False))
    print("\nInfinity column (F):")
    print(np.array2string(infinity_column, precision=10, suppress_small=False))
    print("\nRelative asymmetry:", matrix_result["relative_asymmetry"])
    print("Max |reciprocity z| (conditional on H):", max_abs_reciprocity_z)
    print("\nCompact outputs:")
    print(" ", json_path)
    print(" ", matrix_path)
    print(" ", se_path)
    print(" ", row_path)


if __name__ == "__main__":
    main()
