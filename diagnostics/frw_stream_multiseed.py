#!/usr/bin/env python3
"""Low-memory, multi-seed FRW diagnostic.

This script mirrors the current FRW walk and capacitance estimator but never
stores individual paths. It is intended for numerical diagnosis only and does
not modify the VGSS/FRW source tree.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bgs.matlab_parser import parse_matlab_geometry
from bgs.sampling import VGSSSamplingStats, sample_on_vgss
from bgs.topology import find_touching_nets
from bgs.vgss import build_all_net_vgss
from FRW.capacitance import E0, L_UNIT
from FRW.random_walk import get_system_bounds, _load_default_gft


INFINITY = "Infinity"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run FRW with multiple independent seeds while retaining only "
            "compact coefficient statistics."
        )
    )
    p.add_argument("input_file", type=Path)
    p.add_argument("--walks-per-seed", type=int, default=20_000)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1234, 5678, 9012, 3456],
    )
    p.add_argument("--scale-factor", type=float, default=1.25)
    p.add_argument("--max-distance", type=float, default=10.0)
    p.add_argument("--max-hops", type=int, default=200)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output/diagnostics/frw_multiseed_summary.json"),
    )
    return p


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


def _contribution_se(
    n: int,
    sum_x: float,
    sum_x2: float,
    physical_scale: float,
) -> float:
    """SE of physical_scale * mean(X), where unobserved entries of X are zero."""
    if n <= 1:
        return 0.0
    centered_ss = sum_x2 - (sum_x * sum_x) / n
    sample_variance = max(centered_ss / (n - 1), 0.0)
    return abs(physical_scale) * math.sqrt(sample_variance / n)


def run_master_row(
    conductors,
    master_vgss,
    rng: np.random.Generator,
    *,
    net_order: list[str],
    conductor_to_net: dict[str, str],
    num_walks: int,
    max_hops: int,
    gft,
    system_center: np.ndarray,
    system_size: float,
) -> dict[str, Any]:
    labels = net_order + [INFINITY]
    sums = {label: 0.0 for label in labels}
    sums2 = {label: 0.0 for label in labels}
    counts = {label: 0 for label in labels}
    positive_counts = {label: 0 for label in labels}
    negative_counts = {label: 0 for label in labels}

    residual_sum = 0.0
    residual_sum2 = 0.0
    unresolved = 0

    sampling_stats = VGSSSamplingStats()
    kill_radius = system_size * 500_000.0
    progress_step = max(1, num_walks // 20)

    for walk_index in range(num_walks):
        if (
            (walk_index + 1) % progress_step == 0
            or walk_index + 1 == num_walks
        ):
            pct = 100.0 * (walk_index + 1) / num_walks
            print(
                f"\r  {master_vgss.net.name}: "
                f"{pct:5.1f}% ({walk_index + 1}/{num_walks})",
                end="",
                flush=True,
            )

        start_sample = sample_on_vgss(
            master_vgss.sampling_context,
            rng,
            stats=sampling_stats,
        )
        start_point = start_sample.point
        current_point = start_point.copy()

        distances = [
            conductor.distance_to_point(current_point)
            for conductor in conductors
        ]
        first_half_size = min(distances)

        current_point, _panel, _probability, omega = (
            gft.sample_first_transition(
                start_point,
                first_half_size,
                start_sample.normal,
                rng,
            )
        )

        target: str | None = None

        for _hop in range(max_hops):
            if (
                np.max(np.abs(current_point - system_center))
                > kill_radius
            ):
                target = INFINITY
                break

            distances = [
                conductor.distance_to_point(current_point)
                for conductor in conductors
            ]
            min_distance = min(distances)
            nearest_index = int(np.argmin(distances))

            if min_distance < 1.0e-4:
                conductor_name = conductors[nearest_index].name
                target = conductor_to_net[conductor_name]
                break

            current_point, _panel, _probability = gft.sample_next_point(
                current_point=current_point,
                cube_half_size=min_distance,
                rng=rng,
            )

        if target is None:
            unresolved += 1
            continue

        sums[target] += omega
        sums2[target] += omega * omega
        counts[target] += 1
        if omega > 0.0:
            positive_counts[target] += 1
        elif omega < 0.0:
            negative_counts[target] += 1

        residual_sum += omega
        residual_sum2 += omega * omega

    print()

    context = master_vgss.sampling_context
    H = sampling_stats.estimate_H(context)
    Sg = sampling_stats.estimate_surface_area(context)
    mean_p = sampling_stats.estimate_mean_p()
    physical_scale = E0 * L_UNIT * H

    estimates = {
        label: physical_scale * sums[label] / num_walks
        for label in labels
    }
    standard_errors = {
        label: _contribution_se(
            num_walks,
            sums[label],
            sums2[label],
            physical_scale,
        )
        for label in labels
    }

    row_residual = physical_scale * residual_sum / num_walks
    row_residual_se = _contribution_se(
        num_walks,
        residual_sum,
        residual_sum2,
        physical_scale,
    )

    omega_statistics = {}
    for label in labels:
        count = counts[label]
        omega_statistics[label] = {
            "count": count,
            "sum": sums[label],
            "sum_squares": sums2[label],
            "mean_on_target": sums[label] / count if count else 0.0,
            "positive_fraction_on_target": (
                positive_counts[label] / count if count else 0.0
            ),
            "negative_fraction_on_target": (
                negative_counts[label] / count if count else 0.0
            ),
        }

    return {
        "master_net": master_vgss.net.name,
        "num_walks": num_walks,
        "H": H,
        "Sg": Sg,
        "mean_p": mean_p,
        "sampling_stats": sampling_stats.as_dict(context),
        "estimates_F": estimates,
        "standard_errors_F": standard_errors,
        "omega_statistics": omega_statistics,
        "unresolved_walks": unresolved,
        "unresolved_rate": unresolved / num_walks,
        "row_sum_with_infinity_F": row_residual,
        "row_sum_standard_error_F": row_residual_se,
        "row_sum_z": (
            row_residual / row_residual_se
            if row_residual_se > 0.0
            else 0.0
        ),
    }


def main() -> None:
    args = parser().parse_args()

    if args.walks_per_seed <= 1:
        raise SystemExit("--walks-per-seed must be greater than 1.")
    if len(args.seeds) < 2:
        raise SystemExit("Provide at least two independent seeds.")
    if args.max_hops <= 0:
        raise SystemExit("--max-hops must be positive.")

    conductors = parse_matlab_geometry(args.input_file)
    nets = find_touching_nets(conductors)
    net_order = [net.name for net in nets]
    conductor_to_net = {
        conductor.name: net.name
        for net in nets
        for conductor in net.conductors
    }
    all_vgss = build_all_net_vgss(
        conductors,
        scale_factor=args.scale_factor,
        max_distance=args.max_distance,
    )

    system_center, system_size = get_system_bounds(conductors)
    gft = _load_default_gft()

    seed_results = []
    matrices = []
    infinity_columns = []
    within_se_matrices = []
    within_infinity_se = []

    print("Input:", args.input_file)
    print("Scale factor:", args.scale_factor)
    print("Walks per seed per master net:", args.walks_per_seed)
    print("Seeds:", args.seeds)
    print(
        "Total walks per master net:",
        args.walks_per_seed * len(args.seeds),
    )
    print(
        "Total walks over all master nets:",
        args.walks_per_seed * len(args.seeds) * len(net_order),
    )
    print("Net order:", net_order)

    for base_seed in args.seeds:
        print("\n" + "=" * 78)
        print(f"BASE SEED {base_seed}")
        print("=" * 78)

        seed_sequence = np.random.SeedSequence(base_seed)
        child_sequences = seed_sequence.spawn(len(all_vgss))

        rows = []
        matrix = np.zeros((len(net_order), len(net_order)), dtype=float)
        infinity_column = np.zeros(len(net_order), dtype=float)
        se_matrix = np.zeros_like(matrix)
        infinity_se = np.zeros(len(net_order), dtype=float)

        for row_index, master_vgss in enumerate(all_vgss):
            rng = np.random.default_rng(child_sequences[row_index])
            row = run_master_row(
                conductors,
                master_vgss,
                rng,
                net_order=net_order,
                conductor_to_net=conductor_to_net,
                num_walks=args.walks_per_seed,
                max_hops=args.max_hops,
                gft=gft,
                system_center=system_center,
                system_size=system_size,
            )
            rows.append(row)

            for column_index, target in enumerate(net_order):
                matrix[row_index, column_index] = (
                    row["estimates_F"][target]
                )
                se_matrix[row_index, column_index] = (
                    row["standard_errors_F"][target]
                )

            infinity_column[row_index] = row["estimates_F"][INFINITY]
            infinity_se[row_index] = row["standard_errors_F"][INFINITY]

        seed_results.append(
            {
                "base_seed": base_seed,
                "rows": rows,
                "matrix_F": matrix,
                "infinity_column_F": infinity_column,
                "within_seed_standard_error_matrix_F": se_matrix,
                "within_seed_infinity_standard_error_F": infinity_se,
            }
        )
        matrices.append(matrix)
        infinity_columns.append(infinity_column)
        within_se_matrices.append(se_matrix)
        within_infinity_se.append(infinity_se)

        print("Matrix for this seed (F):")
        print(np.array2string(matrix, precision=10, suppress_small=False))

    matrix_stack = np.stack(matrices)
    infinity_stack = np.stack(infinity_columns)
    within_se_stack = np.stack(within_se_matrices)
    within_infinity_stack = np.stack(within_infinity_se)

    seed_count = len(args.seeds)
    mean_matrix = matrix_stack.mean(axis=0)
    mean_infinity = infinity_stack.mean(axis=0)

    between_seed_sd_matrix = matrix_stack.std(axis=0, ddof=1)
    between_seed_se_matrix = between_seed_sd_matrix / math.sqrt(seed_count)
    between_seed_sd_infinity = infinity_stack.std(axis=0, ddof=1)
    between_seed_se_infinity = (
        between_seed_sd_infinity / math.sqrt(seed_count)
    )

    # Expected SE of the mean from the within-seed estimates.
    combined_within_se_matrix = (
        np.sqrt(np.sum(within_se_stack**2, axis=0)) / seed_count
    )
    combined_within_se_infinity = (
        np.sqrt(np.sum(within_infinity_stack**2, axis=0)) / seed_count
    )

    asymmetry = mean_matrix - mean_matrix.T
    max_abs_entry = float(np.max(np.abs(mean_matrix)))
    max_abs_asymmetry = float(np.max(np.abs(asymmetry)))

    aggregate = {
        "mean_matrix_F": mean_matrix,
        "mean_infinity_column_F": mean_infinity,
        "between_seed_standard_deviation_matrix_F": (
            between_seed_sd_matrix
        ),
        "between_seed_standard_error_of_mean_matrix_F": (
            between_seed_se_matrix
        ),
        "combined_within_seed_standard_error_of_mean_matrix_F": (
            combined_within_se_matrix
        ),
        "between_seed_standard_deviation_infinity_F": (
            between_seed_sd_infinity
        ),
        "between_seed_standard_error_of_mean_infinity_F": (
            between_seed_se_infinity
        ),
        "combined_within_seed_standard_error_of_mean_infinity_F": (
            combined_within_se_infinity
        ),
        "max_abs_asymmetry_F": max_abs_asymmetry,
        "relative_asymmetry": (
            max_abs_asymmetry / max_abs_entry
            if max_abs_entry > 0.0
            else 0.0
        ),
    }

    output = {
        "input_file": str(args.input_file),
        "parameters": {
            "walks_per_seed_per_master_net": args.walks_per_seed,
            "seeds": args.seeds,
            "total_walks_per_master_net": (
                args.walks_per_seed * seed_count
            ),
            "scale_factor": args.scale_factor,
            "max_distance": args.max_distance,
            "max_hops": args.max_hops,
            "E0": E0,
            "L_UNIT": L_UNIT,
            "gft_resolution": gft.resolution,
        },
        "net_order": net_order,
        "seed_results": seed_results,
        "aggregate": aggregate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_to_builtin(output), indent=2),
        encoding="utf-8",
    )

    print("\n" + "#" * 78)
    print("MULTI-SEED MEAN MATRIX (F)")
    print("#" * 78)
    print(np.array2string(mean_matrix, precision=10, suppress_small=False))
    print("\nBetween-seed SE of mean (F):")
    print(
        np.array2string(
            between_seed_se_matrix,
            precision=4,
            suppress_small=False,
        )
    )
    print("\nCombined within-seed SE of mean (F):")
    print(
        np.array2string(
            combined_within_se_matrix,
            precision=4,
            suppress_small=False,
        )
    )
    print("\nMean infinity column (F):")
    print(np.array2string(mean_infinity, precision=10, suppress_small=False))
    print("\nRelative asymmetry:", aggregate["relative_asymmetry"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
