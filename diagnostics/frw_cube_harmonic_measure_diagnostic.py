#!/usr/bin/env python3
"""Validate the complete repeated cube-walk against exact cube harmonic measure.

This standalone diagnostic isolates the continuation part of the FRW algorithm.
It does not use conductor geometry, BGS construction, capacitance assembly, or an
external capacitance solver.

For several source points inside the outer cube [-1, 1]^3 it compares:

1. ``numerical_n200``
   Repeated maximal-cube transitions using the bundled numerical GGFT table.

2. ``analytical_n200``
   The same repeated walk, panel-centre mapping, hit tolerance, and boundary-face
   classification, but with an independently generated analytical panel-integrated
   transition distribution.

The exact probability of ultimately hitting each outer-cube face is calculated by
an independent separation-of-variables series.  Consequently the test covers:

* repeated transition sampling;
* maximal-cube sizing inside a bounded domain;
* panel-centre propagation over many hops;
* terminal face classification for discontinuous indicator boundary data;
* accumulation of numerical GGFT probability errors over a complete path.

No production source files are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FACE_NAMES = ("PX", "NX", "PY", "NY", "PZ", "NZ")
SIDE_LENGTH = 2.0
DEFAULT_STARTS = {
    "center": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    "off_axis": np.array([0.35, -0.20, 0.15], dtype=np.float64),
    "near_face": np.array([0.75, 0.10, -0.15], dtype=np.float64),
    "near_edge": np.array([0.65, 0.65, -0.10], dtype=np.float64),
    "near_corner": np.array([0.55, 0.55, 0.55], dtype=np.float64),
}


@dataclass(frozen=True)
class TransitionTable:
    name: str
    resolution: int
    probabilities: np.ndarray
    cdf: np.ndarray
    panel_centers: np.ndarray


@dataclass
class RunningCategorical:
    count: int = 0
    face_counts: np.ndarray | None = None
    hop_sum: float = 0.0
    max_hops_count: int = 0

    def __post_init__(self) -> None:
        if self.face_counts is None:
            self.face_counts = np.zeros(6, dtype=np.int64)

    def add(self, face_index: int | None, hops: int) -> None:
        self.count += 1
        self.hop_sum += float(hops)
        if face_index is None:
            self.max_hops_count += 1
        else:
            self.face_counts[face_index] += 1


@dataclass
class RunningPaired:
    count: int = 0
    delta_sum: np.ndarray | None = None
    delta_sq_sum: np.ndarray | None = None
    agreement_count: int = 0

    def __post_init__(self) -> None:
        if self.delta_sum is None:
            self.delta_sum = np.zeros(6, dtype=np.float64)
        if self.delta_sq_sum is None:
            self.delta_sq_sum = np.zeros(6, dtype=np.float64)

    def add(self, first_face: int | None, second_face: int | None) -> None:
        self.count += 1
        if first_face == second_face:
            self.agreement_count += 1

        first = np.zeros(6, dtype=np.float64)
        second = np.zeros(6, dtype=np.float64)
        if first_face is not None:
            first[first_face] = 1.0
        if second_face is not None:
            second[second_face] = 1.0
        delta = second - first
        self.delta_sum += delta
        self.delta_sq_sum += delta * delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        default="src/GGFT/data/cube_tables_n201.npz",
        help="Numerical GGFT .npz table.",
    )
    parser.add_argument(
        "--walks-per-seed", type=int, default=20000,
        help="Paired walks per source point and seed.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1234, 5678, 9012, 3456],
    )
    parser.add_argument(
        "--series-terms", type=int, default=41,
        help="Terms used to build the analytical transition table.",
    )
    parser.add_argument(
        "--reference-terms", type=int, default=201,
        help="Odd/even mode cutoff for exact outer-cube face probabilities.",
    )
    parser.add_argument("--hit-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-hops", type=int, default=1000)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=tuple(DEFAULT_STARTS),
        default=list(DEFAULT_STARTS),
    )
    parser.add_argument(
        "--output-dir",
        default="output/diagnostics/frw_cube_harmonic_measure",
    )
    parser.add_argument("--label", default="cube_harmonic_measure")
    return parser.parse_args()


def normalized_cdf(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("Transition probabilities must be finite and non-negative.")
    total = float(p.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Transition probability sum must be positive.")
    p = p / total
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    return p, cdf


def load_numerical_table(path: str | Path) -> TransitionTable:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GGFT table not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        resolution = int(data["resolution"])
        probabilities = np.asarray(data["probabilities"], dtype=np.float64)
        centers = np.asarray(data["panel_centers"], dtype=np.float64).reshape(-1, 3)
    p, cdf = normalized_cdf(probabilities)
    expected = 6 * resolution * resolution
    if p.shape != (expected,) or centers.shape != (expected, 3):
        raise ValueError("Unexpected numerical GGFT table dimensions.")
    return TransitionTable(
        name=f"numerical_n{resolution}",
        resolution=resolution,
        probabilities=p,
        cdf=cdf,
        panel_centers=centers,
    )


def build_panel_centers(resolution: int) -> np.ndarray:
    axis = -1.0 + (np.arange(resolution, dtype=np.float64) + 0.5) * (2.0 / resolution)
    centers: list[tuple[float, float, float]] = []
    # Ordering matches the project table: z-, z+, y-, y+, x-, x+.
    centers.extend((x, y, -1.0) for y in axis for x in axis)
    centers.extend((x, y, 1.0) for y in axis for x in axis)
    centers.extend((x, -1.0, z) for z in axis for x in axis)
    centers.extend((x, 1.0, z) for z in axis for x in axis)
    centers.extend((-1.0, y, z) for z in axis for y in axis)
    centers.extend((1.0, y, z) for z in axis for y in axis)
    return np.asarray(centers, dtype=np.float64)


def integrated_sine_basis(resolution: int, terms: int) -> np.ndarray:
    edges = np.linspace(0.0, SIDE_LENGTH, resolution + 1, dtype=np.float64)
    modes = np.arange(1, terms + 1, dtype=np.float64)
    k = modes * np.pi / SIDE_LENGTH
    return (np.cos(edges[:-1, None] * k) - np.cos(edges[1:, None] * k)) / k


def stable_center_normal_value_factor(terms: int) -> np.ndarray:
    modes = np.arange(1, terms + 1, dtype=np.float64)
    k = modes * np.pi / SIDE_LENGTH
    lam = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    a = lam * SIDE_LENGTH / 2.0
    exp_minus_a = np.exp(-a)
    return exp_minus_a / (1.0 + np.exp(-2.0 * a))


def build_analytical_center_table(resolution: int, terms: int) -> TransitionTable:
    if terms < 1:
        raise ValueError("--series-terms must be positive.")
    basis = integrated_sine_basis(resolution, terms)
    modes = np.arange(1, terms + 1, dtype=np.int64)
    source_sine = np.zeros(terms, dtype=np.float64)
    odd = modes % 2 == 1
    source_sine[odd] = (-1.0) ** ((modes[odd] - 1) // 2)
    value_factor = stable_center_normal_value_factor(terms)
    coefficient = (
        4.0 / SIDE_LENGTH**2
        * source_sine[:, None]
        * source_sine[None, :]
        * value_factor
    )
    face_grid = basis @ coefficient.T @ basis.T
    probability_faces = [face_grid.ravel() for _ in range(6)]
    probabilities = np.concatenate(probability_faces)
    if np.any(probabilities <= 0.0):
        count = int(np.count_nonzero(probabilities <= 0.0))
        raise RuntimeError(
            f"Analytical transition series produced {count} non-positive panels."
        )
    p, cdf = normalized_cdf(probabilities)
    return TransitionTable(
        name=f"analytical_n{resolution}_t{terms}",
        resolution=resolution,
        probabilities=p,
        cdf=cdf,
        panel_centers=build_panel_centers(resolution),
    )


def stable_sinh_ratio(lam: np.ndarray, a: float) -> np.ndarray:
    """Return sinh(lam*a)/sinh(lam*L) without overflow."""
    if not 0.0 <= a <= SIDE_LENGTH:
        raise ValueError("Normal coordinate must lie inside [0, L].")
    numerator = -np.expm1(-2.0 * lam * a)
    denominator = -np.expm1(-2.0 * lam * SIDE_LENGTH)
    return np.exp(-lam * (SIDE_LENGTH - a)) * numerator / denominator


def exact_outer_face_probabilities(point: np.ndarray, terms: int) -> np.ndarray:
    """Exact harmonic measures of the six faces of [-1,1]^3.

    The boundary condition is one on the requested face and zero on the other
    five faces.  The result is evaluated with a separation-of-variables series.
    """
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or np.any(np.abs(point) >= 1.0):
        raise ValueError("Source point must lie strictly inside [-1,1]^3.")

    u, v, w = point + 1.0
    modes_all = np.arange(1, terms + 1, dtype=np.int64)
    modes = modes_all[modes_all % 2 == 1].astype(np.float64)
    k = modes * np.pi / SIDE_LENGTH
    lam = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    integrated_prefactor = 16.0 / (
        np.pi**2 * modes[:, None] * modes[None, :]
    )

    def one_face(t1: float, t2: float, normal_coordinate: float) -> float:
        s1 = np.sin(k * t1)
        s2 = np.sin(k * t2)
        values = (
            integrated_prefactor
            * s1[:, None]
            * s2[None, :]
            * stable_sinh_ratio(lam, normal_coordinate)
        )
        return float(values.sum())

    result = np.array(
        [
            one_face(v, w, u),                    # PX
            one_face(v, w, SIDE_LENGTH - u),      # NX
            one_face(u, w, v),                    # PY
            one_face(u, w, SIDE_LENGTH - v),      # NY
            one_face(u, v, w),                    # PZ
            one_face(u, v, SIDE_LENGTH - w),      # NZ
        ],
        dtype=np.float64,
    )
    # Tiny truncation noise is harmless, but a material violation is not.
    if np.any(result < -1e-12):
        raise RuntimeError("Exact face series produced a negative probability.")
    result = np.maximum(result, 0.0)
    total = float(result.sum())
    if abs(total - 1.0) > 1e-9:
        raise RuntimeError(
            f"Exact face probabilities do not sum to one: {total:.16g}."
        )
    return result / total


def sample_index(cdf: np.ndarray, u: float) -> int:
    return int(np.searchsorted(cdf, u, side="right"))


@dataclass(frozen=True)
class MaximalCoupling:
    common_mass: float
    common_cdf: np.ndarray
    first_residual_cdf: np.ndarray
    second_residual_cdf: np.ndarray


def build_maximal_coupling(first: np.ndarray, second: np.ndarray) -> MaximalCoupling:
    common = np.minimum(first, second)
    common_mass = float(common.sum())

    def cdf_or_placeholder(values: np.ndarray) -> np.ndarray:
        total = float(values.sum())
        if total <= 0.0:
            placeholder = np.zeros_like(values)
            placeholder[-1] = 1.0
            return placeholder
        cdf = np.cumsum(values / total)
        cdf[-1] = 1.0
        return cdf

    return MaximalCoupling(
        common_mass=common_mass,
        common_cdf=cdf_or_placeholder(common),
        first_residual_cdf=cdf_or_placeholder(first - common),
        second_residual_cdf=cdf_or_placeholder(second - common),
    )


def coupled_indices(
    coupling: MaximalCoupling,
    rng: np.random.Generator,
) -> tuple[int, int]:
    branch_u, first_u, second_u = rng.random(3)
    if branch_u < coupling.common_mass:
        index = sample_index(coupling.common_cdf, float(first_u))
        return index, index
    return (
        sample_index(coupling.first_residual_cdf, float(first_u)),
        sample_index(coupling.second_residual_cdf, float(second_u)),
    )


def boundary_distances(point: np.ndarray) -> np.ndarray:
    x, y, z = point
    return np.array(
        [1.0 - x, x + 1.0, 1.0 - y, y + 1.0, 1.0 - z, z + 1.0],
        dtype=np.float64,
    )


def classify_boundary(point: np.ndarray, tolerance: float) -> int | None:
    distances = boundary_distances(point)
    if float(distances.min()) <= tolerance:
        return int(np.argmin(distances))
    return None


def simulate_paired_walk(
    start: np.ndarray,
    numerical: TransitionTable,
    analytical: TransitionTable,
    coupling: MaximalCoupling,
    rng: np.random.Generator,
    *,
    hit_tolerance: float,
    max_hops: int,
) -> tuple[int | None, int | None, int, int, float]:
    points = [np.asarray(start, dtype=np.float64).copy(), np.asarray(start, dtype=np.float64).copy()]
    tables = [numerical, analytical]
    results: list[int | None] = [None, None]
    done = [False, False]
    hops = [0, 0]
    max_overshoot = 0.0

    for _ in range(max_hops):
        for mode in range(2):
            if done[mode]:
                continue
            face = classify_boundary(points[mode], hit_tolerance)
            if face is not None:
                results[mode] = face
                done[mode] = True

        if all(done):
            return results[0], results[1], hops[0], hops[1], max_overshoot

        if not done[0] and not done[1]:
            indices = coupled_indices(coupling, rng)
        else:
            # Once paths have different termination times, the remaining mode
            # samples from its own exact marginal distribution.
            u = float(rng.random())
            indices = (
                sample_index(numerical.cdf, u),
                sample_index(analytical.cdf, u),
            )

        for mode in range(2):
            if done[mode]:
                continue
            distances = boundary_distances(points[mode])
            half_size = float(distances.min())
            if half_size <= 0.0:
                results[mode] = int(np.argmin(distances))
                done[mode] = True
                continue
            points[mode] = points[mode] + half_size * tables[mode].panel_centers[indices[mode]]
            hops[mode] += 1
            overshoot = float(np.max(np.maximum(np.abs(points[mode]) - 1.0, 0.0)))
            max_overshoot = max(max_overshoot, overshoot)
            if overshoot > 1e-11:
                raise RuntimeError(f"Walk left the outer cube by {overshoot:.6e}.")
            points[mode] = np.clip(points[mode], -1.0, 1.0)

    return results[0], results[1], hops[0], hops[1], max_overshoot


def estimate_and_se(counts: np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray]:
    p = counts.astype(np.float64) / total
    se = np.sqrt(np.maximum(p * (1.0 - p), 0.0) / total)
    return p, se


def paired_mean_and_se(stats: RunningPaired) -> tuple[np.ndarray, np.ndarray]:
    mean = stats.delta_sum / stats.count
    if stats.count <= 1:
        return mean, np.zeros_like(mean)
    second = stats.delta_sq_sum / stats.count
    variance = np.maximum(second - mean * mean, 0.0)
    sample_variance = variance * stats.count / (stats.count - 1)
    return mean, np.sqrt(sample_variance / stats.count)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict], fieldnames: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.walks_per_seed <= 0:
        raise ValueError("--walks-per-seed must be positive.")
    if args.max_hops <= 0:
        raise ValueError("--max-hops must be positive.")
    if args.hit_tolerance <= 0.0:
        raise ValueError("--hit-tolerance must be positive.")

    numerical = load_numerical_table(args.table)
    analytical = build_analytical_center_table(
        numerical.resolution, args.series_terms
    )
    alignment_error = float(
        np.max(np.abs(numerical.panel_centers - analytical.panel_centers))
    )
    if alignment_error > 1e-13:
        raise RuntimeError(
            f"Numerical/analytical panel ordering differs: {alignment_error:.6e}."
        )
    coupling = build_maximal_coupling(
        numerical.probabilities, analytical.probabilities
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Numerical table: {numerical.name}")
    print(f"Analytical table: {analytical.name}")
    print(f"Common panel probability: {coupling.common_mass:.9f}")
    print(f"Sources: {', '.join(args.sources)}")
    print(
        "Walks per source:",
        args.walks_per_seed * len(args.seeds),
        f"({args.walks_per_seed} x {len(args.seeds)} seeds)",
    )

    exact_reference: dict[str, np.ndarray] = {}
    reference_convergence: dict[str, float] = {}
    lower_terms = max(21, args.reference_terms // 2)
    if lower_terms % 2 == 0:
        lower_terms += 1
    for source_name in args.sources:
        point = DEFAULT_STARTS[source_name]
        exact = exact_outer_face_probabilities(point, args.reference_terms)
        lower = exact_outer_face_probabilities(point, lower_terms)
        exact_reference[source_name] = exact
        reference_convergence[source_name] = float(np.max(np.abs(exact - lower)))

    aggregate: dict[str, dict[str, RunningCategorical | RunningPaired]] = {}
    seed_results: list[dict] = []
    global_max_overshoot = 0.0

    for source_index, source_name in enumerate(args.sources):
        aggregate[source_name] = {
            numerical.name: RunningCategorical(),
            analytical.name: RunningCategorical(),
            "paired": RunningPaired(),
        }
        start = DEFAULT_STARTS[source_name]
        for seed in args.seeds:
            mixed_seed = int(seed + 1_000_003 * source_index)
            rng = np.random.default_rng(mixed_seed)
            local_num = RunningCategorical()
            local_ref = RunningCategorical()
            local_pair = RunningPaired()
            local_max_overshoot = 0.0

            for walk_index in range(args.walks_per_seed):
                face_num, face_ref, hops_num, hops_ref, overshoot = simulate_paired_walk(
                    start,
                    numerical,
                    analytical,
                    coupling,
                    rng,
                    hit_tolerance=args.hit_tolerance,
                    max_hops=args.max_hops,
                )
                local_num.add(face_num, hops_num)
                local_ref.add(face_ref, hops_ref)
                local_pair.add(face_num, face_ref)
                aggregate[source_name][numerical.name].add(face_num, hops_num)
                aggregate[source_name][analytical.name].add(face_ref, hops_ref)
                aggregate[source_name]["paired"].add(face_num, face_ref)
                local_max_overshoot = max(local_max_overshoot, overshoot)

                if (walk_index + 1) % max(1, args.walks_per_seed // 10) == 0:
                    print(
                        f"\r{source_name} seed={seed}: "
                        f"{walk_index + 1}/{args.walks_per_seed}",
                        end="",
                        flush=True,
                    )
            print()
            global_max_overshoot = max(global_max_overshoot, local_max_overshoot)

            p_num, se_num = estimate_and_se(local_num.face_counts, local_num.count)
            p_ref, se_ref = estimate_and_se(local_ref.face_counts, local_ref.count)
            delta, delta_se = paired_mean_and_se(local_pair)
            seed_results.append(
                {
                    "source": source_name,
                    "point": start,
                    "seed": seed,
                    "walk_count": args.walks_per_seed,
                    "numerical": {
                        "probabilities": p_num,
                        "standard_errors": se_num,
                        "mean_hops": local_num.hop_sum / local_num.count,
                        "max_hops_count": local_num.max_hops_count,
                    },
                    "analytical": {
                        "probabilities": p_ref,
                        "standard_errors": se_ref,
                        "mean_hops": local_ref.hop_sum / local_ref.count,
                        "max_hops_count": local_ref.max_hops_count,
                    },
                    "paired": {
                        "analytical_minus_numerical": delta,
                        "standard_errors": delta_se,
                        "target_agreement_rate": (
                            local_pair.agreement_count / local_pair.count
                        ),
                    },
                    "max_boundary_overshoot": local_max_overshoot,
                }
            )

    entry_rows: list[dict] = []
    start_rows: list[dict] = []
    paired_rows: list[dict] = []
    aggregate_json: dict[str, dict] = {}

    for source_name in args.sources:
        exact = exact_reference[source_name]
        source_json: dict[str, dict] = {}
        for mode_name in (numerical.name, analytical.name):
            stats = aggregate[source_name][mode_name]
            assert isinstance(stats, RunningCategorical)
            estimates, ses = estimate_and_se(stats.face_counts, stats.count)
            deltas = estimates - exact
            z = np.divide(
                deltas,
                ses,
                out=np.zeros_like(deltas),
                where=ses > 0.0,
            )
            l1 = float(np.abs(deltas).sum())
            tv = 0.5 * l1
            max_abs = float(np.max(np.abs(deltas)))
            weighted_chi2 = float(
                np.sum((deltas * deltas) / np.maximum(ses * ses, 1e-300))
            )
            source_json[mode_name] = {
                "probabilities": estimates,
                "standard_errors": ses,
                "exact_probabilities": exact,
                "errors": deltas,
                "z_scores": z,
                "l1_error": l1,
                "total_variation_error": tv,
                "max_absolute_error": max_abs,
                "chi_square_like_sum_z2": weighted_chi2,
                "mean_hops": stats.hop_sum / stats.count,
                "resolved_rate": 1.0 - stats.max_hops_count / stats.count,
                "max_hops_count": stats.max_hops_count,
            }
            for face_index, face_name in enumerate(FACE_NAMES):
                entry_rows.append(
                    {
                        "source": source_name,
                        "x": DEFAULT_STARTS[source_name][0],
                        "y": DEFAULT_STARTS[source_name][1],
                        "z": DEFAULT_STARTS[source_name][2],
                        "mode": mode_name,
                        "face": face_name,
                        "estimate": estimates[face_index],
                        "standard_error": ses[face_index],
                        "exact_probability": exact[face_index],
                        "error": deltas[face_index],
                        "relative_error": (
                            deltas[face_index] / exact[face_index]
                            if exact[face_index] != 0.0 else math.nan
                        ),
                        "z_score": z[face_index],
                        "walk_count": stats.count,
                        "mean_hops": stats.hop_sum / stats.count,
                        "resolved_rate": 1.0 - stats.max_hops_count / stats.count,
                    }
                )
            start_rows.append(
                {
                    "source": source_name,
                    "mode": mode_name,
                    "walk_count": stats.count,
                    "mean_hops": stats.hop_sum / stats.count,
                    "resolved_rate": 1.0 - stats.max_hops_count / stats.count,
                    "l1_error": l1,
                    "total_variation_error": tv,
                    "max_absolute_error": max_abs,
                    "chi_square_like_sum_z2": weighted_chi2,
                    "reference_series_max_change": reference_convergence[source_name],
                }
            )

        paired_stats = aggregate[source_name]["paired"]
        assert isinstance(paired_stats, RunningPaired)
        delta, delta_se = paired_mean_and_se(paired_stats)
        delta_z = np.divide(
            delta,
            delta_se,
            out=np.zeros_like(delta),
            where=delta_se > 0.0,
        )
        source_json["paired"] = {
            "analytical_minus_numerical": delta,
            "standard_errors": delta_se,
            "z_scores": delta_z,
            "target_agreement_rate": (
                paired_stats.agreement_count / paired_stats.count
            ),
        }
        for face_index, face_name in enumerate(FACE_NAMES):
            paired_rows.append(
                {
                    "source": source_name,
                    "face": face_name,
                    "comparison": "analytical_minus_numerical",
                    "mean_delta": delta[face_index],
                    "paired_standard_error": delta_se[face_index],
                    "paired_z": delta_z[face_index],
                    "target_agreement_rate": (
                        paired_stats.agreement_count / paired_stats.count
                    ),
                    "walk_count": paired_stats.count,
                }
            )
        aggregate_json[source_name] = source_json

    summary = {
        "description": "Full repeated cube-walk validation against exact outer-cube harmonic measure.",
        "parameters": {
            "table": str(args.table),
            "resolution": numerical.resolution,
            "series_terms": args.series_terms,
            "reference_terms": args.reference_terms,
            "walks_per_seed": args.walks_per_seed,
            "seeds": args.seeds,
            "walks_per_source": args.walks_per_seed * len(args.seeds),
            "hit_tolerance": args.hit_tolerance,
            "max_hops": args.max_hops,
            "sources": args.sources,
        },
        "face_order": FACE_NAMES,
        "source_points": {name: DEFAULT_STARTS[name] for name in args.sources},
        "table_comparison": {
            "numerical_name": numerical.name,
            "analytical_name": analytical.name,
            "probability_l1_distance": float(
                np.abs(numerical.probabilities - analytical.probabilities).sum()
            ),
            "total_variation_distance": float(
                0.5 * np.abs(numerical.probabilities - analytical.probabilities).sum()
            ),
            "maximal_coupling_common_panel_probability": coupling.common_mass,
            "panel_center_max_absolute_difference": alignment_error,
        },
        "exact_reference_probabilities": exact_reference,
        "reference_series_convergence_max_change": reference_convergence,
        "aggregate": aggregate_json,
        "seed_results": seed_results,
        "global_max_boundary_overshoot": global_max_overshoot,
    }

    base = output_dir / args.label
    summary_path = base.with_name(base.name + "_summary.json")
    entries_path = base.with_name(base.name + "_entries.csv")
    starts_path = base.with_name(base.name + "_start_summary.csv")
    paired_path = base.with_name(base.name + "_paired_differences.csv")

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2)

    write_csv(
        entries_path,
        entry_rows,
        [
            "source", "x", "y", "z", "mode", "face", "estimate",
            "standard_error", "exact_probability", "error", "relative_error",
            "z_score", "walk_count", "mean_hops", "resolved_rate",
        ],
    )
    write_csv(
        starts_path,
        start_rows,
        [
            "source", "mode", "walk_count", "mean_hops", "resolved_rate",
            "l1_error", "total_variation_error", "max_absolute_error",
            "chi_square_like_sum_z2", "reference_series_max_change",
        ],
    )
    write_csv(
        paired_path,
        paired_rows,
        [
            "source", "face", "comparison", "mean_delta",
            "paired_standard_error", "paired_z", "target_agreement_rate",
            "walk_count",
        ],
    )

    print("\nAggregate summary")
    for source_name in args.sources:
        exact = exact_reference[source_name]
        print(f"  {source_name} exact: " + " ".join(f"{v:.6f}" for v in exact))
        for mode_name in (numerical.name, analytical.name):
            mode = aggregate_json[source_name][mode_name]
            print(
                f"    {mode_name}: TV={mode['total_variation_error']:.6e}, "
                f"max_abs={mode['max_absolute_error']:.6e}, "
                f"mean_hops={mode['mean_hops']:.3f}, "
                f"resolved={mode['resolved_rate']:.6f}"
            )
        paired = aggregate_json[source_name]["paired"]
        print(
            f"    target agreement={paired['target_agreement_rate']:.6f}"
        )

    print("\nWrote:")
    for path in (summary_path, entries_path, starts_path, paired_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
