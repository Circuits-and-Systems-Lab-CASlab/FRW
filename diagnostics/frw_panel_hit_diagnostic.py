#!/usr/bin/env python3
"""Low-memory FRW panel-centre and hit-tolerance diagnostic.

This is a local diagnostic, not production code. It isolates two remaining
numerical approximations while leaving the VGSS construction, conductor
geometry, capacitance normalization, and analytical cube-transition law fixed:

1. Panel-centre resolution
   Runs analytically generated cube-transition tables at several panel
   resolutions (default: 100, 200, 400) with a common spatially coupled random
   sample at every transition. The first-hop weight and all later transitions
   use the analytical table at the corresponding resolution.

2. Conductor hit tolerance
   Reuses the same analytical n=200 path and records where it would terminate
   for several distance thresholds (default: 1e-3, 1e-4, 1e-5, 1e-6).

No walk paths are stored. The script writes compact JSON/CSV summaries and does
not modify the production FRW, BGS, or GGFT source files.

Intended location inside the VGSS checkout:

    diagnostics/frw_panel_hit_diagnostic.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from bgs.matlab_parser import parse_matlab_geometry
from bgs.sampling import VGSSSamplingStats, sample_on_vgss
from bgs.topology import find_touching_nets
from bgs.vgss import build_all_net_vgss
from FRW.capacitance import E0, L_UNIT
from FRW.random_walk import get_system_bounds


FloatArray = NDArray[np.float64]
INFINITY = "Infinity"
REFERENCE_SIDE_LENGTH = 2.0


@dataclass(frozen=True)
class AnalyticalGridTable:
    """Compact analytical cube table exploiting six-face symmetry."""

    resolution: int
    axis_centers: FloatArray
    row_cdf: FloatArray
    conditional_column_cdf: FloatArray
    wvt_tangent_first: FloatArray
    wvt_tangent_second: FloatArray
    wvt_normal: FloatArray
    probability_sum_before_normalization: float

    @staticmethod
    def _sample_cdf(cdf: FloatArray, u: float) -> int:
        index = int(np.searchsorted(cdf, u, side="right"))
        return min(index, len(cdf) - 1)

    def sample_panel(
        self, u_face: float, u_row: float, u_column: float
    ) -> tuple[int, int, int]:
        face = min(int(u_face * 6.0), 5)
        row = self._sample_cdf(self.row_cdf, u_row)
        column = self._sample_cdf(
            self.conditional_column_cdf[row], u_column
        )
        return face, row, column

    def relative_point(self, face: int, row: int, column: int) -> FloatArray:
        first = float(self.axis_centers[column])
        second = float(self.axis_centers[row])
        if face == 0:  # z-
            return np.array([first, second, -1.0], dtype=np.float64)
        if face == 1:  # z+
            return np.array([first, second, 1.0], dtype=np.float64)
        if face == 2:  # y-
            return np.array([first, -1.0, second], dtype=np.float64)
        if face == 3:  # y+
            return np.array([first, 1.0, second], dtype=np.float64)
        if face == 4:  # x-
            return np.array([-1.0, first, second], dtype=np.float64)
        if face == 5:  # x+
            return np.array([1.0, first, second], dtype=np.float64)
        raise ValueError(f"Invalid cube face index: {face}")

    def wvt_vector(self, face: int, row: int, column: int) -> FloatArray:
        tangent_first = float(self.wvt_tangent_first[row, column])
        tangent_second = float(self.wvt_tangent_second[row, column])
        normal = float(self.wvt_normal[row, column])
        if face == 0:
            return np.array(
                [tangent_first, tangent_second, -normal], dtype=np.float64
            )
        if face == 1:
            return np.array(
                [tangent_first, tangent_second, normal], dtype=np.float64
            )
        if face == 2:
            return np.array(
                [tangent_first, -normal, tangent_second], dtype=np.float64
            )
        if face == 3:
            return np.array(
                [tangent_first, normal, tangent_second], dtype=np.float64
            )
        if face == 4:
            return np.array(
                [-normal, tangent_first, tangent_second], dtype=np.float64
            )
        if face == 5:
            return np.array(
                [normal, tangent_first, tangent_second], dtype=np.float64
            )
        raise ValueError(f"Invalid cube face index: {face}")


@dataclass
class ScalarMoments:
    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_squares += value * value

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def sample_variance(self) -> float:
        if self.count <= 1:
            return 0.0
        centered = self.total_squares - self.total * self.total / self.count
        return max(centered / (self.count - 1), 0.0)

    def standard_error_of_mean(self) -> float:
        if self.count <= 1:
            return 0.0
        return math.sqrt(self.sample_variance() / self.count)


@dataclass(frozen=True)
class PathResult:
    target: str | None
    termination: str
    omega: float
    hops: int


@dataclass
class ModeAccumulator:
    labels: tuple[str, ...]
    contributions: dict[str, ScalarMoments] = field(init=False)
    row_sum: ScalarMoments = field(default_factory=ScalarMoments)
    termination_counts: dict[str, int] = field(
        default_factory=lambda: {
            "hit_conductor": 0,
            "escaped": 0,
            "max_hops": 0,
        }
    )
    target_counts: dict[str, int] = field(init=False)
    hop_sum: int = 0
    max_hops_observed: int = 0

    def __post_init__(self) -> None:
        self.contributions = {
            label: ScalarMoments() for label in self.labels
        }
        self.target_counts = {label: 0 for label in self.labels}

    def add(self, result: PathResult) -> None:
        for label in self.labels:
            self.contributions[label].add(
                result.omega if result.target == label else 0.0
            )
        self.row_sum.add(result.omega if result.target is not None else 0.0)
        self.termination_counts[result.termination] += 1
        if result.target is not None:
            self.target_counts[result.target] += 1
        self.hop_sum += result.hops
        self.max_hops_observed = max(self.max_hops_observed, result.hops)


@dataclass
class DifferenceAccumulator:
    labels: tuple[str, ...]
    contributions: dict[str, ScalarMoments] = field(init=False)
    row_sum: ScalarMoments = field(default_factory=ScalarMoments)
    target_agreements: int = 0
    count: int = 0

    def __post_init__(self) -> None:
        self.contributions = {
            label: ScalarMoments() for label in self.labels
        }

    def add(self, first: PathResult, second: PathResult) -> None:
        for label in self.labels:
            value_first = first.omega if first.target == label else 0.0
            value_second = second.omega if second.target == label else 0.0
            self.contributions[label].add(value_second - value_first)
        resolved_first = first.omega if first.target is not None else 0.0
        resolved_second = second.omega if second.target is not None else 0.0
        self.row_sum.add(resolved_second - resolved_first)
        self.target_agreements += int(first.target == second.target)
        self.count += 1

    @property
    def target_agreement_rate(self) -> float:
        return self.target_agreements / self.count if self.count else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a low-memory paired FRW diagnostic for panel-centre "
            "resolution and conductor hit tolerance."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--master", default="Net3")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[100, 200, 400],
        help="Analytical panel resolutions. Default: 100 200 400.",
    )
    parser.add_argument(
        "--fixed-tolerance",
        type=float,
        default=1.0e-4,
        help="Tolerance used for resolution convergence. Default: 1e-4.",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6],
        help="Tolerance sequence tested at n=200.",
    )
    parser.add_argument("--series-terms", type=int, default=41)
    parser.add_argument("--walks-per-seed", type=int, default=5000)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1234, 5678, 9012, 3456]
    )
    parser.add_argument("--scale-factor", type=float, default=1.25)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--max-hops", type=int, default=500)
    parser.add_argument("--kill-multiplier", type=float, default=500_000.0)
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=Path("diagnostics/frw_reference_cases.json"),
    )
    parser.add_argument("--case-number", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/diagnostics/frw_panel_hit"),
    )
    parser.add_argument("--label", default="case7_net3_panel_hit")
    return parser


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def integrated_sine_basis(resolution: int, terms: int) -> FloatArray:
    edges = np.linspace(0.0, REFERENCE_SIDE_LENGTH, resolution + 1)
    modes = np.arange(1, terms + 1, dtype=np.float64)
    wave_numbers = modes * np.pi / REFERENCE_SIDE_LENGTH
    return (
        np.cos(edges[:-1, None] * wave_numbers)
        - np.cos(edges[1:, None] * wave_numbers)
    ) / wave_numbers


def exact_center_sine_cosine(terms: int) -> tuple[FloatArray, FloatArray]:
    modes = np.arange(1, terms + 1, dtype=np.int64)
    sine = np.zeros(terms, dtype=np.float64)
    cosine = np.zeros(terms, dtype=np.float64)
    odd = modes % 2 == 1
    sine[odd] = (-1.0) ** ((modes[odd] - 1) // 2)
    even = ~odd
    cosine[even] = (-1.0) ** (modes[even] // 2)
    return sine, cosine


def stable_normal_factors(
    terms: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    modes = np.arange(1, terms + 1, dtype=np.float64)
    wave_numbers = modes * np.pi / REFERENCE_SIDE_LENGTH
    lambda_mn = np.sqrt(
        wave_numbers[:, None] ** 2 + wave_numbers[None, :] ** 2
    )
    a = lambda_mn * REFERENCE_SIDE_LENGTH / 2.0
    exponential = np.exp(-a)
    value_factor = exponential / (1.0 + np.exp(-2.0 * a))
    derivative_factor = (
        lambda_mn * exponential / (1.0 - np.exp(-2.0 * a))
    )
    return wave_numbers, value_factor, derivative_factor


def face_grid(basis: FloatArray, coefficient: FloatArray) -> FloatArray:
    return basis @ coefficient.T @ basis.T


def build_analytical_grid_table(
    resolution: int, terms: int
) -> AnalyticalGridTable:
    if resolution < 3:
        raise ValueError("Every resolution must be at least 3.")
    if terms < 1:
        raise ValueError("--series-terms must be positive.")

    basis = integrated_sine_basis(resolution, terms)
    sine, cosine = exact_center_sine_cosine(terms)
    wave_numbers, value_factor, derivative_factor = stable_normal_factors(
        terms
    )
    prefactor = 4.0 / (REFERENCE_SIDE_LENGTH**2)

    base = prefactor * sine[:, None] * sine[None, :] * value_factor
    tangent_first = (
        prefactor
        * (wave_numbers * cosine)[:, None]
        * sine[None, :]
        * value_factor
    )
    tangent_second = (
        prefactor
        * sine[:, None]
        * (wave_numbers * cosine)[None, :]
        * value_factor
    )
    normal = (
        prefactor
        * sine[:, None]
        * sine[None, :]
        * derivative_factor
    )

    p_raw = face_grid(basis, base)
    grad_t1 = face_grid(basis, tangent_first)
    grad_t2 = face_grid(basis, tangent_second)
    grad_normal = face_grid(basis, normal)

    if np.any(p_raw <= np.finfo(np.float64).tiny):
        bad = int(np.count_nonzero(p_raw <= 0.0))
        raise RuntimeError(
            f"Resolution {resolution}: analytical series produced {bad} "
            "non-positive panel probabilities. Increase --series-terms."
        )

    total_raw = float(6.0 * p_raw.sum())
    probability_face = p_raw / total_raw
    wvt_t1 = grad_t1 / probability_face
    wvt_t2 = grad_t2 / probability_face
    wvt_normal = grad_normal / probability_face

    conditional_face = probability_face / float(probability_face.sum())
    row_probabilities = conditional_face.sum(axis=1)
    row_cdf = np.cumsum(row_probabilities)
    row_cdf[-1] = 1.0
    conditional_column = conditional_face / row_probabilities[:, None]
    conditional_column_cdf = np.cumsum(conditional_column, axis=1)
    conditional_column_cdf[:, -1] = 1.0

    axis_centers = -1.0 + (
        np.arange(resolution, dtype=np.float64) + 0.5
    ) * (2.0 / resolution)

    return AnalyticalGridTable(
        resolution=resolution,
        axis_centers=axis_centers,
        row_cdf=row_cdf,
        conditional_column_cdf=conditional_column_cdf,
        wvt_tangent_first=wvt_t1,
        wvt_tangent_second=wvt_t2,
        wvt_normal=wvt_normal,
        probability_sum_before_normalization=total_raw,
    )


def first_weight(
    table: AnalyticalGridTable,
    panel: tuple[int, int, int],
    surface_normal: FloatArray,
    half_size: float,
) -> float:
    face, row, column = panel
    normal = np.asarray(surface_normal, dtype=np.float64)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 0.0 or not math.isfinite(normal_length):
        raise ValueError("VGSS surface normal must have positive finite length.")
    unit_normal = normal / normal_length
    return -float(
        np.dot(unit_normal, table.wvt_vector(face, row, column))
    ) / half_size


def transition_point(
    current_point: FloatArray,
    half_size: float,
    table: AnalyticalGridTable,
    panel: tuple[int, int, int],
) -> FloatArray:
    return current_point + half_size * table.relative_point(*panel)


def position_state(
    point: FloatArray,
    conductors: Sequence[Any],
    conductor_to_net: dict[str, str],
    system_center: FloatArray,
    kill_radius: float,
) -> tuple[float, str, bool]:
    if np.max(np.abs(point - system_center)) > kill_radius:
        return math.inf, INFINITY, True
    distances = [conductor.distance_to_point(point) for conductor in conductors]
    nearest_index = int(np.argmin(distances))
    return (
        float(distances[nearest_index]),
        conductor_to_net[conductors[nearest_index].name],
        False,
    )


def run_resolution_and_tolerance_paths(
    *,
    start_point: FloatArray,
    surface_normal: FloatArray,
    first_half_size: float,
    rng: np.random.Generator,
    tables: dict[int, AnalyticalGridTable],
    resolutions: tuple[int, ...],
    fixed_tolerance: float,
    tolerances: tuple[float, ...],
    conductors: Sequence[Any],
    conductor_to_net: dict[str, str],
    system_center: FloatArray,
    kill_radius: float,
    max_hops: int,
) -> dict[str, PathResult]:
    tolerance_resolution = 200
    if tolerance_resolution not in tables:
        raise ValueError("The tolerance diagnostic requires resolution 200.")

    results: dict[str, PathResult] = {}
    points: dict[int, FloatArray] = {}
    omegas: dict[int, float] = {}

    u_face, u_row, u_column = rng.random(3)
    for resolution in resolutions:
        table = tables[resolution]
        panel = table.sample_panel(u_face, u_row, u_column)
        omegas[resolution] = first_weight(
            table, panel, surface_normal, first_half_size
        )
        points[resolution] = transition_point(
            start_point, first_half_size, table, panel
        )

    unresolved_tolerances = set(tolerances)
    active_resolutions = set(resolutions)

    for hops in range(1, max_hops + 1):
        for resolution in tuple(active_resolutions):
            point = points[resolution]
            distance, nearest_target, escaped = position_state(
                point,
                conductors,
                conductor_to_net,
                system_center,
                kill_radius,
            )

            if resolution == tolerance_resolution:
                if escaped:
                    for tolerance in tuple(unresolved_tolerances):
                        mode = tolerance_mode_name(resolution, tolerance)
                        results[mode] = PathResult(
                            INFINITY, "escaped", omegas[resolution], hops
                        )
                        unresolved_tolerances.remove(tolerance)
                else:
                    for tolerance in sorted(
                        tuple(unresolved_tolerances), reverse=True
                    ):
                        if distance < tolerance:
                            mode = tolerance_mode_name(resolution, tolerance)
                            results[mode] = PathResult(
                                nearest_target,
                                "hit_conductor",
                                omegas[resolution],
                                hops,
                            )
                            unresolved_tolerances.remove(tolerance)

            resolution_mode = resolution_mode_name(
                resolution, fixed_tolerance
            )
            if resolution_mode not in results:
                if escaped:
                    results[resolution_mode] = PathResult(
                        INFINITY, "escaped", omegas[resolution], hops
                    )
                elif distance < fixed_tolerance:
                    results[resolution_mode] = PathResult(
                        nearest_target,
                        "hit_conductor",
                        omegas[resolution],
                        hops,
                    )

            resolution_done = resolution_mode in results
            if resolution == tolerance_resolution:
                resolution_done = resolution_done and not unresolved_tolerances
            if resolution_done:
                active_resolutions.remove(resolution)

        if not active_resolutions:
            break

        u_face, u_row, u_column = rng.random(3)
        for resolution in tuple(active_resolutions):
            point = points[resolution]
            distance, _nearest_target, escaped = position_state(
                point,
                conductors,
                conductor_to_net,
                system_center,
                kill_radius,
            )
            if escaped:
                continue
            table = tables[resolution]
            panel = table.sample_panel(u_face, u_row, u_column)
            points[resolution] = transition_point(
                point, distance, table, panel
            )

    for resolution in resolutions:
        resolution_mode = resolution_mode_name(
            resolution, fixed_tolerance
        )
        if resolution_mode not in results:
            results[resolution_mode] = PathResult(
                None, "max_hops", omegas[resolution], max_hops + 1
            )
    for tolerance in tolerances:
        mode = tolerance_mode_name(tolerance_resolution, tolerance)
        if mode not in results:
            results[mode] = PathResult(
                None,
                "max_hops",
                omegas[tolerance_resolution],
                max_hops + 1,
            )

    return results


def format_float_token(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def resolution_mode_name(resolution: int, tolerance: float) -> str:
    return f"n{resolution}_tol_{format_float_token(tolerance)}"


def tolerance_mode_name(resolution: int, tolerance: float) -> str:
    return resolution_mode_name(resolution, tolerance)


def mode_metadata(mode: str) -> dict[str, Any]:
    resolution_token, tolerance_token = mode.split("_tol_")
    resolution = int(resolution_token[1:])
    exponent_text = tolerance_token.replace("m", "-")
    tolerance = float(exponent_text)
    return {"resolution": resolution, "hit_tolerance": tolerance}


def build_comparisons(
    resolutions: tuple[int, ...],
    fixed_tolerance: float,
    tolerances: tuple[float, ...],
) -> tuple[tuple[str, str, str], ...]:
    comparisons: list[tuple[str, str, str]] = []
    ordered_resolutions = sorted(resolutions)
    for first, second in zip(ordered_resolutions, ordered_resolutions[1:]):
        comparisons.append(
            (
                f"resolution_n{second}_minus_n{first}",
                resolution_mode_name(first, fixed_tolerance),
                resolution_mode_name(second, fixed_tolerance),
            )
        )
    if len(ordered_resolutions) >= 3:
        first = ordered_resolutions[0]
        last = ordered_resolutions[-1]
        comparisons.append(
            (
                f"resolution_n{last}_minus_n{first}",
                resolution_mode_name(first, fixed_tolerance),
                resolution_mode_name(last, fixed_tolerance),
            )
        )

    ordered_tolerances = sorted(tolerances, reverse=True)
    for loose, tight in zip(ordered_tolerances, ordered_tolerances[1:]):
        comparisons.append(
            (
                f"tolerance_{format_float_token(tight)}_minus_"
                f"{format_float_token(loose)}",
                tolerance_mode_name(200, loose),
                tolerance_mode_name(200, tight),
            )
        )
    if len(ordered_tolerances) >= 3:
        comparisons.append(
            (
                f"tolerance_{format_float_token(ordered_tolerances[-1])}_"
                f"minus_{format_float_token(fixed_tolerance)}",
                tolerance_mode_name(200, fixed_tolerance),
                tolerance_mode_name(200, ordered_tolerances[-1]),
            )
        )
    return tuple(comparisons)


def summarize_mode(
    accumulator: ModeAccumulator,
    physical_scale: float,
    num_walks: int,
) -> dict[str, Any]:
    estimates = {
        label: physical_scale * accumulator.contributions[label].mean()
        for label in accumulator.labels
    }
    standard_errors = {
        label: abs(physical_scale)
        * accumulator.contributions[label].standard_error_of_mean()
        for label in accumulator.labels
    }
    row_sum = physical_scale * accumulator.row_sum.mean()
    row_sum_se = (
        abs(physical_scale) * accumulator.row_sum.standard_error_of_mean()
    )
    return {
        "estimates_F": estimates,
        "standard_errors_F_conditional_on_H": standard_errors,
        "row_sum_with_infinity_F": row_sum,
        "row_sum_standard_error_F_conditional_on_H": row_sum_se,
        "row_sum_z_conditional_on_H": (
            row_sum / row_sum_se if row_sum_se > 0.0 else 0.0
        ),
        "termination_counts": dict(accumulator.termination_counts),
        "target_counts": dict(accumulator.target_counts),
        "resolved_rate": 1.0
        - accumulator.termination_counts["max_hops"] / num_walks,
        "mean_hops": accumulator.hop_sum / num_walks,
        "max_hops_observed": accumulator.max_hops_observed,
    }


def summarize_difference(
    accumulator: DifferenceAccumulator,
    physical_scale: float,
) -> dict[str, Any]:
    delta = {
        label: physical_scale * accumulator.contributions[label].mean()
        for label in accumulator.labels
    }
    se = {
        label: abs(physical_scale)
        * accumulator.contributions[label].standard_error_of_mean()
        for label in accumulator.labels
    }
    z = {
        label: delta[label] / se[label] if se[label] > 0.0 else 0.0
        for label in accumulator.labels
    }
    row_delta = physical_scale * accumulator.row_sum.mean()
    row_se = abs(physical_scale) * accumulator.row_sum.standard_error_of_mean()
    return {
        "delta_F": delta,
        "paired_standard_errors_F_conditional_on_H": se,
        "paired_z_conditional_on_H": z,
        "row_sum_delta_F": row_delta,
        "row_sum_paired_standard_error_F_conditional_on_H": row_se,
        "row_sum_paired_z_conditional_on_H": (
            row_delta / row_se if row_se > 0.0 else 0.0
        ),
        "target_agreement_rate": accumulator.target_agreement_rate,
    }


def combine_seed_modes(
    seed_results: list[dict[str, Any]],
    mode_names: tuple[str, ...],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    seed_count = len(seed_results)
    for mode in mode_names:
        estimates = np.asarray(
            [
                [seed["modes"][mode]["estimates_F"][label] for label in labels]
                for seed in seed_results
            ],
            dtype=np.float64,
        )
        within_se = np.asarray(
            [
                [
                    seed["modes"][mode][
                        "standard_errors_F_conditional_on_H"
                    ][label]
                    for label in labels
                ]
                for seed in seed_results
            ],
            dtype=np.float64,
        )
        mean = estimates.mean(axis=0)
        between_sd = estimates.std(axis=0, ddof=1) if seed_count > 1 else np.zeros_like(mean)
        combined_within = np.sqrt(np.sum(within_se**2, axis=0)) / seed_count
        resolved = np.mean(
            [seed["modes"][mode]["resolved_rate"] for seed in seed_results]
        )
        mean_hops = np.mean(
            [seed["modes"][mode]["mean_hops"] for seed in seed_results]
        )
        row_sums = np.asarray(
            [seed["modes"][mode]["row_sum_with_infinity_F"] for seed in seed_results]
        )
        combined[mode] = {
            "metadata": mode_metadata(mode),
            "mean_estimates_F": dict(zip(labels, mean.tolist())),
            "between_seed_sd_F": dict(zip(labels, between_sd.tolist())),
            "between_seed_se_of_mean_F": dict(
                zip(labels, (between_sd / math.sqrt(seed_count)).tolist())
            ),
            "combined_within_seed_se_of_mean_F": dict(
                zip(labels, combined_within.tolist())
            ),
            "mean_row_sum_with_infinity_F": float(row_sums.mean()),
            "between_seed_se_of_mean_row_sum_F": float(
                row_sums.std(ddof=1) / math.sqrt(seed_count)
                if seed_count > 1
                else 0.0
            ),
            "mean_resolved_rate": float(resolved),
            "mean_hops": float(mean_hops),
        }
    return combined


def combine_seed_differences(
    seed_results: list[dict[str, Any]],
    comparisons: tuple[tuple[str, str, str], ...],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    seed_count = len(seed_results)
    for comparison_name, _first, _second in comparisons:
        deltas = np.asarray(
            [
                [
                    seed["paired_differences"][comparison_name]["delta_F"][label]
                    for label in labels
                ]
                for seed in seed_results
            ],
            dtype=np.float64,
        )
        within_se = np.asarray(
            [
                [
                    seed["paired_differences"][comparison_name][
                        "paired_standard_errors_F_conditional_on_H"
                    ][label]
                    for label in labels
                ]
                for seed in seed_results
            ],
            dtype=np.float64,
        )
        mean_delta = deltas.mean(axis=0)
        combined_within = np.sqrt(np.sum(within_se**2, axis=0)) / seed_count
        z = np.divide(
            mean_delta,
            combined_within,
            out=np.zeros_like(mean_delta),
            where=combined_within > 0.0,
        )
        agreements = [
            seed["paired_differences"][comparison_name][
                "target_agreement_rate"
            ]
            for seed in seed_results
        ]
        combined[comparison_name] = {
            "mean_delta_F": dict(zip(labels, mean_delta.tolist())),
            "combined_within_seed_paired_se_F": dict(
                zip(labels, combined_within.tolist())
            ),
            "paired_z": dict(zip(labels, z.tolist())),
            "mean_target_agreement_rate": float(np.mean(agreements)),
        }
    return combined


def load_reference_row(
    path: Path,
    case_number: int,
    master_name: str,
    net_order: tuple[str, ...],
) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"WARNING: reference JSON not found: {path}")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    case = data.get("cases", {}).get(str(case_number))
    if case is None:
        print(f"WARNING: case {case_number} absent from {path}")
        return None
    if master_name not in net_order:
        return None
    row_index = net_order.index(master_name)
    references = {
        name: values[row_index]
        for name, values in case.get("reference_matrices", {}).items()
    }
    consensus_matrix = case.get("consensus_median")
    consensus = consensus_matrix[row_index] if consensus_matrix else None
    return {
        "case_number": case_number,
        "spacing": case.get("spacing"),
        "references": references,
        "consensus": consensus,
    }


def write_outputs(
    *,
    output_dir: Path,
    label: str,
    summary: dict[str, Any],
    labels: tuple[str, ...],
    mode_names: tuple[str, ...],
    comparisons: tuple[tuple[str, str, str], ...],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}_summary.json"
    entries_path = output_dir / f"{label}_entries.csv"
    differences_path = output_dir / f"{label}_paired_differences.csv"
    mode_summary_path = output_dir / f"{label}_mode_summary.csv"

    json_path.write_text(
        json.dumps(_to_builtin(summary), indent=2), encoding="utf-8"
    )

    aggregate_modes = summary["aggregate_modes"]
    reference = summary.get("reference")
    consensus = reference.get("consensus") if reference else None
    references = reference.get("references", {}) if reference else {}
    net_order = summary["net_order"]

    with entries_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = [
            "mode",
            "resolution",
            "hit_tolerance",
            "target",
            "estimate_F",
            "SE_F",
            "relative_SE",
            "consensus_reference_F",
            "absolute_relative_error_vs_consensus",
        ]
        for reference_name in references:
            header.extend(
                [
                    f"{reference_name}_F",
                    f"absolute_relative_error_vs_{reference_name}",
                ]
            )
        writer.writerow(header)
        for mode in mode_names:
            mode_data = aggregate_modes[mode]
            metadata = mode_data["metadata"]
            for target in labels:
                estimate = mode_data["mean_estimates_F"][target]
                se = mode_data["combined_within_seed_se_of_mean_F"][target]
                relative_se = abs(se / estimate) if estimate != 0.0 else math.inf
                target_index = net_order.index(target) if target in net_order else None
                consensus_value = (
                    consensus[target_index]
                    if consensus is not None and target_index is not None
                    else None
                )
                consensus_error = (
                    abs((estimate - consensus_value) / consensus_value)
                    if consensus_value not in (None, 0.0)
                    else None
                )
                row = [
                    mode,
                    metadata["resolution"],
                    metadata["hit_tolerance"],
                    target,
                    estimate,
                    se,
                    relative_se,
                    consensus_value,
                    consensus_error,
                ]
                for reference_name, reference_row in references.items():
                    value = (
                        reference_row[target_index]
                        if target_index is not None
                        else None
                    )
                    error = (
                        abs((estimate - value) / value)
                        if value not in (None, 0.0)
                        else None
                    )
                    row.extend([value, error])
                writer.writerow(row)

    aggregate_differences = summary["aggregate_paired_differences"]
    with differences_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "comparison",
                "first_mode",
                "second_mode",
                "target",
                "mean_delta_F",
                "paired_SE_F",
                "paired_z",
                "mean_target_agreement_rate",
            ]
        )
        lookup = {name: (first, second) for name, first, second in comparisons}
        for comparison_name, data in aggregate_differences.items():
            first, second = lookup[comparison_name]
            for target in labels:
                writer.writerow(
                    [
                        comparison_name,
                        first,
                        second,
                        target,
                        data["mean_delta_F"][target],
                        data["combined_within_seed_paired_se_F"][target],
                        data["paired_z"][target],
                        data["mean_target_agreement_rate"],
                    ]
                )

    master = summary["master"]
    with mode_summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "mode",
                "resolution",
                "hit_tolerance",
                "mean_hops",
                "resolved_rate",
                "row_sum_with_infinity_F",
                "direct_diagonal_F",
                "conservation_diagonal_F",
                "direct_minus_conservation_F",
            ]
        )
        for mode in mode_names:
            data = aggregate_modes[mode]
            estimates = data["mean_estimates_F"]
            direct_diagonal = estimates[master]
            conservation_diagonal = -sum(
                value
                for target, value in estimates.items()
                if target != master
            )
            writer.writerow(
                [
                    mode,
                    data["metadata"]["resolution"],
                    data["metadata"]["hit_tolerance"],
                    data["mean_hops"],
                    data["mean_resolved_rate"],
                    data["mean_row_sum_with_infinity_F"],
                    direct_diagonal,
                    conservation_diagonal,
                    direct_diagonal - conservation_diagonal,
                ]
            )

    return [json_path, entries_path, differences_path, mode_summary_path]


def validate_arguments(args: argparse.Namespace) -> None:
    if args.walks_per_seed <= 1:
        raise SystemExit("--walks-per-seed must be greater than 1.")
    if not args.seeds:
        raise SystemExit("At least one seed is required.")
    resolutions = tuple(sorted(set(args.resolutions)))
    if 200 not in resolutions:
        raise SystemExit("--resolutions must include 200.")
    if any(value < 3 for value in resolutions):
        raise SystemExit("Every resolution must be at least 3.")
    tolerances = tuple(sorted(set(args.tolerances), reverse=True))
    if args.fixed_tolerance not in tolerances:
        raise SystemExit("--tolerances must include --fixed-tolerance.")
    if any(value <= 0.0 or not math.isfinite(value) for value in tolerances):
        raise SystemExit("Every tolerance must be finite and positive.")
    if args.max_hops <= 0:
        raise SystemExit("--max-hops must be positive.")


def main() -> None:
    args = build_parser().parse_args()
    validate_arguments(args)

    resolutions = tuple(sorted(set(args.resolutions)))
    tolerances = tuple(sorted(set(args.tolerances), reverse=True))
    comparisons = build_comparisons(
        resolutions, args.fixed_tolerance, tolerances
    )

    conductors = parse_matlab_geometry(args.input_file)
    nets = find_touching_nets(conductors)
    net_order = tuple(net.name for net in nets)
    conductor_to_net = {
        conductor.name: net.name
        for net in nets
        for conductor in net.conductors
    }
    if args.master not in net_order:
        raise SystemExit(
            f"Unknown master net {args.master!r}; available nets: {net_order}"
        )

    vgss_all = build_all_net_vgss(
        conductors,
        scale_factor=args.scale_factor,
        max_distance=args.max_distance,
    )
    master_vgss = next(vgss for vgss in vgss_all if vgss.net.name == args.master)
    system_center, system_size = get_system_bounds(conductors)
    kill_radius = system_size * args.kill_multiplier

    print("Building independent analytical transition tables...")
    tables: dict[int, AnalyticalGridTable] = {}
    for resolution in resolutions:
        print(f"  n={resolution}, terms={args.series_terms}")
        tables[resolution] = build_analytical_grid_table(
            resolution, args.series_terms
        )

    mode_names = tuple(
        sorted(
            {
                *(
                    resolution_mode_name(r, args.fixed_tolerance)
                    for r in resolutions
                ),
                *(tolerance_mode_name(200, t) for t in tolerances),
            },
            key=lambda name: (
                mode_metadata(name)["resolution"],
                -mode_metadata(name)["hit_tolerance"],
            ),
        )
    )
    labels = tuple(net_order) + (INFINITY,)

    print("Input:", args.input_file)
    print("Master:", args.master)
    print("Resolutions:", resolutions)
    print("Fixed tolerance:", args.fixed_tolerance)
    print("Tolerance sequence:", tolerances)
    print("Walks per seed:", args.walks_per_seed)
    print("Seeds:", args.seeds)
    print("Max hops:", args.max_hops)
    print("Modes:", mode_names)

    seed_results: list[dict[str, Any]] = []

    for base_seed in args.seeds:
        print("\n" + "=" * 78)
        print(f"BASE SEED {base_seed}")
        print("=" * 78)
        rng = np.random.default_rng(base_seed)
        stats = VGSSSamplingStats()
        mode_accumulators = {
            mode: ModeAccumulator(labels) for mode in mode_names
        }
        difference_accumulators = {
            name: DifferenceAccumulator(labels)
            for name, _first, _second in comparisons
        }
        progress_step = max(1, args.walks_per_seed // 20)

        for walk_index in range(args.walks_per_seed):
            if (
                (walk_index + 1) % progress_step == 0
                or walk_index + 1 == args.walks_per_seed
            ):
                percentage = 100.0 * (walk_index + 1) / args.walks_per_seed
                print(
                    f"\r  {percentage:5.1f}% "
                    f"({walk_index + 1}/{args.walks_per_seed})",
                    end="",
                    flush=True,
                )

            start_sample = sample_on_vgss(
                master_vgss.sampling_context, rng, stats=stats
            )
            start_point = np.asarray(start_sample.point, dtype=np.float64)
            distances = [
                conductor.distance_to_point(start_point)
                for conductor in conductors
            ]
            first_half_size = float(min(distances))
            if first_half_size <= 0.0:
                raise RuntimeError("Non-positive first maximal-cube half-size.")

            results = run_resolution_and_tolerance_paths(
                start_point=start_point,
                surface_normal=start_sample.normal,
                first_half_size=first_half_size,
                rng=rng,
                tables=tables,
                resolutions=resolutions,
                fixed_tolerance=args.fixed_tolerance,
                tolerances=tolerances,
                conductors=conductors,
                conductor_to_net=conductor_to_net,
                system_center=system_center,
                kill_radius=kill_radius,
                max_hops=args.max_hops,
            )

            for mode in mode_names:
                mode_accumulators[mode].add(results[mode])
            for comparison_name, first_mode, second_mode in comparisons:
                difference_accumulators[comparison_name].add(
                    results[first_mode], results[second_mode]
                )

        print()
        context = master_vgss.sampling_context
        H = stats.estimate_H(context)
        physical_scale = E0 * L_UNIT * H
        seed_result = {
            "base_seed": base_seed,
            "walk_count": args.walks_per_seed,
            "H": H,
            "Sg": stats.estimate_surface_area(context),
            "mean_p": stats.estimate_mean_p(),
            "sampling_stats": stats.as_dict(context),
            "modes": {
                mode: summarize_mode(
                    mode_accumulators[mode],
                    physical_scale,
                    args.walks_per_seed,
                )
                for mode in mode_names
            },
            "paired_differences": {
                name: summarize_difference(
                    difference_accumulators[name], physical_scale
                )
                for name, _first, _second in comparisons
            },
        }
        seed_results.append(seed_result)

        print("Seed diagonal estimates [pF]:")
        for mode in mode_names:
            value = seed_result["modes"][mode]["estimates_F"][args.master]
            print(f"  {mode:20s} {value * 1e12: .8f}")

    aggregate_modes = combine_seed_modes(
        seed_results, mode_names, labels
    )
    aggregate_differences = combine_seed_differences(
        seed_results, comparisons, labels
    )
    reference = load_reference_row(
        args.reference_json,
        args.case_number,
        args.master,
        net_order,
    )

    summary = {
        "description": (
            "FRW analytical panel-centre resolution and hit-tolerance "
            "diagnostic."
        ),
        "parameters": {
            "input_file": args.input_file,
            "master": args.master,
            "resolutions": resolutions,
            "fixed_tolerance": args.fixed_tolerance,
            "tolerances": tolerances,
            "series_terms": args.series_terms,
            "walks_per_seed": args.walks_per_seed,
            "seeds": args.seeds,
            "total_walks": args.walks_per_seed * len(args.seeds),
            "scale_factor": args.scale_factor,
            "max_distance": args.max_distance,
            "max_hops": args.max_hops,
            "kill_multiplier": args.kill_multiplier,
            "E0": E0,
            "L_UNIT": L_UNIT,
        },
        "net_order": net_order,
        "target_order": labels,
        "master": args.master,
        "table_diagnostics": {
            str(resolution): {
                "probability_sum_before_normalization": (
                    tables[resolution].probability_sum_before_normalization
                )
            }
            for resolution in resolutions
        },
        "mode_definitions": {
            mode: mode_metadata(mode) for mode in mode_names
        },
        "comparison_definitions": {
            name: {"first_mode": first, "second_mode": second}
            for name, first, second in comparisons
        },
        "reference": reference,
        "seed_results": seed_results,
        "aggregate_modes": aggregate_modes,
        "aggregate_paired_differences": aggregate_differences,
    }

    outputs = write_outputs(
        output_dir=args.output_dir,
        label=args.label,
        summary=summary,
        labels=labels,
        mode_names=mode_names,
        comparisons=comparisons,
    )

    print("\n" + "#" * 78)
    print("AGGREGATE DIAGONAL ESTIMATES")
    print("#" * 78)
    for mode in mode_names:
        mode_data = aggregate_modes[mode]
        estimate = mode_data["mean_estimates_F"][args.master]
        se = mode_data["combined_within_seed_se_of_mean_F"][args.master]
        print(f"{mode:20s}: {estimate * 1e12: .8f} ± {se * 1e12:.8f} pF")

    print("\nPAIRED DIAGONAL DIFFERENCES")
    for comparison_name, data in aggregate_differences.items():
        delta = data["mean_delta_F"][args.master]
        se = data["combined_within_seed_paired_se_F"][args.master]
        z = data["paired_z"][args.master]
        agreement = data["mean_target_agreement_rate"]
        print(
            f"{comparison_name:36s}: "
            f"{delta * 1e12:+.8f} ± {se * 1e12:.8f} pF, "
            f"z={z:+.3f}, target agreement={agreement:.6f}"
        )

    print("\nOutputs:")
    for path in outputs:
        print(" ", path)


if __name__ == "__main__":
    main()
