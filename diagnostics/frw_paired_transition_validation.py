#!/usr/bin/env python3
"""Paired FRW diagnostic for GGFT probability/gradient effects.

This is a local diagnostic, not production code. It compares four estimators
while keeping the conductor geometry, VGSS samples, random seeds, panel grid,
maximal-cube construction, hit test, and infinity test fixed:

1. numerical
   Numerical GGFT probabilities for all transitions and the stored numerical
   WVT for the first-hop weight. This mirrors the current FRW implementation.

2. reference_gradient_same_path
   Exactly the same sampled panels and paths as ``numerical``, but the first-hop
   numerator uses an independently evaluated analytical cube gradient. The
   panel is still sampled with the numerical probability and the importance
   denominator remains the numerical probability. This isolates the numerical
   first-hop gradient/WVT error with very low paired Monte Carlo noise.

3. analytical_first_numerical_rest
   Analytical probability and analytical gradient for the first transition,
   followed by numerical probabilities for subsequent transitions. Compared
   with mode 2, this measures the first-panel probability-distribution effect
   (including the changed first landing point).

4. analytical_all
   Analytical probabilities for every transition and the analytical first-hop
   gradient/weight. Compared with mode 3, this measures the effect of numerical
   probabilities on subsequent maximal-cube transitions.

The analytical cube table is built independently with a panel-integrated
separation-of-variables Poisson-kernel series on [-1, 1]^3. No walk paths are
stored, so memory usage stays small.

Intended location inside the VGSS checkout:

    diagnostics/frw_paired_transition_validation.py

The repository's local diagnostics/output ignore rules should keep this script
and its results out of Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

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
MODE_NAMES = (
    "numerical",
    "reference_gradient_same_path",
    "analytical_first_numerical_rest",
    "analytical_all",
)
COMPARISONS = (
    (
        "reference_gradient_minus_numerical",
        "numerical",
        "reference_gradient_same_path",
    ),
    (
        "analytical_first_minus_reference_gradient",
        "reference_gradient_same_path",
        "analytical_first_numerical_rest",
    ),
    (
        "analytical_all_minus_analytical_first",
        "analytical_first_numerical_rest",
        "analytical_all",
    ),
    (
        "analytical_all_minus_numerical",
        "numerical",
        "analytical_all",
    ),
)
REFERENCE_SIDE_LENGTH = 2.0


@dataclass(frozen=True)
class TransitionTable:
    """One panel-centre cube transition table with a normalized CDF."""

    name: str
    resolution: int
    probabilities: FloatArray
    cdf: FloatArray
    panel_centers: FloatArray
    gradients: FloatArray
    wvt: FloatArray


@dataclass
class ScalarMoments:
    """Streaming moments of a per-walk scalar contribution."""

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

    def add(
        self,
        target: str | None,
        omega: float,
        termination: str,
        hops: int,
    ) -> None:
        for label in self.labels:
            value = omega if target == label else 0.0
            self.contributions[label].add(value)
        resolved_value = omega if target is not None else 0.0
        self.row_sum.add(resolved_value)
        self.termination_counts[termination] += 1
        if target is not None:
            self.target_counts[target] += 1
        self.hop_sum += hops
        self.max_hops_observed = max(self.max_hops_observed, hops)


@dataclass
class DifferenceAccumulator:
    labels: tuple[str, ...]
    contributions: dict[str, ScalarMoments] = field(init=False)
    row_sum: ScalarMoments = field(default_factory=ScalarMoments)

    def __post_init__(self) -> None:
        self.contributions = {
            label: ScalarMoments() for label in self.labels
        }

    def add(
        self,
        target_a: str | None,
        omega_a: float,
        target_b: str | None,
        omega_b: float,
    ) -> None:
        for label in self.labels:
            value_a = omega_a if target_a == label else 0.0
            value_b = omega_b if target_b == label else 0.0
            self.contributions[label].add(value_b - value_a)
        resolved_a = omega_a if target_a is not None else 0.0
        resolved_b = omega_b if target_b is not None else 0.0
        self.row_sum.add(resolved_b - resolved_a)


@dataclass(frozen=True)
class PathResult:
    target: str | None
    termination: str
    first_panel_index: int
    omega: float
    hops: int


@dataclass(frozen=True)
class MaximalCategoricalCoupler:
    """Maximal coupling of two categorical panel distributions.

    The two sampled indices are identical with probability
    ``sum(min(p_i, q_i)) = 1 - total_variation_distance`` while preserving
    each table's exact marginal distribution.
    """

    common_mass: float
    common_cdf: FloatArray
    p_residual_cdf: FloatArray
    q_residual_cdf: FloatArray

    @classmethod
    def build(
        cls, p: FloatArray, q: FloatArray
    ) -> "MaximalCategoricalCoupler":
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if p.shape != q.shape or p.ndim != 1:
            raise ValueError("Coupled distributions must be equal-length vectors.")
        common = np.minimum(p, q)
        common_mass = float(common.sum())
        residual_mass = 1.0 - common_mass

        def make_cdf(values: FloatArray, mass: float) -> FloatArray:
            if mass <= 1.0e-15:
                # This CDF is never used in the zero-mass branch.
                cdf = np.zeros_like(values)
                cdf[-1] = 1.0
                return cdf
            normalized = np.maximum(values, 0.0) / mass
            cdf = np.cumsum(normalized)
            cdf[-1] = 1.0
            return cdf

        return cls(
            common_mass=common_mass,
            common_cdf=make_cdf(common, common_mass),
            p_residual_cdf=make_cdf(p - common, residual_mass),
            q_residual_cdf=make_cdf(q - common, residual_mass),
        )

    @property
    def total_variation_distance(self) -> float:
        return 1.0 - self.common_mass

    @staticmethod
    def _sample_cdf(cdf: FloatArray, rng: np.random.Generator) -> int:
        return int(np.searchsorted(cdf, float(rng.random()), side="right"))

    def sample(self, rng: np.random.Generator) -> tuple[int, int]:
        if float(rng.random()) < self.common_mass:
            index = self._sample_cdf(self.common_cdf, rng)
            return index, index
        p_index = self._sample_cdf(self.p_residual_cdf, rng)
        q_index = self._sample_cdf(self.q_residual_cdf, rng)
        return p_index, q_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a low-memory paired FRW comparison using numerical and "
            "independent analytical cube-transition tables."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("src/GGFT/data/cube_tables_n200.npz"),
        help="Numerical GGFT NPZ table. Default: n200 bundled table.",
    )
    parser.add_argument(
        "--series-terms",
        type=int,
        default=41,
        help="Analytical sine modes per transverse direction. Default: 41.",
    )
    parser.add_argument(
        "--walks-per-seed",
        type=int,
        default=5000,
        help="Walks per master net for each seed. Default: 5000.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1234, 5678, 9012, 3456],
        help="Independent base seeds. Default: 1234 5678 9012 3456.",
    )
    parser.add_argument(
        "--masters",
        nargs="*",
        help="Optional master-net names, e.g. --masters Net1 Net3. Default: all.",
    )
    parser.add_argument("--scale-factor", type=float, default=1.25)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--max-hops", type=int, default=200)
    parser.add_argument("--hit-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--kill-multiplier", type=float, default=500_000.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/diagnostics/frw_paired_transition"),
    )
    parser.add_argument(
        "--label",
        default="paired_transition",
        help="Basename prefix for JSON/CSV outputs.",
    )
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


def load_numerical_table(path: Path) -> TransitionTable:
    if not path.is_file():
        raise FileNotFoundError(f"GGFT table not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "resolution",
            "probabilities",
            "panel_centers",
            "gradients",
            "wvt",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path}: missing NPZ fields: {missing}")
        resolution = int(data["resolution"])
        probabilities = np.asarray(
            data["probabilities"], dtype=np.float64
        ).reshape(-1)
        panel_centers = np.asarray(
            data["panel_centers"], dtype=np.float64
        ).reshape(-1, 3)
        gradients = np.asarray(
            data["gradients"], dtype=np.float64
        ).reshape(-1, 3)
        wvt = np.asarray(data["wvt"], dtype=np.float64).reshape(-1, 3)

    panel_count = 6 * resolution * resolution
    if probabilities.shape != (panel_count,):
        raise ValueError("Unexpected probability-table shape.")
    for name, array in (
        ("panel_centers", panel_centers),
        ("gradients", gradients),
        ("wvt", wvt),
    ):
        if array.shape != (panel_count, 3):
            raise ValueError(f"Unexpected {name} shape: {array.shape}")
    if np.any(probabilities <= 0.0):
        raise ValueError("Numerical probabilities must be positive.")
    if not all(
        np.all(np.isfinite(array))
        for array in (probabilities, panel_centers, gradients, wvt)
    ):
        raise ValueError("Numerical table contains non-finite values.")

    original_sum = float(probabilities.sum())
    probabilities = probabilities / original_sum
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    if not math.isclose(original_sum, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        print(
            "WARNING: numerical probability sum was "
            f"{original_sum:.16e}; runtime-style normalization was applied."
        )

    return TransitionTable(
        name=f"numerical:{path.name}",
        resolution=resolution,
        probabilities=probabilities,
        cdf=cdf,
        panel_centers=panel_centers,
        gradients=gradients,
        wvt=wvt,
    )


def build_panel_centers(resolution: int) -> FloatArray:
    axis = -1.0 + (
        np.arange(resolution, dtype=np.float64) + 0.5
    ) * (2.0 / resolution)
    centers: list[tuple[float, float, float]] = []
    centers.extend((x, y, -1.0) for y in axis for x in axis)
    centers.extend((x, y, 1.0) for y in axis for x in axis)
    centers.extend((x, -1.0, z) for z in axis for x in axis)
    centers.extend((x, 1.0, z) for z in axis for x in axis)
    centers.extend((-1.0, y, z) for z in axis for y in axis)
    centers.extend((1.0, y, z) for z in axis for y in axis)
    return np.asarray(centers, dtype=np.float64)


def integrated_sine_basis(resolution: int, terms: int) -> FloatArray:
    edges = np.linspace(
        0.0, REFERENCE_SIDE_LENGTH, resolution + 1, dtype=np.float64
    )
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


def build_analytical_table(resolution: int, terms: int) -> TransitionTable:
    if resolution < 3:
        raise ValueError("Resolution must be at least 3.")
    if terms < 1:
        raise ValueError("--series-terms must be positive.")

    basis = integrated_sine_basis(resolution, terms)
    sine, cosine = exact_center_sine_cosine(terms)
    wave_numbers, value_factor, derivative_factor = (
        stable_normal_factors(terms)
    )
    prefactor = 4.0 / (REFERENCE_SIDE_LENGTH**2)

    base = (
        prefactor
        * sine[:, None]
        * sine[None, :]
        * value_factor
    )
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

    p_grid = face_grid(basis, base)
    t1_grid = face_grid(basis, tangent_first)
    t2_grid = face_grid(basis, tangent_second)
    n_grid = face_grid(basis, normal)

    probability_faces: list[FloatArray] = []
    gradient_faces: list[FloatArray] = []

    # z-, z+: grid rows y, columns x.
    for sign in (-1.0, 1.0):
        probability_faces.append(p_grid.ravel())
        gradient_faces.append(
            np.column_stack(
                (
                    t1_grid.ravel(),
                    t2_grid.ravel(),
                    (sign * n_grid).ravel(),
                )
            )
        )
    # y-, y+: grid rows z, columns x.
    for sign in (-1.0, 1.0):
        probability_faces.append(p_grid.ravel())
        gradient_faces.append(
            np.column_stack(
                (
                    t1_grid.ravel(),
                    (sign * n_grid).ravel(),
                    t2_grid.ravel(),
                )
            )
        )
    # x-, x+: grid rows z, columns y.
    for sign in (-1.0, 1.0):
        probability_faces.append(p_grid.ravel())
        gradient_faces.append(
            np.column_stack(
                (
                    (sign * n_grid).ravel(),
                    t1_grid.ravel(),
                    t2_grid.ravel(),
                )
            )
        )

    probabilities = np.concatenate(probability_faces)
    gradients = np.vstack(gradient_faces)
    if np.any(probabilities <= np.finfo(np.float64).tiny):
        bad = int(np.count_nonzero(probabilities <= 0.0))
        raise RuntimeError(
            f"Analytical series produced {bad} non-positive probabilities; "
            "increase --series-terms."
        )
    probabilities = probabilities / float(probabilities.sum())
    wvt = gradients / probabilities[:, None]
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0

    return TransitionTable(
        name=f"analytical:series_{terms}",
        resolution=resolution,
        probabilities=probabilities,
        cdf=cdf,
        panel_centers=build_panel_centers(resolution),
        gradients=gradients,
        wvt=wvt,
    )


def validate_panel_alignment(
    numerical: TransitionTable,
    analytical: TransitionTable,
) -> None:
    if numerical.resolution != analytical.resolution:
        raise ValueError("Numerical and analytical resolutions differ.")
    max_error = float(
        np.max(np.abs(numerical.panel_centers - analytical.panel_centers))
    )
    if max_error > 1.0e-13:
        raise ValueError(
            "Numerical/analytical panel ordering differs; "
            f"max centre error={max_error:.6e}."
        )


def sample_index(table: TransitionTable, rng: np.random.Generator) -> int:
    u = float(rng.random())
    return int(np.searchsorted(table.cdf, u, side="right"))


def first_weight_from_wvt(
    wvt: FloatArray,
    panel_index: int,
    normal: FloatArray,
    half_size: float,
) -> float:
    normal = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("VGSS normal must have positive finite length.")
    return -float(np.dot(normal / length, wvt[panel_index])) / half_size


def first_weight_reference_gradient_over_numerical_probability(
    analytical: TransitionTable,
    numerical: TransitionTable,
    panel_index: int,
    normal: FloatArray,
    half_size: float,
) -> float:
    normal = np.asarray(normal, dtype=np.float64)
    unit = normal / float(np.linalg.norm(normal))
    numerator = analytical.gradients[panel_index]
    denominator = numerical.probabilities[panel_index]
    return -float(np.dot(unit, numerator / denominator)) / half_size


def map_panel(
    current_point: FloatArray,
    half_size: float,
    table: TransitionTable,
    panel_index: int,
) -> FloatArray:
    return current_point + half_size * table.panel_centers[panel_index]


def classify_position(
    current_point: FloatArray,
    conductors: Sequence[Any],
    conductor_to_net: dict[str, str],
    system_center: FloatArray,
    kill_radius: float,
    hit_tolerance: float,
) -> tuple[str | None, str | None]:
    """Return (target, termination), or (None, None) while active."""
    if np.max(np.abs(current_point - system_center)) > kill_radius:
        return INFINITY, "escaped"
    distances = [
        conductor.distance_to_point(current_point)
        for conductor in conductors
    ]
    half_size = min(distances)
    nearest_index = int(np.argmin(distances))
    if half_size < hit_tolerance:
        conductor_name = conductors[nearest_index].name
        return conductor_to_net[conductor_name], "hit_conductor"
    return None, None


def next_half_size(
    current_point: FloatArray, conductors: Sequence[Any]
) -> float:
    return min(
        conductor.distance_to_point(current_point)
        for conductor in conductors
    )


def run_coupled_paths(
    start_point: FloatArray,
    normal: FloatArray,
    first_half_size: float,
    rng: np.random.Generator,
    numerical: TransitionTable,
    analytical: TransitionTable,
    coupler: MaximalCategoricalCoupler,
    conductors: Sequence[Any],
    conductor_to_net: dict[str, str],
    system_center: FloatArray,
    kill_radius: float,
    max_hops: int,
    hit_tolerance: float,
) -> dict[str, PathResult]:
    """Run numerical, analytical-first, and analytical-all paths jointly.

    Maximal categorical coupling keeps the numerical and analytical panel
    indices equal as often as mathematically possible while preserving both
    exact marginal transition distributions. At continuation hops, the same
    numerical index is used for the numerical and analytical-first paths.
    """
    first_num, first_ref = coupler.sample(rng)
    omega_num = first_weight_from_wvt(
        numerical.wvt, first_num, normal, first_half_size
    )
    omega_ref_gradient = (
        first_weight_reference_gradient_over_numerical_probability(
            analytical, numerical, first_num, normal, first_half_size
        )
    )
    omega_ref = first_weight_from_wvt(
        analytical.wvt, first_ref, normal, first_half_size
    )

    points = {
        "numerical": map_panel(
            start_point, first_half_size, numerical, first_num
        ),
        "analytical_first_numerical_rest": map_panel(
            start_point, first_half_size, analytical, first_ref
        ),
        "analytical_all": map_panel(
            start_point, first_half_size, analytical, first_ref
        ),
    }
    first_panels = {
        "numerical": first_num,
        "analytical_first_numerical_rest": first_ref,
        "analytical_all": first_ref,
    }
    omegas = {
        "numerical": omega_num,
        "analytical_first_numerical_rest": omega_ref,
        "analytical_all": omega_ref,
    }
    completed: dict[str, PathResult] = {}

    for continuation_index in range(max_hops):
        transitions_so_far = 1 + continuation_index
        for mode in tuple(points):
            if mode in completed:
                continue
            target, termination = classify_position(
                points[mode],
                conductors,
                conductor_to_net,
                system_center,
                kill_radius,
                hit_tolerance,
            )
            if termination is not None:
                completed[mode] = PathResult(
                    target=target,
                    termination=termination,
                    first_panel_index=first_panels[mode],
                    omega=omegas[mode],
                    hops=transitions_so_far,
                )

        if len(completed) == 3:
            break

        next_num, next_ref = coupler.sample(rng)
        for mode in tuple(points):
            if mode in completed:
                continue
            half_size = next_half_size(points[mode], conductors)
            panel_index = (
                next_ref if mode == "analytical_all" else next_num
            )
            table = (
                analytical if mode == "analytical_all" else numerical
            )
            points[mode] = map_panel(
                points[mode], half_size, table, panel_index
            )

    for mode in points:
        if mode not in completed:
            completed[mode] = PathResult(
                target=None,
                termination="max_hops",
                first_panel_index=first_panels[mode],
                omega=omegas[mode],
                hops=1 + max_hops,
            )

    numerical_result = completed["numerical"]
    completed["reference_gradient_same_path"] = PathResult(
        target=numerical_result.target,
        termination=numerical_result.termination,
        first_panel_index=first_num,
        omega=omega_ref_gradient,
        hops=numerical_result.hops,
    )
    return completed


def physical_summary(
    accumulator: ModeAccumulator,
    physical_scale: float,
) -> dict[str, Any]:
    estimates = {
        label: physical_scale * moments.mean()
        for label, moments in accumulator.contributions.items()
    }
    standard_errors = {
        label: physical_scale * moments.standard_error_of_mean()
        for label, moments in accumulator.contributions.items()
    }
    row_sum = physical_scale * accumulator.row_sum.mean()
    row_sum_se = physical_scale * accumulator.row_sum.standard_error_of_mean()
    resolved = sum(accumulator.target_counts.values())
    total = accumulator.row_sum.count
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
        "resolved_rate": resolved / total if total else 0.0,
        "mean_hops": accumulator.hop_sum / total if total else 0.0,
        "max_hops_observed": accumulator.max_hops_observed,
    }


def physical_difference_summary(
    accumulator: DifferenceAccumulator,
    physical_scale: float,
    baseline_estimates: dict[str, float],
) -> dict[str, Any]:
    delta = {
        label: physical_scale * moments.mean()
        for label, moments in accumulator.contributions.items()
    }
    se = {
        label: physical_scale * moments.standard_error_of_mean()
        for label, moments in accumulator.contributions.items()
    }
    z = {
        label: delta[label] / se[label] if se[label] > 0.0 else 0.0
        for label in accumulator.labels
    }
    relative = {
        label: (
            delta[label] / baseline_estimates[label]
            if baseline_estimates[label] != 0.0
            else math.nan
        )
        for label in accumulator.labels
    }
    row_delta = physical_scale * accumulator.row_sum.mean()
    row_se = physical_scale * accumulator.row_sum.standard_error_of_mean()
    return {
        "delta_F": delta,
        "paired_standard_error_F_conditional_on_H": se,
        "paired_z_conditional_on_H": z,
        "relative_delta_to_baseline": relative,
        "row_sum_delta_F": row_delta,
        "row_sum_paired_standard_error_F_conditional_on_H": row_se,
        "row_sum_paired_z_conditional_on_H": (
            row_delta / row_se if row_se > 0.0 else 0.0
        ),
    }


def run_master_seed(
    conductors: Sequence[Any],
    master_vgss: Any,
    start_rng: np.random.Generator,
    numerical: TransitionTable,
    analytical: TransitionTable,
    coupler: MaximalCategoricalCoupler,
    net_order: list[str],
    conductor_to_net: dict[str, str],
    system_center: FloatArray,
    kill_radius: float,
    num_walks: int,
    max_hops: int,
    hit_tolerance: float,
) -> dict[str, Any]:
    labels = tuple(net_order + [INFINITY])
    mode_accumulators = {
        mode: ModeAccumulator(labels) for mode in MODE_NAMES
    }
    difference_accumulators = {
        name: DifferenceAccumulator(labels)
        for name, _baseline, _candidate in COMPARISONS
    }
    stats = VGSSSamplingStats()
    agreement = {
        "numerical_vs_analytical_first_first_panel": 0,
        "numerical_vs_analytical_first_target": 0,
        "analytical_first_vs_analytical_all_target": 0,
        "numerical_vs_analytical_all_target": 0,
        "numerical_vs_analytical_first_termination": 0,
        "analytical_first_vs_analytical_all_termination": 0,
        "numerical_vs_analytical_all_termination": 0,
    }
    progress_step = max(1, num_walks // 20)

    for walk_index in range(num_walks):
        if (walk_index + 1) % progress_step == 0 or walk_index + 1 == num_walks:
            pct = 100.0 * (walk_index + 1) / num_walks
            print(
                f"\r  {master_vgss.net.name}: {pct:5.1f}% "
                f"({walk_index + 1}/{num_walks})",
                end="",
                flush=True,
            )

        sample = sample_on_vgss(
            master_vgss.sampling_context,
            start_rng,
            stats=stats,
        )
        start_point = sample.point
        normal = sample.normal
        distances = [
            conductor.distance_to_point(start_point)
            for conductor in conductors
        ]
        first_half_size = min(distances)
        if not math.isfinite(first_half_size) or first_half_size <= 0.0:
            raise RuntimeError(
                "A VGSS start point produced a non-positive first cube size: "
                f"{first_half_size}."
            )

        path_seed = int(
            start_rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )

        results = run_coupled_paths(
            start_point,
            normal,
            first_half_size,
            np.random.default_rng(path_seed),
            numerical,
            analytical,
            coupler,
            conductors,
            conductor_to_net,
            system_center,
            kill_radius,
            max_hops,
            hit_tolerance,
        )
        numerical_result = results["numerical"]
        analytical_first_result = results[
            "analytical_first_numerical_rest"
        ]
        analytical_all_result = results["analytical_all"]
        for mode, result in results.items():
            mode_accumulators[mode].add(
                result.target,
                result.omega,
                result.termination,
                result.hops,
            )
        for name, baseline, candidate in COMPARISONS:
            result_a = results[baseline]
            result_b = results[candidate]
            difference_accumulators[name].add(
                result_a.target,
                result_a.omega,
                result_b.target,
                result_b.omega,
            )

        agreement["numerical_vs_analytical_first_first_panel"] += int(
            numerical_result.first_panel_index
            == analytical_first_result.first_panel_index
        )
        agreement["numerical_vs_analytical_first_target"] += int(
            numerical_result.target == analytical_first_result.target
        )
        agreement["analytical_first_vs_analytical_all_target"] += int(
            analytical_first_result.target == analytical_all_result.target
        )
        agreement["numerical_vs_analytical_all_target"] += int(
            numerical_result.target == analytical_all_result.target
        )
        agreement["numerical_vs_analytical_first_termination"] += int(
            numerical_result.termination
            == analytical_first_result.termination
        )
        agreement["analytical_first_vs_analytical_all_termination"] += int(
            analytical_first_result.termination
            == analytical_all_result.termination
        )
        agreement["numerical_vs_analytical_all_termination"] += int(
            numerical_result.termination
            == analytical_all_result.termination
        )

    print()
    context = master_vgss.sampling_context
    H = stats.estimate_H(context)
    Sg = stats.estimate_surface_area(context)
    mean_p = stats.estimate_mean_p()
    physical_scale = E0 * L_UNIT * H

    mode_summaries = {
        mode: physical_summary(accumulator, physical_scale)
        for mode, accumulator in mode_accumulators.items()
    }
    difference_summaries = {}
    for name, baseline, _candidate in COMPARISONS:
        difference_summaries[name] = physical_difference_summary(
            difference_accumulators[name],
            physical_scale,
            mode_summaries[baseline]["estimates_F"],
        )

    return {
        "master_net": master_vgss.net.name,
        "num_walks": num_walks,
        "H": H,
        "Sg": Sg,
        "mean_p": mean_p,
        "sampling_stats": stats.as_dict(context),
        "modes": mode_summaries,
        "paired_differences": difference_summaries,
        "agreement_rates": {
            key: value / num_walks for key, value in agreement.items()
        },
    }


def matrix_from_rows(
    rows: list[dict[str, Any]],
    mode: str,
    net_order: list[str],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    matrix = np.zeros((len(net_order), len(net_order)), dtype=np.float64)
    se = np.zeros_like(matrix)
    infinity = np.zeros(len(net_order), dtype=np.float64)
    for i, row in enumerate(rows):
        summary = row["modes"][mode]
        for j, target in enumerate(net_order):
            matrix[i, j] = summary["estimates_F"][target]
            se[i, j] = summary[
                "standard_errors_F_conditional_on_H"
            ][target]
        infinity[i] = summary["estimates_F"][INFINITY]
    return matrix, se, infinity


def difference_matrix_from_rows(
    rows: list[dict[str, Any]],
    comparison: str,
    net_order: list[str],
) -> tuple[FloatArray, FloatArray]:
    matrix = np.zeros((len(net_order), len(net_order)), dtype=np.float64)
    se = np.zeros_like(matrix)
    for i, row in enumerate(rows):
        summary = row["paired_differences"][comparison]
        for j, target in enumerate(net_order):
            matrix[i, j] = summary["delta_F"][target]
            se[i, j] = summary[
                "paired_standard_error_F_conditional_on_H"
            ][target]
    return matrix, se


def aggregate_seed_arrays(
    values: list[FloatArray],
    within_seed_se: list[FloatArray],
) -> dict[str, FloatArray]:
    stack = np.stack(values)
    se_stack = np.stack(within_seed_se)
    seed_count = stack.shape[0]
    mean = stack.mean(axis=0)
    if seed_count > 1:
        between_sd = stack.std(axis=0, ddof=1)
        between_se = between_sd / math.sqrt(seed_count)
    else:
        between_sd = np.zeros_like(mean)
        between_se = np.zeros_like(mean)
    combined_within_se = np.sqrt(np.sum(se_stack**2, axis=0)) / seed_count
    return {
        "mean": mean,
        "between_seed_sd": between_sd,
        "between_seed_se_of_mean": between_se,
        "combined_within_seed_se_of_mean": combined_within_se,
    }


def matrix_diagnostics(matrix: FloatArray) -> dict[str, float]:
    asymmetry = matrix - matrix.T
    max_entry = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    max_asymmetry = float(np.max(np.abs(asymmetry))) if matrix.size else 0.0
    return {
        "max_abs_asymmetry_F": max_asymmetry,
        "relative_asymmetry": (
            max_asymmetry / max_entry if max_entry > 0.0 else 0.0
        ),
    }


def write_matrix_csv(
    path: Path,
    aggregate_modes: dict[str, Any],
    master_order: list[str],
    target_order: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    for mode, data in aggregate_modes.items():
        mean = np.asarray(data["matrix"]["mean"])
        between_se = np.asarray(data["matrix"]["between_seed_se_of_mean"])
        within_se = np.asarray(
            data["matrix"]["combined_within_seed_se_of_mean"]
        )
        for i, master in enumerate(master_order):
            for j, target in enumerate(target_order):
                rows.append(
                    {
                        "mode": mode,
                        "master_net": master,
                        "target_net": target,
                        "estimate_F": mean[i, j],
                        "between_seed_SE_of_mean_F": between_se[i, j],
                        "combined_within_seed_SE_of_mean_F": within_se[i, j],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_difference_csv(
    path: Path,
    aggregate_differences: dict[str, Any],
    master_order: list[str],
    target_order: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    for comparison, data in aggregate_differences.items():
        delta = np.asarray(data["matrix"]["mean"])
        between_se = np.asarray(data["matrix"]["between_seed_se_of_mean"])
        within_se = np.asarray(
            data["matrix"]["combined_within_seed_se_of_mean"]
        )
        for i, master in enumerate(master_order):
            for j, target in enumerate(target_order):
                conservative_se = max(between_se[i, j], within_se[i, j])
                rows.append(
                    {
                        "comparison": comparison,
                        "master_net": master,
                        "target_net": target,
                        "paired_delta_F": delta[i, j],
                        "between_seed_SE_of_mean_F": between_se[i, j],
                        "combined_within_seed_paired_SE_of_mean_F": within_se[i, j],
                        "delta_z_using_larger_SE": (
                            delta[i, j] / conservative_se
                            if conservative_se > 0.0
                            else 0.0
                        ),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_agreement_csv(path: Path, seed_results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for seed_result in seed_results:
        seed = seed_result["base_seed"]
        for row in seed_result["rows"]:
            item = {
                "base_seed": seed,
                "master_net": row["master_net"],
                "H": row["H"],
                "Sg": row["Sg"],
                "mean_p": row["mean_p"],
            }
            item.update(row["agreement_rates"])
            for mode in MODE_NAMES:
                mode_summary = row["modes"][mode]
                item[f"{mode}_resolved_rate"] = mode_summary["resolved_rate"]
                item[f"{mode}_mean_hops"] = mode_summary["mean_hops"]
                item[f"{mode}_row_sum_F"] = mode_summary[
                    "row_sum_with_infinity_F"
                ]
                item[f"{mode}_row_sum_z"] = mode_summary[
                    "row_sum_z_conditional_on_H"
                ]
            rows.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.walks_per_seed <= 1:
        raise SystemExit("--walks-per-seed must be greater than 1.")
    if not args.seeds:
        raise SystemExit("At least one seed is required.")
    if args.max_hops <= 0:
        raise SystemExit("--max-hops must be positive.")
    if args.hit_tolerance <= 0.0:
        raise SystemExit("--hit-tolerance must be positive.")
    if args.kill_multiplier <= 0.0:
        raise SystemExit("--kill-multiplier must be positive.")

    numerical = load_numerical_table(args.table)
    print(
        f"Building independent analytical table for n={numerical.resolution}, "
        f"terms={args.series_terms}..."
    )
    analytical = build_analytical_table(
        numerical.resolution, args.series_terms
    )
    validate_panel_alignment(numerical, analytical)
    coupler = MaximalCategoricalCoupler.build(
        numerical.probabilities, analytical.probabilities
    )

    probability_l1 = float(
        np.sum(np.abs(numerical.probabilities - analytical.probabilities))
    )
    numerical_wvt_from_fields = (
        numerical.gradients / numerical.probabilities[:, None]
    )
    numerical_wvt_consistency_max_abs = float(
        np.max(np.abs(numerical.wvt - numerical_wvt_from_fields))
    )
    numerical_wvt_consistency_relative_rms = math.sqrt(
        float(
            np.sum(
                numerical.probabilities[:, None]
                * (numerical.wvt - numerical_wvt_from_fields) ** 2
            )
        )
        / float(
            np.sum(
                numerical.probabilities[:, None]
                * numerical_wvt_from_fields**2
            )
        )
    )
    wvt_relative_rms = math.sqrt(
        float(
            np.sum(
                analytical.probabilities[:, None]
                * (numerical.wvt - analytical.wvt) ** 2
            )
        )
        / float(
            np.sum(
                analytical.probabilities[:, None] * analytical.wvt**2
            )
        )
    )

    conductors = parse_matlab_geometry(args.input_file)
    nets = find_touching_nets(conductors)
    all_vgss = build_all_net_vgss(
        conductors,
        scale_factor=args.scale_factor,
        max_distance=args.max_distance,
    )
    available = [item.net.name for item in all_vgss]
    if args.masters:
        unknown = sorted(set(args.masters).difference(available))
        if unknown:
            raise SystemExit(
                f"Unknown master nets {unknown}; available: {available}"
            )
        selected_vgss = [
            item for item in all_vgss if item.net.name in args.masters
        ]
    else:
        selected_vgss = list(all_vgss)

    net_order = [net.name for net in nets]
    conductor_to_net = {
        conductor.name: net.name
        for net in nets
        for conductor in net.conductors
    }
    system_center, system_size = get_system_bounds(conductors)
    kill_radius = system_size * args.kill_multiplier

    print("Input:", args.input_file)
    print("Numerical table:", args.table)
    print("Resolution:", numerical.resolution)
    print("Analytical series terms:", args.series_terms)
    print("Numerical/reference probability L1 distance:", probability_l1)
    print(
        "Maximal-coupling common panel probability:",
        coupler.common_mass,
    )
    print(
        "Probability total-variation distance:",
        coupler.total_variation_distance,
    )
    print(
        "Stored WVT vs gradients/probabilities relative RMS:",
        numerical_wvt_consistency_relative_rms,
    )
    print(
        "Stored WVT vs gradients/probabilities max abs:",
        numerical_wvt_consistency_max_abs,
    )
    print("Numerical/reference WVT weighted relative RMS:", wvt_relative_rms)
    print("Net order:", net_order)
    print("Selected masters:", [item.net.name for item in selected_vgss])
    print("Walks per seed per master:", args.walks_per_seed)
    print("Seeds:", args.seeds)
    print(
        "Base walk starts:",
        args.walks_per_seed * len(args.seeds) * len(selected_vgss),
    )
    print(
        "Path simulations (3 per start):",
        3 * args.walks_per_seed * len(args.seeds) * len(selected_vgss),
    )

    seed_results: list[dict[str, Any]] = []
    mode_matrices: dict[str, list[FloatArray]] = {
        mode: [] for mode in MODE_NAMES
    }
    mode_ses: dict[str, list[FloatArray]] = {
        mode: [] for mode in MODE_NAMES
    }
    mode_infinity: dict[str, list[FloatArray]] = {
        mode: [] for mode in MODE_NAMES
    }
    mode_infinity_ses: dict[str, list[FloatArray]] = {
        mode: [] for mode in MODE_NAMES
    }
    difference_matrices: dict[str, list[FloatArray]] = {
        name: [] for name, _a, _b in COMPARISONS
    }
    difference_ses: dict[str, list[FloatArray]] = {
        name: [] for name, _a, _b in COMPARISONS
    }

    for base_seed in args.seeds:
        print("\n" + "=" * 78)
        print(f"BASE SEED {base_seed}")
        print("=" * 78)
        seed_sequence = np.random.SeedSequence(base_seed)
        child_sequences = seed_sequence.spawn(len(selected_vgss))
        rows = []
        for master_vgss, child in zip(selected_vgss, child_sequences):
            row = run_master_seed(
                conductors,
                master_vgss,
                np.random.default_rng(child),
                numerical,
                analytical,
                coupler,
                net_order,
                conductor_to_net,
                system_center,
                kill_radius,
                args.walks_per_seed,
                args.max_hops,
                args.hit_tolerance,
            )
            rows.append(row)

        # Matrix output requires every master row. For a selected subset, keep
        # rectangular selected-master by all-target matrices.
        selected_order = [row["master_net"] for row in rows]
        for mode in MODE_NAMES:
            matrix = np.zeros(
                (len(selected_order), len(net_order)), dtype=np.float64
            )
            se = np.zeros_like(matrix)
            infinity = np.zeros(len(selected_order), dtype=np.float64)
            infinity_se = np.zeros(len(selected_order), dtype=np.float64)
            for i, row in enumerate(rows):
                summary = row["modes"][mode]
                for j, target in enumerate(net_order):
                    matrix[i, j] = summary["estimates_F"][target]
                    se[i, j] = summary[
                        "standard_errors_F_conditional_on_H"
                    ][target]
                infinity[i] = summary["estimates_F"][INFINITY]
                infinity_se[i] = summary[
                    "standard_errors_F_conditional_on_H"
                ][INFINITY]
            mode_matrices[mode].append(matrix)
            mode_ses[mode].append(se)
            mode_infinity[mode].append(infinity)
            mode_infinity_ses[mode].append(infinity_se)

        for comparison, _a, _b in COMPARISONS:
            matrix = np.zeros(
                (len(selected_order), len(net_order)), dtype=np.float64
            )
            se = np.zeros_like(matrix)
            for i, row in enumerate(rows):
                summary = row["paired_differences"][comparison]
                for j, target in enumerate(net_order):
                    matrix[i, j] = summary["delta_F"][target]
                    se[i, j] = summary[
                        "paired_standard_error_F_conditional_on_H"
                    ][target]
            difference_matrices[comparison].append(matrix)
            difference_ses[comparison].append(se)

        seed_results.append(
            {
                "base_seed": base_seed,
                "selected_master_order": selected_order,
                "rows": rows,
            }
        )

    aggregate_modes: dict[str, Any] = {}
    for mode in MODE_NAMES:
        matrix_aggregate = aggregate_seed_arrays(
            mode_matrices[mode], mode_ses[mode]
        )
        infinity_aggregate = aggregate_seed_arrays(
            mode_infinity[mode], mode_infinity_ses[mode]
        )
        matrix_mean = matrix_aggregate["mean"]
        diagnostics = (
            matrix_diagnostics(matrix_mean)
            if matrix_mean.shape[0] == matrix_mean.shape[1]
            else {}
        )
        aggregate_modes[mode] = {
            "matrix": matrix_aggregate,
            "infinity_column": infinity_aggregate,
            "diagnostics": diagnostics,
        }

    aggregate_differences: dict[str, Any] = {}
    for comparison, baseline, candidate in COMPARISONS:
        matrix_aggregate = aggregate_seed_arrays(
            difference_matrices[comparison],
            difference_ses[comparison],
        )
        baseline_matrix = np.asarray(
            aggregate_modes[baseline]["matrix"]["mean"]
        )
        delta_matrix = np.asarray(matrix_aggregate["mean"])
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(
                baseline_matrix != 0.0,
                delta_matrix / baseline_matrix,
                np.nan,
            )
        aggregate_differences[comparison] = {
            "baseline_mode": baseline,
            "candidate_mode": candidate,
            "matrix": matrix_aggregate,
            "relative_delta_to_baseline": relative,
        }

    selected_master_order = seed_results[0]["selected_master_order"]
    output = {
        "input_file": str(args.input_file),
        "parameters": {
            "numerical_table": str(args.table),
            "resolution": numerical.resolution,
            "series_terms": args.series_terms,
            "walks_per_seed_per_master": args.walks_per_seed,
            "seeds": args.seeds,
            "total_walk_starts_per_master": (
                args.walks_per_seed * len(args.seeds)
            ),
            "scale_factor": args.scale_factor,
            "max_distance": args.max_distance,
            "max_hops": args.max_hops,
            "hit_tolerance": args.hit_tolerance,
            "kill_multiplier": args.kill_multiplier,
            "E0": E0,
            "L_UNIT": L_UNIT,
        },
        "table_comparison": {
            "probability_L1_distance": probability_l1,
            "probability_total_variation_distance": (
                coupler.total_variation_distance
            ),
            "maximal_coupling_common_panel_probability": (
                coupler.common_mass
            ),
            "stored_WVT_vs_gradients_over_probabilities_relative_RMS": (
                numerical_wvt_consistency_relative_rms
            ),
            "stored_WVT_vs_gradients_over_probabilities_max_abs": (
                numerical_wvt_consistency_max_abs
            ),
            "WVT_reference_probability_weighted_relative_RMS": (
                wvt_relative_rms
            ),
            "panel_center_max_absolute_difference": float(
                np.max(
                    np.abs(
                        numerical.panel_centers
                        - analytical.panel_centers
                    )
                )
            ),
        },
        "net_order": net_order,
        "selected_master_order": selected_master_order,
        "mode_definitions": {
            "numerical": (
                "Numerical probabilities for all transitions and stored "
                "numerical WVT for the first weight."
            ),
            "reference_gradient_same_path": (
                "Same numerical panel/path/target as numerical, with "
                "analytical first-gradient numerator divided by numerical "
                "panel probability."
            ),
            "analytical_first_numerical_rest": (
                "Analytical first probability and weight; numerical "
                "probabilities after the first transition."
            ),
            "analytical_all": (
                "Analytical probabilities for all transitions and analytical "
                "first weight."
            ),
        },
        "seed_results": seed_results,
        "aggregate_modes": aggregate_modes,
        "aggregate_paired_differences": aggregate_differences,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.label}_summary.json"
    matrix_csv = args.output_dir / f"{args.label}_matrices.csv"
    difference_csv = args.output_dir / f"{args.label}_paired_differences.csv"
    agreement_csv = args.output_dir / f"{args.label}_agreement.csv"
    json_path.write_text(
        json.dumps(_to_builtin(output), indent=2), encoding="utf-8"
    )
    write_matrix_csv(
        matrix_csv,
        aggregate_modes,
        selected_master_order,
        net_order,
    )
    write_difference_csv(
        difference_csv,
        aggregate_differences,
        selected_master_order,
        net_order,
    )
    write_agreement_csv(agreement_csv, seed_results)

    print("\n" + "#" * 78)
    print("AGGREGATE MATRICES (F)")
    print("#" * 78)
    for mode in MODE_NAMES:
        print(f"\n{mode}:")
        print(
            np.array2string(
                np.asarray(aggregate_modes[mode]["matrix"]["mean"]),
                precision=10,
                suppress_small=False,
            )
        )
    print("\nPAIRED DIFFERENCES (candidate - baseline), F")
    for comparison, _baseline, _candidate in COMPARISONS:
        print(f"\n{comparison}:")
        print(
            np.array2string(
                np.asarray(
                    aggregate_differences[comparison]["matrix"]["mean"]
                ),
                precision=5,
                suppress_small=False,
            )
        )
    print("\nOutputs:")
    print(" ", json_path)
    print(" ", matrix_csv)
    print(" ", difference_csv)
    print(" ", agreement_csv)


if __name__ == "__main__":
    main()
