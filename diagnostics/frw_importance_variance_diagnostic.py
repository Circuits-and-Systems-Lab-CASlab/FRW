#!/usr/bin/env python3
"""Low-memory FRW variance diagnostic with first-transition importance sampling.

This is a local diagnostic, not production code. It compares three unbiased
panel-discrete FRW estimators while keeping the geometry, VGSS sampling,
maximal-cube rule, numerical n200 transition table, conductor hit test, and
infinity test unchanged:

1. baseline
   Current production sampling: first panel is sampled from P_i and the weight
   is -n·grad(P_i)/(P_i h), equivalently the stored WVT divided by h.

2. abs_gradient_importance
   First panel is sampled from q_i proportional to |n·grad(P_i)|. The weight is
   -n·grad(P_i)/(q_i h). This is the standard first-transition importance
   distribution and makes the first weight constant in magnitude for a fixed
   surface normal and cube half-size.

3. sign_stratified_importance
   Uses the same |n·grad(P_i)| distribution, but forces a balanced number of
   positive- and negative-weight first panels. The allocation correction keeps
   the estimator unbiased and removes random sign-count imbalance.

Baseline and abs-gradient modes use maximal categorical coupling for their
first panels. All modes use the same subsequent random uniforms, which keeps
paired differences informative even after paths diverge. No walk paths are
stored.

The script can compare results with a compact JSON reference file extracted
from the user's Raphael, FasterCap, and Caplet workbook. MATLAB results are not
used.

Intended local locations inside the VGSS checkout:

    diagnostics/frw_importance_variance_diagnostic.py
    diagnostics/frw_reference_cases.json

The repository's /diagnostics/ and output/ ignore rules keep the files local.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

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
MODES = (
    "baseline",
    "abs_gradient_importance",
    "sign_stratified_importance",
)
TARGET_RELATIVE_SE = 0.01


@dataclass(frozen=True)
class TableData:
    resolution: int
    probabilities: FloatArray
    cdf: FloatArray
    panel_centers: FloatArray
    gradients: FloatArray
    wvt: FloatArray


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


@dataclass
class ModeAccumulator:
    labels: tuple[str, ...]
    contributions: list[ScalarMoments] = field(init=False)
    row_sum: ScalarMoments = field(default_factory=ScalarMoments)
    target_counts: list[int] = field(init=False)
    termination_counts: dict[str, int] = field(
        default_factory=lambda: {
            "hit_conductor": 0,
            "escaped": 0,
            "max_hops": 0,
        }
    )
    hop_sum: int = 0
    max_hops_observed: int = 0
    omega_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.contributions = [ScalarMoments() for _ in self.labels]
        self.target_counts = [0 for _ in self.labels]

    def add(
        self,
        target_index: int | None,
        omega: float,
        termination: str,
        hops: int,
    ) -> None:
        for index, moment in enumerate(self.contributions):
            moment.add(omega if target_index == index else 0.0)
        self.row_sum.add(omega if target_index is not None else 0.0)
        if target_index is not None:
            self.target_counts[target_index] += 1
        self.termination_counts[termination] += 1
        self.hop_sum += hops
        self.max_hops_observed = max(self.max_hops_observed, hops)
        self.omega_values.append(omega)


@dataclass
class DifferenceAccumulator:
    labels: tuple[str, ...]
    contributions: list[ScalarMoments] = field(init=False)
    row_sum: ScalarMoments = field(default_factory=ScalarMoments)

    def __post_init__(self) -> None:
        self.contributions = [ScalarMoments() for _ in self.labels]

    def add(
        self,
        target_a: int | None,
        omega_a: float,
        target_b: int | None,
        omega_b: float,
    ) -> None:
        for index, moment in enumerate(self.contributions):
            a = omega_a if target_a == index else 0.0
            b = omega_b if target_b == index else 0.0
            moment.add(b - a)
        self.row_sum.add(
            (omega_b if target_b is not None else 0.0)
            - (omega_a if target_a is not None else 0.0)
        )


@dataclass(frozen=True)
class PathResult:
    target_index: int | None
    termination: str
    hops: int


@dataclass(frozen=True)
class MaximalCategoricalCoupler:
    common_mass: float
    common_cdf: FloatArray
    p_residual_cdf: FloatArray
    q_residual_cdf: FloatArray

    @staticmethod
    def _normalized_cdf(values: FloatArray) -> FloatArray:
        total = float(values.sum())
        if total <= 1.0e-15:
            cdf = np.zeros_like(values)
            cdf[-1] = 1.0
            return cdf
        cdf = np.cumsum(np.maximum(values, 0.0) / total)
        cdf[-1] = 1.0
        return cdf

    @classmethod
    def build(
        cls, p: FloatArray, q: FloatArray
    ) -> "MaximalCategoricalCoupler":
        common = np.minimum(p, q)
        common_mass = float(common.sum())
        return cls(
            common_mass=common_mass,
            common_cdf=cls._normalized_cdf(common),
            p_residual_cdf=cls._normalized_cdf(p - common),
            q_residual_cdf=cls._normalized_cdf(q - common),
        )

    @staticmethod
    def _sample_cdf(cdf: FloatArray, u: float) -> int:
        return int(np.searchsorted(cdf, u, side="right"))

    def sample(
        self, rng: np.random.Generator
    ) -> tuple[int, int, bool]:
        if float(rng.random()) < self.common_mass:
            index = self._sample_cdf(self.common_cdf, float(rng.random()))
            return index, index, True
        p_index = self._sample_cdf(
            self.p_residual_cdf, float(rng.random())
        )
        q_index = self._sample_cdf(
            self.q_residual_cdf, float(rng.random())
        )
        return p_index, q_index, False


@dataclass(frozen=True)
class NormalImportanceData:
    normal_key: str
    raw_numerator: FloatArray
    q_abs: FloatArray
    q_abs_cdf: FloatArray
    positive_indices: NDArray[np.int64]
    positive_cdf: FloatArray
    positive_mass: float
    negative_indices: NDArray[np.int64]
    negative_cdf: FloatArray
    negative_mass: float
    coupler: MaximalCategoricalCoupler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current FRW first-transition sampling with numerical "
            "absolute-gradient importance sampling and sign stratification."
        )
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=Path("../Conductor_Cases"),
        help="Directory containing the conductor-case folders.",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=Path("diagnostics/frw_reference_cases.json"),
        help="Reference matrices extracted from the comparison workbook.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
        help="Test-case numbers to run. Default: 1 2 3 4 5 6 7 8.",
    )
    parser.add_argument(
        "--masters",
        nargs="+",
        default=["Net3"],
        help="Master nets to run, or 'all'. Default: Net3.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("src/GGFT/data/cube_tables_n201.npz"),
    )
    parser.add_argument("--walks-per-seed", type=int, default=2500)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1234, 5678, 9012, 3456],
    )
    parser.add_argument("--scale-factor", type=float, default=1.25)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--max-hops", type=int, default=200)
    parser.add_argument("--hit-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--kill-multiplier", type=float, default=500_000.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/diagnostics/frw_importance_variance"),
    )
    parser.add_argument(
        "--label",
        default="importance_variance",
    )
    return parser


def load_table(path: Path) -> TableData:
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
            raise ValueError(f"{path}: missing fields: {missing}")
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

    expected = 6 * resolution * resolution
    if probabilities.shape != (expected,):
        raise ValueError("Unexpected probability shape.")
    if panel_centers.shape != (expected, 3):
        raise ValueError("Unexpected panel-centre shape.")
    if gradients.shape != (expected, 3) or wvt.shape != (expected, 3):
        raise ValueError("Unexpected gradient/WVT shape.")
    if np.any(probabilities <= 0.0):
        raise ValueError("All panel probabilities must be positive.")

    probabilities = probabilities / float(probabilities.sum())
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    return TableData(
        resolution=resolution,
        probabilities=probabilities,
        cdf=cdf,
        panel_centers=panel_centers,
        gradients=gradients,
        wvt=wvt,
    )


def normal_key(normal: FloatArray) -> str:
    axis = int(np.argmax(np.abs(normal)))
    sign = "+" if normal[axis] > 0.0 else "-"
    return f"{'xyz'[axis]}{sign}"


def unit_normal_from_key(key: str) -> FloatArray:
    result = np.zeros(3, dtype=np.float64)
    axis = "xyz".index(key[0])
    result[axis] = 1.0 if key[1] == "+" else -1.0
    return result


def normalized_cdf(values: FloatArray) -> FloatArray:
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Cannot normalize a zero categorical distribution.")
    cdf = np.cumsum(values / total)
    cdf[-1] = 1.0
    return cdf


def build_importance_data(
    table: TableData,
) -> dict[str, NormalImportanceData]:
    result: dict[str, NormalImportanceData] = {}
    for key in ("x+", "x-", "y+", "y-", "z+", "z-"):
        normal = unit_normal_from_key(key)
        raw = -(table.gradients @ normal)
        absolute = np.abs(raw)
        q_abs = absolute / float(absolute.sum())
        q_abs_cdf = np.cumsum(q_abs)
        q_abs_cdf[-1] = 1.0

        pos = np.flatnonzero(raw > 0.0)
        neg = np.flatnonzero(raw < 0.0)
        pos_values = raw[pos]
        neg_values = -raw[neg]
        pos_mass = float(pos_values.sum())
        neg_mass = float(neg_values.sum())
        if pos_mass <= 0.0 or neg_mass <= 0.0:
            raise RuntimeError(f"Missing a weight sign for normal {key}.")

        result[key] = NormalImportanceData(
            normal_key=key,
            raw_numerator=raw,
            q_abs=q_abs,
            q_abs_cdf=q_abs_cdf,
            positive_indices=pos,
            positive_cdf=normalized_cdf(pos_values),
            positive_mass=pos_mass,
            negative_indices=neg,
            negative_cdf=normalized_cdf(neg_values),
            negative_mass=neg_mass,
            coupler=MaximalCategoricalCoupler.build(
                table.probabilities, q_abs
            ),
        )
    return result


def sample_cdf(cdf: FloatArray, u: float) -> int:
    return int(np.searchsorted(cdf, u, side="right"))


def simulate_after_first(
    first_point: FloatArray,
    hop_uniforms: FloatArray,
    *,
    conductors,
    conductor_to_target_index: dict[str, int],
    table: TableData,
    system_center: FloatArray,
    kill_radius: float,
    hit_tolerance: float,
) -> PathResult:
    current = np.asarray(first_point, dtype=np.float64).copy()
    for hop_index, u in enumerate(hop_uniforms):
        if np.max(np.abs(current - system_center)) > kill_radius:
            return PathResult(
                target_index=conductor_to_target_index[INFINITY],
                termination="escaped",
                hops=hop_index + 1,
            )

        distances = [c.distance_to_point(current) for c in conductors]
        min_distance = min(distances)
        nearest_index = int(np.argmin(distances))
        if min_distance < hit_tolerance:
            name = conductors[nearest_index].name
            return PathResult(
                target_index=conductor_to_target_index[name],
                termination="hit_conductor",
                hops=hop_index + 1,
            )

        panel_index = sample_cdf(table.cdf, float(u))
        current = (
            current + min_distance * table.panel_centers[panel_index]
        )

    return PathResult(
        target_index=None,
        termination="max_hops",
        hops=len(hop_uniforms) + 1,
    )


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"min": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    q = np.quantile(array, [0.0, 0.05, 0.5, 0.95, 1.0])
    return {
        "min": float(q[0]),
        "p05": float(q[1]),
        "median": float(q[2]),
        "p95": float(q[3]),
        "max": float(q[4]),
    }


def row_result(
    accumulator: ModeAccumulator,
    *,
    H: float,
    walk_count: int,
) -> dict[str, Any]:
    scale = E0 * L_UNIT * H
    estimates = [scale * m.mean() for m in accumulator.contributions]
    standard_errors = [
        abs(scale) * m.standard_error_of_mean()
        for m in accumulator.contributions
    ]
    row_sum = scale * accumulator.row_sum.mean()
    row_sum_se = abs(scale) * accumulator.row_sum.standard_error_of_mean()
    return {
        "estimates_F": estimates,
        "standard_errors_F_conditional_on_H": standard_errors,
        "row_sum_with_infinity_F": row_sum,
        "row_sum_standard_error_F_conditional_on_H": row_sum_se,
        "row_sum_z_conditional_on_H": (
            row_sum / row_sum_se if row_sum_se > 0.0 else 0.0
        ),
        "target_counts": accumulator.target_counts,
        "termination_counts": accumulator.termination_counts,
        "resolved_rate": 1.0 - accumulator.termination_counts["max_hops"] / walk_count,
        "mean_hops": accumulator.hop_sum / walk_count,
        "max_hops_observed": accumulator.max_hops_observed,
        "omega_quantiles": quantiles(accumulator.omega_values),
        "omega_standard_deviation": float(
            np.std(accumulator.omega_values, ddof=1)
            if len(accumulator.omega_values) > 1
            else 0.0
        ),
    }


def difference_result(
    accumulator: DifferenceAccumulator,
    *,
    H: float,
) -> dict[str, Any]:
    scale = E0 * L_UNIT * H
    delta = [scale * m.mean() for m in accumulator.contributions]
    se = [
        abs(scale) * m.standard_error_of_mean()
        for m in accumulator.contributions
    ]
    return {
        "delta_F": delta,
        "paired_standard_errors_F_conditional_on_H": se,
        "paired_z_conditional_on_H": [
            d / s if s > 0.0 else 0.0 for d, s in zip(delta, se)
        ],
        "row_sum_delta_F": scale * accumulator.row_sum.mean(),
        "row_sum_paired_standard_error_F_conditional_on_H": (
            abs(scale) * accumulator.row_sum.standard_error_of_mean()
        ),
    }


def run_master_seed(
    *,
    conductors,
    master_vgss,
    table: TableData,
    importance: dict[str, NormalImportanceData],
    net_order: list[str],
    conductor_to_net: dict[str, str],
    rng: np.random.Generator,
    walk_count: int,
    max_hops: int,
    system_center: FloatArray,
    kill_radius: float,
    hit_tolerance: float,
) -> dict[str, Any]:
    labels = tuple(net_order + [INFINITY])
    label_to_index = {label: i for i, label in enumerate(labels)}
    conductor_to_target = {
        name: label_to_index[net]
        for name, net in conductor_to_net.items()
    }
    conductor_to_target[INFINITY] = label_to_index[INFINITY]

    accumulators = {
        mode: ModeAccumulator(labels) for mode in MODES
    }
    paired = {
        "importance_minus_baseline": DifferenceAccumulator(labels),
        "stratified_minus_baseline": DifferenceAccumulator(labels),
        "stratified_minus_importance": DifferenceAccumulator(labels),
    }
    sampling_stats = VGSSSamplingStats()
    first_half_sizes: list[float] = []
    common_first_panels = 0
    same_target_baseline_importance = 0
    same_target_baseline_stratified = 0

    positive_count = (walk_count + 1) // 2
    negative_count = walk_count // 2
    positive_fraction = positive_count / walk_count
    negative_fraction = negative_count / walk_count

    progress_step = max(1, walk_count // 20)
    for walk_index in range(walk_count):
        if (walk_index + 1) % progress_step == 0 or walk_index + 1 == walk_count:
            print(
                f"\r    {master_vgss.net.name}: "
                f"{100.0 * (walk_index + 1) / walk_count:5.1f}%",
                end="",
                flush=True,
            )

        start = sample_on_vgss(
            master_vgss.sampling_context,
            rng,
            stats=sampling_stats,
        )
        distances = [c.distance_to_point(start.point) for c in conductors]
        half_size = min(distances)
        if half_size <= 0.0:
            raise RuntimeError("First maximal cube has non-positive half-size.")
        first_half_sizes.append(half_size)

        key = normal_key(start.normal)
        info = importance[key]
        baseline_index, importance_index, common = info.coupler.sample(rng)
        common_first_panels += int(common)

        choose_positive = walk_index % 2 == 0
        if choose_positive:
            local = sample_cdf(info.positive_cdf, float(rng.random()))
            stratified_index = int(info.positive_indices[local])
            allocation = positive_fraction
            q_conditional = (
                info.raw_numerator[stratified_index] / info.positive_mass
            )
        else:
            local = sample_cdf(info.negative_cdf, float(rng.random()))
            stratified_index = int(info.negative_indices[local])
            allocation = negative_fraction
            q_conditional = (
                -info.raw_numerator[stratified_index] / info.negative_mass
            )

        baseline_omega = -float(
            np.dot(start.normal, table.wvt[baseline_index])
        ) / half_size
        importance_omega = (
            info.raw_numerator[importance_index]
            / info.q_abs[importance_index]
            / half_size
        )
        stratified_omega = (
            info.raw_numerator[stratified_index]
            / (allocation * q_conditional)
            / half_size
        )

        hop_uniforms = rng.random(max_hops)
        first_points = {
            "baseline": start.point
            + half_size * table.panel_centers[baseline_index],
            "abs_gradient_importance": start.point
            + half_size * table.panel_centers[importance_index],
            "sign_stratified_importance": start.point
            + half_size * table.panel_centers[stratified_index],
        }
        omegas = {
            "baseline": baseline_omega,
            "abs_gradient_importance": importance_omega,
            "sign_stratified_importance": stratified_omega,
        }
        paths = {
            mode: simulate_after_first(
                first_points[mode],
                hop_uniforms,
                conductors=conductors,
                conductor_to_target_index=conductor_to_target,
                table=table,
                system_center=system_center,
                kill_radius=kill_radius,
                hit_tolerance=hit_tolerance,
            )
            for mode in MODES
        }

        for mode in MODES:
            path = paths[mode]
            accumulators[mode].add(
                path.target_index,
                omegas[mode],
                path.termination,
                path.hops,
            )

        baseline = paths["baseline"]
        imp = paths["abs_gradient_importance"]
        strat = paths["sign_stratified_importance"]
        paired["importance_minus_baseline"].add(
            baseline.target_index,
            baseline_omega,
            imp.target_index,
            importance_omega,
        )
        paired["stratified_minus_baseline"].add(
            baseline.target_index,
            baseline_omega,
            strat.target_index,
            stratified_omega,
        )
        paired["stratified_minus_importance"].add(
            imp.target_index,
            importance_omega,
            strat.target_index,
            stratified_omega,
        )
        same_target_baseline_importance += int(
            baseline.target_index == imp.target_index
        )
        same_target_baseline_stratified += int(
            baseline.target_index == strat.target_index
        )

    print()
    context = master_vgss.sampling_context
    H = sampling_stats.estimate_H(context)
    return {
        "master_net": master_vgss.net.name,
        "walk_count": walk_count,
        "H": H,
        "Sg": sampling_stats.estimate_surface_area(context),
        "mean_p": sampling_stats.estimate_mean_p(),
        "sampling_stats": sampling_stats.as_dict(context),
        "first_half_size_quantiles": quantiles(first_half_sizes),
        "first_panel_common_rate_baseline_importance": (
            common_first_panels / walk_count
        ),
        "target_agreement_baseline_importance": (
            same_target_baseline_importance / walk_count
        ),
        "target_agreement_baseline_stratified": (
            same_target_baseline_stratified / walk_count
        ),
        "modes": {
            mode: row_result(
                accumulators[mode], H=H, walk_count=walk_count
            )
            for mode in MODES
        },
        "paired_differences": {
            name: difference_result(acc, H=H)
            for name, acc in paired.items()
        },
    }


def combine_seed_rows(
    rows: list[dict[str, Any]],
    target_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "master_net": rows[0]["master_net"],
        "H_mean": float(np.mean([r["H"] for r in rows])),
        "Sg_mean": float(np.mean([r["Sg"] for r in rows])),
        "first_half_size_quantiles_by_seed": [
            r["first_half_size_quantiles"] for r in rows
        ],
        "first_panel_common_rate_baseline_importance": float(
            np.mean(
                [r["first_panel_common_rate_baseline_importance"] for r in rows]
            )
        ),
        "target_agreement_baseline_importance": float(
            np.mean([r["target_agreement_baseline_importance"] for r in rows])
        ),
        "target_agreement_baseline_stratified": float(
            np.mean([r["target_agreement_baseline_stratified"] for r in rows])
        ),
        "modes": {},
        "paired_differences": {},
    }

    for mode in MODES:
        estimates = np.asarray(
            [r["modes"][mode]["estimates_F"] for r in rows], dtype=float
        )
        within_se = np.asarray(
            [
                r["modes"][mode]["standard_errors_F_conditional_on_H"]
                for r in rows
            ],
            dtype=float,
        )
        seed_count = len(rows)
        mean = estimates.mean(axis=0)
        between_sd = (
            estimates.std(axis=0, ddof=1)
            if seed_count > 1
            else np.zeros(target_count)
        )
        combined_within_se = np.sqrt(np.sum(within_se**2, axis=0)) / seed_count
        result["modes"][mode] = {
            "mean_estimates_F": mean.tolist(),
            "between_seed_sd_F": between_sd.tolist(),
            "between_seed_se_of_mean_F": (
                between_sd / math.sqrt(seed_count)
            ).tolist(),
            "combined_within_seed_se_of_mean_F": combined_within_se.tolist(),
            "mean_row_sum_with_infinity_F": float(
                np.mean([r["modes"][mode]["row_sum_with_infinity_F"] for r in rows])
            ),
            "mean_resolved_rate": float(
                np.mean([r["modes"][mode]["resolved_rate"] for r in rows])
            ),
            "mean_hops": float(
                np.mean([r["modes"][mode]["mean_hops"] for r in rows])
            ),
            "omega_sd_by_seed": [
                r["modes"][mode]["omega_standard_deviation"] for r in rows
            ],
        }

    for name in rows[0]["paired_differences"]:
        deltas = np.asarray(
            [r["paired_differences"][name]["delta_F"] for r in rows],
            dtype=float,
        )
        within = np.asarray(
            [
                r["paired_differences"][name][
                    "paired_standard_errors_F_conditional_on_H"
                ]
                for r in rows
            ],
            dtype=float,
        )
        seed_count = len(rows)
        mean = deltas.mean(axis=0)
        combined = np.sqrt(np.sum(within**2, axis=0)) / seed_count
        result["paired_differences"][name] = {
            "mean_delta_F": mean.tolist(),
            "combined_within_seed_se_of_mean_F": combined.tolist(),
            "paired_z": [
                d / s if s > 0.0 else 0.0
                for d, s in zip(mean, combined)
            ],
        }

    baseline_se = np.asarray(
        result["modes"]["baseline"]["combined_within_seed_se_of_mean_F"]
    )
    imp_se = np.asarray(
        result["modes"]["abs_gradient_importance"][
            "combined_within_seed_se_of_mean_F"
        ]
    )
    strat_se = np.asarray(
        result["modes"]["sign_stratified_importance"][
            "combined_within_seed_se_of_mean_F"
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        vrf_imp = np.where(imp_se > 0.0, (baseline_se / imp_se) ** 2, np.nan)
        vrf_strat = np.where(strat_se > 0.0, (baseline_se / strat_se) ** 2, np.nan)
    result["variance_reduction_factor_vs_baseline"] = {
        "abs_gradient_importance": vrf_imp.tolist(),
        "sign_stratified_importance": vrf_strat.tolist(),
    }
    return result


def relative_error(estimate: float, reference: float) -> float | None:
    if not math.isfinite(reference) or reference == 0.0:
        return None
    return abs(estimate - reference) / abs(reference)


def required_walks(
    estimate: float,
    se: float,
    current_walks: int,
    target_relative_se: float = TARGET_RELATIVE_SE,
) -> float | None:
    if estimate == 0.0 or se <= 0.0:
        return None
    relative = se / abs(estimate)
    return current_walks * (relative / target_relative_se) ** 2


def write_outputs(
    *,
    output_dir: Path,
    label: str,
    summary: dict[str, Any],
    references: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}_summary.json"
    entries_path = output_dir / f"{label}_entries.csv"
    cases_path = output_dir / f"{label}_case_summary.csv"
    paired_path = output_dir / f"{label}_paired_differences.csv"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    entry_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    total_walks = summary["parameters"]["total_walks_per_master"]

    for case_key, case_result in summary["cases"].items():
        reference_case = references["cases"][case_key]
        target_labels = summary["target_order"]
        historical = reference_case.get("historical_frw", {}).get("matrix")
        reference_matrices = reference_case["reference_matrices"]
        consensus = reference_case["consensus_median"]
        selected_rows = case_result["selected_master_rows"]

        case_vrfs_imp: list[float] = []
        case_vrfs_strat: list[float] = []
        case_baseline_errors: list[float] = []
        case_imp_errors: list[float] = []
        case_strat_errors: list[float] = []

        for master, combined in selected_rows.items():
            row_index = summary["net_order"].index(master)
            for target_index, target in enumerate(target_labels):
                if target == INFINITY:
                    continue
                ref_consensus = float(consensus[row_index][target_index])
                historical_value = (
                    float(historical[row_index][target_index])
                    if historical is not None
                    else None
                )
                vrf_imp = combined["variance_reduction_factor_vs_baseline"][
                    "abs_gradient_importance"
                ][target_index]
                vrf_strat = combined["variance_reduction_factor_vs_baseline"][
                    "sign_stratified_importance"
                ][target_index]
                if math.isfinite(vrf_imp):
                    case_vrfs_imp.append(vrf_imp)
                if math.isfinite(vrf_strat):
                    case_vrfs_strat.append(vrf_strat)

                for mode in MODES:
                    mode_result = combined["modes"][mode]
                    estimate = float(mode_result["mean_estimates_F"][target_index])
                    se = float(
                        mode_result["combined_within_seed_se_of_mean_F"][target_index]
                    )
                    error_consensus = relative_error(estimate, ref_consensus)
                    if error_consensus is not None:
                        if mode == "baseline":
                            case_baseline_errors.append(error_consensus)
                        elif mode == "abs_gradient_importance":
                            case_imp_errors.append(error_consensus)
                        else:
                            case_strat_errors.append(error_consensus)
                    row = {
                        "case": int(case_key),
                        "spacing": reference_case["spacing"],
                        "master": master,
                        "target": target,
                        "mode": mode,
                        "BGS_area": combined["Sg_mean"],
                        "estimate_F": estimate,
                        "SE_F": se,
                        "relative_SE": se / abs(estimate) if estimate else None,
                        "walks_for_1pct_relative_SE": required_walks(
                            estimate, se, total_walks
                        ),
                        "consensus_reference_F": ref_consensus,
                        "absolute_relative_error_vs_consensus": error_consensus,
                        "historical_FRW_F": historical_value,
                        "VRF_abs_gradient_vs_baseline": vrf_imp,
                        "VRF_stratified_vs_baseline": vrf_strat,
                    }
                    for tool, matrix in reference_matrices.items():
                        reference_value = float(matrix[row_index][target_index])
                        row[f"{tool}_F"] = reference_value
                        row[f"absolute_relative_error_vs_{tool}"] = relative_error(
                            estimate, reference_value
                        )
                    entry_rows.append(row)

            for pair_name, pair in combined["paired_differences"].items():
                for target_index, target in enumerate(target_labels):
                    paired_rows.append(
                        {
                            "case": int(case_key),
                            "spacing": reference_case["spacing"],
                            "master": master,
                            "target": target,
                            "comparison": pair_name,
                            "mean_delta_F": pair["mean_delta_F"][target_index],
                            "paired_SE_F": pair[
                                "combined_within_seed_se_of_mean_F"
                            ][target_index],
                            "paired_z": pair["paired_z"][target_index],
                        }
                    )

        def median_or_none(values: list[float]) -> float | None:
            return float(np.median(values)) if values else None

        case_rows.append(
            {
                "case": int(case_key),
                "spacing": reference_case["spacing"],
                "selected_masters": " ".join(selected_rows),
                "mean_BGS_area": float(
                    np.mean([row["Sg_mean"] for row in selected_rows.values()])
                ),
                "median_VRF_abs_gradient": median_or_none(case_vrfs_imp),
                "median_VRF_sign_stratified": median_or_none(case_vrfs_strat),
                "median_abs_relative_error_baseline_vs_consensus": median_or_none(
                    case_baseline_errors
                ),
                "median_abs_relative_error_importance_vs_consensus": median_or_none(
                    case_imp_errors
                ),
                "median_abs_relative_error_stratified_vs_consensus": median_or_none(
                    case_strat_errors
                ),
            }
        )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(entries_path, entry_rows)
    write_csv(cases_path, case_rows)
    write_csv(paired_path, paired_rows)

    print("\nOutputs:")
    for path in (json_path, entries_path, cases_path, paired_path):
        print(" ", path)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = build_parser().parse_args()
    if args.walks_per_seed <= 1:
        raise SystemExit("--walks-per-seed must be greater than 1.")
    if not args.seeds:
        raise SystemExit("At least one seed is required.")
    if args.max_hops <= 0:
        raise SystemExit("--max-hops must be positive.")
    if not args.reference_json.is_file():
        raise FileNotFoundError(
            f"Reference JSON not found: {args.reference_json}"
        )

    references = json.loads(args.reference_json.read_text(encoding="utf-8"))
    table = load_table(args.table)
    importance = build_importance_data(table)

    print("GGFT resolution:", table.resolution)
    for key, data in importance.items():
        print(
            f"  {key}: baseline/importance common panel probability="
            f"{data.coupler.common_mass:.6f}, "
            f"positive mass={data.positive_mass:.10f}, "
            f"negative mass={data.negative_mass:.10f}"
        )

    requested_cases = [str(case) for case in args.cases]
    missing = [case for case in requested_cases if case not in references["cases"]]
    if missing:
        raise KeyError(f"Cases absent from reference JSON: {missing}")

    summary: dict[str, Any] = {
        "description": (
            "FRW first-transition variance and importance-sampling diagnostic."
        ),
        "parameters": {
            "cases_root": str(args.cases_root),
            "reference_json": str(args.reference_json),
            "table": str(args.table),
            "resolution": table.resolution,
            "cases": args.cases,
            "requested_masters": args.masters,
            "walks_per_seed": args.walks_per_seed,
            "seeds": args.seeds,
            "total_walks_per_master": args.walks_per_seed * len(args.seeds),
            "scale_factor": args.scale_factor,
            "max_distance": args.max_distance,
            "max_hops": args.max_hops,
            "hit_tolerance": args.hit_tolerance,
            "kill_multiplier": args.kill_multiplier,
            "E0": E0,
            "L_UNIT": L_UNIT,
        },
        "mode_definitions": {
            "baseline": (
                "Production first-panel probability and WVT weight."
            ),
            "abs_gradient_importance": (
                "First panel sampled proportional to absolute numerical "
                "normal gradient, with exact importance correction."
            ),
            "sign_stratified_importance": (
                "Absolute-gradient first-panel sampling with balanced positive "
                "and negative first-weight strata."
            ),
        },
        "target_order": [f"Net{i}" for i in range(1, len(references["matrix_order"]) + 1)] + [INFINITY],
        "net_order": [f"Net{i}" for i in range(1, len(references["matrix_order"]) + 1)],
        "importance_table_diagnostics": {
            key: {
                "baseline_importance_common_mass": data.coupler.common_mass,
                "positive_gradient_mass": data.positive_mass,
                "negative_gradient_mass": data.negative_mass,
            }
            for key, data in importance.items()
        },
        "cases": {},
    }

    for case_key in requested_cases:
        case_reference = references["cases"][case_key]
        geometry = args.cases_root / case_reference["geometry_file"]
        if not geometry.is_file():
            raise FileNotFoundError(
                f"Case {case_key} geometry not found: {geometry}"
            )
        conductors = parse_matlab_geometry(geometry)
        nets = find_touching_nets(conductors)
        net_order = [net.name for net in nets]
        if len(net_order) != len(references["matrix_order"]):
            raise ValueError(
                f"Case {case_key}: found {len(net_order)} nets but the reference "
                f"matrix contains {len(references['matrix_order'])} rows."
            )
        all_vgss = build_all_net_vgss(
            conductors,
            scale_factor=args.scale_factor,
            max_distance=args.max_distance,
        )
        vgss_by_name = {item.net.name: item for item in all_vgss}
        if len(args.masters) == 1 and args.masters[0].lower() == "all":
            selected_masters = net_order
        else:
            selected_masters = args.masters
        unknown = [m for m in selected_masters if m not in vgss_by_name]
        if unknown:
            raise KeyError(f"Case {case_key}: unknown masters {unknown}")

        conductor_to_net = {
            conductor.name: net.name
            for net in nets
            for conductor in net.conductors
        }
        system_center, system_size = get_system_bounds(conductors)
        kill_radius = system_size * args.kill_multiplier

        print("\n" + "=" * 80)
        print(
            f"CASE {case_key}: spacing={case_reference['spacing']}, "
            f"geometry={geometry}"
        )
        print("Selected masters:", selected_masters)
        print("=" * 80)

        seed_rows_by_master: dict[str, list[dict[str, Any]]] = {
            master: [] for master in selected_masters
        }
        raw_seed_results = []
        for seed in args.seeds:
            print(f"\n  Seed {seed}")
            seed_result = {"base_seed": seed, "rows": []}
            seed_sequence = np.random.SeedSequence(
                [seed, int(case_key), table.resolution]
            )
            child_sequences = seed_sequence.spawn(len(selected_masters))
            for master, child in zip(selected_masters, child_sequences):
                row = run_master_seed(
                    conductors=conductors,
                    master_vgss=vgss_by_name[master],
                    table=table,
                    importance=importance,
                    net_order=net_order,
                    conductor_to_net=conductor_to_net,
                    rng=np.random.default_rng(child),
                    walk_count=args.walks_per_seed,
                    max_hops=args.max_hops,
                    system_center=system_center,
                    kill_radius=kill_radius,
                    hit_tolerance=args.hit_tolerance,
                )
                seed_result["rows"].append(row)
                seed_rows_by_master[master].append(row)
            raw_seed_results.append(seed_result)

        combined = {
            master: combine_seed_rows(
                seed_rows_by_master[master],
                target_count=len(net_order) + 1,
            )
            for master in selected_masters
        }
        summary["cases"][case_key] = {
            "test_case_number": int(case_key),
            "spacing": case_reference["spacing"],
            "geometry": str(geometry),
            "selected_masters": selected_masters,
            "seed_results": raw_seed_results,
            "selected_master_rows": combined,
        }

        print("\n  Combined variance-reduction factors:")
        for master, row in combined.items():
            imp = np.asarray(
                row["variance_reduction_factor_vs_baseline"][
                    "abs_gradient_importance"
                ],
                dtype=float,
            )
            strat = np.asarray(
                row["variance_reduction_factor_vs_baseline"][
                    "sign_stratified_importance"
                ],
                dtype=float,
            )
            print(
                f"    {master}: median importance={np.nanmedian(imp):.3f}x, "
                f"median stratified={np.nanmedian(strat):.3f}x"
            )

    summary = to_builtin(summary)
    write_outputs(
        output_dir=args.output_dir,
        label=args.label,
        summary=summary,
        references=references,
    )


if __name__ == "__main__":
    main()
