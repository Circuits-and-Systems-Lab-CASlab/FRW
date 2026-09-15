#!/usr/bin/env python3
"""Low-memory paired FRW diagnostic for maximal-cube scale and escape radius.

The production source tree is not modified.  The script uses the bundled GGFT
transition table, common VGSS starts, and common panel uniforms to compare:

* maximal-cube half-size multipliers at a fixed large escape radius; and
* escape-radius multipliers at the production cube scale.

It also independently audits Conductor.distance_to_point against the exact
L-infinity clearance from a point to an axis-aligned conductor box and checks
that generated cubes do not overlap conductor interiors.
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

from bgs.matlab_parser import parse_matlab_geometry
from bgs.sampling import VGSSSamplingStats, sample_on_vgss
from bgs.topology import find_touching_nets
from bgs.vgss import build_all_net_vgss
from FRW.capacitance import E0, L_UNIT
from FRW.random_walk import get_system_bounds

INFINITY = "Infinity"
FloatArray = np.ndarray


@dataclass(frozen=True)
class TransitionTable:
    probabilities: FloatArray
    cdf: FloatArray
    panel_centers: FloatArray
    wvt: FloatArray
    resolution: int


@dataclass(frozen=True)
class ModeSpec:
    name: str
    cube_scale: float
    kill_multiplier: float
    family: str


@dataclass(frozen=True)
class PathResult:
    target: str | None
    termination: str
    omega: float
    hops: int


@dataclass
class ScalarMoments:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.total_sq += value * value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def sample_variance(self) -> float:
        if self.count <= 1:
            return 0.0
        centered = self.total_sq - self.total * self.total / self.count
        return max(centered / (self.count - 1), 0.0)

    @property
    def standard_error_of_mean(self) -> float:
        if self.count <= 1:
            return 0.0
        return math.sqrt(self.sample_variance / self.count)


@dataclass
class ModeAccumulator:
    labels: tuple[str, ...]
    contributions: dict[str, ScalarMoments] = field(init=False)
    row_sum: ScalarMoments = field(default_factory=ScalarMoments)
    terminations: dict[str, int] = field(
        default_factory=lambda: {"hit_conductor": 0, "escaped": 0, "max_hops": 0}
    )
    hops: ScalarMoments = field(default_factory=ScalarMoments)

    def __post_init__(self) -> None:
        self.contributions = {label: ScalarMoments() for label in self.labels}

    def add(self, result: PathResult) -> None:
        for label in self.labels:
            self.contributions[label].add(
                result.omega if result.target == label else 0.0
            )
        self.row_sum.add(result.omega if result.target is not None else 0.0)
        self.terminations[result.termination] += 1
        self.hops.add(float(result.hops))


@dataclass
class DifferenceAccumulator:
    labels: tuple[str, ...]
    contributions: dict[str, ScalarMoments] = field(init=False)
    row_sum: ScalarMoments = field(default_factory=ScalarMoments)
    target_agreements: int = 0
    count: int = 0

    def __post_init__(self) -> None:
        self.contributions = {label: ScalarMoments() for label in self.labels}

    def add(self, first: PathResult, second: PathResult) -> None:
        for label in self.labels:
            a = first.omega if first.target == label else 0.0
            b = second.omega if second.target == label else 0.0
            self.contributions[label].add(b - a)
        a_row = first.omega if first.target is not None else 0.0
        b_row = second.omega if second.target is not None else 0.0
        self.row_sum.add(b_row - a_row)
        self.target_agreements += int(first.target == second.target)
        self.count += 1

    @property
    def target_agreement_rate(self) -> float:
        return self.target_agreements / self.count if self.count else 0.0


@dataclass
class GeometryAudit:
    evaluated_points: int = 0
    method_clearance_max_abs_error: float = 0.0
    method_clearance_nonzero_mismatches: int = 0
    invalid_nonpositive_clearance: int = 0
    cube_interior_overlap_count: int = 0
    max_cube_interior_penetration: float = 0.0

    def merge(self, other: "GeometryAudit") -> None:
        self.evaluated_points += other.evaluated_points
        self.method_clearance_max_abs_error = max(
            self.method_clearance_max_abs_error,
            other.method_clearance_max_abs_error,
        )
        self.method_clearance_nonzero_mismatches += (
            other.method_clearance_nonzero_mismatches
        )
        self.invalid_nonpositive_clearance += other.invalid_nonpositive_clearance
        self.cube_interior_overlap_count += other.cube_interior_overlap_count
        self.max_cube_interior_penetration = max(
            self.max_cube_interior_penetration,
            other.max_cube_interior_penetration,
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run a paired FRW diagnostic for maximal-cube size invariance, "
            "escape-radius convergence, and exact cube-clearance validity."
        )
    )
    p.add_argument("input_file", type=Path)
    p.add_argument("--master", default="Net3")
    p.add_argument(
        "--table",
        type=Path,
        default=Path("src/GGFT/data/cube_tables_n201.npz"),
    )
    p.add_argument(
        "--cube-scales",
        type=float,
        nargs="+",
        default=[0.50, 0.75, 0.90, 0.99, 1.00],
    )
    p.add_argument(
        "--kill-multipliers",
        type=float,
        nargs="+",
        default=[50.0, 500.0, 5000.0, 500000.0],
    )
    p.add_argument("--fixed-kill-multiplier", type=float, default=500000.0)
    p.add_argument("--fixed-cube-scale", type=float, default=1.0)
    p.add_argument("--walks-per-seed", type=int, default=2500)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1234, 5678, 9012, 3456],
    )
    p.add_argument("--scale-factor", type=float, default=1.25)
    p.add_argument("--max-distance", type=float, default=10.0)
    p.add_argument("--max-hops", type=int, default=2000)
    p.add_argument("--hit-tolerance", type=float, default=1.0e-4)
    p.add_argument("--audit-tolerance", type=float, default=1.0e-9)
    p.add_argument(
        "--reference-json",
        type=Path,
        default=Path("diagnostics/frw_reference_cases.json"),
    )
    p.add_argument("--case-number", type=int, default=7)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/diagnostics/frw_cube_escape"),
    )
    p.add_argument("--label", default="case7_net3_cube_escape")
    return p


def load_table(path: Path) -> TransitionTable:
    if not path.is_file():
        raise FileNotFoundError(f"GGFT table not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        p = np.asarray(data["probabilities"], dtype=np.float64)
        centers = np.asarray(data["panel_centers"], dtype=np.float64)
        wvt = np.asarray(data["wvt"], dtype=np.float64)
        resolution = int(np.asarray(data["resolution"]).item())
    p = p / float(p.sum())
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    if centers.shape != (len(p), 3) or wvt.shape != (len(p), 3):
        raise ValueError("GGFT probability, panel-centre, and WVT shapes disagree.")
    return TransitionTable(p, cdf, centers, wvt, resolution)


def sample_index(cdf: FloatArray, u: float) -> int:
    return min(int(np.searchsorted(cdf, u, side="right")), len(cdf) - 1)


def exact_linf_distance(point: FloatArray, conductor: Any) -> float:
    point = np.asarray(point, dtype=np.float64)
    lower = conductor.min_bounds - point
    upper = point - conductor.max_bounds
    gaps = np.maximum(np.maximum(lower, upper), 0.0)
    return float(np.max(gaps))


def exact_nearest(
    point: FloatArray,
    conductors: Sequence[Any],
    audit: GeometryAudit,
    audit_tolerance: float,
) -> tuple[float, int]:
    exact = np.asarray(
        [exact_linf_distance(point, conductor) for conductor in conductors],
        dtype=np.float64,
    )
    method = np.asarray(
        [float(conductor.distance_to_point(point)) for conductor in conductors],
        dtype=np.float64,
    )
    difference = np.abs(method - exact)
    audit.evaluated_points += 1
    audit.method_clearance_max_abs_error = max(
        audit.method_clearance_max_abs_error,
        float(difference.max(initial=0.0)),
    )
    audit.method_clearance_nonzero_mismatches += int(
        np.any(difference > audit_tolerance)
    )
    index = int(np.argmin(exact))
    return float(exact[index]), index


def audit_cube(
    center: FloatArray,
    half_size: float,
    conductors: Sequence[Any],
    audit: GeometryAudit,
    audit_tolerance: float,
) -> None:
    if not math.isfinite(half_size) or half_size <= 0.0:
        audit.invalid_nonpositive_clearance += 1
        return
    cube_min = np.asarray(center, dtype=np.float64) - half_size
    cube_max = np.asarray(center, dtype=np.float64) + half_size
    for conductor in conductors:
        overlap = np.minimum(cube_max, conductor.max_bounds) - np.maximum(
            cube_min, conductor.min_bounds
        )
        penetration = float(np.min(overlap))
        if np.all(overlap > audit_tolerance):
            audit.cube_interior_overlap_count += 1
            audit.max_cube_interior_penetration = max(
                audit.max_cube_interior_penetration,
                penetration,
            )


def scale_token(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def kill_token(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def build_modes(args: argparse.Namespace) -> list[ModeSpec]:
    modes: dict[tuple[float, float], ModeSpec] = {}
    for scale in args.cube_scales:
        if not (0.0 < scale <= 1.0):
            raise ValueError("Every --cube-scales value must satisfy 0 < s <= 1.")
        key = (float(scale), float(args.fixed_kill_multiplier))
        modes[key] = ModeSpec(
            name=f"cube_s{scale_token(scale)}_k{kill_token(args.fixed_kill_multiplier)}",
            cube_scale=float(scale),
            kill_multiplier=float(args.fixed_kill_multiplier),
            family="cube_scale",
        )
    for multiplier in args.kill_multipliers:
        if multiplier <= 0.0 or not math.isfinite(multiplier):
            raise ValueError("Every --kill-multipliers value must be positive.")
        key = (float(args.fixed_cube_scale), float(multiplier))
        modes[key] = ModeSpec(
            name=f"escape_s{scale_token(args.fixed_cube_scale)}_k{kill_token(multiplier)}",
            cube_scale=float(args.fixed_cube_scale),
            kill_multiplier=float(multiplier),
            family="escape_radius",
        )
    return sorted(modes.values(), key=lambda item: (item.cube_scale, item.kill_multiplier))


def build_comparisons(
    modes: Sequence[ModeSpec], args: argparse.Namespace
) -> list[tuple[str, str, str]]:
    by_key = {(m.cube_scale, m.kill_multiplier): m.name for m in modes}
    comparisons: list[tuple[str, str, str]] = []

    cube = sorted(set(float(v) for v in args.cube_scales))
    for first, second in zip(cube, cube[1:]):
        comparisons.append(
            (
                f"cube_scale_{scale_token(second)}_minus_{scale_token(first)}",
                by_key[(first, float(args.fixed_kill_multiplier))],
                by_key[(second, float(args.fixed_kill_multiplier))],
            )
        )
    if len(cube) >= 2:
        comparisons.append(
            (
                f"cube_scale_{scale_token(cube[-1])}_minus_{scale_token(cube[0])}",
                by_key[(cube[0], float(args.fixed_kill_multiplier))],
                by_key[(cube[-1], float(args.fixed_kill_multiplier))],
            )
        )

    kills = sorted(set(float(v) for v in args.kill_multipliers))
    for first, second in zip(kills, kills[1:]):
        comparisons.append(
            (
                f"kill_{kill_token(second)}_minus_{kill_token(first)}",
                by_key[(float(args.fixed_cube_scale), first)],
                by_key[(float(args.fixed_cube_scale), second)],
            )
        )
    if len(kills) >= 2:
        comparisons.append(
            (
                f"kill_{kill_token(kills[-1])}_minus_{kill_token(kills[0])}",
                by_key[(float(args.fixed_cube_scale), kills[0])],
                by_key[(float(args.fixed_cube_scale), kills[-1])],
            )
        )
    # Remove accidental duplicate comparisons while preserving order.
    seen: set[tuple[str, str]] = set()
    unique = []
    for name, first, second in comparisons:
        key = (first, second)
        if first != second and key not in seen:
            unique.append((name, first, second))
            seen.add(key)
    return unique


def first_weight(
    table: TransitionTable,
    panel_index: int,
    surface_normal: FloatArray,
    half_size: float,
) -> float:
    normal = np.asarray(surface_normal, dtype=np.float64)
    normal /= float(np.linalg.norm(normal))
    return -float(np.dot(normal, table.wvt[panel_index])) / half_size


def run_coupled_paths(
    *,
    start_point: FloatArray,
    surface_normal: FloatArray,
    first_clearance: float,
    rng: np.random.Generator,
    table: TransitionTable,
    modes: Sequence[ModeSpec],
    conductors: Sequence[Any],
    conductor_to_net: dict[str, str],
    system_center: FloatArray,
    system_size: float,
    max_hops: int,
    hit_tolerance: float,
    audit_tolerance: float,
) -> tuple[dict[str, PathResult], GeometryAudit]:
    audit = GeometryAudit()
    panel_uniforms = rng.random(max_hops + 1)
    first_panel = sample_index(table.cdf, float(panel_uniforms[0]))

    states: dict[str, dict[str, Any]] = {}
    for mode in modes:
        h = mode.cube_scale * first_clearance
        audit_cube(start_point, h, conductors, audit, audit_tolerance)
        omega = first_weight(table, first_panel, surface_normal, h)
        point = np.asarray(start_point, dtype=np.float64) + h * table.panel_centers[first_panel]
        states[mode.name] = {
            "point": point,
            "omega": omega,
            "spec": mode,
        }

    results: dict[str, PathResult] = {}
    active = set(states)

    for hop in range(1, max_hops + 1):
        for name in tuple(active):
            state = states[name]
            mode: ModeSpec = state["spec"]
            point = state["point"]
            kill_radius = system_size * mode.kill_multiplier

            if np.max(np.abs(point - system_center)) > kill_radius:
                results[name] = PathResult(INFINITY, "escaped", state["omega"], hop)
                active.remove(name)
                continue

            clearance, nearest_index = exact_nearest(
                point, conductors, audit, audit_tolerance
            )
            if clearance < hit_tolerance:
                conductor_name = conductors[nearest_index].name
                results[name] = PathResult(
                    conductor_to_net[conductor_name],
                    "hit_conductor",
                    state["omega"],
                    hop,
                )
                active.remove(name)

        if not active:
            break

        panel_index = sample_index(table.cdf, float(panel_uniforms[hop]))
        for name in tuple(active):
            state = states[name]
            mode: ModeSpec = state["spec"]
            point = state["point"]
            clearance, _nearest_index = exact_nearest(
                point, conductors, audit, audit_tolerance
            )
            if clearance <= 0.0 or not math.isfinite(clearance):
                audit.invalid_nonpositive_clearance += 1
                results[name] = PathResult(None, "max_hops", state["omega"], hop)
                active.remove(name)
                continue
            h = mode.cube_scale * clearance
            audit_cube(point, h, conductors, audit, audit_tolerance)
            state["point"] = point + h * table.panel_centers[panel_index]

    for name in active:
        state = states[name]
        results[name] = PathResult(None, "max_hops", state["omega"], max_hops)

    return results, audit


def estimate_mode(
    accumulator: ModeAccumulator,
    physical_scale: float,
) -> dict[str, Any]:
    estimates = {
        label: physical_scale * moments.mean
        for label, moments in accumulator.contributions.items()
    }
    standard_errors = {
        label: abs(physical_scale) * moments.standard_error_of_mean
        for label, moments in accumulator.contributions.items()
    }
    total = sum(accumulator.terminations.values())
    return {
        "estimates_F": estimates,
        "standard_errors_F_conditional_on_H": standard_errors,
        "row_sum_with_infinity_F": physical_scale * accumulator.row_sum.mean,
        "row_sum_standard_error_F_conditional_on_H": (
            abs(physical_scale) * accumulator.row_sum.standard_error_of_mean
        ),
        "termination_counts": accumulator.terminations,
        "resolved_rate": (
            1.0 - accumulator.terminations["max_hops"] / total if total else 0.0
        ),
        "mean_hops": accumulator.hops.mean,
    }


def estimate_difference(
    accumulator: DifferenceAccumulator,
    physical_scale: float,
) -> dict[str, Any]:
    delta = {
        label: physical_scale * moments.mean
        for label, moments in accumulator.contributions.items()
    }
    se = {
        label: abs(physical_scale) * moments.standard_error_of_mean
        for label, moments in accumulator.contributions.items()
    }
    z = {label: delta[label] / se[label] if se[label] > 0.0 else 0.0 for label in delta}
    row_delta = physical_scale * accumulator.row_sum.mean
    row_se = abs(physical_scale) * accumulator.row_sum.standard_error_of_mean
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


def run_seed(
    *,
    seed: int,
    num_walks: int,
    conductors: Sequence[Any],
    master_vgss: Any,
    table: TransitionTable,
    modes: Sequence[ModeSpec],
    comparisons: Sequence[tuple[str, str, str]],
    conductor_to_net: dict[str, str],
    system_center: FloatArray,
    system_size: float,
    max_hops: int,
    hit_tolerance: float,
    audit_tolerance: float,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    stats = VGSSSamplingStats()
    mode_acc = {mode.name: ModeAccumulator(labels) for mode in modes}
    diff_acc = {
        name: DifferenceAccumulator(labels) for name, _first, _second in comparisons
    }
    audit = GeometryAudit()
    progress_step = max(1, num_walks // 20)

    for walk_index in range(num_walks):
        if (walk_index + 1) % progress_step == 0 or walk_index + 1 == num_walks:
            pct = 100.0 * (walk_index + 1) / num_walks
            print(
                f"\r  seed {seed}: {pct:5.1f}% ({walk_index + 1}/{num_walks})",
                end="",
                flush=True,
            )

        sample = sample_on_vgss(master_vgss.sampling_context, rng, stats=stats)
        start = sample.point
        first_clearance, _nearest = exact_nearest(
            start, conductors, audit, audit_tolerance
        )
        if first_clearance <= 0.0:
            raise RuntimeError("VGSS start point has non-positive exact cube clearance.")

        path_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        results, path_audit = run_coupled_paths(
            start_point=start,
            surface_normal=sample.normal,
            first_clearance=first_clearance,
            rng=np.random.default_rng(path_seed),
            table=table,
            modes=modes,
            conductors=conductors,
            conductor_to_net=conductor_to_net,
            system_center=system_center,
            system_size=system_size,
            max_hops=max_hops,
            hit_tolerance=hit_tolerance,
            audit_tolerance=audit_tolerance,
        )
        audit.merge(path_audit)

        for name, result in results.items():
            mode_acc[name].add(result)
        for comparison_name, first, second in comparisons:
            diff_acc[comparison_name].add(results[first], results[second])

    print()
    context = master_vgss.sampling_context
    H = stats.estimate_H(context)
    physical_scale = E0 * L_UNIT * H

    return {
        "base_seed": seed,
        "walk_count": num_walks,
        "H": H,
        "Sg": stats.estimate_surface_area(context),
        "mean_p": stats.estimate_mean_p(),
        "sampling_stats": stats.as_dict(context),
        "modes": {
            name: estimate_mode(accumulator, physical_scale)
            for name, accumulator in mode_acc.items()
        },
        "paired_differences": {
            name: estimate_difference(accumulator, physical_scale)
            for name, accumulator in diff_acc.items()
        },
        "geometry_audit": audit.__dict__,
    }


def aggregate_seed_modes(
    seed_results: Sequence[dict[str, Any]],
    modes: Sequence[ModeSpec],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    nseed = len(seed_results)
    for mode in modes:
        stack = np.asarray(
            [
                [seed["modes"][mode.name]["estimates_F"][label] for label in labels]
                for seed in seed_results
            ],
            dtype=np.float64,
        )
        within = np.asarray(
            [
                [
                    seed["modes"][mode.name]["standard_errors_F_conditional_on_H"][label]
                    for label in labels
                ]
                for seed in seed_results
            ],
            dtype=np.float64,
        )
        mean = stack.mean(axis=0)
        between_sd = stack.std(axis=0, ddof=1) if nseed > 1 else np.zeros(len(labels))
        between_se = between_sd / math.sqrt(nseed) if nseed else between_sd
        combined_within = np.sqrt(np.sum(within * within, axis=0)) / nseed
        output[mode.name] = {
            "cube_scale": mode.cube_scale,
            "kill_multiplier": mode.kill_multiplier,
            "family": mode.family,
            "mean_estimates_F": dict(zip(labels, mean.tolist())),
            "between_seed_sd_F": dict(zip(labels, between_sd.tolist())),
            "between_seed_se_of_mean_F": dict(zip(labels, between_se.tolist())),
            "combined_within_seed_se_of_mean_F": dict(
                zip(labels, combined_within.tolist())
            ),
            "mean_row_sum_with_infinity_F": float(
                np.mean(
                    [seed["modes"][mode.name]["row_sum_with_infinity_F"] for seed in seed_results]
                )
            ),
            "mean_resolved_rate": float(
                np.mean([seed["modes"][mode.name]["resolved_rate"] for seed in seed_results])
            ),
            "mean_hops": float(
                np.mean([seed["modes"][mode.name]["mean_hops"] for seed in seed_results])
            ),
        }
    return output


def aggregate_differences(
    seed_results: Sequence[dict[str, Any]],
    comparisons: Sequence[tuple[str, str, str]],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    nseed = len(seed_results)
    for name, first, second in comparisons:
        deltas = np.asarray(
            [
                [seed["paired_differences"][name]["delta_F"][label] for label in labels]
                for seed in seed_results
            ],
            dtype=np.float64,
        )
        within = np.asarray(
            [
                [
                    seed["paired_differences"][name][
                        "paired_standard_errors_F_conditional_on_H"
                    ][label]
                    for label in labels
                ]
                for seed in seed_results
            ],
            dtype=np.float64,
        )
        mean_delta = deltas.mean(axis=0)
        combined = np.sqrt(np.sum(within * within, axis=0)) / nseed
        z = np.divide(
            mean_delta,
            combined,
            out=np.zeros_like(mean_delta),
            where=combined > 0.0,
        )
        output[name] = {
            "first_mode": first,
            "second_mode": second,
            "mean_delta_F": dict(zip(labels, mean_delta.tolist())),
            "combined_within_seed_se_of_mean_F": dict(zip(labels, combined.tolist())),
            "paired_z": dict(zip(labels, z.tolist())),
            "mean_target_agreement_rate": float(
                np.mean(
                    [
                        seed["paired_differences"][name]["target_agreement_rate"]
                        for seed in seed_results
                    ]
                )
            ),
        }
    return output


def load_reference(path: Path, case_number: int, master_index: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    case = data.get("cases", {}).get(str(case_number))
    if not case:
        return None
    refs = {
        name: matrix[master_index]
        for name, matrix in case["reference_matrices"].items()
    }
    consensus = case["consensus_median"][master_index]
    return {
        "case_number": case_number,
        "spacing": case.get("spacing"),
        "references": refs,
        "consensus": consensus,
    }


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parser().parse_args()
    if args.walks_per_seed <= 1:
        raise SystemExit("--walks-per-seed must be greater than 1.")
    if args.max_hops <= 0:
        raise SystemExit("--max-hops must be positive.")
    if args.fixed_cube_scale <= 0.0 or args.fixed_cube_scale > 1.0:
        raise SystemExit("--fixed-cube-scale must satisfy 0 < s <= 1.")

    conductors = parse_matlab_geometry(args.input_file)
    nets = find_touching_nets(conductors)
    net_order = [net.name for net in nets]
    if args.master not in net_order:
        raise SystemExit(f"Unknown master {args.master!r}; available nets: {net_order}")
    master_index = net_order.index(args.master)
    all_vgss = build_all_net_vgss(
        conductors,
        scale_factor=args.scale_factor,
        max_distance=args.max_distance,
    )
    master_vgss = all_vgss[master_index]
    conductor_to_net = {
        conductor.name: net.name for net in nets for conductor in net.conductors
    }
    table = load_table(args.table)
    modes = build_modes(args)
    comparisons = build_comparisons(modes, args)
    labels = tuple(net_order + [INFINITY])
    system_center, system_size = get_system_bounds(conductors)

    print("Input:", args.input_file)
    print("Master:", args.master)
    print("GGFT resolution:", table.resolution)
    print("Walks per seed:", args.walks_per_seed)
    print("Seeds:", args.seeds)
    print("Total walks:", args.walks_per_seed * len(args.seeds))
    print("Modes:")
    for mode in modes:
        print(
            f"  {mode.name}: cube_scale={mode.cube_scale}, "
            f"kill_multiplier={mode.kill_multiplier}"
        )

    seed_results = []
    for seed in args.seeds:
        print("\n" + "=" * 78)
        print("SEED", seed)
        print("=" * 78)
        seed_results.append(
            run_seed(
                seed=seed,
                num_walks=args.walks_per_seed,
                conductors=conductors,
                master_vgss=master_vgss,
                table=table,
                modes=modes,
                comparisons=comparisons,
                conductor_to_net=conductor_to_net,
                system_center=system_center,
                system_size=system_size,
                max_hops=args.max_hops,
                hit_tolerance=args.hit_tolerance,
                audit_tolerance=args.audit_tolerance,
                labels=labels,
            )
        )

    aggregate_modes = aggregate_seed_modes(seed_results, modes, labels)
    aggregate_pair_differences = aggregate_differences(seed_results, comparisons, labels)
    aggregate_audit = GeometryAudit()
    for seed in seed_results:
        aggregate_audit.merge(GeometryAudit(**seed["geometry_audit"]))

    reference = load_reference(args.reference_json, args.case_number, master_index)
    output = {
        "description": "FRW maximal-cube scale, escape-radius, and geometry-clearance diagnostic.",
        "parameters": {
            "input_file": str(args.input_file),
            "master": args.master,
            "table": str(args.table),
            "resolution": table.resolution,
            "cube_scales": args.cube_scales,
            "kill_multipliers": args.kill_multipliers,
            "fixed_kill_multiplier": args.fixed_kill_multiplier,
            "fixed_cube_scale": args.fixed_cube_scale,
            "walks_per_seed": args.walks_per_seed,
            "seeds": args.seeds,
            "total_walks": args.walks_per_seed * len(args.seeds),
            "scale_factor": args.scale_factor,
            "max_distance": args.max_distance,
            "max_hops": args.max_hops,
            "hit_tolerance": args.hit_tolerance,
            "E0": E0,
            "L_UNIT": L_UNIT,
        },
        "net_order": net_order,
        "target_order": labels,
        "modes": [mode.__dict__ for mode in modes],
        "comparisons": [
            {"name": name, "first_mode": first, "second_mode": second}
            for name, first, second in comparisons
        ],
        "reference": reference,
        "seed_results": seed_results,
        "aggregate_modes": aggregate_modes,
        "aggregate_paired_differences": aggregate_pair_differences,
        "aggregate_geometry_audit": aggregate_audit.__dict__,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(to_builtin(output), indent=2), encoding="utf-8")

    entry_rows: list[dict[str, Any]] = []
    for mode in modes:
        aggregate = aggregate_modes[mode.name]
        for target_index, target in enumerate(labels):
            estimate = aggregate["mean_estimates_F"][target]
            se = aggregate["combined_within_seed_se_of_mean_F"][target]
            row = {
                "mode": mode.name,
                "family": mode.family,
                "cube_scale": mode.cube_scale,
                "kill_multiplier": mode.kill_multiplier,
                "target": target,
                "estimate_F": estimate,
                "SE_F": se,
                "relative_SE": abs(se / estimate) if estimate else math.inf,
            }
            if reference and target != INFINITY:
                row["consensus_reference_F"] = reference["consensus"][target_index]
                for solver, values in reference["references"].items():
                    row[f"{solver}_F"] = values[target_index]
            entry_rows.append(row)
    write_csv(args.output_dir / f"{args.label}_entries.csv", entry_rows)

    paired_rows: list[dict[str, Any]] = []
    for name, data in aggregate_pair_differences.items():
        for target in labels:
            paired_rows.append(
                {
                    "comparison": name,
                    "first_mode": data["first_mode"],
                    "second_mode": data["second_mode"],
                    "target": target,
                    "mean_delta_F": data["mean_delta_F"][target],
                    "paired_SE_F": data["combined_within_seed_se_of_mean_F"][target],
                    "paired_z": data["paired_z"][target],
                    "mean_target_agreement_rate": data["mean_target_agreement_rate"],
                }
            )
    write_csv(
        args.output_dir / f"{args.label}_paired_differences.csv", paired_rows
    )

    mode_rows: list[dict[str, Any]] = []
    for mode in modes:
        data = aggregate_modes[mode.name]
        finite = [data["mean_estimates_F"][name] for name in net_order]
        infinity = data["mean_estimates_F"][INFINITY]
        direct = finite[master_index]
        conservation = -(
            sum(value for index, value in enumerate(finite) if index != master_index)
            + infinity
        )
        mode_rows.append(
            {
                "mode": mode.name,
                "family": mode.family,
                "cube_scale": mode.cube_scale,
                "kill_multiplier": mode.kill_multiplier,
                "mean_hops": data["mean_hops"],
                "resolved_rate": data["mean_resolved_rate"],
                "row_sum_with_infinity_F": data["mean_row_sum_with_infinity_F"],
                "direct_diagonal_F": direct,
                "conservation_diagonal_F": conservation,
                "direct_minus_conservation_F": direct - conservation,
            }
        )
    write_csv(args.output_dir / f"{args.label}_mode_summary.csv", mode_rows)
    write_csv(
        args.output_dir / f"{args.label}_geometry_audit.csv",
        [aggregate_audit.__dict__],
    )

    print("\nOutputs:")
    for suffix in (
        "summary.json",
        "entries.csv",
        "paired_differences.csv",
        "mode_summary.csv",
        "geometry_audit.csv",
    ):
        print(" ", args.output_dir / f"{args.label}_{suffix}")
    print("\nAggregate geometry audit:")
    for key, value in aggregate_audit.__dict__.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
