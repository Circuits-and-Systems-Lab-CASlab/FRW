#!/usr/bin/env python3
"""Face-stratified BGS/VGSS surface-integration diagnostic for FRW.

This script does not modify production source files. It isolates the remaining
Gaussian-surface part of the estimator after GGFT, hit-tolerance, escape-radius,
and maximal-cube checks.

For each BGS face it draws an equal number of uniform (u, v) samples and forms
an unbiased stratified surface integral

    C_j = eps0 * sum_f A_f * mean_f[omega * I(target=j)].

Corresponding faces at different BGS scale factors use the same (u, v), first
panel uniform, and later-hop uniforms. This gives paired BGS-scale differences
with substantially lower noise than independent whole-surface runs.
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

from bgs.matlab_parser import parse_matlab_geometry
from bgs.topology import find_touching_nets
from bgs.vgss import build_all_net_vgss
from FRW.capacitance import E0, L_UNIT
from FRW.random_walk import get_system_bounds

INFINITY = "Infinity"
FloatArray = np.ndarray


@dataclass(frozen=True)
class Table:
    probabilities: FloatArray
    cdf: FloatArray
    panel_centers: FloatArray
    wvt: FloatArray
    resolution: int


@dataclass(frozen=True)
class Stratum:
    surface_index: int
    master_conductor: str
    face_index: int
    direction: str
    area: float
    vertices: FloatArray
    normal: FloatArray

    @property
    def key(self) -> str:
        return f"S{self.surface_index}:{self.master_conductor}:{self.direction}"


@dataclass(frozen=True)
class PathResult:
    target: str | None
    termination: str
    omega: float
    hops: int
    first_clearance: float


@dataclass
class Moments:
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
    def se_mean(self) -> float:
        return math.sqrt(self.sample_variance / self.count) if self.count > 1 else 0.0


@dataclass
class StratumAccumulator:
    labels: tuple[str, ...]
    contributions: dict[str, Moments] = field(init=False)
    hops: Moments = field(default_factory=Moments)
    clearances: list[float] = field(default_factory=list)
    terminations: dict[str, int] = field(
        default_factory=lambda: {"hit_conductor": 0, "escaped": 0, "max_hops": 0}
    )
    target_counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.contributions = {label: Moments() for label in self.labels}
        self.target_counts = {label: 0 for label in self.labels}

    def add(self, result: PathResult) -> None:
        for label in self.labels:
            self.contributions[label].add(
                result.omega if result.target == label else 0.0
            )
        if result.target is not None:
            self.target_counts[result.target] += 1
        self.terminations[result.termination] += 1
        self.hops.add(float(result.hops))
        self.clearances.append(float(result.first_clearance))


@dataclass
class DifferenceStratumAccumulator:
    labels: tuple[str, ...]
    # Stores already area-weighted physical differences per sample.
    differences: dict[str, Moments] = field(init=False)
    target_agreements: int = 0
    count: int = 0

    def __post_init__(self) -> None:
        self.differences = {label: Moments() for label in self.labels}

    def add(
        self,
        first: PathResult,
        second: PathResult,
        first_area: float,
        second_area: float,
    ) -> None:
        physical = E0 * L_UNIT
        for label in self.labels:
            a = first.omega if first.target == label else 0.0
            b = second.omega if second.target == label else 0.0
            self.differences[label].add(
                physical * (second_area * b - first_area * a)
            )
        self.target_agreements += int(first.target == second.target)
        self.count += 1

    @property
    def agreement_rate(self) -> float:
        return self.target_agreements / self.count if self.count else 0.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run a low-memory face-stratified FRW diagnostic for BGS surface "
            "integration, face contributions, and BGS-scale invariance."
        )
    )
    p.add_argument("input_file", type=Path)
    p.add_argument("--master", default="Net3")
    p.add_argument(
        "--table",
        type=Path,
        default=Path("src/GGFT/data/cube_tables_n200.npz"),
    )
    p.add_argument(
        "--bgs-scale-factors",
        type=float,
        nargs="+",
        default=[1.0, 1.25, 2.0],
    )
    p.add_argument("--max-distance", type=float, default=10.0)
    p.add_argument("--samples-per-face-per-seed", type=int, default=1000)
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[1234, 5678, 9012, 3456]
    )
    p.add_argument("--kill-multiplier", type=float, default=500.0)
    p.add_argument("--max-hops", type=int, default=500)
    p.add_argument("--hit-tolerance", type=float, default=1.0e-4)
    p.add_argument(
        "--reference-json",
        type=Path,
        default=Path("diagnostics/frw_reference_cases.json"),
    )
    p.add_argument("--case-number", type=int, default=7)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/diagnostics/frw_bgs_face"),
    )
    p.add_argument("--label", default="case7_net3_bgs_face")
    return p


def load_table(path: Path) -> Table:
    if not path.is_file():
        raise FileNotFoundError(f"GGFT table not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        probabilities = np.asarray(data["probabilities"], dtype=np.float64).reshape(-1)
        centers = np.asarray(data["panel_centers"], dtype=np.float64).reshape(-1, 3)
        wvt = np.asarray(data["wvt"], dtype=np.float64).reshape(-1, 3)
        resolution = int(np.asarray(data["resolution"]).item())
    probabilities = probabilities / float(probabilities.sum())
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    expected = 6 * resolution * resolution
    if probabilities.shape != (expected,):
        raise ValueError("Unexpected probability-table shape.")
    if centers.shape != (expected, 3) or wvt.shape != (expected, 3):
        raise ValueError("GGFT panel-centre/WVT shape mismatch.")
    return Table(probabilities, cdf, centers, wvt, resolution)


def select_master(all_vgss: Sequence[Any], master: str) -> Any:
    for item in all_vgss:
        if item.net.name == master:
            return item
    available = ", ".join(item.net.name for item in all_vgss)
    raise KeyError(f"Unknown master {master!r}; available: {available}")


def build_strata(net_vgss: Any) -> tuple[Stratum, ...]:
    strata: list[Stratum] = []
    for surface_index, result in enumerate(net_vgss.bgs_results):
        surface = result.gaussian_surface
        for face_index, face in enumerate(surface.faces):
            vertices = np.asarray(face.vertices, dtype=np.float64)
            normal = np.asarray(face.normal, dtype=np.float64)
            strata.append(
                Stratum(
                    surface_index=surface_index,
                    master_conductor=result.master.name,
                    face_index=face_index,
                    direction=face.direction,
                    area=float(face.area),
                    vertices=vertices,
                    normal=normal,
                )
            )
    return tuple(strata)


def audit_strata(strata: Sequence[Stratum]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in strata:
        e1 = s.vertices[1] - s.vertices[0]
        e2 = s.vertices[3] - s.vertices[0]
        exact_area = float(np.linalg.norm(np.cross(e1, e2)))
        unit_length = float(np.linalg.norm(s.normal))
        axis_nonzero = int(np.count_nonzero(np.abs(s.normal) > 1.0e-12))
        rows.append(
            {
                "stratum": s.key,
                "surface_index": s.surface_index,
                "master_conductor": s.master_conductor,
                "face_direction": s.direction,
                "stored_area": s.area,
                "cross_product_area": exact_area,
                "area_abs_error": abs(s.area - exact_area),
                "normal_length": unit_length,
                "normal_nonzero_components": axis_nonzero,
            }
        )
    return rows


def sample_face_point(stratum: Stratum, u: float, v: float) -> FloatArray:
    origin = stratum.vertices[0]
    return origin + u * (stratum.vertices[1] - origin) + v * (
        stratum.vertices[3] - origin
    )


def sample_index(cdf: FloatArray, u: float) -> int:
    return min(int(np.searchsorted(cdf, u, side="right")), len(cdf) - 1)


def first_weight(
    table: Table,
    panel_index: int,
    surface_normal: FloatArray,
    half_size: float,
) -> float:
    normal = np.asarray(surface_normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("Invalid BGS face normal.")
    return -float(np.dot(normal / norm, table.wvt[panel_index])) / half_size


def run_path(
    *,
    start: FloatArray,
    normal: FloatArray,
    uniforms: FloatArray,
    table: Table,
    conductors: Sequence[Any],
    conductor_to_net: dict[str, str],
    system_center: FloatArray,
    kill_radius: float,
    max_hops: int,
    hit_tolerance: float,
) -> PathResult:
    distances = [float(c.distance_to_point(start)) for c in conductors]
    first_clearance = min(distances)
    if first_clearance <= 0.0 or not math.isfinite(first_clearance):
        raise RuntimeError("A face-stratified start produced non-positive clearance.")

    first_panel = sample_index(table.cdf, float(uniforms[0]))
    omega = first_weight(table, first_panel, normal, first_clearance)
    current = np.asarray(start, dtype=np.float64) + (
        first_clearance * table.panel_centers[first_panel]
    )

    for hop in range(1, max_hops + 1):
        if np.max(np.abs(current - system_center)) > kill_radius:
            return PathResult(INFINITY, "escaped", omega, hop, first_clearance)

        distances = [float(c.distance_to_point(current)) for c in conductors]
        clearance = min(distances)
        nearest = int(np.argmin(distances))
        if clearance < hit_tolerance:
            return PathResult(
                conductor_to_net[conductors[nearest].name],
                "hit_conductor",
                omega,
                hop,
                first_clearance,
            )

        panel = sample_index(table.cdf, float(uniforms[hop]))
        current = current + clearance * table.panel_centers[panel]

    return PathResult(None, "max_hops", omega, max_hops, first_clearance)


def quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    q = np.quantile(np.asarray(values, dtype=np.float64), [0, 0.05, 0.5, 0.95, 1])
    return {
        "min": float(q[0]),
        "p05": float(q[1]),
        "median": float(q[2]),
        "p95": float(q[3]),
        "max": float(q[4]),
    }


def build_scale_data(
    conductors: Sequence[Any],
    scales: Sequence[float],
    master: str,
    max_distance: float,
) -> dict[float, dict[str, Any]]:
    result: dict[float, dict[str, Any]] = {}
    expected_keys: tuple[str, ...] | None = None
    for scale in scales:
        if scale < 1.0:
            raise ValueError("BGS scale factors must be >= 1.0.")
        net_vgss = select_master(
            build_all_net_vgss(
                conductors,
                scale_factor=float(scale),
                max_distance=max_distance,
            ),
            master,
        )
        strata = build_strata(net_vgss)
        keys = tuple(s.key for s in strata)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise RuntimeError("BGS strata do not correspond across scale factors.")
        result[float(scale)] = {
            "net_vgss": net_vgss,
            "strata": strata,
            "strata_by_key": {s.key: s for s in strata},
            "total_area": float(sum(s.area for s in strata)),
            "audit": audit_strata(strata),
        }
    return result


def run_seed(
    *,
    seed: int,
    samples_per_face: int,
    scale_data: dict[float, dict[str, Any]],
    table: Table,
    conductors: Sequence[Any],
    conductor_to_net: dict[str, str],
    labels: tuple[str, ...],
    system_center: FloatArray,
    kill_radius: float,
    max_hops: int,
    hit_tolerance: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scales = sorted(scale_data)
    stratum_keys = tuple(s.key for s in scale_data[scales[0]]["strata"])

    accumulators: dict[float, dict[str, StratumAccumulator]] = {
        scale: {key: StratumAccumulator(labels) for key in stratum_keys}
        for scale in scales
    }
    comparisons = list(zip(scales, scales[1:]))
    if len(scales) > 2:
        comparisons.append((scales[0], scales[-1]))
    difference_acc: dict[tuple[float, float], dict[str, DifferenceStratumAccumulator]] = {
        pair: {key: DifferenceStratumAccumulator(labels) for key in stratum_keys}
        for pair in comparisons
    }

    total_samples = len(stratum_keys) * samples_per_face
    done = 0
    progress_step = max(1, total_samples // 20)

    for key in stratum_keys:
        for _sample_index in range(samples_per_face):
            u, v = rng.random(2)
            path_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            path_rng = np.random.default_rng(path_seed)
            uniforms = path_rng.random(max_hops + 1)

            paths: dict[float, PathResult] = {}
            for scale in scales:
                stratum = scale_data[scale]["strata_by_key"][key]
                start = sample_face_point(stratum, float(u), float(v))
                path = run_path(
                    start=start,
                    normal=stratum.normal,
                    uniforms=uniforms,
                    table=table,
                    conductors=conductors,
                    conductor_to_net=conductor_to_net,
                    system_center=system_center,
                    kill_radius=kill_radius,
                    max_hops=max_hops,
                    hit_tolerance=hit_tolerance,
                )
                paths[scale] = path
                accumulators[scale][key].add(path)

            for first, second in comparisons:
                a = scale_data[first]["strata_by_key"][key]
                b = scale_data[second]["strata_by_key"][key]
                difference_acc[(first, second)][key].add(
                    paths[first], paths[second], a.area, b.area
                )

            done += 1
            if done % progress_step == 0 or done == total_samples:
                print(
                    f"\r  seed {seed}: {100.0 * done / total_samples:5.1f}% "
                    f"({done}/{total_samples})",
                    end="",
                    flush=True,
                )
    print()

    modes: dict[str, Any] = {}
    for scale in scales:
        total_estimate = {label: 0.0 for label in labels}
        total_variance = {label: 0.0 for label in labels}
        face_results: dict[str, Any] = {}
        unresolved = 0
        total_paths = 0

        for key in stratum_keys:
            stratum = scale_data[scale]["strata_by_key"][key]
            acc = accumulators[scale][key]
            face_estimates: dict[str, float] = {}
            face_ses: dict[str, float] = {}
            for label in labels:
                m = acc.contributions[label]
                estimate = E0 * L_UNIT * stratum.area * m.mean
                se = E0 * L_UNIT * stratum.area * m.se_mean
                face_estimates[label] = estimate
                face_ses[label] = se
                total_estimate[label] += estimate
                total_variance[label] += se * se
            face_results[key] = {
                "surface_index": stratum.surface_index,
                "master_conductor": stratum.master_conductor,
                "face_direction": stratum.direction,
                "area": stratum.area,
                "estimates_F": face_estimates,
                "standard_errors_F": face_ses,
                "target_counts": acc.target_counts,
                "termination_counts": acc.terminations,
                "mean_hops": acc.hops.mean,
                "first_clearance_quantiles": quantiles(acc.clearances),
            }
            unresolved += acc.terminations["max_hops"]
            total_paths += sum(acc.terminations.values())

        modes[str(scale)] = {
            "scale_factor": scale,
            "total_surface_area": scale_data[scale]["total_area"],
            "total_estimates_F": total_estimate,
            "total_standard_errors_F": {
                label: math.sqrt(total_variance[label]) for label in labels
            },
            "resolved_rate": 1.0 - unresolved / total_paths if total_paths else 0.0,
            "total_paths": total_paths,
            "face_results": face_results,
        }

    paired: dict[str, Any] = {}
    for first, second in comparisons:
        delta = {label: 0.0 for label in labels}
        variance = {label: 0.0 for label in labels}
        agreement_values: list[float] = []
        per_face: dict[str, Any] = {}
        for key in stratum_keys:
            acc = difference_acc[(first, second)][key]
            face_delta = {label: acc.differences[label].mean for label in labels}
            face_se = {label: acc.differences[label].se_mean for label in labels}
            for label in labels:
                delta[label] += face_delta[label]
                variance[label] += face_se[label] ** 2
            agreement_values.append(acc.agreement_rate)
            per_face[key] = {
                "delta_F": face_delta,
                "paired_standard_error_F": face_se,
                "target_agreement_rate": acc.agreement_rate,
            }
        se = {label: math.sqrt(variance[label]) for label in labels}
        name = f"scale_{second:g}_minus_{first:g}"
        paired[name] = {
            "first_scale": first,
            "second_scale": second,
            "delta_F": delta,
            "paired_standard_error_F": se,
            "paired_z": {
                label: delta[label] / se[label] if se[label] > 0.0 else 0.0
                for label in labels
            },
            "mean_face_target_agreement_rate": float(np.mean(agreement_values)),
            "face_differences": per_face,
        }

    return {
        "seed": seed,
        "samples_per_face": samples_per_face,
        "modes": modes,
        "paired_differences": paired,
    }


def combine_seeds(
    seed_results: Sequence[dict[str, Any]],
    labels: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode_keys = list(seed_results[0]["modes"])
    aggregate_modes: dict[str, Any] = {}
    for mode_key in mode_keys:
        estimates = {
            label: np.asarray(
                [r["modes"][mode_key]["total_estimates_F"][label] for r in seed_results],
                dtype=np.float64,
            )
            for label in labels
        }
        within = {
            label: np.asarray(
                [r["modes"][mode_key]["total_standard_errors_F"][label] for r in seed_results],
                dtype=np.float64,
            )
            for label in labels
        }
        nseed = len(seed_results)
        aggregate_modes[mode_key] = {
            "scale_factor": seed_results[0]["modes"][mode_key]["scale_factor"],
            "total_surface_area": seed_results[0]["modes"][mode_key]["total_surface_area"],
            "mean_estimates_F": {label: float(estimates[label].mean()) for label in labels},
            "between_seed_standard_error_F": {
                label: float(estimates[label].std(ddof=1) / math.sqrt(nseed))
                if nseed > 1
                else 0.0
                for label in labels
            },
            "combined_within_seed_standard_error_F": {
                label: float(math.sqrt(np.sum(within[label] ** 2)) / nseed)
                for label in labels
            },
            "mean_resolved_rate": float(
                np.mean([r["modes"][mode_key]["resolved_rate"] for r in seed_results])
            ),
            "total_paths": int(
                sum(r["modes"][mode_key]["total_paths"] for r in seed_results)
            ),
        }

    pair_keys = list(seed_results[0]["paired_differences"])
    aggregate_pairs: dict[str, Any] = {}
    for pair_key in pair_keys:
        nseed = len(seed_results)
        deltas = {
            label: np.asarray(
                [r["paired_differences"][pair_key]["delta_F"][label] for r in seed_results],
                dtype=np.float64,
            )
            for label in labels
        }
        within = {
            label: np.asarray(
                [
                    r["paired_differences"][pair_key]["paired_standard_error_F"][label]
                    for r in seed_results
                ],
                dtype=np.float64,
            )
            for label in labels
        }
        mean_delta = {label: float(deltas[label].mean()) for label in labels}
        combined = {
            label: float(math.sqrt(np.sum(within[label] ** 2)) / nseed)
            for label in labels
        }
        aggregate_pairs[pair_key] = {
            "first_scale": seed_results[0]["paired_differences"][pair_key]["first_scale"],
            "second_scale": seed_results[0]["paired_differences"][pair_key]["second_scale"],
            "mean_delta_F": mean_delta,
            "combined_within_seed_standard_error_F": combined,
            "paired_z": {
                label: mean_delta[label] / combined[label] if combined[label] > 0.0 else 0.0
                for label in labels
            },
            "between_seed_standard_error_F": {
                label: float(deltas[label].std(ddof=1) / math.sqrt(nseed))
                if nseed > 1
                else 0.0
                for label in labels
            },
            "mean_face_target_agreement_rate": float(
                np.mean(
                    [
                        r["paired_differences"][pair_key][
                            "mean_face_target_agreement_rate"
                        ]
                        for r in seed_results
                    ]
                )
            ),
        }
    return aggregate_modes, aggregate_pairs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_reference(path: Path, case_number: int, master: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    case = data.get("cases", {}).get(str(case_number))
    if case is None:
        return None
    net_order = data.get("net_order", ["Net1", "Net2", "Net3", "Net4", "Net5"])
    if master not in net_order:
        return None
    row = net_order.index(master)
    return {
        "case_number": case_number,
        "spacing": case.get("spacing"),
        "consensus": case.get("consensus_median", [])[row],
        "references": {
            tool: matrix[row] for tool, matrix in case.get("reference_matrices", {}).items()
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.samples_per_face_per_seed <= 1:
        raise SystemExit("--samples-per-face-per-seed must be greater than 1.")
    if not args.seeds:
        raise SystemExit("At least one seed is required.")
    if args.max_hops <= 0:
        raise SystemExit("--max-hops must be positive.")

    scales = sorted(set(float(v) for v in args.bgs_scale_factors))
    conductors = parse_matlab_geometry(args.input_file)
    nets = find_touching_nets(conductors)
    net_order = [net.name for net in nets]
    conductor_to_net = {
        conductor.name: net.name for net in nets for conductor in net.conductors
    }
    labels = tuple(net_order + [INFINITY])
    table = load_table(args.table)
    scale_data = build_scale_data(
        conductors, scales, args.master, args.max_distance
    )

    system_center, system_size = get_system_bounds(conductors)
    kill_radius = float(args.kill_multiplier) * float(system_size)
    reference = load_reference(args.reference_json, args.case_number, args.master)

    print("GGFT resolution:", table.resolution)
    print("Master:", args.master)
    print("BGS scales:", " ".join(f"{v:g}" for v in scales))
    print("Strata per scale:", len(scale_data[scales[0]]["strata"]))
    for scale in scales:
        print(
            f"  scale {scale:g}: area={scale_data[scale]['total_area']:.12g}"
        )

    seed_results = []
    for seed in args.seeds:
        seed_results.append(
            run_seed(
                seed=seed,
                samples_per_face=args.samples_per_face_per_seed,
                scale_data=scale_data,
                table=table,
                conductors=conductors,
                conductor_to_net=conductor_to_net,
                labels=labels,
                system_center=system_center,
                kill_radius=kill_radius,
                max_hops=args.max_hops,
                hit_tolerance=args.hit_tolerance,
            )
        )

    aggregate_modes, aggregate_pairs = combine_seeds(seed_results, labels)
    summary = {
        "description": "FRW face-stratified BGS surface-integration diagnostic.",
        "parameters": {
            "input_file": str(args.input_file),
            "master": args.master,
            "table": str(args.table),
            "resolution": table.resolution,
            "bgs_scale_factors": scales,
            "max_distance": args.max_distance,
            "samples_per_face_per_seed": args.samples_per_face_per_seed,
            "seeds": args.seeds,
            "strata_per_scale": len(scale_data[scales[0]]["strata"]),
            "total_paths_per_scale": (
                args.samples_per_face_per_seed
                * len(scale_data[scales[0]]["strata"])
                * len(args.seeds)
            ),
            "kill_multiplier": args.kill_multiplier,
            "max_hops": args.max_hops,
            "hit_tolerance": args.hit_tolerance,
            "E0": E0,
            "L_UNIT": L_UNIT,
        },
        "net_order": net_order,
        "target_order": list(labels),
        "reference": reference,
        "bgs_geometry_audit": {
            str(scale): scale_data[scale]["audit"] for scale in scales
        },
        "seed_results": seed_results,
        "aggregate_modes": aggregate_modes,
        "aggregate_paired_differences": aggregate_pairs,
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{args.label}_summary.json"
    entries_path = output / f"{args.label}_entries.csv"
    faces_path = output / f"{args.label}_face_contributions.csv"
    paired_path = output / f"{args.label}_paired_differences.csv"
    geometry_path = output / f"{args.label}_bgs_geometry.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    reference_consensus = reference.get("consensus") if reference else None
    reference_tools = reference.get("references", {}) if reference else {}
    entries: list[dict[str, Any]] = []
    for mode_key, mode in aggregate_modes.items():
        scale = mode["scale_factor"]
        for target_index, target in enumerate(labels):
            row = {
                "scale_factor": scale,
                "target": target,
                "estimate_F": mode["mean_estimates_F"][target],
                "within_seed_SE_F": mode[
                    "combined_within_seed_standard_error_F"
                ][target],
                "between_seed_SE_F": mode["between_seed_standard_error_F"][target],
                "surface_area": mode["total_surface_area"],
                "resolved_rate": mode["mean_resolved_rate"],
            }
            if target != INFINITY and reference_consensus is not None:
                row["consensus_reference_F"] = reference_consensus[target_index]
                for tool, values in reference_tools.items():
                    row[f"{tool}_F"] = values[target_index]
            entries.append(row)

    face_rows: list[dict[str, Any]] = []
    # Aggregate face contributions across seeds.
    for scale in scales:
        mode_key = str(scale)
        stratum_keys = list(seed_results[0]["modes"][mode_key]["face_results"])
        for key in stratum_keys:
            first_face = seed_results[0]["modes"][mode_key]["face_results"][key]
            for target in labels:
                values = np.asarray(
                    [
                        r["modes"][mode_key]["face_results"][key]["estimates_F"][target]
                        for r in seed_results
                    ],
                    dtype=np.float64,
                )
                ses = np.asarray(
                    [
                        r["modes"][mode_key]["face_results"][key][
                            "standard_errors_F"
                        ][target]
                        for r in seed_results
                    ],
                    dtype=np.float64,
                )
                face_rows.append(
                    {
                        "scale_factor": scale,
                        "stratum": key,
                        "surface_index": first_face["surface_index"],
                        "master_conductor": first_face["master_conductor"],
                        "face_direction": first_face["face_direction"],
                        "face_area": first_face["area"],
                        "target": target,
                        "mean_contribution_F": float(values.mean()),
                        "within_seed_SE_F": float(math.sqrt(np.sum(ses**2)) / len(ses)),
                        "between_seed_SE_F": float(values.std(ddof=1) / math.sqrt(len(values)))
                        if len(values) > 1
                        else 0.0,
                        "mean_hops": float(
                            np.mean(
                                [
                                    r["modes"][mode_key]["face_results"][key][
                                        "mean_hops"
                                    ]
                                    for r in seed_results
                                ]
                            )
                        ),
                    }
                )

    paired_rows: list[dict[str, Any]] = []
    for name, pair in aggregate_pairs.items():
        for target in labels:
            paired_rows.append(
                {
                    "comparison": name,
                    "first_scale": pair["first_scale"],
                    "second_scale": pair["second_scale"],
                    "target": target,
                    "mean_delta_F": pair["mean_delta_F"][target],
                    "within_seed_paired_SE_F": pair[
                        "combined_within_seed_standard_error_F"
                    ][target],
                    "between_seed_paired_SE_F": pair[
                        "between_seed_standard_error_F"
                    ][target],
                    "paired_z": pair["paired_z"][target],
                    "mean_face_target_agreement_rate": pair[
                        "mean_face_target_agreement_rate"
                    ],
                }
            )

    geometry_rows: list[dict[str, Any]] = []
    for scale in scales:
        for row in scale_data[scale]["audit"]:
            geometry_rows.append({"scale_factor": scale, **row})

    write_csv(entries_path, entries)
    write_csv(faces_path, face_rows)
    write_csv(paired_path, paired_rows)
    write_csv(geometry_path, geometry_rows)

    print("\nAggregate estimates:")
    for mode_key, mode in aggregate_modes.items():
        print(f"  BGS scale {mode['scale_factor']:g}")
        for target in labels:
            value = mode["mean_estimates_F"][target]
            se = mode["combined_within_seed_standard_error_F"][target]
            print(f"    {target:>8s}: {value: .12e} F  SE={se:.3e}")

    print("\nPaired BGS-scale differences:")
    for name, pair in aggregate_pairs.items():
        print(" ", name)
        for target in labels:
            print(
                f"    {target:>8s}: delta={pair['mean_delta_F'][target]: .3e} "
                f"SE={pair['combined_within_seed_standard_error_F'][target]:.3e} "
                f"z={pair['paired_z'][target]: .3f}"
            )

    print("\nOutputs:")
    for path in (json_path, entries_path, faces_path, paired_path, geometry_path):
        print(" ", path)


if __name__ == "__main__":
    main()
