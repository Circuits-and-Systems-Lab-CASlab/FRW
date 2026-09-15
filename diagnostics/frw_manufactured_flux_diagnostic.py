#!/usr/bin/env python3
"""Manufactured-solution flux diagnostic for the FRW/BGS first-hop estimator.

This diagnostic does not modify production code.  It uses the harmonic
potential

    phi(r) = 1 / ||r - r_s||

for a point r_s strictly inside the master conductor.  On every closed
Gaussian surface enclosing r_s,

    integral_G [-grad(phi) dot n] dS = 4*pi.

The script compares three evaluations of this same flux:

1. deterministic Gauss-Legendre quadrature of the exact flux density;
2. Monte Carlo integration of the exact flux density at the sampled BGS points;
3. the production numerical WVT first-hop estimator
       omega * phi(r_1),
   integrated face-by-face with the exact face areas.

The third mode validates, end to end, BGS face areas, outward normals,
first-cube clearance, panel sampling, panel-centre mapping, WVT sign and scale,
and the outer surface-integration normalization.  No continuation walks or
reference capacitance solver are involved.
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
from numpy.polynomial.legendre import leggauss

from bgs.matlab_parser import parse_matlab_geometry
from bgs.vgss import build_all_net_vgss

FloatArray = np.ndarray
EXACT_FLUX = 4.0 * math.pi


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
    direction: str
    area: float
    vertices: FloatArray
    normal: FloatArray

    @property
    def key(self) -> str:
        return f"S{self.surface_index}:{self.master_conductor}:{self.direction}"


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
class FaceAccumulator:
    exact: Moments = field(default_factory=Moments)
    numerical: Moments = field(default_factory=Moments)
    difference: Moments = field(default_factory=Moments)
    clearances: Moments = field(default_factory=Moments)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Validate the complete BGS/WVT first-hop surface integral against "
            "the manufactured point-source flux 4*pi."
        )
    )
    p.add_argument("input_file", type=Path)
    p.add_argument("--master", default="Net3")
    p.add_argument(
        "--table", type=Path, default=Path("src/GGFT/data/cube_tables_n201.npz")
    )
    p.add_argument(
        "--bgs-scale-factors", type=float, nargs="+", default=[1.0, 1.25, 2.0]
    )
    p.add_argument("--max-distance", type=float, default=10.0)
    p.add_argument("--samples-per-face-per-seed", type=int, default=5000)
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[1234, 5678, 9012, 3456]
    )
    p.add_argument("--quadrature-order", type=int, default=40)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/diagnostics/frw_manufactured_flux"),
    )
    p.add_argument("--label", default="case7_net3_manufactured_flux")
    return p


def load_table(path: Path) -> Table:
    if not path.is_file():
        raise FileNotFoundError(f"GGFT table not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        p = np.asarray(data["probabilities"], dtype=np.float64).reshape(-1)
        centers = np.asarray(data["panel_centers"], dtype=np.float64).reshape(-1, 3)
        wvt = np.asarray(data["wvt"], dtype=np.float64).reshape(-1, 3)
        resolution = int(np.asarray(data["resolution"]).item())
    p = p / float(p.sum())
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    expected = 6 * resolution * resolution
    if p.shape != (expected,) or centers.shape != (expected, 3) or wvt.shape != (expected, 3):
        raise ValueError("Unexpected GGFT table shape.")
    return Table(p, cdf, centers, wvt, resolution)


def select_master(all_vgss: Sequence[Any], master: str) -> Any:
    for item in all_vgss:
        if item.net.name == master:
            return item
    available = ", ".join(item.net.name for item in all_vgss)
    raise KeyError(f"Unknown master {master!r}; available: {available}")


def build_strata(net_vgss: Any) -> tuple[Stratum, ...]:
    strata: list[Stratum] = []
    for surface_index, result in enumerate(net_vgss.bgs_results):
        for face in result.gaussian_surface.faces:
            strata.append(
                Stratum(
                    surface_index=surface_index,
                    master_conductor=result.master.name,
                    direction=face.direction,
                    area=float(face.area),
                    vertices=np.asarray(face.vertices, dtype=np.float64),
                    normal=np.asarray(face.normal, dtype=np.float64),
                )
            )
    return tuple(strata)


def build_scale_data(
    conductors: Sequence[Any], scales: Sequence[float], master: str, max_distance: float
) -> dict[float, dict[str, Any]]:
    result: dict[float, dict[str, Any]] = {}
    expected_keys: tuple[str, ...] | None = None
    expected_masters: tuple[str, ...] | None = None
    for scale in scales:
        if scale < 1.0:
            raise ValueError("BGS scale factors must be at least 1.0.")
        net_vgss = select_master(
            build_all_net_vgss(
                conductors, scale_factor=float(scale), max_distance=max_distance
            ),
            master,
        )
        strata = build_strata(net_vgss)
        keys = tuple(s.key for s in strata)
        masters = tuple(r.master.name for r in net_vgss.bgs_results)
        if expected_keys is None:
            expected_keys = keys
            expected_masters = masters
        elif keys != expected_keys or masters != expected_masters:
            raise RuntimeError("BGS strata do not correspond across scale factors.")
        result[float(scale)] = {
            "net_vgss": net_vgss,
            "strata": strata,
            "strata_by_key": {s.key: s for s in strata},
            "total_area": float(sum(s.area for s in strata)),
        }
    return result


def source_points(net_vgss: Any) -> dict[str, FloatArray]:
    # This diagnostic is intentionally focused on the one-block Case-7 master.
    if len(net_vgss.bgs_results) != 1:
        raise RuntimeError(
            "Manufactured-flux diagnostic currently requires a master net with "
            "exactly one BGS/conductor block."
        )
    c = net_vgss.bgs_results[0].master
    lo = c.min_bounds
    hi = c.max_bounds
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    fractions = {
        "center": np.array([0.0, 0.0, 0.0]),
        "off_a": np.array([0.23, -0.19, 0.31]),
        "off_b": np.array([-0.29, 0.27, -0.17]),
    }
    result = {name: center + frac * half for name, frac in fractions.items()}
    for name, point in result.items():
        if np.any(point <= lo) or np.any(point >= hi):
            raise RuntimeError(f"Manufactured source {name} is not strictly inside conductor.")
    return result


def sample_face_point(s: Stratum, u: float, v: float) -> FloatArray:
    origin = s.vertices[0]
    return origin + u * (s.vertices[1] - origin) + v * (s.vertices[3] - origin)


def exact_flux_density(point: FloatArray, normal: FloatArray, source: FloatArray) -> float:
    delta = np.asarray(point, dtype=np.float64) - source
    radius = float(np.linalg.norm(delta))
    if radius <= 0.0:
        raise RuntimeError("Point-source singularity reached unexpectedly.")
    return float(np.dot(delta, normal) / (radius**3))


def potential(point: FloatArray, source: FloatArray) -> float:
    radius = float(np.linalg.norm(np.asarray(point, dtype=np.float64) - source))
    if radius <= 0.0:
        raise RuntimeError("Point-source singularity reached unexpectedly.")
    return 1.0 / radius


def first_clearance(point: FloatArray, conductors: Sequence[Any]) -> float:
    value = min(float(c.distance_to_point(point)) for c in conductors)
    if value <= 0.0 or not math.isfinite(value):
        raise RuntimeError(f"Non-positive first clearance: {value}")
    return value


def sample_index(cdf: FloatArray, u: float) -> int:
    return min(int(np.searchsorted(cdf, u, side="right")), len(cdf) - 1)


def first_hop_geometry(
    *,
    start: FloatArray,
    normal: FloatArray,
    panel_u: float,
    table: Table,
    conductors: Sequence[Any],
) -> tuple[float, FloatArray, float]:
    """Return (omega, next_point, clearance) once for all manufactured sources."""
    h = first_clearance(start, conductors)
    panel = sample_index(table.cdf, panel_u)
    unit_normal = normal / float(np.linalg.norm(normal))
    omega = -float(np.dot(unit_normal, table.wvt[panel])) / h
    next_point = start + h * table.panel_centers[panel]
    return omega, next_point, h


def deterministic_flux(strata: Sequence[Stratum], source: FloatArray, order: int) -> tuple[float, dict[str, float]]:
    nodes, weights = leggauss(order)
    uv = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    total = 0.0
    per_face: dict[str, float] = {}
    for s in strata:
        accum = 0.0
        for i, u in enumerate(uv):
            for j, v in enumerate(uv):
                point = sample_face_point(s, float(u), float(v))
                accum += float(w[i] * w[j]) * exact_flux_density(point, s.normal, source)
        value = s.area * accum
        per_face[s.key] = value
        total += value
    return total, per_face


def combine_seed_values(values: Sequence[float], within_ses: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    s = np.asarray(within_ses, dtype=np.float64)
    n = len(a)
    return {
        "mean": float(a.mean()),
        "combined_within_seed_se": float(math.sqrt(np.sum(s * s)) / n),
        "between_seed_se": float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parser().parse_args()
    if args.samples_per_face_per_seed <= 1:
        raise SystemExit("--samples-per-face-per-seed must be greater than 1.")
    if args.quadrature_order < 4:
        raise SystemExit("--quadrature-order must be at least 4.")
    if not args.seeds:
        raise SystemExit("At least one seed is required.")

    scales = sorted(set(float(x) for x in args.bgs_scale_factors))
    conductors = parse_matlab_geometry(args.input_file)
    table = load_table(args.table)
    scale_data = build_scale_data(conductors, scales, args.master, args.max_distance)
    sources = source_points(scale_data[scales[0]]["net_vgss"])
    stratum_keys = tuple(s.key for s in scale_data[scales[0]]["strata"])

    deterministic: dict[float, dict[str, Any]] = {}
    for scale in scales:
        deterministic[scale] = {}
        for source_name, source in sources.items():
            total, per_face = deterministic_flux(
                scale_data[scale]["strata"], source, args.quadrature_order
            )
            deterministic[scale][source_name] = {
                "total": total,
                "per_face": per_face,
                "relative_error_vs_4pi": total / EXACT_FLUX - 1.0,
            }

    print("GGFT resolution:", table.resolution)
    print("Master:", args.master)
    print("Exact manufactured flux: 4*pi =", f"{EXACT_FLUX:.15g}")
    print("Sources:", ", ".join(sources))
    for scale in scales:
        print(f"  scale {scale:g}: area={scale_data[scale]['total_area']:.12g}")

    seed_results: list[dict[str, Any]] = []
    total_per_seed = len(stratum_keys) * args.samples_per_face_per_seed
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        accum: dict[float, dict[str, dict[str, FaceAccumulator]]] = {
            scale: {
                source_name: {key: FaceAccumulator() for key in stratum_keys}
                for source_name in sources
            }
            for scale in scales
        }
        done = 0
        progress_step = max(1, total_per_seed // 20)
        for key in stratum_keys:
            for _ in range(args.samples_per_face_per_seed):
                u, v, panel_u = rng.random(3)
                for scale in scales:
                    s = scale_data[scale]["strata_by_key"][key]
                    start = sample_face_point(s, float(u), float(v))
                    omega, next_point, h = first_hop_geometry(
                        start=start,
                        normal=s.normal,
                        panel_u=float(panel_u),
                        table=table,
                        conductors=conductors,
                    )
                    for source_name, source in sources.items():
                        exact = exact_flux_density(start, s.normal, source)
                        numerical = omega * potential(next_point, source)
                        a = accum[scale][source_name][key]
                        a.exact.add(exact)
                        a.numerical.add(numerical)
                        a.difference.add(numerical - exact)
                        a.clearances.add(h)
                done += 1
                if done % progress_step == 0 or done == total_per_seed:
                    print(
                        f"\r  seed {seed}: {100.0*done/total_per_seed:5.1f}% "
                        f"({done}/{total_per_seed})",
                        end="",
                        flush=True,
                    )
        print()

        seed_out: dict[str, Any] = {"seed": seed, "scales": {}}
        for scale in scales:
            scale_out: dict[str, Any] = {}
            for source_name in sources:
                exact_total = 0.0
                numerical_total = 0.0
                difference_total = 0.0
                exact_var = 0.0
                numerical_var = 0.0
                difference_var = 0.0
                face_out: dict[str, Any] = {}
                for key in stratum_keys:
                    s = scale_data[scale]["strata_by_key"][key]
                    a = accum[scale][source_name][key]
                    ex = s.area * a.exact.mean
                    nu = s.area * a.numerical.mean
                    di = s.area * a.difference.mean
                    ex_se = s.area * a.exact.se_mean
                    nu_se = s.area * a.numerical.se_mean
                    di_se = s.area * a.difference.se_mean
                    exact_total += ex
                    numerical_total += nu
                    difference_total += di
                    exact_var += ex_se**2
                    numerical_var += nu_se**2
                    difference_var += di_se**2
                    face_out[key] = {
                        "area": s.area,
                        "direction": s.direction,
                        "exact_mc": ex,
                        "numerical_wvt": nu,
                        "numerical_minus_exact": di,
                        "exact_mc_se": ex_se,
                        "numerical_wvt_se": nu_se,
                        "difference_se": di_se,
                        "mean_first_clearance": a.clearances.mean,
                    }
                scale_out[source_name] = {
                    "exact_mc": exact_total,
                    "numerical_wvt": numerical_total,
                    "numerical_minus_exact": difference_total,
                    "exact_mc_se": math.sqrt(exact_var),
                    "numerical_wvt_se": math.sqrt(numerical_var),
                    "difference_se": math.sqrt(difference_var),
                    "face_results": face_out,
                }
            seed_out["scales"][str(scale)] = scale_out
        seed_results.append(seed_out)

    aggregate: dict[str, Any] = {}
    for scale in scales:
        aggregate[str(scale)] = {}
        for source_name in sources:
            exact_vals = [r["scales"][str(scale)][source_name]["exact_mc"] for r in seed_results]
            exact_ses = [r["scales"][str(scale)][source_name]["exact_mc_se"] for r in seed_results]
            num_vals = [r["scales"][str(scale)][source_name]["numerical_wvt"] for r in seed_results]
            num_ses = [r["scales"][str(scale)][source_name]["numerical_wvt_se"] for r in seed_results]
            diff_vals = [r["scales"][str(scale)][source_name]["numerical_minus_exact"] for r in seed_results]
            diff_ses = [r["scales"][str(scale)][source_name]["difference_se"] for r in seed_results]
            exact_combined = combine_seed_values(exact_vals, exact_ses)
            num_combined = combine_seed_values(num_vals, num_ses)
            diff_combined = combine_seed_values(diff_vals, diff_ses)
            aggregate[str(scale)][source_name] = {
                "deterministic_exact": deterministic[scale][source_name]["total"],
                "deterministic_relative_error_vs_4pi": deterministic[scale][source_name]["relative_error_vs_4pi"],
                "exact_mc": exact_combined,
                "numerical_wvt": num_combined,
                "numerical_minus_exact_mc": diff_combined,
                "numerical_relative_error_vs_4pi": num_combined["mean"] / EXACT_FLUX - 1.0,
                "numerical_over_exact_mc_minus_one": num_combined["mean"] / exact_combined["mean"] - 1.0,
            }

    summary = {
        "description": "Manufactured point-source flux validation of BGS/WVT first-hop estimator.",
        "parameters": {
            "input_file": str(args.input_file),
            "master": args.master,
            "table": str(args.table),
            "resolution": table.resolution,
            "bgs_scale_factors": scales,
            "samples_per_face_per_seed": args.samples_per_face_per_seed,
            "seeds": args.seeds,
            "quadrature_order": args.quadrature_order,
            "exact_flux": EXACT_FLUX,
        },
        "sources": {name: point.tolist() for name, point in sources.items()},
        "surface_areas": {str(s): scale_data[s]["total_area"] for s in scales},
        "deterministic": {str(s): deterministic[s] for s in scales},
        "seed_results": seed_results,
        "aggregate": aggregate,
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{args.label}_summary.json"
    entries_path = out / f"{args.label}_entries.csv"
    faces_path = out / f"{args.label}_face_contributions.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    entries: list[dict[str, Any]] = []
    face_rows: list[dict[str, Any]] = []
    for scale in scales:
        for source_name in sources:
            a = aggregate[str(scale)][source_name]
            entries.append(
                {
                    "scale_factor": scale,
                    "source": source_name,
                    "exact_flux_4pi": EXACT_FLUX,
                    "deterministic_exact": a["deterministic_exact"],
                    "deterministic_rel_error": a["deterministic_relative_error_vs_4pi"],
                    "exact_mc": a["exact_mc"]["mean"],
                    "exact_mc_within_SE": a["exact_mc"]["combined_within_seed_se"],
                    "exact_mc_between_SE": a["exact_mc"]["between_seed_se"],
                    "numerical_wvt": a["numerical_wvt"]["mean"],
                    "numerical_wvt_within_SE": a["numerical_wvt"]["combined_within_seed_se"],
                    "numerical_wvt_between_SE": a["numerical_wvt"]["between_seed_se"],
                    "numerical_rel_error_vs_4pi": a["numerical_relative_error_vs_4pi"],
                    "numerical_minus_exact_mc": a["numerical_minus_exact_mc"]["mean"],
                    "paired_difference_within_SE": a["numerical_minus_exact_mc"]["combined_within_seed_se"],
                    "paired_difference_between_SE": a["numerical_minus_exact_mc"]["between_seed_se"],
                    "numerical_over_exact_mc_minus_one": a["numerical_over_exact_mc_minus_one"],
                }
            )
            for key in stratum_keys:
                det_face = deterministic[scale][source_name]["per_face"][key]
                exact_values = np.asarray(
                    [r["scales"][str(scale)][source_name]["face_results"][key]["exact_mc"] for r in seed_results],
                    dtype=np.float64,
                )
                num_values = np.asarray(
                    [r["scales"][str(scale)][source_name]["face_results"][key]["numerical_wvt"] for r in seed_results],
                    dtype=np.float64,
                )
                diff_values = np.asarray(
                    [r["scales"][str(scale)][source_name]["face_results"][key]["numerical_minus_exact"] for r in seed_results],
                    dtype=np.float64,
                )
                first = seed_results[0]["scales"][str(scale)][source_name]["face_results"][key]
                face_rows.append(
                    {
                        "scale_factor": scale,
                        "source": source_name,
                        "stratum": key,
                        "direction": first["direction"],
                        "area": first["area"],
                        "deterministic_exact": det_face,
                        "exact_mc": float(exact_values.mean()),
                        "numerical_wvt": float(num_values.mean()),
                        "numerical_minus_exact_mc": float(diff_values.mean()),
                        "mean_first_clearance": float(
                            np.mean([
                                r["scales"][str(scale)][source_name]["face_results"][key]["mean_first_clearance"]
                                for r in seed_results
                            ])
                        ),
                    }
                )

    write_csv(entries_path, entries)
    write_csv(faces_path, face_rows)

    print("\nAggregate manufactured-flux results:")
    for row in entries:
        print(
            f"  scale={row['scale_factor']:g} source={row['source']:>6s} "
            f"WVT={row['numerical_wvt']:.9f} "
            f"rel_err={100.0*row['numerical_rel_error_vs_4pi']:+.4f}% "
            f"paired(WVT-exact)={row['numerical_minus_exact_mc']:+.3e}"
        )
    print("\nOutputs:")
    for p in (json_path, entries_path, faces_path):
        print(" ", p)


if __name__ == "__main__":
    main()
