#!/usr/bin/env python3
"""Exact end-to-end FRW benchmark using an isolated conducting sphere.

The physical benchmark is a sphere of radius ``a`` held at unit potential in
free space.  Its exact capacitance is

    C_exact = 4*pi*epsilon_0*a.

A cubical Gaussian surface encloses the sphere.  The diagnostic samples every
Gaussian-surface face separately, applies the FRW first-hop WVT weight, follows
repeated maximal axis-aligned cube transitions, and scores whether the path
ultimately reaches the sphere rather than infinity.

Two coupled modes are evaluated:

* ``numerical_n200``: bundled numerical probabilities and WVT;
* ``analytical_n200``: independently generated panel-integrated probabilities
  and gradients at the same resolution and panel centres.

At a configurable far radius, the remaining exterior problem is closed exactly:
from radius r the probability of ever hitting a sphere of radius a is a/r.
This avoids a finite-kill-radius bias while keeping the run inexpensive.

No production source file is modified, and no external capacitance solver is
used.
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

E0 = 8.8541878128e-12
REFERENCE_SIDE_LENGTH = 2.0
FACE_NAMES = ("PX", "NX", "PY", "NY", "PZ", "NZ")


@dataclass(frozen=True)
class TransitionTable:
    name: str
    resolution: int
    probabilities: np.ndarray
    cdf: np.ndarray
    panel_centers: np.ndarray
    gradients: np.ndarray
    wvt: np.ndarray


@dataclass(frozen=True)
class MaximalCoupling:
    common_mass: float
    common_cdf: np.ndarray
    numerical_residual_cdf: np.ndarray
    analytical_residual_cdf: np.ndarray


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
class FaceModeStats:
    values: Moments = field(default_factory=Moments)
    hops: Moments = field(default_factory=Moments)
    hits: int = 0
    escaped: int = 0
    unresolved: int = 0
    first_clearance: Moments = field(default_factory=Moments)


@dataclass
class FacePairStats:
    difference: Moments = field(default_factory=Moments)
    agreements: int = 0
    count: int = 0

    def add(self, first: float, second: float, first_hit: bool, second_hit: bool) -> None:
        self.difference.add(second - first)
        self.count += 1
        if first_hit == second_hit:
            self.agreements += 1


@dataclass(frozen=True)
class PathResult:
    value: float
    hit: bool
    resolved: bool
    hops: int
    first_clearance: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=Path("src/GGFT/data/cube_tables_n201.npz"))
    p.add_argument("--sphere-radius", type=float, default=1.0)
    p.add_argument(
        "--bgs-half-size-factors", type=float, nargs="+", default=[1.25, 2.0, 4.0],
        help="Cubical BGS half-size divided by sphere radius; every value must exceed 1.",
    )
    p.add_argument("--samples-per-face-per-seed", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 5678, 9012, 3456])
    p.add_argument("--series-terms", type=int, default=41)
    p.add_argument(
        "--tail-radius-factor", type=float, default=50.0,
        help="At r >= factor*a, close the exterior exactly with hit probability a/r.",
    )
    p.add_argument("--hit-tolerance-factor", type=float, default=1e-6)
    p.add_argument("--max-hops", type=int, default=2000)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("output/diagnostics/frw_isolated_sphere"),
    )
    p.add_argument("--label", default="isolated_sphere")
    return p.parse_args()


def normalized_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("Probabilities must be finite and non-negative.")
    total = float(p.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Probability sum must be finite and positive.")
    p = p / total
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    return p, cdf


def load_numerical_table(path: Path) -> TransitionTable:
    if not path.is_file():
        raise FileNotFoundError(f"GGFT table not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        resolution = int(np.asarray(data["resolution"]).item())
        probabilities = np.asarray(data["probabilities"], dtype=np.float64).reshape(-1)
        centers = np.asarray(data["panel_centers"], dtype=np.float64).reshape(-1, 3)
        gradients = np.asarray(data["gradients"], dtype=np.float64).reshape(-1, 3)
        wvt = np.asarray(data["wvt"], dtype=np.float64).reshape(-1, 3)
    p, cdf = normalized_cdf(probabilities)
    expected = 6 * resolution * resolution
    for name, array, shape in (
        ("probabilities", p, (expected,)),
        ("panel_centers", centers, (expected, 3)),
        ("gradients", gradients, (expected, 3)),
        ("wvt", wvt, (expected, 3)),
    ):
        if array.shape != shape:
            raise ValueError(f"Unexpected {name} shape {array.shape}; expected {shape}.")
    return TransitionTable(
        name=f"numerical_n{resolution}", resolution=resolution,
        probabilities=p, cdf=cdf, panel_centers=centers,
        gradients=gradients, wvt=wvt,
    )


def build_panel_centers(resolution: int) -> np.ndarray:
    axis = -1.0 + (np.arange(resolution, dtype=np.float64) + 0.5) * (2.0 / resolution)
    centers: list[tuple[float, float, float]] = []
    centers.extend((x, y, -1.0) for y in axis for x in axis)
    centers.extend((x, y, 1.0) for y in axis for x in axis)
    centers.extend((x, -1.0, z) for z in axis for x in axis)
    centers.extend((x, 1.0, z) for z in axis for x in axis)
    centers.extend((-1.0, y, z) for z in axis for y in axis)
    centers.extend((1.0, y, z) for z in axis for y in axis)
    return np.asarray(centers, dtype=np.float64)


def integrated_sine_basis(resolution: int, terms: int) -> np.ndarray:
    edges = np.linspace(0.0, REFERENCE_SIDE_LENGTH, resolution + 1, dtype=np.float64)
    modes = np.arange(1, terms + 1, dtype=np.float64)
    k = modes * np.pi / REFERENCE_SIDE_LENGTH
    return (np.cos(edges[:-1, None] * k) - np.cos(edges[1:, None] * k)) / k


def exact_center_sine_cosine(terms: int) -> tuple[np.ndarray, np.ndarray]:
    modes = np.arange(1, terms + 1, dtype=np.int64)
    sine = np.zeros(terms, dtype=np.float64)
    cosine = np.zeros(terms, dtype=np.float64)
    odd = modes % 2 == 1
    sine[odd] = (-1.0) ** ((modes[odd] - 1) // 2)
    even = ~odd
    cosine[even] = (-1.0) ** (modes[even] // 2)
    return sine, cosine


def stable_normal_factors(terms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modes = np.arange(1, terms + 1, dtype=np.float64)
    wave_numbers = modes * np.pi / REFERENCE_SIDE_LENGTH
    lam = np.sqrt(wave_numbers[:, None] ** 2 + wave_numbers[None, :] ** 2)
    a = lam * REFERENCE_SIDE_LENGTH / 2.0
    exp_minus_a = np.exp(-a)
    value_factor = exp_minus_a / (1.0 + np.exp(-2.0 * a))
    derivative_factor = lam * exp_minus_a / (1.0 - np.exp(-2.0 * a))
    return wave_numbers, value_factor, derivative_factor


def face_grid(basis: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return basis @ coefficient.T @ basis.T


def build_analytical_table(resolution: int, terms: int) -> TransitionTable:
    basis = integrated_sine_basis(resolution, terms)
    sine, cosine = exact_center_sine_cosine(terms)
    wave_numbers, value_factor, derivative_factor = stable_normal_factors(terms)
    prefactor = 4.0 / (REFERENCE_SIDE_LENGTH**2)

    base = prefactor * sine[:, None] * sine[None, :] * value_factor
    tangent_first = (
        prefactor * (wave_numbers * cosine)[:, None] * sine[None, :] * value_factor
    )
    tangent_second = (
        prefactor * sine[:, None] * (wave_numbers * cosine)[None, :] * value_factor
    )
    normal = prefactor * sine[:, None] * sine[None, :] * derivative_factor

    p_grid = face_grid(basis, base)
    t1_grid = face_grid(basis, tangent_first)
    t2_grid = face_grid(basis, tangent_second)
    n_grid = face_grid(basis, normal)

    probability_faces: list[np.ndarray] = []
    gradient_faces: list[np.ndarray] = []
    for sign in (-1.0, 1.0):
        probability_faces.append(p_grid.ravel())
        gradient_faces.append(np.column_stack((t1_grid.ravel(), t2_grid.ravel(), (sign*n_grid).ravel())))
    for sign in (-1.0, 1.0):
        probability_faces.append(p_grid.ravel())
        gradient_faces.append(np.column_stack((t1_grid.ravel(), (sign*n_grid).ravel(), t2_grid.ravel())))
    for sign in (-1.0, 1.0):
        probability_faces.append(p_grid.ravel())
        gradient_faces.append(np.column_stack(((sign*n_grid).ravel(), t1_grid.ravel(), t2_grid.ravel())))

    probabilities_raw = np.concatenate(probability_faces)
    gradients = np.vstack(gradient_faces)
    if np.any(probabilities_raw <= np.finfo(np.float64).tiny):
        count = int(np.count_nonzero(probabilities_raw <= 0.0))
        raise RuntimeError(f"Analytical series produced {count} non-positive panels.")
    total = float(probabilities_raw.sum())
    probabilities = probabilities_raw / total
    gradients = gradients / total
    wvt = gradients / probabilities[:, None]
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    return TransitionTable(
        name=f"analytical_n{resolution}_t{terms}", resolution=resolution,
        probabilities=probabilities, cdf=cdf,
        panel_centers=build_panel_centers(resolution),
        gradients=gradients, wvt=wvt,
    )


def cdf_or_placeholder(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0.0:
        out = np.zeros_like(values)
        out[-1] = 1.0
        return out
    out = np.cumsum(values / total)
    out[-1] = 1.0
    return out


def build_maximal_coupling(numerical: np.ndarray, analytical: np.ndarray) -> MaximalCoupling:
    common = np.minimum(numerical, analytical)
    return MaximalCoupling(
        common_mass=float(common.sum()),
        common_cdf=cdf_or_placeholder(common),
        numerical_residual_cdf=cdf_or_placeholder(numerical-common),
        analytical_residual_cdf=cdf_or_placeholder(analytical-common),
    )


def sample_index(cdf: np.ndarray, u: float) -> int:
    return min(int(np.searchsorted(cdf, u, side="right")), len(cdf)-1)


def coupled_indices(coupling: MaximalCoupling, rng: np.random.Generator) -> tuple[int, int]:
    branch, u1, u2 = rng.random(3)
    if branch < coupling.common_mass:
        idx = sample_index(coupling.common_cdf, float(u1))
        return idx, idx
    return (
        sample_index(coupling.numerical_residual_cdf, float(u1)),
        sample_index(coupling.analytical_residual_cdf, float(u2)),
    )


def l_inf_distance_to_sphere(point: np.ndarray, radius: float) -> float:
    """Exact L-infinity distance from an exterior point to a solid sphere."""
    point = np.asarray(point, dtype=np.float64)
    if float(np.dot(point, point)) <= radius*radius:
        return 0.0
    a = np.sort(np.abs(point))[::-1]

    t1 = a[0] - radius
    if t1 >= a[1] - 1e-14*max(1.0, radius):
        return max(float(t1), 0.0)

    disc2 = 2.0*radius*radius - (a[0]-a[1])**2
    if disc2 >= -1e-13*radius*radius:
        t2 = 0.5*(a[0]+a[1]-math.sqrt(max(float(disc2), 0.0)))
        if t2 >= a[2] - 1e-14*max(1.0, radius):
            return max(float(t2), 0.0)

    s = float(a.sum())
    disc3 = s*s - 3.0*(float(np.dot(a, a))-radius*radius)
    t3 = (s-math.sqrt(max(disc3, 0.0)))/3.0
    return max(float(t3), 0.0)


def face_point_and_normal(face: str, half_size: float, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    p = -half_size + 2.0*half_size*np.array([u, v], dtype=np.float64)
    if face == "PX": return np.array([half_size, p[0], p[1]]), np.array([1.0,0.0,0.0])
    if face == "NX": return np.array([-half_size, p[0], p[1]]), np.array([-1.0,0.0,0.0])
    if face == "PY": return np.array([p[0], half_size, p[1]]), np.array([0.0,1.0,0.0])
    if face == "NY": return np.array([p[0], -half_size, p[1]]), np.array([0.0,-1.0,0.0])
    if face == "PZ": return np.array([p[0], p[1], half_size]), np.array([0.0,0.0,1.0])
    if face == "NZ": return np.array([p[0], p[1], -half_size]), np.array([0.0,0.0,-1.0])
    raise KeyError(face)


def simulate_coupled(
    start: np.ndarray,
    normal: np.ndarray,
    radius: float,
    numerical: TransitionTable,
    analytical: TransitionTable,
    coupling: MaximalCoupling,
    rng: np.random.Generator,
    tail_radius: float,
    hit_tolerance: float,
    max_hops: int,
) -> tuple[PathResult, PathResult]:
    first_clearance = l_inf_distance_to_sphere(start, radius)
    if first_clearance <= 0.0:
        raise RuntimeError("BGS point is not outside the sphere.")

    tail_u = float(rng.random())
    idx_n, idx_a = coupled_indices(coupling, rng)
    omega_n = -float(np.dot(normal, numerical.wvt[idx_n])) / first_clearance
    omega_a = -float(np.dot(normal, analytical.wvt[idx_a])) / first_clearance
    points = [
        start + first_clearance*numerical.panel_centers[idx_n],
        start + first_clearance*analytical.panel_centers[idx_a],
    ]
    tables = (numerical, analytical)
    omegas = (omega_n, omega_a)
    active = [True, True]
    hits = [False, False]
    resolved = [False, False]
    hops = [1, 1]

    for _ in range(max_hops):
        for m in range(2):
            if not active[m]:
                continue
            d = l_inf_distance_to_sphere(points[m], radius)
            if d <= hit_tolerance:
                hits[m] = True
                resolved[m] = True
                active[m] = False
                continue
            r = float(np.linalg.norm(points[m]))
            if r >= tail_radius:
                hits[m] = tail_u < radius/r
                resolved[m] = True
                active[m] = False

        if not any(active):
            break

        if active[0] and active[1]:
            idx_n, idx_a = coupled_indices(coupling, rng)
            for m, idx in enumerate((idx_n, idx_a)):
                d = l_inf_distance_to_sphere(points[m], radius)
                points[m] = points[m] + d*tables[m].panel_centers[idx]
                hops[m] += 1
        elif active[0]:
            idx = sample_index(numerical.cdf, float(rng.random()))
            d = l_inf_distance_to_sphere(points[0], radius)
            points[0] = points[0] + d*numerical.panel_centers[idx]
            hops[0] += 1
        else:
            idx = sample_index(analytical.cdf, float(rng.random()))
            d = l_inf_distance_to_sphere(points[1], radius)
            points[1] = points[1] + d*analytical.panel_centers[idx]
            hops[1] += 1

    results = []
    for m in range(2):
        value = omegas[m] if hits[m] and resolved[m] else 0.0
        results.append(PathResult(
            value=float(value), hit=bool(hits[m]), resolved=bool(resolved[m]),
            hops=int(hops[m]), first_clearance=float(first_clearance),
        ))
    return results[0], results[1]


def combine_seed_uncertainty(values: list[float], within_ses: list[float]) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    n = len(values)
    within = math.sqrt(sum(se*se for se in within_ses))/n
    between = float(np.std(values, ddof=1)/math.sqrt(n)) if n > 1 else 0.0
    return mean, within, between


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.sphere_radius <= 0.0:
        raise ValueError("--sphere-radius must be positive.")
    if args.samples_per_face_per_seed < 2:
        raise ValueError("At least two samples per face per seed are required.")
    factors = sorted(set(float(x) for x in args.bgs_half_size_factors))
    if any(x <= 1.0 for x in factors):
        raise ValueError("Every BGS half-size factor must exceed 1.0.")
    if args.tail_radius_factor <= max(factors):
        raise ValueError("--tail-radius-factor must exceed every BGS factor.")

    numerical = load_numerical_table(args.table)
    analytical = build_analytical_table(numerical.resolution, args.series_terms)
    center_error = float(np.max(np.abs(numerical.panel_centers-analytical.panel_centers)))
    if center_error > 1e-13:
        raise RuntimeError(f"Panel-centre alignment error: {center_error}")
    coupling = build_maximal_coupling(numerical.probabilities, analytical.probabilities)

    radius = float(args.sphere_radius)
    exact_c = 4.0*math.pi*E0*radius
    tail_radius = args.tail_radius_factor*radius
    hit_tolerance = args.hit_tolerance_factor*radius
    modes = (numerical.name, analytical.name)

    print(f"Numerical table: {numerical.name}")
    print(f"Analytical table: {analytical.name}")
    print(f"Common panel probability: {coupling.common_mass:.12f}")
    print(f"Exact capacitance: {exact_c:.12e} F")
    print(f"BGS factors: {' '.join(f'{x:g}' for x in factors)}")

    seed_records: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"Seed {seed}...")
        stats: dict[tuple[float,str,str], FaceModeStats] = {}
        mode_pairs: dict[tuple[float,str], FacePairStats] = {}
        scale_pairs: dict[tuple[str,float,float,str], FacePairStats] = {}
        for factor in factors:
            for face in FACE_NAMES:
                for mode in modes:
                    stats[(factor, face, mode)] = FaceModeStats()
                mode_pairs[(factor, face)] = FacePairStats()
        for mode in modes:
            for f1, f2 in zip(factors[:-1], factors[1:]):
                for face in FACE_NAMES:
                    scale_pairs[(mode,f1,f2,face)] = FacePairStats()
            if len(factors) > 2:
                for face in FACE_NAMES:
                    scale_pairs[(mode,factors[0],factors[-1],face)] = FacePairStats()

        for face_index, face in enumerate(FACE_NAMES):
            surface_rng = np.random.default_rng(np.random.SeedSequence([seed, face_index, 99173]))
            for sample_index in range(args.samples_per_face_per_seed):
                u, v = surface_rng.random(2)
                sample_results: dict[tuple[float,str], PathResult] = {}
                path_seed = np.random.SeedSequence([seed, face_index, sample_index, 71237])
                seed_words = path_seed.generate_state(4, dtype=np.uint32)
                for factor in factors:
                    half = factor*radius
                    start, normal = face_point_and_normal(face, half, float(u), float(v))
                    # Identical random stream across BGS factors for this face/sample.
                    rng = np.random.default_rng(seed_words)
                    result_n, result_a = simulate_coupled(
                        start=start, normal=normal, radius=radius,
                        numerical=numerical, analytical=analytical, coupling=coupling,
                        rng=rng, tail_radius=tail_radius,
                        hit_tolerance=hit_tolerance, max_hops=args.max_hops,
                    )
                    for mode, result in zip(modes, (result_n, result_a)):
                        s = stats[(factor,face,mode)]
                        s.values.add(result.value)
                        s.hops.add(result.hops)
                        s.first_clearance.add(result.first_clearance)
                        if not result.resolved:
                            s.unresolved += 1
                        elif result.hit:
                            s.hits += 1
                        else:
                            s.escaped += 1
                        sample_results[(factor,mode)] = result
                    mode_pairs[(factor,face)].add(
                        result_n.value, result_a.value, result_n.hit, result_a.hit
                    )

                for mode in modes:
                    comparisons = list(zip(factors[:-1], factors[1:]))
                    if len(factors) > 2:
                        comparisons.append((factors[0], factors[-1]))
                    for f1, f2 in comparisons:
                        r1, r2 = sample_results[(f1,mode)], sample_results[(f2,mode)]
                        a1 = 4.0*(f1*radius)**2
                        a2 = 4.0*(f2*radius)**2
                        # Store already area-weighted per-face physical contributions.
                        scale_pairs[(mode,f1,f2,face)].add(
                            E0*a1*r1.value, E0*a2*r2.value, r1.hit, r2.hit
                        )

        seed_mode_results: dict[str, Any] = {}
        for factor in factors:
            factor_key = f"{factor:g}"
            seed_mode_results[factor_key] = {}
            face_area = 4.0*(factor*radius)**2
            for mode in modes:
                face_rows = {}
                estimate = 0.0
                variance = 0.0
                total_hops = 0.0
                total_count = 0
                total_hits = total_escaped = total_unresolved = 0
                for face in FACE_NAMES:
                    s = stats[(factor,face,mode)]
                    contribution = E0*face_area*s.values.mean
                    contribution_se = E0*face_area*s.values.se_mean
                    estimate += contribution
                    variance += contribution_se**2
                    total_hops += s.hops.total
                    total_count += s.hops.count
                    total_hits += s.hits
                    total_escaped += s.escaped
                    total_unresolved += s.unresolved
                    face_rows[face] = {
                        "area": face_area,
                        "contribution_F": contribution,
                        "within_seed_SE_F": contribution_se,
                        "hit_rate": s.hits/s.values.count,
                        "mean_hops": s.hops.mean,
                        "mean_first_clearance": s.first_clearance.mean,
                    }
                seed_mode_results[factor_key][mode] = {
                    "estimate_F": estimate,
                    "within_seed_SE_F": math.sqrt(variance),
                    "relative_error_vs_exact": estimate/exact_c-1.0,
                    "resolved_rate": 1.0-total_unresolved/total_count,
                    "hit_rate": total_hits/total_count,
                    "mean_hops": total_hops/total_count,
                    "termination_counts": {
                        "hit": total_hits, "escaped": total_escaped,
                        "unresolved": total_unresolved,
                    },
                    "faces": face_rows,
                }

            pair = mode_pairs[(factor, FACE_NAMES[0])]
            delta = 0.0
            variance = 0.0
            agreements = count = 0
            face_deltas = {}
            for face in FACE_NAMES:
                p = mode_pairs[(factor,face)]
                d = E0*face_area*p.difference.mean
                se = E0*face_area*p.difference.se_mean
                delta += d
                variance += se**2
                agreements += p.agreements
                count += p.count
                face_deltas[face] = {"delta_F": d, "within_seed_SE_F": se}
            seed_mode_results[factor_key]["analytical_minus_numerical"] = {
                "delta_F": delta,
                "within_seed_SE_F": math.sqrt(variance),
                "target_agreement_rate": agreements/count,
                "faces": face_deltas,
            }

        seed_scale_results: dict[str, Any] = {}
        for mode in modes:
            comparisons = list(zip(factors[:-1], factors[1:]))
            if len(factors) > 2:
                comparisons.append((factors[0], factors[-1]))
            for f1,f2 in comparisons:
                key = f"{mode}:factor_{f2:g}_minus_{f1:g}"
                delta = variance = 0.0
                agreements = count = 0
                for face in FACE_NAMES:
                    p = scale_pairs[(mode,f1,f2,face)]
                    delta += p.difference.mean
                    variance += p.difference.se_mean**2
                    agreements += p.agreements
                    count += p.count
                seed_scale_results[key] = {
                    "mode": mode, "first_factor": f1, "second_factor": f2,
                    "delta_F": delta, "within_seed_SE_F": math.sqrt(variance),
                    "target_agreement_rate": agreements/count,
                }

        seed_records.append({
            "seed": int(seed),
            "samples_per_face": int(args.samples_per_face_per_seed),
            "modes": seed_mode_results,
            "scale_differences": seed_scale_results,
        })

    aggregate: dict[str, Any] = {}
    entries_rows: list[dict[str, Any]] = []
    face_rows_out: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for factor in factors:
        fkey = f"{factor:g}"
        aggregate[fkey] = {}
        for mode in modes:
            vals = [r["modes"][fkey][mode]["estimate_F"] for r in seed_records]
            ses = [r["modes"][fkey][mode]["within_seed_SE_F"] for r in seed_records]
            mean, within, between = combine_seed_uncertainty(vals, ses)
            combined = math.hypot(within, between)
            aggregate[fkey][mode] = {
                "estimate_F": mean,
                "within_seed_SE_of_mean_F": within,
                "between_seed_SE_F": between,
                "combined_SE_F": combined,
                "relative_error_vs_exact": mean/exact_c-1.0,
                "z_vs_exact_combined": (mean-exact_c)/combined if combined else 0.0,
                "mean_hops": float(np.mean([r["modes"][fkey][mode]["mean_hops"] for r in seed_records])),
                "mean_hit_rate": float(np.mean([r["modes"][fkey][mode]["hit_rate"] for r in seed_records])),
                "mean_resolved_rate": float(np.mean([r["modes"][fkey][mode]["resolved_rate"] for r in seed_records])),
            }
            entries_rows.append({
                "bgs_half_size_factor": factor, "mode": mode,
                "estimate_F": mean, "exact_F": exact_c,
                "relative_error_vs_exact": mean/exact_c-1.0,
                "within_seed_SE_of_mean_F": within,
                "between_seed_SE_F": between,
                "combined_SE_F": combined,
                "z_vs_exact_combined": (mean-exact_c)/combined if combined else 0.0,
                "mean_hops": aggregate[fkey][mode]["mean_hops"],
                "hit_rate": aggregate[fkey][mode]["mean_hit_rate"],
                "resolved_rate": aggregate[fkey][mode]["mean_resolved_rate"],
            })
            face_area = 4.0*(factor*radius)**2
            for face in FACE_NAMES:
                values = [r["modes"][fkey][mode]["faces"][face]["contribution_F"] for r in seed_records]
                face_ses = [r["modes"][fkey][mode]["faces"][face]["within_seed_SE_F"] for r in seed_records]
                fm, fw, fb = combine_seed_uncertainty(values, face_ses)
                face_rows_out.append({
                    "bgs_half_size_factor": factor, "mode": mode, "face": face,
                    "face_area": face_area, "mean_contribution_F": fm,
                    "within_seed_SE_of_mean_F": fw, "between_seed_SE_F": fb,
                    "combined_SE_F": math.hypot(fw,fb),
                    "mean_hops": float(np.mean([r["modes"][fkey][mode]["faces"][face]["mean_hops"] for r in seed_records])),
                    "mean_first_clearance": float(np.mean([r["modes"][fkey][mode]["faces"][face]["mean_first_clearance"] for r in seed_records])),
                })

        vals = [r["modes"][fkey]["analytical_minus_numerical"]["delta_F"] for r in seed_records]
        ses = [r["modes"][fkey]["analytical_minus_numerical"]["within_seed_SE_F"] for r in seed_records]
        mean, within, between = combine_seed_uncertainty(vals, ses)
        combined = math.hypot(within,between)
        paired_rows.append({
            "comparison": "analytical_minus_numerical",
            "mode": "analytical-numerical", "first_factor": factor, "second_factor": factor,
            "mean_delta_F": mean, "within_seed_SE_of_mean_F": within,
            "between_seed_SE_F": between, "combined_SE_F": combined,
            "z_combined": mean/combined if combined else 0.0,
            "target_agreement_rate": float(np.mean([r["modes"][fkey]["analytical_minus_numerical"]["target_agreement_rate"] for r in seed_records])),
        })

    scale_comparisons = list(zip(factors[:-1], factors[1:]))
    if len(factors) > 2:
        scale_comparisons.append((factors[0],factors[-1]))
    for mode in modes:
        for f1,f2 in scale_comparisons:
            key = f"{mode}:factor_{f2:g}_minus_{f1:g}"
            vals = [r["scale_differences"][key]["delta_F"] for r in seed_records]
            ses = [r["scale_differences"][key]["within_seed_SE_F"] for r in seed_records]
            mean, within, between = combine_seed_uncertainty(vals,ses)
            combined = math.hypot(within,between)
            paired_rows.append({
                "comparison": f"factor_{f2:g}_minus_{f1:g}", "mode": mode,
                "first_factor": f1, "second_factor": f2,
                "mean_delta_F": mean, "within_seed_SE_of_mean_F": within,
                "between_seed_SE_F": between, "combined_SE_F": combined,
                "z_combined": mean/combined if combined else 0.0,
                "target_agreement_rate": float(np.mean([r["scale_differences"][key]["target_agreement_rate"] for r in seed_records])),
            })

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir/args.label
    entries_path = Path(str(prefix)+"_entries.csv")
    faces_path = Path(str(prefix)+"_face_contributions.csv")
    paired_path = Path(str(prefix)+"_paired_differences.csv")
    summary_path = Path(str(prefix)+"_summary.json")

    write_csv(entries_path, list(entries_rows[0]), entries_rows)
    write_csv(faces_path, list(face_rows_out[0]), face_rows_out)
    write_csv(paired_path, list(paired_rows[0]), paired_rows)

    summary = {
        "description": "Exact end-to-end FRW isolated-sphere capacitance benchmark.",
        "parameters": {
            "table": str(args.table), "resolution": numerical.resolution,
            "series_terms": args.series_terms, "sphere_radius": radius,
            "bgs_half_size_factors": factors,
            "samples_per_face_per_seed": args.samples_per_face_per_seed,
            "seeds": args.seeds,
            "total_paths_per_mode": len(factors)*len(FACE_NAMES)*args.samples_per_face_per_seed*len(args.seeds),
            "tail_radius_factor": args.tail_radius_factor,
            "hit_tolerance_factor": args.hit_tolerance_factor,
            "max_hops": args.max_hops, "E0": E0,
        },
        "exact_capacitance_F": exact_c,
        "table_comparison": {
            "probability_l1_distance": float(np.abs(numerical.probabilities-analytical.probabilities).sum()),
            "total_variation_distance": 0.5*float(np.abs(numerical.probabilities-analytical.probabilities).sum()),
            "maximal_coupling_common_panel_probability": coupling.common_mass,
            "panel_center_max_absolute_difference": center_error,
        },
        "aggregate": aggregate,
        "paired_differences": paired_rows,
        "seed_results": seed_records,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAggregate results:")
    for row in entries_rows:
        print(
            f"  factor={row['bgs_half_size_factor']:g} {row['mode']}: "
            f"C={row['estimate_F']:.6e} F, rel={100*row['relative_error_vs_exact']:+.3f}%, "
            f"z={row['z_vs_exact_combined']:+.2f}, hops={row['mean_hops']:.2f}"
        )
    print("\nWrote:")
    for path in (summary_path, entries_path, faces_path, paired_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
