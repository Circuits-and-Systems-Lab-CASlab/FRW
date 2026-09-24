from dataclasses import dataclass
import math

import numpy as np

from geometry import Bounds
from bgs.sampling import VGSSSamplingStats, sample_on_vgss
from GGFT.ggft import (
    FIRST_TRANSITION_MODES,
    STANDARD_FIRST_TRANSITION,
    STRATIFIED_FIRST_TRANSITION,
    CubeGFT,
)
from .boundary import (
    conductor_distance,
    inside_window,
    reflect_point,
    validate_window,
    window_distance,
)


@dataclass
class WalkOutcome:
    """Compact result of one FRW walk, with an optional visualization trace."""

    start_point: list[float]
    start_normal: list[float]
    first_panel_index: int
    first_transition_mode: str
    first_transition_stratum: str | None
    omega: float
    hit_conductor: str | None
    termination: str
    hop_count: int
    path_data: list[dict] | None = None
    reflection_count: int = 0
    wall_potential: float | None = None

    def as_dict(self) -> dict:
        result = {
            "start_point": self.start_point,
            "start_normal": self.start_normal,
            "first_panel_index": self.first_panel_index,
            "first_transition_mode": self.first_transition_mode,
            "first_transition_stratum": self.first_transition_stratum,
            "omega": self.omega,
            "hit_conductor": self.hit_conductor,
            "termination": self.termination,
            "hop_count": self.hop_count,
        }
        if self.path_data is not None:
            result["path_data"] = self.path_data
        if self.reflection_count:
            result["reflection_count"] = self.reflection_count
        if self.wall_potential is not None:
            result["wall_potential"] = self.wall_potential
        return result


# ==========================================
# 1. GEOMETRY HELPERS
# ==========================================

def get_system_bounds(conductors):
    all_mins = np.array([c.min_bounds for c in conductors])
    all_maxs = np.array([c.max_bounds for c in conductors])
    g_min = np.min(all_mins, axis=0)
    g_max = np.max(all_maxs, axis=0)
    system_center = (g_max + g_min) / 2.0
    system_size = np.max(g_max - g_min)
    return system_center, system_size


def nearest_conductor(point, conductors, window=None, neumann_axes=()):
    """Return the index and distance of the conductor nearest to a point."""

    if not conductors:
        raise ValueError("At least one conductor is required.")

    if not neumann_axes:
        nearest_idx = 0
        min_dist = conductors[0].distance_to_point(point)
        for index in range(1, len(conductors)):
            distance = conductors[index].distance_to_point(point)
            if distance < min_dist:
                nearest_idx = index
                min_dist = distance
        return nearest_idx, min_dist

    nearest_idx = 0
    min_dist = conductor_distance(
        point,
        conductors[0],
        window,
        neumann_axes,
    )

    for index in range(1, len(conductors)):
        distance = conductor_distance(
            point,
            conductors[index],
            window,
            neumann_axes,
        )
        if distance < min_dist:
            nearest_idx = index
            min_dist = distance

    return nearest_idx, min_dist


def walk_radius(point, conductors, window=None, neumann_axes=()):
    """Return the nearest conductor, cube radius, and wall-limit flag."""

    nearest_idx, radius = nearest_conductor(
        point,
        conductors,
        window,
        neumann_axes,
    )
    wall_is_nearest = False

    if window is not None:
        wall_radius = window_distance(point, window, neumann_axes)
        if wall_radius <= radius:
            radius = wall_radius
            wall_is_nearest = True

    return nearest_idx, radius, wall_is_nearest


# ==========================================
# 2. PURE RANDOM WALK LOGIC
# ==========================================

def sample_point_on_bgs(bgs_min, bgs_max, rng):
    dx, dy, dz = bgs_max - bgs_min
    areas = [dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy]
    total_area = sum(areas)

    face_probs = [area / total_area for area in areas]
    face_idx = rng.choice(6, p=face_probs)

    u, v = rng.random(2)

    if face_idx == 0:
        point = [bgs_min[0], bgs_min[1] + u * dy, bgs_min[2] + v * dz]
    elif face_idx == 1:
        point = [bgs_max[0], bgs_min[1] + u * dy, bgs_min[2] + v * dz]
    elif face_idx == 2:
        point = [bgs_min[0] + u * dx, bgs_min[1], bgs_min[2] + v * dz]
    elif face_idx == 3:
        point = [bgs_min[0] + u * dx, bgs_max[1], bgs_min[2] + v * dz]
    elif face_idx == 4:
        point = [bgs_min[0] + u * dx, bgs_min[1] + v * dy, bgs_min[2]]
    else:
        point = [bgs_min[0] + u * dx, bgs_min[1] + v * dy, bgs_max[2]]

    return np.asarray(point, dtype=float)


def sample_point_on_maximal_cube(p_center, radius, rng):
    cube_min = p_center - radius
    cube_max = p_center + radius
    return sample_point_on_bgs(cube_min, cube_max, rng)


def _validate_run_configuration(num_walks, first_transition_mode):
    if num_walks <= 0:
        raise ValueError("num_walks must be a positive integer.")
    if first_transition_mode not in FIRST_TRANSITION_MODES:
        choices = ", ".join(FIRST_TRANSITION_MODES)
        raise ValueError(f"first_transition_mode must be one of: {choices}")
    if (
        first_transition_mode == STRATIFIED_FIRST_TRANSITION
        and num_walks % 2 != 0
    ):
        raise ValueError(
            "Stratified first-transition sampling requires an even number "
            "of walks for exact inward/outward allocation."
        )


def run_one_walk(
    conductors,
    master_vgss,
    gft,
    rng,
    walk_index,
    sampling_stats,
    system_center,
    math_kill_radius,
    max_hops=200,
    first_transition_mode=STANDARD_FIRST_TRANSITION,
    record_path=False,
    window: Bounds | None = None,
    neumann_axes=(),
    wall_potential=0.0,
) -> WalkOutcome:
    """Run one physical FRW walk and return its compact terminal outcome."""

    start_sample = sample_on_vgss(
        master_vgss.sampling_context,
        rng,
        stats=sampling_stats,
    )
    current_p = start_sample.point
    surface_normal = start_sample.normal
    start_point = current_p.copy()
    current_walk_path = [] if record_path else None

    hit_target = None
    termination = "max_hops"
    reflection_count = 0

    if window is not None and not inside_window(
        current_p,
        window,
        tolerance=1.0e-12,
    ):
        raise ValueError(
            "A VGSS start point lies outside the finite window."
        )

    # ---------- first transition ----------
    _nearest_idx, min_dist, _wall_is_nearest = walk_radius(
        current_p,
        conductors,
        window,
        neumann_axes,
    )
    if not math.isfinite(min_dist) or min_dist <= 0.0:
        raise RuntimeError(
            "The first-transition cube has a non-positive radius."
        )
    side_a = 2 * min_dist

    first_transition_stratum = None
    allocation_probability = None
    if first_transition_mode == STRATIFIED_FIRST_TRANSITION:
        first_transition_stratum = (
            "inward" if walk_index % 2 == 0 else "outward"
        )
        allocation_probability = 0.5

    next_point, panel_index, probability, omega = gft.sample_first_transition(
        start_point,
        min_dist,
        surface_normal,
        rng,
        mode=first_transition_mode,
        stratum=first_transition_stratum,
        allocation_probability=allocation_probability,
    )

    table_probability = float(gft.probabilities[panel_index])
    if current_walk_path is not None:
        first_path_record = {
            "center": current_p.tolist(),
            "a": float(side_a),
            "P(r,r')": table_probability,
        }
        if first_transition_mode == STRATIFIED_FIRST_TRANSITION:
            first_path_record.update(
                {
                    "sampling_probability": float(probability),
                    "first_transition_stratum": first_transition_stratum,
                }
            )
        current_walk_path.append(first_path_record)

    # The old path metric counted the first transition cube as one hop.
    hop_count = 1
    probability = table_probability
    current_p = next_point

    if window is not None:
        current_p, reflected, absorbed = reflect_point(current_p, window, neumann_axes, )
        reflection_count += reflected
        if absorbed: termination = "hit_window"

    # ---------- remaining transitions ----------
    for _hop in range(max_hops):
        if termination != "max_hops":
            break

        if window is None:
            dist_to_center = np.max(np.abs(current_p - system_center))
            if dist_to_center > math_kill_radius:
                hit_target = "Infinity (Ground)"
                termination = "escaped"
                break

        nearest_idx, min_dist, wall_is_nearest = walk_radius(
            current_p,
            conductors,
            window,
            neumann_axes,
        )
        side_a = 2 * min_dist
        hop_count += 1

        if current_walk_path is not None:
            current_walk_path.append({ "center": current_p.tolist(), "a": float(side_a), "P(r,r')": float(probability), } )

        if min_dist < 1e-4:
            if wall_is_nearest:
                termination = "hit_window"
            else:
                hit_target = conductors[nearest_idx].name
                termination = "hit_conductor"
            break

        current_p, _index, probability = gft.sample_next_point(
            current_point=current_p,
            cube_half_size=min_dist,
            rng=rng,
        )

        if window is not None:
            current_p, reflected, absorbed = reflect_point(current_p, window, neumann_axes, )
            reflection_count += reflected
            if absorbed:
                termination = "hit_window"
                break

    return WalkOutcome(
        start_point=start_point.tolist(),
        start_normal=surface_normal.tolist(),
        first_panel_index=int(panel_index),
        first_transition_mode=first_transition_mode,
        first_transition_stratum=first_transition_stratum,
        omega=float(omega),
        hit_conductor=hit_target,
        termination=termination,
        hop_count=hop_count,
        path_data=current_walk_path,
        reflection_count=reflection_count,
        wall_potential=( float(wall_potential) if termination == "hit_window" else None ),
    )


def run_frw_geometry(
    conductors,
    master_vgss,
    rng,
    num_walks=1000,
    max_hops=200,
    first_transition_mode=STANDARD_FIRST_TRANSITION,
    window: Bounds | None = None,
    neumann_axes=(),
    wall_potential=0.0,
):
    """Compatibility batch API that stores every walk and its full path.

    Capacitance is intentionally not accumulated here.  The first-hop weight
    ``omega`` and the terminal condition are stored for post-processing by
    ``FRW.capacitance``.
    """
    _validate_run_configuration(num_walks, first_transition_mode)
    validate_window(conductors, window, neumann_axes)

    sys_center, sys_size = get_system_bounds(conductors)
    math_kill_radius = sys_size * 500_000.0

    all_walk_data = []
    sampling_stats = VGSSSamplingStats()

    gft = CubeGFT.load_default()

    for walk in range(num_walks):
        percent_done = (walk + 1) / num_walks * 100.0
        print(f"\rProgress: {percent_done:.1f}% ({walk + 1}/{num_walks})", end="", flush=True,)

        outcome = run_one_walk(
            conductors=conductors,
            master_vgss=master_vgss,
            gft=gft,
            rng=rng,
            walk_index=walk,
            sampling_stats=sampling_stats,
            system_center=sys_center,
            math_kill_radius=math_kill_radius,
            max_hops=max_hops,
            first_transition_mode=first_transition_mode,
            record_path=True,
            window=window,
            neumann_axes=neumann_axes,
            wall_potential=wall_potential,
        )
        all_walk_data.append(outcome.as_dict())

    print()

    context = master_vgss.sampling_context
    h_estimate = sampling_stats.estimate_H(context)

    result = {
        "master_net": master_vgss.net.name,
        "Sg": sampling_stats.estimate_surface_area(context),
        "mean_p": sampling_stats.estimate_mean_p(),
        "H": h_estimate,
        "sampling_stats": sampling_stats.as_dict(context),
        "first_transition_mode": first_transition_mode,
        "walks": all_walk_data,
    }
    if window is not None:
        result["window"] = {
            "xmin": window.xmin,
            "xmax": window.xmax,
            "ymin": window.ymin,
            "ymax": window.ymax,
            "zmin": window.zmin,
            "zmax": window.zmax,
        }
        result["neumann_axes"] = list(neumann_axes)
        result["wall_potential"] = float(wall_potential)
    return result


if __name__ == "__main__":
    print("Run FRW via CLI module: python -m FRW.cli <input_file.m>")
