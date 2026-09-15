import numpy as np

from bgs.sampling import VGSSSamplingStats, sample_on_vgss
from GGFT.ggft import CubeGFT


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


def run_frw_geometry(
    conductors,
    master_vgss,
    rng,
    num_walks=1000,
    max_hops=200,
):
    """Run FRW paths and return raw walk data.

    Capacitance is intentionally not accumulated here.  The first-hop weight
    ``omega`` and the terminal condition are stored for post-processing by
    ``FRW.capacitance``.
    """
    sys_center, sys_size = get_system_bounds(conductors)
    math_kill_radius = sys_size * 500_000.0

    all_walk_data = []
    sampling_stats = VGSSSamplingStats()

    gft = CubeGFT.load_default()

    for walk in range(num_walks):
        percent_done = (walk + 1) / num_walks * 100.0
        print(f"\rProgress: {percent_done:.1f}% ({walk + 1}/{num_walks})", end="", flush=True,)

        start_sample = sample_on_vgss(
            master_vgss.sampling_context,
            rng,
            stats=sampling_stats,
        )
        current_p = start_sample.point
        surface_normal = start_sample.normal
        start_point = current_p.copy()
        current_walk_path = []

        hit_target = None
        termination = "max_hops"

        # ---------- first transition ----------
        distances = [c.distance_to_point(current_p) for c in conductors]
        min_dist = min(distances)
        side_a = 2 * min_dist

        next_point, panel_index, probability, omega = gft.sample_first_transition(
            start_point,
            min_dist,
            surface_normal,
            rng,
        )

        current_walk_path.append({
                "center": current_p.tolist(),
                "a": float(side_a),
                "P(r,r')": float(probability), }
        )
        current_p = next_point

        # ---------- remaining transitions ----------
        for _hop in range(max_hops):
            dist_to_center = np.max(np.abs(current_p - sys_center))
            if dist_to_center > math_kill_radius:
                hit_target = "Infinity (Ground)"
                termination = "escaped"
                break

            distances = [c.distance_to_point(current_p) for c in conductors]
            min_dist = min(distances)
            nearest_idx = int(np.argmin(distances))
            side_a = 2 * min_dist

            current_walk_path.append(
                {
                    "center": current_p.tolist(),
                    "a": float(side_a),
                    "P(r,r')": float(probability),
                }
            )

            if min_dist < 1e-4:
                hit_target = conductors[nearest_idx].name
                termination = "hit_conductor"
                break

            current_p, _index, probability = gft.sample_next_point(current_point=current_p, cube_half_size=min_dist, rng=rng, )

        all_walk_data.append(
            {
                "start_point": start_point.tolist(),
                "start_normal": surface_normal.tolist(),
                "first_panel_index": int(panel_index),
                "omega": float(omega),
                "path_data": current_walk_path,
                "hit_conductor": hit_target,
                "termination": termination,
            }
        )

    print()

    context = master_vgss.sampling_context
    h_estimate = sampling_stats.estimate_H(context)

    return {
        "master_net": master_vgss.net.name,
        "Sg": sampling_stats.estimate_surface_area(context),
        "mean_p": sampling_stats.estimate_mean_p(),
        "H": h_estimate,
        "sampling_stats": sampling_stats.as_dict(context),
        "walks": all_walk_data,
    }


if __name__ == "__main__":
    print("Run FRW via CLI module: python -m FRW.cli <input_file.m>")
