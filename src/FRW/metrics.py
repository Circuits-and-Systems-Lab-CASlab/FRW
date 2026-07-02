from collections import Counter

import numpy as np

from bgs.topology import find_touching_nets


def calculate_frw_metrics( walks, conductors, master_vgss, ):
    nets = find_touching_nets(conductors)

    conductor_to_net = {
        conductor.name: net.name
        for net in nets
        for conductor in net.conductors
    }

    conductor_hit_counts = Counter()
    net_hit_counts = Counter()

    termination_counts = Counter(
        {
        "hit_conductor": 0,
        "escaped": 0,
        "max_hops": 0, 
        }
    )

    hop_counts = []

    for walk in walks:
        hit = walk.get("hit_conductor")
        hop_counts.append(len(walk.get("path_data", [])))

        if hit == "Infinity (Ground)":
            termination_counts["escaped"] += 1
            continue

        if hit is None:
            termination_counts["max_hops"] += 1
            continue

        termination_counts["hit_conductor"] += 1
        conductor_hit_counts[hit] += 1

        hit_net = conductor_to_net[hit]
        net_hit_counts[hit_net] += 1

    master_net_name = master_vgss.net.name

    self_net_hits = net_hit_counts.get(
        master_net_name,
        0,
    )

    other_net_hits = (
        termination_counts["hit_conductor"]
        - self_net_hits
    )

    num_walks = len(walks)

    termination_rates = {
        name: count / num_walks if num_walks else 0.0
        for name, count in termination_counts.items()
    }

    hop_statistics = _calculate_hop_statistics( hop_counts )

    return {
        "conductor_hit_counts": dict( conductor_hit_counts ),
        "net_hit_counts": dict(net_hit_counts),
        "self_net_hits": self_net_hits,
        "other_net_hits": other_net_hits,
        "termination_counts": dict( termination_counts ),
        "termination_rates": termination_rates,
        "hop_statistics": hop_statistics,
    }


def _calculate_hop_statistics(hop_counts):
    if not hop_counts:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "std": 0.0,
            "p95": 0.0,
        }

    values = np.asarray(hop_counts, dtype=float)

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
    }


def print_metrics(metrics: dict) -> None:
    print("\n" + "-" * 30)
    print("FRW Metrics:\n")

    for key in (
        "termination_counts",
        "termination_rates",
        "conductor_hit_counts",
        "net_hit_counts",
        "self_net_hits",
        "other_net_hits",
        "hop_statistics",
    ):
        if key not in metrics:
            continue

        print(f"{key}:")
        value = metrics[key]
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if key == "termination_rates":
                    print(f"  {sub_key}: {sub_value * 100:.2f}%")
                else:
                    print(f"  {sub_key}: {sub_value}")
        else:
            print(f"  {value}")
        print()
