from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import numpy as np

from bgs.topology import find_touching_nets


def _outcome_value(outcome, name, default=None):
    if isinstance(outcome, Mapping):
        return outcome.get(name, default)
    return getattr(outcome, name, default)


class MetricsAccumulator:
    """Accumulate FRW diagnostic metrics without retaining complete walks."""

    def __init__(self, conductors, master_vgss):
        nets = find_touching_nets(conductors)
        self.conductor_to_net = {
            conductor.name: net.name
            for net in nets
            for conductor in net.conductors
        }
        self.master_net_name = master_vgss.net.name
        self.conductor_hit_counts = Counter()
        self.net_hit_counts = Counter()
        self.termination_counts = Counter(
            {
                "hit_conductor": 0,
                "escaped": 0,
                "max_hops": 0,
            }
        )
        # Exact median and p95 require only one integer per walk.
        self.hop_counts = []
        self.reflection_count = 0

    def update(self, outcome) -> None:
        termination = _outcome_value(outcome, "termination")
        hit = _outcome_value(outcome, "hit_conductor")
        hop_count = _outcome_value(outcome, "hop_count")
        if hop_count is None:
            hop_count = len(_outcome_value(outcome, "path_data", []))

        if termination == "hit_window":
            self.termination_counts.setdefault("hit_window", 0)
        elif termination not in self.termination_counts:
            raise ValueError(f"Unknown walk termination: {termination!r}")
        self.termination_counts[termination] += 1
        self.hop_counts.append(int(hop_count))
        self.reflection_count += int(
            _outcome_value(outcome, "reflection_count", 0)
        )

        if termination != "hit_conductor":
            return

        self.conductor_hit_counts[hit] += 1
        hit_net = self.conductor_to_net[hit]
        self.net_hit_counts[hit_net] += 1

    def finalize(self) -> dict:
        num_walks = len(self.hop_counts)
        self_net_hits = self.net_hit_counts.get(self.master_net_name, 0)
        other_net_hits = (
            self.termination_counts["hit_conductor"] - self_net_hits
        )
        termination_rates = {
            name: count / num_walks if num_walks else 0.0
            for name, count in self.termination_counts.items()
        }

        result = {
            "conductor_hit_counts": dict(self.conductor_hit_counts),
            "net_hit_counts": dict(self.net_hit_counts),
            "self_net_hits": self_net_hits,
            "other_net_hits": other_net_hits,
            "termination_counts": dict(self.termination_counts),
            "termination_rates": termination_rates,
            "hop_statistics": _calculate_hop_statistics(self.hop_counts),
        }
        if "hit_window" in self.termination_counts or self.reflection_count:
            result["reflection_count"] = self.reflection_count
        return result


def calculate_frw_metrics(walks, conductors, master_vgss):
    """Compatibility batch API using the online metrics accumulator."""

    accumulator = MetricsAccumulator(conductors, master_vgss)
    for walk in walks:
        accumulator.update(walk)
    return accumulator.finalize()


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
        "reflection_count",
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
