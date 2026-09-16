from __future__ import annotations

import math


THREE_SIGMA = 3.0


def check_row_convergence(
    statistics: dict,
    relative_tolerance: float,
    minimum_target_hits: int = 10,
) -> dict:
    """Evaluate the row-max 3SE convergence criterion.

    H, epsilon_0, and the length scale cancel from this relative test, so the
    check uses the accumulator's raw weight mean and standard error.
    """

    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    if minimum_target_hits < 0:
        raise ValueError("minimum_target_hits must be non-negative")

    net_order = statistics["net_order"]
    mean = statistics["mean_weight_contribution"]
    standard_error = statistics["weight_standard_error"]
    hit_counts = statistics["net_hit_counts"]

    row_scale = max((abs(float(mean[name])) for name in net_order), default=0.0)
    insufficient_hits = [
        name
        for name in net_order
        if int(hit_counts[name]) < minimum_target_hits
    ]

    per_entry_three_se = {}
    uncertain_entries = []
    errors_available = True
    for name in net_order:
        entry_standard_error = standard_error[name]
        if entry_standard_error is None:
            per_entry_three_se[name] = None
            errors_available = False
            continue

        three_se = THREE_SIGMA * float(entry_standard_error)
        per_entry_three_se[name] = three_se
        if row_scale <= 0.0 or three_se > relative_tolerance * row_scale:
            uncertain_entries.append(name)

    finite_errors = [
        value for value in per_entry_three_se.values() if value is not None
    ]
    max_three_se = max(finite_errors, default=None)
    ratio = (
        max_three_se / row_scale
        if max_three_se is not None and row_scale > 0.0
        else None
    )
    converged = (
        errors_available
        and row_scale > 0.0
        and not insufficient_hits
        and not uncertain_entries
    )

    return {
        "converged": converged,
        "criterion": "row-max-3se",
        "relative_tolerance": float(relative_tolerance),
        "minimum_target_hits": int(minimum_target_hits),
        "row_scale": float(row_scale),
        "max_three_se": max_three_se,
        "final_ratio": ratio,
        "per_entry_three_se": per_entry_three_se,
        "target_hit_counts": dict(hit_counts),
        "insufficient_hit_entries": insufficient_hits,
        "uncertain_entries": uncertain_entries,
    }
