from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from bgs.topology import find_touching_nets
from GGFT.ggft import STANDARD_FIRST_TRANSITION, STRATIFIED_FIRST_TRANSITION


E0 = 8.8541878128e-12
L_UNIT = 1.0e0
INFINITY_LABEL = "Infinity"


def _outcome_value(outcome, name, default=None):
    if isinstance(outcome, Mapping):
        return outcome.get(name, default)
    return getattr(outcome, name, default)


class CapacitanceRowAccumulator:
    """Online row estimator with pair-level statistics for stratification."""

    def __init__(
        self,
        conductors,
        master_vgss,
        first_transition_mode=STANDARD_FIRST_TRANSITION,
        finite_window=False,
    ):
        if first_transition_mode not in {
            STANDARD_FIRST_TRANSITION,
            STRATIFIED_FIRST_TRANSITION,
        }:
            raise ValueError(f"Unknown first-transition mode: {first_transition_mode}")

        nets = find_touching_nets(conductors)
        self.net_order = [net.name for net in nets]
        self.master_net = master_vgss.net.name
        self.first_transition_mode = first_transition_mode
        self.finite_window = finite_window
        self.conductor_to_net = {
            conductor.name: net.name
            for net in nets
            for conductor in net.conductors
        }
        self.net_index = {
            net_name: index
            for index, net_name in enumerate(self.net_order)
        }

        # The final two components represent infinity and a Dirichlet window.
        size = len(self.net_order) + 2
        self.infinity_index = len(self.net_order)
        self.window_index = len(self.net_order) + 1
        self.walk_count = 0
        self.sample_count = 0
        self.mean = np.zeros(size, dtype=float)
        self.M2 = np.zeros(size, dtype=float)
        self.pending_inward = None

        self.raw_weight_sums = np.zeros(len(self.net_order), dtype=float)
        self.infinity_weight_sum = 0.0
        self.window_weight_sum = 0.0
        self.window_potential_sum = 0.0
        self.net_hit_counts = np.zeros(len(self.net_order), dtype=int)
        self.infinity_hits = 0
        self.window_hits = 0
        self.unresolved_walks = 0

    def update(self, outcome) -> bool:
        """Consume one outcome and report whether one sample was completed."""

        self.walk_count += 1
        omega = float(_outcome_value(outcome, "omega"))
        termination = _outcome_value(outcome, "termination")
        hit_conductor = _outcome_value(outcome, "hit_conductor")

        contribution = np.zeros(len(self.net_order) + 2, dtype=float)
        if termination == "hit_conductor":
            target_net = self.conductor_to_net[hit_conductor]
            target_index = self.net_index[target_net]
            contribution[target_index] = omega
            self.raw_weight_sums[target_index] += omega
            self.net_hit_counts[target_index] += 1
        elif termination == "escaped":
            contribution[self.infinity_index] = omega
            self.infinity_weight_sum += omega
            self.infinity_hits += 1
        elif termination == "hit_window":
            self.finite_window = True
            contribution[self.window_index] = omega
            self.window_weight_sum += omega
            self.window_potential_sum += omega * float(
                _outcome_value(outcome, "wall_potential", 0.0)
            )
            self.window_hits += 1
        elif termination == "max_hops":
            self.unresolved_walks += 1
        else:
            raise ValueError(f"Unknown walk termination: {termination!r}")

        if self.first_transition_mode == STANDARD_FIRST_TRANSITION:
            self._update_statistics(contribution)
            return True

        stratum = _outcome_value(outcome, "first_transition_stratum")
        if stratum == "inward":
            if self.pending_inward is not None:
                raise ValueError("Received two inward outcomes without an outward pair.")
            self.pending_inward = contribution
            return False
        if stratum != "outward":
            raise ValueError(
                "Stratified outcomes must identify the inward or outward stratum."
            )
        if self.pending_inward is None:
            raise ValueError("Received an outward outcome without an inward pair.")

        pair_contribution = (self.pending_inward + contribution) / 2.0
        self.pending_inward = None
        self._update_statistics(pair_contribution)
        return True

    def _update_statistics(self, observation):
        """Update the vector mean and M2 using Welford's algorithm."""

        self.sample_count += 1
        delta = observation - self.mean
        self.mean += delta / self.sample_count
        self.M2 += delta * (observation - self.mean)

    def _weight_standard_error(self):
        if self.sample_count < 2:
            return None
        variance = np.maximum(self.M2 / (self.sample_count - 1), 0.0)
        return np.sqrt(variance / self.sample_count)

    def statistics(self) -> dict:
        """Return the current H-independent statistics for convergence checks."""

        standard_error = self._weight_standard_error()
        result = {
            "walk_count": self.walk_count,
            "statistical_sample_count": self.sample_count,
            "net_order": list(self.net_order),
            "mean_weight_contribution": {
                name: float(self.mean[index])
                for index, name in enumerate(self.net_order)
            },
            "weight_standard_error": {
                name: (
                    None
                    if standard_error is None
                    else float(standard_error[index])
                )
                for index, name in enumerate(self.net_order)
            },
            "net_hit_counts": {
                name: int(self.net_hit_counts[index])
                for index, name in enumerate(self.net_order)
            },
            "pair_complete": self.pending_inward is None,
        }
        return result

    def finalize(self, H: float) -> dict:
        """Convert the running estimator into a capacitance-row result."""

        if self.walk_count == 0:
            raise ValueError("At least one walk is required to estimate capacitance.")
        if self.pending_inward is not None:
            raise ValueError("Cannot finalize an incomplete inward/outward pair.")
        if not np.isfinite(H) or H < 0.0:
            raise ValueError("H must be finite and non-negative.")

        net_count = len(self.net_order)
        mean_finite = self.mean[:net_count]
        geometric_values = H * mean_finite
        physical_values = E0 * L_UNIT * geometric_values
        geometric_infinity = H * self.mean[self.infinity_index]
        infinity_capacitance = E0 * L_UNIT * geometric_infinity
        geometric_window = H * self.mean[self.window_index]
        window_capacitance = E0 * L_UNIT * geometric_window
        window_potential_contribution = (
            E0 * L_UNIT * H * self.window_potential_sum / self.walk_count
        )

        weight_standard_error = self._weight_standard_error()
        if weight_standard_error is not None:
            geometric_standard_error = H * weight_standard_error
            physical_standard_error = E0 * L_UNIT * geometric_standard_error
        else:
            geometric_standard_error = None
            physical_standard_error = None

        geometric_row = {
            name: float(geometric_values[index])
            for index, name in enumerate(self.net_order)
        }
        capacitance_row = {
            name: float(physical_values[index])
            for index, name in enumerate(self.net_order)
        }
        raw_weight_sums = {
            name: float(self.raw_weight_sums[index])
            for index, name in enumerate(self.net_order)
        }
        net_hit_counts = {
            name: int(self.net_hit_counts[index])
            for index, name in enumerate(self.net_order)
        }

        if weight_standard_error is None:
            weight_se_row = {name: None for name in self.net_order}
            geometric_se_row = {name: None for name in self.net_order}
            physical_se_row = {name: None for name in self.net_order}
            infinity_weight_se = None
            infinity_geometric_se = None
            infinity_physical_se = None
            window_weight_se = None
            window_geometric_se = None
            window_physical_se = None
        else:
            weight_se_row = {
                name: float(weight_standard_error[index])
                for index, name in enumerate(self.net_order)
            }
            geometric_se_row = {
                name: float(geometric_standard_error[index])
                for index, name in enumerate(self.net_order)
            }
            physical_se_row = {
                name: float(physical_standard_error[index])
                for index, name in enumerate(self.net_order)
            }
            infinity_weight_se = float(
                weight_standard_error[self.infinity_index]
            )
            infinity_geometric_se = float(
                geometric_standard_error[self.infinity_index]
            )
            infinity_physical_se = float(
                physical_standard_error[self.infinity_index]
            )
            window_weight_se = float(
                weight_standard_error[self.window_index]
            )
            window_geometric_se = float(
                geometric_standard_error[self.window_index]
            )
            window_physical_se = float(
                physical_standard_error[self.window_index]
            )

        unresolved_rate = self.unresolved_walks / self.walk_count
        row_sum_finite = float(sum(capacitance_row.values()))
        row_sum_with_infinity = row_sum_finite + infinity_capacitance
        row_sum_with_boundary = row_sum_with_infinity + window_capacitance

        result = {
            "master_net": self.master_net,
            "net_order": self.net_order,
            "num_walks": self.walk_count,
            "statistical_sample_count": self.sample_count,
            "first_transition_mode": self.first_transition_mode,
            "H": float(H),
            "E0": E0,
            "L_UNIT": L_UNIT,
            "raw_weight_sums": raw_weight_sums,
            "infinity_weight_sum": float(self.infinity_weight_sum),
            "net_hit_counts": net_hit_counts,
            "infinity_hits": self.infinity_hits,
            "mean_weight_contribution": {
                name: float(mean_finite[index])
                for index, name in enumerate(self.net_order)
            },
            "weight_standard_error": weight_se_row,
            "geometric_row": geometric_row,
            "geometric_standard_error": geometric_se_row,
            "capacitance_row_F": capacitance_row,
            "standard_error_F": physical_se_row,
            "geometric_infinity": float(geometric_infinity),
            "infinity_capacitance_F": float(infinity_capacitance),
            "infinity_weight_standard_error": infinity_weight_se,
            "infinity_geometric_standard_error": infinity_geometric_se,
            "infinity_standard_error_F": infinity_physical_se,
            "unresolved_walks": self.unresolved_walks,
            "unresolved_rate": float(unresolved_rate),
            "is_complete": self.unresolved_walks == 0,
            "row_sum_finite_F": row_sum_finite,
            "row_sum_with_infinity_F": float(row_sum_with_infinity),
        }
        if self.finite_window:
            result.update(
                {
                    "window_weight_sum": float(self.window_weight_sum),
                    "window_hits": self.window_hits,
                    "geometric_window": float(geometric_window),
                    "window_capacitance_F": float(window_capacitance),
                    "window_potential_contribution_F": float(
                        window_potential_contribution
                    ),
                    "window_weight_standard_error": window_weight_se,
                    "window_geometric_standard_error": window_geometric_se,
                    "window_standard_error_F": window_physical_se,
                    "row_sum_with_boundary_F": float(row_sum_with_boundary),
                }
            )
        return result


def calculate_capacitance_row(
    walks: Sequence[Mapping],
    H: float,
    conductors,
    master_vgss,
) -> dict:
    """Convert a stored batch through the same online row accumulator."""

    if not walks:
        raise ValueError("At least one walk is required to estimate capacitance.")

    first_transition_mode = _outcome_value(
        walks[0],
        "first_transition_mode",
        STANDARD_FIRST_TRANSITION,
    )
    accumulator = CapacitanceRowAccumulator(
        conductors,
        master_vgss,
        first_transition_mode=first_transition_mode,
        finite_window=any(
            _outcome_value(walk, "termination") == "hit_window"
            or _outcome_value(walk, "reflection_count", 0) > 0
            for walk in walks
        ),
    )
    for walk in walks:
        accumulator.update(walk)
    return accumulator.finalize(H)


def assemble_capacitance_matrix(
    row_results: Sequence[Mapping],
    net_order: Sequence[str] | None = None,
) -> dict:
    """Assemble independently estimated rows into a full net capacitance matrix."""

    if not row_results:
        raise ValueError("At least one capacitance row is required.")

    if net_order is None:
        net_order = list(row_results[0]["net_order"])
    else:
        net_order = list(net_order)

    row_by_master = {row["master_net"]: row for row in row_results}

    matrix = np.zeros((len(net_order), len(net_order)), dtype=float)
    infinity_column = np.zeros(len(net_order), dtype=float)
    window_column = np.zeros(len(net_order), dtype=float)

    for row_index, master_net in enumerate(net_order):
        row_result = row_by_master[master_net]
        row_values = row_result["capacitance_row_F"]

        for column_index, target_net in enumerate(net_order):
            matrix[row_index, column_index] = float(row_values[target_net])

        infinity_column[row_index] = float(
            row_result["infinity_capacitance_F"]
        )
        window_column[row_index] = float(
            row_result.get("window_capacitance_F", 0.0)
        )

    difference = matrix - matrix.T
    max_abs_entry = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    max_abs_asymmetry = float(np.max(np.abs(difference))) if matrix.size else 0.0
    relative_asymmetry = (
        max_abs_asymmetry / max_abs_entry
        if max_abs_entry > 0.0
        else 0.0
    )
    convergence_records = [row.get("convergence") for row in row_results]
    all_rows_converged = (
        all(bool(record["converged"]) for record in convergence_records)
        if all(
            record is not None and bool(record.get("enabled"))
            for record in convergence_records
        )
        else None
    )

    result = {
        "net_order": net_order,
        "matrix_F": matrix.tolist(),
        "infinity_column_F": infinity_column.tolist(),
        "diagonal_F": np.diag(matrix).tolist(),
        "max_abs_asymmetry_F": max_abs_asymmetry,
        "relative_asymmetry": float(relative_asymmetry),
        "all_rows_complete": all(bool(row["is_complete"]) for row in row_results),
        "all_rows_converged": all_rows_converged,
    }
    if any("window_capacitance_F" in row for row in row_results):
        result["window_column_F"] = window_column.tolist()
    return result


def print_capacitance_row(row_result: Mapping) -> None:
    master_net = row_result["master_net"]

    print("\n" + "-" * 30)
    print(f"Capacitance row for {master_net}:")
    for target_net in row_result["net_order"]:
        value = row_result["capacitance_row_F"][target_net]
        print(f"  C[{master_net}, {target_net}] = {value:.6e} F")

    print(
        "  C["
        f"{master_net}, {INFINITY_LABEL}] = "
        f"{row_result['infinity_capacitance_F']:.6e} F"
    )
    if row_result.get("window_hits", 0):
        print(
            "  C["
            f"{master_net}, Window] = "
            f"{row_result['window_capacitance_F']:.6e} F"
        )
    print(
        "  finite row sum = "
        f"{row_result['row_sum_finite_F']:.6e} F"
    )
    print(
        "  row sum including infinity = "
        f"{row_result['row_sum_with_infinity_F']:.6e} F"
    )
    if row_result.get("window_hits", 0):
        print(
            "  row sum including window = "
            f"{row_result['row_sum_with_boundary_F']:.6e} F"
        )
    print(
        "  unresolved walks = "
        f"{row_result['unresolved_walks']} "
        f"({row_result['unresolved_rate'] * 100.0:.3f}%)"
    )
    if not row_result["is_complete"]:
        print("  WARNING: this row contains unresolved max-hop walks.")


def print_capacitance_matrix(matrix_result: Mapping) -> None:
    matrix = np.asarray(matrix_result["matrix_F"], dtype=float)
    net_order = matrix_result["net_order"]

    print("\nCapacitance matrix (F):")
    print("Net order: " + ", ".join(net_order))
    print(np.array2string(matrix, precision=4, suppress_small=False))

    print(
        "Relative symmetry error: "
        f"{matrix_result['relative_asymmetry']:.3e}"
    )
