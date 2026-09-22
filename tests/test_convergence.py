from __future__ import annotations

import numpy as np
import pytest

from FRW.cli import _build_parser, _row_rng, _worker_count, main
from FRW.convergence import check_row_convergence


def _statistics(*, hits=(20, 20), row_scale=10.0):
    return {
        "net_order": ["Net1", "Net2"],
        "mean_weight_contribution": {
            "Net1": row_scale,
            "Net2": -2.0,
        },
        "weight_standard_error": {
            "Net1": 0.05,
            "Net2": 0.02,
        },
        "net_hit_counts": {
            "Net1": hits[0],
            "Net2": hits[1],
        },
    }


def test_row_max_three_se_converges() -> None:
    result = check_row_convergence(
        _statistics(),
        relative_tolerance=0.03,
        minimum_target_hits=10,
    )

    assert result["converged"] is True
    assert result["final_ratio"] == pytest.approx(0.015)
    assert result["uncertain_entries"] == []


def test_convergence_requires_minimum_hits_for_every_entry() -> None:
    result = check_row_convergence(
        _statistics(hits=(20, 3)),
        relative_tolerance=0.03,
        minimum_target_hits=10,
    )

    assert result["converged"] is False
    assert result["insufficient_hit_entries"] == ["Net2"]


def test_zero_row_does_not_report_false_convergence() -> None:
    statistics = _statistics(row_scale=0.0)
    statistics["mean_weight_contribution"]["Net2"] = 0.0

    result = check_row_convergence(
        statistics,
        relative_tolerance=0.03,
        minimum_target_hits=0,
    )

    assert result["converged"] is False
    assert result["final_ratio"] is None


def test_cli_defaults_to_adaptive_convergence() -> None:
    args = _build_parser().parse_args(["input.m"])

    assert args.walks is None
    assert args.max_walks == 1000000
    assert args.min_walks == 2000
    assert args.check_every == 200
    assert args.tolerance == 0.03
    assert args.min_hits == 10
    assert args.workers == 4


def test_explicit_walk_count_selects_fixed_mode() -> None:
    args = _build_parser().parse_args(["input.m", "--walks", "5000"])

    assert args.walks == 5000


def test_adaptive_counts_must_preserve_complete_pairs() -> None:
    with pytest.raises(SystemExit):
        main(["input.m", "--check-every", "101"])


def test_workers_are_limited_by_the_number_of_rows() -> None:
    assert _worker_count(4, 5) == 4
    assert _worker_count(4, 2) == 2
    assert _worker_count(4, 1) == 1


def test_worker_count_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        main(["input.m", "--workers", "0"])


def test_each_row_has_an_independent_reproducible_rng_stream() -> None:
    first = _row_rng(1234, 2).random(5)
    repeated = _row_rng(1234, 2).random(5)
    different_row = _row_rng(1234, 3).random(5)

    np.testing.assert_allclose(first, repeated)
    assert not np.allclose(first, different_row)
