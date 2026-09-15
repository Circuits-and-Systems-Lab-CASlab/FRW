from __future__ import annotations

import numpy as np
import pytest

from FRW.random_walk import run_frw_geometry
from GGFT.ggft import CubeGFT


@pytest.fixture(scope="module")
def production_table() -> CubeGFT:
    return CubeGFT.load_default()


@pytest.mark.parametrize(
    "normal",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ],
)
def test_stratified_expectation_matches_standard_first_transition(
    production_table: CubeGFT,
    normal: np.ndarray,
) -> None:
    table = production_table
    signed_derivative = -table.probabilities * (table.wvt @ normal)
    boundary_values = (
        1.0
        + 0.3 * table.panel_centers[:, 0]
        - 0.2 * table.panel_centers[:, 1]
        + 0.1 * table.panel_centers[:, 2] ** 2
    )
    cube_half_size = 1.7

    standard_expectation = float(
        signed_derivative @ boundary_values / cube_half_size
    )

    stratified_expectation = 0.0
    for stratum_name in ("inward", "outward"):
        stratum, _unit_normal = table._get_first_transition_stratum(
            normal,
            stratum_name,
        )
        indices = stratum.panel_indices
        derivatives = signed_derivative[indices]
        conditional_probability = np.abs(derivatives) / stratum.absolute_mass
        weights = (
            np.sign(derivatives)
            * stratum.absolute_mass
            / 0.5
            / cube_half_size
        )
        stratified_expectation += 0.5 * float(
            conditional_probability
            @ (weights * boundary_values[indices])
        )

    assert stratified_expectation == pytest.approx(
        standard_expectation,
        rel=2.0e-13,
        abs=2.0e-13,
    )


def test_existing_first_transition_call_remains_standard(
    production_table: CubeGFT,
) -> None:
    table = production_table
    arguments = (
        np.array([2.0, 3.0, 4.0]),
        1.25,
        np.array([1.0, 0.0, 0.0]),
    )

    default_result = table.sample_first_transition(
        *arguments,
        np.random.default_rng(1234),
    )
    explicit_result = table.sample_first_transition(
        *arguments,
        np.random.default_rng(1234),
        mode="standard",
    )

    np.testing.assert_allclose(default_result[0], explicit_result[0])
    assert default_result[1:] == explicit_result[1:]


def _mixed_sign_table() -> CubeGFT:
    probabilities = np.full(6, 1.0 / 6.0)
    panel_centers = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # For n=+x, d_j=-P_j*WVT_x has both signs inside x<=0.
    wvt = np.zeros((6, 3))
    wvt[:, 0] = [-6.0, 6.0, 3.0, -3.0, 1.5, -1.5]
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    return CubeGFT(
        probabilities=probabilities,
        cdf=cdf,
        panel_centers=panel_centers,
        wvt=wvt,
        resolution=1,
    )


def test_stratified_weight_preserves_mixed_derivative_signs() -> None:
    table = _mixed_sign_table()
    normal = np.array([1.0, 0.0, 0.0])
    stratum, _unit_normal = table._get_first_transition_stratum(
        normal,
        "inward",
    )
    expected_magnitude = stratum.absolute_mass / 0.5 / 2.0
    observed_signs = set()
    rng = np.random.default_rng(9)

    for _ in range(200):
        _point, panel_index, probability, omega = table.sample_first_transition(
            np.zeros(3),
            2.0,
            normal,
            rng,
            mode="stratified",
            stratum="inward",
            allocation_probability=0.5,
        )
        derivative = -table.probabilities[panel_index] * table.wvt[panel_index, 0]
        assert probability > 0.0
        assert np.sign(omega) == np.sign(derivative)
        assert abs(omega) == pytest.approx(expected_magnitude)
        observed_signs.add(int(np.sign(omega)))

    assert observed_signs == {-1, 1}


def test_stratified_transition_rejects_non_axis_aligned_normal(
    production_table: CubeGFT,
) -> None:
    with pytest.raises(ValueError, match="axis-aligned"):
        production_table.sample_first_transition(
            np.zeros(3),
            1.0,
            np.array([1.0, 1.0, 0.0]),
            np.random.default_rng(1),
            mode="stratified",
            stratum="inward",
            allocation_probability=0.5,
        )


def test_stratified_geometry_requires_an_even_walk_count() -> None:
    with pytest.raises(ValueError, match="even number"):
        run_frw_geometry(
            [],
            None,
            np.random.default_rng(1),
            num_walks=3,
            first_transition_mode="stratified",
        )
