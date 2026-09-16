from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from FRW.capacitance import CapacitanceRowAccumulator, calculate_capacitance_row
from FRW.random_walk import get_system_bounds, run_frw_geometry, run_one_walk
from GGFT.ggft import STRATIFIED_FIRST_TRANSITION, CubeGFT
from bgs.matlab_parser import parse_matlab_geometry
from bgs.models import Bounds, GaussianSurface, make_surface_faces
from bgs.sampling import VGSSSamplingStats, prepare_vgss_sampling, sample_on_vgss
from bgs.vgss import build_vgss_for_conductor


def _surface(name: str, bounds: Bounds) -> GaussianSurface:
    return GaussianSurface(name, bounds, make_surface_faces(bounds))


def test_sampling_stats_return_area_when_p_is_one() -> None:
    surface = _surface("C1", Bounds(0, 1, 0, 2, 0, 3))
    context = prepare_vgss_sampling([surface])
    stats = VGSSSamplingStats()
    rng = np.random.default_rng(11)

    for _ in range(25):
        sample_on_vgss(context, rng, stats=stats).point

    assert stats.proposal_count == 25
    assert stats.accepted_count == 25
    assert stats.estimate_Sg(context) == pytest.approx(surface.area)


def test_sampling_stats_estimate_weighted_H() -> None:
    surface = _surface(
        "C1",
        Bounds(0, 1, 0, 1, 0, 1),
    )

    context = prepare_vgss_sampling([surface], importance_function=lambda _point: 0.5, upper_bound=1.0, )

    stats = VGSSSamplingStats()
    rng = np.random.default_rng(2026)

    for _ in range(4000):
        sample_on_vgss(context,rng,stats=stats,)

    assert stats.estimate_Sg(context) == pytest.approx(6.0)
    assert stats.estimate_mean_p() == pytest.approx(0.5)
    assert stats.estimate_H(context) == pytest.approx(3.0)

    assert stats.importance_rejections > 0


def test_one_walk_can_skip_the_full_visualization_trace() -> None:
    project_root = Path(__file__).resolve().parents[1]
    conductors = parse_matlab_geometry(project_root / "input" / "INPUT_EXAMPLE.m")
    master_vgss = build_vgss_for_conductor(conductors, "C1")
    system_center, system_size = get_system_bounds(conductors)

    outcome = run_one_walk(
        conductors=conductors,
        master_vgss=master_vgss,
        gft=CubeGFT.load_default(),
        rng=np.random.default_rng(5),
        walk_index=0,
        sampling_stats=VGSSSamplingStats(),
        system_center=system_center,
        math_kill_radius=system_size * 500_000.0,
        record_path=False,
    )

    assert outcome.path_data is None
    assert "path_data" not in outcome.as_dict()
    assert outcome.hop_count >= 1


def test_online_accumulation_matches_the_stored_batch() -> None:
    project_root = Path(__file__).resolve().parents[1]
    conductors = parse_matlab_geometry(project_root / "input" / "INPUT_EXAMPLE.m")
    master_vgss = build_vgss_for_conductor(conductors, "C1")

    batch = run_frw_geometry(
        conductors,
        master_vgss,
        np.random.default_rng(123),
        num_walks=4,
        first_transition_mode=STRATIFIED_FIRST_TRANSITION,
    )
    batch_row = calculate_capacitance_row(
        batch["walks"],
        batch["H"],
        conductors,
        master_vgss,
    )

    system_center, system_size = get_system_bounds(conductors)
    online_stats = VGSSSamplingStats()
    online_accumulator = CapacitanceRowAccumulator(
        conductors,
        master_vgss,
        first_transition_mode=STRATIFIED_FIRST_TRANSITION,
    )
    online_rng = np.random.default_rng(123)
    gft = CubeGFT.load_default()

    for walk_index in range(4):
        outcome = run_one_walk(
            conductors=conductors,
            master_vgss=master_vgss,
            gft=gft,
            rng=online_rng,
            walk_index=walk_index,
            sampling_stats=online_stats,
            system_center=system_center,
            math_kill_radius=system_size * 500_000.0,
            first_transition_mode=STRATIFIED_FIRST_TRANSITION,
            record_path=False,
        )
        online_accumulator.update(outcome)

    online_h = online_stats.estimate_H(master_vgss.sampling_context)
    online_row = online_accumulator.finalize(online_h)

    assert online_h == pytest.approx(batch["H"])
    assert online_row["geometric_row"] == pytest.approx(
        batch_row["geometric_row"]
    )
    assert online_row["standard_error_F"] == pytest.approx(
        batch_row["standard_error_F"]
    )
