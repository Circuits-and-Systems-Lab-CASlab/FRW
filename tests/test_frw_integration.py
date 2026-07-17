from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from FRW.random_walk import run_frw_geometry
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
