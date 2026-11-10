"""BGS generation and virtual Gaussian-surface sampling."""

from .gaussian_surface import (
    BGSGenerationResult,
    UnsupportedGeometryError,
    generate_all_bgs,
    generate_bgs,
)
from geometry import Bounds, Conductor
from .models import GaussianSurface, SurfaceFace
from .sampling import (
    VGSSSamplingContext,
    VGSSSamplingStats,
    prepare_vgss_sampling,
    sample_on_vgss,
    uniform_importance,
)
from .topology import (
    ConductorNet,
    InvalidConductorGeometryError,
    find_net_containing,
    find_touching_nets,
)
from .vgss import (
    NetVGSS,
    build_all_net_vgss,
    build_net_vgss,
    build_vgss_for_conductor,
)

__all__ = [
    "Bounds",
    "Conductor",
    "GaussianSurface",
    "SurfaceFace",
    "BGSGenerationResult",
    "UnsupportedGeometryError",
    "generate_bgs",
    "generate_all_bgs",
    "ConductorNet",
    "InvalidConductorGeometryError",
    "find_touching_nets",
    "find_net_containing",
    "VGSSSamplingContext",
    "VGSSSamplingStats",
    "prepare_vgss_sampling",
    "sample_on_vgss",
    "uniform_importance",
    "NetVGSS",
    "build_net_vgss",
    "build_all_net_vgss",
    "build_vgss_for_conductor",
    "plot_vgss_results",
]


def __getattr__(name: str):
    # Lazy import so the package works without matplotlib installed.
    if name == "plot_vgss_results":
        from .visualization import plot_vgss_results

        return plot_vgss_results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
