from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np

from .models import GaussianSurface, SurfaceFace
from .vgss_geometry import build_neighbor_lists, classify_candidate

ImportanceFunction = Callable[[np.ndarray], float]


def uniform_importance(_point: np.ndarray) -> float:
    """Default Algorithm 4.1 importance function: p(r) = 1."""
    return 1.0


def sample_point_uniformly_on_face(
    face: SurfaceFace,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample a point on a rectangular face."""

    origin = face.vertices[0]
    edge_1 = face.vertices[1] - origin
    edge_2 = face.vertices[3] - origin
    u, v = rng.random(2)
    return origin + u * edge_1 + v * edge_2


def _normalized_area_probabilities(areas: np.ndarray) -> np.ndarray:
    if areas.ndim != 1 or len(areas) == 0:
        raise ValueError("At least one positive area is required.")
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("All surface areas must be finite and positive.")

    total_area = float(areas.sum())
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("The total surface area must be finite and positive.")

    return areas / total_area


def choose_surface_index_by_area(
    surfaces: Sequence[GaussianSurface],
    rng: np.random.Generator,
) -> int:
    """Return an area-weighted BGS index for Algorithm 4.1 Step 1."""

    if not surfaces:
        raise ValueError("At least one Gaussian surface is required.")

    areas = np.asarray([surface.area for surface in surfaces], dtype=float)
    probabilities = _normalized_area_probabilities(areas)
    return int(rng.choice(len(surfaces), p=probabilities))


def choose_surface_by_area(
    surfaces: Sequence[GaussianSurface],
    rng: np.random.Generator,
) -> GaussianSurface:
    """Area-weighted BGS choice: Algorithm 4.1 Step 1."""

    return surfaces[choose_surface_index_by_area(surfaces, rng)]


def choose_face_index_by_area(
    surface: GaussianSurface,
    rng: np.random.Generator,
) -> int:
    """Return an area-weighted face index for uniform BGS sampling."""

    areas = np.asarray([face.area for face in surface.faces], dtype=float)
    probabilities = _normalized_area_probabilities(areas)
    return int(rng.choice(len(surface.faces), p=probabilities))


def choose_face_by_area(
    surface: GaussianSurface,
    rng: np.random.Generator,
) -> SurfaceFace:
    """Area-weighted face choice for uniform sampling over one cuboid BGS."""

    return surface.faces[choose_face_index_by_area(surface, rng)]


@dataclass(frozen=True)
class VGSSSamplingContext:
    """Precomputed data reused by repeated SampleOnVGSS calls for one net."""

    surfaces: tuple[GaussianSurface, ...]
    neighbor_lists: tuple[tuple[int, ...], ...]
    importance_function: ImportanceFunction
    upper_bound: float
    total_component_area: float
    abs_tol: float


@dataclass(frozen=True)
class VGSSSample:
    """Accepted VGSS point together with its exact source-face normal."""

    point: np.ndarray
    normal: np.ndarray
    source_surface_index: int
    face_direction: str


@dataclass
class VGSSSamplingStats:
    """Statistics for the VGSS area, mean p(r), and H estimators."""

    proposal_count: int = 0
    uniform_vgss_count: int = 0
    accepted_count: int = 0
    geometry_rejections: int = 0
    multiplicity_rejections: int = 0
    importance_rejections: int = 0
    importance_value_sum: float = 0.0

    @property
    def rejection_count(self) -> int:
        return self.proposal_count - self.accepted_count

    @property
    def acceptance_rate(self) -> float:
        """Final Algorithm 4.1 acceptance rate, after p(r)/U rejection."""
        if self.proposal_count == 0:
            return 0.0
        return self.accepted_count / self.proposal_count

    @property
    def geometric_acceptance_rate(self) -> float:
        """Rate after geometry and multiplicity correction, before p(r)/U."""
        if self.proposal_count == 0:
            return 0.0
        return self.uniform_vgss_count / self.proposal_count

    @property
    def importance_acceptance_rate(self) -> float:
        """Conditional acceptance rate of the p(r)/U rejection step."""

        if self.uniform_vgss_count == 0:
            return 0.0
        return self.accepted_count / self.uniform_vgss_count

    def estimate_surface_area(self, context: VGSSSamplingContext) -> float:
        """Estimate the actual VGSS area Sg. Sg = A * N"""
        if self.proposal_count == 0:
            raise ValueError("S cannot be estimated before any VGSS proposals.")
        return context.total_component_area * self.geometric_acceptance_rate

    def estimate_Sg(self, context: VGSSSamplingContext) -> float:
        """Backward-compatible function for estimate_surface_area()."""
        return self.estimate_surface_area(context)

    def estimate_mean_p(self) -> float:
        """Estimate the statistical mean of p(r) over uniform VGSS samples."""
        if self.uniform_vgss_count == 0:
            raise ValueError("mean p(r) cannot be estimated before a uniform VGSS sample.")
        return self.importance_value_sum / self.uniform_vgss_count

    def estimate_H(self, context: VGSSSamplingContext) -> float:
        """Estimate H = S * mean(p(r)) using the book's formulation."""
        return self.estimate_surface_area(context) * self.estimate_mean_p()

    def as_dict(self, context: VGSSSamplingContext):
        """Return JSON-serializable sampling metadata."""

        surface_area = self.estimate_surface_area(context)
        mean_p = self.estimate_mean_p()
        h_estimate = surface_area * mean_p

        return {
            "proposal_count": self.proposal_count,
            "uniform_vgss_count": self.uniform_vgss_count,
            "accepted_count": self.accepted_count,
            "rejection_count": self.rejection_count,
            "geometry_rejections": self.geometry_rejections,
            "multiplicity_rejections": self.multiplicity_rejections,
            "importance_rejections": self.importance_rejections,
            "acceptance_rate": self.acceptance_rate,
            "geometric_acceptance_rate": self.geometric_acceptance_rate,
            "importance_acceptance_rate": self.importance_acceptance_rate,
            "upper_bound_U": context.upper_bound,
            "total_component_area": context.total_component_area,
            "Sg_estimate": surface_area,
            "mean_p": mean_p,
            "H_estimate": h_estimate,
        }


def prepare_vgss_sampling(
    surfaces: Sequence[GaussianSurface],
    *,
    importance_function: ImportanceFunction = uniform_importance,
    upper_bound: float = 1.0,
    abs_tol: float = 1e-12,
) -> VGSSSamplingContext:
    """Validate and precompute one net's VGSS sampling context."""

    if not math.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound U must be finite and positive.")

    surface_tuple = tuple(surfaces)
    surface_areas = np.asarray(
        [surface.area for surface in surface_tuple],
        dtype=float,
    )
    _normalized_area_probabilities(surface_areas)
    for surface in surface_tuple:
        _normalized_area_probabilities(
            np.asarray([face.area for face in surface.faces], dtype=float)
        )

    return VGSSSamplingContext(
        surfaces=surface_tuple,
        neighbor_lists=build_neighbor_lists(surface_tuple, abs_tol=abs_tol),
        importance_function=importance_function,
        upper_bound=float(upper_bound),
        total_component_area=float(surface_areas.sum()),
        abs_tol=float(abs_tol),
    )

def sample_on_vgss(
    context: VGSSSamplingContext,
    rng: np.random.Generator,
    *,
    max_attempts: int = 10000,
    stats: VGSSSamplingStats | None = None,
) -> VGSSSample:
    """Implement Algorithm 4.1 and return the accepted point and face normal."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")

    surfaces = context.surfaces

    for _attempt in range(1, max_attempts + 1):
        if stats is not None:
            stats.proposal_count += 1

        # Steps 1 and 2: area-weighted BGS selection and uniform point proposal.
        source_surface_index = choose_surface_index_by_area(surfaces, rng)
        source_surface = surfaces[source_surface_index]
        source_face = choose_face_by_area(source_surface, rng)
        point = sample_point_uniformly_on_face(source_face, rng)

        # Steps 3 and 4: reject buried surfaces and opposite-normal interfaces.
        classification = classify_candidate(
            surfaces,
            source_surface_index,
            source_face,
            point,
            neighbor_indices=context.neighbor_lists[source_surface_index],
            abs_tol=context.abs_tol,
        )
        if not classification.is_geometrically_valid:
            if stats is not None:
                stats.geometry_rejections += 1
            continue

        # Step 5.
        x1, x2 = rng.random(2)

        # Step 6: remove duplicate representation with probability 1/n_c(r).
        multiplicity_acceptance = 1.0 / classification.coincident_surface_count
        if x1 > multiplicity_acceptance:
            if stats is not None:
                stats.multiplicity_rejections += 1
            continue

        # The point is now a uniform sample on the actual VGSS.  Record
        # p(r) here, before the p(r)/U rejection, so that
        # H = S * statistical_mean(p(r)) can be estimated directly.
        importance_value = float(context.importance_function(point))
        if not math.isfinite(importance_value) or importance_value < 0.0:
            raise ValueError("p(r) must be finite and non-negative.")
        if importance_value > context.upper_bound + context.abs_tol:
            raise ValueError(f"p(r)={importance_value} exceeds upper bound U={context.upper_bound}.")

        if stats is not None:
            stats.uniform_vgss_count += 1
            stats.importance_value_sum += importance_value

        # Step 7: importance rejection p(r)/U.
        if x2 > importance_value / context.upper_bound:
            if stats is not None:
                stats.importance_rejections += 1
            continue

        # Step 8.  Keep the normal from the exact accepted VGSS face.
        if stats is not None:
            stats.accepted_count += 1
        return VGSSSample(
            point=np.asarray(point, dtype=float).copy(),
            normal=np.asarray(source_face.normal, dtype=float).copy(),
            source_surface_index=source_surface_index,
            face_direction=source_face.direction,
        )

    raise RuntimeError(
        "SampleOnVGSS did not accept a point within "
        f"{max_attempts} attempts. Check the BGS geometry, p(r), and U."
    )
