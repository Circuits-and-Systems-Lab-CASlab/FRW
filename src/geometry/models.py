from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DIRECTIONS = ("PX", "PY", "PZ", "NX", "NY", "NZ")


@dataclass
class Bounds:
    """Axis-aligned bounds for a cuboid."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def __post_init__(self) -> None:
        if not self.xmin < self.xmax:
            raise ValueError("Expected xmin < xmax.")
        if not self.ymin < self.ymax:
            raise ValueError("Expected ymin < ymax.")
        if not self.zmin < self.zmax:
            raise ValueError("Expected zmin < zmax.")

    @classmethod
    def from_vertices(cls, vertices: np.ndarray) -> "Bounds":
        points = np.asarray(vertices, dtype=float).reshape(-1, 3)
        return cls(
            xmin=float(points[:, 0].min()),
            xmax=float(points[:, 0].max()),
            ymin=float(points[:, 1].min()),
            ymax=float(points[:, 1].max()),
            zmin=float(points[:, 2].min()),
            zmax=float(points[:, 2].max()),
        )

    def expanded(self, extents: dict[str, float]) -> "Bounds":
        return Bounds(
            xmin=self.xmin - extents["NX"],
            xmax=self.xmax + extents["PX"],
            ymin=self.ymin - extents["NY"],
            ymax=self.ymax + extents["PY"],
            zmin=self.zmin - extents["NZ"],
            zmax=self.zmax + extents["PZ"],
        )

    def vertices(self) -> np.ndarray:
        return np.array(
            [
                [self.xmin, self.ymin, self.zmin],
                [self.xmax, self.ymin, self.zmin],
                [self.xmax, self.ymax, self.zmin],
                [self.xmin, self.ymax, self.zmin],
                [self.xmin, self.ymin, self.zmax],
                [self.xmax, self.ymin, self.zmax],
                [self.xmax, self.ymax, self.zmax],
                [self.xmin, self.ymax, self.zmax],
            ],
            dtype=float,
        )


@dataclass
class Conductor:
    """Axis-aligned conductor block used by BGS and FRW.

    ``relative_permittivity`` is retained only as legacy input metadata for the
    original single-value ``eps_r`` MATLAB format.  It is not interpreted as a
    material property of the conductor and is not used by the current physics.
    """

    name: str
    bounds: Bounds
    face_vertices: np.ndarray
    relative_permittivity: float | None = None

    @property
    def min_bounds(self) -> np.ndarray:
        return np.array(
            [self.bounds.xmin, self.bounds.ymin, self.bounds.zmin],
            dtype=float,
        )

    @property
    def max_bounds(self) -> np.ndarray:
        return np.array(
            [self.bounds.xmax, self.bounds.ymax, self.bounds.zmax],
            dtype=float,
        )

    def distance_to_point(self, point: np.ndarray) -> float:
        """Return the Chebyshev distance from a point to this conductor."""

        x, y, z = point
        bounds = self.bounds

        dx = max(bounds.xmin - x, 0.0, x - bounds.xmax)
        dy = max(bounds.ymin - y, 0.0, y - bounds.ymax)
        dz = max(bounds.zmin - z, 0.0, z - bounds.zmax)

        return float(max(dx, dy, dz))

    def get_surface_normal(self, point: np.ndarray) -> np.ndarray:
        """Return the outward-facing normal of the nearest face to a point.

        The direction ordering is intentionally unchanged from the existing
        implementation so this project reorganization does not alter FRW/BGS
        numerical behaviour.
        """

        point = np.asarray(point, dtype=float)
        distances = [
            abs(point[0] - self.bounds.xmin),
            abs(point[0] - self.bounds.xmax),
            abs(point[1] - self.bounds.ymin),
            abs(point[1] - self.bounds.ymax),
            abs(point[2] - self.bounds.zmin),
            abs(point[2] - self.bounds.zmax),
        ]
        nearest_face_index = int(np.argmin(distances))
        direction = DIRECTIONS[nearest_face_index]

        if direction == "PX":
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if direction == "NX":
            return np.array([-1.0, 0.0, 0.0], dtype=float)
        if direction == "PY":
            return np.array([0.0, 1.0, 0.0], dtype=float)
        if direction == "NY":
            return np.array([0.0, -1.0, 0.0], dtype=float)
        if direction == "PZ":
            return np.array([0.0, 0.0, 1.0], dtype=float)
        if direction == "NZ":
            return np.array([0.0, 0.0, -1.0], dtype=float)
        raise ValueError(f"Unexpected face direction: {direction}")


@dataclass(frozen=True)
class DielectricRegion:
    """Axis-aligned dielectric volume reconstructed from an input object."""

    name: str
    bounds: Bounds
    relative_permittivity: float

    def __post_init__(self) -> None:
        if self.relative_permittivity <= 0.0:
            raise ValueError("Dielectric relative permittivity must be positive.")


@dataclass
class GeometryModel:
    """Format-independent geometry consumed by BGS/FRW.

    Parsing and source-file conventions stop at this boundary.  The current
    BGS/FRW algorithms still consume only ``conductors``; dielectric regions are
    carried as interpreted geometry for the later multidielectric implementation.
    """

    conductors: tuple[Conductor, ...]
    dielectric_regions: tuple[DielectricRegion, ...] = ()
    background_permittivity: float | None = None
    boundary_permittivity_pairs: dict[str, np.ndarray] = field(default_factory=dict)
    source_path: Path | None = None
    source_format: str | None = None

    def __post_init__(self) -> None:
        conductor_names = [item.name for item in self.conductors]
        dielectric_names = [item.name for item in self.dielectric_regions]
        all_names = conductor_names + dielectric_names

        if not self.conductors:
            raise ValueError("GeometryModel requires at least one conductor.")
        if len(all_names) != len(set(all_names)):
            raise ValueError("Geometry object names must be unique.")
        if (
            self.background_permittivity is not None
            and self.background_permittivity <= 0.0
        ):
            raise ValueError("Background relative permittivity must be positive.")

    @property
    def dielectrics(self) -> tuple[DielectricRegion, ...]:
        """Short compatibility alias for ``dielectric_regions``."""

        return self.dielectric_regions
