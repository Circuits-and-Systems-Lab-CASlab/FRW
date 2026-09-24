from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Bounds/Conductor are shared geometry primitives.  Re-exporting them here keeps
# the existing public API stable while new code can import them from ``geometry``.
from geometry.models import Bounds, Conductor, DIRECTIONS


@dataclass
class SurfaceFace:
    """One rectangular face of a generated cuboid BGS."""

    direction: str
    vertices: np.ndarray
    normal: np.ndarray
    area: float


@dataclass
class GaussianSurface:
    """Cuboid Boundary Gaussian Surface enclosing one master conductor."""

    master_conductor: str
    bounds: Bounds
    faces: tuple[SurfaceFace, ...]

    @property
    def area(self) -> float:
        return float(sum(face.area for face in self.faces))

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


def make_surface_faces(bounds: Bounds) -> tuple[SurfaceFace, ...]:
    """Construct the six outward-facing rectangles of a cuboid."""

    x0, x1 = bounds.xmin, bounds.xmax
    y0, y1 = bounds.ymin, bounds.ymax
    z0, z1 = bounds.zmin, bounds.zmax

    raw_faces = {
        "NX": (
            [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
            [-1.0, 0.0, 0.0],
            (y1 - y0) * (z1 - z0),
        ),
        "PX": (
            [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
            [1.0, 0.0, 0.0],
            (y1 - y0) * (z1 - z0),
        ),
        "NY": (
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
            [0.0, -1.0, 0.0],
            (x1 - x0) * (z1 - z0),
        ),
        "PY": (
            [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]],
            [0.0, 1.0, 0.0],
            (x1 - x0) * (z1 - z0),
        ),
        "NZ": (
            [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]],
            [0.0, 0.0, -1.0],
            (x1 - x0) * (y1 - y0),
        ),
        "PZ": (
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
            [0.0, 0.0, 1.0],
            (x1 - x0) * (y1 - y0),
        ),
    }

    return tuple(
        SurfaceFace(
            direction=direction,
            vertices=np.asarray(raw_faces[direction][0], dtype=float),
            normal=np.asarray(raw_faces[direction][1], dtype=float),
            area=float(raw_faces[direction][2]),
        )
        for direction in DIRECTIONS
    )


__all__ = [
    "Bounds",
    "Conductor",
    "DIRECTIONS",
    "SurfaceFace",
    "GaussianSurface",
    "make_surface_faces",
]
