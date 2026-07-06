from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CubeGFT:
    probabilities: FloatArray
    cdf: FloatArray
    panel_centers: FloatArray
    resolution: int

    @classmethod
    def load(cls, path: str | Path) -> "CubeGFT":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"GFT file not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            resolution = int(data["resolution"])

            probabilities = np.asarray(
                data["probabilities"],
                dtype=np.float64,
            ).reshape(-1)

            panel_centers = np.asarray(
                data["panel_centers"],
                dtype=np.float64,
            ).reshape(-1, 3)

        expected_panels = 6 * resolution * resolution

        if probabilities.shape != (expected_panels,):
            raise ValueError(
                f"Expected {expected_panels} probabilities, "
                f"got {probabilities.shape}"
            )

        if panel_centers.shape != (expected_panels, 3):
            raise ValueError(
                f"Expected panel centers with shape "
                f"({expected_panels}, 3), "
                f"got {panel_centers.shape}"
            )

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("GFT contains non-finite probabilities")

        if np.any(probabilities < 0.0):
            raise ValueError("GFT contains negative probabilities")

        probability_sum = probabilities.sum()

        if probability_sum <= 0.0:
            raise ValueError("GFT probability sum must be positive")

        probabilities = probabilities / probability_sum

        cdf = np.cumsum(probabilities)
        cdf[-1] = 1.0

        return cls(
            probabilities=probabilities,
            cdf=cdf,
            panel_centers=panel_centers,
            resolution=resolution,
        )

    def sample_relative_point( self, rng: np.random.Generator, ) -> FloatArray:

        u = rng.random()

        index = int( np.searchsorted(self.cdf, u, side="right", ) )
        index = min( index, len(self.panel_centers) - 1, )

        return self.panel_centers[index].copy()

    def sample_next_point(
        self,
        current_point: FloatArray,
        cube_half_size: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        if cube_half_size <= 0.0:
            raise ValueError( "cube_half_size must be positive" )

        current_point = np.asarray( current_point, dtype=np.float64, )

        if current_point.shape != (3,):
            raise ValueError( "current_point must have shape (3,)" )

        relative_point = self.sample_relative_point(rng)

        #next_point = current_point + a/2 * ξ
        return (current_point + cube_half_size * relative_point )