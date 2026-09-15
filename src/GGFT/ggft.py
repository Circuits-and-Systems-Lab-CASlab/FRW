from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import as_file, files
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
DEFAULT_GFT_TABLE = "cube_tables_n201.npz"
STANDARD_FIRST_TRANSITION = "standard"
STRATIFIED_FIRST_TRANSITION = "stratified"
FIRST_TRANSITION_MODES = (
    STANDARD_FIRST_TRANSITION,
    STRATIFIED_FIRST_TRANSITION,
)
FIRST_TRANSITION_STRATA = ("inward", "outward")


@dataclass(frozen=True)
class _FirstTransitionStratum:
    """Cached derivative-importance distribution for one geometric half."""

    panel_indices: NDArray[np.int32]
    cdf: FloatArray
    absolute_mass: float


@dataclass(frozen=True)
class CubeGFT:
    """GFT probabilities and first-hop weight values for the unit cube [-1, 1]^3."""

    probabilities: FloatArray
    cdf: FloatArray
    panel_centers: FloatArray
    wvt: FloatArray
    resolution: int
    _first_transition_strata: dict[
        tuple[int, int, str], _FirstTransitionStratum
    ] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> "CubeGFT":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GFT file not found: {path}")

        #-------- load
        with np.load(path, allow_pickle=False) as data:
            resolution = int(data["resolution"])
            probabilities = np.asarray(data["probabilities"], dtype=np.float64 ).reshape(-1)
            panel_centers = np.asarray(data["panel_centers"], dtype=np.float64).reshape(-1, 3)
            wvt = np.asarray(data["wvt"], dtype=np.float64).reshape(-1, 3)

        #------ validity checks ------
        panel_count = 6 * resolution * resolution
        if probabilities.shape != (panel_count,):
            raise ValueError(f"Expected {panel_count} probabilities, got {probabilities.shape}")
        if panel_centers.shape != (panel_count, 3):
            raise ValueError(f"Expected panel_centers shape ({panel_count}, 3), "f"got {panel_centers.shape}")
        if wvt.shape != (panel_count, 3):
            raise ValueError(f"Expected wvt shape ({panel_count}, 3), got {wvt.shape}")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("Probabilities must be finite and non-negative")
        if not np.all(np.isfinite(panel_centers)):
            raise ValueError("Panel centers must be finite")
        if not np.all(np.isfinite(wvt)):
            raise ValueError("WVT values must be finite")

        #-------- compute cdf -------
        probability_sum = float(probabilities.sum())
        probabilities = probabilities / probability_sum
        cdf = np.cumsum(probabilities)
        cdf[-1] = 1.0

        return cls(
            probabilities=probabilities,
            cdf=cdf,
            panel_centers=panel_centers,
            wvt=wvt,
            resolution=resolution,
        )

    @classmethod
    def load_default(cls) -> "CubeGFT":
        """Load the production GGFT table bundled with this package.

        The resource lookup is independent of the process working directory,
        so installed commands and Python API calls use the same table.
        """

        resource = files("GGFT").joinpath("data", DEFAULT_GFT_TABLE)
        try:
            with as_file(resource) as table_path:
                return cls.load(table_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Bundled GFT table {DEFAULT_GFT_TABLE!r} was not found. "
                "Generate it with src/GGFT/generate_gft.sh 201 or "
                "reinstall the project."
            ) from exc
        
    #debug
    def print_panels(self, start: int = 0, end: int | None = None, ) -> None:
        """Print probability and derivatives for each selected panel.

        By default, every panel is printed. Use ``start`` and ``end`` to print
        only a smaller index range, for example ``print_panels(0, 20)``.
        """
        panel_count = len(self.probabilities)
        if end is None:
            end = panel_count

        if not 0 <= start <= end <= panel_count:
            raise ValueError(
                f"Expected 0 <= start <= end <= {panel_count}"
            )
        print("index face         x         y         z              P           Wx              Wy              Wz")

        for index in range(start, end):
            x, y, z = self.panel_centers[index]
            # dp_dx, dp_dy, dp_dz = self.gradients[index]

            if np.isclose(x, -1.0):
                face = "-X"
            elif np.isclose(x, 1.0):
                face = "+X"
            elif np.isclose(y, -1.0):
                face = "-Y"
            elif np.isclose(y, 1.0):
                face = "+Y"
            elif np.isclose(z, -1.0):
                face = "-Z"
            elif np.isclose(z, 1.0):
                face = "+Z"
            else:
                face = "??"

            row = (
                f"{index:5d} {face:>4s}"
                f" {x:9.5f} {y:9.5f} {z:9.5f}"
                f" {self.probabilities[index]:14.7e}"
                # f" {dp_dx:15.7e} {dp_dy:15.7e} {dp_dz:15.7e}"
            )
            wx, wy, wz = self.wvt[index]
            row += f" {wx:15.7e} {wy:15.7e} {wz:15.7e}"

            print(row)

    def sample_relative_point(self, rng: np.random.Generator, ) -> tuple[FloatArray, int, float]:
        """Return (relative point, panel index, panel probability)."""
        u = float(rng.random())
        index = int(np.searchsorted(self.cdf, u, side="right"))

        #next point, index, probability
        return ( self.panel_centers[index].copy(), index, float(self.probabilities[index]), )

    @staticmethod
    def _axis_normal(
        surface_normal: FloatArray,
    ) -> tuple[FloatArray, int, int]:
        """Return a validated axis-aligned unit normal and cache key."""

        normal = np.asarray(surface_normal, dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("surface_normal must contain three finite values")

        normal_length = float(np.linalg.norm(normal))
        if not math.isfinite(normal_length) or normal_length <= 0.0:
            raise ValueError("surface_normal must be non-zero")

        unit_normal = normal / normal_length
        axis = int(np.argmax(np.abs(unit_normal)))
        direction = 1 if unit_normal[axis] > 0.0 else -1
        expected = np.zeros(3, dtype=np.float64)
        expected[axis] = direction
        if not np.allclose(unit_normal, expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Stratified first transitions require an axis-aligned "
                "Gaussian-surface normal."
            )

        return expected, axis, direction

    def _get_first_transition_stratum(
        self,
        surface_normal: FloatArray,
        stratum: str,
    ) -> tuple[_FirstTransitionStratum, FloatArray]:
        """Build or return one geometric half-cube proposal distribution.

        Panels are sampled in proportion to ``abs(-n.grad(P_j))``.  The
        proposal deliberately retains panels whose derivative sign differs
        from the usual homogeneous inward/outward sign, which keeps the
        estimator valid for future asymmetric dielectric GGFT tables.
        """

        if stratum not in FIRST_TRANSITION_STRATA:
            choices = ", ".join(FIRST_TRANSITION_STRATA)
            raise ValueError(f"stratum must be one of: {choices}")

        unit_normal, axis, direction = self._axis_normal(surface_normal)
        cache_key = (axis, direction, stratum)
        cached = self._first_transition_strata.get(cache_key)
        if cached is not None:
            return cached, unit_normal

        signed_derivative = -self.probabilities * (
            self.wvt @ unit_normal
        )
        signed_coordinate = self.panel_centers @ unit_normal

        # Assign the splitting plane to the inward half.  Its derivative is
        # zero for a homogeneous symmetric cube, but it need not be zero for
        # a future dielectric table, so it must not be discarded.
        geometric_mask = (
            signed_coordinate <= 0.0
            if stratum == "inward"
            else signed_coordinate > 0.0
        )
        candidate_indices = np.flatnonzero(geometric_mask)
        weights = np.abs(signed_derivative[candidate_indices])
        positive = weights > 0.0
        panel_indices = candidate_indices[positive].astype(np.int32, copy=False)
        weights = weights[positive]

        absolute_mass = float(weights.sum())
        if not math.isfinite(absolute_mass) or absolute_mass <= 0.0:
            raise ValueError(
                f"The GGFT {stratum} stratum has no finite derivative mass."
            )

        cdf = np.cumsum(weights / absolute_mass, dtype=np.float64)
        cdf[-1] = 1.0
        cached = _FirstTransitionStratum(
            panel_indices=panel_indices,
            cdf=cdf,
            absolute_mass=absolute_mass,
        )
        self._first_transition_strata[cache_key] = cached
        return cached, unit_normal

    def sample_next_point(self, current_point: FloatArray, cube_half_size: float, rng: np.random.Generator, ) -> tuple[FloatArray, int, float]:
        """Map a sampled unit-cube point onto an axis-aligned physical cube."""

        relative_point, index, probability = self.sample_relative_point(rng)
        current_point = np.asarray(current_point, dtype=np.float64)

        #r' = r + a/2 * ξ where a the whole side of the cube
        next_point = current_point + cube_half_size * relative_point
        return next_point, index, probability

    def first_hop_weight( self, panel_index: int, surface_normal: FloatArray, cube_half_size: float, ) -> float:
        """Return omega for the selected panel of the first transition cube."""

        surface_normal = np.asarray(surface_normal, dtype=np.float64)

        normal_length = float(np.linalg.norm(surface_normal))

        unit_normal = surface_normal / normal_length

        #omega = - WVT * n  in WVT <- grad(P(r,r')/P(r,r')
        omega = -float(np.dot(unit_normal, self.wvt[panel_index]))
        omega = omega / cube_half_size # scale by 1 / cube_half_size to convert to maximal cube value
        return omega

    def sample_first_transition(
        self,
        current_point: FloatArray,
        cube_half_size: float,
        surface_normal: FloatArray,
        rng: np.random.Generator,
        *,
        mode: str = STANDARD_FIRST_TRANSITION,
        stratum: str | None = None,
        allocation_probability: float | None = None,
    ) -> tuple[FloatArray, int, float, float]:
        """Return ``(next point, panel index, sampling probability, omega)``.
        Stratified calls select a geometric half and sample by absolute derivative mass.
        """

        if mode not in FIRST_TRANSITION_MODES:
            choices = ", ".join(FIRST_TRANSITION_MODES)
            raise ValueError(f"mode must be one of: {choices}")

        if mode == STANDARD_FIRST_TRANSITION:
            next_point, panel_index, probability = self.sample_next_point(
                current_point=current_point,
                cube_half_size=cube_half_size,
                rng=rng,
            )
            omega = self.first_hop_weight(
                panel_index=panel_index,
                surface_normal=surface_normal,
                cube_half_size=cube_half_size,
            )
            return next_point, panel_index, probability, omega

        #--------stratified first transition--------
        if not math.isfinite(cube_half_size) or cube_half_size <= 0.0:
            raise ValueError("cube_half_size must be finite and positive")
        if stratum is None:
            raise ValueError("stratum is required in stratified mode")
        if allocation_probability is None:
            raise ValueError("allocation_probability is required in stratified mode")
        if ( not math.isfinite(allocation_probability) or not 0.0 < allocation_probability <= 1.0):
            raise ValueError("allocation_probability must be finite and in (0, 1]")

        #alternate stratum

        # Split panels into inward/outward halves and sample within each half
        # proportionally to |d_j|, where d_j = -P_j * (n · WVT_j).
        # The corrected weight sign(d_j) * A_s / (alpha_s * a) preserves the
        # standard estimator's expected value while reducing variance.
        table, unit_normal = self._get_first_transition_stratum(
            surface_normal,
            stratum,
        )
        local_index = int(np.searchsorted(table.cdf, float(rng.random()), side="right"))
        panel_index = int(table.panel_indices[local_index])
        relative_point = self.panel_centers[panel_index]
        current_point = np.asarray(current_point, dtype=np.float64)
        next_point = current_point + cube_half_size * relative_point

        signed_derivative = -float(
            np.dot(unit_normal, self.probabilities[panel_index] * self.wvt[panel_index],)
        )
        probability = abs(signed_derivative) / table.absolute_mass
        omega = (
            math.copysign(table.absolute_mass, signed_derivative)
            / allocation_probability
            / cube_half_size
        )

        return next_point, panel_index, probability, omega
