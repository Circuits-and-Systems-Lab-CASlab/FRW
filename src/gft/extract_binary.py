from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
FACE_NAMES = ("z-", "z+", "y-", "y+", "x-", "x+")


def build_panel_centers(resolution: int) -> FloatArray:
    axis = -1.0 + (np.arange(resolution, dtype=np.float64) + 0.5) * (
        2.0 / resolution
    )
    centers: list[tuple[float, float, float]] = []

    # This ordering matches the C++ solver and DeepRWCap's six-face layout.
    centers.extend((x, y, -1.0) for y in axis for x in axis)  # z-
    centers.extend((x, y, 1.0) for y in axis for x in axis)   # z+
    centers.extend((x, -1.0, z) for z in axis for x in axis)  # y-
    centers.extend((x, 1.0, z) for z in axis for x in axis)   # y+
    centers.extend((-1.0, y, z) for z in axis for y in axis)  # x-
    centers.extend((1.0, y, z) for z in axis for y in axis)   # x+

    return np.asarray(centers, dtype=np.float64)


def read_deeprwcap_binary(
    path: str | Path,
) -> tuple[int, int, FloatArray, FloatArray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Binary GFT file not found: {path}")

    raw = np.fromfile(path, dtype="<f8")
    if raw.size < 2:
        raise ValueError("Binary file is too small to contain the header")

    resolution_value = float(raw[0])
    block_width_value = float(raw[1])
    resolution = int(round(resolution_value))
    block_width = int(round(block_width_value))

    if not np.isclose(resolution_value, resolution) or resolution < 3:
        raise ValueError(f"Invalid resolution in binary header: {resolution_value}")
    if not np.isclose(block_width_value, block_width) or block_width <= 0:
        raise ValueError(f"Invalid block width in binary header: {block_width_value}")

    dielectric_count = resolution**3
    probability_count = 6 * resolution * resolution
    expected_values = 2 + dielectric_count + probability_count

    if raw.size != expected_values:
        raise ValueError(
            "Unexpected binary size. "
            f"Expected {expected_values} float64 values, found {raw.size}. "
            "This converter expects one homogeneous sample with no structure data."
        )

    dielectric_start = 2
    probability_start = dielectric_start + dielectric_count
    dielectric = np.asarray(
        raw[dielectric_start:probability_start], dtype=np.float64
    )
    probabilities = np.asarray(raw[probability_start:], dtype=np.float64)

    if not np.all(np.isfinite(dielectric)):
        raise ValueError("Dielectric array contains non-finite values")
    if not np.allclose(dielectric, dielectric[0]):
        raise ValueError(
            "Binary contains dielectric interfaces; this homogeneous converter "
            "does not support them yet"
        )
    if dielectric[0] <= 0.0:
        raise ValueError("Homogeneous dielectric value must be positive")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("GFT contains non-finite probabilities")
    if np.any(probabilities < -1.0e-12):
        raise ValueError("GFT contains negative probabilities")

    probabilities = np.clip(probabilities, 0.0, None)
    probability_sum = float(probabilities.sum())
    if probability_sum <= 0.0:
        raise ValueError("GFT probability sum must be positive")

    probabilities = probabilities / probability_sum
    return resolution, block_width, dielectric, probabilities


def convert_binary_to_npz(
    binary_path: str | Path,
    npz_path: str | Path,
) -> Path:
    resolution, block_width, _, probabilities = read_deeprwcap_binary(
        binary_path
    )
    panel_centers = build_panel_centers(resolution)

    output = Path(npz_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        resolution=np.int64(resolution),
        probabilities=probabilities,
        panel_centers=panel_centers,
    )

    panels_per_face = resolution * resolution
    print(f"Resolution: {resolution}")
    print(f"Block width: {block_width}")
    print(f"Panels: {probabilities.size}")
    print(f"Probability sum: {probabilities.sum():.16g}")
    for face_index, face_name in enumerate(FACE_NAMES):
        start = face_index * panels_per_face
        end = start + panels_per_face
        print(f"{face_name}: {probabilities[start:end].sum():.16g}")
    print(f"NPZ: {output}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one homogeneous DeepRWCap-style GGFT binary sample "
            "and save it as a CubeGFT-compatible NPZ file."
        )
    )
    parser.add_argument("binary", type=Path, help="Input .bin file")
    parser.add_argument("output", type=Path, help="Output .npz file")
    args = parser.parse_args()

    convert_binary_to_npz(args.binary, args.output)


if __name__ == "__main__":
    main()
