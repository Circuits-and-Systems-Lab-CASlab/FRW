from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
FACE_NAMES = ("z-", "z+", "y-", "y+", "x-", "x+")
MAGIC = b"GGFTTBL1"
HEADER = struct.Struct("<8sIIIIdQQ")


@dataclass(frozen=True)
class BinaryTables:
    version: int
    resolution: int
    block_width: int
    reference_half_size: float
    dielectric: FloatArray
    probabilities: FloatArray
    gradients: FloatArray


def build_panel_centers(resolution: int) -> FloatArray:
    axis = -1.0 + (np.arange(resolution, dtype=np.float64) + 0.5) * (
        2.0 / resolution
    )
    centers: list[tuple[float, float, float]] = []

    centers.extend((x, y, -1.0) for y in axis for x in axis)  # z-
    centers.extend((x, y, 1.0) for y in axis for x in axis)   # z+
    centers.extend((x, -1.0, z) for z in axis for x in axis)  # y-
    centers.extend((x, 1.0, z) for z in axis for x in axis)   # y+
    centers.extend((-1.0, y, z) for z in axis for y in axis)  # x-
    centers.extend((1.0, y, z) for z in axis for y in axis)   # x+

    return np.asarray(centers, dtype=np.float64)


def read_binary(path: str | Path) -> BinaryTables:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GGFT binary not found: {path}")

    with path.open("rb") as stream:
        header_bytes = stream.read(HEADER.size)
        if len(header_bytes) != HEADER.size:
            raise ValueError("Binary file is too small to contain the GGFT header")

        (
            magic,
            version,
            resolution,
            block_width,
            _reserved,
            reference_half_size,
            dielectric_count,
            panel_count,
        ) = HEADER.unpack(header_bytes)

        if magic != MAGIC:
            raise ValueError(
                f"Invalid GGFT magic {magic!r}; expected {MAGIC!r}"
            )
        if version != 1:
            raise ValueError(f"Unsupported GGFT binary version: {version}")
        if resolution < 3:
            raise ValueError(f"Invalid resolution: {resolution}")
        if block_width <= 0 or resolution % block_width != 0:
            raise ValueError(f"Invalid block width: {block_width}")
        if not np.isfinite(reference_half_size) or reference_half_size <= 0.0:
            raise ValueError("reference_half_size must be positive and finite")

        expected_dielectric_count = (resolution // block_width) ** 3
        expected_panel_count = 6 * resolution * resolution
        if dielectric_count != expected_dielectric_count:
            raise ValueError(
                f"Expected {expected_dielectric_count} dielectric values, "
                f"header declares {dielectric_count}"
            )
        if panel_count != expected_panel_count:
            raise ValueError(
                f"Expected {expected_panel_count} panels, header declares {panel_count}"
            )

        count = int(dielectric_count + 4 * panel_count)
        payload = np.frombuffer(stream.read(), dtype="<f8")
        if payload.size != count:
            raise ValueError(
                f"Expected {count} float64 payload values, found {payload.size}"
            )

    offset = 0
    dielectric = np.asarray(
        payload[offset : offset + dielectric_count], dtype=np.float64
    ).copy()
    offset += int(dielectric_count)

    probabilities = np.asarray(
        payload[offset : offset + panel_count], dtype=np.float64
    ).copy()
    offset += int(panel_count)

    gradient_x = np.asarray(
        payload[offset : offset + panel_count], dtype=np.float64
    ).copy()
    offset += int(panel_count)
    gradient_y = np.asarray(
        payload[offset : offset + panel_count], dtype=np.float64
    ).copy()
    offset += int(panel_count)
    gradient_z = np.asarray(
        payload[offset : offset + panel_count], dtype=np.float64
    ).copy()

    if not np.all(np.isfinite(dielectric)) or np.any(dielectric <= 0.0):
        raise ValueError("Dielectric data must be positive and finite")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probability table contains non-finite values")
    if np.any(probabilities < -1.0e-11):
        raise ValueError("Probability table contains negative values")

    probabilities = np.clip(probabilities, 0.0, None)
    probability_sum = float(probabilities.sum())
    if probability_sum <= 0.0:
        raise ValueError("Probability sum must be positive")
    probabilities /= probability_sum

    gradients = np.column_stack((gradient_x, gradient_y, gradient_z))
    if not np.all(np.isfinite(gradients)):
        raise ValueError("Gradient table contains non-finite values")

    return BinaryTables(
        version=version,
        resolution=resolution,
        block_width=block_width,
        reference_half_size=reference_half_size,
        dielectric=dielectric,
        probabilities=probabilities,
        gradients=gradients,
    )


def convert_binary_to_npz(
    binary_path: str | Path,
    npz_path: str | Path,
) -> Path:
    tables = read_binary(binary_path)
    panel_centers = build_panel_centers(tables.resolution)

    probability_floor = np.finfo(np.float64).tiny
    if np.any(tables.probabilities <= probability_floor):
        bad = int(np.count_nonzero(tables.probabilities <= probability_floor))
        raise ValueError(
            f"Cannot construct WVT: {bad} panels have zero probability"
        )
    wvt = tables.gradients / tables.probabilities[:, None]

    output = Path(npz_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format_version=np.int64(tables.version),
        resolution=np.int64(tables.resolution),
        block_width=np.int64(tables.block_width),
        reference_half_size=np.float64(tables.reference_half_size),
        dielectric=tables.dielectric,
        probabilities=tables.probabilities,
        panel_centers=panel_centers,
        gradients=tables.gradients,
        wvt=wvt,
    )

    panels_per_face = tables.resolution * tables.resolution
    print(f"Resolution: {tables.resolution}")
    print(f"Block width: {tables.block_width}")
    print(f"Dielectric cells: {tables.dielectric.size}")
    print(f"Panels: {tables.probabilities.size}")
    print(f"Probability sum: {tables.probabilities.sum():.16g}")
    print(f"Gradient sums: {tables.gradients.sum(axis=0)}")
    for face_index, face_name in enumerate(FACE_NAMES):
        start = face_index * panels_per_face
        end = start + panels_per_face
        print(f"{face_name}: {tables.probabilities[start:end].sum():.16g}")
    print(f"NPZ: {output}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a deterministic GGFT binary bundle into an NPZ containing "
            "P, raw gradients and the normalized weight-value table."
        )
    )
    parser.add_argument("binary", type=Path, help="Input .bin file")
    parser.add_argument("output", type=Path, help="Output .npz file")
    args = parser.parse_args()

    convert_binary_to_npz(args.binary, args.output)


if __name__ == "__main__":
    main()
