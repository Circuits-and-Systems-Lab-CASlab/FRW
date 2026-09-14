#!/usr/bin/env python3
"""Independent analytical validation of homogeneous-cube GGFT/WVT tables.

This diagnostic does not import the production ``GGFT`` or ``FRW`` modules.
It reads NPZ tables directly and compares them with a separate separation-of-
variables solution of the Dirichlet problem on the reference cube ``[-1, 1]^3``.

For a boundary panel on (for example) the face x = +1, the panel transition
probability is obtained by integrating the rectangular-box Poisson-kernel
series over that panel. Derivatives with respect to the source coordinates at
the cube centre are differentiated analytically term by term. The reference
WVT is then

    WVT_j = grad(P_j) / P_j.

The implementation uses matrix products of integrated sine bases, so an n=200
table can be checked without constructing a dense 240000-by-M^2 matrix.

Outputs are written under ``output/diagnostics`` by default, which is ignored
by the repository's .gitignore. This script itself is intended to live under
``diagnostics/``, which is also ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
FACE_NAMES = ("z-", "z+", "y-", "y+", "x-", "x+")
COMPONENT_NAMES = ("x", "y", "z")
REFERENCE_HALF_SIZE = 1.0
REFERENCE_SIDE_LENGTH = 2.0


@dataclass(frozen=True)
class TableData:
    path: Path
    resolution: int
    probabilities: FloatArray
    panel_centers: FloatArray
    gradients: FloatArray
    wvt: FloatArray


@dataclass(frozen=True)
class ReferenceData:
    resolution: int
    terms: int
    probabilities: FloatArray
    panel_centers: FloatArray
    gradients: FloatArray
    wvt: FloatArray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate homogeneous-cube GGFT probability, gradient, and WVT "
            "tables against an independent analytical eigenfunction series."
        )
    )
    parser.add_argument(
        "--tables",
        type=Path,
        nargs="+",
        help=(
            "One or more GGFT NPZ files. If omitted, all "
            "src/GGFT/data/cube_tables_n*.npz files are used."
        ),
    )
    parser.add_argument(
        "--series-terms",
        type=int,
        default=41,
        help=(
            "Number of sine modes per transverse direction in the analytical "
            "reference. The centre-source series converges exponentially; "
            "41 is normally ample. Default: 41."
        ),
    )
    parser.add_argument(
        "--convergence-terms",
        type=int,
        nargs="*",
        default=[11, 21, 31, 41],
        help=(
            "Series truncations used to report analytical-reference "
            "convergence. Default: 11 21 31 41."
        ),
    )
    parser.add_argument(
        "--harmonic-k",
        type=float,
        nargs="*",
        default=[0.5, 1.0, 1.5],
        help=(
            "Wave numbers for independent harmonic-function reproduction "
            "tests. Default: 0.5 1.0 1.5."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/diagnostics/ggft_wvt_validation"),
        help="Directory for compact JSON/CSV/plot outputs.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save face heatmaps of probability and WVT errors.",
    )
    parser.add_argument(
        "--plot-face",
        choices=FACE_NAMES,
        default="x+",
        help="Face used for optional heatmaps. Default: x+.",
    )
    return parser


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def discover_tables(repo_root: Path) -> list[Path]:
    return sorted(
        repo_root.glob("src/GGFT/data/cube_tables_n*.npz"),
        key=lambda path: path.name,
    )


def load_table(path: Path) -> TableData:
    if not path.is_file():
        raise FileNotFoundError(f"GGFT table not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {
            "resolution",
            "probabilities",
            "panel_centers",
            "gradients",
            "wvt",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path}: missing NPZ fields: {missing}")

        resolution = int(data["resolution"])
        probabilities = np.asarray(
            data["probabilities"], dtype=np.float64
        ).reshape(-1)
        panel_centers = np.asarray(
            data["panel_centers"], dtype=np.float64
        ).reshape(-1, 3)
        gradients = np.asarray(
            data["gradients"], dtype=np.float64
        ).reshape(-1, 3)
        wvt = np.asarray(data["wvt"], dtype=np.float64).reshape(-1, 3)

    panel_count = 6 * resolution * resolution
    expected_vector = (panel_count,)
    expected_matrix = (panel_count, 3)
    if probabilities.shape != expected_vector:
        raise ValueError(
            f"{path}: probabilities shape {probabilities.shape}; "
            f"expected {expected_vector}"
        )
    for name, array in (
        ("panel_centers", panel_centers),
        ("gradients", gradients),
        ("wvt", wvt),
    ):
        if array.shape != expected_matrix:
            raise ValueError(
                f"{path}: {name} shape {array.shape}; expected {expected_matrix}"
            )

    for name, array in (
        ("probabilities", probabilities),
        ("panel_centers", panel_centers),
        ("gradients", gradients),
        ("wvt", wvt),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{path}: {name} contains non-finite values")

    if np.any(probabilities <= 0.0):
        bad = int(np.count_nonzero(probabilities <= 0.0))
        raise ValueError(f"{path}: {bad} panel probabilities are non-positive")

    return TableData(
        path=path,
        resolution=resolution,
        probabilities=probabilities,
        panel_centers=panel_centers,
        gradients=gradients,
        wvt=wvt,
    )


def build_panel_centers(resolution: int) -> FloatArray:
    axis = -1.0 + (
        np.arange(resolution, dtype=np.float64) + 0.5
    ) * (2.0 / resolution)
    centers: list[tuple[float, float, float]] = []

    centers.extend((x, y, -1.0) for y in axis for x in axis)
    centers.extend((x, y, 1.0) for y in axis for x in axis)
    centers.extend((x, -1.0, z) for z in axis for x in axis)
    centers.extend((x, 1.0, z) for z in axis for x in axis)
    centers.extend((-1.0, y, z) for z in axis for y in axis)
    centers.extend((1.0, y, z) for z in axis for y in axis)
    return np.asarray(centers, dtype=np.float64)


def integrated_sine_basis(resolution: int, terms: int) -> FloatArray:
    """Return exact panel integrals of sin(m*pi*x/L) on [0, L]."""
    length = REFERENCE_SIDE_LENGTH
    edges = np.linspace(0.0, length, resolution + 1, dtype=np.float64)
    modes = np.arange(1, terms + 1, dtype=np.float64)
    wave_numbers = modes * np.pi / length
    return (
        np.cos(edges[:-1, None] * wave_numbers)
        - np.cos(edges[1:, None] * wave_numbers)
    ) / wave_numbers


def exact_center_sine_cosine(terms: int) -> tuple[FloatArray, FloatArray]:
    """sin(m*pi/2) and cos(m*pi/2), formed exactly by parity."""
    mode_numbers = np.arange(1, terms + 1, dtype=np.int64)
    sine = np.zeros(terms, dtype=np.float64)
    cosine = np.zeros(terms, dtype=np.float64)

    odd = mode_numbers % 2 == 1
    sine[odd] = (-1.0) ** ((mode_numbers[odd] - 1) // 2)

    even = ~odd
    cosine[even] = (-1.0) ** (mode_numbers[even] // 2)
    return sine, cosine


def stable_normal_factors(terms: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return transverse k, centre value factor, and normal derivative factor.

    On a box [0, L]^3, the upper-face separated solution contains

        sinh(lambda*x0) / sinh(lambda*L).

    At x0=L/2 this equals exp(-a)/(1+exp(-2a)), where a=lambda*L/2.
    Its x0 derivative equals lambda*exp(-a)/(1-exp(-2a)).
    These forms avoid overflow for high modes.
    """
    length = REFERENCE_SIDE_LENGTH
    modes = np.arange(1, terms + 1, dtype=np.float64)
    wave_numbers = modes * np.pi / length
    lambda_mn = np.sqrt(
        wave_numbers[:, None] ** 2 + wave_numbers[None, :] ** 2
    )
    a = lambda_mn * length / 2.0
    exponential = np.exp(-a)
    value_factor = exponential / (1.0 + np.exp(-2.0 * a))
    derivative_factor = (
        lambda_mn * exponential / (1.0 - np.exp(-2.0 * a))
    )
    return wave_numbers, value_factor, derivative_factor


def face_grid(
    basis: FloatArray,
    coefficient: FloatArray,
) -> FloatArray:
    """Map coefficient[axis_1, axis_2] to grid[axis_2_panel, axis_1_panel]."""
    return basis @ coefficient.T @ basis.T


def build_analytical_reference(resolution: int, terms: int) -> ReferenceData:
    if resolution < 3:
        raise ValueError("resolution must be at least 3")
    if terms < 1:
        raise ValueError("series terms must be positive")

    basis = integrated_sine_basis(resolution, terms)
    sine, cosine = exact_center_sine_cosine(terms)
    wave_numbers, value_factor, normal_derivative_factor = (
        stable_normal_factors(terms)
    )

    prefactor = 4.0 / (REFERENCE_SIDE_LENGTH**2)
    base = (
        prefactor
        * sine[:, None]
        * sine[None, :]
        * value_factor
    )
    tangent_first = (
        prefactor
        * (wave_numbers * cosine)[:, None]
        * sine[None, :]
        * value_factor
    )
    tangent_second = (
        prefactor
        * sine[:, None]
        * (wave_numbers * cosine)[None, :]
        * value_factor
    )
    normal = (
        prefactor
        * sine[:, None]
        * sine[None, :]
        * normal_derivative_factor
    )

    probability_grid = face_grid(basis, base)
    tangent_first_grid = face_grid(basis, tangent_first)
    tangent_second_grid = face_grid(basis, tangent_second)
    normal_grid = face_grid(basis, normal)

    probability_faces: list[FloatArray] = []
    gradient_faces: list[FloatArray] = []

    # z-, z+: grid rows are y, columns are x.
    for normal_sign in (-1.0, 1.0):
        probability_faces.append(probability_grid.ravel())
        gradient_faces.append(
            np.column_stack(
                (
                    tangent_first_grid.ravel(),
                    tangent_second_grid.ravel(),
                    (normal_sign * normal_grid).ravel(),
                )
            )
        )

    # y-, y+: grid rows are z, columns are x.
    for normal_sign in (-1.0, 1.0):
        probability_faces.append(probability_grid.ravel())
        gradient_faces.append(
            np.column_stack(
                (
                    tangent_first_grid.ravel(),
                    (normal_sign * normal_grid).ravel(),
                    tangent_second_grid.ravel(),
                )
            )
        )

    # x-, x+: grid rows are z, columns are y.
    for normal_sign in (-1.0, 1.0):
        probability_faces.append(probability_grid.ravel())
        gradient_faces.append(
            np.column_stack(
                (
                    (normal_sign * normal_grid).ravel(),
                    tangent_first_grid.ravel(),
                    tangent_second_grid.ravel(),
                )
            )
        )

    probabilities = np.concatenate(probability_faces)
    gradients = np.vstack(gradient_faces)

    tiny = np.finfo(np.float64).tiny
    if np.any(probabilities <= tiny):
        count = int(np.count_nonzero(probabilities <= tiny))
        raise RuntimeError(
            f"Analytical truncation produced {count} non-positive panel probabilities; "
            "increase --series-terms."
        )
    wvt = gradients / probabilities[:, None]

    return ReferenceData(
        resolution=resolution,
        terms=terms,
        probabilities=probabilities,
        panel_centers=build_panel_centers(resolution),
        gradients=gradients,
        wvt=wvt,
    )


def relative_l2(numerical: FloatArray, reference: FloatArray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(numerical - reference))
    return float(np.linalg.norm(numerical - reference) / denominator)


def regression_slope(numerical: FloatArray, reference: FloatArray) -> float:
    denominator = float(np.dot(reference, reference))
    if denominator == 0.0:
        return math.nan
    return float(np.dot(reference, numerical) / denominator)


def correlation(numerical: FloatArray, reference: FloatArray) -> float:
    num = numerical - float(np.mean(numerical))
    ref = reference - float(np.mean(reference))
    denominator = float(np.linalg.norm(num) * np.linalg.norm(ref))
    if denominator == 0.0:
        return math.nan
    return float(np.dot(num, ref) / denominator)


def face_sums(values: FloatArray, resolution: int) -> dict[str, float]:
    panels_per_face = resolution * resolution
    return {
        face: float(
            values[index * panels_per_face : (index + 1) * panels_per_face].sum()
        )
        for index, face in enumerate(FACE_NAMES)
    }


def harmonic_probes(
    probabilities: FloatArray,
    gradients: FloatArray,
    centers: FloatArray,
    wave_numbers: Iterable[float],
) -> list[dict[str, Any]]:
    """Test the discrete boundary operator on exact harmonic functions."""
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    probes: list[dict[str, Any]] = []
    for kappa in wave_numbers:
        if not np.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("all --harmonic-k values must be positive and finite")

        definitions = (
            (
                "exp(k*x)*cos(k*y)",
                np.exp(kappa * x) * np.cos(kappa * y),
                1.0,
                np.array([kappa, 0.0, 0.0]),
            ),
            (
                "exp(k*y)*cos(k*z)",
                np.exp(kappa * y) * np.cos(kappa * z),
                1.0,
                np.array([0.0, kappa, 0.0]),
            ),
            (
                "exp(k*z)*cos(k*x)",
                np.exp(kappa * z) * np.cos(kappa * x),
                1.0,
                np.array([0.0, 0.0, kappa]),
            ),
            (
                "exp(k*x)*sin(k*y)",
                np.exp(kappa * x) * np.sin(kappa * y),
                0.0,
                np.array([0.0, kappa, 0.0]),
            ),
        )

        for name, boundary_values, exact_value, exact_gradient in definitions:
            estimated_value = float(np.dot(probabilities, boundary_values))
            estimated_gradient = gradients.T @ boundary_values
            probes.append(
                {
                    "name": name,
                    "k": float(kappa),
                    "exact_value_at_center": exact_value,
                    "estimated_value_at_center": estimated_value,
                    "value_error": estimated_value - exact_value,
                    "exact_gradient_at_center": exact_gradient,
                    "estimated_gradient_at_center": estimated_gradient,
                    "gradient_error": estimated_gradient - exact_gradient,
                    "gradient_error_norm": float(
                        np.linalg.norm(estimated_gradient - exact_gradient)
                    ),
                }
            )
    return probes


def analytical_convergence(
    resolution: int,
    term_counts: list[int],
) -> list[dict[str, Any]]:
    unique_terms = sorted(set(term_counts))
    if not unique_terms:
        return []
    references = [
        build_analytical_reference(resolution, terms)
        for terms in unique_terms
    ]
    finest = references[-1]
    rows = []
    for reference in references:
        rows.append(
            {
                "terms": reference.terms,
                "probability_sum": float(reference.probabilities.sum()),
                "probability_relative_l2_to_finest": relative_l2(
                    reference.probabilities,
                    finest.probabilities,
                ),
                "gradient_relative_l2_to_finest": [
                    relative_l2(
                        reference.gradients[:, component],
                        finest.gradients[:, component],
                    )
                    for component in range(3)
                ],
            }
        )
    return rows


def validate_table(
    table: TableData,
    reference: ReferenceData,
    harmonic_k: list[float],
) -> dict[str, Any]:
    if table.resolution != reference.resolution:
        raise ValueError("table/reference resolution mismatch")

    center_error = table.panel_centers - reference.panel_centers
    probability_error = table.probabilities - reference.probabilities
    gradient_error = table.gradients - reference.gradients
    wvt_error = table.wvt - reference.wvt

    probability_metrics = {
        "numerical_sum": float(table.probabilities.sum()),
        "reference_sum": float(reference.probabilities.sum()),
        "face_sums_numerical": face_sums(
            table.probabilities, table.resolution
        ),
        "face_sums_reference": face_sums(
            reference.probabilities, table.resolution
        ),
        "absolute_l1": float(np.sum(np.abs(probability_error))),
        "relative_l1": float(
            np.sum(np.abs(probability_error))
            / np.sum(np.abs(reference.probabilities))
        ),
        "relative_l2": relative_l2(
            table.probabilities, reference.probabilities
        ),
        "max_absolute": float(np.max(np.abs(probability_error))),
        "regression_slope": regression_slope(
            table.probabilities, reference.probabilities
        ),
        "correlation": correlation(
            table.probabilities, reference.probabilities
        ),
    }

    gradient_metrics: dict[str, Any] = {}
    for component, name in enumerate(COMPONENT_NAMES):
        gradient_metrics[name] = {
            "numerical_sum": float(table.gradients[:, component].sum()),
            "reference_sum": float(reference.gradients[:, component].sum()),
            "relative_l2": relative_l2(
                table.gradients[:, component],
                reference.gradients[:, component],
            ),
            "max_absolute": float(
                np.max(np.abs(gradient_error[:, component]))
            ),
            "regression_slope": regression_slope(
                table.gradients[:, component],
                reference.gradients[:, component],
            ),
            "correlation": correlation(
                table.gradients[:, component],
                reference.gradients[:, component],
            ),
        }

    probability_weights = reference.probabilities
    reference_wvt_energy = float(
        np.sum(probability_weights[:, None] * reference.wvt**2)
    )
    wvt_error_energy = float(
        np.sum(probability_weights[:, None] * wvt_error**2)
    )
    wvt_metrics = {
        "reference_probability_weighted_rms_error": math.sqrt(
            wvt_error_energy
        ),
        "reference_probability_weighted_relative_rms_error": math.sqrt(
            wvt_error_energy / reference_wvt_energy
        ),
        "max_absolute_component_error": float(np.max(np.abs(wvt_error))),
        "component_metrics": {},
    }
    for component, name in enumerate(COMPONENT_NAMES):
        reference_energy = float(
            np.dot(
                probability_weights,
                reference.wvt[:, component] ** 2,
            )
        )
        error_energy = float(
            np.dot(
                probability_weights,
                wvt_error[:, component] ** 2,
            )
        )
        wvt_metrics["component_metrics"][name] = {
            "weighted_rms_error": math.sqrt(error_energy),
            "weighted_relative_rms_error": math.sqrt(
                error_energy / reference_energy
            ),
            "regression_slope": regression_slope(
                table.wvt[:, component],
                reference.wvt[:, component],
            ),
            "correlation": correlation(
                table.wvt[:, component],
                reference.wvt[:, component],
            ),
        }

    numerical_linear_gradient = table.panel_centers.T @ table.gradients
    reference_linear_gradient = (
        reference.panel_centers.T @ reference.gradients
    )

    return {
        "table": str(table.path),
        "resolution": table.resolution,
        "panel_count": int(table.probabilities.size),
        "series_terms": reference.terms,
        "panel_center_max_absolute_error": float(
            np.max(np.abs(center_error))
        ),
        "probability": probability_metrics,
        "gradient": gradient_metrics,
        "wvt": wvt_metrics,
        "moment_identities": {
            "mean_exit_point_numerical": (
                table.probabilities @ table.panel_centers
            ),
            "mean_exit_point_reference": (
                reference.probabilities @ reference.panel_centers
            ),
            "linear_gradient_matrix_numerical": numerical_linear_gradient,
            "linear_gradient_matrix_reference": reference_linear_gradient,
            "linear_gradient_identity_error_frobenius_numerical": float(
                np.linalg.norm(numerical_linear_gradient - np.eye(3))
            ),
            "linear_gradient_identity_error_frobenius_reference": float(
                np.linalg.norm(reference_linear_gradient - np.eye(3))
            ),
        },
        "harmonic_reproduction_numerical": harmonic_probes(
            table.probabilities,
            table.gradients,
            table.panel_centers,
            harmonic_k,
        ),
        "harmonic_reproduction_reference_at_panel_centers": harmonic_probes(
            reference.probabilities,
            reference.gradients,
            reference.panel_centers,
            harmonic_k,
        ),
    }


def write_metrics_csv(results: list[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "table": result["table"],
            "resolution": result["resolution"],
            "panel_count": result["panel_count"],
            "series_terms": result["series_terms"],
            "probability_relative_l1": result["probability"]["relative_l1"],
            "probability_relative_l2": result["probability"]["relative_l2"],
            "probability_slope": result["probability"]["regression_slope"],
            "wvt_weighted_relative_rms": result["wvt"][
                "reference_probability_weighted_relative_rms_error"
            ],
            "linear_gradient_identity_error": result["moment_identities"][
                "linear_gradient_identity_error_frobenius_numerical"
            ],
        }
        for component in COMPONENT_NAMES:
            row[f"gradient_{component}_relative_l2"] = result["gradient"][
                component
            ]["relative_l2"]
            row[f"gradient_{component}_slope"] = result["gradient"][component][
                "regression_slope"
            ]
            row[f"wvt_{component}_weighted_relative_rms"] = result["wvt"][
                "component_metrics"
            ][component]["weighted_relative_rms_error"]
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_harmonic_csv(results: list[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        for source_name, key in (
            ("numerical", "harmonic_reproduction_numerical"),
            (
                "analytical_panel_center_quadrature",
                "harmonic_reproduction_reference_at_panel_centers",
            ),
        ):
            for probe in result[key]:
                rows.append(
                    {
                        "table": result["table"],
                        "resolution": result["resolution"],
                        "source": source_name,
                        "probe": probe["name"],
                        "k": probe["k"],
                        "value_error": probe["value_error"],
                        "gradient_error_norm": probe["gradient_error_norm"],
                        "gradient_error_x": probe["gradient_error"][0],
                        "gradient_error_y": probe["gradient_error"][1],
                        "gradient_error_z": probe["gradient_error"][2],
                    }
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_error_plots(
    table: TableData,
    reference: ReferenceData,
    output_dir: Path,
    face_name: str,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plots requires matplotlib") from exc

    face_index = FACE_NAMES.index(face_name)
    count = table.resolution * table.resolution
    start = face_index * count
    end = start + count
    shape = (table.resolution, table.resolution)

    probability_reference = reference.probabilities[start:end].reshape(shape)
    probability_error = (
        table.probabilities[start:end] - reference.probabilities[start:end]
    ).reshape(shape)
    probability_relative = probability_error / probability_reference

    paths: list[Path] = []
    plot_specs: list[tuple[str, FloatArray, str]] = [
        ("probability_relative_error", probability_relative, "relative error"),
    ]

    for component, name in enumerate(COMPONENT_NAMES):
        wvt_error = (
            table.wvt[start:end, component]
            - reference.wvt[start:end, component]
        ).reshape(shape)
        plot_specs.append((f"wvt_{name}_error", wvt_error, "absolute error"))

    for suffix, values, label in plot_specs:
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(values, origin="lower", aspect="equal")
        axis.set_title(
            f"{table.path.name}: {face_name} {suffix.replace('_', ' ')}"
        )
        axis.set_xlabel("panel index 1")
        axis.set_ylabel("panel index 2")
        figure.colorbar(image, ax=axis, label=label)
        figure.tight_layout()

        output_path = output_dir / (
            f"{table.path.stem}_{face_name}_{suffix}.png"
        )
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        paths.append(output_path)

    return paths


def print_summary(result: dict[str, Any]) -> None:
    probability = result["probability"]
    wvt = result["wvt"]
    print("\n" + "=" * 78)
    print(Path(result["table"]).name)
    print("=" * 78)
    print(f"Resolution: {result['resolution']}")
    print(f"Panels: {result['panel_count']}")
    print(
        "Probability relative L1/L2: "
        f"{probability['relative_l1']:.6e} / "
        f"{probability['relative_l2']:.6e}"
    )
    print(
        "Probability regression slope: "
        f"{probability['regression_slope']:.10f}"
    )
    for component in COMPONENT_NAMES:
        metrics = result["gradient"][component]
        print(
            f"Gradient {component}: relative L2="
            f"{metrics['relative_l2']:.6e}, "
            f"slope={metrics['regression_slope']:.10f}, "
            f"corr={metrics['correlation']:.10f}"
        )
    print(
        "WVT probability-weighted relative RMS error: "
        f"{wvt['reference_probability_weighted_relative_rms_error']:.6e}"
    )
    print(
        "Linear harmonic identity ||X^T grad(P)-I||_F: "
        f"{result['moment_identities']['linear_gradient_identity_error_frobenius_numerical']:.6e}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.series_terms < 1:
        raise SystemExit("--series-terms must be positive")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    table_paths = args.tables or discover_tables(repo_root)
    if not table_paths:
        raise SystemExit(
            "No tables found. Pass --tables or place the script under "
            "<repo>/diagnostics/."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_tables = [load_table(path) for path in table_paths]
    references: dict[int, ReferenceData] = {}
    convergence: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    plot_paths: list[Path] = []

    for table in loaded_tables:
        print(f"Building independent analytical reference for n={table.resolution}...")
        reference = references.get(table.resolution)
        if reference is None:
            reference = build_analytical_reference(
                table.resolution,
                args.series_terms,
            )
            references[table.resolution] = reference
            convergence[str(table.resolution)] = analytical_convergence(
                table.resolution,
                args.convergence_terms + [args.series_terms],
            )

        result = validate_table(
            table,
            reference,
            args.harmonic_k,
        )
        results.append(result)
        print_summary(result)

        if args.plots:
            plot_paths.extend(
                save_error_plots(
                    table,
                    reference,
                    output_dir,
                    args.plot_face,
                )
            )

    payload = {
        "method": {
            "domain": "[-1, 1]^3",
            "source": [0.0, 0.0, 0.0],
            "reference": (
                "Panel-integrated rectangular-box Poisson-kernel eigenfunction "
                "series, analytically differentiated with respect to source "
                "coordinates."
            ),
            "series_terms": args.series_terms,
            "face_order": FACE_NAMES,
        },
        "analytical_series_convergence": convergence,
        "tables": results,
        "plots": plot_paths,
    }

    summary_path = output_dir / "ggft_wvt_validation_summary.json"
    summary_path.write_text(
        json.dumps(_to_builtin(payload), indent=2),
        encoding="utf-8",
    )
    metrics_path = output_dir / "ggft_wvt_validation_metrics.csv"
    harmonic_path = output_dir / "ggft_harmonic_reproduction.csv"
    write_metrics_csv(results, metrics_path)
    write_harmonic_csv(results, harmonic_path)

    print("\nOutputs:")
    print(f"  {summary_path}")
    print(f"  {metrics_path}")
    print(f"  {harmonic_path}")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
