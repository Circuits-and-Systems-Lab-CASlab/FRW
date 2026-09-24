from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from geometry import Bounds, Conductor, DielectricRegion, GeometryModel
from .matlab_parser import (
    MatlabDocument,
    MatlabMatrixAssignment,
    MatlabScalarAssignment,
    parse_matlab_file,
)

_OBJECT_RE = re.compile(r"^(?P<kind>[CD])(?P<index>\d+)$")
_EPS_OBJECT_RE = re.compile(r"^EPS_(?P<object>[CD]\d+)$")


class MatlabGeometryInterpretationError(ValueError):
    """Raised when parsed MATLAB data cannot be mapped to the internal geometry."""


def _numeric_matrix(
    assignment: MatlabMatrixAssignment,
    scalars: dict[str, float],
    *,
    allow_symbols: bool,
) -> np.ndarray:
    rows: list[list[float]] = []
    widths: set[int] = set()

    for row in assignment.rows:
        resolved: list[float] = []
        for token in row:
            if isinstance(token, float):
                resolved.append(token)
                continue
            if not allow_symbols:
                raise MatlabGeometryInterpretationError(
                    f"{assignment.name}: geometry rows must contain numeric coordinates."
                )
            if token not in scalars:
                raise MatlabGeometryInterpretationError(
                    f"{assignment.name}: unknown scalar reference {token!r}."
                )
            resolved.append(float(scalars[token]))
        rows.append(resolved)
        widths.add(len(resolved))

    if len(widths) != 1:
        raise MatlabGeometryInterpretationError(
            f"{assignment.name}: matrix rows have inconsistent lengths {sorted(widths)}."
        )
    return np.asarray(rows, dtype=float)


def _face_vertices(assignment: MatlabMatrixAssignment) -> np.ndarray:
    matrix = _numeric_matrix(assignment, {}, allow_symbols=False)
    if matrix.shape[1] != 12:
        raise MatlabGeometryInterpretationError(
            f"{assignment.name}: expected 12 coordinates per stored face, "
            f"found {matrix.shape[1]}."
        )
    return matrix.reshape(matrix.shape[0], 4, 3)


def _validate_object_face_count(name: str, faces: np.ndarray, kind: str) -> None:
    if kind == "C" and faces.shape[0] != 6:
        raise MatlabGeometryInterpretationError(
            f"{name}: conductor cuboids must store exactly 6 faces, found {faces.shape[0]}."
        )
    if kind == "D" and not 1 <= faces.shape[0] <= 6:
        raise MatlabGeometryInterpretationError(
            f"{name}: dielectric cuboids must store between 1 and 6 owned faces, "
            f"found {faces.shape[0]}."
        )


def interpret_matlab_geometry(
    document: MatlabDocument,
    *,
    source_path: str | Path | None = None,
) -> GeometryModel:
    """Convert parsed MATLAB assignments into the format-independent geometry model.

    Supported object naming for this first adapter is intentionally explicit:
    ``C<number>`` denotes a conductor and ``D<number>`` denotes a dielectric
    region.  Dielectric ``Dk`` requires a scalar ``eps_dk``.  ``eps_air`` becomes
    the background permittivity when present.  The legacy ``eps_r`` scalar placed
    before a conductor is retained as compatibility metadata only.
    """

    scalars: dict[str, float] = {}
    latest_legacy_eps_r: float | None = None
    object_faces: dict[str, np.ndarray] = {}
    object_order: list[str] = []
    eps_assignments: dict[str, MatlabMatrixAssignment] = {}
    legacy_eps_by_conductor: dict[str, float | None] = {}

    for assignment in document.assignments:
        if isinstance(assignment, MatlabScalarAssignment):
            scalars[assignment.name] = assignment.value
            if assignment.name == "eps_r":
                latest_legacy_eps_r = assignment.value
            continue

        eps_match = _EPS_OBJECT_RE.fullmatch(assignment.name)
        if eps_match:
            eps_assignments[eps_match.group("object")] = assignment
            continue

        object_match = _OBJECT_RE.fullmatch(assignment.name)
        if object_match is None:
            # Aggregate matrices such as Panels/EpsInOut are source-format details;
            # the internal model is reconstructed from the individual C*/D* blocks.
            continue

        name = assignment.name
        if name in object_faces:
            raise MatlabGeometryInterpretationError(f"Duplicate geometry object {name!r}.")

        kind = object_match.group("kind")
        faces = _face_vertices(assignment)
        _validate_object_face_count(name, faces, kind)
        object_faces[name] = faces
        object_order.append(name)
        if kind == "C":
            legacy_eps_by_conductor[name] = latest_legacy_eps_r

    conductors: list[Conductor] = []
    dielectric_regions: list[DielectricRegion] = []

    for name in object_order:
        faces = object_faces[name]
        try:
            bounds = Bounds.from_vertices(faces)
        except ValueError as exc:
            raise MatlabGeometryInterpretationError(
                f"{name}: stored faces do not define a non-degenerate cuboid volume."
            ) from exc

        if name.startswith("C"):
            conductors.append(
                Conductor(
                    name=name,
                    bounds=bounds,
                    face_vertices=faces,
                    relative_permittivity=legacy_eps_by_conductor.get(name),
                )
            )
            continue

        eps_name = f"eps_{name.lower()}"
        if eps_name not in scalars:
            raise MatlabGeometryInterpretationError(
                f"{name}: missing required dielectric scalar {eps_name!r}."
            )
        dielectric_regions.append(
            DielectricRegion(
                name=name,
                bounds=bounds,
                relative_permittivity=float(scalars[eps_name]),
            )
        )

    if not conductors:
        raise MatlabGeometryInterpretationError("No conductor matrices C<number> were found.")

    boundary_pairs: dict[str, np.ndarray] = {}
    for object_name, eps_assignment in eps_assignments.items():
        if object_name not in object_faces:
            raise MatlabGeometryInterpretationError(
                f"{eps_assignment.name}: no matching geometry object {object_name!r}."
            )
        pairs = _numeric_matrix(eps_assignment, scalars, allow_symbols=True)
        if pairs.shape[1] != 2:
            raise MatlabGeometryInterpretationError(
                f"{eps_assignment.name}: expected two permittivity values per stored face, "
                f"found {pairs.shape[1]}."
            )
        expected_rows = object_faces[object_name].shape[0]
        if pairs.shape[0] != expected_rows:
            raise MatlabGeometryInterpretationError(
                f"{eps_assignment.name}: expected {expected_rows} rows to match "
                f"{object_name}, found {pairs.shape[0]}."
            )
        if np.any(pairs <= 0.0):
            raise MatlabGeometryInterpretationError(
                f"{eps_assignment.name}: relative permittivities must be positive."
            )
        boundary_pairs[object_name] = pairs

    source = Path(source_path) if source_path is not None else None
    return GeometryModel(
        conductors=tuple(conductors),
        dielectric_regions=tuple(dielectric_regions),
        background_permittivity=scalars.get("eps_air"),
        boundary_permittivity_pairs=boundary_pairs,
        source_path=source,
        source_format="matlab",
    )


def load_matlab_geometry(path: str | Path) -> GeometryModel:
    file_path = Path(path)
    return interpret_matlab_geometry(
        parse_matlab_file(file_path),
        source_path=file_path,
    )
