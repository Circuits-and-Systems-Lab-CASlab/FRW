"""Input adapters that parse source formats into the shared geometry model."""

from pathlib import Path

from geometry import GeometryModel
from .matlab_interpreter import (
    MatlabGeometryInterpretationError,
    interpret_matlab_geometry,
    load_matlab_geometry,
)
from .matlab_parser import (
    MatlabDocument,
    MatlabMatrixAssignment,
    MatlabScalarAssignment,
    MatlabSyntaxParseError,
    parse_matlab_file,
    parse_matlab_text,
)


class UnsupportedInputFormatError(ValueError):
    """Raised when no input adapter exists for a requested file type."""


def load_geometry(path: str | Path) -> GeometryModel:
    """Load a supported external geometry file into ``GeometryModel``."""

    file_path = Path(path)
    if file_path.suffix.lower() == ".m":
        return load_matlab_geometry(file_path)
    raise UnsupportedInputFormatError(
        f"Unsupported geometry input format {file_path.suffix!r} for {file_path}."
    )


__all__ = [
    "MatlabDocument",
    "MatlabMatrixAssignment",
    "MatlabScalarAssignment",
    "MatlabSyntaxParseError",
    "MatlabGeometryInterpretationError",
    "UnsupportedInputFormatError",
    "parse_matlab_text",
    "parse_matlab_file",
    "interpret_matlab_geometry",
    "load_matlab_geometry",
    "load_geometry",
]
