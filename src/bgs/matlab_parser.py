"""Backward-compatible MATLAB geometry loader.

New code should use :func:`input_formats.load_geometry` or
:func:`input_formats.load_matlab_geometry`. This module keeps the historical
``parse_matlab_geometry() -> list[Conductor]`` API so existing callers are not
broken by the parser/interpreter reorganization.
"""

from __future__ import annotations

from pathlib import Path

from geometry import Conductor
from input_formats.matlab_interpreter import (
    MatlabGeometryInterpretationError,
    load_matlab_geometry,
)
from input_formats.matlab_parser import MatlabSyntaxParseError


class MatlabGeometryParseError(ValueError):
    """Compatibility error raised by the historical conductor-only loader."""


def parse_matlab_geometry(path: str | Path) -> list[Conductor]:
    """Return only conductors from a MATLAB geometry file.

    The full interpreted result, including dielectric regions and background
    permittivity, is available from ``input_formats.load_matlab_geometry``.
    """

    try:
        return list(load_matlab_geometry(path).conductors)
    except (MatlabSyntaxParseError, MatlabGeometryInterpretationError) as exc:
        raise MatlabGeometryParseError(str(exc)) from exc


__all__ = ["MatlabGeometryParseError", "parse_matlab_geometry"]
