from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Union


_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_IDENTIFIER_PATTERN = r"[A-Za-z_]\w*"
_ASSIGNMENT_RE = re.compile(
    rf"(?P<name>{_IDENTIFIER_PATTERN})\s*=\s*(?:"
    rf"(?P<matrix>\[(?P<body>.*?)\])|"
    rf"(?P<number>{_NUMBER_PATTERN})"
    rf")\s*;",
    re.DOTALL,
)
_TOKEN_RE = re.compile(rf"{_NUMBER_PATTERN}|{_IDENTIFIER_PATTERN}")
_NUMBER_RE = re.compile(rf"^(?:{_NUMBER_PATTERN})$")


class MatlabSyntaxParseError(ValueError):
    """Raised when the supported MATLAB input subset cannot be parsed safely."""


@dataclass(frozen=True)
class MatlabScalarAssignment:
    name: str
    value: float


@dataclass(frozen=True)
class MatlabMatrixAssignment:
    name: str
    rows: tuple[tuple[float | str, ...], ...]


MatlabAssignment = Union[MatlabScalarAssignment, MatlabMatrixAssignment]


@dataclass(frozen=True)
class MatlabDocument:
    """Ordered assignments parsed from the supported MATLAB input subset."""

    assignments: tuple[MatlabAssignment, ...]

    def scalar_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for assignment in self.assignments:
            if isinstance(assignment, MatlabScalarAssignment):
                values[assignment.name] = assignment.value
        return values


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("%", maxsplit=1)[0] for line in text.splitlines())


def _parse_matrix_rows(body: str, variable_name: str) -> tuple[tuple[float | str, ...], ...] | None:
    """Parse simple numeric/symbolic MATLAB matrix rows.

    Function calls, cell arrays, quoted strings, arithmetic expressions, and other
    executable MATLAB constructs are deliberately not evaluated.  Such aggregate
    assignments (for example ``ObjTag = [repmat(...)]``) are ignored by returning
    ``None``; geometry/EPS matrices in the supported input format contain only
    numbers and scalar variable names.
    """

    if any(char in body for char in "(){}'\""):
        return None

    rows: list[tuple[float | str, ...]] = []
    for raw_row in body.split(";"):
        row_text = raw_row.replace(",", " ").strip()
        if not row_text:
            continue

        tokens = _TOKEN_RE.findall(row_text)
        if not tokens:
            continue

        # Remove the tokens and ordinary matrix separators. Anything left means
        # the row contains syntax outside the intentionally small supported subset.
        residue = _TOKEN_RE.sub(" ", row_text)
        if residue.strip():
            return None

        parsed: list[float | str] = []
        for token in tokens:
            if _NUMBER_RE.fullmatch(token):
                parsed.append(float(token))
            else:
                parsed.append(token)
        rows.append(tuple(parsed))

    if not rows:
        raise MatlabSyntaxParseError(f"{variable_name}: matrix contains no rows.")
    return tuple(rows)


def parse_matlab_text(text: str) -> MatlabDocument:
    """Parse the non-executable MATLAB subset used by geometry input files.

    Supported assignments are numeric scalars and matrices containing numeric
    literals and/or scalar variable references.  Assignment order is preserved so
    the interpreter can support the legacy ``eps_r``-before-conductor convention.
    """

    uncommented = _strip_comments(text)
    assignments: list[MatlabAssignment] = []

    for match in _ASSIGNMENT_RE.finditer(uncommented):
        name = match.group("name")
        number = match.group("number")
        if number is not None:
            assignments.append(MatlabScalarAssignment(name=name, value=float(number)))
            continue

        body = match.group("body")
        if body is None:
            continue
        rows = _parse_matrix_rows(body, name)
        if rows is None:
            # Deliberately ignore MATLAB expressions that are outside the input
            # subset instead of attempting to execute or partially evaluate them.
            continue
        assignments.append(MatlabMatrixAssignment(name=name, rows=rows))

    if not assignments:
        raise MatlabSyntaxParseError("No supported MATLAB assignments were found.")

    return MatlabDocument(assignments=tuple(assignments))


def parse_matlab_file(path: str | Path) -> MatlabDocument:
    file_path = Path(path)
    return parse_matlab_text(file_path.read_text(encoding="utf-8"))
