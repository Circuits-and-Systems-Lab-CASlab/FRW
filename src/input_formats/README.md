# `input_formats`: geometry input files

Source-format parsers and interpreters. They turn an input file into the
format-independent `GeometryModel` defined in [`src/geometry`](../geometry/README.md).
Back to the [main README](../../README.md).

Examples assume the project root as the working directory and an installed
package (`pip install -e .`) or `PYTHONPATH=src`.

## Input geometry format

The MATLAB adapter has two explicit stages:

1. `input_formats.matlab_parser` parses a deliberately small, non-executable MATLAB subset: numeric scalar assignments and matrix assignments containing numeric values or scalar variable references.
2. `input_formats.matlab_interpreter` assigns physical meaning and constructs the shared `GeometryModel`.

The parser does **not** execute arbitrary MATLAB expressions. Aggregate/helper statements such as `repmat(...)` and `save(...)` are not required by the geometry interpreter.

### Legacy conductor-only form

Each conductor is represented by a MATLAB-style matrix with six rows. Each row contains four 3D face vertices, so every row contains twelve numbers.

```matlab
C1 = [
    0 0 0   0 1 0   1 1 0   1 0 0;
    0 0 1   1 0 1   1 1 1   0 1 1;
    0 0 0   1 0 0   1 0 1   0 0 1;
    0 1 0   0 1 1   1 1 1   1 1 0;
    0 0 0   0 0 1   0 1 1   0 1 0;
    1 0 0   1 1 0   1 1 1   1 0 1;
];
```

MATLAB `%` comments are allowed. A legacy `eps_r = value` declaration before a conductor matrix is retained as compatibility metadata; it is not treated as a conductor material property and is not used by the current physics.

### Standard conductor + dielectric form

The included `input/DIELECTRIC_INPUT_3CONDUCTORS.m` demonstrates the new standard adapter shape. The interpreter currently recognizes:

- `C<number>` as a conductor cuboid with six stored faces;
- `D<number>` as a dielectric cuboid reconstructed from its owned/stored faces, which may be fewer than six;
- `eps_d<number>` as that dielectric region's relative permittivity;
- `eps_air` as the background relative permittivity;
- `EPS_C*` / `EPS_D*` as resolved per-stored-face `(eps_in, eps_out)` metadata used for input consistency and retained for later interface work.

The external ownership convention can omit shared dielectric faces. The interpreter reconstructs the full dielectric **volume** from the vertices that remain. BGS and FRW currently consume only `GeometryModel.conductors`; dielectric volumes are parsed and stored but do not yet change the walk.

The geometry is expected to describe valid axis-aligned cuboids with unique `C*`/`D*` object names.

## Parse and interpret a geometry file

```python
from input_formats import load_geometry

geometry = load_geometry("input/DIELECTRIC_INPUT_3CONDUCTORS.m")
conductors = list(geometry.conductors)

for conductor in geometry.conductors:
    print(conductor.name, conductor.min_bounds, conductor.max_bounds)

for dielectric in geometry.dielectrics:
    print(dielectric.name, dielectric.bounds, dielectric.relative_permittivity)

print("background eps_r =", geometry.background_permittivity)
```

`load_geometry()` returns a format-independent `GeometryModel`. This is the preferred API for new code and is the extension point for future input formats.

For older code, `bgs.matlab_parser.parse_matlab_geometry(path)` is preserved and returns `list[Conductor]`.

`bgs.matlab_parser.parse_matlab_geometry()` remains as a compatibility wrapper returning only conductors, but new code should use `input_formats.load_geometry()`.
