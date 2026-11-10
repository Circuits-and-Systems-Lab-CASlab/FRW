# `geometry`: shared geometry model

Format-independent geometry objects shared by the input adapters, BGS/VGSS and
FRW. Back to the [main README](../../README.md).

## Dependency direction

The dependency direction is intentionally one-way: an input adapter creates a `GeometryModel`; BGS/FRW consume that model's conductor geometry; later multidielectric FRW code can consume its dielectric volumes. The solver does not parse MATLAB syntax directly.

```text
input file -> input_formats (parser + interpreter) -> GeometryModel -> bgs / FRW
```

## Objects

All are defined in `models.py` and exported from `geometry`:

```python
from geometry import DIRECTIONS, Bounds, Conductor, DielectricRegion, GeometryModel
```

| Object | Contents |
|---|---|
| `Bounds` | `xmin, xmax, ymin, ymax, zmin, zmax` of an axis-aligned cuboid; each `min` must be below its `max`. `Bounds.from_vertices()`, `expanded()`, `vertices()` |
| `Conductor` | `name`, `bounds`, `face_vertices` (the stored faces), and `relative_permittivity`, which is legacy `eps_r` metadata only and not used by the physics |
| `DielectricRegion` | `name`, `bounds`, positive `relative_permittivity` |
| `GeometryModel` | `conductors` (at least one), `dielectric_regions` (alias `dielectrics`), `background_permittivity`, `boundary_permittivity_pairs` (`EPS_*` metadata), `source_path`, `source_format`. Object names must be unique |
| `DIRECTIONS` | face names `PX, PY, PZ, NX, NY, NZ` (positive and negative x, y, z) |

The current BGS and FRW algorithms consume only `GeometryModel.conductors`.
Dielectric regions are carried as interpreted geometry for the
multi-dielectric work.
