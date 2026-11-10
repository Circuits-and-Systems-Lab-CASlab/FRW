# Capacitance Extraction with the Floating Random Walk

A Python implementation of capacitance extraction for axis-aligned conductor
structures using the Floating Random Walk (FRW) method. Walks start on Virtual
Gaussian Surfaces built around each electrical net, use precomputed GGFT
reference-cube transition tables, and are accumulated into a capacitance
matrix.

## Current state

What works today:

- MATLAB-style geometry input (conductors and dielectric boxes) through a
  parser and interpreter that produce a format-independent geometry model;
- touch-connected electrical-net detection (face, edge and corner contact);
- Boundary Gaussian Surfaces (Algorithm 4.2) and net-level Virtual Gaussian
  Surface Sets with `SampleOnVGSS` (Algorithm 4.1);
- FRW in a homogeneous medium, in open space or a finite window with
  per-axis Dirichlet (absorbing) or Neumann (reflecting) faces;
- standard or derivative-stratified first transitions (the CLI uses
  stratified);
- adaptive convergence on the row-max 3SE criterion, or a fixed walk count;
- the full capacitance matrix, computed row by row in parallel worker
  processes;
- JSON, CSV and SPICE subcircuit export, plus 3D views of walks, BGSs and
  VGSS samples (matplotlib or an interactive HTML file).

Not done yet:

- Dielectric boxes and background permittivity are parsed and stored, but the
  walk still assumes one homogeneous medium. Multi-dielectric physics is being
  developed on the `dielectrics` and `debug/dielectric-frw` branches.
- `EPS_*` boundary-permittivity metadata is parsed but not used by FRW.
- Conductors must be axis-aligned cuboids, and overlapping conductor volumes
  are rejected as invalid input.
- A custom VGSS importance function `p(r)` must satisfy `0 <= p(r) <= U` with
  a valid positive `U`; this is not checked.

## Installation

Python 3.9 or newer is required. The only required runtime dependency is
**NumPy**; everything else is optional:

| Extra | Packages | Needed for |
|---|---|---|
| *(core)* | `numpy` | BGS/VGSS construction, sampling, GGFT, FRW capacitance extraction |
| `plot` | `matplotlib` | `--visualization`, `--save-plot`, interactive figures, `plot_vgss_results` |
| `excel` | `openpyxl` | `scripts/run_excel_cases.py` |
| `dev` | `pytest`, `matplotlib`, `openpyxl` | the test suite |

No compiler is needed: the precomputed GGFT tables
(`src/GGFT/data/cube_tables_n201.npz`) are part of the repository. A C++
toolchain is only needed to regenerate them (see `src/GGFT/README.md`).

matplotlib is imported lazily. Without it, the computational commands still
work, and plotting is skipped with a message. Explicitly requesting a plot
(`--visualization`, `--save-plot`) without matplotlib is an error. The
`--html` export of `FRW.cli` does not need matplotlib.

A virtual environment is recommended, but not required. Pick one of the
options below.

### Option A: virtual environment (recommended)

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"     # or: -e .   /  -e ".[plot]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The venv lives in your home or project directory, so it needs no root access.

### Option B: user-space install on a server (no venv, no root)

```bash
python3 -m pip install --user -e .            # core only (numpy)
python3 -m pip install --user -e ".[plot]"    # with plotting
```

This also installs the `frw`, `bgs-generate`, and `vgss-sample` commands into
`~/.local/bin`. Make sure that directory is on your `PATH`.

Newer Debian and Ubuntu systems refuse `pip install --user` with
`externally-managed-environment` (PEP 668). On those systems, use Option A,
your cluster's `module load python` or conda, or (not recommended) add
`--break-system-packages`.

### Option C: run from the source tree without installing

If NumPy is already available, for example from a system or cluster module,
no installation is needed:

```bash
export PYTHONPATH="$PWD/src"
python3 -m FRW.cli input/DIELECTRIC_INPUT_3CONDUCTORS.m --walks 5000
```

### Headless servers

On machines without a display, set `MPLBACKEND=Agg` if matplotlib is
installed, and pass `--no-show` to the BGS/VGSS commands. Use `--save-plot`
to write figures to files instead.

## Quick start

Compute the capacitance matrix of the included three-conductor example with
5,000 walks per row, and also write it as CSV:

```bash
frw input/DIELECTRIC_INPUT_3CONDUCTORS.m --walks 5000 --csv
```

The matrix is printed in pF and saved to `output/frw_exported_data.json` and
`output/frw_exported_data.csv`. Without `--walks`, each row runs until it
converges (see below).

Look at the geometry and the Boundary Gaussian Surfaces first:

```bash
bgs-generate input/DIELECTRIC_INPUT_3CONDUCTORS.m --all --save-plot output/all_bgs.png --no-show
```

Every command also runs as a module from the source tree, for example
`python -m FRW.cli`, `python -m bgs.main` and `python -m bgs.vgss_cli` (see
[Option C](#option-c-run-from-the-source-tree-without-installing)).

## Command-line usage

### `frw`: capacitance extraction

```bash
frw INPUT_FILE [options]
```

By default every detected net is run as the master once, and the rows are
assembled into the full capacitance matrix. Each row stops independently when
its row-max relative three-standard-error ratio is at most `--tolerance`,
after at least `--min-walks` walks and `--min-hits` hits per entry. Each row
has its own RNG stream derived from `--seed`, so results do not depend on the
number of workers.

**Walk count and convergence**

| Option | Default | Meaning |
|---|---|---|
| `--walks N` | off | Run exactly `N` walks per row (even number) and disable adaptive convergence. |
| `--tolerance` | `0.03` | Adaptive target for the row-max relative 3SE ratio. |
| `--min-walks` | `2000` | Minimum walks per row before convergence is checked. |
| `--max-walks` | `1000000` | Hard limit per row. |
| `--check-every` | `200` | Walks between convergence checks. |
| `--min-hits` | `10` | Minimum hits required for every matrix entry. |
| `--max-hops` | `200` | Maximum cube transitions in one walk. |

**What to run**

| Option | Default | Meaning |
|---|---|---|
| `--master NET` | all nets | Run only one row, for example `--master Net1`. No matrix is assembled. |
| `--workers` | `5` | Rows run in parallel. Capped at the number of rows. |
| `--seed` | `1234` | Random seed. |
| `--scale-factor` | `1.25` | VGSS size relative to the nearest neighbour (Algorithm 4.2). Must be at least 1. |
| `--max-distance` | `10.0` | VGSS extent for a conductor with no neighbours. |

**Boundaries**

| Option | Default | Meaning |
|---|---|---|
| `--window XMIN XMAX YMIN YMAX ZMIN ZMAX` | open space | Finite simulation window. Its faces are Dirichlet (absorbing) unless listed in `--neumann-axes`. |
| `--neumann-axes x y z` | none | Make these window axes reflecting. |
| `--wall-potential` | `0.0` | Potential of the Dirichlet window faces. |

**Output**

| Option | Default | Meaning |
|---|---|---|
| `--output PATH` | `output/frw_exported_data.json` | Full results as JSON. |
| `--unit` | `pF` | `pF`, `fF`, `F`, or `auto`. |
| `--csv [PATH]` | off | Capacitance matrix as CSV (`<output>.csv` if no path). Needs all nets. |
| `--spice [PATH]` | off | Lumped SPICE subcircuit (`<output>.subckt` if no path). Needs all nets. |
| `--html [PATH]` | off | Interactive 3D view of one walk per row (`<output>_3d.html` if no path). No matplotlib needed. |
| `--visualization` | off | Show one walk per row in a matplotlib window after the run. |

Examples:

```bash
# Tighter convergence, eight workers.
frw input/DIELECTRIC_INPUT_3CONDUCTORS.m --tolerance 0.01 --workers 8

# Finite window: absorbing in x and y, reflecting in z.
frw input/FRW_MAT_test_cases/Testcase1_W0.14toS0.14/MATLABinputfile.m \
    --window -1.96 2.1 -0.75 2.3 -1.4 2.8 --neumann-axes z

# One row only, exported as an interactive HTML walk view.
frw input/DIELECTRIC_INPUT_3CONDUCTORS.m --master Net1 --walks 2000 --html
```

More on how the runner works: [`src/FRW/README.md`](src/FRW/README.md).

### `bgs-generate`: Boundary Gaussian Surfaces

```bash
bgs-generate INPUT_FILE (--master C1 | --all) [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--master NAME` | | Build the BGS around one conductor, for example `C1`. |
| `--all` | | Build the BGS around every conductor in one plot. |
| `--scale-factor` | `1` | Algorithm 4.2 scale factor. |
| `--max-distance` | `10` | BGS extent for a conductor with no neighbours. |
| `--touch-policy` | `skip` | `skip` ignores directly touching conductors; `error` stops instead. |
| `--save-plot PATH` | | Save the 3D plot (with `--all`, defaults to `output/all_bgs.png`). |
| `--no-show` | | Do not open a plot window. |

### `vgss-sample`: VGSS sampling

```bash
vgss-sample INPUT_FILE (--net-master C1 | --all-nets) [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--net-master NAME` | | Sample the net that contains this conductor. |
| `--all-nets` | | Sample every net independently. |
| `--count` | `1` | Accepted points per net. |
| `--seed` | random | NumPy seed for reproducible points. |
| `--scale-factor` | `1` | Algorithm 4.2 scale factor. |
| `--max-distance` | `10` | BGS extent for a conductor with no neighbours. |
| `--abs-tol` | `1e-12` | Absolute geometric tolerance. |
| `--save-points PATH` | | Write the points as CSV (`x,y,z`, or `net,x,y,z` with `--all-nets`). |
| `--save-plot PATH` | | Save the 3D plot. |
| `--no-show` | | Do not open a plot window. |

More examples and the Python API: [`src/bgs/README.md`](src/bgs/README.md).

## Input files

Geometry is described in a small, non-executable subset of MATLAB. Each
conductor `C<n>` is a matrix of its six faces, one row of four 3D vertices
per face. Dielectric boxes `D<n>` use the same layout but may list fewer
faces.

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

Examples are in [`input/`](input/). The full format, including dielectric
regions and permittivity variables, is described in
[`src/input_formats/README.md`](src/input_formats/README.md).

## Project structure and documentation

```text
src/
├── input_formats/   # MATLAB parser and interpreter -> GeometryModel
├── geometry/        # format-independent Bounds, Conductor, DielectricRegion, GeometryModel
├── bgs/             # BGS, net detection, VGSS construction and sampling
├── FRW/             # floating random walk, capacitance, convergence, CLI
└── GGFT/            # reference-cube GFT/WVT tables and the C++ generator
scripts/             # batch replay of the comparison workbook, worker benchmark
tests/               # pytest suite and the QA convergence runner
input/               # example geometries and test cases
```

| Folder | Documentation |
|---|---|
| `src/input_formats` | [Input file format and parser API](src/input_formats/README.md) |
| `src/geometry` | [Shared geometry model](src/geometry/README.md) |
| `src/bgs` | [BGS/VGSS algorithms, Python API and parameters](src/bgs/README.md) |
| `src/FRW` | [Walk engine, Python API and runner details](src/FRW/README.md) |
| `src/GGFT` | [Transition tables and how to regenerate them](src/GGFT/README.md) |
| `scripts` | [Workbook replay and worker benchmark](scripts/README.md) |
| `tests` | [Test suite and QA convergence runner](tests/README.md) |

## Tests

```bash
pytest
```

The plotting test is skipped automatically when matplotlib is not installed.
See [`tests/README.md`](tests/README.md) for what each test module covers.
