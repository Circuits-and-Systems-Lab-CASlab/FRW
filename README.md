# Capacitance Geometry, BGS/VGSS Sampling, GGFT, and FRW Integration

This project provides an implementation of:

- format-independent axis-aligned conductor/dielectric geometry models;
- a parser/interpreter pipeline for MATLAB-style geometry input;
- conductor-level Boundary Gaussian Surface generation based on Algorithm 4.2;
- touch-connected electrical-net detection;
- net-level Virtual Gaussian Surface Set construction;
- `SampleOnVGSS` based on Algorithm 4.1;
- GGFT reference-cube transition tables;
- integration with the existing geometric Floating Random Walk implementation.

The parser reorganization is intentionally separate from the numerical algorithms. BGS, VGSS, GGFT, and FRW physics are unchanged in this milestone. Dielectric volumes are now represented in the internal geometry model so the later multidielectric FRW work can consume them without tying the solver to one external file format.

## Current scope

The implementation currently supports:

- MATLAB-style geometry files containing conductor matrices such as `C1 = [ ... ];`;
- axis-aligned cuboid conductor blocks;
- one cuboid BGS for each conductor block;
- face, edge, and corner contact as electrical connectivity;
- indirect net connectivity through intermediate touching blocks;
- rejection of overlapping conductor volumes as invalid input;
- one independent VGSS sampling context for each detected net;
- area-weighted BGS and face selection;
- geometric rejection of buried and internal interfaces;
- the `1 / n_c(r)` multiplicity correction;
- configurable importance rejection using `p(r) / U`;
- estimation of the normalization integral `H` from sampling statistics;
- 3D visualization of conductors, BGSs, VGSSs, and accepted sample points;
- use of accepted VGSS points as FRW starting points.

The default importance function is:

```text
p(r) = 1
U = 1
```

The input layer now reconstructs dielectric volumes and background permittivity, but the current FRW transition logic remains the original homogeneous-medium implementation. No dielectric-transition physics is enabled by this parser milestone. General non-Manhattan geometry is also outside the current scope.

## Important touching-conductor rule

Electrical nets are found as connected components. Therefore, if `C1` touches `C2` and `C2` touches `C3`, all three conductors belong to the same net even when `C1` does not directly touch `C3`.

BGS generation applies a more local rule:

- a conductor directly touching the current master conductor is ignored for that master's BGS calculation;
- a same-net conductor that does not directly touch the current master is still considered;
- conductors outside the net are also considered;
- overlapping conductor volumes are rejected.

The implementation therefore ignores only the directly touching conductor or conductors, not the complete net.


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

## Project structure

The source tree separates external formats from physical geometry and numerical algorithms:

```text
src/
├── geometry/        # format-independent Bounds, Conductor, DielectricRegion, GeometryModel
├── input_formats/   # source-format parsers and semantic interpreters
├── bgs/             # BGS, topology, VGSS construction/sampling
├── FRW/             # existing floating-random-walk implementation
└── GGFT/            # reference-cube GFT/WVT generation and loading
```

The dependency direction is intentionally one-way: an input adapter creates a `GeometryModel`; BGS/FRW consume that model's conductor geometry; later multidielectric FRW code can consume its dielectric volumes. The solver does not parse MATLAB syntax directly.

`bgs.matlab_parser.parse_matlab_geometry()` remains as a compatibility wrapper returning only conductors, but new code should use `input_formats.load_geometry()`.

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

# Python API

## Imports

The main BGS and VGSS functions are exported from `bgs`:

```python
from bgs import (
    VGSSSamplingStats,
    build_all_net_vgss,
    build_net_vgss,
    build_vgss_for_conductor,
    find_touching_nets,
    generate_all_bgs,
    generate_bgs,
    sample_on_vgss,
)

from input_formats import load_geometry
```

The original FRW functions remain in `FRW.random_walk`:

```python
from FRW.random_walk import (
    get_system_bounds,
    run_frw_geometry,
    sample_point_on_bgs,
    sample_point_on_maximal_cube,
)
```

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

## Original conductor-level BGS API

### Generate one BGS

```python
from bgs import generate_bgs
from input_formats import load_geometry

geometry = load_geometry("input/DIELECTRIC_INPUT_3CONDUCTORS.m")
conductors = list(geometry.conductors)
master = next(c for c in conductors if c.name == "C1")

result = generate_bgs(
    master,
    conductors,
    scale_factor=1.0,
    max_distance=10.0,
    touch_policy="skip",
)

surface = result.gaussian_surface
print(surface.bounds)
print(surface.area)
print(result.initial_extents)
print(result.final_extents)
```

`generate_bgs()` returns a `BGSGenerationResult` containing:

- `master`;
- `gaussian_surface`;
- `initial_extents`;
- `final_extents`;
- `cap_distance`;
- `pairwise_rows`;
- `used_isolated_fallback`.

`touch_policy="skip"` ignores directly touching blocks for this master's BGS. `touch_policy="error"` instead raises an error when direct contact is found.

### Generate a BGS for every conductor

```python
from bgs import generate_all_bgs

results = generate_all_bgs(
    conductors,
    scale_factor=1.0,
    max_distance=10.0,
    touch_policy="skip",
)

for result in results:
    print(result.master.name, result.gaussian_surface.area)
```

These functions are retained for code that needs only the original conductor-level Algorithm 4.2 behavior.

## Detect electrical nets

```python
from bgs import find_touching_nets

nets = find_touching_nets(conductors)

for net in nets:
    print(net.name, net.conductor_names)
```

For example, a touching chain `C1-C2-C3` is detected as one transitive electrical net even when `C1` and `C3` do not directly touch.

## Build the VGSS for a selected conductor's net

`build_vgss_for_conductor()` is the simplest entry point when a conductor name is known.

```python
from bgs import build_vgss_for_conductor

net_vgss = build_vgss_for_conductor(
    conductors,
    "C1",
    scale_factor=1.0,
    max_distance=10.0,
)

print(net_vgss.net.name)
print(net_vgss.net.conductor_names)
print(len(net_vgss.surfaces))
```

The returned `NetVGSS` contains:

- `net`: the detected `ConductorNet`;
- `bgs_results`: one `BGSGenerationResult` per conductor in the net;
- `sampling_context`: precomputed Algorithm 4.1 data;
- `surfaces`: a convenience property returning the net's BGS objects.

## Build the VGSS for an existing net

```python
from bgs import build_net_vgss, find_touching_nets

nets = find_touching_nets(conductors)
selected_net = nets[0]

net_vgss = build_net_vgss(
    selected_net,
    conductors,
    scale_factor=1.0,
    max_distance=10.0,
)
```

Use this form when net detection and net selection are handled separately.

## Build every net-level VGSS

```python
from bgs import build_all_net_vgss

all_net_vgss = build_all_net_vgss(
    conductors,
    scale_factor=1.0,
    max_distance=10.0,
)

for item in all_net_vgss:
    print(item.net.name, item.net.conductor_names)
```

`build_all_net_vgss()` returns a tuple containing one independent `NetVGSS` object for each detected net.

## Sample one accepted VGSS point

```python
import numpy as np

from bgs import build_vgss_for_conductor, sample_on_vgss

net_vgss = build_vgss_for_conductor(conductors, "C1")
rng = np.random.default_rng(1234)

r = sample_on_vgss(net_vgss.sampling_context, rng)
print(r)
```

`sample_on_vgss()` returns only the accepted coordinate:

```text
[x, y, z]
```

The returned value is a NumPy array with shape `(3,)`.

A different point is proposed for every call. Reuse the same NumPy generator when deterministic repeatability is required.

## Sample multiple points from every net

```python
import numpy as np

from bgs import build_all_net_vgss, sample_on_vgss

rng = np.random.default_rng(1234)
all_net_vgss = build_all_net_vgss(conductors)

sampled_points = {
    item.net.name: np.vstack(
        [
            sample_on_vgss(item.sampling_context, rng)
            for _ in range(200)
        ]
    )
    for item in all_net_vgss
}
```

## Custom importance function `p(r)`

The importance function and its upper bound are attached to the sampling context when the VGSS is built.

```python
import numpy as np

from bgs import build_vgss_for_conductor, sample_on_vgss


def F(r: np.ndarray) -> float:
    return 0.5


net_vgss = build_vgss_for_conductor(
    conductors,
    "C1",
    importance_function=F,
    upper_bound=1.0,
)

rng = np.random.default_rng(1234)
r = sample_on_vgss(net_vgss.sampling_context, rng)
```

The caller must currently ensure that:

```text
U > 0
0 <= p(r) <= U
```

for every possible VGSS point. An invalid `p(r)` or `U` can make the rejection sampler incorrect or prevent acceptance.

## Estimate `H`

`VGSSSamplingStats` records raw proposals and all rejection categories.

```python
import numpy as np

from bgs import VGSSSamplingStats, sample_on_vgss

rng = np.random.default_rng(1234)
stats = VGSSSamplingStats()

points = np.vstack(
    [
        sample_on_vgss(
            net_vgss.sampling_context,
            rng,
            stats=stats,
        )
        for _ in range(5000)
    ]
)

H = stats.estimate_H(net_vgss.sampling_context)
print(H)
print(stats.as_dict(net_vgss.sampling_context))
```

The estimator is:

```text
H = U * A_raw * accepted_count / proposal_count
```

where `A_raw` is the sum of the areas of the component BGSs stored in the sampling context.

For `p(r) = 1`, `H` estimates the actual exterior VGSS area after the geometry and multiplicity corrections.

## Visualize VGSS results

Requires the optional `plot` extra (`pip install -e ".[plot]"`).

```python
from bgs import plot_vgss_results

plot_vgss_results(
    conductors,
    all_net_vgss,
    sampled_points,
    save_path="output/all_net_vgss_samples.png",
    show=False,
)
```

The `sampled_points` dictionary must be keyed by net name, for example `Net1`.

# FRW integration

The FRW implementation keeps its original maximal-cube walk logic. Every walk
begins from a fresh point accepted by `SampleOnVGSS`. The first cube supports
both the original transition and derivative-importance stratification through
the same `sample_first_transition()` method.

`run_one_walk()` returns a compact `WalkOutcome` containing the terminal net,
first-hop weight, termination reason, and hop count. Its complete cube trace is
optional. The command-line runner consumes each outcome immediately, updates
the capacitance and metric accumulators, and then discards it.

## Run FRW for one selected net

```python
import numpy as np

from FRW.random_walk import run_frw_geometry
from bgs import build_vgss_for_conductor
from input_formats import load_geometry

geometry = load_geometry("input/DIELECTRIC_INPUT_3CONDUCTORS.m")
conductors = list(geometry.conductors)

master_vgss = build_vgss_for_conductor(
    conductors,
    "C1",
    scale_factor=1.25,
    max_distance=10.0,
)

result = run_frw_geometry(
    conductors,
    master_vgss,
    np.random.default_rng(1234),
    num_walks=5000,
    max_hops=100,
)

print(result["H"])
print(result["sampling_stats"])
```

The Python API defaults to `first_transition_mode="standard"`, preserving the
original behavior. Opt in to balanced stratification with:

```python
result = run_frw_geometry(
    conductors,
    master_vgss,
    np.random.default_rng(1234),
    num_walks=5000,
    first_transition_mode="stratified",
)
```

Stratified mode assigns half the walks to the inward geometric half of the
first cube and half to the outward half, so `num_walks` must be even. Within
each half, panels are sampled in proportion to the absolute normal derivative.
The weight retains the sampled panel's derivative sign; this does not assume a
homogeneous or sign-symmetric table.

## Run FRW for every detected net

```python
import numpy as np

from FRW.random_walk import run_frw_geometry
from bgs import build_all_net_vgss

rng = np.random.default_rng(1234)
all_net_vgss = build_all_net_vgss(conductors)

results = {
    item.net.name: run_frw_geometry(
        conductors,
        item,
        rng,
        num_walks=5000,
        max_hops=100,
    )
    for item in all_net_vgss
}
```

## FRW return value

`run_frw_geometry()` returns:

```python
{
    "H": H_estimate,
    "first_transition_mode": "standard_or_stratified",
    "sampling_stats": {
        "proposal_count": ...,
        "accepted_count": ...,
        "rejection_count": ...,
        "geometry_rejections": ...,
        "multiplicity_rejections": ...,
        "importance_rejections": ...,
        "acceptance_rate": ...,
        "upper_bound_U": ...,
        "total_component_area": ...,
        "H_estimate": ...,
    },
    "walks": [
        {
            "start_point": [x, y, z],
            "first_transition_mode": "standard_or_stratified",
            "first_transition_stratum": "inward_outward_or_null",
            "path_data": [
                {
                    "center": [x, y, z],
                    "a": maximal_cube_side_length,
                }
            ],
            "hit_conductor": conductor_name_or_ground,
        }
    ],
}
```

The compatibility `run_frw_geometry()` batch API still returns the stored
`walks` list shown above, including `path_data`. The CLI uses the compact online
path instead and does not export individual walks.

# Command-line usage

The module commands below work directly with the current source tree after installation.

## Generate one conductor-level BGS

```bash
python -m bgs.main input/DIELECTRIC_INPUT_3CONDUCTORS.m \
    --master C1 \
    --scale-factor 2
```

Save the plot without opening a window:

```bash
python -m bgs.main input/DIELECTRIC_INPUT_3CONDUCTORS.m \
    --master C1 \
    --scale-factor 2 \
    --save-plot output/C1_bgs.png \
    --no-show
```

## Generate BGSs for every conductor

```bash
python -m bgs.main input/DIELECTRIC_INPUT_3CONDUCTORS.m \
    --all \
    --scale-factor 2 \
    --save-plot output/all_bgs.png \
    --no-show
```

When `--save-plot` is omitted in `--all` mode, the combined plot is saved as `output/all_bgs.png`.

## Sample one selected net

`--net-master` identifies a conductor belonging to the desired net.

```bash
python -m bgs.vgss_cli input/DIELECTRIC_INPUT_3CONDUCTORS.m \
    --net-master C1 \
    --count 10 \
    --seed 1234 \
    --no-show
```

Save coordinates and a visualization:

```bash
python -m bgs.vgss_cli input/DIELECTRIC_INPUT_3CONDUCTORS.m \
    --net-master C1 \
    --count 1000 \
    --seed 1234 \
    --save-points output/vgss_points.csv \
    --save-plot output/net1_vgss.png \
    --no-show
```

Single-net CSV output contains:

```text
x,y,z
```

## Sample every detected net

```bash
python -m bgs.vgss_cli input/DIELECTRIC_INPUT_3CONDUCTORS.m \
    --all-nets \
    --count 1000 \
    --seed 1234 \
    --save-points output/all_net_vgss_points.csv \
    --save-plot output/all_net_vgss_samples.png \
    --no-show
```

With `--all-nets`, `--count` is the number of accepted points generated for each net. The CSV contains:

```text
net,x,y,z
```

The installed `vgss-sample` console command is equivalent to `python -m bgs.vgss_cli`.

## Run the integrated FRW

```bash
python -m FRW.cli input/DIELECTRIC_INPUT_3CONDUCTORS.m \
    --scale-factor 1.25 \
    --max-distance 10 \
    --seed 1234 \
    --workers 4 \
    --output output/frw_exported_data.json
```

Adaptive convergence is the CLI default. Each master row stops independently
when its row-max three-standard-error ratio is at most `0.03`, after at least
2,000 walks and 10 hits per target entry. The default hard limit is 1,000,000
walks, with checks every 200 walks. All counts are even so checks occur only
after complete inward/outward pairs.

Use the adaptive options to change those limits:

```bash
python -m FRW.cli input/DIELECTRIC_INPUT_3CONDUCTORS.m \
    --tolerance 0.02 \
    --min-walks 4000 \
    --max-walks 200000 \
    --check-every 400 \
    --min-hits 20
```

Supplying `--walks` selects fixed mode and disables convergence:

```bash
python -m FRW.cli input/DIELECTRIC_INPUT_3CONDUCTORS.m --walks 5000
```

If `--master` is omitted, the CLI runs every detected net and assembles the
full capacitance matrix. Rows are independent and run in separate worker
processes. The default is at most five workers; the actual number is limited to
the number of selected rows. For example, two rows use two workers, while
`--master Net1` runs sequentially without creating a process pool. Use
`--workers 1` to force sequential execution.

Each row has an independent deterministic RNG stream derived from `--seed`, so
parallel scheduling and one row stopping earlier do not change another row's
random sequence. Visualization is disabled by default; add `--visualization`
to display one saved walk for each master net after all calculations finish.

Capacitance and metrics are accumulated as each walk finishes. In stratified
mode, each inward/outward pair forms one statistical sample. The output reports
the pair count, online standard error, convergence ratio, stopping reason, and
actual walks used for each row. Matrix symmetry remains a diagnostic and is not
used as a stopping condition. A full path is retained only for the first walk
when visualization is requested.

The Python API remains backward compatible and defaults to the standard first
transition. To revert the CLI itself to standard sampling, change its
`first_transition_mode="stratified"` call to `"standard"`; the GGFT API and
standard implementation remain intact.

## Replay the three Excel comparison cases

Requires the optional `excel` extra (`pip install -e ".[excel]"`).

The comparison workbook records the exact windows and boundary conditions used
for `W0.14_S0.14`, `W0.14_S0.70`, and `W0.14_S2.8`. To replay its stratified
jobs with a `0.01` convergence tolerance:

```bash
python scripts/run_excel_cases.py
```

The runner adds `Current FRW Jobs`, `Current FRW Matrices`, and
`Current Comparisons` sheets. It writes the workbook and a JSON checkpoint
after every job. Press `Ctrl+C` to stop the current FRW process group; running
the same command again skips completed jobs and resumes at the interrupted job.
The original workbook is copied once to
`frw_raphael_window_comparison.original.xlsx`.

Useful controls include:

```bash
# Inspect every command without running FRW.
python scripts/run_excel_cases.py --list

# Run or resume one case.
python scripts/run_excel_cases.py --case W0.14_S0.70

# Recalculate completed jobs instead of skipping them.
python scripts/run_excel_cases.py --rerun
```

By default the runner keeps the recorded seed, scale factor, 20,000-walk
minimum, 20,000,000-walk limit, and 5,000-hop limit. It uses four row workers.
The output JSON files, logs, and checkpoint are stored under
`output/current_excel_runs`.

# Parameters

## `scale_factor`

Must be at least `1.0`.

For a non-isolated conductor, the BGS cap is calculated as:

```text
min(finite initial extents) * scale_factor
```

Each final directional extent is limited by that cap.

## `max_distance`

Must be positive.

It is used as a fallback extent only when no finite non-touching conductor constraint exists for the current master conductor.

## `abs_tol`

Controls floating-point comparisons for touching, overlap, surface coincidence, and candidate classification. The default is `1e-12`.

## `max_attempts`

`sample_on_vgss()` defaults to `10000` proposal attempts for one accepted point. It raises `RuntimeError` when no proposal is accepted within that limit.

# Tests

Run the complete test suite from the repository root:

```bash
pytest
```

The current suite covers:

- directional distances;
- BGS generation;
- raw MATLAB parser and semantic interpreter behavior;
- legacy parser compatibility;
- standard conductor/dielectric volume reconstruction;
- direct-touch skipping;
- overlap rejection;
- transitive net construction;
- Algorithm 4.1 geometry rejection;
- multiplicity rejection;
- all-net VGSS construction;
- VGSS visualization;
- `H` estimation;
- FRW integration and output compatibility.

# Current limitations

- Conductors must be axis-aligned cuboids.
- Overlapping conductor volumes are invalid input.
- Dielectric volumes/background permittivity are represented internally, but dielectric transition physics is not implemented yet.
- `EPS_*` boundary-permittivity metadata is parsed and retained but is not yet consumed by FRW.
- The caller must currently provide a valid positive `U` and a function satisfying `0 <= p(r) <= U`.
- The implementation prioritizes correctness and clarity over large-scale performance optimization.
