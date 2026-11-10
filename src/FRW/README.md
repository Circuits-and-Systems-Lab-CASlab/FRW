# `FRW`: floating random walk capacitance extraction

The walk engine, capacitance accumulation, convergence checks, finite-window
boundaries, metrics, visualization, and the `frw` command-line runner.
Back to the [main README](../../README.md).

Examples assume the project root as the working directory and an installed
package (`pip install -e .`) or `PYTHONPATH=src`.

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | `frw` command: argument parsing, per-row worker processes, convergence loop, JSON/CSV/SPICE/HTML export |
| `random_walk.py` | one walk (`run_one_walk`), the batch API `run_frw_geometry()`, VGSS start points, maximal cubes |
| `capacitance.py` | online capacitance-row accumulation (standard and stratified pairs), matrix assembly, CSV and SPICE writers |
| `convergence.py` | row-max 3SE convergence check with minimum-hit requirements |
| `boundary.py` | finite-window validation, Dirichlet absorption and Neumann reflection |
| `metrics.py` | hop counts, termination reasons and other walk diagnostics |
| `visualization.py` | walk traces, matplotlib 3D views, and the WebGL HTML export (no matplotlib needed) |

## Python API

The original FRW functions remain in `FRW.random_walk`:

```python
from FRW.random_walk import (
    get_system_bounds,
    run_frw_geometry,
    sample_point_on_bgs,
    sample_point_on_maximal_cube,
)
```

## FRW integration

The FRW implementation keeps its original maximal-cube walk logic. Every walk
begins from a fresh point accepted by `SampleOnVGSS`. The first cube supports
both the original transition and derivative-importance stratification through
the same `sample_first_transition()` method.

`run_one_walk()` returns a compact `WalkOutcome` containing the terminal net,
first-hop weight, termination reason, and hop count. Its complete cube trace is
optional. The command-line runner consumes each outcome immediately, updates
the capacitance and metric accumulators, and then discards it.

### Run FRW for one selected net

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

### Run FRW for every detected net

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

### FRW return value

`run_frw_geometry()` returns a dictionary shaped like this:

```text
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

## Command-line runner details

The main README lists every option. This section explains how the runner behaves.

### Run the integrated FRW

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
