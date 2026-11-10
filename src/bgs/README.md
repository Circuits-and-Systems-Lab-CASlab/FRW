# `bgs`: Boundary Gaussian Surfaces and VGSS sampling

Boundary Gaussian Surface generation (Algorithm 4.2), touch-connected net
detection, net-level Virtual Gaussian Surface Sets, and `SampleOnVGSS`
(Algorithm 4.1). Back to the [main README](../../README.md).

Examples assume the project root as the working directory and an installed
package (`pip install -e .`) or `PYTHONPATH=src`.

## What the sampler implements

- one cuboid BGS for each conductor block;
- one independent VGSS sampling context for each detected net;
- area-weighted BGS and face selection;
- geometric rejection of buried and internal interfaces;
- the `1 / n_c(r)` multiplicity correction;
- configurable importance rejection using `p(r) / U`;
- estimation of the normalization integral `H` from sampling statistics.

The default importance function is:

```text
p(r) = 1
U = 1
```

## Important touching-conductor rule

Electrical nets are found as connected components. Therefore, if `C1` touches `C2` and `C2` touches `C3`, all three conductors belong to the same net even when `C1` does not directly touch `C3`.

BGS generation applies a more local rule:

- a conductor directly touching the current master conductor is ignored for that master's BGS calculation;
- a same-net conductor that does not directly touch the current master is still considered;
- conductors outside the net are also considered;
- overlapping conductor volumes are rejected.

The implementation therefore ignores only the directly touching conductor or conductors, not the complete net.

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

sample = sample_on_vgss(net_vgss.sampling_context, rng)
print(sample.point)
```

`sample_on_vgss()` returns a `VGSSSample` with these fields:

| Field | Meaning |
|---|---|
| `point` | accepted coordinate `[x, y, z]`, a NumPy array with shape `(3,)` |
| `normal` | outward unit normal of the face the point was sampled on |
| `source_surface_index` | index of the BGS (within the net's VGSS) that produced the point |
| `face_direction` | name of that face: `PX`, `PY`, `PZ`, `NX`, `NY` or `NZ` |

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
            sample_on_vgss(item.sampling_context, rng).point
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
sample = sample_on_vgss(net_vgss.sampling_context, rng)
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
        ).point
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

## Command-line tools

`bgs-generate` is `python -m bgs.main`, and `vgss-sample` is
`python -m bgs.vgss_cli`. The main README lists their options.

### Generate one conductor-level BGS

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

### Generate BGSs for every conductor

```bash
python -m bgs.main input/DIELECTRIC_INPUT_3CONDUCTORS.m \
    --all \
    --scale-factor 2 \
    --save-plot output/all_bgs.png \
    --no-show
```

When `--save-plot` is omitted in `--all` mode, the combined plot is saved as `output/all_bgs.png`.

### Sample one selected net

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

### Sample every detected net

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

## Parameters

### `scale_factor`

Must be at least `1.0`.

For a non-isolated conductor, the BGS cap is calculated as:

```text
min(finite initial extents) * scale_factor
```

Each final directional extent is limited by that cap.

### `max_distance`

Must be positive.

It is used as a fallback extent only when no finite non-touching conductor constraint exists for the current master conductor.

### `abs_tol`

Controls floating-point comparisons for touching, overlap, surface coincidence, and candidate classification. The default is `1e-12`.

### `max_attempts`

`sample_on_vgss()` defaults to `10000` proposal attempts for one accepted point. It raises `RuntimeError` when no proposal is accepted within that limit.
