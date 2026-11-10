# GGFT transition tables for FRW

This project generates all reference-cube data needed by the FRW runtime:

- the transition probability table `P`,
- `dP/dxi_x`,
- `dP/dxi_y`,
- `dP/dxi_z`,
- and the normalized weight-value table `WVT = grad(P) / P`.

The source construction follows the four DeepRWCap GGFT modes (`GFT`, `WVTX`, `WVTY`, `WVTZ`). 
The surrounding dataset-generation program was replaced by a
small deterministic API that produces one table bundle for one dielectric
configuration.

The numerical implementation is matrix-free and supports a positive relative-
permittivity grid. It solves the variable-coefficient electrostatic operator on
the reference cube `[-1, 1]^3` and writes raw signed gradients. It does not turn
gradients into absolute-value CDFs.

## Project layout

```text
gft/
├── src/
│   ├── ggft_solver.h       # C++ API and result types
│   ├── ggft_solver.cpp     # P and x/y/z derivative solves
│   └── main.cpp            # deterministic command-line wrapper
├── tests/
│   └── test_tables.py
├── data/
│   ├── cube_tables_n201.bin
│   └── cube_tables_n201.npz
├── extract_binary.py       # binary -> validated NPZ + WVT
├── ggft.py                 # runtime sampling and first-hop weight API
├── generate_gft.sh
├── build_gft.sh
└── Makefile
```

## Build and generate

```bash
python3 -m pip install -r requirements.txt
./generate_gft.sh
```

Defaults:

```text
resolution      = 23
relative tol.   = 1e-10
max iterations  = 20000
block width     = 1
```

Custom values:

```bash
./generate_gft.sh 31 1e-10 30000 1
```

This creates:

```text
data/cube_tables_n31.bin
data/cube_tables_n31.npz
```

## C++ command-line API

```bash
make

./bin/ggft \
  --output data/cube_tables_n201.bin \
  --resolution 23 \
  --block-width 1 \
  --tolerance 1e-10 \
  --max-iterations 20000
```

For a dielectric configuration:

```bash
./bin/ggft \
  --output data/custom_tables.bin \
  --resolution 24 \
  --block-width 2 \
  --dielectric data/epsilon_r.bin
```

`epsilon_r.bin` is a raw little-endian/native `float64` array with shape:

```text
(N / block_width, N / block_width, N / block_width)
```

flattened with `x` as the fastest-changing coordinate. Every value must be
positive. When `--dielectric` is omitted, every value is `1.0`.

## C++ library API

```cpp
std::vector<double> epsilon_r;  // empty means homogeneous epsilon_r = 1

gft::GGFTSolver solver(
    /* resolution = */ 23,
    /* block_width = */ 1,
    epsilon_r
);

gft::GGFTResult tables = solver.solve_all();

// tables.probabilities
// tables.gradient_x
// tables.gradient_y
// tables.gradient_z

solver.write_binary("data/cube_tables_n201.bin", tables);
```

All four panel arrays have the same order:

```text
z-, z+, y-, y+, x-, x+
```

with `N*N` panels per face.

## NPZ contents

```text
resolution
block_width
reference_half_size
dielectric
probabilities       shape (6*N*N,)
panel_centers       shape (6*N*N, 3)
gradients           shape (6*N*N, 3)
wvt                 shape (6*N*N, 3)
```

For panel `j`:

```text
gradients[j] = [dP_j/dxi_x, dP_j/dxi_y, dP_j/dxi_z]
wvt[j]       = gradients[j] / probabilities[j]
```

## Runtime use for a general cube

Load once. `load_default()` finds the bundled table regardless of the working
directory; `CubeGFT.load(path)` loads any other `.npz` file:

```python
import numpy as np

from GGFT import CubeGFT

gft = CubeGFT.load_default()  # src/GGFT/data/cube_tables_n201.npz
rng = np.random.default_rng(1234)
center = np.array([0.0, 0.0, 0.0])  # current walk position
a = 0.5                             # half-size of the maximal cube
```

Sample a normal transition:

```python
r_prime, panel_index, transition_probability = gft.sample_next_point(
    current_point=center,
    cube_half_size=a,
    rng=rng,
)
```

For the first transition from the Gaussian surface:

```python
r0 = center                          # accepted VGSS point
normal = np.array([0.0, 0.0, 1.0])   # outward normal at r0

r1, panel_index, probability, omega = gft.sample_first_transition(
    current_point=r0,
    cube_half_size=a,
    surface_normal=normal,
    rng=rng,
)
```

This is the standard first transition. `mode="stratified"` (with `stratum`
and `allocation_probability`) samples one geometric half of the cube in
proportion to the absolute derivative; see `sample_first_transition()` in `ggft.py`.

The reference cube is `[-1, 1]^3`. For an axis-aligned physical cube with
centre `c` and half-size `a`:

```text
r' = c + a * xi'
```

The probability table is dimensionless and does not change with `a`. The
weight scales as:

```text
omega_j = -(normal . WVT[j]) / a
```

For a rotated cube, pass a `3x3` orthonormal local-to-world rotation matrix to
both runtime methods.

The complete VGSS contribution is still applied by the FRW code:

```text
Omega = H * epsilon(r0) / p_surface(r0) * omega
```

Only the first cube uses `omega`; later hops use the probability table only.

## Validation

```bash
make test
```

The tests check:

- `sum(P) = 1`,
- `sum(dP/dxi_x) = sum(dP/dxi_y) = sum(dP/dxi_z) = 0`,
- inverse scaling of `omega` with physical cube half-size,
- sampled points lie on the physical cube boundary.

## Binary format

The binary begins with a versioned header:

```text
char magic[8] = "GGFTTBL1"
uint32 version
uint32 resolution
uint32 block_width
uint32 reserved
double reference_half_size
uint64 dielectric_count
uint64 panel_count
```

followed by `float64` arrays:

```text
dielectric[dielectric_count]
probabilities[panel_count]
gradient_x[panel_count]
gradient_y[panel_count]
gradient_z[panel_count]
```
