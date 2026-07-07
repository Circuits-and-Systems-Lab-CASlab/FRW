# Homogeneous Cube GFT — C++ solver and Python extractor

This folder generates a cube Green Function Table for FRW without dielectric
interfaces.

The numerical solve stays in C++. Python does not solve the finite-difference
system; it only extracts the generated binary, builds the normalized panel
coordinates, validates the probabilities and writes the `.npz` consumed by
`ggft.py`.

## Included files

```text
gft/
├── src/
│   ├── ggft_solver.h
│   ├── ggft_solver.cpp
│   └── main.cpp
├── bin/
├── data/
│   ├── cube_gft_n23.bin
│   └── cube_gft_n23.npz
├── tests/
│   └── test_extract_binary.py
├── ggft.py
├── extract_binary.py
├── generate_gft.sh
├── Makefile
├── requirements.txt
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

The random dielectric generator, OpenMP dataset loop, training code, neural
networks, CUDA code and C++ inference code from DeepRWCap are intentionally not
included.

## Requirements

- Linux or WSL
- `g++` with C++17 support
- GNU `make`
- Python 3
- NumPy

Install the Python requirement:

```bash
python3 -m pip install -r requirements.txt
```

No Eigen installation is required. The homogeneous solver uses a small
matrix-free conjugate-gradient implementation.

## Generate the binary and NPZ

From inside `gft/`:

```bash
./generate_gft.sh
```

The default resolution is 23. To use another resolution:

```bash
./generate_gft.sh 31
```

The script performs these steps:

```text
C++ FDM solve
    ↓
data/cube_gft_n23.bin
    ↓ extract_binary.py
data/cube_gft_n23.npz
```

## Run the stages separately

Compile:

```bash
make
```

Generate only the DeepRWCap-style binary:

```bash
./bin/ggft_homogeneous data/cube_gft_n23.bin 23
```

Convert the binary to NPZ:

```bash
python3 extract_binary.py \
    data/cube_gft_n23.bin \
    data/cube_gft_n23.npz
```

## Binary layout

The output follows the part of DeepRWCap's dataset layout needed for one sample:

```text
double N
double block_width
double dielectric[N^3]       # all values are 1.0
double probability[6*N^2]
```

There is no structure-data section because this generator contains no
dielectric interfaces.

The face order is:

```text
z-, z+, y-, y+, x-, x+
```

## Use the generated table

```python
from pathlib import Path

import numpy as np

from ggft import CubeGFT


gft = CubeGFT.load(Path("data/cube_gft_n23.npz"))
rng = np.random.default_rng(1234)

current_point = np.array([2.0, 3.0, 1.0])
cube_half_size = 0.75
next_point = gft.sample_next_point(
    current_point=current_point,
    cube_half_size=cube_half_size,
    rng=rng,
)
```

`cube_half_size` is half the full side length of the maximal cube.

## Tests

```bash
python3 -m pytest -q
```

The tests compile the C++ solver, generate a small binary, convert it to NPZ,
load it through `CubeGFT`, and verify normalization and cube symmetry.

## Current limitation

Use this table only when the maximal transition cube does not cross a
dielectric interface. Dielectric support can be added later without changing
the `CubeGFT` loading API.
