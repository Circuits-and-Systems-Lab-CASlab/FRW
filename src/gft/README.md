# Homogeneous Cube GFT — C++ solver and Python extractor

This folder generates a cube Green Function Table for FRW without dielectric
interfaces.

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

## Requirements

- GNU `make`
- Python 3
- NumPy

Install the Python requirement:
```bash
python3 -m pip install -r requirements.txt
```

## Generate the binary and NPZ

From inside `gft/`:
```bash
./generate_gft.sh
```

The default resolution is 100. To use another resolution:
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
python3 extract_binary.py data/cube_gft_n23.bin data/cube_gft_n23.npz
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

## Current limitation

Use this table only when the maximal transition cube does not cross a
dielectric interface. Dielectric support can be added later without changing
the `CubeGFT` loading API.
