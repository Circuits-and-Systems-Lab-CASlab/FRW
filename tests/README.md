# Testing and Quality Assurance (QA) Guide

This project includes a comprehensive, two-tiered testing framework designed to ensure mathematical correctness and evaluate large-scale algorithmic convergence. 

1. **The Pytest Suite:** Core unit and integration tests that validate the fundamental mathematical rules, geometric intersections, and logic.
2. **The QA Convergence Tester (`qa_tester.py`):** A high-performance, multi-core batch processing script designed to test the Floating Random Walk (FRW) algorithm across millions of iterations to prove statistical convergence.

---

## 1. Unit & Integration Testing (Pytest)

The `tests/` directory contains automated checks for the individual components of the Virtual Gaussian Surface Set (VGSS) and FRW pipeline. 

### What the Pytest Suite Covers

* **`test_parser.py`**: Ensures MATLAB `.m` geometry files are correctly parsed into Python `Conductor` objects, accurately capturing bounds and relative permittivity.
* **`test_topology.py`**: Validates the detection of electrical nets, ensuring that face, edge, and corner contacts are correctly grouped, while overlapping geometries are flagged as invalid.
* **`test_distances.py`**: Verifies the mathematical calculations of 3D directional distances between cuboids (Algorithm 4.2 definitions).
* **`test_gaussian_surface.py`**: Tests the generation of Boundary Gaussian Surfaces (BGS), ensuring that directly touching blocks are skipped while non-touching constraints limit the BGS extents.
* **`test_vgss_geometry.py`**: Enforces strict geometric rules, ensuring that proposed points inside other surfaces or on opposite normals are correctly rejected.
* **`test_sampling.py` & `test_vgss_integration.py`**: Validates Algorithm 4.1 (SampleOnVGSS), ensuring points are correctly sampled, rejected based on multiplicity, and assigned to valid surface faces.
* **`test_frw_integration.py`**: End-to-end checks to guarantee that the Random Walk correctly initializes on the VGSS and tracks its paths, hops, and terminal targets.

### How to Run the Pytest Suite

Ensure your virtual environment is active, then simply run `pytest` from the project root:

```bash
# Run all tests
pytest

# Run tests with detailed console output
pytest -v

# Run a specific test file
pytest tests/test_topology.py
```

## 2. Automated QA & Convergence Testing (`qa_tester.py`)

While Pytest guarantees the code works, the Automated QA Tester guarantees the math converges.

Because Monte Carlo simulations like the Floating Random Walk require testing across thousands of walks to stabilize (e.g., estimating the $S_g$ integral), `qa_tester.py` acts as an automated, multi-core orchestrator.

### Features
* **Parallel Processing:** Utilizes Python's `concurrent.futures` standard library to execute dozens of complex random walk simulations simultaneously across your CPU cores.
* **Batch Execution:** Automatically parses and tests multiple `.m` files from the `input/` directory without manual intervention.
* **Parameter Sweeping:** Runs an automated grid matrix testing different Algorithm 4.2 scale factors, max distances, and predefined RNG seeds.
* **Convergence Tracking:** Iteratively scales up the number of proposed points ($N = 100, 1000, 5000, 10000$) on the same geometry to track the convergence curve of the $S_g$ metric.
* **Headless Visualization:** Automatically renders and saves 3D walk visualizations as `.png` files without blocking execution with interactive windows.
* **Excel-Native Export:** Aggregates execution data (hit rates, hop standard deviations, rejection counts, and final estimates) into a flat `qa_convergence_results.csv` format that opens natively in Microsoft Excel.

### CLI Flags and Configuration
You can fully control the QA script from your terminal. Use `python qa_tester.py --help` for a quick reference.

#### Target Overrides (What to Test)
* `--files`: Specify exact `.m` files to test (e.g., `--files INPUTFILE_01.m`). If omitted, it runs every `.m` file in the `input/` directory.
* `--seeds`: Override the deterministic RNG seeds (e.g., `--seeds 42 99`).
* `--scale-factors`: Override the Algorithm 4.2 bounding scale factors (e.g., `--scale-factors 1.0 1.5 2.0`).
* `--max-dists`: Override the maximum fallback distance for isolated conductors.
* `--walks`: Define the sequence of walk counts ($N$) to test convergence (e.g., `--walks 100 5000 50000`).

#### Optimization Flags (How to Test)
* `--workers`: Control how many CPU cores to use. Defaults to half your available cores to prevent system freeze. Use `--workers 1` for a background run.
* `--no-plots`: Highly recommended for massive batch runs. Skips generating 3D images, which dramatically speeds up testing time.
* `--verbose`: Stops swallowing the pipeline print statements and floods the terminal with raw output (useful for debugging hangs).

### Example Workflows

#### 1. The Quick Diagnostic Check
Test a single file on a single core, generating plots for a low number of walks to ensure the geometry parses and walks correctly.
```bash
python qa_tester.py --files INPUTFILE_01.m --walks 100 500 --workers 1