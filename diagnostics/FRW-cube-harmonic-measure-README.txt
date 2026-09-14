FRW FULL CONTINUATION / HARMONIC-MEASURE DIAGNOSTIC
===================================================

Purpose
-------
This standalone diagnostic validates the complete repeated cube-walk against an
independent exact solution.  It does not use Raphael, FasterCap, Caplet, MATLAB,
BGS generation, capacitance assembly, or conductor geometry.

The outer domain is the cube [-1,1]^3.  For five interior source points, the
script computes the exact probability of ultimately hitting each of the six
outer faces with an independent separation-of-variables series.

It compares:
  1. numerical_n200: the bundled numerical GGFT transition table;
  2. analytical_n200: an independently generated analytical panel-integrated
     transition table with the same panel centres.

Both modes use repeated maximal cubes, panel-centre mapping, hit tolerance, and
terminal face classification.  This directly tests continuation paths and the
discontinuous face-indicator boundary condition.

No production source files are changed.

Installation
------------
From the VGSS repository root:

  mkdir -p diagnostics
  cp /mnt/c/Users/alexa/Downloads/frw_cube_harmonic_measure_diagnostic.py diagnostics/
  python3 -m py_compile diagnostics/frw_cube_harmonic_measure_diagnostic.py
  git check-ignore -v diagnostics/frw_cube_harmonic_measure_diagnostic.py

Smoke test
----------

  python3 diagnostics/frw_cube_harmonic_measure_diagnostic.py \
      --table src/GGFT/data/cube_tables_n200.npz \
      --walks-per-seed 500 \
      --seeds 1234 \
      --sources center off_axis near_face \
      --series-terms 41 \
      --reference-terms 201 \
      --hit-tolerance 1e-10 \
      --max-hops 1000 \
      --output-dir output/diagnostics/frw_cube_harmonic_measure \
      --label smoke

Main run
--------

  python3 diagnostics/frw_cube_harmonic_measure_diagnostic.py \
      --table src/GGFT/data/cube_tables_n200.npz \
      --walks-per-seed 20000 \
      --seeds 1234 5678 9012 3456 \
      --series-terms 41 \
      --reference-terms 201 \
      --hit-tolerance 1e-10 \
      --max-hops 1000 \
      --output-dir output/diagnostics/frw_cube_harmonic_measure \
      --label cube_harmonic_4x20000 \
      2>&1 | tee output/diagnostics/frw_cube_harmonic_measure.log

The main run performs 80,000 paired starts for each of five source points.
Because the domain is simple and the average walk has only a few hops, it is
much cheaper than a capacitance run.

Output files
------------
  cube_harmonic_4x20000_summary.json
  cube_harmonic_4x20000_entries.csv
  cube_harmonic_4x20000_start_summary.csv
  cube_harmonic_4x20000_paired_differences.csv

Interpretation
--------------
The most useful file is start_summary.csv.

* total_variation_error below roughly 0.005 at 80,000 walks is compatible with
  an accurate continuation chain, subject to the per-face z-scores.
* a coherent face error with |z| >= 3 in both numerical and analytical modes
  would implicate repeated panel-centre propagation or terminal classification.
* an error present only in numerical_n200 but absent in analytical_n200 would
  implicate accumulation of numerical GGFT probability-table error.
* agreement of both modes with the exact face probabilities would rule out the
  repeated continuation walk and leave Monte Carlo/reference-solver differences
  as the main explanation for the capacitance comparisons.
