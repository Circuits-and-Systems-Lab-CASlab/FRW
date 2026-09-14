FRW exact isolated-sphere capacitance diagnostic
================================================

Purpose
-------
This standalone test validates the complete FRW estimator against the exact
capacitance of an isolated conducting sphere:

    C = 4*pi*epsilon_0*a.

It tests the cubical BGS integral, first-hop WVT, repeated maximal-cube walk,
conductor hit classification, and free-space escape. It compares the bundled
numerical n200 GGFT table with an independently generated analytical n200 table.
No production source is modified and no external capacitance solver is used.

Install from the VGSS repository root
-------------------------------------
mkdir -p diagnostics
cp /mnt/c/Users/alexa/Downloads/frw_isolated_sphere_capacitance_diagnostic.py diagnostics/
python3 -m py_compile diagnostics/frw_isolated_sphere_capacitance_diagnostic.py
git check-ignore -v diagnostics/frw_isolated_sphere_capacitance_diagnostic.py

Smoke test
----------
python3 diagnostics/frw_isolated_sphere_capacitance_diagnostic.py \
  --table src/GGFT/data/cube_tables_n200.npz \
  --sphere-radius 1.0 \
  --bgs-half-size-factors 1.25 \
  --samples-per-face-per-seed 100 \
  --seeds 1234 \
  --series-terms 41 \
  --tail-radius-factor 50 \
  --hit-tolerance-factor 1e-6 \
  --max-hops 2000 \
  --output-dir output/diagnostics/frw_isolated_sphere \
  --label smoke

Focused main run
----------------
python3 diagnostics/frw_isolated_sphere_capacitance_diagnostic.py \
  --table src/GGFT/data/cube_tables_n200.npz \
  --sphere-radius 1.0 \
  --bgs-half-size-factors 1.25 \
  --samples-per-face-per-seed 5000 \
  --seeds 1234 5678 9012 3456 \
  --series-terms 41 \
  --tail-radius-factor 50 \
  --hit-tolerance-factor 1e-6 \
  --max-hops 2000 \
  --output-dir output/diagnostics/frw_isolated_sphere \
  --label sphere_r1_bgs1p25_4x5000 \
  2>&1 | tee output/diagnostics/frw_isolated_sphere.log

Optional BGS-invariance run
---------------------------
Run this only if the focused result shows a significant discrepancy:

python3 diagnostics/frw_isolated_sphere_capacitance_diagnostic.py \
  --table src/GGFT/data/cube_tables_n200.npz \
  --sphere-radius 1.0 \
  --bgs-half-size-factors 1.25 2.0 4.0 \
  --samples-per-face-per-seed 2500 \
  --seeds 1234 5678 9012 3456 \
  --series-terms 41 \
  --tail-radius-factor 50 \
  --hit-tolerance-factor 1e-6 \
  --max-hops 2000 \
  --output-dir output/diagnostics/frw_isolated_sphere \
  --label sphere_r1_scales_4x2500

Focused-run outputs
-------------------
sphere_r1_bgs1p25_4x5000_summary.json
sphere_r1_bgs1p25_4x5000_entries.csv
sphere_r1_bgs1p25_4x5000_face_contributions.csv
sphere_r1_bgs1p25_4x5000_paired_differences.csv
