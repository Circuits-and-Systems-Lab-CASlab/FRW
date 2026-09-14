FRW panel-centre and hit-tolerance diagnostic
=============================================

Purpose
-------
This standalone local diagnostic tests two approximations that remain after the
previous sign, BGS, GGFT/WVT, and first-transition tests:

1. Transition-location discretization: analytical panel grids n=100, n=200,
   and n=400 are compared with common spatially coupled random samples.
2. Conductor hit tolerance: the same analytical n=200 paths are evaluated at
   1e-3, 1e-4, 1e-5, and 1e-6.

The script does not modify production source and stores no path histories.

Install locally
---------------
From the VGSS repository root:

    mkdir -p diagnostics
    cp /mnt/c/Users/alexa/Downloads/frw_panel_hit_diagnostic.py diagnostics/
    python3 -m py_compile diagnostics/frw_panel_hit_diagnostic.py
    git check-ignore -v diagnostics/frw_panel_hit_diagnostic.py

Smoke test
----------

    python3 diagnostics/frw_panel_hit_diagnostic.py \
        ../Conductor_Cases/Testcase10_W0.14toS1.4/MATLABinputfile.m \
        --master Net3 \
        --walks-per-seed 200 \
        --seeds 1234 \
        --resolutions 100 200 400 \
        --tolerances 1e-3 1e-4 1e-5 1e-6 \
        --fixed-tolerance 1e-4 \
        --max-hops 500 \
        --reference-json diagnostics/frw_reference_cases.json \
        --case-number 7 \
        --output-dir output/diagnostics/frw_panel_hit \
        --label smoke

Main paired run
---------------

    python3 diagnostics/frw_panel_hit_diagnostic.py \
        ../Conductor_Cases/Testcase10_W0.14toS1.4/MATLABinputfile.m \
        --master Net3 \
        --walks-per-seed 5000 \
        --seeds 1234 5678 9012 3456 \
        --resolutions 100 200 400 \
        --tolerances 1e-3 1e-4 1e-5 1e-6 \
        --fixed-tolerance 1e-4 \
        --series-terms 41 \
        --scale-factor 1.25 \
        --max-distance 10 \
        --max-hops 500 \
        --reference-json diagnostics/frw_reference_cases.json \
        --case-number 7 \
        --output-dir output/diagnostics/frw_panel_hit \
        --label case7_net3_4x5000 \
        2>&1 | tee output/diagnostics/frw_panel_hit_case7.log

Outputs
-------

    case7_net3_4x5000_summary.json
    case7_net3_4x5000_entries.csv
    case7_net3_4x5000_paired_differences.csv
    case7_net3_4x5000_mode_summary.csv

Interpretation
--------------
Resolution comparisons:

    resolution_n200_minus_n100
    resolution_n400_minus_n200
    resolution_n400_minus_n100

If n=200 and n=400 differ by much less than the FRW/golden-tool mismatch and
have paired |z| below about 2, panel-centre resolution is not the main cause.
A coherent n=100 -> n=200 -> n=400 trend toward the golden reference indicates
transition-location discretization.

Tolerance comparisons:

    tolerance_1em04_minus_1em03
    tolerance_1em05_minus_1em04
    tolerance_1em06_minus_1em05
    tolerance_1em06_minus_1em04

If 1e-4, 1e-5, and 1e-6 are effectively identical, the production 1e-4 hit
threshold is converged. A monotonic statistically significant shift as the
threshold decreases identifies hit-tolerance bias.

The analytical transition law is used deliberately so the test isolates panel
placement and hit tolerance rather than re-testing the already validated n200
GGFT/WVT table.
