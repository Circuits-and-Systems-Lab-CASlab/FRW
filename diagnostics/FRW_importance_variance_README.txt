FRW IMPORTANCE / VARIANCE DIAGNOSTIC
====================================

Files
-----
frw_importance_variance_diagnostic.py
frw_reference_cases.json

These are local diagnostics. Put both files under the ignored `diagnostics/`
directory of the VGSS repository. No production source file is modified.

Purpose
-------
The comparison workbook shows that the 10,000-walk runs can have very large
sampling instability, including positive mutual coefficients that every
reference solver reports as negative. The effect is strongest for the central
master conductor as its BGS area grows with spacing.

This diagnostic compares three unbiased estimators using the same numerical
n200 GGFT table:

1. baseline
   The current production first-panel probability and WVT weight.

2. abs_gradient_importance
   Samples the first panel proportional to |n dot grad(P)| and applies the
   exact importance correction.

3. sign_stratified_importance
   Uses the same importance distribution but balances positive and negative
   first-weight strata.

The script reports coefficient standard errors, variance-reduction factors,
required walk estimates for 1% relative standard error, paired differences,
and errors against Raphael, FasterCap, Caplet, and their per-entry median.
MATLAB results are intentionally absent from the reference JSON.

Install locally
---------------
From the VGSS repository root:

    mkdir -p diagnostics
    cp /mnt/c/Users/alexa/Downloads/frw_importance_variance_diagnostic.py diagnostics/
    cp /mnt/c/Users/alexa/Downloads/frw_reference_cases.json diagnostics/

Check that Git ignores them:

    git check-ignore -v \
        diagnostics/frw_importance_variance_diagnostic.py \
        diagnostics/frw_reference_cases.json

Compile check
-------------

    python3 -m py_compile diagnostics/frw_importance_variance_diagnostic.py

Quick smoke test
----------------

    python3 diagnostics/frw_importance_variance_diagnostic.py \
        --cases-root ../Conductor_Cases \
        --cases 5 \
        --masters Net3 \
        --walks-per-seed 200 \
        --seeds 1234 \
        --output-dir output/diagnostics/frw_importance_variance \
        --label smoke

Main diagnostic
---------------
This tests the central conductor row across all eight spreadsheet test cases:

    python3 diagnostics/frw_importance_variance_diagnostic.py \
        --cases-root ../Conductor_Cases \
        --cases 1 2 3 4 5 6 7 8 \
        --masters Net3 \
        --walks-per-seed 2500 \
        --seeds 1234 5678 9012 3456 \
        --scale-factor 1.25 \
        --max-distance 10 \
        --max-hops 200 \
        --hit-tolerance 1e-4 \
        --output-dir output/diagnostics/frw_importance_variance \
        --label all_cases_net3_4x2500 \
        2>&1 | tee output/diagnostics/frw_importance_variance_net3.log

Outputs
-------

output/diagnostics/frw_importance_variance/
    all_cases_net3_4x2500_summary.json
    all_cases_net3_4x2500_entries.csv
    all_cases_net3_4x2500_case_summary.csv
    all_cases_net3_4x2500_paired_differences.csv

Upload the four compact output files. The log is needed only if an error occurs.

Optional complete-matrix run
----------------------------
After interpreting the central-row test, the same script can run all masters.
For a bounded first full-matrix experiment, use cases 1, 5, and 8:

    python3 diagnostics/frw_importance_variance_diagnostic.py \
        --cases-root ../Conductor_Cases \
        --cases 1 5 8 \
        --masters all \
        --walks-per-seed 2500 \
        --seeds 1234 5678 9012 3456 \
        --output-dir output/diagnostics/frw_importance_variance \
        --label cases_1_5_8_all_masters_4x2500
