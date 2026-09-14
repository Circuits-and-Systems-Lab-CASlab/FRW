FRW face-stratified BGS/VGSS diagnostic
========================================

Purpose
-------
This standalone diagnostic tests the remaining Gaussian-surface part of the
FRW estimator. It uses equal sampling on every BGS face and applies the exact
face area explicitly:

    C_j = epsilon_0 * sum_f A_f * mean_f(omega * I[target=j]).

Corresponding faces at different BGS scale factors share the same face
coordinates and path random numbers. No production source file is modified.

Install
-------
From the VGSS repository root:

    mkdir -p diagnostics
    cp /mnt/c/Users/alexa/Downloads/frw_bgs_face_diagnostic.py diagnostics/
    python3 -m py_compile diagnostics/frw_bgs_face_diagnostic.py
    git check-ignore -v diagnostics/frw_bgs_face_diagnostic.py

Smoke test
----------

    python3 diagnostics/frw_bgs_face_diagnostic.py \
        ../Conductor_Cases/Testcase10_W0.14toS1.4/MATLABinputfile.m \
        --master Net3 \
        --bgs-scale-factors 1.0 1.25 \
        --samples-per-face-per-seed 50 \
        --seeds 1234 \
        --kill-multiplier 500 \
        --max-hops 500 \
        --hit-tolerance 1e-4 \
        --reference-json diagnostics/frw_reference_cases.json \
        --case-number 7 \
        --output-dir output/diagnostics/frw_bgs_face \
        --label smoke

Main run
--------

    python3 diagnostics/frw_bgs_face_diagnostic.py \
        ../Conductor_Cases/Testcase10_W0.14toS1.4/MATLABinputfile.m \
        --master Net3 \
        --bgs-scale-factors 1.0 1.25 2.0 \
        --samples-per-face-per-seed 1000 \
        --seeds 1234 5678 9012 3456 \
        --max-distance 10 \
        --kill-multiplier 500 \
        --max-hops 500 \
        --hit-tolerance 1e-4 \
        --reference-json diagnostics/frw_reference_cases.json \
        --case-number 7 \
        --output-dir output/diagnostics/frw_bgs_face \
        --label case7_net3_bgs_face_4x1000 \
        2>&1 | tee output/diagnostics/frw_bgs_face_case7.log

Outputs
-------
case7_net3_bgs_face_4x1000_summary.json
case7_net3_bgs_face_4x1000_entries.csv
case7_net3_bgs_face_4x1000_face_contributions.csv
case7_net3_bgs_face_4x1000_paired_differences.csv
case7_net3_bgs_face_4x1000_bgs_geometry.csv

The face-contributions file is especially important because it shows which BGS
face and normal contribute to each capacitance coefficient.
