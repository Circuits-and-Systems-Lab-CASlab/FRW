FRW manufactured point-source flux diagnostic
============================================

Purpose
-------
This local diagnostic validates the complete BGS/WVT first-hop surface
integral against an exact manufactured solution.

For phi(r) = 1 / ||r-r_s||, with r_s strictly inside the master conductor,

the exact outward flux of -grad(phi) through every enclosing closed Gaussian
surface is 4*pi = 12.566370614359...

The script compares:
1. deterministic Gauss-Legendre integration of the exact flux density;
2. Monte Carlo integration of the exact flux density on the same BGS samples;
3. the production n200 WVT estimator omega * phi(r1).

It uses three source locations inside C3 (one centered and two off-center), so
symmetry cannot hide a face-order or normal-direction error. It does not run
continuation walks and does not depend on Raphael, FasterCap, Caplet, or MATLAB.
No production source files are modified.

Installation
------------
From the VGSS repository root:

    mkdir -p diagnostics
    cp /mnt/c/Users/alexa/Downloads/frw_manufactured_flux_diagnostic.py diagnostics/
    python3 -m py_compile diagnostics/frw_manufactured_flux_diagnostic.py
    git check-ignore -v diagnostics/frw_manufactured_flux_diagnostic.py

Smoke test
----------

    python3 diagnostics/frw_manufactured_flux_diagnostic.py \
        ../Conductor_Cases/FRW_MAT_test_cases/Testcase10_W0.14toS1.4/MATLABinputfile.m \
        --master Net3 \
        --bgs-scale-factors 1.0 1.25 \
        --samples-per-face-per-seed 200 \
        --seeds 1234 \
        --quadrature-order 32 \
        --output-dir output/diagnostics/frw_manufactured_flux \
        --label smoke

The smoke run is only an execution check. Its WVT result can be very noisy.
The deterministic_exact values should already be extremely close to 4*pi.

Main run
--------
This is a first-hop-only test and is much cheaper than a full FRW run.

    python3 diagnostics/frw_manufactured_flux_diagnostic.py \
        ../Conductor_Cases/FRW_MAT_test_cases/Testcase10_W0.14toS1.4/MATLABinputfile.m \
        --master Net3 \
        --bgs-scale-factors 1.0 1.25 2.0 \
        --samples-per-face-per-seed 10000 \
        --seeds 1234 5678 9012 3456 \
        --quadrature-order 40 \
        --output-dir output/diagnostics/frw_manufactured_flux \
        --label case7_net3_manufactured_flux_4x10000 \
        2>&1 | tee output/diagnostics/frw_manufactured_flux_case7.log

Outputs
-------

    case7_net3_manufactured_flux_4x10000_summary.json
    case7_net3_manufactured_flux_4x10000_entries.csv
    case7_net3_manufactured_flux_4x10000_face_contributions.csv

Most important fields in entries.csv
------------------------------------
- deterministic_exact: direct quadrature result; should equal 4*pi.
- numerical_wvt: complete numerical first-hop/BGS estimate.
- numerical_rel_error_vs_4pi: total end-to-end relative error.
- numerical_minus_exact_mc: paired WVT-minus-exact difference.
- paired_difference_within_SE and paired_difference_between_SE: uncertainty.

Interpretation
--------------
- Errors around 0-1% and compatible with the paired uncertainty rule out a
  missing BGS area, normal, WVT sign, or first-hop scale factor.
- A stable 3-7% error for all three source locations and all BGS scales would
  identify a genuine normalization problem in the first-hop surface integral.
- A source-dependent or face-localized error would indicate a panel/normal
  mapping or spatial quadrature problem.
