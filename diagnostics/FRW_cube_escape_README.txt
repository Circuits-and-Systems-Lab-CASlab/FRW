FRW MAXIMAL-CUBE / ESCAPE-RADIUS DIAGNOSTIC

Purpose
-------
This standalone diagnostic tests:
1. invariance to using a smaller valid transition cube;
2. convergence with the numerical escape (infinity) radius;
3. exact L-infinity conductor clearance used for maximal cubes;
4. whether any generated cube overlaps a conductor interior.

It does not modify production source files and stores no walk histories.

Place frw_cube_escape_diagnostic.py in the ignored diagnostics/ directory.
Run from the VGSS repository root with the commands supplied in ChatGPT.

Expected output files
---------------------
<label>_summary.json
<label>_entries.csv
<label>_paired_differences.csv
<label>_mode_summary.csv
<label>_geometry_audit.csv
