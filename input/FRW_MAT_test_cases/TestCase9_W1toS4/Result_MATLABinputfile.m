
--- BOUNDING BOX ---
Name  x-range (Δ)           y-range (Δ)       z-range (Δ) 
----------------------------------------------------------
C1    [0.0, 4.0] Δ=4.0          [0.0, 1.0] Δ=1.0      [0.0, 1.0] Δ=1.0
C2    [0.0, 4.0] Δ=4.0          [5.0, 6.0] Δ=1.0      [0.0, 1.0] Δ=1.0

--- FACE CONSISTENCY ---
  C1 | All 6 faces OK
  C2 | All 6 faces OK

--- GAP CHECK ---
  C1 y_max = 1.0
  C2 y_min = 5.0
  Gap      = 4.0 units

--- DIELECTRIC PANELS ---
  D1 | flat at y=3.0  OK
  D2 | flat at y=4.0  OK

Detected 2 conductors: C1 C2 

=== Iteration 1 ===
Total panels: 12
  Constructing A (12 x 12) in 3D...
Distance matrix time: 0.0003 seconds
A matrix construction time: 0.0002 seconds
Solver time for conductor 1: 0.0001 seconds
Solver time for conductor 2: 0.0000 seconds
C11 = 1.1420e-10 F
C22 = 1.1420e-10 F
Capacitance matrix (F) at iteration 1:
   1.0e-09 *

    0.1142   -0.0214
   -0.0214    0.1142

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 2 ===
Total panels: 48
  Constructing A (48 x 48) in 3D...
Distance matrix time: 0.0003 seconds
A matrix construction time: 0.0000 seconds
Solver time for conductor 1: 0.0001 seconds
Solver time for conductor 2: 0.0001 seconds
C11 = 1.2738e-10 F
C22 = 1.2738e-10 F
Auto-stop check: ratio = 1.077e-01
Capacitance matrix (F) at iteration 2:
   1.0e-09 *

    0.1274   -0.0261
   -0.0261    0.1274

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 3 ===
Total panels: 192
  Constructing A (192 x 192) in 3D...
Distance matrix time: 0.0004 seconds
A matrix construction time: 0.0001 seconds
Solver time for conductor 1: 0.0009 seconds
Solver time for conductor 2: 0.0006 seconds
C11 = 1.3461e-10 F
C22 = 1.3461e-10 F
Auto-stop check: ratio = 5.655e-02
Capacitance matrix (F) at iteration 3:
   1.0e-09 *

    0.1346   -0.0290
   -0.0290    0.1346

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 4 ===
Total panels: 768
  Constructing A (768 x 768) in 3D...
Distance matrix time: 0.0008 seconds
A matrix construction time: 0.0003 seconds
Solver time for conductor 1: 0.0072 seconds
Solver time for conductor 2: 0.0069 seconds
C11 = 1.3806e-10 F
C22 = 1.3806e-10 F
Auto-stop check: ratio = 2.640e-02
Capacitance matrix (F) at iteration 4:
   1.0e-09 *

    0.1381   -0.0305
   -0.0305    0.1381

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 5 ===
Total panels: 3072
  Constructing A (3072 x 3072) in 3D...
Distance matrix time: 0.0068 seconds
A matrix construction time: 0.0059 seconds
Solver time for conductor 1: 0.1264 seconds
Solver time for conductor 2: 0.1117 seconds
C11 = 1.3968e-10 F
C22 = 1.3968e-10 F
Auto-stop check: ratio = 1.226e-02
Capacitance matrix (F) at iteration 5:
   1.0e-09 *

    0.1397   -0.0311
   -0.0311    0.1397

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 6 ===
Total panels: 12288
  Constructing A (12288 x 12288) in 3D...
Distance matrix time: 0.0945 seconds
A matrix construction time: 0.0778 seconds
Solver time for conductor 1: 2.2568 seconds
Solver time for conductor 2: 2.3217 seconds
C11 = 1.4044e-10 F
C22 = 1.4044e-10 F
Auto-stop check: ratio = 5.789e-03

=====================================
Convergence achieved.
Iteration = 6
Ratio = 5.789e-03
Tolerance = 1.000e-02
=====================================
Total time for iteration 6 = 4.9770 seconds