
--- BOUNDING BOX ---
Name  x-range (Δ)             y-range (Δ)         z-range (Δ)   
-----------------------------------------------------------------
C1    [ -0.91,-0.77] Δ=0.14   [0.65,0.90] Δ=0.25   [0.0,2.0] Δ=2.0
C2    [ -0.63,-0.49] Δ=0.14   [0.65,0.90] Δ=0.25   [0.0,2.0] Δ=2.0
C3    [ -0.35,-0.21] Δ=0.14   [0.65,0.90] Δ=0.25   [0.0,2.0] Δ=2.0
C4    [ -0.07, 0.07] Δ=0.14   [0.65,0.90] Δ=0.25   [0.0,2.0] Δ=2.0
C5    [  0.21, 0.35] Δ=0.14   [0.65,0.90] Δ=0.25   [0.0,2.0] Δ=2.0

--- FACE CONSISTENCY ---
  C1 | All 6 faces OK
  C2 | All 6 faces OK
  C3 | All 6 faces OK
  C4 | All 6 faces OK
  C5 | All 6 faces OK

--- GAP CHECK ---
  Gap C1→C2 = 0.14  OK
  Gap C2→C3 = 0.14  OK
  Gap C3→C4 = 0.14  OK
  Gap C4→C5 = 0.14  OK

Detected 5 conductors: C1 C2 C3 C4 C5 

=== Iteration 1 ===
Total panels: 30
  Constructing A (30 x 30) in 3D...
Distance matrix time: 0.2716 seconds
A matrix construction time: 0.0008 seconds
Solver time for conductor 1: 0.2688 seconds
Solver time for conductor 2: 0.0001 seconds
Solver time for conductor 3: 0.0000 seconds
Solver time for conductor 4: 0.0000 seconds
Solver time for conductor 5: 0.0001 seconds
C11 = 3.5459e-11 F
C22 = 4.2948e-11 F
C33 = 4.1417e-11 F
C44 = 4.2948e-11 F
C55 = 3.5459e-11 F
Capacitance matrix (F) at iteration 1:
   1.0e-10 *

    0.3546   -0.1850    0.0451   -0.0590    0.0052
   -0.1850    0.4295   -0.1916    0.0711   -0.0590
    0.0451   -0.1916    0.4142   -0.1916    0.0451
   -0.0590    0.0711   -0.1916    0.4295   -0.1850
    0.0052   -0.0590    0.0451   -0.1850    0.3546

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 2 ===
Total panels: 120
  Constructing A (120 x 120) in 3D...
Distance matrix time: 0.0011 seconds
A matrix construction time: 0.0001 seconds
Solver time for conductor 1: 0.0002 seconds
Solver time for conductor 2: 0.0001 seconds
Solver time for conductor 3: 0.0001 seconds
Solver time for conductor 4: 0.0001 seconds
Solver time for conductor 5: 0.0001 seconds
C11 = 4.9417e-11 F
C22 = 5.7881e-11 F
C33 = 5.8662e-11 F
C44 = 5.7881e-11 F
C55 = 4.9417e-11 F
Auto-stop check: ratio = 3.388e-01
Capacitance matrix (F) at iteration 2:
   1.0e-10 *

    0.4942   -0.1969   -0.0702   -0.0363   -0.0271
   -0.1969    0.5788   -0.1686   -0.0589   -0.0363
   -0.0702   -0.1686    0.5866   -0.1686   -0.0702
   -0.0363   -0.0589   -0.1686    0.5788   -0.1969
   -0.0271   -0.0363   -0.0702   -0.1969    0.4942

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 3 ===
Total panels: 480
  Constructing A (480 x 480) in 3D...
Distance matrix time: 0.0020 seconds
A matrix construction time: 0.0002 seconds
Solver time for conductor 1: 0.0264 seconds
Solver time for conductor 2: 0.0026 seconds
Solver time for conductor 3: 0.0025 seconds
Solver time for conductor 4: 0.0025 seconds
Solver time for conductor 5: 0.0025 seconds
C11 = 6.1345e-11 F
C22 = 7.8454e-11 F
C33 = 7.8768e-11 F
C44 = 7.8454e-11 F
C55 = 6.1345e-11 F
Auto-stop check: ratio = 2.832e-01
Capacitance matrix (F) at iteration 3:
   1.0e-10 *

    0.6134   -0.3133   -0.0513   -0.0275   -0.0261
   -0.3133    0.7845   -0.2894   -0.0404   -0.0275
   -0.0513   -0.2894    0.7877   -0.2894   -0.0513
   -0.0275   -0.0404   -0.2894    0.7845   -0.3133
   -0.0261   -0.0275   -0.0513   -0.3133    0.6134

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 4 ===
Total panels: 1920
  Constructing A (1920 x 1920) in 3D...
Distance matrix time: 0.0041 seconds
A matrix construction time: 0.0025 seconds
Solver time for conductor 1: 0.0395 seconds
Solver time for conductor 2: 0.0375 seconds
Solver time for conductor 3: 0.0396 seconds
Solver time for conductor 4: 0.0346 seconds
Solver time for conductor 5: 0.0418 seconds
C11 = 6.9808e-11 F
C22 = 9.1232e-11 F
C33 = 9.1562e-11 F
C44 = 9.1232e-11 F
C55 = 6.9808e-11 F
Auto-stop check: ratio = 1.438e-01
Capacitance matrix (F) at iteration 4:
   1.0e-10 *

    0.6981   -0.3773   -0.0554   -0.0281   -0.0285
   -0.3773    0.9123   -0.3495   -0.0433   -0.0281
   -0.0554   -0.3495    0.9156   -0.3495   -0.0554
   -0.0281   -0.0433   -0.3495    0.9123   -0.3773
   -0.0285   -0.0281   -0.0554   -0.3773    0.6981

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 5 ===
Total panels: 7680
  Constructing A (7680 x 7680) in 3D...
Distance matrix time: 0.0376 seconds
A matrix construction time: 0.0296 seconds
Solver time for conductor 1: 0.7000 seconds
Solver time for conductor 2: 0.6778 seconds
Solver time for conductor 3: 0.6614 seconds
Solver time for conductor 4: 0.6763 seconds
Solver time for conductor 5: 0.6913 seconds
C11 = 7.6411e-11 F
C22 = 1.0251e-10 F
C33 = 1.0280e-10 F
C44 = 1.0251e-10 F
C55 = 7.6411e-11 F
Auto-stop check: ratio = 1.127e-01
Capacitance matrix (F) at iteration 5:
   1.0e-09 *

    0.0764   -0.0435   -0.0054   -0.0028   -0.0030
   -0.0435    0.1025   -0.0407   -0.0042   -0.0028
   -0.0054   -0.0407    0.1028   -0.0407   -0.0054
   -0.0028   -0.0042   -0.0407    0.1025   -0.0435
   -0.0030   -0.0028   -0.0054   -0.0435    0.0764

  Subdividing 3D panels into 4 sub-panels each...

=== Iteration 6 ===
Total panels: 30720
  Constructing A (30720 x 30720) in 3D...
Distance matrix time: 0.5809 seconds
A matrix construction time: 0.4268 seconds
Solver time for conductor 1: 25.6145 seconds
Solver time for conductor 2: 24.5664 seconds
Solver time for conductor 3: 24.8149 seconds
Solver time for conductor 4: 26.1060 seconds
Solver time for conductor 5: 25.3998 seconds
C11 = 8.0893e-11 F
C22 = 1.1029e-10 F
C33 = 1.1057e-10 F
C44 = 1.1029e-10 F
C55 = 8.0893e-11 F
Auto-stop check: ratio = 7.152e-02
Capacitance matrix (F) at iteration 6:
   1.0e-09 *

    0.0809   -0.0474   -0.0055   -0.0029   -0.0031
   -0.0474    0.1103   -0.0445   -0.0042   -0.0029
   -0.0055   -0.0445    0.1106   -0.0445   -0.0055
   -0.0029   -0.0042   -0.0445    0.1103   -0.0474
   -0.0031   -0.0029   -0.0055   -0.0474    0.0809