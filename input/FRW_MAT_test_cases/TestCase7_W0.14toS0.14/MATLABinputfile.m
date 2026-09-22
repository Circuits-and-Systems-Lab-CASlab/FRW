% ==============================
% Geometry of rectangular conductors (cuboids)
% Each conductor is defined by 6 faces
% Each face has 4 corner points (x, y, z)
%
% Axis convention:
%   x = width direction  (changes per conductor)
%   y = height direction (fixed: 0.65 to 0.90, Δy = 0.25)
%   z = depth direction  (fixed: 0.00 to 2.00, Δz = 2.00)
%
% Conductor layout (width=0.14, gap=0.14):
%   C1: x = -0.91 → -0.77
%   C2: x = -0.63 → -0.49
%   C3: x = -0.35 → -0.21
%   C4: x = -0.07 →  0.07
%   C5: x =  0.21 →  0.35
%
% Face row order (same for all conductors):
%   Row 1 — Bottom face (y = 0.65)
%   Row 2 — Top    face (y = 0.90)
%   Row 3 — Front  face (z = 0.00)
%   Row 4 — Back   face (z = 2.00)
%   Row 5 — Left   face (x = x_min)
%   Row 6 — Right  face (x = x_max)
% ==============================
%% -------- Conductor C1 --------
% x: -0.91 → -0.77
C1 = [
% Bottom face (y = 0.65)
   -0.91  0.65  0     -0.77  0.65  0     -0.77  0.65  2     -0.91  0.65  2;
% Top    face (y = 0.90)
   -0.91  0.90  0     -0.77  0.90  0     -0.77  0.90  2     -0.91  0.90  2;
% Front  face (z = 0)
   -0.91  0.65  0     -0.77  0.65  0     -0.77  0.90  0     -0.91  0.90  0;
% Back   face (z = 2)
   -0.91  0.65  2     -0.77  0.65  2     -0.77  0.90  2     -0.91  0.90  2;
% Left   face (x = -0.91)
   -0.91  0.65  0     -0.91  0.65  2     -0.91  0.90  2     -0.91  0.90  0;
% Right  face (x = -0.77)
   -0.77  0.65  0     -0.77  0.65  2     -0.77  0.90  2     -0.77  0.90  0;
];
%% -------- Conductor C2 --------
% x: -0.63 → -0.49  (gap from C1 = 0.14)
C2 = [
% Bottom face (y = 0.65)
   -0.63  0.65  0     -0.49  0.65  0     -0.49  0.65  2     -0.63  0.65  2;
% Top    face (y = 0.90)
   -0.63  0.90  0     -0.49  0.90  0     -0.49  0.90  2     -0.63  0.90  2;
% Front  face (z = 0)
   -0.63  0.65  0     -0.49  0.65  0     -0.49  0.90  0     -0.63  0.90  0;
% Back   face (z = 2)
   -0.63  0.65  2     -0.49  0.65  2     -0.49  0.90  2     -0.63  0.90  2;
% Left   face (x = -0.70)
   -0.63  0.65  0     -0.63  0.65  2     -0.63  0.90  2     -0.63  0.90  0;
% Right  face (x = -0.56)
   -0.49  0.65  0     -0.49  0.65  2     -0.49  0.90  2     -0.49  0.90  0;
];
%% -------- Conductor C3 --------
% x: -0.35 → -0.21  (gap from C2 = 0.14)
% FIX: Front face first point corrected from z=2 → z=0
C3 = [
% Bottom face (y = 0.65)
   -0.35  0.65  0     -0.21  0.65  0     -0.21  0.65  2     -0.35  0.65  2;
% Top    face (y = 0.90)
   -0.35  0.90  0     -0.21  0.90  0     -0.21  0.90  2     -0.35  0.90  2;
% Front  face (z = 0)
   -0.35  0.65  0     -0.21  0.65  0     -0.21  0.90  0     -0.35  0.90  0;
% Back   face (z = 2)
   -0.35  0.65  2     -0.21  0.65  2     -0.21  0.90  2     -0.35  0.90  2;
% Left   face (x = -0.42)
   -0.35  0.65  0     -0.35  0.65  2     -0.35  0.90  2     -0.35  0.90  0;
% Right  face (x = -0.28)
   -0.21  0.65  0     -0.21  0.65  2     -0.21  0.90  2     -0.21  0.90  0;
];
%% -------- Conductor C4 --------
% x: -0.07 → 0.07  (gap from C3 = 0.14)
C4 = [
% Bottom face (y = 0.65)
   -0.07  0.65  0      0.07  0.65  0      0.07  0.65  2     -0.07  0.65  2;
% Top    face (y = 0.90)
   -0.07  0.90  0      0.07  0.90  0      0.07  0.90  2     -0.07  0.90  2;
% Front  face (z = 0)
   -0.07  0.65  0      0.07  0.65  0      0.07  0.90  0     -0.07  0.90  0;
% Back   face (z = 2)
   -0.07  0.65  2      0.07  0.65  2      0.07  0.90  2     -0.07  0.90  2;
% Left   face (x = -0.14)
   -0.07  0.65  0     -0.07  0.65  2     -0.07  0.90  2     -0.07  0.90  0;
% Right  face (x = 0.00)
    0.07  0.65  0      0.07  0.65  2      0.07  0.90  2      0.07  0.90  0;
];
%% -------- Conductor C5 --------
% x: 0.21 → 0.28  (gap from C4 = 0.14)
C5 = [
% Bottom face (y = 0.65)
    0.21  0.65  0      0.35  0.65  0      0.35  0.65  2      0.21  0.65  2;
% Top    face (y = 0.90)
    0.21  0.90  0      0.35  0.90  0      0.35  0.90  2      0.21  0.90  2;
% Front  face (z = 0)
    0.21  0.65  0      0.35  0.65  0      0.35  0.90  0      0.21  0.90  0;
% Back   face (z = 2)
    0.21  0.65  2      0.35  0.65  2      0.35  0.90  2      0.21  0.90  2;
% Left   face (x = 0.14)
    0.21  0.65  0      0.21  0.65  2      0.21  0.90  2      0.21  0.90  0;
% Right  face (x = 0.28)
    0.35  0.65  0      0.35  0.65  2      0.35  0.90  2      0.35  0.90  0;
];
%% Store all conductors
Conductors = {C1, C2, C3, C4, C5};
%% =====================================================================
%  Sanity Check — bounding box + face consistency + gap verification
% =====================================================================
names      = {'C1','C2','C3','C4','C5'};
face_label = {'Bottom(y=0.65)','Top(y=0.90)','Front(z=0)','Back(z=2)','Left','Right'};
face_axis = horzcat(2, 2, 3, 3, 1, 1);   % 1=x, 2=y, 3=z  (MATLAB 1-based col groups)
fprintf('\n--- BOUNDING BOX ---\n');
fprintf('%-4s  %-22s  %-18s  %-14s\n','Name','x-range (Δ)','y-range (Δ)','z-range (Δ)');
fprintf('%s\n', repmat('-',1,65));
for k = 1:numel(Conductors)
   pts = Conductors{k};
   X = pts(:,1:3:end); Y = pts(:,2:3:end); Z = pts(:,3:3:end);
   fprintf('%-4s  [%6.2f,%5.2f] Δ=%.2f   [%.2f,%.2f] Δ=%.2f   [%.1f,%.1f] Δ=%.1f\n',...
       names{k},...
       min(X(:)),max(X(:)),max(X(:))-min(X(:)),...
       min(Y(:)),max(Y(:)),max(Y(:))-min(Y(:)),...
       min(Z(:)),max(Z(:)),max(Z(:))-min(Z(:)));
end
fprintf('\n--- FACE CONSISTENCY ---\n');
for k = 1:numel(Conductors)
   pts  = Conductors{k};
   X    = pts(:,1:3:end);
   xmin = min(X(:)); xmax = max(X(:));
   face_expect = horzcat(0.65, 0.90, 0.0, 2.0, xmin, xmax);
   all_ok = true;
   for f = 1:6
       ax   = face_axis(f);
       col  = pts(f, ax:3:end);
       uval = unique(round(col,9));
       if numel(uval)~=1 || abs(uval-face_expect(f))>1e-9
           fprintf('  %s | %-18s | values=[%s] expected=%.2f  FAIL\n',...
               names{k}, face_label{f}, num2str(col,'%.4f '), face_expect(f));
           all_ok = false;
       end
   end
   if all_ok
       fprintf('  %s | All 6 faces OK\n', names{k});
   end
end
fprintf('\n--- GAP CHECK ---\n');
for k = 1:numel(Conductors)-1
   a = Conductors{k}; b = Conductors{k+1};
   Xa = a(:,1:3:end); Xb = b(:,1:3:end);
   gap = round(min(Xb(:)) - max(Xa(:)), 4);
   if abs(gap - 0.14) < 1e-9
       fprintf('  Gap %s→%s = %.2f  OK\n', names{k}, names{k+1}, gap);
   else
       fprintf('  Gap %s→%s = %.2f  FAIL (expected 0.14)\n', names{k}, names{k+1}, gap);
   end
end
fprintf('\n');