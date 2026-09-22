%Input file 
clc; clear;
% ======================================================================
% Test Case 2 — Two Parallel Conductor Geometry
%
% Axis convention:
%   x = length direction  (0 → 4,  Δx = 4)
%   y = separation axis   (C1: 0→1, gap: 1→5, C2: 5→6)
%   z = depth direction   (0 → 1,  Δz = 1)
%
% Conductors:
%   C1: x=[0,4]  y=[0,1]  z=[0,1]
%   C2: x=[0,4]  y=[5,6]  z=[0,1]
%   Gap between C1 and C2 in y = 4 units (y=1 to y=5)
%
% Dielectric panels (flat surfaces between conductors):
%   D1: flat panel at y=3  (gap midpoint)
%   D2: flat panel at y=4  (FIX: was non-planar, corrected to y=4)
%
% Face row order (both conductors):
%   Row 1 — y_min face
%   Row 2 — y_max face
%   Row 3 — back  face (x=0)
%   Row 4 — front face (x=4)
%   Row 5 — Bottom face (z=0)
%   Row 6 — Top    face (z=1)
% ======================================================================
%% -------- Conductor C1 --------
% x: 0→4,  y: 0→1,  z: 0→1
C1 = [
% y_min face (y = 0)
   0  0  0     4  0  0     4  0  1     0  0  1;
% y_max face (y = 1)
   0  1  0     4  1  0     4  1  1     0  1  1;
% Back  face (x = 0)
   0  0  0     0  1  0     0  1  1     0  0  1;
% Front face (x = 4)
   4  0  0     4  1  0     4  1  1     4  0  1;
% Bottom face (z = 0)
   0  0  0     4  0  0     4  1  0     0  1  0;
% Top    face (z = 1)
   0  0  1     4  0  1     4  1  1     0  1  1;
];
%% -------- Conductor C2 --------
% x: 0→4,  y: 5→6,  z: 0→1
C2 = [
% y_min face (y = 5)
   0  5  0     4  5  0     4  5  1     0  5  1;
% y_max face (y = 6)
   0  6  0     4  6  0     4  6  1     0  6  1;
% Back  face (x = 0)
   0  5  0     0  6  0     0  6  1     0  5  1;
% Front face (x = 4)
   4  5  0     4  6  0     4  6  1     4  5  1;
% Bottom face (z = 0)
   0  5  0     4  5  0     4  6  0     0  6  0;
% Top    face (z = 1)
   0  5  1     4  5  1     4  6  1     0  6  1;
];
%% -------- Dielectric Panels --------
% Each panel is a single flat face (4 corner points)
% Used as POLY3D DIEL= surfaces in Raphael
% D1: flat panel at y = 3.0  (gap midpoint: (1+5)/2 = 3)
D1 = [0  3  0     4  3  0     4  3  1     0  3  1];
% D2: flat panel at y = 4.0
% FIX: original had y=[4,5,6,7] and z=[0,0,1,1] — non-planar tilted face
%      corrected to a flat panel at y=4 matching the same xz extents as D1
D2 = [0  4  0     4  4  0     4  4  1     0  4  1];
%% -------- Store --------
Conductors = {C1, C2};
Dielectrics = {D1, D2};
%% ======================================================================
%  Sanity Check
% ======================================================================
cnames = {'C1','C2'};
dnames = {'D1','D2'};
% Expected constant axis values per face per conductor
expected = {
   [0, 1, 0, 4, 0, 1];   % C1: y0,y1,x0,x4,z0,z1
   [5, 6, 0, 4, 0, 1];   % C2: y5,y6,x0,x4,z0,z1
};
faxes  = [2, 2, 1, 1, 3, 3];   % 1=x,2=y,3=z (MATLAB column groups)
flabels = {'y_min','y_max','back(x=0)','front(x=4)','bottom(z=0)','top(z=1)'};
fprintf('\n--- BOUNDING BOX ---\n');
fprintf('%-4s  %-20s  %-16s  %-12s\n','Name','x-range (Δ)','y-range (Δ)','z-range (Δ)');
fprintf('%s\n',repmat('-',1,58));
for k = 1:2
   pts = Conductors{k};
   X=pts(:,1:3:end); Y=pts(:,2:3:end); Z=pts(:,3:3:end);
   fprintf('%-4s  [%.1f, %.1f] Δ=%.1f          [%.1f, %.1f] Δ=%.1f      [%.1f, %.1f] Δ=%.1f\n',...
       cnames{k},...
       min(X(:)),max(X(:)),max(X(:))-min(X(:)),...
       min(Y(:)),max(Y(:)),max(Y(:))-min(Y(:)),...
       min(Z(:)),max(Z(:)),max(Z(:))-min(Z(:)));
end
fprintf('\n--- FACE CONSISTENCY ---\n');
for k = 1:2
   pts = Conductors{k};
   exp = expected{k};
   all_ok = true;
   for f = 1:6
       ax  = faxes(f);
       col = pts(f, ax:3:end);
       uv  = unique(round(col,9));
       if numel(uv)~=1 || abs(uv - exp(f)) > 1e-9
           fprintf('  %s | %-14s | values=[%s] expected=%.1f  FAIL\n',...
               cnames{k}, flabels{f}, num2str(col,'%.2f '), exp(f));
           all_ok = false;
       end
   end
   if all_ok
       fprintf('  %s | All 6 faces OK\n', cnames{k});
   end
end
fprintf('\n--- GAP CHECK ---\n');
y_top_C1 = max(Conductors{1}(:,2:3:end),[],'all');
y_bot_C2 = min(Conductors{2}(:,2:3:end),[],'all');
fprintf('  C1 y_max = %.1f\n', y_top_C1);
fprintf('  C2 y_min = %.1f\n', y_bot_C2);
fprintf('  Gap      = %.1f units\n', y_bot_C2 - y_top_C1);
fprintf('\n--- DIELECTRIC PANELS ---\n');
for k = 1:2
   d   = Dielectrics{k};
   y_v = d(1, 2:3:end);
   z_v = d(1, 3:3:end);
   uy  = unique(round(y_v,9));
   uz  = unique(round(z_v,9));
   y_ok = (numel(uy)==1);
   z_ok = (numel(uz) >1);   % z should span 0→1
   if y_ok
       fprintf('  %s | flat at y=%.1f  OK\n', dnames{k}, uy);
   else
       fprintf('  %s | NOT flat — y values=[%s]  FAIL\n', dnames{k}, num2str(y_v,'%.1f '));
   end
end
fprintf('\n');
%Input file 
clc; clear;
% ======================================================================
% Test Case 2 — Two Parallel Conductor Geometry
%
% Axis convention:
%   x = length direction  (0 → 4,  Δx = 4)
%   y = separation axis   (C1: 0→1, gap: 1→5, C2: 5→6)
%   z = depth direction   (0 → 1,  Δz = 1)
%
% Conductors:
%   C1: x=[0,4]  y=[0,1]  z=[0,1]
%   C2: x=[0,4]  y=[5,6]  z=[0,1]
%   Gap between C1 and C2 in y = 4 units (y=1 to y=5)
%
% Dielectric panels (flat surfaces between conductors):
%   D1: flat panel at y=3  (gap midpoint)
%   D2: flat panel at y=4  (FIX: was non-planar, corrected to y=4)
%
% Face row order (both conductors):
%   Row 1 — y_min face
%   Row 2 — y_max face
%   Row 3 — back  face (x=0)
%   Row 4 — front face (x=4)
%   Row 5 — Bottom face (z=0)
%   Row 6 — Top    face (z=1)
% ======================================================================
%% -------- Conductor C1 --------
% x: 0→4,  y: 0→1,  z: 0→1
C1 = [
% y_min face (y = 0)
   0  0  0     4  0  0     4  0  1     0  0  1;
% y_max face (y = 1)
   0  1  0     4  1  0     4  1  1     0  1  1;
% Back  face (x = 0)
   0  0  0     0  1  0     0  1  1     0  0  1;
% Front face (x = 4)
   4  0  0     4  1  0     4  1  1     4  0  1;
% Bottom face (z = 0)
   0  0  0     4  0  0     4  1  0     0  1  0;
% Top    face (z = 1)
   0  0  1     4  0  1     4  1  1     0  1  1;
];
%% -------- Conductor C2 --------
% x: 0→4,  y: 5→6,  z: 0→1
C2 = [
% y_min face (y = 5)
   0  5  0     4  5  0     4  5  1     0  5  1;
% y_max face (y = 6)
   0  6  0     4  6  0     4  6  1     0  6  1;
% Back  face (x = 0)
   0  5  0     0  6  0     0  6  1     0  5  1;
% Front face (x = 4)
   4  5  0     4  6  0     4  6  1     4  5  1;
% Bottom face (z = 0)
   0  5  0     4  5  0     4  6  0     0  6  0;
% Top    face (z = 1)
   0  5  1     4  5  1     4  6  1     0  6  1;
];
%% -------- Dielectric Panels --------
% Each panel is a single flat face (4 corner points)
% Used as POLY3D DIEL= surfaces in Raphael
% D1: flat panel at y = 3.0  (gap midpoint: (1+5)/2 = 3)
D1 = [0  3  0     4  3  0     4  3  1     0  3  1];
% D2: flat panel at y = 4.0
% FIX: original had y=[4,5,6,7] and z=[0,0,1,1] — non-planar tilted face
%      corrected to a flat panel at y=4 matching the same xz extents as D1
D2 = [0  4  0     4  4  0     4  4  1     0  4  1];
%% -------- Store --------
Conductors = {C1, C2};
Dielectrics = {D1, D2};
%% ======================================================================
%  Sanity Check
% ======================================================================
cnames = {'C1','C2'};
dnames = {'D1','D2'};
% Expected constant axis values per face per conductor
expected = {
   [0, 1, 0, 4, 0, 1];   % C1: y0,y1,x0,x4,z0,z1
   [5, 6, 0, 4, 0, 1];   % C2: y5,y6,x0,x4,z0,z1
};
faxes  = [2, 2, 1, 1, 3, 3];   % 1=x,2=y,3=z (MATLAB column groups)
flabels = {'y_min','y_max','back(x=0)','front(x=4)','bottom(z=0)','top(z=1)'};
fprintf('\n--- BOUNDING BOX ---\n');
fprintf('%-4s  %-20s  %-16s  %-12s\n','Name','x-range (Δ)','y-range (Δ)','z-range (Δ)');
fprintf('%s\n',repmat('-',1,58));
for k = 1:2
   pts = Conductors{k};
   X=pts(:,1:3:end); Y=pts(:,2:3:end); Z=pts(:,3:3:end);
   fprintf('%-4s  [%.1f, %.1f] Δ=%.1f          [%.1f, %.1f] Δ=%.1f      [%.1f, %.1f] Δ=%.1f\n',...
       cnames{k},...
       min(X(:)),max(X(:)),max(X(:))-min(X(:)),...
       min(Y(:)),max(Y(:)),max(Y(:))-min(Y(:)),...
       min(Z(:)),max(Z(:)),max(Z(:))-min(Z(:)));
end
fprintf('\n--- FACE CONSISTENCY ---\n');
for k = 1:2
   pts = Conductors{k};
   exp = expected{k};
   all_ok = true;
   for f = 1:6
       ax  = faxes(f);
       col = pts(f, ax:3:end);
       uv  = unique(round(col,9));
       if numel(uv)~=1 || abs(uv - exp(f)) > 1e-9
           fprintf('  %s | %-14s | values=[%s] expected=%.1f  FAIL\n',...
               cnames{k}, flabels{f}, num2str(col,'%.2f '), exp(f));
           all_ok = false;
       end
   end
   if all_ok
       fprintf('  %s | All 6 faces OK\n', cnames{k});
   end
end
fprintf('\n--- GAP CHECK ---\n');
y_top_C1 = max(Conductors{1}(:,2:3:end),[],'all');
y_bot_C2 = min(Conductors{2}(:,2:3:end),[],'all');
fprintf('  C1 y_max = %.1f\n', y_top_C1);
fprintf('  C2 y_min = %.1f\n', y_bot_C2);
fprintf('  Gap      = %.1f units\n', y_bot_C2 - y_top_C1);
fprintf('\n--- DIELECTRIC PANELS ---\n');
for k = 1:2
   d   = Dielectrics{k};
   y_v = d(1, 2:3:end);
   z_v = d(1, 3:3:end);
   uy  = unique(round(y_v,9));
   uz  = unique(round(z_v,9));
   y_ok = (numel(uy)==1);
   z_ok = (numel(uz) >1);   % z should span 0→1
   if y_ok
       fprintf('  %s | flat at y=%.1f  OK\n', dnames{k}, uy);
   else
       fprintf('  %s | NOT flat — y values=[%s]  FAIL\n', dnames{k}, num2str(y_v,'%.1f '));
   end
end
fprintf('\n');
