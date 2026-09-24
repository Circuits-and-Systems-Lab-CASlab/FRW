%% ========================================================================
%  PANEL INPUT FILE:  3-conductor lateral test case
%     Row (side by side along x):   C1 - D1 - D2 - C2 - D3 - C3
%     Common capping dielectric on top of everything:   D4
% =========================================================================
%
%  Layout (x grows left->right, all objects share y:[0,1]):
%     C1 : x = 0    -> 1     z = 0 -> 1   (conductor)
%     D1 : x = 1    -> 1.5   z = 0 -> 1   (dielectric)
%     D2 : x = 1.5  -> 2     z = 0 -> 1   (dielectric)
%     C2 : x = 2    -> 3     z = 0 -> 1   (conductor)
%     D3 : x = 3    -> 3.5   z = 0 -> 1   (dielectric)
%     C3 : x = 3.5  -> 4.5   z = 0 -> 1   (conductor)
%     D4 : x = 0    -> 4.5   z = 1 -> 2   (dielectric, blanket cap over ALL)
%
%  Panel row order for a full box: y0, y1, x0, x1, z0(bottom), z1(top)
%
%  OWNERSHIP RULE (no interface stored twice):
%   - conductor <-> dielectric shared face  -> stored on the CONDUCTOR side
%     (single eps = eps_in = eps_out = eps of the touching dielectric)
%   - dielectric <-> dielectric shared face -> stored on the LEFT dielectric
%     (eps_in = its own eps, eps_out = neighbour's eps)
%   - anything <-> D4 (the cap) shared face -> stored on the object BELOW,
%     with eps_out = eps_d4.  D4 itself has NO bottom panel (already
%     represented, piecewise, by the 6 objects underneath it).
%   - true outer boundaries (leftmost x0, rightmost x1, all y0/y1, all
%     bottom z0, and D4's own top) -> touching air, eps = eps_air
% =========================================================================

eps_air = 1.006;
eps_d1  = 3.9;   % edit to your actual dielectric values
eps_d2  = 4.2;
eps_d3  = 3.0;
eps_d4  = 7.5;   % common capping dielectric

%% ---------------------- C1 : conductor (full 6 panels) -------------------
C1 = [
    0 0 0   1 0 0   1 0 1   0 0 1;   % y0            -> air
    0 1 0   1 1 0   1 1 1   0 1 1;   % y1            -> air
    0 0 0   0 1 0   0 1 1   0 0 1;   % x0 (outer left)-> air
    1 0 0   1 1 0   1 1 1   1 0 1;   % x1 SHARED with D1
    0 0 0   1 0 0   1 1 0   0 1 0;   % z0 (bottom)   -> air
    0 0 1   1 0 1   1 1 1   0 1 1    % z1 (top) SHARED with D4
];
EPS_C1 = [
    eps_air  eps_air;
    eps_air  eps_air;
    eps_air  eps_air;
    eps_d1   eps_d1;    % touches D1
    eps_air  eps_air;
    eps_d4   eps_d4     % touches D4
];

%% ---------------------- D1 : dielectric (x0 owned by C1, so omitted) ----
D1 = [
    1 0 0   1.5 0 0   1.5 0 1   1 0 1;   % y0            -> air
    1 1 0   1.5 1 0   1.5 1 1   1 1 1;   % y1            -> air
    1.5 0 0 1.5 1 0   1.5 1 1   1.5 0 1; % x1 SHARED with D2
    1 0 0   1.5 0 0   1.5 1 0   1 1 0;   % z0 (bottom)   -> air
    1 0 1   1.5 0 1   1.5 1 1   1 1 1    % z1 (top) SHARED with D4
];
EPS_D1 = [
    eps_d1  eps_air;
    eps_d1  eps_air;
    eps_d1  eps_d2;     % touches D2
    eps_d1  eps_air;
    eps_d1  eps_d4      % touches D4
];

%% ---------------------- D2 : dielectric (both x-faces owned elsewhere) --
D2 = [
    1.5 0 0   2 0 0   2 0 1   1.5 0 1;   % y0            -> air
    1.5 1 0   2 1 0   2 1 1   1.5 1 1;   % y1            -> air
    1.5 0 0   2 0 0   2 1 0   1.5 1 0;   % z0 (bottom)   -> air
    1.5 0 1   2 0 1   2 1 1   1.5 1 1    % z1 (top) SHARED with D4
];
EPS_D2 = [
    eps_d2  eps_air;
    eps_d2  eps_air;
    eps_d2  eps_air;
    eps_d2  eps_d4      % touches D4
];

%% ---------------------- C2 : conductor (full 6 panels) -------------------
C2 = [
    2 0 0   3 0 0   3 0 1   2 0 1;   % y0            -> air
    2 1 0   3 1 0   3 1 1   2 1 1;   % y1            -> air
    2 0 0   2 1 0   2 1 1   2 0 1;   % x0 SHARED with D2
    3 0 0   3 1 0   3 1 1   3 0 1;   % x1 SHARED with D3
    2 0 0   3 0 0   3 1 0   2 1 0;   % z0 (bottom)   -> air
    2 0 1   3 0 1   3 1 1   2 1 1    % z1 (top) SHARED with D4
];
EPS_C2 = [
    eps_air  eps_air;
    eps_air  eps_air;
    eps_d2   eps_d2;    % touches D2
    eps_d3   eps_d3;    % touches D3
    eps_air  eps_air;
    eps_d4   eps_d4     % touches D4
];

%% ---------------------- D3 : dielectric (both x-faces owned by C2/C3) ---
D3 = [
    3 0 0   3.5 0 0   3.5 0 1   3 0 1;   % y0            -> air
    3 1 0   3.5 1 0   3.5 1 1   3 1 1;   % y1            -> air
    3 0 0   3.5 0 0   3.5 1 0   3 1 0;   % z0 (bottom)   -> air
    3 0 1   3.5 0 1   3.5 1 1   3 1 1    % z1 (top) SHARED with D4
];
EPS_D3 = [
    eps_d3  eps_air;
    eps_d3  eps_air;
    eps_d3  eps_air;
    eps_d3  eps_d4      % touches D4
];

%% ---------------------- C3 : conductor (full 6 panels) -------------------
C3 = [
    3.5 0 0   4.5 0 0   4.5 0 1   3.5 0 1;   % y0            -> air
    3.5 1 0   4.5 1 0   4.5 1 1   3.5 1 1;   % y1            -> air
    3.5 0 0   3.5 1 0   3.5 1 1   3.5 0 1;   % x0 SHARED with D3
    4.5 0 0   4.5 1 0   4.5 1 1   4.5 0 1;   % x1 (outer right) -> air
    3.5 0 0   4.5 0 0   4.5 1 0   3.5 1 0;   % z0 (bottom)   -> air
    3.5 0 1   4.5 0 1   4.5 1 1   3.5 1 1    % z1 (top) SHARED with D4
];
EPS_C3 = [
    eps_air  eps_air;
    eps_air  eps_air;
    eps_d3   eps_d3;    % touches D3
    eps_air  eps_air;
    eps_air  eps_air;
    eps_d4   eps_d4     % touches D4
];

%% ---------------------- D4 : blanket capping dielectric ------------------
%  Bottom face (z=1) is OMITTED on purpose: it's already fully represented,
%  piecewise, by the top panels of C1/D1/D2/C2/D3/C3 above.
D4 = [
    0 0 1     4.5 0 1   4.5 0 2   0 0 2;   % y0            -> air
    0 1 1     4.5 1 1   4.5 1 2   0 1 2;   % y1            -> air
    0 0 1     0 1 1     0 1 2     0 0 2;   % x0 (left end) -> air
    4.5 0 1   4.5 1 1   4.5 1 2   4.5 0 2; % x1 (right end)-> air
    0 0 2     4.5 0 2   4.5 1 2   0 1 2    % z1 (top)      -> air
];
EPS_D4 = [
    eps_d4  eps_air;
    eps_d4  eps_air;
    eps_d4  eps_air;
    eps_d4  eps_air;
    eps_d4  eps_air
];

%% ------------------------- Assemble full model ---------------------------
Panels   = [C1; D1; D2; C2; D3; C3; D4];
EpsInOut = [EPS_C1; EPS_D1; EPS_D2; EPS_C2; EPS_D3; EPS_C3; EPS_D4];

ObjTag = [ repmat({'C1'},size(C1,1),1); repmat({'D1'},size(D1,1),1);
           repmat({'D2'},size(D2,1),1); repmat({'C2'},size(C2,1),1);
           repmat({'D3'},size(D3,1),1); repmat({'C3'},size(C3,1),1);
           repmat({'D4'},size(D4,1),1) ];

save('panel_input_3cond.mat','C1','D1','D2','C2','D3','C3','D4', ...
     'EPS_C1','EPS_D1','EPS_D2','EPS_C2','EPS_D3','EPS_C3','EPS_D4', ...
     'Panels','EpsInOut','ObjTag');