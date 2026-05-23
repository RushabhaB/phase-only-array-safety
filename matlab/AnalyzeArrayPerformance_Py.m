% AnalyzeArrayPerformance_Py.m
close all; clc;
% NOTE: 'clear' is intentionally not called here so that batch-mode env
% overrides (AAP_CASE / AAP_DATA below) and any pre-set workspace vars
% (e.g. via -batch "CaseID=...; DataFileName=...; AnalyzeArrayPerformance_Py")
% are preserved.

%% 0. User Configuration & Data Loading
% --- Batch-mode overrides via environment variables ---
%   AAP_CASE=<int>   AAP_DATA=<path-to-mat-file>
% These take precedence over the defaults set below; allows scripted
% batch invocation without editing this file each run.
if ~exist('CaseID', 'var');       CaseID = 4; end
if ~exist('DataFileName', 'var'); DataFileName = "./Weights/Gmin/phase_only_w/optimized_results_iMM_case_4.mat"; end
override_case = getenv('AAP_CASE');
override_file = getenv('AAP_DATA');
if ~isempty(override_case);  CaseID = str2double(override_case); end
if ~isempty(override_file);  DataFileName = string(override_file); end

% --- Add ArrayGeometry helper from sibling project to the MATLAB path ---
geom_dirs = { ...
    fullfile(getenv('HOME'), 'Documents/workspace/ARS_UCLA/Experiment36x36/functions'), ...
    fullfile(getenv('HOME'), 'workspace/ARS_UCLA/Experiment36x36/functions') ...
};
for gi_ = 1:numel(geom_dirs)
    if isfolder(geom_dirs{gi_})
        addpath(geom_dirs{gi_});
        break;
    end
end


if ~isfile(DataFileName)
    error('Data file "%s" not found in current directory.', DataFileName);
end

fprintf('Loading data from: %s\n', DataFileName);
data = load(DataFileName);

% Extract Weights
if isfield(data, 'w_opt')
    UserWeights = data.w_opt;
else
    vars = fieldnames(data);
    warning('Variable "w_opt_iMM" not found. Using first variable "%s".', vars{1});
    UserWeights = data.(vars{1});
end

% Detect optimization method from the file path so we can name EPS outputs
% with the paper convention (case_X_reflection_coeff_{baseline,iMM,convex}.eps,
% case_X_array_gain.eps).
if contains(DataFileName, 'phase_only_w') || contains(DataFileName, 'iMM')
    method_label = 'iMM';
elseif contains(DataFileName, 'inf_norm_w')
    method_label = 'convex';
else
    method_label = 'user';
end

% Detect non-unit-modulus weights (e.g. the convex relaxation produces
% |w_p| in [0, 1]). For those, dividing by |w_p|^2 in the per-port active
% reflection coefficient blows up at near-zero ports and is *not* the
% quantity the relaxation actually minimizes. The apples-to-apples metric
% vs. the unit-modulus iMM/baseline columns is the per-port reflected
% energy |b_p|^2 / Sum_k |X_k|^2 = Gamma^2 * |w_p|^2.
non_unit_modulus = max(abs(abs(UserWeights(:)) - 1)) > 1e-3;


%% 1. Python Environment Setup
% --- Auto-pick the first available env (works on both local and coresnix) ---
candidatePython = { ...
    '/home/rushabha/miniconda3/envs/manifold_opt/bin/python', ... % local mac/linux
    '/home/rushabha/anaconda3/envs/sbl_tran/bin/python', ...      % coresnix
    '/home/rushabha/miniconda/envs/wideband_DoA/bin/python' ...   % coresws1
};
targetPython = '';
for ci_ = 1:numel(candidatePython)
    if isfile(candidatePython{ci_})
        targetPython = candidatePython{ci_};
        break;
    end
end
if isempty(targetPython)
    error('No known Python env found. Add the path to candidatePython.');
end
fprintf('Using Python env: %s\n', targetPython);

pe = pyenv;
if pe.Status == "NotLoaded"
    try
        pyenv('Version', targetPython);
    catch ME
        warning('Failed to set Python path.'); rethrow(ME);
    end
elseif pe.Executable ~= targetPython
    error('Incorrect Python Environment Loaded. Restart MATLAB.');
end

if count(py.sys.path, pwd) == 0
    insert(py.sys.path, int32(0), pwd);
end

try
    py.importlib.import_module('numpy');
    py.importlib.import_module('torch');
    arrMod = py.importlib.import_module('antennaeArray');
    py.importlib.reload(arrMod);
    mod = py.importlib.import_module('MatlabInterface');
    py.importlib.reload(mod);
catch ME
    error('Error loading Python modules. Check environment.');
end

%% 2. Initialize Python Physics Engine
fprintf('Initializing Python Physics Engine for Case %d...\n', CaseID);
bridge = py.MatlabInterface.ArrayCalcBridge(int32(CaseID));
[elem] = ArrayGeometry_36x36_Vivaldi(false);

num_elements = length(elem.ID);
if length(UserWeights) ~= num_elements
    error('UserWeights length (%d) does not match array geometry.', length(UserWeights));
end

%% 3. Calculate Reflection Coefficient (Baseline vs User)
fprintf('Calculating Reflection Coefficient via Python...\n');

% A. BASELINE Calculation (All 1s, steering applied in Python)
w_base = ones(num_elements, 1);
RC_base_Raw = double(bridge.calculate_max_reflected_power(real(w_base), imag(w_base), true))';
RC_base_Raw = 10*log10(RC_base_Raw);

% B. USER WEIGHTS Calculation (optimized weights used as-is). The bridge
% returns Gamma^2 = max_p |b_p|^2 / (|w_p|^2 * Sum_k|X_k|^2). For
% non-unit-modulus weights we multiply back by |w_p|^2 to plot the
% per-port reflected energy instead -- see comment above and discussion
% in solver.py::solveArraySafety_inf.
RC_user_linear = double(bridge.calculate_max_reflected_power(real(UserWeights), imag(UserWeights), false));
RC_user_linear = RC_user_linear(:);   % force column to avoid broadcast mismatch
if non_unit_modulus
    RC_user_linear = RC_user_linear .* abs(UserWeights(:)).^2;
end
RC_user_Raw = 10*log10(RC_user_linear).';   % row 1x1296, matches RC_base_Raw

% --- REORDER DATA (Python Row-Major -> MATLAB Column-Major) ---
% 1. Reshape 1D vector to 36x36 Matrix.
RC_base_Matrix_T = reshape(RC_base_Raw, 36, 36);
RC_user_Matrix_T = reshape(RC_user_Raw, 36, 36);

% 2. Transpose back to get correct physical alignment [Row, Col]
RC_base_Matrix = RC_base_Matrix_T.';
RC_user_Matrix = RC_user_Matrix_T.';

% 3. Flatten using MATLAB's default Column-Major order to match 'elem' geometry
RC_base = RC_base_Matrix(:);
RC_user = RC_user_Matrix(:);
% -----------------------------------------------------------------------------

% C. Define Limits
RC_CLim_Min = min([RC_base; RC_user]);
RC_CLim_Max = max([RC_base; RC_user]);
fprintf('Reflection Coeff Range: [%0.4f, %0.4f]\n', RC_CLim_Min, RC_CLim_Max);

% Smooth colormap for reflection coefficient plots
RC_cmap = turbo(256);

%% 4. Calculate UV Array Gain (User Weights Only)
fprintf('Calculating UV Array Gain via Python...\n');

uv_result = bridge.calculate_uv_gain_grid(real(UserWeights), imag(UserWeights), pyargs('n_grid', int32(200), 'baseline', false));
U_grid = double(uv_result{1});
V_grid = double(uv_result{2});
% Gain_grid comes back as 3D array: (Grid x Grid x NumSegments)
Gain_stack = double(uv_result{3}); 
NumSegments = size(Gain_stack, 3);

% Pick the right axis label for this run. Baseline (always unit-modulus)
% and iMM (unit-modulus) both display Gamma^2; the convex column displays
% per-port reflected energy. Both are plotted in dB.
% (R2024b's 'latex' interpreter rejects \sum here for unknown reasons -- use
%  the 'tex' interpreter throughout, which renders \Gamma and \Sigma cleanly.)
if non_unit_modulus
    user_cbar_label = '\Sigma_{k=1}^{K} |S_k w X(f_k)|^2';
    user_title_fmt  = 'USER Per-Port Reflected Energy\nMax: %0.4f | Avg: %0.4f';
else
    user_cbar_label = '\Gamma^2_i (dB)';
    user_title_fmt  = 'USER Reflection Coefficient\nMax: %0.4f | Avg: %0.4f';
end
user_cbar_interp = 'tex';
base_cbar_label  = '\Gamma^2_i (dB)';
base_cbar_interp = 'tex';

%% 5. FIGURE 1: Baseline Reflection Coefficient
figure('Name', 'Baseline Reflection Coefficient', 'Units', 'Normalized', 'Position', [0.05 0.05 0.4 0.4]);

scatter(elem.x, elem.y, 40, RC_base, 'filled', 'markeredgecolor', 'k');
axis equal tight;
xlabel('X (inches)','FontSize',14); ylabel('Y (inches)','FontSize',14);
c = colorbar; c.Label.String = base_cbar_label; c.Label.Interpreter = base_cbar_interp; c.Label.FontSize=20;
c.Color = 'k'; c.Label.Color = 'k';
colormap(gca, RC_cmap);
clim([RC_CLim_Min RC_CLim_Max]);
% (no title -- paper figures do not carry a per-panel title; the LaTeX caption supplies context)
fprintf('BASELINE: Max=%.4f | Avg=%.4f dB\n', max(RC_base), mean(RC_base));
box on; grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
set(gcf, 'Color', 'w');
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 4.57 3.90]);
print(gcf, fullfile('./Figures', sprintf('case_%d_reflection_coeff_baseline.eps', CaseID)), '-depsc', '-painters');

%% 6. FIGURE 2: User Reflection Coefficient (or per-port energy for convex)
figure('Name', 'User Reflection Coefficient', 'Units', 'Normalized', 'Position', [0.5 0.05 0.4 0.4]);

scatter(elem.x, elem.y, 40, RC_user, 'filled', 'markeredgecolor', 'k');
axis equal tight;
xlabel('X (inches)','FontSize', 14); ylabel('Y (inches)','FontSize',14);
c = colorbar; c.Label.String = user_cbar_label; c.Label.Interpreter = user_cbar_interp; c.Label.FontSize = 20;
c.Color = 'k'; c.Label.Color = 'k';
colormap(gca, RC_cmap);
clim([RC_CLim_Min RC_CLim_Max]);
% (no title -- LaTeX caption supplies context)
fprintf('USER (%s): Max=%.4f | Avg=%.4f dB\n', method_label, max(RC_user), mean(RC_user));
box on; grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
set(gcf, 'Color', 'w');
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 4.57 3.90]);
print(gcf, fullfile('./Figures', sprintf('case_%d_reflection_coeff_%s.eps', CaseID, method_label)), '-depsc', '-painters');

%% 7. FIGURE 3: UV Gain Subplots
% Tight per-case layout sized so the three EPS files line up at the same
% column height in the paper:
%   case 1 (1 seg)  -> 1x1 in a 4x4-inch frame
%   case 2 (2 segs) -> 1x2 in an 8x4-inch frame (2:1 wide; LaTeX side
%                       vertically centres it inside a square minipage)
%   case 4 (4 segs) -> 2x2 in a 4x4-inch frame
% Inner whitespace is killed by a single shared xlabel/ylabel on the
% tiledlayout, hidden tick labels on inner edges, and 'tight' spacing.

thetas_deg = double(py.array.array('d', bridge.thetas));
phis_deg   = double(py.array.array('d', bridge.phis));
u_beam = sin(deg2rad(thetas_deg)) .* cos(deg2rad(phis_deg));
v_beam = sin(deg2rad(thetas_deg)) .* sin(deg2rad(phis_deg));
th_circle = 0:pi/50:2*pi;

if NumSegments == 1
    rows = 1; cols = 1; order = 1; paper = [4 4];

elseif NumSegments == 2
    % 1x2 in a SQUARE 4x4 frame (so the EPS aspect matches case 1 and
    % case 4 and the column fills at width=\linewidth without vertical
    % whitespace inside the LaTeX minipage).  Each tile is 2 wide x 4
    % tall, circles stay round at 2-inch diameter via axis equal --
    % surrounding tile space gives the visual presence the user asked
    % for.  Larger-u beam goes LEFT so beams converge at the seam.
    if u_beam(1) >= u_beam(2); order = [1, 2]; else; order = [2, 1]; end
    rows = 1; cols = 2; paper = [4 4];

elseif NumSegments == 4
    % 2x2: each segment in the OPPOSITE quadrant of its beam direction so
    % the four beams converge toward the centre of the figure.
    seg_to_tile = zeros(1, 4);
    for s = 1:4
        if v_beam(s) >= 0  % beam in top half  -> place in bottom half
            if u_beam(s) >= 0; seg_to_tile(s) = 3; else; seg_to_tile(s) = 4; end
        else               % beam in bottom half -> place in top half
            if u_beam(s) >= 0; seg_to_tile(s) = 1; else; seg_to_tile(s) = 2; end
        end
    end
    used = false(1, 4);
    for s = 1:4
        if used(seg_to_tile(s))
            for k = 1:4
                if ~used(k); seg_to_tile(s) = k; break; end
            end
        end
        used(seg_to_tile(s)) = true;
    end
    order = zeros(1, 4);
    for s = 1:4; order(seg_to_tile(s)) = s; end
    % Case 4 carries the shared dB colorbar at its east edge (i.e. at the
    % extreme right of the paper row), so the figure is wider than 4x4.
    rows = 2; cols = 2; paper = [5 4];

else
    error('Unsupported number of segments: %d', NumSegments);
end

figure('Name', sprintf('Case %d UV Array Response', CaseID), ...
       'Units', 'Inches', 'Position', [1 1 paper(1) paper(2)]);
% 'none' kills the inter-tile gap entirely; case 4's 2x2 grid then reads
% as a tightly packed cluster of circles.
t = tiledlayout(rows, cols, 'TileSpacing','none', 'Padding','tight');

for tile_idx = 1:numel(order)
    seg = order(tile_idx);
    if seg < 1 || seg > NumSegments; continue; end
    nexttile(tile_idx);
    Gain_seg = Gain_stack(:, :, seg);
    h = pcolor(U_grid, V_grid, Gain_seg);
    set(h, 'EdgeColor', 'none');
    hold on;
    plot(cos(th_circle), sin(th_circle), 'k--', 'LineWidth', 1.0);
    axis equal;
    xlim([-1.02 1.02]); ylim([-1.02 1.02]);
    % Uniform [-40, 0] dB scale across every panel in every case so the
    % single colorbar at the extreme right of the paper row reads against
    % the same reference.  0 dB = max gain achievable by conjugate
    % beamforming (caption); -40 dB caps the dynamic range.
    clim([-40, 0]);
    colormap(gca, 'parula');

    cur_row = ceil(tile_idx / cols);
    cur_col = rem(tile_idx - 1, cols) + 1;   % 'mod' is shadowed by py.module above

    % With TileSpacing='none' adjacent tiles share an edge; force each
    % tile to label only the OUTER end of that edge so labels don't
    % collide at the seam (e.g. left tile shows -1,0 and right tile 0,1).
    if cols > 1
        if cur_col == 1
            set(gca, 'XTick', [-1 0]);
        elseif cur_col == cols
            set(gca, 'XTick', [0 1]);
        else
            set(gca, 'XTick', [-1 0 1]);
        end
    end
    if rows > 1
        if cur_row == 1
            set(gca, 'YTick', [0 1]);
        elseif cur_row == rows
            set(gca, 'YTick', [-1 0]);
        else
            set(gca, 'YTick', [-1 0 1]);
        end
    end

    if cur_row < rows         % not bottom row -> drop x tick labels
        set(gca, 'XTickLabel', []);
    end
    if cur_col > 1            % not leftmost col -> drop y tick labels
        set(gca, 'YTickLabel', []);
    end
    set(gca, 'FontSize', 10, 'TickDir', 'in', ...
             'Color', 'w', 'XColor', 'k', 'YColor', 'k');
end

% Shared axis labels on the outer edges of the tiledlayout.
xlabel(t, 'u', 'FontSize', 12, 'Color', 'k');
ylabel(t, 'v', 'FontSize', 12, 'Color', 'k');

% Single shared colorbar -- placed only on case 4 (the rightmost figure
% in the paper row), so the dB scale sits at the extreme right.  All
% panels in cases 1, 2, 4 use the same clim ([-40, 0] dB), so this one
% bar is the reference for every panel.
if NumSegments == 4
    cb = colorbar;
    cb.Label.String = 'Gain (dB)';
    cb.Label.FontSize = 11;
    cb.FontSize = 9;
    cb.Color = 'k';
    cb.Label.Color = 'k';
    cb.Layout.Tile = 'east';
end

if strcmp(method_label, 'iMM')
    set(gcf, 'Color', 'w');
    set(gcf, 'PaperUnits','inches', 'PaperPosition', [0 0 paper(1) paper(2)]);
    out = fullfile('./Figures', sprintf('case_%d_array_gain.eps', CaseID));
    print(gcf, out, '-depsc', '-painters', '-loose');  % -loose: BBox = PaperPosition
    fprintf('Saved %s\n', out);
end

fprintf('Analysis Complete.\n');