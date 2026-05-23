%% plot_uv_heatmaps.m
% Plots max per-port Gamma^2 (dB) over UV for saved Case 3 / 5 baseline runs
% and the adaptive iMM Case 5 sweep. Uses the post-fix Gamma^2 stored in
% reflection_coeff_per_port (when present) so no impedance/nEl heuristic is
% applied. Falls back to the dBm-derived conversion only for legacy files
% that don't carry the per-port field.

clear; clc; close all;

%% --- Configuration ---
results_dir = './Weights/Gmin/inf_norm_w';

cases = [3,5];
case_labels = {'Case 3 (Single Segment UV Sweep)', ...
               'Case 5 (Two Segments, Seg 1 Sweeping)'};

%% --- Store baseline Case 5 colorbar limits ---
baseline_clim_5 = [];

for ci = 1:length(cases)
    case_num = cases(ci);
    mat_file = fullfile(results_dir, sprintf('baseline_results_case_%d.mat', case_num));

    if ~isfile(mat_file)
        mat_file = fullfile('./Weights/Gmin/inf_norm_w', ...
                            sprintf('baseline_results_case_%d.mat', case_num));
    end

    if ~isfile(mat_file)
        warning('File not found for Case %d. Skipping.', case_num);
        continue;
    end

    fprintf('Loading %s ...\n', mat_file);
    D = load(mat_file);

    %% --- Extract Data ---
    u = double(D.u(:));
    v = double(D.v(:));

    % Use saved post-fix per-port Gamma^2 directly: max over ports, then 10*log10.
    % This matches the convention of the 3x3 figure exactly (no impedance/nEl
    % factor inferred from dBm). Legacy files without this field fall through
    % to the dBm-derived path.
    if isfield(D, 'reflection_coeff_per_port')
        rc = double(real(D.reflection_coeff_per_port));   % [N_tasks x nEl]
        max_gamma2 = max(rc, [], 2);                       % max over ports per task
        refl_coeff_dB = 10*log10(max_gamma2(:));
    else
        warning('reflection_coeff_per_port absent; using legacy dBm conversion.');
        Z_0 = 50; nEl = 1296;
        max_ref_power_dbm = double(real(D.max_reflected_power_dbm(:)));
        refl_coeff_dB = 10*log10(10.^((max_ref_power_dbm - 30)/10) * Z_0 * nEl);
    end

    %% --- Interpolate onto regular grid for plotting ---
    n_grid = 100;
    u_grid = linspace(-1, 1, n_grid);
    v_grid = linspace(-1, 1, n_grid);
    [U, V] = meshgrid(u_grid, v_grid);

    % Mask points outside the unit circle
    outside = (U.^2 + V.^2) > 1;

    % Natural interpolation
    F = scatteredInterpolant(u, v, refl_coeff_dB, 'natural', 'none');
    Z = F(U, V);

    % Fill NaN gaps inside the circle using nearest-neighbor
    F_nn = scatteredInterpolant(u, v, refl_coeff_dB, 'nearest', 'nearest');
    nan_mask = isnan(Z) & ~outside;
    Z(nan_mask) = F_nn(U(nan_mask), V(nan_mask));
    Z(outside) = NaN;

    % Save Case 5 colorbar limits for the adaptive plot
    if case_num == 5
        baseline_clim_5 = [min(refl_coeff_dB), max(refl_coeff_dB)];
    end

    %% --- Plot Corrected Reflection Coefficient (dB) ---
    figure('Position', [100, 100, 650, 500]);
    imagesc(u_grid, v_grid, Z, 'AlphaData', ~outside);
    set(gca, 'YDir', 'normal', 'Color', [1 1 1], 'FontSize', 11);
    hold on;
    theta_circle = linspace(0, 2*pi, 361);
    plot(cos(theta_circle), sin(theta_circle), 'k-', 'LineWidth', 1.5);
    hold off;

    cb = colorbar;
    cb.Label.String = '$\Gamma^2$ (dB)';
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = 20;
    colormap(jet);
    axis equal tight;
    xlim([-1.05 1.05]); ylim([-1.05 1.05]);
    xlabel('u', 'FontSize', 14);
    ylabel('v', 'FontSize', 14);

    % Save (PNG for quick view + EPS in paper-naming convention so the file
    % can drop directly into paper/Figures/)
    saveas(gcf, fullfile('./Figures', sprintf('refl_coeff_dB_uv_case_%d.png', case_num)));
    print(gcf, fullfile('./Figures', ...
        sprintf('baseline_case_%d_heatmap_matlab.eps', case_num)), '-depsc', '-painters');

    fprintf('Case %d plot done.\n\n', case_num);
end

%% --- Plot Adaptive iMM Case 5 ---
adaptive_file = './speed_comparison_results/speed_comparison_case_5_adaptive_cuda.mat';
sweep_file    = './speed_comparison_results/case_5_sweep.mat';

if isfile(adaptive_file) && isfile(sweep_file)
    fprintf('Loading adaptive iMM Case 5 ...\n');
    A = load(adaptive_file);
    S = load(sweep_file);

    u_a = double(S.u(:));
    v_a = double(S.v(:));

    % Prefer the linear Gamma^2 already stored in case_5_sweep.mat
    % (max_reflection_coeff_per_port). Fall back to dBm conversion only if
    % that field is missing.
    if isfield(S, 'max_reflection_coeff_per_port')
        adap_refl_dB = 10*log10(double(real(S.max_reflection_coeff_per_port(:))));
    else
        warning('max_reflection_coeff_per_port absent; using legacy dBm conversion.');
        Z_0 = 50; nEl = 1296;
        adap_dbm = double(real(A.adaptive_obj_finals(:)));
        adap_refl_dB = 10*log10(10.^((adap_dbm - 30)/10) * Z_0 * nEl);
    end

    %% --- Interpolate onto regular grid ---
    n_grid = 100;
    u_grid = linspace(-1, 1, n_grid);
    v_grid = linspace(-1, 1, n_grid);
    [U, V] = meshgrid(u_grid, v_grid);
    outside = (U.^2 + V.^2) > 1;

    F = scatteredInterpolant(u_a, v_a, adap_refl_dB, 'natural', 'none');
    Z_a = F(U, V);

    F_nn = scatteredInterpolant(u_a, v_a, adap_refl_dB, 'nearest', 'nearest');
    nan_mask = isnan(Z_a) & ~outside;
    Z_a(nan_mask) = F_nn(U(nan_mask), V(nan_mask));
    Z_a(outside) = NaN;

    %% --- Plot with baseline Case 5 colorbar limits ---
    figure('Position', [100, 100, 650, 500]);
    imagesc(u_grid, v_grid, Z_a, 'AlphaData', ~outside);
    set(gca, 'YDir', 'normal', 'Color', [1 1 1], 'FontSize', 11);
    if ~isempty(baseline_clim_5)
        caxis(baseline_clim_5);
    end
    hold on;
    theta_circle = linspace(0, 2*pi, 361);
    plot(cos(theta_circle), sin(theta_circle), 'k-', 'LineWidth', 1.5);
    hold off;

    cb = colorbar;
    cb.Label.String = '$\Gamma^2$ (dB)';
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = 20;
    colormap(jet);
    axis equal tight;
    xlim([-1.05 1.05]); ylim([-1.05 1.05]);
    xlabel('u', 'FontSize', 14);
    ylabel('v', 'FontSize', 14);

    saveas(gcf, fullfile('./Figures', 'refl_coeff_dB_uv_case_5_adaptive_iMM.png'));
    % EPS export for paper inclusion. Matches the figure currently referenced
    % in paper/main.tex as fig:case_5_imm (case_5_imm_sweep.eps).
    print(gcf, fullfile('./Figures', 'case_5_imm_sweep.eps'), '-depsc', '-painters');
    fprintf('Adaptive iMM Case 5 plot done.\n\n');
else
    warning('Adaptive iMM or sweep file not found. Skipping.');
end

fprintf('All plots complete.\n');
