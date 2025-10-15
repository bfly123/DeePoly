%% Allen-Cahn Train vs Reference Comparison Plotting
% 这个脚本读取Python导出的train数据和参考解，生成对比图

function plot_comparison()
    % 检查是否存在Python导出的数据
    if ~exist('train_data.mat', 'file')
        fprintf('Error: train_data.mat not found. Please run Python solver first to export data.\n');
        return;
    end

    if ~exist('allen_cahn.mat', 'file')
        fprintf('Error: allen_cahn.mat not found. Please generate reference solution first.\n');
        return;
    end

    % 加载数据
    fprintf('Loading train data and reference solution...\n');
    train_data = load('train_data.mat');
    ref_data = load('allen_cahn.mat');

    % 提取数据
    t_train = train_data.time_history;
    x_train = train_data.x_coords;
    U_train = train_data.solution_history;

    t_ref = ref_data.t;
    x_ref = ref_data.x;
    U_ref = ref_data.usol;

    fprintf('Train data: %d time steps, %d spatial points\n', length(t_train), length(x_train));
    fprintf('Reference data: %d time steps, %d spatial points\n', length(t_ref), length(x_ref));

    % 1. 最终时刻对比
    plot_final_time_comparison(t_train, x_train, U_train, t_ref, x_ref, U_ref);

    % 2. 时空二维图对比
    plot_spacetime_comparison(t_train, x_train, U_train, t_ref, x_ref, U_ref);

    % 3. 误差统计分析
    compute_error_statistics(t_train, x_train, U_train, t_ref, x_ref, U_ref);

    fprintf('All plots generated successfully!\n');
end

function plot_final_time_comparison(t_train, x_train, U_train, t_ref, x_ref, U_ref)
    % 最终时刻对比
    fprintf('Generating final time comparison...\n');

    % 获取最终时刻数据
    T_final = t_train(end);
    U_final_train = U_train(:, end);

    % 插值参考解到最终时刻
    [~, idx_ref] = min(abs(t_ref - T_final));
    U_final_ref = U_ref(idx_ref, :);

    % 插值参考解到train网格
    U_ref_interp = interp1d(x_ref, U_final_ref, x_train, 'cubic');

    % 计算误差
    error = abs(U_final_train - U_ref_interp');

    % 绘图
    figure('Position', [100, 100, 800, 600]);

    % 上图：解对比
    subplot(2,1,1);
    scatter(x_train, U_final_train, 30, 'r', 'filled', 'DisplayName', 'Train Solution');
    hold on;
    plot(x_ref, U_final_ref, 'b--', 'LineWidth', 2, 'DisplayName', 'Reference Solution');
    xlabel('x');
    ylabel('u');
    title(sprintf('Final Time Comparison (T = %.4f)', T_final));
    legend('Location', 'best');
    grid on;

    % 下图：逐点误差
    subplot(2,1,2);
    scatter(x_train, error, 20, 'g', 'filled');
    xlabel('x');
    ylabel('|Error|');
    title('Pointwise Error at Final Time');
    set(gca, 'YScale', 'log');
    grid on;

    % 添加误差统计
    max_error = max(error);
    l2_error = sqrt(mean(error.^2));
    text(0.02, 0.98, sprintf('Max Error: %.2e\nL2 Error: %.2e', max_error, l2_error), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');

    % 保存图片
    saveas(gcf, 'final_time_comparison_matlab.png');
    saveas(gcf, 'final_time_comparison_matlab.fig');

    fprintf('  Final time comparison saved (Max Error: %.2e, L2 Error: %.2e)\n', max_error, l2_error);
end

function plot_spacetime_comparison(t_train, x_train, U_train, t_ref, x_ref, U_ref)
    % 时空二维图对比
    fprintf('Generating spacetime comparison...\n');

    figure('Position', [200, 100, 1200, 400]);

    % 子图1: Train解 (散点图)
    subplot(1,3,1);
    [T_mesh, X_mesh] = meshgrid(t_train, x_train);
    scatter(X_mesh(:), T_mesh(:), 10, U_train(:), 'filled');
    colorbar;
    colormap(gca, 'RdBu');
    xlabel('x');
    ylabel('t');
    title('Train Solution (Scatter)');
    axis tight;

    % 子图2: 参考解 (连续图)
    subplot(1,3,2);
    pcolor(t_ref, x_ref, U_ref');
    shading interp;
    colorbar;
    colormap(gca, 'RdBu');
    xlabel('t');
    ylabel('x');
    title('Reference Solution (Continuous)');
    axis tight;

    % 子图3: 误差分布
    subplot(1,3,3);
    % 计算误差
    error_spacetime = compute_spacetime_error(t_train, x_train, U_train, t_ref, x_ref, U_ref);
    [T_err_mesh, X_err_mesh] = meshgrid(t_train, x_train);
    scatter(X_err_mesh(:), T_err_mesh(:), 10, error_spacetime(:), 'filled');
    colorbar;
    colormap(gca, 'Reds');
    xlabel('x');
    ylabel('t');
    title('Pointwise Error (Scatter)');
    axis tight;

    % 保存图片
    saveas(gcf, 'spacetime_comparison_matlab.png');
    saveas(gcf, 'spacetime_comparison_matlab.fig');

    fprintf('  Spacetime comparison saved\n');
end

function error_matrix = compute_spacetime_error(t_train, x_train, U_train, t_ref, x_ref, U_ref)
    % 计算时空误差矩阵
    nt = length(t_train);
    nx = length(x_train);
    error_matrix = zeros(nx, nt);

    for i = 1:nt
        t_current = t_train(i);

        % 找到最近的参考时间点
        [~, idx_ref] = min(abs(t_ref - t_current));
        U_ref_current = U_ref(idx_ref, :);

        % 插值到train空间网格
        U_ref_interp = interp1d(x_ref, U_ref_current, x_train, 'cubic');

        % 计算误差
        error_matrix(:, i) = abs(U_train(:, i) - U_ref_interp');
    end
end

function compute_error_statistics(t_train, x_train, U_train, t_ref, x_ref, U_ref)
    % 计算并绘制误差统计
    fprintf('Computing error statistics...\n');

    nt = length(t_train);
    l2_errors = zeros(nt, 1);
    linf_errors = zeros(nt, 1);

    for i = 1:nt
        t_current = t_train(i);

        % 找到最近的参考时间点
        [~, idx_ref] = min(abs(t_ref - t_current));
        U_ref_current = U_ref(idx_ref, :);

        % 插值到train空间网格
        U_ref_interp = interp1d(x_ref, U_ref_current, x_train, 'cubic');

        % 计算误差
        error = abs(U_train(:, i) - U_ref_interp');
        l2_errors(i) = sqrt(mean(error.^2));
        linf_errors(i) = max(error);
    end

    % 绘制误差演化
    figure('Position', [300, 100, 800, 600]);

    subplot(2,1,1);
    scatter(t_train, l2_errors, 30, 'b', 'filled');
    xlabel('Time');
    ylabel('L2 Error (log scale)');
    title('L2 Error Evolution (Train Data Points)');
    set(gca, 'YScale', 'log');
    grid on;

    subplot(2,1,2);
    scatter(t_train, linf_errors, 30, 'r', 'filled');
    xlabel('Time');
    ylabel('L∞ Error (log scale)');
    title('L∞ Error Evolution (Train Data Points)');
    set(gca, 'YScale', 'log');
    grid on;

    % 保存图片
    saveas(gcf, 'error_evolution_matlab.png');
    saveas(gcf, 'error_evolution_matlab.fig');

    % 计算统计量
    mean_l2 = mean(l2_errors);
    mean_linf = mean(linf_errors);
    max_l2 = max(l2_errors);
    max_linf = max(linf_errors);

    % 保存统计报告
    fid = fopen('error_statistics_matlab.txt', 'w');
    fprintf(fid, '============================================================\n');
    fprintf(fid, 'Allen-Cahn Train Data Error Statistics (MATLAB Generated)\n');
    fprintf(fid, '============================================================\n\n');
    fprintf(fid, 'Time range: [%.6f, %.6f]\n', t_train(1), t_train(end));
    fprintf(fid, 'Total time steps: %d\n', nt);
    fprintf(fid, 'Spatial points: %d\n\n', length(x_train));
    fprintf(fid, 'Time-averaged Errors:\n');
    fprintf(fid, '------------------------------\n');
    fprintf(fid, 'Mean L2 Error:   %.6e\n', mean_l2);
    fprintf(fid, 'Mean L∞ Error:   %.6e\n\n', mean_linf);
    fprintf(fid, 'Maximum Errors:\n');
    fprintf(fid, '------------------------------\n');
    fprintf(fid, 'Max L2 Error:    %.6e\n', max_l2);
    fprintf(fid, 'Max L∞ Error:    %.6e\n', max_linf);
    fclose(fid);

    fprintf('  Error statistics saved\n');
    fprintf('  Time-averaged L2 Error:  %.6e\n', mean_l2);
    fprintf('  Time-averaged L∞ Error:  %.6e\n', mean_linf);
    fprintf('  Maximum L2 Error:        %.6e\n', max_l2);
    fprintf('  Maximum L∞ Error:        %.6e\n', max_linf);
end

function y_interp = interp1d(x, y, x_new, method)
    % 简单的1D插值函数
    if nargin < 4
        method = 'linear';
    end

    switch method
        case 'linear'
            y_interp = interp1(x, y, x_new, 'linear', 'extrap');
        case 'cubic'
            y_interp = interp1(x, y, x_new, 'cubic', 'extrap');
        otherwise
            y_interp = interp1(x, y, x_new, 'linear', 'extrap');
    end
end

% 执行主函数
plot_comparison();