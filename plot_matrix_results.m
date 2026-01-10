% plot_matrix_results.m
% MATLAB script to visualize experiment results

% 1. 读取数据
filename = 'experiment_matrix_results.csv';
if ~isfile(filename)
    error('File %s not found. Run python run_matrix_experiments.py first.', filename);
end

opts = detectImportOptions(filename);
opts.VariableTypes{'Optimizer_Name'} = 'categorical';
T = readtable(filename, opts);

% 获取所有优化器名称
optimizers = categories(T.Optimizer_Name);
n_opt = length(optimizers);

% 2. 绘制分面图 (Subplots)
figure('Name', 'Optimizer Comparison', 'Color', 'w', 'Position', [100, 100, 1200, 800]);

n_cols = 2;
n_rows = ceil(n_opt / n_cols);

for i = 1:n_opt
    opt_name = optimizers{i};
    sub_data = T(T.Optimizer_Name == opt_name, :);
    
    % 计算均值和标准差
    [ratios, ~, idx] = unique(sub_data.Train_Ratio);
    mean_val = accumarray(idx, sub_data.Final_Val_Acc, [], @mean);
    std_val = accumarray(idx, sub_data.Final_Val_Acc, [], @std);
    mean_train = accumarray(idx, sub_data.Final_Train_Acc, [], @mean);
    std_train = accumarray(idx, sub_data.Final_Train_Acc, [], @std);
    
    subplot(n_rows, n_cols, i);
    hold on;
    
    % 绘制带误差棒的曲线
    errorbar(ratios, mean_train, std_train, '-s', 'LineWidth', 1.5, 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Train Acc');
    errorbar(ratios, mean_val, std_val, '-o', 'LineWidth', 2, 'Color', [0 0.4470 0.7410], 'DisplayName', 'Val Acc');
    
    title(char(opt_name), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Training Ratio');
    ylabel('Accuracy');
    ylim([-0.05, 1.05]);
    grid on;
    legend('Location', 'best');
    hold off;
end

sgtitle('Performance of Different Optimizers vs Training Ratio');
saveas(gcf, 'optimizer_comparison_matlab.png');

% 3. 绘制汇总对比图
figure('Name', 'All Optimizers Comparison', 'Color', 'w', 'Position', [150, 150, 800, 600]);
hold on;
colors = lines(n_opt);

for i = 1:n_opt
    opt_name = optimizers{i};
    sub_data = T(T.Optimizer_Name == opt_name, :);
    
    [ratios, ~, idx] = unique(sub_data.Train_Ratio);
    mean_val = accumarray(idx, sub_data.Final_Val_Acc, [], @mean);
    
    plot(ratios, mean_val, '-o', 'LineWidth', 2, 'Color', colors(i,:), 'DisplayName', char(opt_name));
end

title('Validation Accuracy Comparison', 'FontSize', 16);
xlabel('Training Ratio', 'FontSize', 12);
ylabel('Validation Accuracy', 'FontSize', 12);
ylim([0, 1.05]);
grid on;
legend('Location', 'southeast', 'FontSize', 10);
hold off;

saveas(gcf, 'all_optimizers_comparison_matlab.png');

disp('Plots saved as optimizer_comparison_matlab.png and all_optimizers_comparison_matlab.png');
