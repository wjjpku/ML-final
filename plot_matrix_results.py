import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(csv_file='experiment_matrix_results.csv', output_file='optimizer_comparison.png'):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run run_matrix_experiments.py first.")
        return

    df = pd.read_csv(csv_file)
    
    # 设置风格
    sns.set(style="whitegrid")
    
    # 获取优化器列表
    optimizers = df['Optimizer_Name'].unique()
    n_opt = len(optimizers)
    
    # 创建大图
    # 假设每行2个图
    n_cols = 2
    n_rows = (n_opt + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
    axes = axes.flatten()
    
    # 遍历优化器绘制子图
    for i, opt_name in enumerate(optimizers):
        ax = axes[i]
        data = df[df['Optimizer_Name'] == opt_name]
        
        # 绘制 验证集准确率
        sns.lineplot(
            data=data, 
            x='Train_Ratio', 
            y='Final_Val_Acc', 
            marker='o', 
            label='Val Acc',
            ax=ax,
            color='blue',
            errorbar='sd' # 显示标准差
        )
        
        # 绘制 训练集准确率
        sns.lineplot(
            data=data, 
            x='Train_Ratio', 
            y='Final_Train_Acc', 
            marker='s', 
            label='Train Acc',
            ax=ax,
            color='green',
            linestyle='--',
            errorbar='sd'
        )
        
        ax.set_title(f"Optimizer: {opt_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Ratio", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
    
    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # 额外图表：所有优化器对比
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x='Train_Ratio', 
        y='Final_Val_Acc', 
        hue='Optimizer_Name', 
        marker='o',
        errorbar=None # 对比图不显示误差棒以免混乱
    )
    plt.title("All Optimizers Comparison (Val Acc)", fontsize=16)
    plt.xlabel("Training Ratio", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.savefig('all_optimizers_comparison.png', dpi=300)
    print(f"Comparison plot saved to all_optimizers_comparison.png")

if __name__ == "__main__":
    plot_results()
