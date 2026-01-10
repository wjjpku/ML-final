import os
import csv
import torch
import pandas as pd
from datetime import datetime
from mlfinal.config import Config
from mlfinal.trainer import train_loop
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_sgd_study():
    """
    研究 SGD 在不同 Learning Rate 和 Batch Size 下的表现 (独立分析)
    """
    
    # 实验配置
    seeds = [42]
    common_params = {
        'p': 97,
        'op': 'mod_add',
        'train_ratio': 0.5,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'steps': 20000,
        'enable_visualization': False,
        'plot_interval': 0,
        'data_storage_dir': None,
        'target_val_acc': 0.95
    }
    
    results_file = 'sgd_study_results.csv'
    
    # 初始化 CSV
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Study_Type', 'Param_Value', 'Seed', 
                'Final_Train_Acc', 'Final_Val_Acc', 'Steps'
            ])
            
    print(f"开始 SGD 独立参数研究...")
    
    # --- 实验 1: 固定 Batch Size，变化 LR ---
    # 8个点
    lrs = [0.2, 0.1, 0.08, 0.05, 0.03, 0.01, 0.005, 0.001]
    fixed_bs = 64
    
    print("\n[Study 1] Varying LR (Fixed Batch Size = 64)")
    for lr in lrs:
        for seed in seeds:
            print(f"Running: LR={lr}, Batch={fixed_bs}")
            cfg = Config(
                lr=lr,
                batch_size=fixed_bs,
                full_batch=False,
                seed=seed,
                **common_params
            )
            try:
                result = train_loop(cfg)
                with open(results_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'LR_Study', lr, seed,
                        result['train_acc'], result['val_acc'], result['step']
                    ])
            except Exception as e:
                print(f"Error: {e}")

    # --- 实验 2: 固定 LR，变化 Batch Size ---
    # 8个点
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 0] # 0 = Full
    fixed_lr = 0.05
    
    print("\n[Study 2] Varying Batch Size (Fixed LR = 0.05)")
    for bs in batch_sizes:
        for seed in seeds:
            bs_val = bs if bs != 0 else 'Full'
            print(f"Running: LR={fixed_lr}, Batch={bs_val}")
            
            is_full = (bs == 0)
            cfg = Config(
                lr=fixed_lr,
                batch_size=bs,
                full_batch=is_full,
                seed=seed,
                **common_params
            )
            try:
                result = train_loop(cfg)
                # 如果是 Full Batch，我们需要记录一个数值以便绘图
                # 假设 Full Batch 大小约为 train_data size (约 4500 * 0.5 = 2250)
                # 但为了绘图方便，我们可以用字符串 'Full' 或者一个大数值，或者就在绘图时处理
                # 这里我们存真实数值或特殊标记。为了方便 CSV 读取，存 -1 代表 Full? 或者就在这里获取真实 batch size?
                # train_loop 不返回真实 batch size。
                # 简单起见，我们存 -1 表示 Full，绘图时手动标
                param_val = bs if bs != 0 else -1 
                
                with open(results_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'BS_Study', param_val, seed,
                        result['train_acc'], result['val_acc'], result['step']
                    ])
            except Exception as e:
                print(f"Error: {e}")

    print(f"SGD 研究完成。结果保存在 {results_file}")

def plot_sgd_study():
    """绘制 SGD 研究结果 (两张独立图)"""
    if not os.path.exists('sgd_study_results.csv'):
        print("未找到结果文件")
        return

    df = pd.read_csv('sgd_study_results.csv')
    
    # 阈值
    FAIL_THRESHOLD = 10000
    
    # 分割数据
    df_lr = df[df['Study_Type'] == 'LR_Study'].copy()
    df_bs = df[df['Study_Type'] == 'BS_Study'].copy()
    
    # 处理 Steps：超过阈值视为失败
    # 我们创建一个新的列 'Plot_Steps'，超过阈值的设为阈值
    # 另外创建一个 mask 用于标记失败点
    
    def process_data(data):
        data['Is_Fail'] = data['Steps'] >= FAIL_THRESHOLD
        data['Plot_Steps'] = data['Steps'].clip(upper=FAIL_THRESHOLD)
        return data

    df_lr = process_data(df_lr)
    df_bs = process_data(df_bs)
    
    sns.set(style="whitegrid")
    
    # --- 图 1: LR vs Steps ---
    plt.figure(figsize=(8, 6))
    
    # 成功点
    success_data = df_lr[~df_lr['Is_Fail']]
    if not success_data.empty:
        plt.plot(success_data['Param_Value'], success_data['Plot_Steps'], 'bo-', label='Success (<10k steps)')
    
    # 失败点
    fail_data = df_lr[df_lr['Is_Fail']]
    if not fail_data.empty:
        plt.plot(fail_data['Param_Value'], fail_data['Plot_Steps'], 'r^', markersize=10, label='Failed (>=10k steps)')
        
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)', fontsize=12)
    plt.ylabel('Steps to reach 95% Acc', fontsize=12)
    plt.title('SGD: Effect of Learning Rate (Batch=64)', fontsize=14)
    plt.ylim(0, FAIL_THRESHOLD * 1.1)
    plt.axhline(y=FAIL_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('sgd_lr_study.png', dpi=150)
    print("Saved sgd_lr_study.png")
    
    # --- 图 2: Batch Size vs Steps ---
    plt.figure(figsize=(8, 6))
    
    # 处理 Full Batch (-1)
    # 将 -1 替换为实际的大数值用于绘图，或者作为分类轴
    # 让我们假设 Full Batch 约为 2000 (取决于 p=97, train_ratio=0.5, total~97*97/2=4700 -> 2350)
    # 为了绘图连续性，我们可以把 -1 映射为 2500 并在轴上标注 'Full'
    
    # 简单起见，我们把 Param_Value 视为数值，如果是 -1，改为 2500
    df_bs['Plot_Param'] = df_bs['Param_Value'].apply(lambda x: 2500 if x == -1 else x)
    
    # 成功点
    success_data = df_bs[~df_bs['Is_Fail']]
    if not success_data.empty:
        plt.plot(success_data['Plot_Param'], success_data['Plot_Steps'], 'go-', label='Success (<10k steps)')
        
    # 失败点
    fail_data = df_bs[df_bs['Is_Fail']]
    if not fail_data.empty:
        plt.plot(fail_data['Plot_Param'], fail_data['Plot_Steps'], 'r^', markersize=10, label='Failed (>=10k steps)')

    plt.xscale('log')
    plt.xlabel('Batch Size (log scale)', fontsize=12)
    plt.ylabel('Steps to reach 95% Acc', fontsize=12)
    plt.title('SGD: Effect of Batch Size (LR=0.05)', fontsize=14)
    plt.ylim(0, FAIL_THRESHOLD * 1.1)
    plt.axhline(y=FAIL_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    
    # 自定义 X 轴刻度以显示 Full
    ticks = [16, 32, 64, 128, 256, 512, 1024, 2500]
    labels = ['16', '32', '64', '128', '256', '512', '1024', 'Full']
    plt.xticks(ticks, labels, rotation=45)
    
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('sgd_bs_study.png', dpi=150)
    print("Saved sgd_bs_study.png")

if __name__ == "__main__":
    run_sgd_study()
    plot_sgd_study()
