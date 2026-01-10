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

def run_reg_study():
    """
    研究正则化 (Weight Decay, Dropout) 对 Grokking (泛化延迟) 的影响
    独立分析 Weight Decay 和 Dropout
    """
    
    # 实验配置
    seeds = [42]
    common_params = {
        'p': 97,
        'op': 'mod_add',
        'train_ratio': 0.3, # 较难，易发生 Grokking
        'optimizer': 'adamw',
        'lr': 1e-3,
        'steps': 20000, 
        'enable_visualization': False, 
        'plot_interval': 0,
        'data_storage_dir': None, 
        'eval_interval': 100,
        'target_val_acc': 0.95
    }
    
    # 定义日志列表用于后续绘图
    # 结构: {'Study_Type': 'WD'/'Dropout', 'Param_Value': float, 'log_path': str}
    experiment_logs = []
    
    print("开始正则化与 Grokking 研究...")
    
    # --- Study 1: Weight Decay ---
    # 控制变量: Dropout=0.0 (为了更纯粹地看 WD 影响) 或者 Baseline 0.1
    # 假设控制 Dropout=0.0 以隔离变量
    fixed_drop = 0.0
    wds = [0.0, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5] # 8个点
    
    print("\n[Study 1] Varying Weight Decay (Fixed Dropout=0.0)")
    for wd in wds:
        for seed in seeds:
            print(f"Running: WD={wd}, Drop={fixed_drop}")
            cfg = Config(
                weight_decay=wd,
                dropout=fixed_drop,
                seed=seed,
                **common_params
            )
            try:
                result = train_loop(cfg)
                log_path = os.path.join(result['data_dir'], 'training_log.csv')
                experiment_logs.append({
                    'Study_Type': 'WD_Study',
                    'Param_Value': wd,
                    'log_path': log_path
                })
            except Exception as e:
                print(f"Error: {e}")

    # --- Study 2: Dropout ---
    # 控制变量: WD=1.0 (Baseline)
    fixed_wd = 1.0
    drops = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 8个点
    
    print("\n[Study 2] Varying Dropout (Fixed WD=1.0)")
    for drop in drops:
        for seed in seeds:
            print(f"Running: WD={fixed_wd}, Drop={drop}")
            cfg = Config(
                weight_decay=fixed_wd,
                dropout=drop,
                seed=seed,
                **common_params
            )
            try:
                result = train_loop(cfg)
                log_path = os.path.join(result['data_dir'], 'training_log.csv')
                experiment_logs.append({
                    'Study_Type': 'Drop_Study',
                    'Param_Value': drop,
                    'log_path': log_path
                })
            except Exception as e:
                print(f"Error: {e}")
                
    return experiment_logs

def calculate_grokking_step(log_path, target_acc=0.95):
    """从日志中计算达到目标准确率的步数"""
    if not os.path.exists(log_path):
        return 20000 # Max steps
        
    try:
        df = pd.read_csv(log_path)
        df['Val_Acc'] = pd.to_numeric(df['Val_Acc'], errors='coerce')
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
        df = df.dropna(subset=['Val_Acc', 'Step'])
        
        # 找到第一个 >= target_acc 的行
        success_rows = df[df['Val_Acc'] >= target_acc]
        if not success_rows.empty:
            return success_rows.iloc[0]['Step']
        else:
            return 20000 # Max steps / Fail
    except:
        return 20000

def plot_reg_study(experiment_logs):
    """
    绘制正则化参数 vs Grokking Step
    """
    if not experiment_logs:
        print("无实验数据")
        return

    # 提取数据
    data = []
    for exp in experiment_logs:
        step = calculate_grokking_step(exp['log_path'])
        data.append({
            'Study_Type': exp['Study_Type'],
            'Param_Value': exp['Param_Value'],
            'Grok_Step': step
        })
    
    df = pd.DataFrame(data)
    
    # 阈值
    FAIL_THRESHOLD = 10000 # 超过此步数视为 Grokking 失败/太慢
    
    def process_data(d):
        d['Is_Fail'] = d['Grok_Step'] >= FAIL_THRESHOLD
        d['Plot_Step'] = d['Grok_Step'].clip(upper=FAIL_THRESHOLD)
        return d
    
    df_wd = process_data(df[df['Study_Type'] == 'WD_Study'].copy())
    df_drop = process_data(df[df['Study_Type'] == 'Drop_Study'].copy())
    
    sns.set(style="whitegrid")
    
    # --- 图 1: Weight Decay ---
    plt.figure(figsize=(8, 6))
    success = df_wd[~df_wd['Is_Fail']]
    fail = df_wd[df_wd['Is_Fail']]
    
    if not success.empty:
        plt.plot(success['Param_Value'], success['Plot_Step'], 'bo-', label='Success (<10k steps)')
    if not fail.empty:
        plt.plot(fail['Param_Value'], fail['Plot_Step'], 'r^', markersize=10, label='Failed (>=10k steps)')
        
    plt.xlabel('Weight Decay', fontsize=12)
    plt.ylabel('Steps to reach 95% Acc', fontsize=12)
    plt.title('Effect of Weight Decay on Grokking Speed', fontsize=14)
    plt.ylim(0, FAIL_THRESHOLD * 1.1)
    plt.axhline(y=FAIL_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('reg_wd_study.png', dpi=150)
    print("Saved reg_wd_study.png")
    
    # --- 图 2: Dropout ---
    plt.figure(figsize=(8, 6))
    success = df_drop[~df_drop['Is_Fail']]
    fail = df_drop[df_drop['Is_Fail']]
    
    if not success.empty:
        plt.plot(success['Param_Value'], success['Plot_Step'], 'go-', label='Success (<10k steps)')
    if not fail.empty:
        plt.plot(fail['Param_Value'], fail['Plot_Step'], 'r^', markersize=10, label='Failed (>=10k steps)')
        
    plt.xlabel('Dropout Rate', fontsize=12)
    plt.ylabel('Steps to reach 95% Acc', fontsize=12)
    plt.title('Effect of Dropout on Grokking Speed', fontsize=14)
    plt.ylim(0, FAIL_THRESHOLD * 1.1)
    plt.axhline(y=FAIL_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('reg_drop_study.png', dpi=150)
    print("Saved reg_drop_study.png")

if __name__ == "__main__":
    logs = run_reg_study()
    plot_reg_study(logs)
