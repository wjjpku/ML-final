import os
import csv
import torch
import pandas as pd
from datetime import datetime
from mlfinal.config import Config
from mlfinal.trainer import train_loop

def run_matrix_experiments():
    # 实验配置矩阵
    optimizers = [
        {'name': 'AdamW (beta1=0.9)', 'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 1.0, 'momentum': 0.9},
        {'name': 'AdamW (beta1=0.5)', 'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 1.0, 'momentum': 0.5},
        {'name': 'AdamW (beta1=0.0)', 'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 1.0, 'momentum': 0.0},
        {'name': 'SGD', 'optimizer': 'sgd', 'lr': 1e-2, 'weight_decay': 0.0, 'momentum': 0.0},
        {'name': 'SGD+Momentum', 'optimizer': 'sgd', 'lr': 1e-2, 'weight_decay': 0.0, 'momentum': 0.9},
        {'name': 'RMSprop', 'optimizer': 'rmsprop', 'lr': 1e-3, 'weight_decay': 0.0, 'momentum': 0.0},
    ]
    
    train_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    seeds = [42, 43, 44]  # 重复3次以获取误差棒
    
    # 固定参数
    common_params = {
        'p': 97,
        'op': 'mod_add',
        'steps': 3000,
        'enable_visualization': False,  # 矩阵实验不需要每次都画Loss图，节省时间
        'data_storage_dir': None, # 让它自动创建
        'plot_interval': 0,
        'target_val_acc': 1,
        'eval_interval': 20
    }
    
    # 结果存储文件
    results_file = 'experiment_matrix_results.csv'
    
    # 初始化CSV头
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Optimizer_Name', 'Optimizer_Type', 'LR', 'Weight_Decay', 'Momentum',
                'Train_Ratio', 'Seed', 'Final_Train_Acc', 'Final_Val_Acc', 'Steps', 'Data_Dir'
            ])
    
    print(f"开始矩阵实验，总计 {len(optimizers) * len(train_ratios) * len(seeds)} 次运行...")
    
    count = 0
    total = len(optimizers) * len(train_ratios) * len(seeds)
    
    for opt_conf in optimizers:
        for ratio in train_ratios:
            for seed in seeds:
                count += 1
                print(f"\n[{count}/{total}] Running: {opt_conf['name']}, Ratio={ratio}, Seed={seed}")
                
                # 检查是否已经跑过（可选，简单的断点续传）
                # 这里略过，直接跑
                
                cfg = Config(
                    optimizer=opt_conf['optimizer'],
                    lr=opt_conf['lr'],
                    weight_decay=opt_conf['weight_decay'],
                    momentum=opt_conf['momentum'],
                    train_ratio=ratio,
                    seed=seed,
                    **common_params
                )
                
                try:
                    result = train_loop(cfg)
                    
                    # 写入结果
                    with open(results_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            opt_conf['name'],
                            opt_conf['optimizer'],
                            opt_conf['lr'],
                            opt_conf['weight_decay'],
                            opt_conf['momentum'],
                            ratio,
                            seed,
                            result['train_acc'],
                            result['val_acc'],
                            result['step'],
                            result['data_dir']
                        ])
                        
                except Exception as e:
                    print(f"Error running experiment: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\n所有实验完成。结果已保存至 {results_file}")

if __name__ == "__main__":
    run_matrix_experiments()
