import os
import csv
from datetime import datetime


class DataStorageManager:
    """数据存储管理器，负责将训练数据存储为 CSV"""
    
    def __init__(self, data_dir=None):
        """
        初始化数据存储管理器
        
        Args:
            data_dir (str): 数据存储目录。如果为 None，自动创建带时间戳的目录
        """
        if data_dir is None:
            # 默认创建带时间戳的目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_dir = os.path.join('data_storage', timestamp)
        
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # CSV 文件路径
        self.csv_path = os.path.join(data_dir, 'training_log.csv')
        
        # 初始化 CSV 文件（如果不存在）
        self._init_csv()
    
    def _init_csv(self):
        """初始化 CSV 文件，如果不存在则创建并添加表头"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'Step',
                    'Train_Acc',
                    'Val_Acc',
                    'Train_Loss',
                    'Val_Loss',
                    'Status',
                    'Success_Step',
                    'Final_Train_Acc',
                    'Final_Val_Acc',
                    'Final_Loss'
                ])
    
    def save_config(self, cfg, command_line_args=None):
        """
        保存配置信息到 txt 文件
        
        Args:
            cfg: 配置对象
            command_line_args (str): 命令行参数字符串
        """
        config_path = os.path.join(self.data_dir, 'config.txt')
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write('训练配置信息\n')
            f.write('='*80 + '\n\n')
            
            # 记录命令行
            if command_line_args:
                f.write('命令行参数:\n')
                f.write('-'*80 + '\n')
                f.write(command_line_args + '\n')
                f.write('-'*80 + '\n\n')
            
            # 记录所有配置参数
            f.write('详细配置参数:\n')
            f.write('-'*80 + '\n')
            
            # 数据参数
            f.write('\n【数据参数】\n')
            f.write(f"  运算类型 (op): {cfg.op}\n")
            f.write(f"  模数 (p): {cfg.p}\n")
            f.write(f"  运算元数 (k): {cfg.k}\n")
            f.write(f"  训练集比例 (train_ratio): {cfg.train_ratio}\n")
            
            # 模型参数
            f.write('\n【模型参数】\n')
            f.write(f"  模型架构 (architecture): {cfg.architecture}\n")
            f.write(f"  隐藏维度 (d_model): {cfg.d_model}\n")
            f.write(f"  层数 (n_layers): {cfg.n_layers}\n")
            f.write(f"  注意力头数 (n_heads): {cfg.n_heads}\n")
            f.write(f"  Dropout比例 (dropout): {cfg.dropout}\n")
            
            # 优化器参数
            f.write('\n【优化器参数】\n')
            f.write(f"  优化器类型 (optimizer): {cfg.optimizer}\n")
            f.write(f"  学习率 (lr): {cfg.lr}\n")
            f.write(f"  权重衰减 (weight_decay): {cfg.weight_decay}\n")
            f.write(f"  预热步数 (warmup_steps): {cfg.warmup_steps}\n")
            
            # 训练参数
            f.write('\n【训练参数】\n')
            f.write(f"  总步数 (steps): {cfg.steps}\n")
            f.write(f"  批处理大小 (batch_size): {cfg.batch_size}\n")
            f.write(f"  全批次训练 (full_batch): {cfg.full_batch}\n")
            f.write(f"  评估间隔 (eval_interval): {cfg.eval_interval}\n")
            f.write(f"  目标验证准确率 (target_val_acc): {cfg.target_val_acc}\n")
            f.write(f"  梯度剪裁 (grad_clip): {cfg.grad_clip}\n")
            
            # 正则化与噪声
            f.write('\n【正则化与噪声】\n')
            f.write(f"  梯度噪声标准差 (grad_noise_std): {cfg.grad_noise_std}\n")
            f.write(f"  权重噪声标准差 (weight_noise_std): {cfg.weight_noise_std}\n")
            f.write(f"  衰减到初始值 (decay_to_init): {cfg.decay_to_init}\n")
            f.write(f"  衰减系数 (decay_to_init_lambda): {cfg.decay_to_init_lambda}\n")
            
            # 可视化参数
            f.write('\n【可视化参数】\n')
            f.write(f"  启用可视化 (enable_visualization): {cfg.enable_visualization}\n")
            f.write(f"  绘图间隔 (plot_interval): {cfg.plot_interval}\n")
            f.write(f"  图表注释 (plot_note): {cfg.plot_note}\n")
            
            # 其他参数
            f.write('\n【其他参数】\n')
            f.write(f"  随机种子 (seed): {cfg.seed}\n")
            f.write(f"  设备 (device): {cfg.device}\n")
            
            f.write('\n' + '='*80 + '\n')
            f.write(f'配置保存时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('='*80 + '\n')
        
        print(f"配置已保存到: {config_path}")
        return config_path
    
    def log_evaluation(self, step, train_acc, val_acc, train_loss, val_loss):
        """
        记录单次评估数据到 CSV
        
        Args:
            step (int): 当前步数
            train_acc (float): 训练准确率
            val_acc (float): 验证准确率
            train_loss (float): 训练损失
            val_loss (float): 验证损失
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                step,
                f'{train_acc:.6f}',
                f'{val_acc:.6f}',
                f'{train_loss:.6f}',
                f'{val_loss:.6f}',
                '',  # Status - 留空
                '',  # Success_Step
                '',  # Final_Train_Acc
                '',  # Final_Val_Acc
                '',  # Final_Loss
            ])
        
        # 同时输出到终端
        print(f"步数: {step:6d} | 训练准确率: {train_acc:.6f} | 验证准确率: {val_acc:.6f} | 损失: {train_loss:.6f}")
    
    def log_success(self, success_step, train_acc, val_acc, loss):
        """
        记录训练成功的结果
        
        Args:
            success_step (int): 成功时的步数
            train_acc (float): 成功时的训练准确率
            val_acc (float): 成功时的验证准确率
            loss (float): 成功时的损失
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                success_step,
                f'{train_acc:.6f}',
                f'{val_acc:.6f}',
                f'{loss:.6f}',
                f'{loss:.6f}',
                '成功',
                success_step,
                f'{train_acc:.6f}',
                f'{val_acc:.6f}',
                f'{loss:.6f}',
            ])
        
        # 同时输出到终端
        print(f"\n{'='*80}")
        print(f"[成功] 步数: {success_step} | 训练准确率: {train_acc:.6f} | 验证准确率: {val_acc:.6f} | 损失: {loss:.6f}")
        print(f"{'='*80}")
    
    def log_failure(self, final_step, train_acc, val_acc, loss):
        """
        记录训练失败/结束的结果
        
        Args:
            final_step (int): 最终步数
            train_acc (float): 最终训练准确率
            val_acc (float): 最终验证准确率
            loss (float): 最终损失
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                final_step,
                f'{train_acc:.6f}',
                f'{val_acc:.6f}',
                f'{loss:.6f}',
                f'{loss:.6f}',
                '失败',
                '',
                f'{train_acc:.6f}',
                f'{val_acc:.6f}',
                f'{loss:.6f}',
            ])
        
        # 同时输出到终端
        print(f"\n{'='*80}")
        print(f"[失败] 步数: {final_step} | 训练准确率: {train_acc:.6f} | 验证准确率: {val_acc:.6f} | 最终损失: {loss:.6f}")
        print(f"{'='*80}")
    
    def get_csv_path(self):
        """获取 CSV 文件路径"""
        return self.csv_path
    
    def get_data_dir(self):
        """获取数据存储目录"""
        return self.data_dir