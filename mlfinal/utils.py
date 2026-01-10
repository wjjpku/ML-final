import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import random


def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_vis_suffix(cfg=None) -> str:
    """
    构建可视化文件名后缀，包含详细参数
    
    格式: __op__{op}__p{p}__k{k}__train_ratio_{train_ratio}__arch_{architecture}__opt_{optimizer}
    """
    if cfg is None:
        return ""
    
    suffix = f"__op__{cfg.op}__p{cfg.p}__k{cfg.k}__train_ratio_{cfg.train_ratio:.2f}__arch_{cfg.architecture}__opt_{cfg.optimizer}"
    
    return suffix


def _add_step_to_suffix(suffix: str, step) -> str:
    """在后缀中添加步数信息"""
    if step is None or step == 'final':
        step_str = '__step_final'
    else:
        step_str = f'__step_{step}'
    
    return suffix + step_str


def plot_training_curves(history, storage_manager, cfg=None, step=None):
    """
    绘制训练曲线（Accuracy曲线）
    
    Args:
        history (dict): 训练历史 {'steps': [...], 'losses': [...], 'train_accs': [...], 'val_accs': [...]}
        storage_manager: DataStorageManager 对象，用于获取保存目录
        cfg: 配置对象
        step: 当前步数或'final'
    
    Returns:
        保存文件的路径，如果未保存则返回 None
    """
    if len(history['steps']) == 0:
        return None
    
    # 使用 storage_manager 提供的目录作为输出目录
    out_dir = storage_manager.get_data_dir()
    os.makedirs(out_dir, exist_ok=True)
    
    # 构建文件名
    vis_suffix = _build_vis_suffix(cfg)
    vis_suffix = _add_step_to_suffix(vis_suffix, step)
    filename = f"training_curves{vis_suffix}.png"
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = np.array(history['steps'])
    train_accs = np.array(history['train_accs'])
    val_accs = np.array(history['val_accs'])
    
    # 绘制准确率曲线
    ax.plot(steps, train_accs, 'g-', label='Train Acc', linewidth=2.5, marker='o', markersize=4)
    ax.plot(steps, val_accs, 'r-', label='Val Acc', linewidth=2.5, marker='s', markersize=4)
    
    # 设置横坐标：以2为底数的指数间距
    # 找到合适的刻度范围
    min_step = steps.min()
    max_step = steps.max()
    
    # 计算以2为底的对数范围
    log2_min = np.log2(max(min_step, 1))
    log2_max = np.log2(max_step)
    
    # 生成以2的幂次为刻度的位置
    tick_positions = []
    tick_labels = []
    power = int(np.floor(log2_min))
    while 2 ** power <= max_step:
        tick_pos = 2 ** power
        if tick_pos >= min_step:
            tick_positions.append(tick_pos)
            tick_labels.append(str(int(tick_pos)))
        power += 1
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xscale('log', base=2)
    
    # 设置纵坐标：正常刻度
    ax.set_ylim([0, 1.05])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # 标签和标题
    ax.set_xlabel('Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    
    # 构建标题，包含所有配置参息
    if cfg:
        title = f"Training Accuracy Curves\n"
        title += f"Op: {cfg.op}, p: {cfg.p}, k: {cfg.k}, train_ratio: {cfg.train_ratio:.2f}, "
        title += f"arch: {cfg.architecture}, optimizer: {cfg.optimizer}"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    else:
        ax.set_title('Training Accuracy Curves', fontsize=13, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存训练曲线: {filepath}")
    return filepath


def build_optimizer(model, cfg):
    """
    构建优化器
    
    Args:
        model: 模型
        cfg: 配置对象
    
    Returns:
        优化器对象
    """
    optimizer_type = cfg.optimizer.lower()
    
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer