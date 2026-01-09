import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _build_vis_suffix(cfg=None) -> str:
    """构建可视化文件名后缀"""
    if cfg is None:
        return ""
    
    suffix = f"_op{cfg.op}_p{cfg.p}_k{cfg.k}"
    
    if cfg.plot_note:
        suffix += f"_{cfg.plot_note}"
    
    return suffix

def _add_step_to_suffix(suffix: str, step) -> str:
    """在后缀中添加步数信息"""
    if step is None or step == 'final':
        step_str = '_final'
    else:
        step_str = f'_step{step}'
    
    return suffix + step_str

def plot_training_curves(history, storage_manager, cfg=None, step=None):
    """
    绘制训练曲线（Loss和Accuracy）
    
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
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = np.array(history['steps'])
    losses = np.array(history['losses'])
    train_accs = np.array(history['train_accs'])
    val_accs = np.array(history['val_accs'])
    
    # 左图：损失曲线（双对数刻度）
    axes[0].plot(steps, losses, 'b-', label='Loss', linewidth=2)
    axes[0].set_xlabel('Step (log scale)', fontsize=12)
    axes[0].set_ylabel('Loss (log scale)', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(fontsize=11)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    # 右图：准确率曲线（半对数刻度）
    axes[1].plot(steps, train_accs, 'g-', label='Train Acc', linewidth=2)
    axes[1].plot(steps, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Step (log scale)', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=11)
    axes[1].set_ylim([0, 1.05])
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存训练曲线: {filepath}")
    return filepath

def save_tsne_head(model: nn.Module, out_dir: str, cfg=None, perplexity: float = 30, 
                   n_iter: int = 1000, note: str = ""):
    """
    使用 t-SNE 可视化输出层权重
    
    文件名格式: tsne_head__op_pX__k_Y__train_ratio_Z__architecture__optimizer__[note].png
    
    Args:
        model: 训练好的模型
        out_dir: 输出目录
        cfg: 配置对象，用于生成文件名后缀
        perplexity: t-SNE perplexity 参数
        n_iter: t-SNE 迭代次数
        note: 额外的注释信息（已弃用，由 cfg.plot_note 提供）
    
    Returns:
        保存文件的路径，如果未保存则返回None
    """
    if not _HAS_SKLEARN:
        print("警告: sklearn 未安装，跳过 t-SNE 可视化")
        return None
    
    # 构建文件名后缀
    vis_suffix = _build_vis_suffix(cfg)
    
    # 构建完整文件名
    filename = f"tsne_head{vis_suffix}.png"
    
    # 获取输出层权重
    head_weight = model.head.weight.data.cpu().numpy()
    
    print("运行 t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    head_tsne = tsne.fit_transform(head_weight)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(head_tsne[:, 0], head_tsne[:, 1], c=range(len(head_tsne)), 
                        cmap='tab20', s=100, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Output Layer Weights', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Token ID', fontsize=11)
    
    plt.tight_layout()
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存 t-SNE 可视化: {filepath}")
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