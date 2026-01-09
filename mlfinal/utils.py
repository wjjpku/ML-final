import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _build_vis_suffix(cfg=None) -> str:
    """
    构建可视化文件后缀，包含所有相关参数信息
    
    格式: op_pX__train_ratio_Y__architecture__optimizer__note__step_Z
    例如: mod_add_p97__train_ratio_0.4__transformer__adamw__exp1__step_3000
    
    Args:
        cfg: 配置对象，包含所有参数信息
    
    Returns:
        文件名后缀字符串，或空字符串（如果 cfg 为 None）
    """
    if cfg is None:
        return ""
    
    # 构建基础信息：操作和模数
    op_info = f"{cfg.op}_p{cfg.p}"
    
    # 构建训练比例信息
    train_ratio_info = f"train_ratio_{cfg.train_ratio}"
    
    # 构建架构和优化器信息
    architecture = getattr(cfg, 'architecture', 'transformer').lower()
    optimizer = getattr(cfg, 'optimizer', 'adamw').lower()
    
    # 构建 note 信息（如果存在）
    note = getattr(cfg, 'plot_note', "").lower().strip()
    
    # 拼接所有部分
    suffix_parts = [op_info, train_ratio_info, architecture, optimizer]
    
    # 添加 note（如果不为空）
    if note:
        suffix_parts.append(note)
    
    return "__" + "__".join(suffix_parts)


def _add_step_to_suffix(suffix: str, step: int) -> str:
    """
    将步数信息添加到后缀末尾
    
    Args:
        suffix: 已有的后缀
        step: 当前步数
    
    Returns:
        添加了步数的后缀
    """
    if step is not None:
        suffix += f"__step_{step}"
    return suffix


def plot_training_curves(history: dict, out_dir: str, cfg=None, step=None, note: str = ""):
    """
    绘制训练曲线（损失和准确率）
    
    文件名格式: training_curves__op_pX__train_ratio_Y__architecture__optimizer__[note__]step_Z.png
    
    Args:
        history: 包含训练历史的字典
        out_dir: 输出目录
        cfg: 配置对象，用于生成文件名后缀
        step: 当前步数（可选，用于动态保存）
        note: 额外的注释信息（已弃用，由 cfg.plot_note 提供）
    """
    if len(history['steps']) == 0:
        return
    
    # 构建文件名后缀
    vis_suffix = _build_vis_suffix(cfg)
    
    # 添加步数信息
    vis_suffix = _add_step_to_suffix(vis_suffix, step)
    
    # 构建完整文件名
    filename = f"training_curves{vis_suffix}.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['steps'], history['losses'], 'b-', label='Loss', linewidth=2)
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_yscale('log')
    
    axes[1].plot(history['steps'], history['train_accs'], 'g-', label='Train Acc', linewidth=2)
    axes[1].plot(history['steps'], history['val_accs'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存训练曲线: {filepath}")

def save_tsne_head(model: nn.Module, out_dir: str, cfg=None, perplexity: float = 30, 
                   n_iter: int = 1000, note: str = ""):
    """
    使用 t-SNE 可视化输出层权重
    
    文件名格式: tsne_head__op_pX__train_ratio_Y__architecture__optimizer__[note__]step_Z.png
    
    Args:
        model: 训练好的模型
        out_dir: 输出目录
        cfg: 配置对象，用于生成文件名后缀
        perplexity: t-SNE perplexity 参数
        n_iter: t-SNE 迭代次数
        note: 额外的注释信息（已弃用，由 cfg.plot_note 提供）
    """
    if not _HAS_SKLEARN:
        print("警告: sklearn 未安装，跳过 t-SNE 可视化")
        return
    
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

def build_optimizer(model: nn.Module, cfg):
    """
    构建优化器
    
    Args:
        model: 神经网络模型
        cfg: 配置对象
    
    Returns:
        优化器实例
    """
    optimizer_name = cfg.optimizer.lower()
    
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.98)
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.98)
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"不支持的优化器: {cfg.optimizer}")