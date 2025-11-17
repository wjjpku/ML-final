import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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

def plot_training_curves(history: dict, out_dir: str):
    if len(history['steps']) == 0:
        return
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
    plot_path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def build_optimizer(model: nn.Module, cfg):
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.0)
    return optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=cfg.weight_decay)

def save_tsne_head(model: nn.Module, out_dir: str, perplexity: float, n_iter: int):
    try:
        if not _HAS_SKLEARN:
            return None
        W = model.head.weight.detach().cpu().numpy()
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, learning_rate='auto', init='random')
        Z = tsne.fit_transform(W)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.8)
        path = os.path.join(out_dir, 'tsne_head.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    except Exception:
        return None