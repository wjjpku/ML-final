from dataclasses import dataclass
import torch

@dataclass
class Config:
    p: int = 97
    op: str = "mod_add"
    k: int = 2  # 新增：运算的元数（变量数量）
    train_ratio: float = 0.4
    architecture: str = "transformer"  # 新增：模型架构
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1
    optimizer: str = "adamw"
    warmup_steps: int = 10
    steps: int = 100000
    batch_size: int = 0
    grad_clip: float = 1.0
    target_val_acc: float = 0.95
    full_batch: bool = False
    grad_noise_std: float = 0.0
    weight_noise_std: float = 0.0
    decay_to_init: bool = False
    decay_to_init_lambda: float = 0.0
    eval_interval: int = 20
    tsne: bool = True
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    plot_note: str = ""
    plot_interval: int = 0
    seed: int = 42
    out_dir: str = "outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"