from dataclasses import dataclass
import torch

@dataclass
class Config:
    p: int = 97
    train_ratio: float = 0.4
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1
    warmup_steps: int = 100
    steps: int = 20000
    batch_size: int = 0
    grad_clip: float = 1.0
    target_val_acc: float = 0.95
    seed: int = 42
    out_dir: str = "outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"