import os
import random
from dataclasses import dataclass, asdict
# import argparse # Removed argparse import

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，兼容服务器环境


# =========================================================
# 模加法数据集 + 解码器型 Transformer（因果掩码）
# - 序列形式：⟨x⟩ ⟨op⟩ ⟨y⟩ ⟨=⟩ ⟨x + y mod p⟩，仅在最后一位计算损失与准确率。
# - 数据：模 p 加法，p=97，共 97×97 = 9409 个样本。
# - 模型参数：层数 2、隐藏宽度 128、注意力头数 4、残差 dropout 可设。
# - 优化器：默认 AdamW(lr=1e-3, wd=0.1, betas=(0.9,0.98))，线性预热。
# =========================================================


# ------------------------- 配置 -------------------------
@dataclass
class Config:
    p: int = 97                                 # 模数 p
    train_ratio: float = 0.4                    # 训练集比例（其余为验证集）

    d_model: int = 128                          # 隐藏维度
    n_layers: int = 2                           # 层数
    n_heads: int = 4                            # 注意力头数
    dropout: float = 0.1                        # 残差/注意力/嵌入 dropout

    lr: float = 1e-3                            # 学习率
    weight_decay: float = 0.1                   # 权重衰减
    warmup_steps: int = 100                     # 线性预热步数

    steps: int = 20000                          # 最大训练步数
    batch_size: int = 0                         # 0 表示 min(512, 训练集一半)
    grad_clip: float = 1.0                      # 梯度范数裁剪（防止梯度爆炸）

    target_val_acc: float = 0.95               # 目标验证准确率，达到此值时停止训练

    seed: int = 42
    out_dir: str = "outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------- 词表与编码 ----------------------
class Vocab:
    # 词表：包含 数字 0..p-1、操作符 token、等号 token
    def __init__(self, p: int):
        self.p = p
        self.num_offset = 0
        self.num_size = p
        self.op_offset = self.num_offset + self.num_size
        self.op_size = 1  # 只有一个操作：模加法
        self.eq_id = self.op_offset + self.op_size
        self.vocab_size = self.eq_id + 1
        self.op_id = self.op_offset  # 模加法的token id

    def num_id(self, x: int) -> int:
        return self.num_offset + int(x)


# ----------------------- 数据生成 -----------------------
def build_dataset(cfg: Config):
    # 生成模 p 加法数据集，并进行训练/验证划分
    vocab = Vocab(cfg.p)
    p = cfg.p
    samples = []  # 每条样本为长度 5 的 token 序列: [x, op, y, =, (x+y) mod p]

    # 生成所有模加法的样本：x + y mod p，其中 x, y ∈ [0, p-1]
    for x in range(p):
        for y in range(p):
            c = (x + y) % p
            samples.append([
                vocab.num_id(x),
                vocab.op_id,
                vocab.num_id(y),
                vocab.eq_id,
                vocab.num_id(c)
            ])

    # 转为张量并划分
    data = torch.tensor(samples, dtype=torch.long)
    N = data.size(0)
    print(f"生成的数据样本数: {N:,} (模 p={p} 加法)")
    # 使用固定随机种子确保数据划分可复现
    generator = torch.Generator()
    generator.manual_seed(42)
    idx = torch.randperm(N, generator=generator)
    N_train = int(N * cfg.train_ratio)
    train_data = data[idx[:N_train]]
    val_data = data[idx[N_train:]]
    print(f"训练集: {train_data.size(0):,}, 验证集: {val_data.size(0):,}")

    return train_data, val_data, vocab


# ----------------------- 模型定义 -----------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        # 重塑为多头形式：[B, L, n_heads, head_dim] -> [B, n_heads, L, head_dim]
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 因果掩码：下三角矩阵，防止看到未来信息
        mask = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_drop(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, dropout: float, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.size()
        pos = torch.arange(0, L, device=idx.device)
        # 词嵌入 + 位置嵌入
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits  # [B, L, V]


# ----------------------- 训练与评估 -----------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_training_curves(history: dict, out_dir: str):
    # 绘制训练曲线（loss、train acc、val acc）
    if len(history['steps']) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制Loss曲线
    axes[0].plot(history['steps'], history['losses'], 'b-', label='Loss', linewidth=2)
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_yscale('log')  # 使用对数刻度，便于观察loss变化

    # 绘制准确率曲线
    axes[1].plot(history['steps'], history['train_accs'], 'g-', label='Train Acc', linewidth=2)
    axes[1].plot(history['steps'], history['val_accs'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存


def build_optimizer(model: nn.Module, cfg: Config):
    return optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=cfg.weight_decay)


def train_loop(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_data, val_data, vocab = build_dataset(cfg)

    if cfg.batch_size == 0:
        cfg.batch_size = min(512, max(1, train_data.size(0) // 2))

    model = DecoderOnlyTransformer(
        vocab_size=vocab.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        seq_len=5,
    ).to(cfg.device)

    opt = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数总数: {total_params:,}")
    print(f"词表大小: {vocab.vocab_size}")
    print(f"训练集大小: {train_data.size(0):,}, 验证集大小: {val_data.size(0):,}\n")

    def lr_for_step(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
        return cfg.lr

    def batch_from(data: torch.Tensor, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, data.size(0), (batch_size,))
        return data[idx]

    def evaluate(data: torch.Tensor, max_samples: int = None) -> float:
        # 评估模型在给定数据上的准确率（仅使用前4个token预测，不使用真实标签）
        model.eval()
        with torch.no_grad():
            # 如果指定了最大样本数且数据量超过，则随机采样
            if max_samples is not None and data.size(0) > max_samples:
                indices = torch.randperm(data.size(0))[:max_samples]
                eval_data = data[indices]
            else:
                eval_data = data
            eval_data = eval_data.to(cfg.device)
            x, y = eval_data[:, :4], eval_data[:, 4]
            # 评估时只输入前4个token，第5个位置用等号token填充（已在序列中出现，不影响预测）
            # 因果掩码确保位置4的预测只依赖位置0-3的输入
            dummy_token = torch.full((x.size(0), 1), vocab.eq_id, dtype=torch.long, device=cfg.device)
            logits = model(torch.cat([x, dummy_token], dim=1))
            pred = logits[:, 4, :].argmax(dim=-1)  # 取位置4的预测（对应第5个token）
            acc = (pred == y).float().mean().item()
        return acc

    best_val_acc = 0.0
    best_state = None
    eval_interval = 10  # 更频繁的评估，便于观察训练进度

    # 记录训练历史用于可视化
    history = {
        'steps': [],
        'losses': [],
        'train_accs': [],
        'val_accs': [],
    }

    for step in range(cfg.steps):
        model.train()
        # 动态调整学习率（线性预热）
        for g in opt.param_groups:
            g['lr'] = lr_for_step(step)

        batch = batch_from(train_data, cfg.batch_size).to(cfg.device)
        x = batch[:, :4]
        y = batch[:, 4]

        opt.zero_grad(set_to_none=True)
        # 训练时与评估保持一致：只输入前4个token，使用dummy token作为第5个位置
        # 因果掩码确保位置4的预测只依赖位置0-3的输入
        dummy_token = torch.full((x.size(0), 1), vocab.eq_id, dtype=torch.long, device=cfg.device)
        input_seq = torch.cat([x, dummy_token], dim=1)  # [B, 5]
        logits = model(input_seq)  # [B, 5, V]
        pred_last = logits[:, 4, :]  # 仅计算最后一位的损失
        loss = criterion(pred_last, y)
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        if cfg.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        opt.step()

        # 周期性评估与日志
        if (step + 1) % eval_interval == 0 or step == cfg.steps - 1:
            acc_tr = evaluate(train_data, max_samples=4096)
            acc_val = evaluate(val_data)

            # 记录历史（使用当前batch的loss，与训练一致）
            history['steps'].append(step + 1)
            history['losses'].append(loss.item())
            history['train_accs'].append(acc_tr)
            history['val_accs'].append(acc_val)

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_state = {
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "vocab": {
                        "p": vocab.p,
                        "vocab_size": vocab.vocab_size,
                        "num_size": vocab.num_size,
                    },
                }

            print(f"Step {step+1:06d} | Train Acc {acc_tr:.4f} | Val Acc {acc_val:.4f} | Loss {loss.item():.4f}")

            # 绘制并保存曲线图
            plot_training_curves(history, cfg.out_dir)

            # 如果验证准确率达到目标，则停止训练
            if acc_val >= cfg.target_val_acc:
                print(f"Validation accuracy {acc_val:.4f} reached {cfg.target_val_acc:.2f}%. Stopping training.")
                break

    # 保存最佳模型
    if best_state is not None:
        save_path = os.path.join(cfg.out_dir, "best_model.pt")
        torch.save(best_state, save_path)
        print(f"最佳模型已保存: {save_path}")
        print(f"最佳验证准确率: {best_val_acc:.4f}")

    # 最终保存一次训练曲线
    if len(history['steps']) > 0:
        plot_training_curves(history, cfg.out_dir)
        print(f"训练曲线已保存: {os.path.join(cfg.out_dir, 'training_curves.png')}")


# ----------------------- 命令行接口 -----------------------
def main():
    cfg = Config(
        p=97,
        train_ratio=0.2,
        d_model=128,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1,
        warmup_steps=10,
        steps=20000, # Max steps if target_val_acc is not reached
        batch_size=0,
        grad_clip=1.0,
        target_val_acc=0.9995, # Target validation accuracy
        seed=42,
        out_dir="outputs",
    )
    print("配置:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    train_loop(cfg)


if __name__ == "__main__":
    main()
