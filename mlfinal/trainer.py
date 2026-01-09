import os
import torch
import torch.nn as nn
from dataclasses import asdict
from .config import Config
from .data import build_dataset
from .architectures import get_model
from .utils import set_seed, plot_training_curves, build_optimizer, save_tsne_head

def train_loop(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_data, val_data, vocab = build_dataset(cfg)
    if cfg.batch_size == 0:
        cfg.batch_size = min(512, max(1, train_data.size(0) // 2))
    if cfg.full_batch:
        cfg.batch_size = train_data.size(0)
    
    # 计算序列长度：对于K元运算，格式为 x1 op x2 op ... op xk = result
    # 长度为 2*k + 1 (k个数字，k-1个op，1个eq)
    k = getattr(cfg, 'k', 2)
    seq_len = 2 * k + 1
    
    # 使用工厂函数创建模型，支持多种架构
    model = get_model(
        architecture=cfg.architecture,
        vocab_size=vocab.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        seq_len=seq_len,
    ).to(cfg.device)
    
    opt = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型架构: {cfg.architecture}")
    print(f"模型参数总数: {total_params:,}")
    print(f"词表大小: {vocab.vocab_size}")
    print(f"运算元数 (K): {k}")
    print(f"序列长度: {seq_len}")
    print(f"训练集大小: {train_data.size(0):,}, 验证集大小: {val_data.size(0):,}\n")
    
    def lr_for_step(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
        return cfg.lr
    
    def batch_from(data: torch.Tensor, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, data.size(0), (batch_size,))
        return data[idx]
    
    def evaluate(data: torch.Tensor, max_samples: int = None):
        model.eval()
        with torch.no_grad():
            if max_samples is not None and data.size(0) > max_samples:
                indices = torch.randperm(data.size(0))[:max_samples]
                eval_data = data[indices]
            else:
                eval_data = data
            eval_data = eval_data.to(cfg.device)
            
            # 提取输入和目标
            # 对于K元运算，最后一个元素是结果
            x = eval_data[:, :-1]
            y = eval_data[:, -1]
            
            # 创建虚拟token（等号位置）
            dummy_token = torch.full((x.size(0), 1), vocab.eq_id, dtype=torch.long, device=cfg.device)
            logits = model(torch.cat([x, dummy_token], dim=1))
            
            # 预测最后一个位置
            pred = logits[:, -1, :].argmax(dim=-1)
            acc = (pred == y).float().mean().item()
            loss = criterion(logits[:, -1, :], y).item()
        return acc, loss
    
    best_val_acc = 0.0
    best_state = None
    eval_interval = cfg.eval_interval if cfg.eval_interval and cfg.eval_interval > 0 else max(100, cfg.steps // 100)
    history = {
        'steps': [],
        'losses': [],
        'train_accs': [],
        'val_accs': [],
    }
    
    for step in range(cfg.steps):
        model.train()
        for g in opt.param_groups:
            g['lr'] = lr_for_step(step)
        
        batch = batch_from(train_data, cfg.batch_size).to(cfg.device)
        
        # 提取输入和目标
        x = batch[:, :-1]
        y = batch[:, -1]
        
        opt.zero_grad(set_to_none=True)
        
        # 创建虚拟token（等号位置）
        dummy_token = torch.full((x.size(0), 1), vocab.eq_id, dtype=torch.long, device=cfg.device)
        input_seq = torch.cat([x, dummy_token], dim=1)
        
        if cfg.weight_noise_std > 0.0:
            for p in model.parameters():
                p.data.add_(torch.randn_like(p) * cfg.weight_noise_std)
        
        logits = model(input_seq)
        pred_last = logits[:, -1, :]
        loss = criterion(pred_last, y)
        loss.backward()
        
        if cfg.grad_noise_std > 0.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p) * cfg.grad_noise_std)
        
        if cfg.decay_to_init and cfg.decay_to_init_lambda > 0.0:
            if 'init_params' not in locals():
                init_params = [q.detach().clone() for q in model.parameters()]
            for p, p0 in zip(model.parameters(), init_params):
                if p.grad is not None:
                    p.grad.add_((p.data - p0) * cfg.decay_to_init_lambda)
        
        if cfg.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        opt.step()
        
        if (step + 1) % eval_interval == 0 or step == cfg.steps - 1:
            acc_tr, loss_tr = evaluate(train_data, max_samples=4096)
            acc_val, loss_val = evaluate(val_data)
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
                        "vocab_size": vocab.vocab_size,
                        "num_size": vocab.num_size,
                        "k": vocab.k,
                    },
                }
            
            print(f"Step {step+1:06d} | Train Acc {acc_tr:.4f} | Val Acc {acc_val:.4f} | Loss {loss.item():.4f}")
            
            if cfg.plot_interval and (step + 1) % cfg.plot_interval == 0:
                plot_path = plot_training_curves(history, cfg.out_dir, cfg, step=step+1)
            
            if acc_val >= cfg.target_val_acc:
                print(f"Validation accuracy {acc_val:.4f} reached {cfg.target_val_acc:.2f}%. Stopping training.")
                break
    
    if best_state is not None:
        save_path = os.path.join(cfg.out_dir, "best_model.pt")
        torch.save(best_state, save_path)
        print(f"最佳模型已保存: {save_path}")
        print(f"最佳验证准确率: {best_val_acc:.4f}")
    
    if len(history['steps']) > 0:
        plot_path = plot_training_curves(history, cfg.out_dir, cfg, step='final')
    
    if cfg.tsne:
        path = save_tsne_head(model, cfg.out_dir, cfg, cfg.tsne_perplexity, cfg.tsne_n_iter)
