import os
import sys
import torch
import torch.nn as nn
from dataclasses import asdict
from .config import Config
from .data import build_dataset
from .architectures import get_model
from .utils import set_seed, plot_training_curves, build_optimizer
from .storage import DataStorageManager


def train_loop(cfg: Config):
    """
    主训练循环
    
    Args:
        cfg (Config): 训练配置对象
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # 初始化数据存储管理器
    storage_manager = DataStorageManager(data_dir=cfg.data_storage_dir)
    csv_path = storage_manager.get_csv_path()
    data_dir = storage_manager.get_data_dir()
    
    # 获取命令行参数字符串
    command_line = ' '.join(sys.argv)
    
    # 保存配置信息到 txt 文件
    storage_manager.save_config(cfg, command_line_args=command_line)
    
    # 打印初始配置
    print(f"\n{'='*80}")
    print(f"训练配置")
    print(f"{'='*80}")
    print(f"运算: {cfg.op}, 模数: {cfg.p}, 元数: {cfg.k}")
    print(f"模型: {cfg.architecture}, D_model: {cfg.d_model}, Layers: {cfg.n_layers}")
    print(f"优化器: {cfg.optimizer}, LR: {cfg.lr}, Weight_decay: {cfg.weight_decay}")
    print(f"训练集比例: {cfg.train_ratio}")
    print(f"数据存储目录: {data_dir}")
    print(f"CSV日志路径: {csv_path}")
    print(f"可视化: {'启用' if cfg.enable_visualization else '禁用'}")
    print(f"{'='*80}\n")
    
    # 构建数据集
    train_data, val_data, vocab = build_dataset(cfg)
    
    # 设置批处理大小
    if cfg.batch_size == 0:
        cfg.batch_size = min(512, max(1, train_data.size(0) // 2))
    if cfg.full_batch:
        cfg.batch_size = train_data.size(0)
    
    # 计算序列长度：对于K元运算，格式为 x1 op x2 op ... op xk = result
    # 长度为 2*k + 1 (k个数字，k-1个op，1个等号）
    k = getattr(cfg, 'k', 2)
    seq_len = 2 * k + 1
    
    # 创建模型
    model = get_model(
        architecture=cfg.architecture,
        vocab_size=vocab.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        seq_len=seq_len,
    ).to(cfg.device)
    
    # 构建优化器
    opt = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    
    # 计算总参数数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型架构: {cfg.architecture}")
    print(f"模型参数总数: {total_params:,}")
    print(f"词表大小: {vocab.vocab_size}")
    print(f"运算元数 (K): {k}")
    print(f"序列长度: {seq_len}")
    print(f"训练集大小: {train_data.size(0):,}, 验证集大小: {val_data.size(0):,}\n")
    
    # 学习率预热函数
    def lr_for_step(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
        return cfg.lr
    
    # 批处理采样函数
    def batch_from(data: torch.Tensor, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, data.size(0), (batch_size,))
        return data[idx]
    
    # 评估函数
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
    
    # 训练历史
    history = {
        'steps': [],
        'losses': [],
        'train_accs': [],
        'val_accs': [],
    }
    
    print(f"{'='*80}")
    print(f"开始训练")
    print(f"{'='*80}\n")
    
    # 主训练循环
    for step in range(cfg.steps):
        model.train()
        
        # 更新学习率
        for g in opt.param_groups:
            g['lr'] = lr_for_step(step)
        
        # 采样批处理
        batch = batch_from(train_data, cfg.batch_size).to(cfg.device)
        
        # 提取输入和目标
        x = batch[:, :-1]
        y = batch[:, -1]
        
        opt.zero_grad(set_to_none=True)
        
        # 创建虚拟token（等号位置）
        dummy_token = torch.full((x.size(0), 1), vocab.eq_id, dtype=torch.long, device=cfg.device)
        input_seq = torch.cat([x, dummy_token], dim=1)
        
        # 权重噪声
        if cfg.weight_noise_std > 0.0:
            for p in model.parameters():
                p.data.add_(torch.randn_like(p) * cfg.weight_noise_std)
        
        # 前向传播
        logits = model(input_seq)
        pred_last = logits[:, -1, :]
        loss = criterion(pred_last, y)
        
        # 反向传播
        loss.backward()
        
        # 梯度噪声
        if cfg.grad_noise_std > 0.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p) * cfg.grad_noise_std)
        
        # 权重衰减到初始值
        if cfg.decay_to_init and cfg.decay_to_init_lambda > 0.0:
            if 'init_params' not in locals():
                init_params = [q.detach().clone() for q in model.parameters()]
            for p, p0 in zip(model.parameters(), init_params):
                if p.grad is not None:
                    p.grad.add_((p.data - p0) * cfg.decay_to_init_lambda)
        
        # 梯度剪裁
        if cfg.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        # 优化器更新
        opt.step()
        
        # 定期评估
        if (step + 1) % eval_interval == 0 or step == cfg.steps - 1:
            acc_tr, loss_tr = evaluate(train_data, max_samples=4096)
            acc_val, loss_val = evaluate(val_data)
            
            history['steps'].append(step + 1)
            history['losses'].append(loss.item())
            history['train_accs'].append(acc_tr)
            history['val_accs'].append(acc_val)
            
            # 记录到CSV并打印
            storage_manager.log_evaluation(step + 1, acc_tr, acc_val, loss_tr, loss_val)
            
            # 保存最好的模型
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
            
            # 定期绘制曲线（仅当启用可视化时）
            if cfg.plot_interval and (step + 1) % cfg.plot_interval == 0:
                if cfg.enable_visualization:
                    plot_training_curves(history, storage_manager, cfg, step=step+1)
            
            # 检查是否达到目标准确率
            if acc_val >= cfg.target_val_acc:
                # 记录成功到CSV并打印
                storage_manager.log_success(step + 1, acc_tr, acc_val, loss_val)
                
                # 绘制最终曲线（仅当启用可视化时）
                if cfg.enable_visualization and len(history['steps']) > 0:
                    plot_training_curves(history, storage_manager, cfg, step='final')
                
                # 保存最好的模型
                if best_state is not None:
                    save_path = os.path.join(data_dir, "best_model.pt")
                    torch.save(best_state, save_path)
                    print(f"\n最佳模型已保存: {save_path}")
                    print(f"最佳验证准确率: {best_val_acc:.6f}")
                
                # 打印最终信息
                print(f"\n{'='*80}")
                print(f"训练完成")
                print(f"{'='*80}")
                print(f"数据已保存到: {data_dir}")
                print(f"CSV日志路径: {csv_path}")
                print(f"{'='*80}\n")
                
                return {
                    'status': 'success',
                    'step': step + 1,
                    'train_acc': acc_tr,
                    'val_acc': acc_val,
                    'loss': loss_val,
                    'data_dir': data_dir
                }
    
    # 训练结束（未达到目标）
    acc_tr, loss_tr = evaluate(train_data, max_samples=4096)
    acc_val, loss_val = evaluate(val_data)
    
    # 记录失败到CSV并打印
    storage_manager.log_failure(cfg.steps, acc_tr, acc_val, loss_val)
    
    # 绘制最终曲线（仅当启用可视化时）
    if cfg.enable_visualization and len(history['steps']) > 0:
        plot_training_curves(history, storage_manager, cfg, step='final')
    
    # 保存最好的模型
    if best_state is not None:
        save_path = os.path.join(data_dir, "best_model.pt")
        torch.save(best_state, save_path)
        print(f"\n最佳模型已保存: {save_path}")
        print(f"最佳验证准确率: {best_val_acc:.6f}")
    
    # 打印最终信息
    print(f"\n{'='*80}")
    print(f"训练完成")
    print(f"{'='*80}")
    print(f"数据已保存到: {data_dir}")
    print(f"CSV日志路径: {csv_path}")
    print(f"{'='*80}\n")

    return {
        'status': 'finished',
        'step': cfg.steps,
        'train_acc': acc_tr,
        'val_acc': acc_val,
        'loss': loss_val,
        'data_dir': data_dir
    }
