import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse
import torch
import torch.nn as nn
from mlfinal.config import Config
from mlfinal.trainer import train_loop


def get_supported_architectures():
    """获取支持的模型架构"""
    return ['transformer', 'mlp', 'lstm', 'gru']


def main():
    parser = argparse.ArgumentParser(description='训练神经网络模型')
    
    # 数据参数
    parser.add_argument('--p', type=int, default=None, help='模数，默认97')
    parser.add_argument('--op', type=str, default=None, help='运算类型，默认mod_add')
    parser.add_argument('--k', type=int, default=None, 
                        help='运算的元数（变量数量），默认为2（二元运算）')
    parser.add_argument('--train-ratio', type=float, default=None, help='训练集比例，默认0.4')
    
    # 模型参数
    parser.add_argument('--architecture', type=str, default=None,
                        choices=get_supported_architectures(),
                        help='模型架构: transformer, mlp, lstm, gru')
    parser.add_argument('--d-model', type=int, default=None, help='隐藏维度，默认128')
    parser.add_argument('--n-layers', type=int, default=None, help='层数，默认2')
    parser.add_argument('--n-heads', type=int, default=None, help='注意力头数，默认4')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout比例，默认0.1')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default=None, help='优化器类型，默认adamw')
    parser.add_argument('--lr', type=float, default=None, help='学习率，默认1e-3')
    parser.add_argument('--weight-decay', type=float, default=None, help='权重衰减，默认1')
    parser.add_argument('--warmup-steps', type=int, default=None, help='预热步数，默认100')
    
    # 训练参数
    parser.add_argument('--steps', type=int, default=None, help='总步数，默认20000')
    parser.add_argument('--batch-size', type=int, default=None, help='批处理大小，默认自动计算')
    parser.add_argument('--full-batch', action='store_true', help='全批次训练')
    parser.add_argument('--eval-interval', type=int, default=None, help='评估间隔，默认20')
    parser.add_argument('--target-val-acc', type=float, default=None, help='目标验证准确率，默认0.95')
    parser.add_argument('--grad-clip', type=float, default=None, help='梯度剪裁，默认1.0')
    
    # 正则化与噪声
    parser.add_argument('--grad-noise-std', type=float, default=None, help='梯度噪声标准差，默认0.0')
    parser.add_argument('--weight-noise-std', type=float, default=None, help='权重噪声标准差，默认0.0')
    parser.add_argument('--decay-to-init', action='store_true', help='衰减到初始值')
    parser.add_argument('--decay-to-init-lambda', type=float, default=None, help='衰减系数，默认0.0')
    
    # 可视化参数
    parser.add_argument('--plot-note', type=str, default=None, help='图表注释')
    parser.add_argument('--plot-interval', type=int, default=None, help='绘图间隔')
    parser.add_argument('--enable-visualization', action='store_true', default=True,
                        help='启用可视化（保存Loss曲线）')
    parser.add_argument('--no-visualization', dest='enable_visualization', action='store_false',
                        help='禁用可视化')
    
    # 数据存储参数
    parser.add_argument('--data-storage-dir', type=str, default=None,
                        help='数据存储目录，默认自动创建带时间戳的目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=None, help='随机种子，默认42')
    parser.add_argument('--out-dir', type=str, default=None, help='输出目录，默认outputs')
    
    args = parser.parse_args()
    
    # 构建配置对象，使用默认值填充 None 值
    cfg = Config(
        p=args.p if args.p is not None else 97,
        op=args.op if args.op is not None else "mod_add",
        k=args.k if args.k is not None else 2,
        train_ratio=args.train_ratio if args.train_ratio is not None else 0.4,
        architecture=args.architecture if args.architecture is not None else "transformer",
        d_model=args.d_model if args.d_model is not None else 128,
        n_layers=args.n_layers if args.n_layers is not None else 2,
        n_heads=args.n_heads if args.n_heads is not None else 4,
        dropout=args.dropout if args.dropout is not None else 0.1,
        optimizer=args.optimizer if args.optimizer is not None else "adamw",
        lr=args.lr if args.lr is not None else 1e-3,
        weight_decay=args.weight_decay if args.weight_decay is not None else 1.0,
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else 100,
        steps=args.steps if args.steps is not None else 20000,
        batch_size=args.batch_size if args.batch_size is not None else 0,
        full_batch=args.full_batch,
        eval_interval=args.eval_interval if args.eval_interval is not None else 20,
        target_val_acc=args.target_val_acc if args.target_val_acc is not None else 0.95,
        grad_clip=args.grad_clip if args.grad_clip is not None else 1.0,
        grad_noise_std=args.grad_noise_std if args.grad_noise_std is not None else 0.0,
        weight_noise_std=args.weight_noise_std if args.weight_noise_std is not None else 0.0,
        decay_to_init=args.decay_to_init,
        decay_to_init_lambda=args.decay_to_init_lambda if args.decay_to_init_lambda is not None else 0.0,
        enable_visualization=args.enable_visualization,
        plot_note=args.plot_note if args.plot_note is not None else "",
        plot_interval=args.plot_interval if args.plot_interval is not None else 0,
        data_storage_dir=args.data_storage_dir,
        seed=args.seed if args.seed is not None else 42,
        out_dir=args.out_dir if args.out_dir is not None else "outputs",
    )
    
    # 启动训练
    train_loop(cfg)


if __name__ == "__main__":
    main()

