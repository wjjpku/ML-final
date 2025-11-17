import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse
from dataclasses import asdict
from mlfinal.config import Config
from mlfinal.trainer import train_loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int)
    parser.add_argument('--train-ratio', type=float)
    parser.add_argument('--d-model', type=int)
    parser.add_argument('--n-layers', type=int)
    parser.add_argument('--n-heads', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--warmup-steps', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--grad-clip', type=float)
    parser.add_argument('--target-val-acc', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--out-dir', type=str)
    args = parser.parse_args()
    cfg = Config(
        p=args.p if args.p is not None else 97,
        train_ratio=args.train_ratio if args.train_ratio is not None else 0.4,
        d_model=args.d_model if args.d_model is not None else 128,
        n_layers=args.n_layers if args.n_layers is not None else 2,
        n_heads=args.n_heads if args.n_heads is not None else 4,
        dropout=args.dropout if args.dropout is not None else 0.1,
        lr=args.lr if args.lr is not None else 1e-3,
        weight_decay=args.weight_decay if args.weight_decay is not None else 1,
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else 100,
        steps=args.steps if args.steps is not None else 20000,
        batch_size=args.batch_size if args.batch_size is not None else 0,
        grad_clip=args.grad_clip if args.grad_clip is not None else 1.0,
        target_val_acc=args.target_val_acc if args.target_val_acc is not None else 0.95,
        seed=args.seed if args.seed is not None else 42,
        out_dir=args.out_dir if args.out_dir is not None else "outputs",
    )
    print("配置:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    train_loop(cfg)

if __name__ == "__main__":
    main()
