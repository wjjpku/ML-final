import torch
import itertools

class Vocab:
    def __init__(self, num_size: int):
        self.num_offset = 0
        self.num_size = num_size
        self.op_offset = self.num_offset + self.num_size
        self.op_size = 1
        self.eq_id = self.op_offset + self.op_size
        self.vocab_size = self.eq_id + 1
        self.op_id = self.op_offset

    def num_id(self, x: int) -> int:
        return self.num_offset + int(x)

def _mod_inv(y: int, p: int) -> int:
    if y % p == 0:
        return 0
    try:
        return pow(y, -1, p)
    except Exception:
        a, b = y % p, p
        u0, u1 = 1, 0
        while b:
            q = a // b
            a, b = b, a - q * b
            u0, u1 = u1, u0 - q * u1
        return u0 % p

def _build_mod_dataset(cfg, op: str):
    p = cfg.p
    vocab = Vocab(p)
    samples = []
    for x in range(p):
        for y in range(p):
            if op == "mod_add":
                c = (x + y) % p
            elif op == "mod_sub":
                c = (x - y) % p
            elif op == "mod_div":
                if y == 0:
                    continue
                c = (x * _mod_inv(y, p)) % p
            elif op == "div_or_sub_by_y_parity":
                if y % 2 == 1 and y != 0:
                    c = (x * _mod_inv(y, p)) % p
                else:
                    c = (x - y) % p
            elif op == "x2_y2":
                c = (x * x + y * y) % p
            elif op == "x2_xy_y2":
                c = (x * x + x * y + y * y) % p
            elif op == "x2_xy_y2_plus_x":
                c = (x * x + x * y + y * y + x) % p
            elif op == "x3_xy":
                c = (x * x * x + x * y) % p
            elif op == "x3_xy2_plus_y":
                c = (x * x * x + x * y * y + y) % p
            else:
                raise ValueError(op)
            samples.append([vocab.num_id(x), vocab.op_id, vocab.num_id(y), vocab.eq_id, vocab.num_id(c)])
    data = torch.tensor(samples, dtype=torch.long)
    return data, vocab

def _compose_perm(a, b):
    return [a[b[i]] for i in range(len(b))]

def _inv_perm(a):
    inv = [0] * len(a)
    for i, v in enumerate(a):
        inv[v] = i
    return inv

def _build_s5_dataset(cfg, op: str):
    elems = list(itertools.permutations(range(5)))
    id_of = {tuple(e): i for i, e in enumerate(elems)}
    vocab = Vocab(len(elems))
    samples = []
    for e1 in elems:
        for e2 in elems:
            x_id = id_of[tuple(e1)]
            y_id = id_of[tuple(e2)]
            if op == "s5_mul":
                c = _compose_perm(list(e1), list(e2))
            elif op == "s5_conj":
                inv = _inv_perm(list(e1))
                c = _compose_perm(list(e1), _compose_perm(list(e2), inv))
            elif op == "s5_x_y_x":
                c = _compose_perm(list(e1), _compose_perm(list(e2), list(e1)))
            else:
                raise ValueError(op)
            c_id = id_of[tuple(c)]
            samples.append([vocab.num_id(x_id), vocab.op_id, vocab.num_id(y_id), vocab.eq_id, vocab.num_id(c_id)])
    data = torch.tensor(samples, dtype=torch.long)
    return data, vocab

def build_dataset(cfg):
    if cfg.op.startswith("s5_"):
        data, vocab = _build_s5_dataset(cfg, cfg.op)
        domain_desc = "S5"
    else:
        data, vocab = _build_mod_dataset(cfg, cfg.op)
        domain_desc = f"模 p={cfg.p}"
    N = data.size(0)
    print(f"生成的数据样本数: {N:,} ({domain_desc}，操作 {cfg.op})")
    generator = torch.Generator()
    generator.manual_seed(42)
    idx = torch.randperm(N, generator=generator)
    N_train = int(N * cfg.train_ratio)
    train_data = data[idx[:N_train]]
    val_data = data[idx[N_train:]]
    print(f"训练集: {train_data.size(0):,}, 验证集: {val_data.size(0):,}")
    return train_data, val_data, vocab