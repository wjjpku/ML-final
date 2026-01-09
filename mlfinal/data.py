import torch
import itertools
import random

class Vocab:
    def __init__(self, num_size: int, k: int = 2):
        """
        Args:
            num_size: 数字符号数量
            k: 运算的元数（变量数量）
        """
        self.num_offset = 0
        self.num_size = num_size
        self.op_offset = self.num_offset + self.num_size
        self.op_size = 1
        self.eq_id = self.op_offset + self.op_size
        self.vocab_size = self.eq_id + 1
        self.op_id = self.op_offset
        self.k = k

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

def _validate_operation_arity(op: str, k: int):
    """
    验证运算是否支持给定的元数
    
    Args:
        op: 运算名称
        k: 元数
    
    Raises:
        ValueError: 如果运算不支持给定的元数
    """
    # 只支持二元运算的操作
    binary_only_ops = {
        "mod_sub", "mod_div", "div_or_sub_by_y_parity",
        "x2_y2", "x2_xy_y2", "x2_xy_y2_plus_x",
        "x3_xy", "x3_xy2_plus_y",
        "s5_mul", "s5_conj", "s5_x_y_x"
    }
    
    # 支持K元运算的操作
    kary_ops = {"mod_add", "sum_mod"}
    
    if op in binary_only_ops and k != 2:
        raise ValueError(f"运算 '{op}' 仅支持二元运算（k=2），但指定 k={k}")
    
    if op not in binary_only_ops and op not in kary_ops:
        raise ValueError(f"未知的运算: {op}")

def _build_mod_dataset(cfg, op: str):
    """
    生成模运算数据集
    
    支持的二元运算: mod_add, mod_sub, mod_div, div_or_sub_by_y_parity, 
                   x2_y2, x2_xy_y2, x2_xy_y2_plus_x, x3_xy, x3_xy2_plus_y
    支持的K元运算: sum_mod
    """
    p = cfg.p
    k = getattr(cfg, 'k', 2)
    
    # 验证运算是否支持给定的元数
    _validate_operation_arity(op, k)
    
    vocab = Vocab(p, k=k)
    samples = []
    
    # 生成所有k元组合
    if k == 2:
        # 二元运算
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
                    raise ValueError(f"未知的二元运算: {op}")
                
                sample = [vocab.num_id(x), vocab.op_id, vocab.num_id(y), vocab.eq_id, vocab.num_id(c)]
                samples.append(sample)
    else:
        # K元运算 (k > 2)
        if op == "mod_add" or op == "sum_mod":
            # 生成k元组合的所有可能性
            # 对于大的k，使用采样以避免组合爆炸
            num_samples = min(p ** k, 100000)  # 限制样本数
            random.seed(42)
            
            generated = set()
            while len(generated) < num_samples:
                variables = tuple(random.randint(0, p-1) for _ in range(k))
                if variables not in generated:
                    generated.add(variables)
                    c = sum(variables) % p
                    
                    sample = []
                    for i, var in enumerate(variables):
                        if i > 0:
                            sample.append(vocab.op_id)
                        sample.append(vocab.num_id(var))
                    sample.append(vocab.eq_id)
                    sample.append(vocab.num_id(c))
                    samples.append(sample)
        else:
            raise ValueError(f"运算 '{op}' 不支持K元运算（k={k}）")
    
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
    """
    生成S5群数据集（仅支持二元运算）
    """
    k = getattr(cfg, 'k', 2)
    if k != 2:
        raise ValueError(f"S5群运算仅支持二元运算（k=2），但指定 k={k}")
    
    elems = list(itertools.permutations(range(5)))
    id_of = {tuple(e): i for i, e in enumerate(elems)}
    vocab = Vocab(len(elems), k=k)
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
    """
    构建数据集
    
    Args:
        cfg: 配置对象，包含op、k等参数
    
    Returns:
        train_data, val_data, vocab
    """
    k = getattr(cfg, 'k', 2)
    
    if cfg.op.startswith("s5_"):
        if k != 2:
            raise ValueError(f"S5群运算不支持K元运算（k={k}）")
        data, vocab = _build_s5_dataset(cfg, cfg.op)
        domain_desc = "S5"
    else:
        data, vocab = _build_mod_dataset(cfg, cfg.op)
        domain_desc = f"模 p={cfg.p}"
    
    # 如果k > 2，添加k的信息
    if k > 2:
        domain_desc += f"（{k}元运算）"
    
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