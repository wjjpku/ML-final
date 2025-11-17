import torch

class Vocab:
    def __init__(self, p: int):
        self.p = p
        self.num_offset = 0
        self.num_size = p
        self.op_offset = self.num_offset + self.num_size
        self.op_size = 1
        self.eq_id = self.op_offset + self.op_size
        self.vocab_size = self.eq_id + 1
        self.op_id = self.op_offset

    def num_id(self, x: int) -> int:
        return self.num_offset + int(x)

def build_dataset(cfg):
    vocab = Vocab(cfg.p)
    p = cfg.p
    samples = []
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
    data = torch.tensor(samples, dtype=torch.long)
    N = data.size(0)
    print(f"生成的数据样本数: {N:,} (模 p={p} 加法)")
    generator = torch.Generator()
    generator.manual_seed(42)
    idx = torch.randperm(N, generator=generator)
    N_train = int(N * cfg.train_ratio)
    train_data = data[idx[:N_train]]
    val_data = data[idx[N_train:]]
    print(f"训练集: {train_data.size(0):,}, 验证集: {val_data.size(0):,}")
    return train_data, val_data, vocab