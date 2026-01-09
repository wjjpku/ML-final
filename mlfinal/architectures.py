import torch
import torch.nn as nn


# ============ Transformer 相关类 ============

class CausalSelfAttention(nn.Module):
    """因果自注意力层"""
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
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_drop(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    """Transformer 块：自注意力 + 前馈网络"""
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
    """仅解码器 Transformer 模型"""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, 
                 dropout: float, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) 
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.size()
        pos = torch.arange(0, L, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ============ MLP 模型 ============

class MLPModel(nn.Module):
    """多层感知机模型"""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, dropout: float, 
                 seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        # 将序列展平为一个向量
        input_dim = seq_len * d_model
        
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(prev_dim, d_model * 4))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = d_model * 4
        
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, vocab_size * seq_len)
        self.vocab_size = vocab_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.size()
        pos = torch.arange(0, L, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        
        # 展平为 (B, L * d_model)
        x = x.view(B, -1)
        
        # 通过 MLP
        x = self.mlp(x)
        
        # 输出层
        logits = self.head(x)
        
        # 重新 reshape 为 (B, L, vocab_size)
        logits = logits.view(B, L, self.vocab_size)
        return logits


# ============ LSTM 模型 ============

class LSTMModel(nn.Module):
    """LSTM 模型"""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, dropout: float, 
                 seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.size()
        pos = torch.arange(0, L, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        
        # LSTM
        x, (h, c) = self.lstm(x)
        
        # 输出层
        logits = self.head(x)
        return logits


# ============ GRU 模型 ============

class GRUModel(nn.Module):
    """GRU 模型"""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, dropout: float, 
                 seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.size()
        pos = torch.arange(0, L, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        
        # GRU
        x, h = self.gru(x)
        
        # 输出层
        logits = self.head(x)
        return logits


# ============ 工厂函数 ============

def get_model(architecture: str, vocab_size: int, d_model: int, n_layers: int, 
              n_heads: int, dropout: float, seq_len: int) -> nn.Module:
    """
    工厂函数：根据架构名称返回对应的模型实例
    
    Args:
        architecture: 模型架构名称 ('transformer', 'mlp', 'lstm', 'gru')
        vocab_size: 词汇表大小
        d_model: 模型维度
        n_layers: 层数
        n_heads: 注意力头数（仅用于 Transformer）
        dropout: Dropout 比例
        seq_len: 序列长度
    
    Returns:
        初始化的模型实例
    
    Raises:
        ValueError: 如果架构名称不被支持
    """
    arch_lower = architecture.lower()
    
    if arch_lower == "transformer":
        return DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            seq_len=seq_len
        )
    elif arch_lower == "mlp":
        return MLPModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            seq_len=seq_len
        )
    elif arch_lower == "lstm":
        return LSTMModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            seq_len=seq_len
        )
    elif arch_lower == "gru":
        return GRUModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            seq_len=seq_len
        )
    else:
        raise ValueError(
            f"未知的架构: {architecture}。支持的架构: {', '.join(get_supported_architectures())}"
        )


def get_supported_architectures() -> list:
    """返回所有支持的架构列表"""
    return ["transformer", "mlp", "lstm", "gru"]