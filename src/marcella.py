from torch.nn import RMSNorm, Module, Linear, Dropout, ModuleList
from src.attention import Attention, KV_Cache
from bitsandbytes.nn.modules import StableEmbedding
import torch
import torch.nn.functional as F

class TransformerBlock(Module):
    def __init__(self,
                 embed_dim:int,
                 num_heads:int,
                 ffn_dropout:float,
                 attn_dropout:float):
        
        super().__init__()
        
        self.norm1 = RMSNorm(embed_dim)
        self.attn = Attention(embed_dim=embed_dim,
                              num_heads=num_heads,
                              flash_attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim=embed_dim,
                                      dropout=ffn_dropout)
        
    def forward(self, x, kv_cache=None):
        attn_out = self.attn(self.norm1(x), kv_cache)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x  

class FeedForwardNetwork(Module):
    def __init__(self,
                 embed_dim:int,
                 dropout:float):
        super().__init__()

        hidden_dim = int(embed_dim * 4 * 2 / 3)

        self.gate_proj = Linear(embed_dim, hidden_dim,bias=False)
        self.up_proj = Linear(embed_dim, hidden_dim,bias=False)
        self.down_proj = Linear(hidden_dim, embed_dim,bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        swiglu = self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))
        return self.dropout(swiglu)

class Marcella(Module):
    def __init__(self,
                 vocab_size:int=32000,
                 embed_dim:int=384,
                 num_transformer_layers:int=32,
                 num_heads:int=12,
                 attn_dropout:float=0.0,
                 ffn_dropout:float=0.1):
        
        super().__init__()
        
        self.token_embed = StableEmbedding(vocab_size, embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.transformer_blocks = ModuleList([
            TransformerBlock(embed_dim=embed_dim,
                             num_heads=num_heads,
                             ffn_dropout=ffn_dropout,
                             attn_dropout=attn_dropout)
            for _ in range(num_transformer_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        
    def init_kv_cache(self,
                      batch_size: int,
                      max_seq_len: int,
                      device=torch.device('cuda'),
                      dtype=torch.bfloat16):

        caches = []

        for _ in self.transformer_blocks:

            cache = KV_Cache(batch_size=batch_size,
                             num_heads=self.num_heads,
                             max_seq_len=max_seq_len,
                             head_dim=self.head_dim)
            
            if device is not None:
                cache.k = cache.k.to(device)
                cache.v = cache.v.to(device)
            if dtype is not None:
                cache.k = cache.k.to(dtype)
                cache.v = cache.v.to(dtype)
            caches.append(cache)
        return caches
        
    def forward(self, input_ids, kv_cache=None):
        x = self.token_embed(input_ids)

        if kv_cache is None:
            kv_cache = [None] * len(self.transformer_blocks)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, kv_cache[i])

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits