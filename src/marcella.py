import torch
from torch.nn import LayerNorm, GELU, Module, Linear, Dropout
from torch.optim import AdamW
from torch.nn.functional import scaled_dot_product_attention

N_HEADS = 12
D_MODEL = 384
VOCAB = 32000
ATTN_DROPOUT = 0.1

class Attention(Module):
    def __init__(self, dropout, attn_dropout=ATTN_DROPOUT):
        super().__init__()
        
        assert D_MODEL%N_HEADS == 0, f'D_MODEL: {D_MODEL} must be divisible by N_HEADS: {N_HEADS}'
        
        self.qkv = Linear(D_MODEL, 3*D_MODEL, bias=False)
        self.d_head = D_MODEL // N_HEADS
        self.attn_dropout = attn_dropout
        self.linear = Linear(D_MODEL, D_MODEL, bias=False)
        self.dropout = Dropout(dropout)
        
    def forward(self, x, kv_cache):
        B, S, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        k = self.k(x).view(B, N_HEADS, S, self.d_head).transpose(1, 2) 
        v = self.v(x).view(B, N_HEADS, S, self.d_head).transpose(1, 2)
        q = self.q(x).view(B, N_HEADS, S, self.d_head).transpose(1, 2)
        
        if self.cache is None:
            out = scaled_dot_product_attention(query=q,
                                            key=k,
                                            value=v,
                                            dropout_p=self.attn_dropout if self.training else 0.0,
                                            is_causal=True)
        else:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)

            kv_cache["k"] = k
            kv_cache["v"] = v

            attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
            attn = attn.softmax(dim=-1)
            out = attn @ v
            
        out = out.transpose(1, 2).contiguous().view(B, S, D_MODEL)
        out = self.linear(out)
        return self.dropout(out)