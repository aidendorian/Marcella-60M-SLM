from torch import cat
from torch.nn import Module, Linear, Dropout
from torch.nn.functional import scaled_dot_product_attention

class Attention(Module):
    def __init__(self,
                 embed_dim:int,
                 num_heads:int,
                 flash_attn_dropout:float,
                 ln_dropout:float=0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0, f'D_MODEL: {embed_dim} must be divisible by N_HEADS: {num_heads}'
        
        self.qkv = Linear(embed_dim, 3 * embed_dim, bias=False)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.flash_attn_dropout = flash_attn_dropout
        self.linear = Linear(embed_dim, embed_dim, bias=False)
        self.ln_dropout = Dropout(ln_dropout)
        
    def forward(self, x, kv_cache):
        B, S, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        k = k.view(B, S, self.num_heads, self.d_head).transpose(1, 2) 
        v = v.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        q = q.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
            
        if kv_cache is None:
            out = scaled_dot_product_attention(query=q,
                                               key=k,
                                               value=v,
                                               dropout_p=self.flash_attn_dropout if self.training else 0.0,
                                               is_causal=True)
        else:
            if kv_cache.get("k") is not None:
                k = cat([kv_cache["k"], k], dim=2)
                v = cat([kv_cache["v"], v], dim=2)

            kv_cache["k"] = k
            kv_cache["v"] = v

            attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
            attn = attn.softmax(dim=-1)
            out = attn @ v
            
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        out = self.linear(out)
        return self.ln_dropout(out)