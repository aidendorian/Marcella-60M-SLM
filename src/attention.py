from torch import cat
from torch.nn import Module, Linear, Dropout
from torch.nn.functional import scaled_dot_product_attention
import torch
from training.config import Config

config = Config()

def precompute_freqs_cis(dim: int,
                         max_seq_len: int,
                         theta: float = 10000.0,
                         device=config.device,
                         dtype=torch.bfloat16):
    
    freq_seq = torch.arange(0, dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (freq_seq / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    sin = freqs.sin()
    cos = freqs.cos()
    return sin.to(dtype), cos.to(dtype)    

def apply_rope(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor):

    _, _, S, _ = x.shape

    sin = sin[:S]
    cos = cos[:S]

    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]

    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    rot_even = x_even*cos - x_odd*sin
    rot_odd  = x_even*sin + x_odd*cos

    x_out = torch.stack((rot_even, rot_odd),dim=-1).flatten(-2)
    return x_out

def apply_rope_offset(q, k, sin, cos, cache_len):
    S = q.size(2)
    sin_slice = sin[cache_len: cache_len + S]
    cos_slice = cos[cache_len: cache_len + S]

    q = apply_rope(q, sin_slice, cos_slice)
    k = apply_rope(k, sin_slice, cos_slice)
    return q, k

class KV_Cache:
    def __init__(self,
                 batch_size=config.batch_size,
                 num_heads=config.num_heads,
                 max_seq_len=config.max_seq_len,
                 head_dim=config.embed_dim//config.num_heads):
        
        self.k = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=config.device, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.cache_len = 0
        self.max_seq_len = max_seq_len

    def update(self, k, v):
        """
        k, v : (B, H, S, D)
        """

        S = k.size(2)
        start = self.cache_len
        end   = start + S

        if end > self.max_seq_len:
            raise RuntimeError("KV cache overflow")

        self.k[:, :, start:end, :] = k
        self.v[:, :, start:end, :] = v

        self.cache_len = end

    def get_kv(self):
        return (
            self.k[:, :, :self.cache_len, :],
            self.v[:, :, :self.cache_len, :]
        )

    def get_len(self):
        return self.cache_len

    def reset(self):
        self.cache_len = 0

class Attention(Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 flash_attn_dropout: float,
                 ln_dropout: float = 0.1):
        
        super().__init__()

        assert embed_dim % num_heads == 0

        self.qkv = Linear(embed_dim, 3 * embed_dim, bias=False)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.flash_attn_dropout = flash_attn_dropout
        self.linear = Linear(embed_dim, embed_dim, bias=False)
        self.ln_dropout = Dropout(ln_dropout)
        
        sin, cos = precompute_freqs_cis(embed_dim//num_heads,
                                        max_seq_len=config.max_seq_len,
                                        device=config.device)
        
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
                
        if kv_cache is None:
            q = apply_rope(q, self.rope_sin, self.rope_cos) # type: ignore
            k = apply_rope(k, self.rope_sin, self.rope_cos) # type: ignore
            
            out = scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.flash_attn_dropout if self.training else 0.0, 
                is_causal=True
            )
        else:
            cache_len = kv_cache.get_len()
            
            if cache_len == 0:
                q = apply_rope(q, self.rope_sin[:S], self.rope_cos[:S]) # type: ignore
                k = apply_rope(k, self.rope_sin[:S], self.rope_cos[:S]) # type: ignore

                kv_cache.update(k, v)

                out = scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            else:
                offset = cache_len
                q, k = apply_rope_offset(q, k, self.rope_sin, self.rope_cos, offset)
                
                kv_cache.update(k, v)
                k_all, v_all = kv_cache.get_kv()
                
                attn = (q @ k_all.transpose(-2, -1)) * (self.d_head ** -0.5)
                attn = attn.softmax(dim=-1)
                out = attn @ v_all
                
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        out = self.linear(out)
        return self.ln_dropout(out)