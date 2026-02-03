from torch.nn import LayerNorm, GELU, Module, Linear, Dropout, ModuleList
from src.attention import Attention
from bitsandbytes.nn.modules import StableEmbedding

class TransformerBlock(Module):
    def __init__(self,
                 embed_dim:int,
                 num_heads:int,
                 ffn_dropout:float,
                 attn_dropout:float):
        
        super().__init__()
        
        self.norm1 = LayerNorm(embed_dim)
        self.attn = Attention(embed_dim=embed_dim,
                              num_heads=num_heads,
                              flash_attn_dropout=attn_dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim=embed_dim,
                                      dropout=ffn_dropout)
        
    def forward(self, x, kv_cache=None, freqs_cis=None):
        attn_out = self.attn(self.norm1(x), kv_cache, freqs_cis)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x  

class FeedForwardNetwork(Module):
    def __init__(self,
                 embed_dim:int,
                 dropout:float):
        super().__init__()

        self.layer1 = Linear(embed_dim, 4 * embed_dim, bias=False)
        self.act = GELU()
        self.layer2 = Linear(4 * embed_dim, embed_dim, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return self.dropout(x)

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
        self.norm = LayerNorm(embed_dim)
        
    def forward(self, input_ids, kv_cache=None, freqs_cis=None):
        x = self.token_embed(input_ids)

        if kv_cache is None:
            kv_cache = [None] * len(self.transformer_blocks)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, kv_cache[i], freqs_cis)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits