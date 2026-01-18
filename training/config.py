class Config:
    def __init__(self):
        self.vocab_size = 32000
        self.embed_dim = 384
        self.num_transformer_layers = 32
        self.num_heads = 12
        self.attn_dropout = 0.0
        self.ffn_dropout = 0.1
