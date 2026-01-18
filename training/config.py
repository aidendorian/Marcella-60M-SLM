class Config:
    def __init__(self):
        self.vocab_size = 32000
        self.embed_dim = 384
        self.num_transformer_layers = 32
        self.num_heads = 12
        self.attn_dropout = 0.0
        self.ffn_dropout = 0.1
        self.dataset_name = "draco976/wikipedia-bookcorpus"
        self.dataset_split = 'train'
        self.tkn_model = 'models/Marcella_vocab_32K.model'
        self.block_size = 512
        self.batch_size = 2
        self.num_workers = 4
        self.pin_memory = True
        self.prefetch_factor = 2
        self.persistent_workers = True
        self.max_samples = None