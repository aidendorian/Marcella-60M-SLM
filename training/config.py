import torch

class Config:
    def __init__(self):
        self.vocab_size = 32000
        self.embed_dim = 576
        self.num_transformer_layers = 12
        self.num_heads = 12
        self.attn_dropout = 0.05
        self.ffn_dropout = 0.1
        self.dataset_name = "draco976/wikipedia-bookcorpus"
        self.dataset_split = 'train'
        self.tkn_model = 'models/Marcella_vocab_32K.model'
        self.block_size = 512
        self.batch_size = 6
        self.num_workers = 0
        self.pin_memory = False
        self.prefetch_factor = None
        self.persistent_workers = False
        self.max_samples = None
        self.validation_prompt = "The Moon has this magical creature on it's surface called"
        self.device = torch.device('cuda')
        self.max_seq_len = 1024
        self.accumulation_steps = 6