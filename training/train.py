import torch
from bitsandbytes.optim.adamw import AdamW8bit
from tqdm import tqdm
from config import Config
import os
import random
from datasets import load_dataset
from src.marcella import Marcella

config = Config()

model = Marcella(vocab_size=config.vocab_size, 
                 embed_dim=config.embed_dim,
                 num_transformer_layers=config.num_transformer_layers, 
                 num_heads=config.num_heads,
                 attn_dropout=config.attn_dropout,
                 ffn_dropout=config.ffn_dropout
                 )

optim = AdamW8bit(model.parameters(), lr=1e-4)

