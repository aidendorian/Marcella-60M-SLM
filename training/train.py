import torch
from bitsandbytes.optim.adamw import AdamW8bit
from tqdm import tqdm
from config import Config
from torch.nn import CrossEntropyLoss
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import os
from dataloader import get_data
from src.marcella import Marcella

config = Config()

model = Marcella(vocab_size=config.vocab_size, 
                 embed_dim=config.embed_dim,
                 num_transformer_layers=config.num_transformer_layers, 
                 num_heads=config.num_heads,
                 attn_dropout=config.attn_dropout,
                 ffn_dropout=config.ffn_dropout
                 )

optimizer = AdamW8bit(model.parameters(), lr=1e-4)

data = get_data(dataset_name=config.dataset_name,
                dataset_split=config.dataset_split,
                tkn_model=config.tkn_model,
                block_size=config.block_size,
                batch_size=config.batch_size,
                num_workers=config.batch_size,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                persistent_workers=config.persistent_workers,
                max_samples=config.max_samples)

loss = CrossEntropyLoss()

scaler = GradScaler("cuda")