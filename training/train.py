import torch
from bitsandbytes.optim.adamw import AdamW8bit
from tqdm import tqdm
from training.config import Config
from torch.nn import CrossEntropyLoss
from torch.amp.autocast_mode import autocast
import os
from training.dataloader import get_data
from src.marcella import Marcella
from src.tokenizer import Tokenizer
from src.attention import KV_Cache
from training.checkpoint import load_checkpoint, save_checkpoint

os.makedirs('training/checkpoints', exist_ok=True)

config = Config()
device = config.device
print(f'Device: {device}')

model = Marcella(vocab_size=config.vocab_size,
                 embed_dim=config.embed_dim,
                 num_transformer_layers=config.num_transformer_layers,
                 num_heads=config.num_heads,
                 attn_dropout=config.attn_dropout,
                 ffn_dropout=config.ffn_dropout).to(device)

tkn = Tokenizer()

optimizer = AdamW8bit(model.parameters(), lr=1e-4)

data = get_data(dataset_name=config.dataset_name,
                dataset_split=config.dataset_split,
                tkn_model=config.tkn_model,
                block_size=config.block_size,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                persistent_workers=config.persistent_workers,
                max_samples=config.max_samples)

loss_fn = CrossEntropyLoss()

@torch.no_grad()
def validate(model, val_prompt, max_tokens, top_k):
    model.eval()

    input_ids = torch.tensor(tkn.encode(val_prompt), device=config.device).unsqueeze(0)

    B, S = input_ids.shape
    total_seq_len = S + max_tokens
    
    kv_cache = [
        KV_Cache(
            batch_size=B,
            max_seq_len=total_seq_len
        )
        for _ in range(config.num_transformer_layers)
    ]

    generated = input_ids

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids, kv_cache)
        
        for _ in range(max_tokens):
            next_token_logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)
            logits = model(next_token, kv_cache)
            
    return tkn.decode(generated[0].tolist())

RESUME_FROM_CHECKPOINT = False
checkpoint_path = ''

start_iter = 0

if RESUME_FROM_CHECKPOINT:
    _, model, optimizer, start_iter = load_checkpoint(model, optimizer, checkpoint_path)
    print(f'Resuming Training from Iter: {start_iter} from {checkpoint_path}')

MAX_ITERS = 300 + start_iter
print(f'Starting Training from iter {start_iter} to {MAX_ITERS}')

for i, (x, y) in enumerate(tqdm(data, desc='Training', total=MAX_ITERS)):
    if start_iter > i:
        continue
    
    model.train()
    
    x, y = x.to(device), y.to(device)
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(x)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        loss = loss_fn(logits, y.view(B*T))
    
    loss = loss / config.accumulation_steps
    loss.backward()

    if (i + 1) % config.accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    if i == MAX_ITERS - 1:
        
        validation_output = validate(model, config.validation_prompt, max_tokens=256, top_k=10)

        with open('training/validation.txt', 'a') as file:
            file.write(f"Iteration {i+1}:\n")
            file.write(validation_output + '\n')
            file.write('-' * 50 + '\n\n')
        
        save_checkpoint(model, optimizer, 'training/checkpoints', f'{i+1}_chkpnt.pth', loss, i+1)
        
        print(f"\nValidation output:\n{validation_output}\n")
        print(f"Checkpoint saved: training/checkpoints/{i+1}_chkpnt.pth")
        break