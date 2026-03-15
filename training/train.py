import torch
import math
from torch.optim import lr_scheduler
from bitsandbytes.optim.adamw import AdamW8bit
from tqdm import tqdm
from training.config import Config
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from torch.amp.autocast_mode import autocast
import os
from training.dataloader import get_data
from src.marcella import Marcella
from src.tokenizer import Tokenizer
from src.attention import KV_Cache
from training.checkpoint import load_checkpoint, save_checkpoint

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

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

no_decay = {"bias", "rmsnorm.weight"}

param_groups = [
    {"params": [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)]},
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)]},
]

optimizer = AdamW8bit(param_groups, lr=config.LR_MAX, betas=(0.9, 0.95), eps=1e-8)

def lr_lambda(step:int) -> float:
    if step < config.WARMUP_STEPS:
        return step/config.WARMUP_STEPS
    t= step - config.WARMUP_STEPS
    cosine = 0.5 * (1.0+math.cos(math.pi*t / config.T_MAX))
    lr = config.LR_MIN + (config.LR_MAX - config.LR_MIN) * cosine
    return lr / config.LR_MAX

start_iter = 0

RESUME_FROM_CHECKPOINT = True
checkpoint_path = 'training/checkpoints/15000_chkpnt.pth'

resume_chunks = 0

if RESUME_FROM_CHECKPOINT:
    _, model, optimizer, start_iter, resume_chunks = load_checkpoint(model, optimizer, checkpoint_path)
    data, text_data = get_data(dataset_name=config.dataset_name,
                               dataset_split=config.dataset_split,
                               tkn_model=config.tkn_model,
                               block_size=config.block_size,
                               batch_size=config.batch_size,
                               num_workers=config.num_workers,
                               pin_memory=config.pin_memory,
                               prefetch_factor=config.prefetch_factor,
                               persistent_workers=config.persistent_workers,
                               max_samples=config.max_samples,
                               resume_chunks=resume_chunks)
    
    print(f'Resuming Training from Iter: {start_iter} from {checkpoint_path}')
else:
    data, text_data = get_data(dataset_name=config.dataset_name,
                               dataset_split=config.dataset_split,
                               tkn_model=config.tkn_model,
                               block_size=config.block_size,
                               batch_size=config.batch_size,
                               num_workers=config.num_workers,
                               pin_memory=config.pin_memory,
                               prefetch_factor=config.prefetch_factor,
                               persistent_workers=config.persistent_workers,
                               max_samples=config.max_samples)

scheduler = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda = lr_lambda,
    last_epoch = start_iter - 1
)

loss_fn = CrossEntropyLoss()

@torch.no_grad()
def validate(model, val_prompt, max_tokens, top_k):
    model.eval()
    eos_id = tkn.eos_id

    input_ids = torch.tensor(tkn.encode(val_prompt), device=config.device).unsqueeze(0)
    B, S = input_ids.shape
    total_seq_len = S + max_tokens

    kv_cache = [
        KV_Cache(batch_size=B, max_seq_len=total_seq_len)
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

            if next_token.item() == eos_id:
                break

            generated = torch.cat([generated, next_token], dim=1)
            logits = model(next_token, kv_cache)

    return tkn.decode(generated[0].tolist())

MAX_ITERS = 15000 + start_iter
print(f'Starting Training from iter {start_iter} to {MAX_ITERS}')

for i, (x, y) in enumerate(tqdm(data, desc='Training', total=MAX_ITERS), start=start_iter):
    
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
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    if i == MAX_ITERS - 1:
        
        validation_output = validate(model, config.validation_prompt, max_tokens=256, top_k=10)

        with open('training/validation.txt', 'a') as file:
            file.write(f"Iteration {i+1}:\n")
            file.write(validation_output + '\n')
            file.write('-' * 50 + '\n\n')
        
        save_checkpoint(model, optimizer, 'training/checkpoints', f'{i+1}_chkpnt.pth', loss, i+1, chunks_yielded=resume_chunks + text_data.chunks_yielded)
        
        print(f"\nValidation output:\n{validation_output}\n")
        print(f"Checkpoint saved: training/checkpoints/{i+1}_chkpnt.pth")
        break