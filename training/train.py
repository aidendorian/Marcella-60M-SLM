import torch
import math
import time
from torch.optim import lr_scheduler
from bitsandbytes.optim.adamw import AdamW8bit
from tqdm import tqdm
from training.config import Config
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from torch.amp.autocast_mode import autocast
import os
from training.dataloader import get_data, get_val_batch
from src.marcella import Marcella
from src.tokenizer import Tokenizer
from src.attention import KV_Cache
from training.checkpoint import load_checkpoint, save_checkpoint
import wandb

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

os.makedirs('training/checkpoints', exist_ok=True)

wandb.login(key=os.getenv('WANDB_API_KEY'))

config = Config()
device = config.device
print(f'Device: {device}')

model = Marcella(vocab_size=config.vocab_size,
                 embed_dim=config.embed_dim,
                 num_transformer_layers=config.num_transformer_layers,
                 num_heads=config.num_heads,
                 attn_dropout=config.attn_dropout,
                 ffn_dropout=config.ffn_dropout).to(device)

tkn = Tokenizer(tokenizer_model=config.tkn_model)

no_decay = {"bias", "rmsnorm.weight"}
param_groups = [
    {"params": [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)]},
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)]},
]

optimizer = AdamW8bit(param_groups, lr=config.LR_MAX, betas=(0.9, 0.95), eps=1e-8)

def lr_lambda(step: int) -> float:
    if step < config.WARMUP_STEPS:
        return step / config.WARMUP_STEPS
    t = step - config.WARMUP_STEPS
    cosine = 0.5 * (1.0 + math.cos(math.pi * t / config.T_MAX))
    lr = config.LR_MIN + (config.LR_MAX - config.LR_MIN) * cosine
    return lr / config.LR_MAX

start_iter    = 0
start_shard   = 0
start_seq     = 0
wandb_run_id  = None

RESUME_FROM_CHECKPOINT = False
checkpoint_path = 'training/checkpoints/15000_chkpnt.pth'

if RESUME_FROM_CHECKPOINT:
    _, model, optimizer, start_iter, start_shard, start_seq, wandb_run_id = load_checkpoint(model, optimizer, checkpoint_path)
    print(f'Resuming from iter {start_iter}, '
          f'shard {start_shard}, seq {start_seq}')

data, text_data = get_data(data_dir=config.data_dir,
                           block_size=config.block_size,
                           batch_size=config.batch_size,
                           num_workers=config.num_workers,
                           pin_memory=config.pin_memory,
                           prefetch_factor=config.prefetch_factor,
                           persistent_workers=config.persistent_workers,
                           start_shard=start_shard,
                           start_seq=start_seq)

val_x, val_y = get_val_batch(data_dir=config.data_dir,
                             block_size=config.block_size,
                             batch_size=config.batch_size,
                             device=device)

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=start_iter - 1)
loss_fn = CrossEntropyLoss()

run = wandb.init(
    project=os.getenv('WANDB_PROJECT'),
    entity=os.getenv('WANDB_ENTITY'),
    id=wandb_run_id,
    resume="allow",
    reinit=False,
    config={
        "vocab_size":         config.vocab_size,
        "embed_dim":          config.embed_dim,
        "num_layers":         config.num_transformer_layers,
        "num_heads":          config.num_heads,
        "block_size":         config.block_size,
        "batch_size":         config.batch_size,
        "accumulation_steps": config.accumulation_steps,
        "lr_max":             config.LR_MAX,
        "lr_min":             config.LR_MIN,
        "warmup_steps":       config.WARMUP_STEPS,
        "t_max":              config.T_MAX,
        "resume_from":        checkpoint_path if RESUME_FROM_CHECKPOINT else None,
    },
)

wandb_run_id = run.id


@torch.no_grad()
def compute_val_loss(model, val_x, val_y):
    model.eval()
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(val_x)
        B, T, C = logits.shape
        loss = loss_fn(logits.view(B * T, C), val_y.view(B * T))
    return loss.item()

@torch.no_grad()
def validate(model, val_prompt, max_tokens, top_k, temperature=0.8, rep_penalty=1.3):
    model.eval()
    eos_id = tkn.eos_id

    input_ids = torch.tensor(
        tkn.encode(val_prompt), device=config.device
    ).unsqueeze(0)
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

            for token_id in set(generated[0].tolist()):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= rep_penalty
                else:
                    next_token_logits[0, token_id] *= rep_penalty

            next_token_logits = next_token_logits / temperature

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
print(f'Starting from iter {start_iter} to {MAX_ITERS}')

tokens_per_iter = config.block_size * config.batch_size
step_start_time = time.time()

for i, (x, y) in enumerate(
    tqdm(data, desc='Training', total=MAX_ITERS, initial=start_iter),
    start=start_iter
):
    model.train()

    x, y = x.to(device), y.to(device)
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(x)
        B, T, C = logits.shape
        logits = logits.view(B*T,C)
        loss = loss_fn(logits, y.view(B * T))
        loss = loss / config.accumulation_steps
    loss.backward()

    if (i + 1) % config.accumulation_steps == 0:
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if (i + 1) % 100 == 0:
        elapsed      = time.time() - step_start_time
        toks_per_sec = (tokens_per_iter * 100) / elapsed
        train_loss   = loss.item() * config.accumulation_steps
        perplexity   = math.exp(min(train_loss, 20))

        run.log({
            "train/loss":           train_loss,
            "train/perplexity":     perplexity,
            "train/tokens_per_sec": toks_per_sec,
            "train/lr":             scheduler.get_last_lr()[0],
        }, step=i)

        step_start_time = time.time()

    if (i + 1) % 2000 == 0:
        val_loss       = compute_val_loss(model, val_x, val_y)
        val_perplexity = math.exp(min(val_loss, 20))

        run.log({
            "val/loss":       val_loss,
            "val/perplexity": val_perplexity,
        }, step=i)

    if (i + 1) % 5000 == 0 and i != MAX_ITERS - 1:
        save_checkpoint(
            model, optimizer, 'training/checkpoints', f'{i+1}_chkpnt.pth',
            loss, i + 1,
            shard_id=text_data.current_shard,
            seq_idx=text_data.current_seq,
            wandb_run_id=run.id,
        )
        print(f"Checkpoint saved: training/checkpoints/{i+1}_chkpnt.pth")

    if i == MAX_ITERS - 1:
        validation_output = validate(
            model, config.validation_prompt,
            max_tokens=256, top_k=50,
            temperature=0.8, rep_penalty=1.3
        )

        with open('training/validation.txt', 'a') as file:
            file.write(f"Iteration {i+1}:\n")
            file.write(validation_output + '\n')
            file.write('-' * 50 + '\n\n')

        save_checkpoint(
            model, optimizer, 'training/checkpoints', f'{i+1}_chkpnt.pth',
            loss, i + 1,
            shard_id=text_data.current_shard,
            seq_idx=text_data.current_seq,
            wandb_run_id=run.id,
        )

        run.log({
            "val/generation": wandb.Html(f"<pre>{validation_output}</pre>")
        }, step=i)

        print(f"\nValidation output:\n{validation_output}\n")
        print(f"Checkpoint saved: training/checkpoints/{i+1}_chkpnt.pth")
        break

run.finish()