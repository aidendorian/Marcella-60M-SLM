<p align="center">
  <img src="dark.png" alt="Marcella Logo" width="120"/>
</p>



<p align="center">
  <img src="https://img.shields.io/badge/parameters-60M-blue" />
  <img src="https://img.shields.io/badge/vocab-32K-lightgrey" />
  <img src="https://img.shields.io/badge/context-1024-green" />
  <img src="https://img.shields.io/badge/license-MIT-orange" />
</p>

---

## Overview

Marcella is a ~60M parameter decoder-only transformer language model trained from scratch using PyTorch. The architecture draws from modern LLM design: RoPE positional embeddings, RMSNorm, SwiGLU FFN, weight-tied embeddings, and a custom per-layer KV cache for inference. Everything — tokenizer, data pipeline, training loop, and model — is built from scratch.

---

## Architecture

| Component | Value |
|---|---|
| Parameters | ~60M |
| Embedding dim | 576 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Head dim | 48 |
| FFN hidden dim | 1,536 (× 2/3 × 4 rule) |
| Context length | 1,024 |
| Vocabulary size | 32,000 |
| Positional encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| FFN activation | SwiGLU |
| Embeddings | StableEmbedding (bitsandbytes) |
| Weight tying | Yes (embedding ↔ LM head) |
| Biases | None (all linear layers are bias-free) |
| dtype | bfloat16 |

### Attention

The attention module is a standard multi-head self-attention with fused QKV projection and two distinct forward paths depending on whether a KV cache is active:

**Training path (`kv_cache=None`):**
RoPE is applied to Q and K over the full sequence, then `scaled_dot_product_attention` (PyTorch's Flash Attention kernel) is used with causal masking and dropout.

**Inference path (prefill — `cache_len == 0`):**
The full prompt is processed at once with Flash Attention (causal, no dropout). The resulting K and V tensors are written into the cache.

**Inference path (decode — `cache_len > 0`):**
Only the new token's Q, K, V are computed. RoPE is applied with a position offset equal to the current cache length so positional encodings stay consistent with the cached keys. The new K/V are appended, then attention is computed manually over the full cached context: `(Q @ K_all.T) * scale → softmax → @ V_all`.

This means Flash Attention is used where sequence length is long (prefill), and manual attention is used where it's short (single decode step), which is the correct tradeoff.

### RoPE

RoPE frequencies are precomputed once up to `2 × max_seq_len` and registered as non-persistent buffers. This gives headroom for generation beyond the training context without recomputation. The implementation rotates even/odd dimension pairs separately and interleaves them back, matching the standard formulation.

```
rot_even = x_even * cos − x_odd * sin
rot_odd  = x_even * sin + x_odd * cos
```

During decode, `apply_rope_offset` slices the correct position from the precomputed tables using the cache length as an offset, keeping Q and K in sync with their absolute positions.

### KV Cache

Each transformer block gets its own `KV_Cache` instance. The cache pre-allocates fixed-size tensors of shape `(B, H, max_seq_len, head_dim)` in bfloat16 on GPU at init time — no dynamic allocation during generation. A `cache_len` pointer tracks how far the cache has been filled, and `update()` writes new K/V slices into the appropriate positions. `get_kv()` returns only the filled slice, so attention never attends over padding.

`Marcella.init_kv_cache()` constructs one cache per layer and returns a list that's passed through the forward call at each decode step.

### FFN (SwiGLU)

The feedforward block uses SwiGLU with the standard hidden dim scaling of `4 × embed_dim × 2/3`:

```
FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
```

All three projections are bias-free.

### TransformerBlock

Pre-norm architecture: RMSNorm is applied before both the attention and FFN sublayers, with residual connections around each.

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

---

## Tokenizer

A custom SentencePiece tokenizer (`Marcella_vocab_32K_v2`) trained on FineWeb-Edu with a vocabulary of 32,000 tokens. The `Tokenizer` wrapper exposes `encode` and `decode` and surfaces the BOS, EOS, PAD, and UNK ids directly.

---

## Dataset

| Source | Weight |
|---|---|
| FineWeb-Edu | 60% |
| Wikipedia | 25% |
| SlimPajama | 15% |

The dataset is pre-tokenized and stored as binary shards (`.bin` + `.idx` pairs). Each `.idx` file holds byte offsets into the corresponding `.bin` so sequences can be sliced without loading the full shard. `ShardedDataset` is an `IterableDataset` that streams shards in sorted order and supports resuming from a specific `(shard_id, seq_idx)` — which is saved into checkpoints so training can be resumed exactly where it left off.

---

## Training

```bash
uv run python -m training.train
```

| Setting | Value |
|---|---|
| Batch size | 4 |
| Gradient accumulation | 8 steps (effective batch = 32) |
| Max LR | 2e-4 |
| Min LR | 2e-5 |
| Warmup steps | 610 |
| Total steps | 61,035 |
| LR schedule | Linear warmup → cosine decay |
| Optimizer | AdamW |

### Checkpointing

Checkpoints save model weights, optimizer state, scheduler state, PyTorch and CUDA RNG states, the current shard and sequence index, and the W&B run ID. This means training can be resumed with exact reproducibility — the dataloader will rewind to the same position in the same shard.

---

## Evaluation

Evaluated on a held-out tail split (shards 40+, 1,140 sequences):

| Metric | Value |
|---|---|
| Checkpoint | step 488,000 |
| Eval tokens | 1,164,288 |
| Loss | 3.4925 |
| Perplexity | 32.87 |
| Eval throughput | 167,259 tokens/sec |
| Generation speed | ~22.8 tokens/sec |

---

## Project Structure

```
MARCELLA-60M/
├── models/
│   ├── Marcella_vocab_32K_v2.model   # SentencePiece model
│   ├── Marcella_vocab_32K_v2.vocab
│   └── marcella.pt                   # Model weights
├── src/
│   ├── attention.py      # RoPE, KV cache, multi-head attention
│   ├── marcella.py       # TransformerBlock, FFN, Marcella model
│   └── tokenizer.py      # SentencePiece wrapper
├── training/
│   ├── checkpoints/
│   ├── data/             # Pre-tokenized binary shards
│   ├── checkpoint.py     # Save/load with full RNG state
│   ├── config.py         # All hyperparameters
│   ├── dataloader.py     # Sharded streaming dataset + resume
│   ├── train.py          # Training loop
│   └── validation.txt
├── dark.png
├── pyproject.toml
└── uv.lock
```

---

## License

MIT