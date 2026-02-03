* No mixed precision training (autocast + GradScaler) → will OOM or be extremely slow on 6 GB VRAM
* No gradient clipping → high risk of exploding gradients in small-model training
* No learning rate warmup or scheduler (fixed LR=1e-4 forever) → poor convergence, instability likely
* **RoPE frequencies (freqs\_cis) never passed during training → positional information never learned**
* **RoPE application broken in generation loop (wrong slicing logic) → garbage after a few generated tokens**
* **Training forward call misses freqs\_cis argument → Attention receives None → no RoPE during pretraining**
* No evaluation loop / perplexity monitoring during training → blind to whether model is improving or diverging
* **No torch.compile usage → missing 15–30% speedup + potential memory reduction on PyTorch 2.3+**
* No gradient accumulation → batch=4 is too small for good optimization at this scale
* No TF32 / allow reduced precision matmul → missing free speedup on Ada GPUs
* ***GELU instead of modern GLU variants (SwiGLU / GeGLU) → 3–10% worse quality at same compute***
* Head dimension 32 is acceptable but on the low side → many recent small models prefer 48–64
* 32 layers × 384 dim is relatively deep \& narrow → recent small-model scaling studies favor wider+shallower
* Validation prompt is never used → no qualitative generation sanity check during training
* ***No torch.backends.cudnn.benchmark = True → potential 5–20% throughput loss***
* Attention still uses manual matmul path when kv\_cache is present → much slower decode, higher memory vs SDPA
* *Dropout in attention is 0.0 → might benefit from small attn dropout (0.05–0.1) for regularization*
* ***max\_samples=None → potentially trains forever on huge dataset without defined stopping point***
* **use register_buffer for kv_cache storage**