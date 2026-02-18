* No mixed precision training (autocast + GradScaler) → will OOM or be extremely slow on 6 GB VRAM
* No gradient clipping → high risk of exploding gradients in small-model training
* No learning rate warmup or scheduler (fixed LR=1e-4 forever) → poor convergence, instability likely
* No evaluation loop / perplexity monitoring during training → blind to whether model is improving or diverging
* **No torch.compile usage → missing 15–30% speedup + potential memory reduction on PyTorch 2.3+**
* No gradient accumulation → batch=4 is too small for good optimization at this scale
* No TF32 / allow reduced precision matmul → missing free speedup on Ada GPUs
* Validation prompt is never used → no qualitative generation sanity check during training
* ***No torch.backends.cudnn.benchmark = True → potential 5–20% throughput loss***
* ***max\_samples=None → potentially trains forever on huge dataset without defined stopping point***