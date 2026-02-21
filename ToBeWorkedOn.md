* No mixed precision training (autocast + GradScaler) → will OOM or be extremely slow on 6 GB VRAM
* No gradient clipping → high risk of exploding gradients in small-model training
* No learning rate warmup or scheduler (fixed LR=1e-4 forever) → poor convergence, instability likely
* **No torch.compile usage → missing 15–30% speedup + potential memory reduction on PyTorch 2.3+**
* No TF32 / allow reduced precision matmul → missing free speedup on Ada GPUs
* ***No torch.backends.cudnn.benchmark = True → potential 5–20% throughput loss***