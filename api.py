"""
Marcella – FastAPI streaming backend
Run: uv run uvicorn api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.marcella import Marcella
from src.tokenizer import Tokenizer
from training.config import Config

cfg = Config()
app = FastAPI(title="Marcella API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = Tokenizer(tokenizer_model=cfg.tkn_model)

model = Marcella(
    vocab_size=cfg.vocab_size,
    embed_dim=cfg.embed_dim,
    num_transformer_layers=cfg.num_transformer_layers,
    num_heads=cfg.num_heads,
)
checkpoint = torch.load("models/marcella.pt", map_location=cfg.device, weights_only=True)
state = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state)
model.to(cfg.device)
model.eval()

print(f"[marcella] model loaded — {sum(p.numel() for p in model.parameters()):,} params")


_DONE = object()  # sentinel — avoids StopIteration crossing into asyncio Future


def token_id_to_text(token_id: int) -> str:
    piece = tokenizer.tokenizer.IdToPiece(token_id)
    return piece.replace("▁", " ")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int    = Field(default=200, ge=1,   le=1024)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int         = Field(default=50,  ge=1,   le=500)


@torch.inference_mode()
def _generate_tokens(req: GenerateRequest):
    """Regular (non-async) generator. Yields str tokens then the _DONE sentinel."""
    ids = tokenizer.encode(req.prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=cfg.device)

    kv_cache = model.init_kv_cache(
        batch_size=1,
        max_seq_len=cfg.max_seq_len,
        device=cfg.device,
        dtype=torch.float32,
    )

    logits = model(input_ids, kv_cache)

    for _ in range(req.max_tokens):
        next_logits = logits[:, -1, :] / req.temperature

        if req.top_k > 0:
            vals, _ = torch.topk(next_logits, req.top_k)
            next_logits = next_logits.masked_fill(
                next_logits < vals[:, -1].unsqueeze(-1), float("-inf")
            )

        probs   = F.softmax(next_logits.float(), dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.eos_id:
            break

        yield token_id_to_text(next_id.item()) # type: ignore
        logits = model(next_id, kv_cache)

    yield _DONE  # signal completion without raising StopIteration


def _next_token(gen):
    """Called in executor thread. Returns token str or _DONE sentinel."""
    return next(gen)


@app.post("/generate")
async def generate(req: GenerateRequest):
    async def event_stream():
        loop = asyncio.get_event_loop()
        gen  = _generate_tokens(req)
        while True:
            # run_in_executor + StopIteration = TypeError in Python 3.12
            # so the generator yields _DONE instead of stopping
            result = await loop.run_in_executor(None, _next_token, gen)
            if result is _DONE:
                yield "data: [DONE]\n\n"
                break
            payload = json.dumps({"token": result})
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(cfg.device)}