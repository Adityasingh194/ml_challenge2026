"""Qwen3-Embedding — the local embedding code, lifted out of the BER pipeline.

Extracted verbatim (logic unchanged) from:
    .../student_resource/code/business_entity_resolution/src/stage2_embed.py
    lines 165-262  (_load_model, _last_token_pool, _embed_batchwise, _text_for)

The original depends on the pipeline's Cfg/select/common modules and its chunked
on-disk store. This file is standalone: config is plain keyword arguments, and it
returns plain numpy arrays. Nothing here talks to the network.

Model weights, already on disk (no download needed):
    /home/niteesh859/ber_bundle/hf/models/Qwen3-Embedding-0.6B   1.19 GB  fp16
    /home/niteesh859/ber_bundle/hf/models/Qwen3-Embedding-4B     8.04 GB  fp16

Both are Apache-2.0 and under the competition's 8B parameter limit. Because this
runs locally on open weights, it is not an "external database, API, or service" —
unlike the jev path.

Settings below are the originals from config.yaml (the laptop profile):
    dim 512 (Matryoshka), name_dim 256, dtype float16,
    max_seq_len 96, batch_tokens 4096, max_batch_size 256

Two views per record, retrieved separately and unioned downstream:
    "name"  business_name    (raw)
    "addr"  business_address (raw, falls back to the name when blank)

RAW text, not cleaned text: Qwen3's tokenizer already handles casing, Unicode and
punctuation, and cleaning throws away signal it can use (legal-suffix rewrites,
transliteration of the ~9% of S2 names that are non-Latin).

Qwen3-Embedding is trained with Matryoshka representation learning, so truncating
to the first `dim` components and renormalising is a supported operation, not a hack.
"""
from __future__ import annotations

import os

import numpy as np

LOCAL_0_6B = "/home/niteesh859/ber_bundle/hf/models/Qwen3-Embedding-0.6B"
LOCAL_4B = "/home/niteesh859/ber_bundle/hf/models/Qwen3-Embedding-4B"

DEFAULTS = {
    "dim": 512,           # Matryoshka width for the addr view
    "name_dim": 256,      # name-only view is cheaper; 0 = same as dim
    "dtype": "float16",
    "max_seq_len": 96,
    "batch_tokens": 4096,     # length-bucketed dynamic batching; lower if OOM on 4 GB
    "max_batch_size": 256,    # upper bound on sequences per batch
}


# --------------------------------------------------------------------- model


def load_model(model: str = LOCAL_0_6B, dtype: str = "float16", device: str | None = None):
    """Returns (tokenizer, model, device). Resolves weights locally only."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch_dtype = torch.float32
        print("[qwen_embed] no CUDA device — embedding on CPU will be very slow")

    # A local path or HF_HUB_OFFLINE both mean: never touch the hub.
    local_only = os.path.isdir(model) or os.environ.get("HF_HUB_OFFLINE") == "1" \
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    kw = {"local_files_only": True} if local_only else {}

    tok = AutoTokenizer.from_pretrained(model, padding_side="left", **kw)
    net = AutoModel.from_pretrained(model, torch_dtype=torch_dtype, **kw).to(device).eval()

    n_par = sum(p.numel() for p in net.parameters())
    print(f"[qwen_embed] {model} on {device} ({torch_dtype}, {n_par/1e9:.3f}B params, "
          f"{n_par*2/2**30:.2f} GiB weights, hidden={net.config.hidden_size})")
    if n_par > 8.5e9:   # competition rule: <= 8B parameters
        raise SystemExit(f"{model} has {n_par/1e9:.2f}B parameters; the 8B limit forbids it")
    return tok, net, device


def view_dim(view: str, dim: int = DEFAULTS["dim"], name_dim: int = DEFAULTS["name_dim"]) -> int:
    """Matryoshka width per view. The name view is short (a business name is a few
    tokens); the address view keeps the full `dim` because addresses carry more."""
    return (name_dim or dim) if view == "name" else dim


# --------------------------------------------------------------------- encode


def last_token_pool(hidden, attention_mask):
    """Qwen3-Embedding pools the final token, not the CLS token or a mean."""
    import torch

    left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padded:
        return hidden[:, -1]
    idx = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def embed_batchwise(texts: list[str], tok, model, device, dim: int,
                    max_seq_len: int = DEFAULTS["max_seq_len"],
                    batch_tokens: int = DEFAULTS["batch_tokens"],
                    max_batch_size: int = DEFAULTS["max_batch_size"]) -> np.ndarray:
    """Length-bucketed encode. Returns float32 (n, dim), L2-normalised.

    Batch size is chosen per batch from a token budget, so short records pack more
    rows in and long ones fewer — that is what keeps peak VRAM flat.
    """
    import torch

    lens = np.fromiter((len(t) for t in texts), dtype=np.int32, count=len(texts))
    order = np.argsort(lens, kind="stable")                    # short-with-short
    out = np.empty((len(texts), dim), dtype=np.float32)

    i = 0
    with torch.inference_mode():
        while i < len(order):
            approx = max(1, min(max_seq_len, lens[order[i]] // 3 + 4))
            bs = max(1, min(max_batch_size, batch_tokens // approx))
            idx = order[i: i + bs]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True,
                      max_length=max_seq_len, return_tensors="pt").to(device)
            hid = model(**enc).last_hidden_state
            vec = last_token_pool(hid, enc["attention_mask"]).float()
            vec = vec[:, :dim]                                  # Matryoshka truncation
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out[idx] = vec.cpu().numpy()
            i += bs
    return out


def text_for(view: str, name, addr) -> list[str]:
    """Raw text per view. No cleaning — the tokenizer is the normaliser here.

    A blank address (3.3% of S2/S3) falls back to the name: every empty string would
    otherwise embed to the same vector, and those rows would become each other's
    nearest neighbours and eat the candidate budget with pure noise.
    """
    if view == "name":
        return [n or "" for n in name]
    return [(a or n or "") for n, a in zip(name, addr)]


# --------------------------------------------------------------------- store


def quantize_int8(vec: np.ndarray) -> np.ndarray:
    """L2-normalised floats -> int8 at scale 127. The pipeline's store_dtype default:
    4x smaller than fp16 on disk, and cosine survives it because every vector is
    already unit-norm."""
    return np.clip(np.rint(vec * 127.0), -127, 127).astype(np.int8)


def dequantize_int8(q: np.ndarray) -> np.ndarray:
    v = q.astype(np.float32) / 127.0
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def cosine_topk(queries: np.ndarray, pool: np.ndarray, k: int = 100,
                block: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Blocked exact top-k by cosine. Both sides must be L2-normalised, so the
    similarity is a plain dot product. Returns (scores, indices), each (n_q, k).

    This is the step that makes a bi-encoder cheap: the model runs once per RECORD,
    and every PAIR comparison after that is a matmul — not a forward pass.
    """
    nq = len(queries)
    k = min(k, len(pool))
    scores = np.empty((nq, k), dtype=np.float32)
    idx = np.empty((nq, k), dtype=np.int64)
    for s in range(0, nq, block):
        e = min(s + block, nq)
        sim = queries[s:e] @ pool.T
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        got = sim[rows, part]
        srt = np.argsort(-got, axis=1)
        idx[s:e] = part[rows, srt]
        scores[s:e] = got[rows, srt]
    return scores, idx


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Embed a TSV's name/address columns with Qwen3.")
    p.add_argument("--model", default=LOCAL_0_6B)
    p.add_argument("--tsv", required=True, help="input TSV with the columns below")
    p.add_argument("--name-col", default="business_name")
    p.add_argument("--addr-col", default="business_address")
    p.add_argument("--view", default="addr", choices=("name", "addr"))
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--out", default="", help="write a .npy here (int8) instead of just timing")
    p.add_argument("--dim", type=int, default=DEFAULTS["dim"])
    p.add_argument("--name-dim", type=int, default=DEFAULTS["name_dim"])
    p.add_argument("--max-seq-len", type=int, default=DEFAULTS["max_seq_len"])
    p.add_argument("--batch-tokens", type=int, default=DEFAULTS["batch_tokens"])
    p.add_argument("--max-batch-size", type=int, default=DEFAULTS["max_batch_size"])
    p.add_argument("--dtype", default=DEFAULTS["dtype"])
    args = p.parse_args()

    import csv
    import time
    csv.field_size_limit(10 ** 9)

    names, addrs = [], []
    with open(args.tsv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            names.append(row.get(args.name_col) or "")
            addrs.append(row.get(args.addr_col) or "")
            if args.limit and len(names) >= args.limit:
                break
    texts = text_for(args.view, names, addrs)
    dim = view_dim(args.view, args.dim, args.name_dim)
    print(f"[qwen_embed] {len(texts):,} rows, view={args.view}, dim={dim}")

    tok, net, device = load_model(args.model, args.dtype)
    t0 = time.time()
    vecs = embed_batchwise(texts, tok, net, device, dim, args.max_seq_len,
                           args.batch_tokens, args.max_batch_size)
    el = time.time() - t0
    print(f"[qwen_embed] {len(texts):,} rows in {el:.1f}s = {len(texts)/el:,.0f} rec/s")
    try:
        import torch
        print(f"[qwen_embed] peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    except Exception:
        pass
    if args.out:
        np.save(args.out, quantize_int8(vecs))
        print(f"[qwen_embed] wrote {args.out} ({os.path.getsize(args.out)/2**20:.1f} MiB, int8)")
