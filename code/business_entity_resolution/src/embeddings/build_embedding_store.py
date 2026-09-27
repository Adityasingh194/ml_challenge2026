"""Build the Qwen3-Embedding-0.6B store the pipeline reads (notebook cells 0c / 12 / 18).

Layout written, one folder per (split, source, view):
    <out>/{train,test}/{s1,s2,s3}/{name,addr}/meta.json, rows.npy, emb_00000.npy, emb_00001.npy, ...

  - emb_XXXXX.npy : int8 (chunk_rows, dim), L2-normalised vectors * 127 (qwen_embed.quantize_int8)
  - rows.npy      : embedding row i belongs to TSV row rows[i]. Written in TSV order here, so rows = arange(n).
  - meta.json     : model, view, dim, store_dtype, chunk_rows, n, file_split, source

Dims: name view 256, addr view 512 (Matryoshka truncation, see qwen_embed.DEFAULTS).
Input text is RAW business_name / business_address (blank address falls back to the name).

Resumable: a folder whose meta.json and all chunks exist is skipped. Run one process per GPU with --device and
--only to split the 12 folders across GPUs, e.g.
    python build_embedding_store.py --device cuda:0 --only train/s2 train/s3
    python build_embedding_store.py --device cuda:1 --only train/s1 test/s1 test/s2 test/s3
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_embed as QE  # noqa: E402

ROOT = os.environ.get("ER_ROOT", "/home/parth/Desktop/trial work")
CHUNK_ROWS = 20_000


def read_tsv(path):
    csv.field_size_limit(10 ** 9)
    names, addrs = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            names.append(row.get("business_name") or "")
            addrs.append(row.get("business_address") or "")
    return names, addrs


def done(d):
    mp = os.path.join(d, "meta.json")
    if not (os.path.exists(mp) and os.path.exists(os.path.join(d, "rows.npy"))):
        return False
    meta = json.load(open(mp))
    return all(os.path.exists(os.path.join(d, f"emb_{c:05d}.npy")) for c in range(math.ceil(meta["n"] / meta["chunk_rows"])))


def build_one(split, s, view, names, addrs, tok, net, device, args):
    d = os.path.join(args.out, split, f"s{s}", view)
    if done(d):
        print(f"[store] {split}/s{s}/{view}: complete, skipped")
        return
    os.makedirs(d, exist_ok=True)
    n, dim = len(names), QE.view_dim(view)
    texts = QE.text_for(view, names, addrs)
    t0 = time.time()
    for c in range(math.ceil(n / CHUNK_ROWS)):
        p = os.path.join(d, f"emb_{c:05d}.npy")
        if os.path.exists(p):
            continue
        lo, hi = c * CHUNK_ROWS, min(n, (c + 1) * CHUNK_ROWS)
        vec = QE.embed_batchwise(texts[lo:hi], tok, net, device, dim, args.max_seq_len, args.batch_tokens,
                                 args.max_batch_size)
        np.save(p + ".tmp.npy", QE.quantize_int8(vec))
        os.replace(p + ".tmp.npy", p)
        print(f"[store] {split}/s{s}/{view} chunk {c} rows {hi:,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(d, "rows.npy"), np.arange(n, dtype=np.int64))
    meta = dict(model=args.model, view=view, dim=dim, store_dtype="int8", chunk_rows=CHUNK_ROWS, n=n,
                variant="qwen3-06b-prod", file_split=split, source=f"s{s}")
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)   # written last = folder complete


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"), help="{train,test}/{split}_source{1,2,3}.tsv")
    ap.add_argument("--out", default=os.path.join(ROOT, "embedding_store"))
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
                    help="HF id or local path of Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="subset like train/s1 test/s3 (default: all 6)")
    ap.add_argument("--max-seq-len", type=int, default=QE.DEFAULTS["max_seq_len"])
    ap.add_argument("--batch-tokens", type=int, default=QE.DEFAULTS["batch_tokens"])
    ap.add_argument("--max-batch-size", type=int, default=QE.DEFAULTS["max_batch_size"])
    args = ap.parse_args()

    todo = [(sp, s) for sp in ("train", "test") for s in (1, 2, 3)
            if args.only is None or f"{sp}/s{s}" in args.only]
    tok, net, device = QE.load_model(args.model, "float16", args.device)
    for sp, s in todo:
        if all(done(os.path.join(args.out, sp, f"s{s}", v)) for v in ("name", "addr")):
            print(f"[store] {sp}/s{s}: complete, skipped")
            continue
        names, addrs = read_tsv(os.path.join(args.data, sp, f"{sp}_source{s}.tsv"))
        for view in ("name", "addr"):
            build_one(sp, s, view, names, addrs, tok, net, device, args)


if __name__ == "__main__":
    main()
