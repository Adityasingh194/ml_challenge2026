# 0c. Precomputed Qwen3-Embedding-0.6B vectors (name 256-d + address 512-d, int8) for 100% of train and test.
# Layout: embedding_store/{train,test}/{s1,s2,s3}/{name,addr}/{meta.json, rows.npy, emb_00000.npy ...}
import json
import math
import os
import subprocess

EMB_S3_URI = "s3://amazon-ml-summer-challenge1/embedding_store/"
EMB_LOCAL = os.path.join(LOCAL_ROOT, "embedding_store")


def embeddings_complete(root):
    """All 12 (split, source, view) folders present with meta.json, rows.npy and ceil(n / chunk_rows) chunks."""
    missing = []
    for split in ("train", "test"):
        for s in ("s1", "s2", "s3"):
            for view in ("name", "addr"):
                d = os.path.join(root, split, s, view)
                mp = os.path.join(d, "meta.json")
                if not (os.path.exists(mp) and os.path.exists(os.path.join(d, "rows.npy"))):
                    missing.append(f"{split}/{s}/{view}: meta/rows")
                    continue
                with open(mp) as f:
                    meta = json.load(f)
                need = math.ceil(meta["n"] / meta["chunk_rows"])
                have = sum(1 for i in range(need) if os.path.exists(os.path.join(d, f"emb_{i:05d}.npy")))
                if have != need:
                    missing.append(f"{split}/{s}/{view}: {have}/{need} chunks")
    return missing


_missing = embeddings_complete(EMB_LOCAL)
if _missing:
    print(f"syncing {EMB_S3_URI} -> {EMB_LOCAL} ({len(_missing)} folder(s) incomplete) ...")
    subprocess.run(["aws", "s3", "sync", EMB_S3_URI, EMB_LOCAL, "--only-show-errors"], check=True)
    _missing = embeddings_complete(EMB_LOCAL)
assert not _missing, f"embeddings still incomplete after sync: {_missing}"
_n_files = sum(len(fs) for _, _, fs in os.walk(EMB_LOCAL))
_gb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(EMB_LOCAL) for f in fs) / 1e9
print(f"embeddings complete: {EMB_LOCAL}  ({_n_files:,} files, {_gb:.2f} GB)")
