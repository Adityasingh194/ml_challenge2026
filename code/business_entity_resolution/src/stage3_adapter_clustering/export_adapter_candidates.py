"""Export the adapter (fused) test candidates: one row per (S1, adapter pair), top-20 per S1.
is_new = 1 -> the pair is NOT among that S1's existing 120 candidates (a pair the adapter adds).
CPU only (~10-12 GB RAM); runs alongside GPU jobs. Writes output/adapter_candidates_test.tsv (+ France-only file)."""
import glob, os, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

T0 = time.time()
W = os.path.join(os.environ.get("ER_ROOT", "/home/parth/Desktop/trial work"), "pipeline_run")   # same ER_ROOT as the notebook
LOG = f"{W}/experiments/logs/export.log"


def log(m):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [export +{time.time() - T0:4.0f}s] {m}"
    print(line, flush=True)
    for f in (LOG, f"{W}/experiments/experiments.log"):
        open(f, "a").write(line + "\n")


z = np.load(f"{W}/experiments/adapter_fast/fused_test_top20.npz"); ids, sc = z["ids"], z["sc"]
fr = glob.glob(f"{W}/scratch/frames/*")[0]
s1 = pq.read_table(f"{fr}/test_s1_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
po = pq.read_table(f"{fr}/test_pool_norm.parquet", columns=["entity_id", "pool_source"]).to_pandas()
K, NP = ids.shape[1], len(po)
r = np.repeat(np.arange(len(s1), dtype=np.int64), K); c = ids.reshape(-1); v = c >= 0
r, c, rk, cs = r[v], c[v], np.tile(np.arange(K), len(s1))[v], sc.reshape(-1)[v]
log(f"adapter top-{K}: {len(r):,} pairs for {len(s1):,} test S1")
ex = pq.read_table(glob.glob(f"{W}/cache/retrieval/test_*/scores_*/final_preds.parquet")[0], columns=["s1_pos", "pool_pos"])
ek = ex.column(0).to_numpy().astype(np.int64) * NP + ex.column(1).to_numpy()
del ex
is_new = ~np.isin(r * NP + c, ek)
del ek
out = pd.DataFrame({"source1_entity_id": s1.entity_id.to_numpy(object)[r],
                    "candidate_entity_id": po.entity_id.to_numpy(object)[c],
                    "fused_rank": rk + 1, "fused_cos": cs.round(5),
                    "cand_source": po.pool_source.to_numpy(object)[c],
                    "country": s1.country_norm.to_numpy(object)[r], "is_new": is_new.astype(np.int8)})
log(f"is_new pairs {int(is_new.sum()):,} ({is_new.mean():.1%}); by country: "
    + str(out.groupby('country')['is_new'].agg(['size', 'sum']).to_dict('index')))
path = f"{W}/output/adapter_candidates_test.tsv"
out.to_csv(path, sep="\t", index=False, chunksize=2_000_000)
log(f"written {path}")
fr_ = out[out["country"] == "France"]
fr_.to_csv(f"{W}/output/adapter_candidates_test_France.tsv", sep="\t", index=False)
log(f"written France-only file: {len(fr_):,} rows; done in {time.time() - T0:.0f}s")
