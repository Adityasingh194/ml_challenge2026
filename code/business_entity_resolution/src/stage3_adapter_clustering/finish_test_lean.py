"""Memory-lean TEST path for 'adapter + clustering' -> output/submission_3.
Same computation as adapter_fast.py's test section, but numpy arrays instead of 216M-row DataFrames, features only for
the scored rows, and a streaming validator. Resumes: fused search (npz), CE chunks, cluster chunks are reused."""
import glob, hashlib, json, os, sys, time, resource
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, xgboost as xgb

sys.argv = [sys.argv[0]]
import adapter_fast as AF        # functions + adapter + CE tokenizer; its __main__ block does not run
from adapter_fast import CE_TAG, log, rss, inb, emb, fused_search, score_pairs, raw, texts, CL_COLS, OUT, W, C
T0 = time.time()
TAG = "finish_test"
K = int(os.environ.get("K_NEW", "10"))
SUB = f"{W}/output/{os.environ.get('SUBDIR', 'submission_3')}"
dec = json.load(open(f"{OUT}/decision_k{K}{CE_TAG}.json" if os.path.exists(f"{OUT}/decision_k{K}{CE_TAG}.json") else f"{OUT}/decision.json"))
assert dec["k_new"] == K
rule = dec["rules"]["adapter + clustering"]
b3 = xgb.Booster(); b3.load_model(f"{OUT}/stage2_adapter_plus_clustering_k{K}{CE_TAG}.json" if os.path.exists(f"{OUT}/stage2_adapter_plus_clustering_k{K}{CE_TAG}.json") else f"{OUT}/stage2_adapter_plus_clustering.json")
log(f"lean test path: K_NEW={K}, CE_TAG={CE_TAG!r}, rule {rule}", TAG)
if CE_TAG: assert os.path.exists(f"{OUT}/decision_k{K}{CE_TAG}.json"), "no CE2 decision file"

tdir = glob.glob(f"{C}/retrieval/test_*")[0]
t = pq.read_table(glob.glob(f"{tdir}/scores_*/final_preds.parquet")[0], columns=["s1_pos", "pool_pos", "p"])
s1 = t.column("s1_pos").to_numpy().astype(np.int32); pool = t.column("pool_pos").to_numpy().astype(np.int32)
p = t.column("p").to_numpy().astype(np.float32); del t
fr = glob.glob(f"{W}/scratch/frames/*")[0]
ts1 = pq.read_table(f"{fr}/test_s1_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
tpo = pq.read_table(f"{fr}/test_pool_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
NT1, NTP = len(ts1), len(tpo)
S1_IDS = ts1["entity_id"].to_numpy(dtype=object); POOL_IDS = tpo["entity_id"].to_numpy(dtype=object)   # object, not Arrow str (slow)
log(f"existing test candidates {len(p):,}; RSS {rss():.1f} GB", TAG)

# ---- new fused-only pairs ----
X1 = emb("test", 1); XP = np.concatenate([emb("test", 2), emb("test", 3)], axis=0)
fid, fsc = AF.fused_search_prefix(X1, XP, np.arange(NT1), ts1["country_norm"].values, tpo["country_norm"].values, K, "test")
fk = np.repeat(np.arange(NT1, dtype=np.int64), K) * np.int64(NTP) + fid.reshape(-1)
frank = np.tile(np.arange(K, dtype=np.int16), NT1); fcos = fsc.reshape(-1)
v = fid.reshape(-1) >= 0; fk, frank, fcos = fk[v], frank[v], fcos[v]
ek = s1.astype(np.int64) * NTP + pool
new = ~np.isin(fk, ek)
nidx = np.flatnonzero(new); nidx = nidx[np.argsort(frank[nidx] >= 10, kind="stable")]   # K<=10 first: CE scores reusable
ns1, npool = (fk[nidx] // NTP).astype(np.int32), (fk[nidx] % NTP).astype(np.int32)
log(f"new fused-only pairs {new.sum():,} ({new.sum() / NT1:.2f}/S1); RSS {rss():.1f} GB", TAG)

# ---- all rows = existing + new (arrays) ----
S1 = np.r_[s1, ns1]; PO = np.r_[pool, npool]; P = np.r_[p, np.zeros(len(ns1), np.float32)]
IS_NEW = np.r_[np.zeros(len(s1), np.int8), np.ones(len(ns1), np.int8)]
n_old = len(s1); del s1, pool, p, ns1, npool
# fused rank / cos per row (existing rows found by the fused search too)
FR_ = np.full(len(S1), 99, np.int16); FC = np.full(len(S1), np.nan, np.float32)
FR_[n_old:] = frank[nidx]; FC[n_old:] = fcos[nidx]
old_hit = ~new
j = pd.Index(ek).get_indexer(fk[old_hit])
FR_[j] = frank[old_hit]; FC[j] = fcos[old_hit]
del ek, fk, frank, fcos, new, old_hit, j, nidx
# entity context (rank by p within S1, top-1 / top-2 p, band count), as entity_ctx()
o = np.lexsort((-P, S1)); ss = S1[o]
st = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]; sz = np.diff(np.r_[st, len(ss)])
RK = np.empty(len(P), np.int32); RK[o] = np.arange(len(ss)) - np.repeat(st, sz)
p1 = np.zeros(NT1, np.float32); p1[ss[st]] = P[o[st]]
p2 = np.zeros(NT1, np.float32); two = sz > 1; p2[ss[st[two]]] = P[o[st[two] + 1]]
nb = np.bincount(S1[inb(P)], minlength=NT1).astype(np.float32)
del o, ss, st, sz, two
SC = np.flatnonzero(inb(P) | (IS_NEW == 1))
log(f"rows {len(P):,}; to score {len(SC):,}; RSS {rss():.1f} GB", TAG)

# fused cos for scored existing rows missing it
need = SC[np.isnan(FC[SC])]
FC[need] = AF.pair_cos(X1, XP, S1[need].astype(np.int64), PO[need].astype(np.int64))
del X1
# CE (resumes from the chunks already scored as 'test')
r1, rp = raw("test", 1), pd.concat([raw("test", 2), raw("test", 3)])
ta, tb = texts(S1_IDS[S1[SC]], POOL_IDS[PO[SC]], r1, rp)
del r1, rp
CE = score_pairs(ta, tb, "test" + CE_TAG)
del ta, tb
log(f"CE done; RSS {rss():.1f} GB", TAG)

# cluster features for scored rows (reuses cluster_features on a small frame of the scored rows + their entity tops)
D = pd.DataFrame({"s1": S1[SC].astype(np.int64), "pool": PO[SC].astype(np.int64), "p": P[SC], "is_new": IS_NEW[SC],
                  "scored": np.ones(len(SC), bool)})
# tops must come from ALL existing rows of the entity: add the entity top-3 existing rows (unscored) as context rows
o = np.lexsort((-np.where(IS_NEW == 0, P, -1.0), S1)); ss = S1[o]
st = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]; sz = np.diff(np.r_[st, len(ss)])
ctx_rows = np.concatenate([o[st[sz > k] + k] for k in range(3)])
ctx_rows = np.setdiff1d(ctx_rows, SC)
Dc = pd.DataFrame({"s1": S1[ctx_rows].astype(np.int64), "pool": PO[ctx_rows].astype(np.int64), "p": P[ctx_rows],
                   "is_new": IS_NEW[ctx_rows], "scored": np.zeros(len(ctx_rows), bool)})
del o, ss, st, sz
D = pd.concat([D, Dc], ignore_index=True); del Dc
AF.K_NEW = K
D = AF.cluster_features(D, XP, NT1, "test")
del XP
D = D[D["scored"].values].reset_index(drop=True)
Xf = np.c_[P[SC], CE, RK[SC], p1[S1[SC]], p2[S1[SC]], nb[S1[SC]], p1[S1[SC]] - P[SC], FR_[SC], FC[SC], IS_NEW[SC],
           np.column_stack([D[c].values for c in CL_COLS])].astype(np.float32)
del D
P[SC] = b3.predict(xgb.DMatrix(Xf)); del Xf
log(f"2nd stage applied to {len(SC):,} rows; RSS {rss():.1f} GB", TAG)

# rule: threshold + exclusivity (each pool record to its highest-p S1, ties -> lowest index)
keep = P >= rule["thr"]
mx = np.full(NTP, -np.inf, np.float32); np.maximum.at(mx, PO, P)
idx = np.flatnonzero(P == mx[PO]); _, first = np.unique(PO[idx], return_index=True)
pm = np.zeros(len(P), bool); pm[idx[first]] = True; keep &= pm
del mx, idx, first, pm
log(f"test: {len(P):,} candidates ({int(IS_NEW.sum()):,} new), {int(keep.sum()):,} matches ({keep.sum() / NT1:.3f}/S1)", TAG)

# ---- write (streamed per S1 range) ----
os.makedirs(SUB, exist_ok=True)
s1_ids, pool_ids = S1_IDS, POOL_IDS
order_file = pq.read_table(f"{W}/aligned_inputs/data_parquet/test_source1_final.parquet", columns=["entity_id"]).column(0).to_pylist()
assert order_file == list(s1_ids), "test S1 order"
o = np.lexsort((-P, S1))
S1o, POo, Ko = S1[o], PO[o], keep[o]; del o, P, S1, PO, keep
bnd = np.searchsorted(S1o, np.arange(NT1 + 1))
with open(f"{SUB}/matching_results.tsv.tmp", "w") as fm_, open(f"{SUB}/candidate_pairs.tsv.tmp", "w") as fc_:
    fm_.write("source1_entity_id\tmatched_entity_ids\n"); fc_.write("source1_entity_id\tcandidate_entity_ids\n")
    for i in range(NT1):
        a_, b_ = bnd[i], bnd[i + 1]
        ids = pool_ids[POo[a_:b_]]
        fm_.write(f"{s1_ids[i]}\t{','.join(ids[Ko[a_:b_]])}\n"); fc_.write(f"{s1_ids[i]}\t{','.join(ids)}\n")
for f_ in ("matching_results.tsv", "candidate_pairs.tsv"):
    os.replace(f"{SUB}/{f_}.tmp", f"{SUB}/{f_}")
del S1o, POo, Ko
log(f"files written; RSS {rss():.1f} GB", TAG)

# ---- streaming validation ----
errs, valid_pool, seen = [], set(pool_ids), set()
with open(f"{SUB}/matching_results.tsv") as fm_, open(f"{SUB}/candidate_pairs.tsv") as fc_:
    if fm_.readline().rstrip("\n") != "source1_entity_id\tmatched_entity_ids": errs.append("matching header")
    if fc_.readline().rstrip("\n") != "source1_entity_id\tcandidate_entity_ids": errs.append("candidates header")
    n = 0
    for i, (lm, lc) in enumerate(zip(fm_, fc_)):
        a_, m_ = lm.rstrip("\n").split("\t"); b_, c_ = lc.rstrip("\n").split("\t")
        if a_ != order_file[i] or b_ != order_file[i]:
            errs.append(f"row {i} order"); break
        ms = m_.split(",") if m_ else []; cs = set(c_.split(",")) if c_ else set()
        if not set(ms) <= cs: errs.append(f"row {i}: match not among candidates"); break
        for x in ms:
            if x in seen: errs.append(f"pool id {x} matched twice"); break
            if x not in valid_pool: errs.append(f"unknown pool id {x}"); break
            seen.add(x)
        n += 1
    if n != len(order_file): errs.append(f"row count {n} != {len(order_file)}")
sha = {f_: hashlib.sha256(open(f"{SUB}/{f_}", "rb").read()).hexdigest() for f_ in ("matching_results.tsv", "candidate_pairs.tsv")}
json.dump(dict(variant="adapter + clustering", rule=rule, k_new=K, errors=errs, sha256=sha,
               rep2_result=dec), open(f"{SUB}/submission_info.json", "w"), indent=1)
log(f"submission_3 {'VALIDATION PASSED' if not errs else 'VALIDATION FAILED ' + str(errs)} sha256 {sha}", TAG)
log(f"done in {time.time() - T0:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", TAG)
