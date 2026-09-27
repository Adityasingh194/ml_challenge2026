"""Phase 2: decision rules on saved predictions only (no retraining, no retrieval).
Select on CV OOF -> check on nested holdout -> report on the 80% evaluation entities."""
import glob, json, os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

T0 = time.time()
W = os.path.join(os.environ.get("ER_ROOT", "/home/parth/Desktop/trial work"), "pipeline_run")   # same ER_ROOT as the notebook
C = f"{W}/cache"
CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1_retrieval_rerank", "cells")   # notebook cell sources
FP = "dd3a402b93"
g = {"np": np, "pd": pd}
src = open(f"{CELLS}/022.py").read()
exec(src[src.index("def macro_f05"):src.index("def blocking_ceiling")], g)   # the pipeline's own metric
macro_f05 = g["macro_f05"]

# ---------------- data ----------------
s1f = pq.read_table(f"{C}/frames/{FP}/train_s1_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
pof = pq.read_table(f"{C}/frames/{FP}/train_pool_norm.parquet", columns=["entity_id", "pool_source"]).to_pandas()
gt = pq.read_table(f"{C}/samples/{FP}/gt.parquet").to_pandas()
rows = gt[gt["matched_entity_ids"] != ""]
sid = np.repeat(rows["source1_entity_id"].values, rows["matched_entity_ids"].str.count(",").values + 1)
cid = np.array([x.strip() for s in rows["matched_entity_ids"].values for x in s.split(",")], dtype=object)
a, b = pd.Index(s1f["entity_id"]).get_indexer(sid), pd.Index(pof["entity_id"]).get_indexer(cid)
ok = (a >= 0) & (b >= 0)
N1, NP = len(s1f), len(pof)
n_true = np.bincount(a[ok], minlength=N1).astype(np.int32)
cc = np.where(s1f["country_norm"].values == "India", 0, 1).astype(np.int8)   # 0 India, 1 US
is_s3_pool = pof["pool_source"].values == "S3"
del gt, rows, sid, cid, a, b, pof

rd = glob.glob(f"{C}/retrieval/train_*")[0]
oos = pq.read_table(glob.glob(f"{rd}/model_*/train_entities_oos.parquet")[0]).to_pandas()
lab = np.concatenate([pq.read_table(f, columns=["label"]).column(0).to_numpy() for f in
                      sorted(glob.glob(f"{rd}/trainset_*/block_*.parquet"))]).astype(bool)
oof = np.load(glob.glob(f"{C}/models/train_*/oof_1.npy")[0])
assert len(oos) == len(lab) == len(oof)
tr = dict(s1=oos["s1_pos"].values.astype(np.int64), pool=oos["pool_pos"].values.astype(np.int64),
          p=oos["p"].values.astype(np.float32), y=lab)
cv_row = ~np.isnan(oof)
ho_row = ~cv_row
cv_ent = np.zeros(N1, bool); cv_ent[tr["s1"][cv_row]] = True
ho_ent = np.zeros(N1, bool); ho_ent[tr["s1"][ho_row]] = True
del oos, lab, oof
ev_files = sorted(glob.glob(f"{rd}/eval_*/block_*.parquet"))
E = [pq.read_table(f).to_pandas() for f in ev_files]
ev = dict(s1=np.concatenate([e["s1_pos"].values for e in E]).astype(np.int64),
          pool=np.concatenate([e["pool_pos"].values for e in E]).astype(np.int64),
          p=np.concatenate([e["p"].values for e in E]).astype(np.float32),
          y=np.concatenate([e["y"].values for e in E]).astype(bool))
del E
ev_ent = np.zeros(N1, bool); ev_ent[ev["s1"]] = True
ev_ent &= ~cv_ent & ~ho_ent
print(f"data loaded {time.time() - T0:.0f}s  CV rows {cv_row.sum():,} ({cv_ent.sum():,} ent)  holdout rows {ho_row.sum():,} "
      f"({ho_ent.sum():,} ent)  eval rows {len(ev['p']):,} ({ev_ent.sum():,} ent)", flush=True)


def prep(d):
    """Per-row entity context: rank in entity, top-1 / top-2 p, pool-record best (exclusivity)."""
    s1, p, pool = d["s1"], d["p"], d["pool"]
    o = np.lexsort((-p, s1))
    ss = s1[o]
    st = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    sizes = np.diff(np.r_[st, len(ss)])
    rk = np.empty(len(p), np.int32)
    rk[o] = (np.arange(len(ss)) - np.repeat(st, sizes)).astype(np.int32)
    p1 = np.full(N1, -1.0, np.float32); p2 = np.full(N1, -1.0, np.float32)
    p1[ss[st]] = p[o[st]]
    two = sizes > 1
    p2[ss[st[two]]] = p[o[st[two] + 1]]
    m = np.full(NP, -np.inf, np.float32)
    np.maximum.at(m, pool, p)
    idx = np.flatnonzero(p == m[pool])
    _, first = np.unique(pool[idx], return_index=True)
    pim = np.zeros(len(p), bool); pim[idx[first]] = True
    d.update(rk=rk, p1=p1[s1], p2=p2[s1], pim=pim, cc=cc[s1], s3=is_s3_pool[pool])
    return d


def keep_mask(d, r):
    p = d["p"]
    t = np.where(d["cc"] == 0, r.get("t_in", r["t"]), r.get("t_us", r["t"])) if ("t_in" in r or "t_us" in r) else r["t"]
    k = p >= t
    if "t_top1" in r:          # rescue: always keep the entity's best candidate if it clears a lower bar
        k |= (d["rk"] == 0) & (p >= r["t_top1"])
    if "rel" in r:             # secondary candidates must be within a factor of the entity's best
        k &= (d["rk"] == 0) | (p >= r["rel"] * d["p1"])
    if "guard" in r:           # predict nothing for an entity whose best candidate is below the guard
        k &= d["p1"] >= r["guard"]
    if "margin" in r:          # when the best is far ahead of the 2nd, keep only the best unless 2nd is very sure
        far = (d["p1"] - d["p2"]) > r["margin"]
        k &= ~far | (d["rk"] == 0) | (p >= r.get("t_hi", 0.95))
    if r.get("excl"):
        k &= d["pim"]
    return k


def score(d, r, ent, full=False):
    k = keep_mask(d, r)
    m = ent[d["s1"]]
    f = macro_f05(d["s1"][k], d["y"][k], n_true, ent)
    if not full:
        return f
    tp = int((k & d["y"] & m).sum()); fp = int((k & ~d["y"] & m).sum()); fn = int(n_true[ent].sum()) - tp
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(f05=round(f, 5), precision=round(P, 5), recall=round(R, 5), micro_f05=round(1.25 * P * R / max(0.25 * P + R, 1e-12), 5),
                n_pred=int((k & m).sum()), fp=fp, fn=fn,
                india_f05=round(macro_f05(d["s1"][k], d["y"][k], n_true, ent & (cc == 0)), 5),
                us_f05=round(macro_f05(d["s1"][k], d["y"][k], n_true, ent & (cc == 1)), 5))


CV = prep({k: v[cv_row] for k, v in tr.items()})
HO = prep({k: v[ho_row] for k, v in tr.items()})
print(f"prepared CV/holdout {time.time() - T0:.0f}s", flush=True)
TG = np.round(np.r_[np.arange(0.30, 0.90, 0.02), np.arange(0.90, 0.99, 0.01)], 3)
cv = lambda r: score(CV, r, cv_ent)
base = dict(t=0.66)
b0 = cv(base)
found = {"baseline global 0.66": base}


def best_over(make, grid):
    best = max(((make(x), cv(make(x))) for x in grid), key=lambda z: z[1])
    return best


# 1 global threshold (re-searched on the finer grid)
found["1 global"] = best_over(lambda t: dict(t=float(t)), TG)[0]
tg = found["1 global"]["t"]
# 2/3 per-country thresholds (coordinate search, 2 passes)
r = dict(t=tg, t_in=tg, t_us=tg)
for _ in range(2):
    for key in ("t_in", "t_us"):
        r = best_over(lambda t: dict(r, **{key: float(t)}), TG)[0]
found["2 per-country"] = r
# 4 per-source
r = dict(t=tg)
def ps(ts2, ts3): return ts2, ts3
best_ps = None
for t2 in TG[::2]:
    for t3 in TG[::2]:
        rr = dict(t=tg, _ps=(float(t2), float(t3)))
        k = CV["p"] >= np.where(CV["s3"], t3, t2)
        f = macro_f05(CV["s1"][k], CV["y"][k], n_true, cv_ent)
        if best_ps is None or f > best_ps[1]:
            best_ps = (rr, f)
found["4 per-source"] = best_ps[0]
# 5 no-match guard, 6 margin, 7 top-1 rescue, relative, exclusivity
found["5 no-match guard"] = best_over(lambda g_: dict(t=tg, guard=float(g_)), TG[TG >= tg])[0]
found["6 margin top1-top2"] = max(((dict(t=tg, margin=m_, t_hi=th), cv(dict(t=tg, margin=m_, t_hi=th)))
                                  for m_ in (0.3, 0.5, 0.7, 0.9) for th in (0.8, 0.9, 0.95, 0.98)), key=lambda z: z[1])[0]
found["7a top-1 rescue"] = best_over(lambda t1: dict(t=tg, t_top1=float(t1)), np.round(np.arange(0.05, tg, 0.02), 3))[0]
found["7b relative to best"] = best_over(lambda x: dict(t=tg, rel=float(x)), (0.3, 0.5, 0.6, 0.7, 0.8, 0.9))[0]
found["7c exclusive"] = dict(t=tg, excl=True)
print(f"single-rule search done {time.time() - T0:.0f}s", flush=True)


def cv_any(r):
    if "_ps" in r:
        t2, t3 = r["_ps"]
        k = CV["p"] >= np.where(CV["s3"], t3, t2)
        return macro_f05(CV["s1"][k], CV["y"][k], n_true, cv_ent)
    return cv(r)


res = {name: cv_any(r) for name, r in found.items()}
# combination: greedy, adding components that help on CV
combo = dict(found["2 per-country"])
for name in ("7a top-1 rescue", "5 no-match guard", "7b relative to best", "6 margin top1-top2", "7c exclusive"):
    extra = {k: v for k, v in found[name].items() if k != "t"}
    if name == "7a top-1 rescue":   # re-tune the rescue bar under the per-country thresholds
        cand = max(((dict(combo, t_top1=float(t1)), cv(dict(combo, t_top1=float(t1)))) for t1 in np.round(np.arange(0.05, 0.7, 0.02), 3)), key=lambda z: z[1])
    else:
        cand = (dict(combo, **extra), cv(dict(combo, **extra)))
    if cand[1] > cv(combo) + 1e-5:
        combo = cand[0]
# re-tune per-country thresholds under the final combination
for _ in range(2):
    for key in ("t_in", "t_us"):
        combo = best_over(lambda t: dict(combo, **{key: float(t)}), TG)[0]
found["COMBINED (greedy on CV)"] = combo
res["COMBINED (greedy on CV)"] = cv(combo)
print(f"combination done {time.time() - T0:.0f}s", flush=True)

# ---------------- report: CV -> holdout -> 80% eval ----------------
EV = prep({k: np.r_[ev[k], tr[k]] for k in ("s1", "pool", "p", "y")})   # all train S1 compete for exclusivity
del ev
out = []
for name, r in found.items():
    if "_ps" in r:
        continue
    h = score(HO, r, ho_ent, full=True); e = score(EV, r, ev_ent, full=True)
    out.append(dict(rule=name, params=json.dumps({k: v for k, v in r.items()}), cv_f05=round(res[name], 5),
                    ho_f05=h["f05"], ev_f05=e["f05"], ev_delta=round(e["f05"] - 0.94826, 5), ev_precision=e["precision"],
                    ev_recall=e["recall"], ev_micro_f05=e["micro_f05"], ev_n_pred=e["n_pred"], ev_fp=e["fp"], ev_fn=e["fn"],
                    ev_india=e["india_f05"], ev_us=e["us_f05"], ho_india=h["india_f05"], ho_us=h["us_f05"]))
t2, t3 = found["4 per-source"]["_ps"]
for nm, D, ent in (("ho", HO, ho_ent), ("ev", EV, ev_ent)):
    k = D["p"] >= np.where(D["s3"], t3, t2)
    res[f"4ps_{nm}"] = macro_f05(D["s1"][k], D["y"][k], n_true, ent)
out.append(dict(rule="4 per-source", params=json.dumps({"thr_s2": t2, "thr_s3": t3}), cv_f05=round(res["4 per-source"], 5),
                ho_f05=round(res["4ps_ho"], 5), ev_f05=round(res["4ps_ev"], 5), ev_delta=round(res["4ps_ev"] - 0.94826, 5)))
T = pd.DataFrame(out)
pd.set_option("display.width", 250)
print(T[["rule", "cv_f05", "ho_f05", "ev_f05", "ev_delta", "ev_precision", "ev_recall", "ev_n_pred", "ev_fp", "ev_fn", "ev_india", "ev_us"]].to_string(index=False))
print("\nparams:\n" + T[["rule", "params"]].to_string(index=False))
T.to_csv(f"{W}/experiments/phase2_rules.tsv", sep="\t", index=False)
import resource
print(f"\nruntime {time.time() - T0:.0f}s  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
