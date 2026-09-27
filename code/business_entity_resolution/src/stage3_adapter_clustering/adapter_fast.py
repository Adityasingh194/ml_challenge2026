"""Adapter FAST integration (no stage-1 retrain):
  new candidates = fused top-K (adapter) pairs NOT already in the 120-candidate union
  -> scored by the existing cross-encoder -> 2nd stage v3 [stage-1 p, CE, entity ctx, fused rank/cos, is_new]
  entities: adapter-train (25% eval 'fit') are EXCLUDED; the remaining report entities are hash-split into
  FIT2 (2nd stage + rule) and REP2 (reported, never used); baseline = submission_2 pipeline on the same REP2 entities.
  test -> output/submission_3 (submission_1/2 untouched). Logs to experiments.log; checkpoints in experiments/adapter_fast/."""
import glob, hashlib, json, os, sys, threading, time, math, resource
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F

T0 = time.time()
W = os.path.join(os.environ.get("ER_ROOT", "/home/parth/Desktop/trial work"), "pipeline_run")   # same ER_ROOT as the notebook
C, A = f"{W}/cache", f"{W}/aligned_inputs/emb_aligned"
OUT = f"{W}/experiments/adapter_fast"; os.makedirs(OUT, exist_ok=True)
SUB = f"{W}/output/submission_3"
FP = "dd3a402b93"
K_NEW = int(os.environ.get("K_NEW", "10"))        # fused top-K considered for new candidates
N_FIT2, N_REP2 = 250_000, 250_000
BAND = (0.05, 0.95)
CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1_retrieval_rerank", "cells")   # notebook cell sources
g = {"np": np, "pd": pd}
src = open(f"{CELLS}/022.py").read(); exec(src[src.index("def macro_f05"):src.index("def blocking_ceiling")], g)
macro_f05 = g["macro_f05"]
LOGF = f"{W}/experiments/experiments.log"
inb = lambda p: (p >= BAND[0]) & (p <= BAND[1])


os.makedirs(f"{W}/experiments/logs", exist_ok=True)


def log(m, tag="adapter_fast"):
    """shared experiments.log + a separate per-component log (logs/<tag>.log)"""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{tag} +{time.time() - T0:5.0f}s] {m}"
    print(line, flush=True); open(LOGF, "a").write(line + "\n"); open(f"{W}/experiments/logs/{tag}.log", "a").write(line + "\n")


def rss():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 1e6


# ---------------- adapter model (same as adapter_pilot.py) ----------------
DIM = 256
class Tower(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Linear(1024, DIM))
        s.skip = nn.Linear(768, DIM, bias=False)
    def forward(s, x8):
        x = x8.float() / 127.0
        return F.normalize(s.net(x) + s.skip(x), dim=-1)
class Adapter(nn.Module):
    def __init__(s):
        super().__init__(); s.fq, s.fd = Tower(), Tower()
ad = Adapter(); ad.load_state_dict(torch.load(f"{W}/experiments/adapter/adapter.pt", map_location="cpu")); ad.eval()


def emb(split, s):
    return np.concatenate([np.load(f"{A}/{split}_s{s}_name.npy"), np.load(f"{A}/{split}_s{s}_addr.npy")], axis=1)


@torch.no_grad()
def project(tower, X, dev, chunk=500_000):
    tower = tower.cpu().to(dev)   # via host: broken P2P
    return torch.cat([tower(torch.from_numpy(np.ascontiguousarray(X[s:s + chunk])).to(dev)).half() for s in range(0, len(X), chunk)])


@torch.no_grad()
def fused_search(X1, XP, q_rows, cc_q, cc_p, K, name):
    """top-K pool rows per query by adapter cosine, per country; GPU1 then GPU0 alternate by country. Checkpointed."""
    path = f"{OUT}/fused_{name}_top{K}.npz"
    if os.path.exists(path):
        z = np.load(path); log(f"fused search {name}: cached", "fused"); return z["ids"], z["sc"]
    ids = np.full((len(q_rows), K), -1, np.int64); scs = np.zeros((len(q_rows), K), np.float32)
    t = time.time()
    for ci, c in enumerate(sorted(set(cc_q[q_rows]))):
        dev = f"cuda:{ci % 2}"
        qi = np.flatnonzero(cc_q[q_rows] == c); pi = np.flatnonzero(cc_p == c)
        Pc = project(ad.fd, XP[pi], dev); Qc = project(ad.fq, X1[q_rows[qi]], dev)
        for s in range(0, len(qi), 4096):
            bs = torch.full((min(4096, len(qi) - s), K), -1e4, device=dev); bi = torch.full_like(bs, -1, dtype=torch.int64)
            for p0 in range(0, len(pi), 400_000):
                v, ix = (Qc[s:s + 4096] @ Pc[p0:p0 + 400_000].T).float().topk(K, dim=1)
                cs, ci_ = torch.cat([bs, v], 1), torch.cat([bi, ix + p0], 1)
                bs, o = cs.topk(K, dim=1); bi = ci_.gather(1, o)
            ids[qi[s:s + 4096]] = pi[bi.cpu().numpy()]; scs[qi[s:s + 4096]] = bs.cpu().numpy()
        del Pc, Qc; torch.cuda.empty_cache()
        log(f"  fused search {name} {c}: {len(qi):,} queries vs {len(pi):,} pool  ({time.time() - t:.0f}s, GPU peak {torch.cuda.max_memory_allocated(ci % 2) / 1e9:.1f} GB, RSS {rss():.1f} GB)", "fused")
    np.savez(path, ids=ids, sc=scs)
    return ids, scs


def fused_search_prefix(X1, XP, q_rows, cc_q, cc_p, K, name):
    """top-K whose first 10 columns are EXACTLY the cached top-10 (so CE scores of the K=10 run stay aligned);
    columns 11..K keep only ids not already in the row."""
    ids, scs = fused_search(X1, XP, q_rows, cc_q, cc_p, K, name)
    p10 = f"{OUT}/fused_{name}_top10.npz"
    if K <= 10 or not os.path.exists(p10):
        return ids, scs
    z = np.load(p10); i10, s10 = z["ids"], z["sc"]
    same = (ids[:, :10] == i10).all(1).mean()
    ext_i, ext_s = ids[:, 10:].copy(), scs[:, 10:].copy()
    dup = (ext_i[:, :, None] == i10[:, None, :]).any(2)
    ext_i[dup] = -1
    log(f"fused {name}: first-10 agreement with cached top-10 {same:.5f}; {int(dup.sum()):,} duplicate extras dropped", "fused")
    return np.c_[i10, ext_i], np.c_[s10, ext_s]


# ---------------- cross-encoder scoring (same as ce_full.py; both GPUs, 50k chunks, resumable) ----------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MDIR = os.environ.get("CE_MDIR", f"{W}/experiments/ce_full/model")
CE_TAG = os.environ.get("CE_TAG", "")   # e.g. "_ce2": separate CE cache / stage2 / decision files
tok = AutoTokenizer.from_pretrained(MDIR)


def score_pairs(ta, tb, name, CH=50_000):
    path = f"{OUT}/ce_{name}.npy"
    n = len(ta); out = np.empty(n, np.float32)
    if os.path.exists(path) and len(np.load(path, mmap_mode="r")) == n:
        log(f"CE {name}: cached", "ce_new"); return np.load(path)
    cdir = f"{OUT}/ce_{name}_chunks"; os.makedirs(cdir, exist_ok=True)
    chunks = [(i, min(i + CH, n)) for i in range(0, n, CH)]
    todo = [c for c in chunks if not (os.path.exists(f"{cdir}/{c[0]:010d}.npy")
                                      and len(np.load(f"{cdir}/{c[0]:010d}.npy", mmap_mode="r")) == c[1] - c[0])]
    log(f"CE {name}: {n:,} pairs, {len(chunks)} chunks ({len(chunks) - len(todo)} done)", "ce_new")
    lock, t, cnt = threading.Lock(), time.time(), [0]

    def work(gi, mine):
        dev = f"cuda:{gi}"; m = AutoModelForSequenceClassification.from_pretrained(MDIR).to(dev).eval()
        for lo, hi in mine:
            buf = np.empty(hi - lo, np.float32)
            order = np.argsort([len(ta[i]) + len(tb[i]) for i in range(lo, hi)], kind="stable")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for k in range(0, hi - lo, 512):
                    ix = order[k:k + 512] + lo
                    enc = tok([ta[i] for i in ix], [tb[i] for i in ix], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
                    buf[ix - lo] = m(**enc).logits.squeeze(-1).float().cpu().numpy()
            np.save(f"{cdir}/{lo:010d}.tmp.npy", buf); os.replace(f"{cdir}/{lo:010d}.tmp.npy", f"{cdir}/{lo:010d}.npy")
            with lock:
                cnt[0] += 1
                if cnt[0] % 10 == 0 or cnt[0] == len(todo):
                    el = time.time() - t
                    log(f"  CE {name}: {cnt[0]}/{len(todo)} chunks, {cnt[0] * CH / el:,.0f} pairs/s, ETA {el / cnt[0] * (len(todo) - cnt[0]) / 60:.1f} min, GPU peaks {[round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(2)]} GB, RSS {rss():.1f} GB", "ce_new")
        del m; torch.cuda.empty_cache()

    ths = [threading.Thread(target=work, args=(gi, todo[gi::2])) for gi in range(2)]
    [th.start() for th in ths]; [th.join() for th in ths]
    for lo, hi in chunks:
        out[lo:hi] = np.load(f"{cdir}/{lo:010d}.npy")
    np.save(path, out); return out


def raw(split, s):
    return pq.read_table(f"{W}/aligned_inputs/data_parquet/{split}_source{s}_final.parquet",
                         columns=["entity_id", "business_name", "business_address"]).to_pandas().set_index("entity_id")


def texts(s1_ids, pool_ids, r1, rp):
    a = r1.reindex(s1_ids); b = rp.reindex(pool_ids)
    return ((a["business_name"].fillna("") + " | " + a["business_address"].fillna("")).tolist(),
            (b["business_name"].fillna("") + " | " + b["business_address"].fillna("")).tolist())


def entity_ctx(D, n_s1):
    s1, p = D["s1"].values, D["p"].values
    o = np.lexsort((-p, s1)); ss = s1[o]
    st = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]; sz = np.diff(np.r_[st, len(ss)])
    rk = np.empty(len(p), np.int32); rk[o] = np.arange(len(ss)) - np.repeat(st, sz)
    p1 = np.zeros(n_s1, np.float32); p1[ss[st]] = p[o[st]]
    p2 = np.zeros(n_s1, np.float32); two = sz > 1; p2[ss[st[two]]] = p[o[st[two] + 1]]
    nb = np.bincount(s1[inb(p)], minlength=n_s1).astype(np.float32)
    D["rk"], D["p1"], D["p2"], D["n_band"] = rk, p1[s1], p2[s1], nb[s1]
    return D


@torch.no_grad()
def pair_cos(X1, XP, s1, pool, dev="cuda:0", ch=250_000):
    out = np.empty(len(s1), np.float32)
    for i in range(0, len(s1), ch):
        fq_, fd_ = ad.fq.cpu().to(dev), ad.fd.cpu().to(dev)   # via host (broken P2P)
        q = fq_(torch.from_numpy(X1[s1[i:i + ch]]).to(dev)); d = fd_(torch.from_numpy(XP[pool[i:i + ch]]).to(dev))
        out[i:i + ch] = (q * d).sum(1).float().cpu().numpy()
    return out


CL_COLS = ["cl_cos_top1", "cl_cos_max3", "cl_cos_wmax3", "cl_name_top1", "cl_addr_top1", "cl_is_top3", "cl_n_near"]


@torch.no_grad()
def cluster_features(D, XP, n_s1, name, dev="cuda:0", CH=250_000):
    """Entity-level context: similarity of each scored candidate to the entity's top-3 confident candidates (by stage-1 p;
    pool<->pool, never the candidate's own label). Adapter cosine, raw name / address cosine, p-weighted max, and how many
    of the top-3 are near-duplicates of it. Batched on GPU, checkpointed per batch."""
    path = f"{OUT}/cluster_{name}_k{K_NEW}.npz"
    sc = np.flatnonzero(D["scored"].values)
    if os.path.exists(path):
        z = np.load(path); log(f"cluster {name}: cached", "cluster"); F_ = z["F"]
    else:
        ex = D["is_new"].values == 0
        s1, p, pool = D["s1"].values, D["p"].values, D["pool"].values
        o = np.lexsort((-np.where(ex, p, -1.0), s1)); ss = s1[o]
        st = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]; sz = np.diff(np.r_[st, len(ss)])
        top = np.full((n_s1, 3), -1, np.int64); topp = np.zeros((n_s1, 3), np.float32)
        for j in range(3):
            okj = sz > j
            top[ss[st[okj]], j] = pool[o[st[okj] + j]]; topp[ss[st[okj]], j] = np.where(ex[o[st[okj] + j]], p[o[st[okj] + j]], 0)
        F_ = np.zeros((len(sc), len(CL_COLS)), np.float32)
        cdir = f"{OUT}/cluster_{name}_k{K_NEW}_chunks"; os.makedirs(cdir, exist_ok=True)
        fd = ad.fd.cpu().to(dev); t = time.time()   # via host (broken P2P)
        for bi, i in enumerate(range(0, len(sc), CH)):
            cp = f"{cdir}/{i:010d}.npy"
            if os.path.exists(cp):
                F_[i:i + CH] = np.load(cp); continue
            r = sc[i:i + CH]; cand = pool[r]; T = top[s1[r]]; TP = torch.from_numpy(topp[s1[r]]).to(dev)
            xc = torch.from_numpy(XP[cand]).to(dev)
            vc = fd(xc)
            cos = torch.full((len(r), 3), float("nan"), device=dev)
            nm = torch.full((len(r),), float("nan"), device=dev); adr = torch.full_like(nm, float("nan"))
            for j in range(3):
                okj = (T[:, j] >= 0) & (T[:, j] != cand)          # skip missing and the candidate itself
                if okj.any():
                    xt = torch.from_numpy(XP[T[okj, j]]).to(dev)
                    ok_t = torch.from_numpy(okj).to(dev)
                    cos[ok_t, j] = (vc[ok_t] * fd(xt)).sum(1)
                    if j == 0:
                        a_, b_ = xc[ok_t].float() / 127, xt.float() / 127
                        nm[ok_t] = (a_[:, :256] * b_[:, :256]).sum(1); adr[ok_t] = (a_[:, 256:] * b_[:, 256:]).sum(1)
            c0 = torch.nan_to_num(cos, nan=-1.0)
            f = torch.stack([cos[:, 0], c0.max(1).values, (c0 * TP).max(1).values, nm, adr,
                             torch.from_numpy((T == cand[:, None]).any(1)).to(dev).float(), (c0 > 0.9).sum(1).float()], 1)
            F_[i:i + CH] = f.cpu().numpy(); np.save(cp, F_[i:i + CH])
            del xc, vc, cos, nm, adr, c0, f, TP; torch.cuda.empty_cache()
            if bi % 10 == 0 or i + CH >= len(sc): log(f"  cluster {name}: batch {bi + 1}/{math.ceil(len(sc) / CH)} ({min(i + CH, len(sc)):,}/{len(sc):,} rows, {time.time() - t:.0f}s, "
                f"GPU peak {torch.cuda.max_memory_allocated(0) / 1e9:.1f} GB, RSS {rss():.1f} GB)", "cluster")
        np.savez(path, F=F_)
        log(f"cluster {name}: {len(sc):,} rows in {time.time() - t:.0f}s", "cluster")
    for k, c in enumerate(CL_COLS):
        col = np.full(len(D), np.nan, np.float32); col[sc] = F_[:, k]; D[c] = col
    return D


def build_rows(D_exist, fid, fsc, q_rows, NP, n_s1, X1, XP, r1, rp, s1_ids, pool_ids, name, ce_band_cached=None):
    """D_exist: existing candidates of q_rows entities (s1,pool,p[,y]); returns all rows incl. new fused-only pairs with features."""
    K = fid.shape[1]
    fk = (np.repeat(np.asarray(q_rows, np.int64), K) * np.int64(NP) + fid.reshape(-1)); frank = np.tile(np.arange(K), len(q_rows)); fcos = fsc.reshape(-1)
    v = fid.reshape(-1) >= 0; fk, frank, fcos = fk[v], frank[v], fcos[v]
    ek = D_exist["s1"].values * NP + D_exist["pool"].values
    pos = pd.Index(fk)
    D_exist["frank"] = np.int16(99); D_exist["fcos"] = np.float32(np.nan); D_exist["is_new"] = np.int8(0)
    j = pos.get_indexer(ek); m = j >= 0
    D_exist.loc[m, "frank"] = frank[j[m]].astype(np.int16); D_exist.loc[m, "fcos"] = fcos[j[m]]
    newm = np.flatnonzero(~np.isin(fk, ek))
    newm = newm[np.argsort(frank[newm] >= 10, kind="stable")]   # K<=10 pairs first (same order as the K=10 run) -> CE scores reusable
    N = pd.DataFrame({"s1": fk[newm] // NP, "pool": fk[newm] % NP, "p": np.float32(0.0), "frank": frank[newm].astype(np.int16),
                      "fcos": fcos[newm], "is_new": np.int8(1)})
    D = pd.concat([D_exist, N], ignore_index=True)
    D = entity_ctx(D, n_s1)
    # fused cosine for existing band pairs not in fused top-K
    need = inb(D["p"].values) & np.isnan(D["fcos"].values)
    D.loc[need, "fcos"] = pair_cos(X1, XP, D["s1"].values[need], D["pool"].values[need])
    # CE: band pairs of existing candidates + all new pairs
    sc = inb(D["p"].values) | (D["is_new"].values == 1)
    ta, tb = texts(s1_ids[D["s1"].values[sc]], pool_ids[D["pool"].values[sc]], r1, rp)
    ce = np.full(len(D), np.nan, np.float32); ce[sc] = score_pairs(ta, tb, name + CE_TAG)
    D["ce"] = ce; D["scored"] = sc
    torch.cuda.empty_cache()
    D = cluster_features(D, XP, n_s1, name)
    log(f"{name}: {len(D_exist):,} existing + {len(N):,} new fused-only pairs ({len(N) / max(len(q_rows), 1):.2f} per S1); CE-scored {sc.sum():,}; RSS {rss():.1f} GB")
    return D


def fmat3(D, cluster=True):
    base = [D["p"].values, D["ce"].values, D["rk"].values, D["p1"].values, D["p2"].values, D["n_band"].values,
            D["p1"].values - D["p"].values, D["frank"].values, D["fcos"].values, D["is_new"].values]
    return np.c_[tuple(base + ([D[c].values for c in CL_COLS] if cluster else []))].astype(np.float32)


def fmat1(D):   # the submission_2 2nd-stage inputs
    return np.c_[D["p"].values, D["ce"].values, D["rk"].values, D["p1"].values, D["p2"].values, D["n_band"].values,
                 D["p1"].values - D["p"].values].astype(np.float32)


import xgboost as xgb
if __name__ == "__main__":
    # ================= TRAIN side =================
    s1f = pq.read_table(f"{C}/frames/{FP}/train_s1_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
    pof = pq.read_table(f"{C}/frames/{FP}/train_pool_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
    N1, NP = len(s1f), len(pof)
    gt = pq.read_table(f"{C}/samples/{FP}/gt.parquet").to_pandas(); rows = gt[gt["matched_entity_ids"] != ""]
    sid = np.repeat(rows["source1_entity_id"].values, rows["matched_entity_ids"].str.count(",").values + 1)
    cid = np.array([x.strip() for s in rows["matched_entity_ids"].values for x in s.split(",")], dtype=object)
    a, b = pd.Index(s1f["entity_id"]).get_indexer(sid), pd.Index(pof["entity_id"]).get_indexer(cid)
    ok = (a >= 0) & (b >= 0); true_keys = np.unique(a[ok].astype(np.int64) * NP + b[ok])
    n_true = np.bincount(a[ok], minlength=N1).astype(np.int32); cc = np.where(s1f["country_norm"].values == "India", 0, 1)
    del gt, rows, sid, cid, a, b
    rd = glob.glob(f"{C}/retrieval/train_*")[0]
    EV = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{rd}/eval_*/block_*.parquet"))], ignore_index=True)
    ev_ids = np.unique(EV["s1_pos"].values).astype(np.int64)
    h = np.frombuffer(hashlib.sha256(b"ce_full_fit_split").digest()[:8], np.uint64)[0]
    u = ((ev_ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + h) >> np.uint64(11)).astype(np.float64) / 2 ** 53
    rep = ev_ids[u >= 0.25]                                   # adapter never saw these
    rng = np.random.default_rng(3); pick = rng.choice(rep, N_FIT2 + N_REP2, replace=False)
    fit2 = np.sort(pick[:N_FIT2]); rep2 = np.sort(pick[N_FIT2:])
    q_rows = np.sort(pick)
    fit2_ent = np.zeros(N1, bool); fit2_ent[fit2] = True; rep2_ent = np.zeros(N1, bool); rep2_ent[rep2] = True
    ev_ce_all = np.load(f"{W}/experiments/ce_full/ce_eval_band.npy")        # submission_2 CE scores (eval band, EV order)
    evb = inb(EV["p"].values)
    ce_ev = np.full(len(EV), np.nan, np.float32); ce_ev[evb] = ev_ce_all
    sel = np.isin(EV["s1_pos"].values, q_rows)
    DE = pd.DataFrame({"s1": EV["s1_pos"].values[sel].astype(np.int64), "pool": EV["pool_pos"].values[sel].astype(np.int64),
                       "p": EV["p"].values[sel].astype(np.float32), "ce_old": ce_ev[sel]})
    del EV, ce_ev
    log(f"train: fit2 {len(fit2):,} / rep2 {len(rep2):,} entities (disjoint from adapter training), existing rows {len(DE):,}; RSS {rss():.1f} GB")
    X1 = emb("train", 1); XP = np.concatenate([emb("train", 2), emb("train", 3)], axis=0)
    fid, fsc = fused_search_prefix(X1, XP, q_rows, s1f["country_norm"].values, pof["country_norm"].values, K_NEW, "train_sample")
    r1, rp = raw("train", 1), pd.concat([raw("train", 2), raw("train", 3)])
    D = build_rows(DE, fid, fsc, q_rows, NP, N1, X1, XP, r1, rp, s1f["entity_id"].values, pof["entity_id"].values, "train_sample")
    D["y"] = np.isin(D["s1"].values * NP + D["pool"].values, true_keys)
    del X1, XP, r1, rp
    # recall of the candidate set with the new pairs (report entities)
    m2 = rep2_ent[D["s1"].values]
    rec_old = D["y"].values[m2 & (D["is_new"].values == 0)].sum() / n_true[rep2_ent].sum()
    rec_new = D["y"].values[m2].sum() / n_true[rep2_ent].sum()
    log(f"REP2 candidate recall: existing {rec_old:.4f} -> with fused top-{K_NEW} new pairs {rec_new:.4f}")
    # 2nd stage v1 (submission_2, loaded) vs v3 (new, fitted on FIT2)
    b1 = xgb.Booster(); b1.load_model(f"{W}/experiments/ce_full/stage2.json")
    old = D["is_new"].values == 0
    p_v1 = D["p"].values.copy(); mb = old & inb(D["p"].values)
    Dv1 = D[mb].copy(); Dv1["ce"] = Dv1["ce_old"]
    p_v1[mb] = b1.predict(xgb.DMatrix(fmat1(Dv1)))
    sc = D["scored"].values
    fm = sc & fit2_ent[D["s1"].values]
    prm = dict(objective="binary:logistic", max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9,
               min_child_weight=5, tree_method="hist", device="cuda:0", eval_metric="logloss", seed=0)
    models, preds = {}, {}
    for variant, cl in (("adapter only", False), ("adapter + clustering", True)):
        mp = f"{OUT}/stage2_{variant.replace(' ', '_').replace('+', 'plus')}_k{K_NEW}{CE_TAG}.json"
        if os.path.exists(mp):
            bst = xgb.Booster(); bst.load_model(mp)
        else:
            bst = xgb.train(prm, xgb.DMatrix(fmat3(D[fm], cl), label=D["y"].values[fm]), 800); bst.save_model(mp)
        pv = D["p"].values.copy(); pv[sc] = bst.predict(xgb.DMatrix(fmat3(D[sc], cl)))
        models[variant], preds[variant] = (bst, cl), pv
        log(f"2nd stage '{variant}' fitted on {fm.sum():,} FIT2 rows")
    TG = np.round(np.r_[np.arange(0.30, 0.90, 0.02), np.arange(0.90, 0.995, 0.005)], 3)

    def pim(pool, p):
        mx = np.full(NP, -np.inf, np.float32); np.maximum.at(mx, pool, p)
        idx = np.flatnonzero(p == mx[pool]); _, first = np.unique(pool[idx], return_index=True)
        v = np.zeros(len(p), bool); v[idx[first]] = True; return v

    def ev_(p, t, excl, ent):
        k = (p >= t) & (pim(D["pool"].values, p) if excl else True)
        s1, y = D["s1"].values, D["y"].values; m = ent[s1]
        tp = int((k & y & m).sum()); fp = int((k & ~y & m).sum()); fn = int(n_true[ent].sum()) - tp
        return dict(f05=round(macro_f05(s1[k], y[k], n_true, ent), 5), precision=round(tp / max(tp + fp, 1), 5),
                    recall=round(tp / max(tp + fn, 1), 5), fp=fp, fn=fn,
                    india=round(macro_f05(s1[k], y[k], n_true, ent & (cc == 0)), 5), us=round(macro_f05(s1[k], y[k], n_true, ent & (cc == 1)), 5))

    res, rules = {}, {}
    for name, p in (("submission_2 pipeline (v1)", p_v1), ("adapter only", preds["adapter only"]),
                    ("adapter + clustering", preds["adapter + clustering"])):
        best = max(((t, e) for t in TG for e in (True,)), key=lambda te: ev_(p, te[0], te[1], fit2_ent)["f05"])
        rules[name] = dict(thr=float(best[0]), exclusive=True)
        res[name] = dict(rule=str(rules[name]), **ev_(p, best[0], True, rep2_ent))
    R = pd.DataFrame(res).T
    log(f"[K={K_NEW} CE={MDIR}] RESULT on REP2 ({len(rep2):,} held-out entities, rule chosen on FIT2):\n" + R.to_string())
    base_f = res["submission_2 pipeline (v1)"]["f05"]
    for v in ("adapter only", "adapter + clustering"):
        log(f"delta on REP2 vs submission_2 pipeline: {v}: {res[v]['f05'] - base_f:+.5f}")
    best_v = max(("adapter only", "adapter + clustering"), key=lambda v: res[v]["f05"])
    dlt = res[best_v]["f05"] - base_f
    log(f"best variant: {best_v} ({dlt:+.5f})")
    b3, use_cl = models[best_v]
    with open(f"{W}/experiments/experiments.tsv", "a") as f:
        for v in ("adapter only", "adapter + clustering"):
            r_ = res[v]
            f.write(f"{v.replace(' ', '_')}\trep2_250k\t{r_['f05']}\t{round(r_['f05'] - base_f, 5)}\t{r_['precision']}\t{r_['recall']}\t{r_['india']}\t{r_['us']}\t\t{r_['fp']}\t{r_['fn']}\t{rec_new:.4f}\t{time.time() - T0:.0f}\t\t\tbaseline(sub2 pipeline, same entities)={base_f} K_NEW={K_NEW} CE={MDIR}\n")
    R.to_csv(f"{OUT}/result_k{K_NEW}{CE_TAG}.tsv", sep="\t")
    json.dump(dict(rules=rules, delta=dlt, k_new=K_NEW, rep2=res), open(f"{OUT}/decision_k{K_NEW}{CE_TAG}.json", "w"), indent=1)
    if os.environ.get("SKIP_TEST") == "1":
        log("SKIP_TEST=1: train-side measurement done; test runs in finish_test_lean.py"); sys.exit(0)
    if dlt <= 0.001 and os.environ.get("FORCE_TEST") != "1":
        log("no measured gain >= 0.001 -> NOT producing submission_3"); sys.exit(0)
    del D
    # ================= TEST -> submission_3 =================
    rule = rules[best_v]
    tdir = glob.glob(f"{C}/retrieval/test_*")[0]
    TE = pq.read_table(glob.glob(f"{tdir}/scores_*/final_preds.parquet")[0], columns=["s1_pos", "pool_pos", "p"]).to_pandas()
    ce_te_all = np.load(f"{W}/experiments/ce_full/ce_test_band.npy")
    TE = pd.DataFrame({"s1": TE["s1_pos"].values.astype(np.int64), "pool": TE["pool_pos"].values.astype(np.int64), "p": TE["p"].values.astype(np.float32)})
    fr = glob.glob(f"{W}/scratch/frames/*")[0]
    ts1 = pq.read_table(f"{fr}/test_s1_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
    tpo = pq.read_table(f"{fr}/test_pool_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
    NT1, NTP = len(ts1), len(tpo)
    X1 = emb("test", 1); XP = np.concatenate([emb("test", 2), emb("test", 3)], axis=0)
    assert len(X1) == NT1 and len(XP) == NTP
    fid, fsc = fused_search(X1, XP, np.arange(NT1), ts1["country_norm"].values, tpo["country_norm"].values, K_NEW, "test")
    r1, rp = raw("test", 1), pd.concat([raw("test", 2), raw("test", 3)])
    # existing band pairs already have CE scores (submission_2): reuse them, score only NEW pairs
    TE["ce_old"] = np.nan; TE.loc[inb(TE["p"].values), "ce_old"] = ce_te_all
    D = build_rows(TE, fid, fsc, np.arange(NT1), NTP, NT1, X1, XP, r1, rp, ts1["entity_id"].values, tpo["entity_id"].values, "test")
    p = D["p"].values.copy(); sc = D["scored"].values
    p[sc] = b3.predict(xgb.DMatrix(fmat3(D[sc], use_cl)))
    keep = (p >= rule["thr"])
    mx = np.full(NTP, -np.inf, np.float32); np.maximum.at(mx, D["pool"].values, p)
    idx = np.flatnonzero(p == mx[D["pool"].values]); _, first = np.unique(D["pool"].values[idx], return_index=True)
    pm = np.zeros(len(p), bool); pm[idx[first]] = True; keep &= pm
    log(f"test: {len(D):,} candidate pairs ({(D['is_new'].values == 1).sum():,} new), {keep.sum():,} predicted matches ({keep.sum() / NT1:.3f}/S1), rule {rule}")
    os.makedirs(SUB, exist_ok=True)
    s1_ids, pool_ids = ts1["entity_id"].values, tpo["entity_id"].values
    order_file = pq.read_table(f"{W}/aligned_inputs/data_parquet/test_source1_final.parquet", columns=["entity_id"]).column(0).to_pylist()
    assert order_file == list(s1_ids)
    o = np.lexsort((-p, D["s1"].values)); s1s, ids, ks = D["s1"].values[o], pool_ids[D["pool"].values[o]], keep[o]
    bnd = np.searchsorted(s1s, np.arange(NT1 + 1))
    with open(f"{SUB}/matching_results.tsv.tmp", "w") as fm_, open(f"{SUB}/candidate_pairs.tsv.tmp", "w") as fc_:
        fm_.write("source1_entity_id\tmatched_entity_ids\n"); fc_.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(NT1):
            a_, b_ = bnd[i], bnd[i + 1]
            fm_.write(f"{s1_ids[i]}\t{','.join(ids[a_:b_][ks[a_:b_]])}\n"); fc_.write(f"{s1_ids[i]}\t{','.join(ids[a_:b_])}\n")
    for f_ in ("matching_results.tsv", "candidate_pairs.tsv"):
        os.replace(f"{SUB}/{f_}.tmp", f"{SUB}/{f_}")
    m = pd.read_csv(f"{SUB}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    cnd = pd.read_csv(f"{SUB}/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    errs = []
    if m["source1_entity_id"].tolist() != order_file or cnd["source1_entity_id"].tolist() != order_file: errs.append("row order")
    used = [x for s in m["matched_entity_ids"] for x in (s.split(",") if s else [])]
    if not set(used) <= set(pool_ids): errs.append("unknown pool ids")
    if len(used) != len(set(used)): errs.append("pool id matched twice")
    if any(not set(x.split(",") if x else []) <= set(y.split(",") if y else []) for x, y in zip(m["matched_entity_ids"], cnd["candidate_entity_ids"])):
        errs.append("match not among candidates")
    sha = {f_: hashlib.sha256(open(f"{SUB}/{f_}", "rb").read()).hexdigest() for f_ in ("matching_results.tsv", "candidate_pairs.tsv")}
    json.dump(dict(rule=rule, variant=best_v, errors=errs, sha256=sha, rep2=res, k_new=K_NEW), open(f"{SUB}/submission_info.json", "w"), indent=1)
    log(f"submission_3 {'VALIDATION PASSED' if not errs else 'VALIDATION FAILED ' + str(errs)} sha256 {sha}")
    log(f"done in {time.time() - T0:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
