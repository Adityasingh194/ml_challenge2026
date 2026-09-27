"""Adapter PILOT (go/no-go): learned fusion of the precomputed Qwen3-0.6B name(256)+addr(512) vectors.
Two towers (S1 / pool), InfoNCE with in-batch + hard negatives and a multi-positive mask.
Trained on ADAPTER entities (the 25% eval 'fit' split, disjoint from the reranker's entities);
recall measured on held-out REPORT entities against the FULL train pool (per country, S2+S3 together).
Go if union(existing 120 candidates + fused top-K) recall >= 0.985 (existing 0.974). Logs to experiments.log."""
import glob, hashlib, os, time, math, resource
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F

T0 = time.time()
W = os.path.join(os.environ.get("ER_ROOT", "/home/parth/Desktop/trial work"), "pipeline_run")   # same ER_ROOT as the notebook
C, A = f"{W}/cache", f"{W}/aligned_inputs/emb_aligned"
OUT = f"{W}/experiments/adapter"; os.makedirs(OUT, exist_ok=True)
FP = "dd3a402b93"
TAU, N_HARD, BS, EPOCHS, DIM = 0.05, 6, 2048, 4, 256
N_EVAL_ENT = 60_000
LOGF = f"{W}/experiments/experiments.log"


def log(m):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [adapter +{time.time() - T0:5.0f}s] {m}"
    print(line, flush=True)
    open(LOGF, "a").write(line + "\n")


def rss():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 1e6


# ---------------- data ----------------
s1f = pq.read_table(f"{C}/frames/{FP}/train_s1_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
pof = pq.read_table(f"{C}/frames/{FP}/train_pool_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
N1, NP = len(s1f), len(pof)
gt = pq.read_table(f"{C}/samples/{FP}/gt.parquet").to_pandas()
rows = gt[gt["matched_entity_ids"] != ""]
sid = np.repeat(rows["source1_entity_id"].values, rows["matched_entity_ids"].str.count(",").values + 1)
cid = np.array([x.strip() for s in rows["matched_entity_ids"].values for x in s.split(",")], dtype=object)
ts1, tpo = pd.Index(s1f["entity_id"]).get_indexer(sid), pd.Index(pof["entity_id"]).get_indexer(cid)
ok = (ts1 >= 0) & (tpo >= 0); ts1, tpo = ts1[ok].astype(np.int64), tpo[ok].astype(np.int64)
n_true = np.bincount(ts1, minlength=N1)
owner = np.full(NP, -1, np.int64); owner[tpo] = ts1
cc1 = s1f["country_norm"].values; ccp = pof["country_norm"].values
del gt, rows, sid, cid

# the same entity split as ce_full.py: 25% of eval entities = adapter training, rest = report (held out)
rd = glob.glob(f"{C}/retrieval/train_*")[0]
ev_files = sorted(glob.glob(f"{rd}/eval_*/block_*.parquet"))
EV = pd.concat([pq.read_table(f).to_pandas() for f in ev_files], ignore_index=True)
ev_ids = np.unique(EV["s1_pos"].values)
h = np.frombuffer(hashlib.sha256(b"ce_full_fit_split").digest()[:8], np.uint64)[0]
u = ((ev_ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + h) >> np.uint64(11)).astype(np.float64) / 2 ** 53
ad_ent = np.zeros(N1, bool); ad_ent[ev_ids[u < 0.25]] = True
rep_ids = ev_ids[u >= 0.25]
log(f"entities: adapter-train {ad_ent.sum():,}, report {len(rep_ids):,}; RSS {rss():.1f} GB")

# hard negatives: highest stage-1 p non-matches of each adapter entity (from the existing candidate lists)
E_ad = EV[ad_ent[EV["s1_pos"].values] & ~EV["y"].values.astype(bool)]
E_ad = E_ad.sort_values(["s1_pos", "p"], ascending=[True, False])
E_ad["r"] = E_ad.groupby("s1_pos").cumcount()
E_ad = E_ad[E_ad["r"] < N_HARD]
hard = np.full((N1, N_HARD), -1, np.int64)
hard[E_ad["s1_pos"].values, E_ad["r"].values] = E_ad["pool_pos"].values
# existing-union membership for the held-out recall comparison
rng = np.random.default_rng(0)
ev_sample = np.sort(rng.choice(rep_ids, N_EVAL_ENT, replace=False))
sm = np.zeros(N1, bool); sm[ev_sample] = True
E_rep = EV[sm[EV["s1_pos"].values]]
existing = set((E_rep["s1_pos"].values.astype(np.int64) * NP + E_rep["pool_pos"].values).tolist())
del EV, E_ad, E_rep
log(f"hard negatives ready; held-out sample {len(ev_sample):,} entities; RSS {rss():.1f} GB")


def emb(split, s):
    return np.concatenate([np.load(f"{A}/{split}_s{s}_name.npy"), np.load(f"{A}/{split}_s{s}_addr.npy")], axis=1)


X1 = emb("train", 1)
XP = np.concatenate([emb("train", 2), emb("train", 3)], axis=0)
assert len(X1) == N1 and len(XP) == NP
log(f"embeddings loaded: S1 {X1.shape}, pool {XP.shape} int8; RSS {rss():.1f} GB")
dev0, dev1 = "cuda:0", "cuda:1"
X1g, XPg = torch.from_numpy(X1).to(dev0), torch.from_numpy(XP).to(dev0)
owner_g, hard_g = torch.from_numpy(owner).to(dev0), torch.from_numpy(hard).to(dev0)


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
        super().__init__()
        s.fq, s.fd = Tower(), Tower()


model = Adapter().to(dev0)
ck = f"{OUT}/adapter.pt"
pa = np.flatnonzero(ad_ent[ts1])          # positive pairs of adapter entities
a_all, p_all = ts1[pa], tpo[pa]
if os.path.exists(ck):
    model.load_state_dict(torch.load(ck, map_location=dev0)); log("adapter loaded from checkpoint")
else:
    steps = EPOCHS * math.ceil(len(pa) / BS)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    log(f"training on {len(pa):,} positive pairs of {ad_ent.sum():,} entities, {steps} steps (batch {BS}, {N_HARD} hard negatives, tau {TAU})")
    step, t = 0, time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(pa), device=dev0)
        at, pt = torch.from_numpy(a_all).to(dev0)[perm], torch.from_numpy(p_all).to(dev0)[perm]
        losses = []
        for i in range(0, len(pa), BS):
            a, p = at[i:i + BS], pt[i:i + BS]
            hn = hard_g[a].reshape(-1)
            docs = torch.cat([p, hn.clamp(min=0)])
            valid = torch.cat([torch.ones_like(p, dtype=torch.bool), hn >= 0])
            q = model.fq(X1g[a]); d = model.fd(XPg[docs])
            logits = q @ d.T / TAU
            mask = owner_g[docs][None, :] == a[:, None]            # other true matches of this anchor: not negatives
            mask[torch.arange(len(a), device=dev0), torch.arange(len(a), device=dev0)] = False
            mask |= ~valid[None, :]
            loss = F.cross_entropy(logits.masked_fill(mask, -1e4), torch.arange(len(a), device=dev0))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            losses.append(loss.item())
        log(f"  epoch {ep + 1}/{EPOCHS}: loss {np.mean(losses):.4f}  ({time.time() - t:.0f}s)  GPU0 peak {torch.cuda.max_memory_allocated(0) / 1e9:.1f} GB")
        torch.save(model.state_dict(), ck)
    log(f"adapter trained in {time.time() - t:.0f}s")
del XPg, hard_g
torch.cuda.empty_cache()


# ---------------- held-out recall against the FULL pool, per country, S2+S3 together ----------------
@torch.no_grad()
def project(tower, X, dev, chunk=500_000):
    tower = tower.cpu().to(dev).eval()   # via host: direct GPU->GPU copies return zeros on this machine (broken P2P)
    return torch.cat([tower(torch.from_numpy(X[s:s + chunk]).to(dev)).half() for s in range(0, len(X), chunk)])


K = 50
res_ids = np.full((len(ev_sample), K), -1, np.int64)
t = time.time()
for c in sorted(set(cc1[ev_sample])):
    qi = np.flatnonzero(cc1[ev_sample] == c)
    pi = np.flatnonzero(ccp == c)
    Pc = project(model.fd, XP[pi], dev1)                           # fp16 on GPU1
    Qc = project(model.fq, X1[ev_sample[qi]], dev1)
    for s in range(0, len(qi), 2048):
        best_s = torch.full((min(2048, len(qi) - s), K), -1e4, device=dev1, dtype=torch.float32)
        best_i = torch.full_like(best_s, -1, dtype=torch.int64)
        for p0 in range(0, len(pi), 500_000):
            sc = (Qc[s:s + 2048] @ Pc[p0:p0 + 500_000].T).float()
            v, ix = sc.topk(K, dim=1)
            cat_s, cat_i = torch.cat([best_s, v], 1), torch.cat([best_i, ix + p0], 1)
            best_s, o = cat_s.topk(K, dim=1); best_i = cat_i.gather(1, o)
        res_ids[qi[s:s + 2048]] = pi[best_i.cpu().numpy()]
    del Pc, Qc; torch.cuda.empty_cache()
    log(f"  fused search {c}: {len(qi):,} queries vs {len(pi):,} pool records  GPU1 peak {torch.cuda.max_memory_allocated(1) / 1e9:.1f} GB")
log(f"fused search done in {time.time() - t:.0f}s")

hm = np.isin(ts1, ev_sample)
hs1, hpo = ts1[hm], tpo[hm]
row = np.searchsorted(ev_sample, hs1)
tot = len(hs1)
in_exist = np.array([k in existing for k in (hs1 * NP + hpo).tolist()])
lines = [f"held-out: {len(ev_sample):,} report entities, {tot:,} true pairs; existing union@120 recall = {in_exist.mean():.4f}"]
for k in (1, 5, 10, 20, 30, 50):
    hit = (res_ids[row, :k] == hpo[:, None]).any(1)
    lines.append(f"  fused recall@{k:<3}= {hit.mean():.4f}   union(existing + fused@{k}) = {(hit | in_exist).mean():.4f}   "
                 f"fused finds {(hit & ~in_exist).sum():,} of {(~in_exist).sum():,} pairs the existing union misses")
for c in sorted(set(cc1[ev_sample])):
    m = cc1[hs1] == c
    hit = (res_ids[row[m], :30] == hpo[m][:, None]).any(1)
    lines.append(f"  {c}: existing {in_exist[m].mean():.4f}  fused@30 {hit.mean():.4f}  union {(hit | in_exist[m]).mean():.4f}")
log("RESULT (held-out, full pool):\n" + "\n".join(lines))
np.save(f"{OUT}/heldout_fused_top{K}.npy", res_ids)
log(f"done; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
