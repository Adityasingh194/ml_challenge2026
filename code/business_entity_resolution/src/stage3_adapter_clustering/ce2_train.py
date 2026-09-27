"""CE epoch 2: continue the ce_full cross-encoder on (a) adapter-only pairs (fused top-20 not in the 120-union) and
(b) a replay sample of uncertain-band pairs. Training entities = the stage-1 CV entities (disjoint from FIT2/REP2 and
from the adapter's training entities), minus 5% kept aside for a quick old-vs-new CE sanity check.
Writes experiments/ce2/model; logs to logs/ce2.log + experiments.log; checkpoints every 1000 steps."""
import glob, os, sys, time, math
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch

sys.argv = [sys.argv[0]]
import adapter_fast as AF
from adapter_fast import emb, raw, texts, inb, rss, W, C, FP

T0 = time.time()
OUT = f"{W}/experiments/ce2"; os.makedirs(OUT, exist_ok=True)
OLD = f"{W}/experiments/ce_full/model"
K, N_NEG, N_BAND, BS, LR, MAXLEN = 20, 250_000, 150_000, 64, 1e-5, 128


def log(m):
    AF.log(m, "ce2")


s1f = pq.read_table(f"{C}/frames/{FP}/train_s1_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
pof = pq.read_table(f"{C}/frames/{FP}/train_pool_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
N1, NP = len(s1f), len(pof)
gt = pq.read_table(f"{C}/samples/{FP}/gt.parquet").to_pandas(); rows = gt[gt["matched_entity_ids"] != ""]
sid = np.repeat(rows["source1_entity_id"].values, rows["matched_entity_ids"].str.count(",").values + 1)
cid = np.array([x.strip() for s in rows["matched_entity_ids"].values for x in s.split(",")], dtype=object)
a, b = pd.Index(s1f["entity_id"]).get_indexer(sid), pd.Index(pof["entity_id"]).get_indexer(cid)
ok = (a >= 0) & (b >= 0); true_keys = np.unique(a[ok].astype(np.int64) * NP + b[ok])
del gt, rows, sid, cid, a, b
rd = glob.glob(f"{C}/retrieval/train_*")[0]
oos = pq.read_table(glob.glob(f"{rd}/model_*/train_entities_oos.parquet")[0]).to_pandas()
oof = np.load(glob.glob(f"{C}/models/train_*/oof_1.npy")[0])
cvm = ~np.isnan(oof)
tr_s1 = oos["s1_pos"].values.astype(np.int64)[cvm]; tr_pool = oos["pool_pos"].values.astype(np.int64)[cvm]
tr_p = oos["p"].values.astype(np.float32)[cvm]
del oos, oof
cv_ent = np.unique(tr_s1)
rng = np.random.default_rng(11)
side = np.zeros(N1, bool); side[rng.choice(cv_ent, len(cv_ent) // 20, replace=False)] = True   # 5% sanity entities
log(f"CV entities {len(cv_ent):,} (sanity 5%: {side.sum():,}); CV candidate rows {len(tr_s1):,}; RSS {rss():.1f} GB")

# ---- adapter-only pairs of the CV entities ----
X1 = emb("train", 1); XP = np.concatenate([emb("train", 2), emb("train", 3)], axis=0)
fid, fsc = AF.fused_search(X1, XP, cv_ent, s1f["country_norm"].values, pof["country_norm"].values, K, "train_cv")
del X1, XP
fk = np.repeat(cv_ent, K) * np.int64(NP) + fid.reshape(-1); v = fid.reshape(-1) >= 0; fk = fk[v]
new = fk[~np.isin(fk, tr_s1 * NP + tr_pool)]
ny = np.isin(new, true_keys)
log(f"adapter-only pairs {len(new):,} ({len(new) / len(cv_ent):.2f}/S1), positives {ny.sum():,} ({ny.mean():.4f})")

# ---- training / sanity sets ----
ns1 = new // NP
tr_new, sd_new = ~side[ns1], side[ns1]
pos = np.flatnonzero(tr_new & ny); neg = np.flatnonzero(tr_new & ~ny)
neg = rng.choice(neg, min(N_NEG, len(neg)), replace=False)
band = np.flatnonzero(inb(tr_p) & ~side[tr_s1])
band = rng.choice(band, min(N_BAND, len(band)), replace=False)
keys = np.r_[new[pos], new[neg], tr_s1[band] * NP + tr_pool[band]]
y = np.isin(keys, true_keys).astype(np.float32)
kind = np.r_[np.zeros(len(pos) + len(neg), np.int8), np.ones(len(band), np.int8)]
log(f"CE2 training pairs {len(keys):,}: new pos {len(pos):,} / new neg {len(neg):,} / band replay {len(band):,} (band pos rate {y[kind == 1].mean():.3f})")
sd_n = rng.choice(np.flatnonzero(sd_new), min(40_000, int(sd_new.sum())), replace=False)
sd_b = np.flatnonzero(inb(tr_p) & side[tr_s1]); sd_b = rng.choice(sd_b, min(20_000, len(sd_b)), replace=False)
sd_keys = np.r_[new[sd_n], tr_s1[sd_b] * NP + tr_pool[sd_b]]
sd_y = np.isin(sd_keys, true_keys); sd_kind = np.r_[np.zeros(len(sd_n), np.int8), np.ones(len(sd_b), np.int8)]

r1, rp = raw("train", 1), pd.concat([raw("train", 2), raw("train", 3)])
S1_IDS = s1f["entity_id"].to_numpy(dtype=object); POOL_IDS = pof["entity_id"].to_numpy(dtype=object)
ta, tb = texts(S1_IDS[keys // NP], POOL_IDS[keys % NP], r1, rp)
sa, sb = texts(S1_IDS[sd_keys // NP], POOL_IDS[sd_keys % NP], r1, rp)
del r1, rp
log(f"texts built; RSS {rss():.1f} GB")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(OLD)
mdir = f"{OUT}/model"
if not os.path.exists(f"{mdir}/done"):
    dev = "cuda:0"
    model = AutoModelForSequenceClassification.from_pretrained(OLD).to(dev)
    # length-bucketed batches (less padding): shuffle, sort inside windows of 50 batches, shuffle batch order
    perm = np.random.default_rng(0).permutation(len(y)); L = np.array([len(ta[i]) + len(tb[i]) for i in range(len(y))])
    batches = []
    for w in range(0, len(perm), BS * 50):
        wi = perm[w:w + BS * 50]; wi = wi[np.argsort(L[wi], kind="stable")]
        batches += [wi[i:i + BS] for i in range(0, len(wi), BS)]
    batches = [batches[i] for i in np.random.default_rng(1).permutation(len(batches))]
    steps = len(batches)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    step, ck = 0, f"{OUT}/train_ckpt.pt"
    if os.path.exists(ck):
        st_ = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st_["model"]); opt.load_state_dict(st_["opt"]); sched.load_state_dict(st_["sched"]); step = st_["step"]
        log(f"  resumed CE2 training at step {step}")
    model.train(); t_tr = time.time(); done0 = step; run_loss = 0.0
    for bi in range(step, steps):
        ix = batches[bi]
        enc = tok([ta[j] for j in ix], [tb[j] for j in ix], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(y[ix]).to(dev))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1; run_loss += loss.item()
        if step % 500 == 0:
            el = time.time() - t_tr
            log(f"  step {step}/{steps} loss(avg500) {run_loss / 500:.4f}  {(step - done0) * BS / el:,.0f} pairs/s  ETA "
                f"{el / (step - done0) * (steps - step) / 60:.1f} min  GPU0 peak {torch.cuda.max_memory_allocated(0) / 1e9:.1f} GB  RSS {rss():.1f} GB")
            run_loss = 0.0
        if step % 1000 == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    model.save_pretrained(mdir); tok.save_pretrained(mdir); open(f"{mdir}/done", "w").write("ok")
    if os.path.exists(ck):
        os.remove(ck)
    log(f"CE2 trained: {steps} steps in {time.time() - t_tr:.0f}s")
    del model, opt; torch.cuda.empty_cache()


# ---- sanity: old CE vs CE2 on the 5% side entities (never trained on) ----
@torch.no_grad()
def score(md, dev):
    m = AutoModelForSequenceClassification.from_pretrained(md).to(dev).eval(); out = np.empty(len(sa), np.float32)
    order = np.argsort([len(sa[i]) + len(sb[i]) for i in range(len(sa))], kind="stable")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(sa), 512):
            ix = order[k:k + 512]
            enc = tok([sa[i] for i in ix], [sb[i] for i in ix], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
            out[ix] = m(**enc).logits.squeeze(-1).float().cpu().numpy()
    del m; torch.cuda.empty_cache(); return out


from sklearn.metrics import roc_auc_score, average_precision_score
so, sn = score(OLD, "cuda:0"), score(mdir, "cuda:1")
for nm, k in (("adapter-only pairs", sd_kind == 0), ("band pairs", sd_kind == 1)):
    yy = sd_y[k]
    log(f"sanity {nm}: n {k.sum():,} pos {yy.mean():.4f} | old CE AUC {roc_auc_score(yy, so[k]):.4f} AP {average_precision_score(yy, so[k]):.4f}"
        f" | CE2 AUC {roc_auc_score(yy, sn[k]):.4f} AP {average_precision_score(yy, sn[k]):.4f}")
log(f"done in {time.time() - T0:.0f}s")
