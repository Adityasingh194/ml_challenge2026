"""Cross-encoder PILOT (go / no-go), leakage-safe:
  train  : XLM-R base on ~200k uncertain-band pairs of the CV training entities (single GPU; P2P is broken here)
  fit    : small 2nd-stage XGBoost [stage-1 p, CE score, entity context] on band pairs of a sample of the 80% EVAL entities
  measure: nested-holdout entities (never used above), macro F0.5 with vs without the CE, same decision rule family.
Steps are checkpointed under experiments/ce_pilot/ (rerun resumes)."""
import glob, json, os, sys, time, math, threading
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch

T0 = time.time()
W = os.path.join(os.environ.get("ER_ROOT", "/home/parth/Desktop/trial work"), "pipeline_run")   # same ER_ROOT as the notebook
C = f"{W}/cache"
OUT = f"{W}/experiments/ce_pilot"
os.makedirs(OUT, exist_ok=True)
FP = "dd3a402b93"
BAND = (0.05, 0.95)
N_TRAIN = 200_000
EVAL_FIT_ENTITIES = 200_000
MODEL = "FacebookAI/xlm-roberta-base"
MAXLEN = 128
CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1_retrieval_rerank", "cells")   # notebook cell sources
g = {"np": np, "pd": pd}
src = open(f"{CELLS}/022.py").read()
exec(src[src.index("def macro_f05"):src.index("def blocking_ceiling")], g)
macro_f05 = g["macro_f05"]
rng = np.random.default_rng(42)


LOGF = f"{W}/experiments/experiments.log"


def log(m):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ce_pilot +{time.time() - T0:5.0f}s] {m}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")


def rss():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 1e6


# ---------------- ids, truth, predictions ----------------
s1f = pq.read_table(f"{C}/frames/{FP}/train_s1_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
pof = pq.read_table(f"{C}/frames/{FP}/train_pool_norm.parquet", columns=["entity_id"]).to_pandas()
N1 = len(s1f)
gt = pq.read_table(f"{C}/samples/{FP}/gt.parquet").to_pandas()
rows = gt[gt["matched_entity_ids"] != ""]
sid = np.repeat(rows["source1_entity_id"].values, rows["matched_entity_ids"].str.count(",").values + 1)
cid = np.array([x.strip() for s in rows["matched_entity_ids"].values for x in s.split(",")], dtype=object)
a, b = pd.Index(s1f["entity_id"]).get_indexer(sid), pd.Index(pof["entity_id"]).get_indexer(cid)
ok = (a >= 0) & (b >= 0)
n_true = np.bincount(a[ok], minlength=N1).astype(np.int32)
cc = np.where(s1f["country_norm"].values == "India", 0, 1).astype(np.int8)
del gt, rows, sid, cid, a, b

rd = glob.glob(f"{C}/retrieval/train_*")[0]
oos = pq.read_table(glob.glob(f"{rd}/model_*/train_entities_oos.parquet")[0]).to_pandas()
lab = np.concatenate([pq.read_table(f, columns=["label"]).column(0).to_numpy() for f in
                      sorted(glob.glob(f"{rd}/trainset_*/block_*.parquet"))]).astype(bool)
oof = np.load(glob.glob(f"{C}/models/train_*/oof_1.npy")[0])
TR = pd.DataFrame({"s1": oos["s1_pos"].values.astype(np.int64), "pool": oos["pool_pos"].values.astype(np.int64),
                   "p": oos["p"].values.astype(np.float32), "y": lab, "cv": ~np.isnan(oof)})
del oos, lab, oof
ho_ent = np.zeros(N1, bool); ho_ent[TR["s1"].values[~TR["cv"].values]] = True
tr_ent = np.zeros(N1, bool); tr_ent[TR["s1"].values] = True
# eval sample for fitting the blend: EVAL_FIT_ENTITIES hash-sampled 80% entities
ev_files = sorted(glob.glob(f"{rd}/eval_*/block_*.parquet"))
EV = pd.concat([pq.read_table(f).to_pandas() for f in ev_files], ignore_index=True)
ev_ids = np.unique(EV["s1_pos"].values)
fit_ent = np.zeros(N1, bool); fit_ent[rng.choice(ev_ids, EVAL_FIT_ENTITIES, replace=False)] = True
EV = EV[fit_ent[EV["s1_pos"].values]].reset_index(drop=True)
EV = pd.DataFrame({"s1": EV["s1_pos"].values.astype(np.int64), "pool": EV["pool_pos"].values.astype(np.int64),
                   "p": EV["p"].values.astype(np.float32), "y": EV["y"].values.astype(bool)})
log(f"loaded: train rows {len(TR):,}  holdout ent {ho_ent.sum():,}  eval-fit ent {fit_ent.sum():,} rows {len(EV):,}  RSS {rss():.1f} GB")


def entity_ctx(D):
    s1, p = D["s1"].values, D["p"].values
    o = np.lexsort((-p, s1)); ss = s1[o]
    st = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]; sz = np.diff(np.r_[st, len(ss)])
    rk = np.empty(len(p), np.int32); rk[o] = np.arange(len(ss)) - np.repeat(st, sz)
    p1 = np.full(N1, 0, np.float32); p1[ss[st]] = p[o[st]]
    p2 = np.full(N1, 0, np.float32); two = sz > 1; p2[ss[st[two]]] = p[o[st[two] + 1]]
    D["rk"], D["p1"], D["p2"] = rk, p1[s1], p2[s1]
    D["n_band"] = pd.Series((p >= BAND[0]) & (p <= BAND[1])).groupby(s1).transform("sum").values.astype(np.float32)
    return D


TR = entity_ctx(TR); EV = entity_ctx(EV)
inb = lambda D: (D["p"].values >= BAND[0]) & (D["p"].values <= BAND[1])

# ---------------- raw texts ----------------
def raw(s):
    return pq.read_table(f"{W}/aligned_inputs/data_parquet/train_source{s}_final.parquet",
                         columns=["entity_id", "business_name", "business_address"]).to_pandas().set_index("entity_id")
r1 = raw(1); rp = pd.concat([raw(2), raw(3)])
s1_ids, pool_ids = s1f["entity_id"].values, pof["entity_id"].values


def texts(D):
    a = r1.reindex(s1_ids[D["s1"].values]); b = rp.reindex(pool_ids[D["pool"].values])
    ta = (a["business_name"].fillna("") + " | " + a["business_address"].fillna("")).tolist()
    tb = (b["business_name"].fillna("") + " | " + b["business_address"].fillna("")).tolist()
    return ta, tb


# ---------------- 1) train the cross-encoder ----------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(MODEL)
mdir = f"{OUT}/model"
trn_band = TR[TR["cv"].values & inb(TR)]
if not os.path.exists(f"{mdir}/done"):
    take = trn_band.sample(n=min(N_TRAIN, len(trn_band)), random_state=1)
    ta, tb = texts(take); y = take["y"].values.astype(np.float32)
    log(f"train pairs {len(take):,} (band pairs available {len(trn_band):,}), positives {y.mean():.3f}")
    dev = "cuda:0"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    BS, EPOCHS = 64, 1
    steps = EPOCHS * math.ceil(len(y) / BS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train(); step = 0; t_tr = time.time()
    ck = f"{OUT}/train_ckpt.pt"
    if os.path.exists(ck):   # resume after a crash
        st_ = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st_["model"]); opt.load_state_dict(st_["opt"]); sched.load_state_dict(st_["sched"])
        step = st_["step"]
        log(f"  resumed training from checkpoint at step {step}")
    for ep in range(EPOCHS):
        perm = np.random.default_rng(ep).permutation(len(y))
        for i in range(0, len(y), BS):
            if ep * math.ceil(len(y) / BS) + i // BS < step:
                continue
            ix = perm[i:i + BS]
            enc = tok([ta[j] for j in ix], [tb[j] for j in ix], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.from_numpy(y[ix]).to(dev))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 1000 == 0:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step), ck + ".tmp")
                os.replace(ck + ".tmp", ck)
                log(f"  checkpoint saved at step {step}")
            if step % 500 == 0:
                log(f"  step {step}/{steps} loss {loss.item():.4f}  {(step * BS) / (time.time() - t_tr):,.0f} pairs/s  "
                    f"GPU0 peak {torch.cuda.max_memory_allocated(0) / 1e9:.1f} GB")
    model.save_pretrained(mdir); tok.save_pretrained(mdir)
    open(f"{mdir}/done", "w").write("ok")
    if os.path.exists(ck):
        os.remove(ck)
    log(f"trained in {time.time() - t_tr:.0f}s")
    del model, opt
    torch.cuda.empty_cache()


# ---------------- 2) score band pairs on both GPUs (independent copies; host merge) ----------------
def score_pairs(D, name):
    path = f"{OUT}/ce_{name}.npy"
    if os.path.exists(path):
        return np.load(path)
    ta, tb = texts(D)
    n = len(ta); out = np.empty(n, np.float32)
    CH = 50_000   # pairs per checkpointed chunk
    cdir = f"{OUT}/ce_{name}_chunks"; os.makedirs(cdir, exist_ok=True)
    chunks = [(i, min(i + CH, n)) for i in range(0, n, CH)]
    todo = [c for c in chunks if not os.path.exists(f"{cdir}/{c[0]:010d}.npy")]
    log(f"scoring {name}: {n:,} pairs in {len(chunks)} chunks ({len(chunks) - len(todo)} already done)")
    lock = threading.Lock()

    def work(gi, mine):
        dev = f"cuda:{gi}"
        m = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).eval()
        for lo, hi in mine:
            tc = time.time(); buf = np.empty(hi - lo, np.float32)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(lo, hi, 512):
                    j = min(i + 512, hi)
                    enc = tok(ta[i:j], tb[i:j], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
                    buf[i - lo:j - lo] = m(**enc).logits.squeeze(-1).float().cpu().numpy()
            np.save(f"{cdir}/{lo:010d}.tmp.npy", buf); os.replace(f"{cdir}/{lo:010d}.tmp.npy", f"{cdir}/{lo:010d}.npy")
            with lock:
                log(f"  {name} chunk {lo:,}-{hi:,} on cuda:{gi}: {(hi - lo) / (time.time() - tc):,.0f} pairs/s  "
                    f"GPU{gi} peak {torch.cuda.max_memory_allocated(gi) / 1e9:.1f} GB  RSS {rss():.1f} GB")
        del m
        torch.cuda.empty_cache()

    t = time.time()
    ths = [threading.Thread(target=work, args=(gi, todo[gi::2])) for gi in range(2)]
    [th.start() for th in ths]; [th.join() for th in ths]
    for lo, hi in chunks:
        out[lo:hi] = np.load(f"{cdir}/{lo:010d}.npy")
    np.save(path, out)
    log(f"scored {name}: {n:,} pairs in {time.time() - t:.0f}s ({n / max(time.time() - t, 1e-9):,.0f} pairs/s)  "
        f"GPU peaks {[round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(2)]} GB")
    return out


HO = TR[~TR["cv"].values].reset_index(drop=True)
HO_b = HO[inb(HO)].reset_index(drop=True)
EV_b = EV[inb(EV)].reset_index(drop=True)
ce_ho = score_pairs(HO_b, "holdout_band")
ce_ev = score_pairs(EV_b, "evalfit_band")

from sklearn.metrics import roc_auc_score
log(f"band AUC  holdout: stage-1 p {roc_auc_score(HO_b['y'], HO_b['p']):.4f}  CE {roc_auc_score(HO_b['y'], ce_ho):.4f}  |  "
    f"eval-fit: stage-1 p {roc_auc_score(EV_b['y'], EV_b['p']):.4f}  CE {roc_auc_score(EV_b['y'], ce_ev):.4f}")

# ---------------- 3) small 2nd stage on eval-fit band, measured on the holdout ----------------
import xgboost as xgb
FEATS = ["p", "ce", "rk", "p1", "p2", "n_band", "gap"]


def fmat(D, ce):
    return np.c_[D["p"].values, ce, D["rk"].values, D["p1"].values, D["p2"].values, D["n_band"].values,
                 D["p1"].values - D["p"].values].astype(np.float32)


def fmat_noce(D):
    return np.delete(fmat(D, np.zeros(len(D), np.float32)), 1, axis=1)


prm = dict(objective="binary:logistic", max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9,
           min_child_weight=5, tree_method="hist", device="cuda:0", eval_metric="logloss", seed=0)
b_ce = xgb.train(prm, xgb.DMatrix(fmat(EV_b, ce_ev), label=EV_b["y"].values), 400)
b_no = xgb.train(prm, xgb.DMatrix(fmat_noce(EV_b), label=EV_b["y"].values), 400)   # same 2nd stage WITHOUT the CE (control)


def new_p(D, booster, X):
    p = D["p"].values.copy()
    m = inb(D)
    p[m] = booster.predict(xgb.DMatrix(X))
    return p


EV_ce = EV["p"].values.copy(); EV_ce[inb(EV)] = b_ce.predict(xgb.DMatrix(fmat(EV_b, ce_ev)))
EV_no = EV["p"].values.copy(); EV_no[inb(EV)] = b_no.predict(xgb.DMatrix(fmat_noce(EV_b)))
HO_ce = HO["p"].values.copy(); HO_ce[inb(HO)] = b_ce.predict(xgb.DMatrix(fmat(HO_b, ce_ho)))
HO_no = HO["p"].values.copy(); HO_no[inb(HO)] = b_no.predict(xgb.DMatrix(fmat_noce(HO_b)))
TG = np.round(np.r_[np.arange(0.30, 0.90, 0.02), np.arange(0.90, 0.99, 0.01)], 3)


def f05(D, p, t, ent):
    k = p >= t
    return macro_f05(D["s1"].values[k], D["y"].values[k], n_true, ent)


def pick_t(D, p, ent):   # threshold chosen on the FIT set only
    return max(TG, key=lambda t: f05(D, p, t, ent))


res = {}
for name, pe, ph in (("stage-1 only (baseline p)", EV["p"].values, HO["p"].values),
                     ("2nd stage WITHOUT CE (control)", EV_no, HO_no), ("2nd stage WITH CE", EV_ce, HO_ce)):
    t = pick_t(EV, pe, fit_ent)
    k = ph >= t; m = ho_ent[HO["s1"].values]
    tp = int((k & HO["y"].values).sum()); fp = int((k & ~HO["y"].values).sum()); fn = int(n_true[ho_ent].sum()) - tp
    res[name] = dict(thr=float(t), holdout_f05=round(f05(HO, ph, t, ho_ent), 5), precision=round(tp / max(tp + fp, 1), 5),
                     recall=round(tp / max(tp + fn, 1), 5), fp=fp, fn=fn,
                     india=round(f05(HO, ph, t, ho_ent & (cc == 0)), 5), us=round(f05(HO, ph, t, ho_ent & (cc == 1)), 5))
R = pd.DataFrame(res).T
print(R.to_string())
log("RESULT (holdout):\n" + R.to_string())
base = res["stage-1 only (baseline p)"]["holdout_f05"]
log(f"CE effect on holdout vs control: {res['2nd stage WITH CE']['holdout_f05'] - res['2nd stage WITHOUT CE (control)']['holdout_f05']:+.5f}")
print(f"\nCE effect on holdout (vs control without CE): {res['2nd stage WITH CE']['holdout_f05'] - res['2nd stage WITHOUT CE (control)']['holdout_f05']:+.5f}"
      f"   vs stage-1 baseline: {res['2nd stage WITH CE']['holdout_f05'] - base:+.5f}")
R.to_csv(f"{OUT}/pilot_result.tsv", sep="\t")
import resource
log(f"done. peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
