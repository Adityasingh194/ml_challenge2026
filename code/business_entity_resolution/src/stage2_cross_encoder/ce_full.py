"""Cross-encoder, full run (after the positive pilot). Leakage-safe protocol:
  CE train   : XLM-R base on ALL uncertain-band pairs of the CV training entities (single GPU: P2P is broken here)
  2nd stage  : small XGBoost [stage-1 p, CE, entity context] + decision rule fitted on 25% of the 80% EVAL entities
  report     : the other 75% of eval entities (never used) + nested holdout
  test       : same 2nd stage + rule -> output/submission_2 (submission_1 / baseline_v1 are never touched)
Every step logs to experiments/experiments.log, checkpoints (training every 1000 steps, scoring per 50k-pair chunk,
2nd stage / results as files) and resumes on rerun."""
import glob, json, os, sys, time, math, threading, hashlib, csv, resource
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch

T0 = time.time()
W = os.path.join(os.environ.get("ER_ROOT", "/home/parth/Desktop/trial work"), "pipeline_run")   # same ER_ROOT as the notebook
C = f"{W}/cache"
OUT = f"{W}/experiments/ce_full"
SUB = f"{W}/output/submission_2"
os.makedirs(OUT, exist_ok=True)
FP_TR, BAND = "dd3a402b93", (0.05, 0.95)
MODEL, MAXLEN, CH = "FacebookAI/xlm-roberta-base", 128, 50_000
CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1_retrieval_rerank", "cells")   # notebook cell sources
g = {"np": np, "pd": pd}
src = open(f"{CELLS}/022.py").read()
exec(src[src.index("def macro_f05"):src.index("def blocking_ceiling")], g)
macro_f05 = g["macro_f05"]
LOGF = f"{W}/experiments/experiments.log"


def log(m):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ce_full +{time.time() - T0:5.0f}s] {m}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")


def rss():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 1e6


inb = lambda p: (p >= BAND[0]) & (p <= BAND[1])

# =============== TRAIN side: ids, truth, predictions ===============
s1f = pq.read_table(f"{C}/frames/{FP_TR}/train_s1_norm.parquet", columns=["entity_id", "country_norm"]).to_pandas()
pof = pq.read_table(f"{C}/frames/{FP_TR}/train_pool_norm.parquet", columns=["entity_id"]).to_pandas()
N1, NP = len(s1f), len(pof)
gt = pq.read_table(f"{C}/samples/{FP_TR}/gt.parquet").to_pandas()
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
EV = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{rd}/eval_*/block_*.parquet"))], ignore_index=True)
EV = pd.DataFrame({"s1": EV["s1_pos"].values.astype(np.int64), "pool": EV["pool_pos"].values.astype(np.int64),
                   "p": EV["p"].values.astype(np.float32), "y": EV["y"].values.astype(bool)})
ev_ids = np.unique(EV["s1"].values)
h = np.frombuffer(hashlib.sha256(b"ce_full_fit_split").digest()[:8], np.uint64)[0]
u = ((ev_ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + h) >> np.uint64(11)).astype(np.float64) / 2 ** 53
fit_ent = np.zeros(N1, bool); fit_ent[ev_ids[u < 0.25]] = True
rep_ent = np.zeros(N1, bool); rep_ent[ev_ids[u >= 0.25]] = True
log(f"train side loaded: train-entity rows {len(TR):,}, eval rows {len(EV):,}; entities fit {fit_ent.sum():,} / report "
    f"{rep_ent.sum():,} / holdout {ho_ent.sum():,}; RSS {rss():.1f} GB")


def raw(split, s):
    return pq.read_table(f"{W}/aligned_inputs/data_parquet/{split}_source{s}_final.parquet",
                         columns=["entity_id", "business_name", "business_address"]).to_pandas().set_index("entity_id")


def texts(s1_ids, pool_ids, r1, rp):
    a = r1.reindex(s1_ids); b = rp.reindex(pool_ids)
    return ((a["business_name"].fillna("") + " | " + a["business_address"].fillna("")).tolist(),
            (b["business_name"].fillna("") + " | " + b["business_address"].fillna("")).tolist())


r1_tr, rp_tr = raw("train", 1), pd.concat([raw("train", 2), raw("train", 3)])
S1_IDS_TR, POOL_IDS_TR = s1f["entity_id"].values, pof["entity_id"].values

# =============== 1) train the cross-encoder ===============
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(MODEL)
mdir = f"{OUT}/model"
if not os.path.exists(f"{mdir}/done"):
    take = TR[TR["cv"].values & inb(TR["p"].values)]
    ta, tb = texts(S1_IDS_TR[take["s1"].values], POOL_IDS_TR[take["pool"].values], r1_tr, rp_tr)
    y = take["y"].values.astype(np.float32)
    log(f"CE training pairs {len(y):,} (all CV-entity band pairs), positives {y.mean():.3f}")
    dev = "cuda:0"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    BS = 64
    steps = math.ceil(len(y) / BS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    step, ck = 0, f"{OUT}/train_ckpt.pt"
    if os.path.exists(ck):
        st_ = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st_["model"]); opt.load_state_dict(st_["opt"]); sched.load_state_dict(st_["sched"]); step = st_["step"]
        log(f"  resumed CE training at step {step}")
    model.train(); t_tr = time.time(); done0 = step
    perm = np.random.default_rng(0).permutation(len(y))
    for i in range(step * BS, len(y), BS):
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
            log(f"  step {step}/{steps} loss {loss.item():.4f}  {(step - done0) * BS / (time.time() - t_tr):,.0f} pairs/s  "
                f"GPU0 peak {torch.cuda.max_memory_allocated(0) / 1e9:.1f} GB  RSS {rss():.1f} GB  (checkpoint saved)")
    model.save_pretrained(mdir); tok.save_pretrained(mdir); open(f"{mdir}/done", "w").write("ok")
    if os.path.exists(ck):
        os.remove(ck)
    log(f"CE trained: {steps} steps in {time.time() - t_tr:.0f}s")
    del model, opt, ta, tb
    torch.cuda.empty_cache()


# =============== 2) score band pairs: both GPUs, 50k-pair chunks, length-sorted batches ===============
def score_pairs(ta, tb, name):
    path = f"{OUT}/ce_{name}.npy"
    if os.path.exists(path):
        log(f"scores {name}: cached")
        return np.load(path)
    n = len(ta); out = np.empty(n, np.float32)
    cdir = f"{OUT}/ce_{name}_chunks"; os.makedirs(cdir, exist_ok=True)
    chunks = [(i, min(i + CH, n)) for i in range(0, n, CH)]
    todo = [c for c in chunks if not os.path.exists(f"{cdir}/{c[0]:010d}.npy")]
    log(f"scoring {name}: {n:,} pairs, {len(chunks)} chunks ({len(chunks) - len(todo)} done)")
    lock, t = threading.Lock(), time.time()
    cnt = [0]

    def work(gi, mine):
        dev = f"cuda:{gi}"
        m = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).eval()
        for lo, hi in mine:
            buf = np.empty(hi - lo, np.float32)
            order = np.argsort([len(ta[i]) + len(tb[i]) for i in range(lo, hi)], kind="stable")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for k in range(0, hi - lo, 512):
                    ix = order[k:k + 512] + lo
                    enc = tok([ta[i] for i in ix], [tb[i] for i in ix], truncation=True, max_length=MAXLEN,
                              padding=True, return_tensors="pt").to(dev)
                    buf[ix - lo] = m(**enc).logits.squeeze(-1).float().cpu().numpy()
            np.save(f"{cdir}/{lo:010d}.tmp.npy", buf); os.replace(f"{cdir}/{lo:010d}.tmp.npy", f"{cdir}/{lo:010d}.npy")
            with lock:
                cnt[0] += 1
                if cnt[0] % 5 == 0 or cnt[0] == len(todo):
                    el = time.time() - t
                    log(f"  {name}: {cnt[0]}/{len(todo)} chunks, {cnt[0] * CH / el:,.0f} pairs/s, ETA "
                        f"{el / cnt[0] * (len(todo) - cnt[0]) / 60:.1f} min, GPU peaks "
                        f"{[round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(2)]} GB, RSS {rss():.1f} GB")
        del m
        torch.cuda.empty_cache()

    ths = [threading.Thread(target=work, args=(gi, todo[gi::2])) for gi in range(2)]
    [th.start() for th in ths]; [th.join() for th in ths]
    for lo, hi in chunks:
        out[lo:hi] = np.load(f"{cdir}/{lo:010d}.npy")
    np.save(path, out)
    log(f"scored {name}: {n:,} pairs in {time.time() - t:.0f}s")
    return out


def attach_ce(D, name, s1_ids, pool_ids, r1, rp):
    m = inb(D["p"].values)
    B = D[m]
    ta, tb = texts(s1_ids[B["s1"].values], pool_ids[B["pool"].values], r1, rp)
    ce = np.full(len(D), np.nan, np.float32)
    ce[m] = score_pairs(ta, tb, name)
    D["ce"] = ce
    return D


TR = attach_ce(TR, "train_entities_band", S1_IDS_TR, POOL_IDS_TR, r1_tr, rp_tr)   # CV rows in-sample: competitors only
EV = attach_ce(EV, "eval_band", S1_IDS_TR, POOL_IDS_TR, r1_tr, rp_tr)
del r1_tr, rp_tr


# =============== 3) 2nd stage (fit on 25% eval entities) ===============
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


def fmat(D):
    return np.c_[D["p"].values, D["ce"].values, D["rk"].values, D["p1"].values, D["p2"].values, D["n_band"].values,
                 D["p1"].values - D["p"].values].astype(np.float32)


TR = entity_ctx(TR, N1); EV = entity_ctx(EV, N1)
import xgboost as xgb
bpath = f"{OUT}/stage2.json"
prm = dict(objective="binary:logistic", max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9,
           min_child_weight=5, tree_method="hist", device="cuda:0", eval_metric="logloss", seed=0)
fitm = inb(EV["p"].values) & fit_ent[EV["s1"].values]
if os.path.exists(bpath):
    bst = xgb.Booster(); bst.load_model(bpath)
else:
    bst = xgb.train(prm, xgb.DMatrix(fmat(EV[fitm]), label=EV["y"].values[fitm]), 600)
    bst.save_model(bpath)
log(f"2nd stage fitted on {fitm.sum():,} band pairs of {fit_ent.sum():,} eval entities")


def p2nd(D):
    p = D["p"].values.copy(); m = inb(p)
    p[m] = bst.predict(xgb.DMatrix(fmat(D[m])))
    return p


EV["p2"] = p2nd(EV); TR["p2"] = p2nd(TR)


def pool_is_max(pool, p, n_pool):
    mx = np.full(n_pool, -np.inf, np.float32); np.maximum.at(mx, pool, p)
    idx = np.flatnonzero(p == mx[pool]); _, first = np.unique(pool[idx], return_index=True)
    v = np.zeros(len(p), bool); v[idx[first]] = True
    return v


ALL = pd.concat([EV[["s1", "pool", "p", "p2", "y"]], TR[["s1", "pool", "p", "p2", "y"]]], ignore_index=True)   # all train S1 compete
pim1, pim2 = pool_is_max(ALL["pool"].values, ALL["p"].values, NP), pool_is_max(ALL["pool"].values, ALL["p2"].values, NP)
TG = np.round(np.r_[np.arange(0.30, 0.90, 0.02), np.arange(0.90, 0.995, 0.005)], 3)


_SUB = {}


def evaluate(pcol, pim, t, excl, ent):
    key = (id(ent), pcol, id(pim))   # rows of these entities, sliced once (exclusivity was decided on ALL rows)
    if key not in _SUB:
        r = np.flatnonzero(ent[ALL["s1"].values])
        _SUB[key] = (ALL[pcol].values[r], pim[r], ALL["s1"].values[r], ALL["y"].values[r])
    p, pm, s1, y = _SUB[key]
    k = (p >= t) & (pm if excl else True)
    m = np.ones(len(p), bool)
    tp = int((k & y & m).sum()); fp = int((k & ~y & m).sum()); fn = int(n_true[ent].sum()) - tp
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(f05=round(macro_f05(s1[k], y[k], n_true, ent), 5), precision=round(P, 5), recall=round(R, 5), fp=fp, fn=fn,
                n_pred=int((k & m).sum()), india=round(macro_f05(s1[k], y[k], n_true, ent & (cc == 0)), 5),
                us=round(macro_f05(s1[k], y[k], n_true, ent & (cc == 1)), 5))


res, rules = {}, {}
for name, pcol, pim in (("baseline stage-1", "p", pim1), ("stage-1 + CE 2nd stage", "p2", pim2)):
    best = max(((t, e) for t in TG for e in (False, True)), key=lambda te: evaluate(pcol, pim, te[0], te[1], fit_ent)["f05"])
    rules[name] = dict(thr=float(best[0]), exclusive=bool(best[1]))
    for split, ent in (("REPORT 75% eval", rep_ent), ("holdout", ho_ent)):
        res[(name, split)] = dict(rule=str(rules[name]), **evaluate(pcol, pim, best[0], best[1], ent))
R = pd.DataFrame(res).T
log("RESULT (rule chosen on the 25% fit entities only):\n" + R.to_string())
R.to_csv(f"{OUT}/result.tsv", sep="\t")
b, n_ = res[("baseline stage-1", "REPORT 75% eval")], res[("stage-1 + CE 2nd stage", "REPORT 75% eval")]
with open(f"{W}/experiments/experiments.tsv", "a") as f:
    f.write(f"ce_full_2nd_stage\teval75\t{n_['f05']}\t{round(n_['f05'] - b['f05'], 5)}\t{n_['precision']}\t{n_['recall']}\t"
            f"{n_['india']}\t{n_['us']}\t{n_['n_pred']}\t{n_['fp']}\t{n_['fn']}\t0.9742\t{time.time() - T0:.0f}\t"
            f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f}\t7\tbaseline(same entities, best rule)={b['f05']} {rules}\n")
log(f"eval-75% delta vs baseline (same entities, each with its best rule from the fit set): {n_['f05'] - b['f05']:+.5f}")
del ALL, EV, TR

# =============== 4) test -> submission_2 ===============
tdir = glob.glob(f"{C}/retrieval/test_*")[0]
TE = pq.read_table(glob.glob(f"{tdir}/scores_*/final_preds.parquet")[0], columns=["s1_pos", "pool_pos", "p"]).to_pandas()
TE = pd.DataFrame({"s1": TE["s1_pos"].values.astype(np.int64), "pool": TE["pool_pos"].values.astype(np.int64),
                   "p": TE["p"].values.astype(np.float32)})
fr = glob.glob(f"{W}/scratch/frames/*")[0]
t_s1 = pq.read_table(f"{fr}/test_s1_norm.parquet", columns=["entity_id"]).column(0).to_numpy(zero_copy_only=False)
t_pool = pq.read_table(f"{fr}/test_pool_norm.parquet", columns=["entity_id"]).column(0).to_numpy(zero_copy_only=False)
r1_te, rp_te = raw("test", 1), pd.concat([raw("test", 2), raw("test", 3)])
TE = attach_ce(TE, "test_band", t_s1, t_pool, r1_te, rp_te)
TE = entity_ctx(TE, len(t_s1))
TE["p2"] = p2nd(TE)
rule = rules["stage-1 + CE 2nd stage"]
keep = TE["p2"].values >= rule["thr"]
if rule["exclusive"]:
    keep &= pool_is_max(TE["pool"].values, TE["p2"].values, len(t_pool))
log(f"test: {len(TE):,} candidate pairs, {keep.sum():,} predicted matches ({keep.sum() / len(t_s1):.3f} per S1), rule {rule}")

# submission files (same format as submission_1), strict checks
os.makedirs(SUB, exist_ok=True)
file_order = pq.read_table(f"{W}/aligned_inputs/data_parquet/test_source1_final.parquet", columns=["entity_id"]).column(0).to_pylist()
assert file_order == list(t_s1), "test S1 order differs from test_source1"
o = np.lexsort((-TE["p2"].values, TE["s1"].values))
s1s, ids, ks = TE["s1"].values[o], t_pool[TE["pool"].values[o]], keep[o]
bounds = np.searchsorted(s1s, np.arange(len(t_s1) + 1))
with open(f"{SUB}/matching_results.tsv.tmp", "w") as fm, open(f"{SUB}/candidate_pairs.tsv.tmp", "w") as fc:
    fm.write("source1_entity_id\tmatched_entity_ids\n"); fc.write("source1_entity_id\tcandidate_entity_ids\n")
    for i in range(len(t_s1)):
        a_, b_ = bounds[i], bounds[i + 1]
        fm.write(f"{t_s1[i]}\t{','.join(ids[a_:b_][ks[a_:b_]])}\n"); fc.write(f"{t_s1[i]}\t{','.join(ids[a_:b_])}\n")
for fn_ in ("matching_results.tsv", "candidate_pairs.tsv"):
    os.replace(f"{SUB}/{fn_}.tmp", f"{SUB}/{fn_}")
# validation
valid_pool = set(t_pool)
m = pd.read_csv(f"{SUB}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
cnd = pd.read_csv(f"{SUB}/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
errs = []
if list(m.columns) != ["source1_entity_id", "matched_entity_ids"]: errs.append("matching header")
if list(cnd.columns) != ["source1_entity_id", "candidate_entity_ids"]: errs.append("candidates header")
if m["source1_entity_id"].tolist() != file_order or cnd["source1_entity_id"].tolist() != file_order: errs.append("row order / count")
used = [x for s in m["matched_entity_ids"] for x in (s.split(",") if s else [])]
if not set(used) <= valid_pool: errs.append("unknown pool ids in matches")
if rule["exclusive"] and len(used) != len(set(used)): errs.append("a pool id matched twice under the exclusive rule")
if any(not set(a.split(",") if a else []) <= set(c.split(",") if c else []) for a, c in zip(m["matched_entity_ids"], cnd["candidate_entity_ids"])):
    errs.append("match not among candidates")
sha = {fn_: hashlib.sha256(open(f"{SUB}/{fn_}", "rb").read()).hexdigest() for fn_ in ("matching_results.tsv", "candidate_pairs.tsv")}
json.dump(dict(rule=rule, stage2=bpath, ce_model=mdir, errors=errs, sha256=sha, predicted=int(keep.sum()),
               eval75=n_, eval75_baseline=b), open(f"{SUB}/submission_info.json", "w"), indent=1)
log(f"submission_2 {'VALIDATION PASSED' if not errs else 'VALIDATION FAILED ' + str(errs)}  sha256 {sha}")
log(f"done in {time.time() - T0:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
