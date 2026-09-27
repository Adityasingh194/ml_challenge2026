def sample_truth():
    gt = load_parquet(locate(f"{SAMPLE_REL}/gt.parquet"))
    return build_truth(gt, FR.s1, FR.pool)


def skip(stage_name, reason):
    print(f"\n[{stage_name}] SKIPPED: {reason}")
    REPORT.setdefault("skipped", {})[stage_name] = reason


def _read_report(name):
    p = locate(f"reports/{name}.json")
    if p is None:
        return None
    with open(p) as f:
        return json.load(f)


# ------------------------- weight sweep: w * cos(name) + (1 - w) * cos(addr) -------------------------
def _weight_sweep_rel():
    return f"reports/weight_sweep_{FP_SAMPLE}_{cfg_fp(['EMB_WEIGHT_SWEEP', 'DIAG_QUERY_FRACTION', 'RECALL_KS', 'DENSE_TOP_K'], extra=sorted(EMB_INFO.items()))}.json"


def resolve_emb_weight():
    """EMB_NAME_WEIGHT, or with "auto" the best w of the weight sweep (run once, then cached)."""
    if cfg.EMB_NAME_WEIGHT != "auto":
        return float(cfg.EMB_NAME_WEIGHT)
    p = locate(_weight_sweep_rel())
    if p is None:
        stage_weight_sweep()
        p = locate(_weight_sweep_rel())
    with open(p) as f:
        w = float(json.load(f)["best_w"])
    cfg.EMB_NAME_WEIGHT = w
    CFG["EMB_NAME_WEIGHT"] = w
    log(f"EMB_NAME_WEIGHT auto -> {w} (from the weight sweep)")
    return w


def stage_weight_sweep():
    """Dense forward Recall@K and dense-only oracle F0.5 on the diagnostic queries vs the FULL train pool, per w."""
    rel = _weight_sweep_rel()
    if locate(rel):
        with open(locate(rel)) as f:
            REPORT["weight_sweep"] = json.load(f)
        print("weight sweep (cached):\n" + pd.DataFrame(REPORT["weight_sweep"]["table"]).to_string(index=False)
              + f"\n  best w = {REPORT['weight_sweep']['best_w']}")
        return REPORT["weight_sweep"]
    with stage("weight_sweep") as st:
        ensure_frames("train")
        q = np.flatnonzero(diag_query_mask())
        ctx_like = SimpleNamespace(q=q, pool=np.arange(len(FR.pool)), rev=np.arange(len(FR.s1)))
        true_keys, n_true = sample_truth()
        n_pool = np.int64(len(FR.pool))
        eval_mask = np.zeros(len(FR.s1), bool)
        eval_mask[q] = True
        total = int(n_true[eval_mask].sum())
        kmax = max(cfg.RECALL_KS)
        rows = []
        for w in cfg.EMB_WEIGHT_SWEEP:
            t0 = time.time()
            v1 = EmbView([CombinedStore("train", "s1", w)], EMB_DIMS["name"] + EMB_DIMS["addr"])
            v2 = EmbView([CombinedStore("train", "pool", w)], v1.dim)
            parts = [_dense_fwd((v1, v2), qr, pr, kmax) for _, _, qr, pr, _ in partitions(ctx_like) if len(qr) and len(pr)]
            d_ = pd.concat(parts, ignore_index=True)
            s1 = d_["s1_pos"].values
            t = np.isin(s1 * n_pool + d_["pool_pos"].values, true_keys)
            pos = group_positions(s1, d_["rank"].values, d_["score"].values)
            row = dict(w=w, **{f"R@{K}": round(float(t[pos < K].sum()) / max(total, 1), 4) for K in cfg.RECALL_KS})
            m = d_["rank"].values < cfg.DENSE_TOP_K
            row[f"oracle_F05@top{cfg.DENSE_TOP_K}/src"] = round(blocking_ceiling(s1[m], t[m], n_true, eval_mask)[1], 4)
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            log(f"  w={w:<5} " + "  ".join(f"R@{K}={row[f'R@{K}']:.4f}" for K in cfg.RECALL_KS)
                + f"  oracle F0.5={row[f'oracle_F05@top{cfg.DENSE_TOP_K}/src']:.4f}  ({row['seconds']}s)")
            del d_, parts
            gc.collect()
        tab = pd.DataFrame(rows)
        best = tab.sort_values(["R@100", "R@50"], ascending=False).iloc[0]
        res = dict(table=rows, best_w=float(best["w"]), queries=int(len(q)), true_pairs=total)
        print(f"\nweight sweep on {len(q):,} diagnostic queries ({total:,} true pairs) vs the full train pool:\n"
              + tab.to_string(index=False) + f"\n  best w (max dense R@100) = {res['best_w']}")
        REPORT["weight_sweep"] = res
        save_json(rel, res)
        st.add(best_w=res["best_w"])
    return res


# ------------------------- diagnostic recall: choose the top-Ks and MAX_CANDIDATES -------------------------
def stage_diag_recall():
    resolve_emb_weight()
    ctx = make_ctx("diag")
    rel = f"reports/recall_diag_{ctx.fp}_{cfg.MAX_CANDIDATES}.json"
    if locate(rel):
        with open(locate(rel)) as f:
            rec = json.load(f)
        log(f"diagnostic recall (cached): union recall@{cfg.MAX_CANDIDATES}={rec['union_recall_at_cap']}  "
            f"oracle F0.5={rec['oracle_macro_f05']}")
    else:
        with stage("diag_recall"):
            run_retrievers(ctx)
            retr = load_retr(ctx)
            true_keys, n_true = sample_truth()
            eval_mask = np.zeros(len(FR.s1), bool)
            eval_mask[ctx.q] = True
            rec = recall_report(retr, len(FR.pool), true_keys, n_true, eval_mask,
                                f"DIAGNOSTIC: {len(ctx.q):,} train S1 vs the FULL train pool")
            del retr
            gc.collect()
            save_json(rel, rec)
    REPORT["recall_diag"] = rec
    track_f05("1a retrieval ceiling: oracle @ configured union (diag queries, full pool)", rec["oracle_macro_f05"],
              union_recall=rec["union_recall_at_cap"], max_candidates=cfg.MAX_CANDIDATES)
    caps = pd.DataFrame(rec["cap_table"])
    deep = caps[caps["union"] == "diagnostic depth"]
    ok = deep[(deep["recall"] >= 0.995) & (deep["oracle_macro_f05"] >= 0.995)]
    print(f"\nchoosing K: configured union @ MAX_CANDIDATES={cfg.MAX_CANDIDATES}: recall={rec['union_recall_at_cap']:.4f} "
          f"oracle F0.5={rec['oracle_macro_f05']:.4f}")
    print("  smallest cap on the diagnostic-depth union with recall >= 0.995 and oracle F0.5 >= 0.995: "
          + (f"{int(ok['cap'].min())}" if len(ok) else "none within the measured caps -- retrieval is the limit"))
    print(f"  leave-one-out recall drop per retriever: {rec['leave_one_out_recall_drop']}")
    return rec


# ------------------------- blocked union over a production context -------------------------
def _union_blocks(ctx):
    B = cfg.QUERY_BLOCK_ROWS
    q = ctx.q
    return [(int(q[a]), int(q[min(a + B, len(q)) - 1]) + 1) for a in range(0, len(q), B)]


def _block_union(ctx, lo, hi):
    return build_union(truncate_retr(load_retr(ctx, lo, hi)), len(FR.pool), cap=cfg.MAX_CANDIDATES)


def full_train_recall(ctx, true_keys, n_true):
    """Configured-union recall and oracle F0.5 over ALL train S1 (production depth, full pool), per country.
    Per-block counts are checkpointed."""
    rel = f"reports/recall_train_full_{ctx.fp}_{cfg.MAX_CANDIDATES}.json"
    if locate(rel):
        with open(locate(rel)) as f:
            return json.load(f)
    with stage("recall_train_full"):
        n_pool = np.int64(len(FR.pool))
        found = np.zeros(len(FR.s1), np.int32)
        cands = np.zeros(len(FR.s1), np.int32)
        xc = 0
        blocks = _union_blocks(ctx)
        for bi, (lo, hi) in enumerate(blocks):
            brel = ctx_rel(ctx, "recall_blocks", f"b{bi:05d}_{cfg.MAX_CANDIDATES}.npy")
            if locate(brel):
                arr = np.load(locate(brel))
            else:
                C = _block_union(ctx, lo, hi)
                s1 = C["s1_pos"].values
                y = np.isin(s1 * n_pool + C["pool_pos"].values, true_keys)
                xc_b = cross_country_count(s1, C["pool_pos"].values)
                arr = np.stack([np.bincount(s1[y] - lo, minlength=hi - lo), np.bincount(s1 - lo, minlength=hi - lo),
                                np.full(hi - lo, xc_b)]).astype(np.int32)
                save_npy(brel, arr)
            found[lo:hi], cands[lo:hi] = arr[0], arr[1]
            xc += int(arr[2][0]) if hi > lo else 0
            if (bi + 1) % 5 == 0 or bi + 1 == len(blocks):
                log(f"  full-train recall: block {bi + 1}/{len(blocks)}  running recall="
                    f"{found[:hi].sum() / max(n_true[:hi].sum(), 1):.4f}")
        if xc and not cfg.ENABLE_GLOBAL_FALLBACK:
            raise RuntimeError(f"country partition violated: {xc} cross-country candidates")
        s1_pos_found = np.repeat(np.arange(len(found)), found)
        allm = np.ones(len(FR.s1), bool)
        oracle = macro_f05(s1_pos_found, np.ones(len(s1_pos_found), bool), n_true, allm)
        country = FR.s1["country_norm"].values
        by_c = {}
        for c in sorted(set(country)):
            m = country == c
            by_c[str(c)] = dict(entities=int(m.sum()), recall=round(float(found[m].sum() / max(n_true[m].sum(), 1)), 4),
                                oracle_macro_f05=round(macro_f05(s1_pos_found, np.ones(len(s1_pos_found), bool), n_true, m), 4))
        rec = dict(entities=int(len(found)), true_pairs=int(n_true.sum()), union_recall_at_cap=round(float(found.sum() / max(n_true.sum(), 1)), 4),
                   oracle_macro_f05=round(oracle, 4), max_candidates=cfg.MAX_CANDIDATES,
                   cands_per_query=dict(mean=round(float(cands.mean()), 2), p50=float(np.percentile(cands, 50)),
                                        p95=float(np.percentile(cands, 95)), zero_frac=round(float((cands == 0).mean()), 4)),
                   entity_all_found_rate=round(float((found[n_true > 0] >= n_true[n_true > 0]).mean()), 4),
                   cross_country_candidates=xc, by_country=by_c, retr_cfg_fp=retr_cfg_fp())
        save_json(rel, rec)
    print(f"\n==== configured union over ALL {rec['entities']:,} train S1 (full pool) ====\n"
          f"  recall@{cfg.MAX_CANDIDATES}={rec['union_recall_at_cap']:.4f}  oracle F0.5={rec['oracle_macro_f05']:.4f}  "
          f"candidates/query={rec['cands_per_query']}\n  by country: {rec['by_country']}")
    return rec


def _trainset_dir():
    return f"trainset_{cfg_fp(['MAX_CANDIDATES', 'GBDT_TRAIN_ENTITY_FRACTION', 'NESTED_HOLDOUT_FRAC', 'N_FOLDS'], extra=[FEATURE_COLS])}"


def build_training_set(ctx, true_keys, fold):
    """Candidates + features + labels of the TRAINING entities, block by block (each block checkpointed),
    then assembled into one float32 matrix."""
    tdir = _trainset_dir()
    n_pool = np.int64(len(FR.pool))
    train_ent = fold >= -1
    blocks = _union_blocks(ctx)
    t0, n_new = time.time(), 0
    for bi, (lo, hi) in enumerate(blocks):
        if is_done(ctx_rel(ctx, tdir, f"block_{bi:05d}.parquet")):
            continue
        if hours_left() <= 0:
            raise RuntimeError("MAX_SESSION_HOURS reached while building the training set -- finished blocks are saved; run again")
        C = _block_union(ctx, lo, hi)
        C = C[train_ent[C["s1_pos"].values]].reset_index(drop=True)
        feats = features_in_chunks(C, ctx, label=f"training block {bi + 1}/{len(blocks)}")
        y = np.isin(C["s1_pos"].values * n_pool + C["pool_pos"].values, true_keys)
        out = pd.concat([C[["s1_pos", "pool_pos"]].astype(np.int32).reset_index(drop=True), feats], axis=1)
        out["label"] = y.astype(np.int8)
        ctx_save_parquet(ctx, out, tdir, f"block_{bi:05d}.parquet")
        n_new += 1
        el = time.time() - t0
        left = sum(1 for b in range(bi + 1, len(blocks)) if not is_done(ctx_rel(ctx, tdir, f"block_{b:05d}.parquet")))
        log(f"  training set block {bi + 1}/{len(blocks)}: {len(C):,} pairs ({int(y.sum()):,} positives)  "
            f"ETA {_fmt_duration(el / n_new * left)}")
        del C, feats, out
        gc.collect()
    paths = [locate(ctx_rel(ctx, tdir, f"block_{bi:05d}.parquet")) for bi in range(len(blocks))]
    n = sum(int(read_done(ctx_rel(ctx, tdir, f"block_{bi:05d}.parquet"))["rows"]) for bi in range(len(blocks)))
    y = np.empty(n, bool)
    s1p = np.empty(n, np.int64)
    pp = np.empty(n, np.int64)
    a = 0
    for p_ in paths:   # ids and labels only; the feature matrix is streamed from these blocks into XGBoost
        t = pq.read_table(p_, columns=["s1_pos", "pool_pos", "label"])
        b = a + t.num_rows
        y[a:b] = t.column("label").to_numpy().astype(bool)
        s1p[a:b] = t.column("s1_pos").to_numpy()
        pp[a:b] = t.column("pool_pos").to_numpy()
        a = b
        del t
    X = TrainBlocks(paths, n)
    log(f"training set: {n:,} pairs, {int(y.sum()):,} positives, {len(np.unique(s1p)):,} entities, "
        f"features {n * len(FEATURE_COLS) * 4 / 1e9:.1f} GB as float32 (streamed from {len(paths)} blocks, never assembled)")
    mem_mark("training set ids + labels", [y, s1p, pp], rows=n)
    return pd.DataFrame({"s1_pos": s1p, "pool_pos": pp}), X, y


def _existing_model_for(name, ctx):
    m = load_model(name)
    if m is not None and m.meta.get("ctx_fp") == ctx.fp and m.meta.get("model_fp") == model_fp(ctx):
        return m
    return None


def stage_train_model():
    resolve_emb_weight()
    ctx = make_ctx("train")
    done = _existing_model_for("train", ctx)
    if done is not None:
        log("[train_model] model already trained for this configuration -- reusing "
            "(delete models/latest_train.json to retrain)")
        rec = _read_report("recall_train_full") or {}
        return rec, done
    with stage("train_model"):
        run_retrievers(ctx)
        true_keys, n_true = sample_truth()
        rec = full_train_recall(ctx, true_keys, n_true)
        save_json("reports/recall_train_full.json", rec)
        REPORT["recall_train_full"] = rec
        track_f05("1b retrieval ceiling: oracle @ configured union (ALL train S1)", rec["oracle_macro_f05"],
                  union_recall=rec["union_recall_at_cap"])
        C, X, y = build_training_set(ctx, true_keys, entity_folds())
        res = train_and_calibrate("train", ctx, C, X, y, n_true)
        del C, X, y
        gc.collect()
    REPORT["model_train"] = res.meta
    return rec, res


# ------------------------- train evaluation: the other train entities, never trained on -------------------------
def stage_train_eval():
    resolve_emb_weight()
    ctx = make_ctx("train")
    model = _existing_model_for("train", ctx)
    if model is None:
        return skip("train_eval", "no trained model for the current configuration -- run train_model first")
    mfp = model.meta["model_fp"]
    rule = model.meta["rule"]
    rel = f"reports/train_eval_{mfp}.json"
    if locate(rel):
        with open(locate(rel)) as f:
            res = json.load(f)
        log(f"train evaluation (cached): macro F0.5={res['macro_f05']}")
        REPORT["train_eval"] = res
        evaluate_gate1(res)
        return res
    with stage("train_eval") as st:
        true_keys, n_true = sample_truth()
        n_pool = np.int64(len(FR.pool))
        fold = entity_folds()
        ev = fold == -3
        edir = f"eval_{mfp}"
        blocks = _union_blocks(ctx)
        t0, n_new = time.time(), 0
        for bi, (lo, hi) in enumerate(blocks):
            if is_done(ctx_rel(ctx, edir, f"block_{bi:05d}.parquet")):
                continue
            if hours_left() <= 0:
                raise RuntimeError("MAX_SESSION_HOURS reached during the train evaluation -- finished blocks are saved; run again")
            C = _block_union(ctx, lo, hi)
            C = C[ev[C["s1_pos"].values]].reset_index(drop=True)
            feats = features_in_chunks(C, ctx, label=f"eval block {bi + 1}/{len(blocks)}")
            p = gbdt_predict(model.booster, feats[FEATURE_COLS])
            s1p, pp = C["s1_pos"].values.astype(np.int64), C["pool_pos"].values.astype(np.int64)
            y = isin_keys(s1p, pp, n_pool, true_keys)
            out = pd.DataFrame({"s1_pos": s1p.astype(np.int32), "pool_pos": pp.astype(np.int32), "p": p.astype(np.float32),
                                "y": y.astype(np.int8)})
            ctx_save_parquet(ctx, out, edir, f"block_{bi:05d}.parquet")   # checkpoint per block
            n_new += 1
            # running estimate: rule applied inside this block only (exclusivity across blocks comes at the end)
            m_blk = np.zeros(len(FR.s1), bool)
            m_blk[lo:hi] = True
            m_blk &= ev
            keep = apply_rule_v3(s1p, pp, FR.pool_is_s3[pp], p, rule)
            f_blk = macro_f05(s1p[keep], y[keep], n_true, m_blk)
            left = sum(1 for b in range(bi + 1, len(blocks)) if not is_done(ctx_rel(ctx, edir, f"block_{b:05d}.parquet")))
            log(f"  train-eval block {bi + 1}/{len(blocks)}: {len(C):,} pairs  block macro F0.5={f_blk:.4f}  "
                f"ETA {_fmt_duration((time.time() - t0) / n_new * left)}")
            del C, feats, out
            gc.collect()
        ev_preds = pd.concat([ctx_read_parquet(ctx, edir, f"block_{bi:05d}.parquet") for bi in range(len(blocks))],
                             ignore_index=True)
        oos = ctx_read_parquet(ctx, f"model_{mfp}", "train_entities_oos.parquet")
        s1p = np.r_[ev_preds["s1_pos"].values, oos["s1_pos"].values].astype(np.int64)
        pp = np.r_[ev_preds["pool_pos"].values, oos["pool_pos"].values].astype(np.int64)
        p = np.r_[ev_preds["p"].values, oos["p"].values].astype(np.float32)
        y = isin_keys(s1p, pp, n_pool, true_keys)
        del ev_preds, oos
        gc.collect()
        # the rule is applied to ALL train S1 at once (out-of-sample scores everywhere), so exclusivity sees every
        # competing S1; the score is reported on the evaluation entities (never trained on / used for the rule)
        res = eval_rule(s1p, pp, FR.pool_is_s3[pp], p, y, rule, n_true, ev,
                        label=f"TRAIN EVALUATION ({int(ev.sum()):,} entities never trained on, full pool)")
        keep = apply_rule_v3(s1p, pp, FR.pool_is_s3[pp], p, rule)
        res["entities_all_train_macro_f05"] = round(macro_f05(s1p[keep], y[keep], n_true, np.ones(len(FR.s1), bool)), 5)
        save_json(rel, res)
        st.add(macro_f05=res["macro_f05"])
    REPORT["train_eval"] = res
    track_f05("3 train evaluation: final model + rule (80% of train, never trained on)", res["macro_f05"],
              precision=res["micro_precision"], recall=res["micro_recall"], entities=res["entities"])
    for c, v in res["by_country_macro_f05"].items():
        track_f05(f"3 train evaluation: {c}", v)
    evaluate_gate1(res)
    return res


def evaluate_gate1(eval_res):
    """GATE: may the test phase start? Written to reports/gate1.json and checked by test_inference."""
    reasons = []
    rec = _read_report("recall_train_full")
    if rec is None:
        reasons.append("full-train recall not computed (run train_model)")
    else:
        if rec["union_recall_at_cap"] < cfg.GATE1_MIN_UNION_RECALL:
            reasons.append(f"configured-union recall {rec['union_recall_at_cap']:.4f} < GATE1_MIN_UNION_RECALL "
                           f"{cfg.GATE1_MIN_UNION_RECALL}")
        if rec.get("cross_country_candidates", 0) and not cfg.ENABLE_GLOBAL_FALLBACK:
            reasons.append(f"{rec['cross_country_candidates']} cross-country candidates")
    inv = REPORT.get("country_invariant") or _read_report("country_invariant")
    if inv is None:
        reasons.append("country invariant not checked (run the sample stage)")
    elif inv["cross_country_pairs"] > cfg.COUNTRY_INVARIANT_MAX_VIOLATIONS:
        reasons.append(f"country invariant violated ({inv['cross_country_pairs']} pairs)")
    if not np.isfinite(eval_res["macro_f05"]):
        reasons.append("train-evaluation macro F0.5 is not finite")
    g = dict(passed=not reasons, reasons=reasons, retr_cfg_fp=retr_cfg_fp(),
             union_recall_at_cap=rec["union_recall_at_cap"] if rec else None,
             oracle_macro_f05=rec["oracle_macro_f05"] if rec else None,
             train_eval_macro_f05=eval_res["macro_f05"], by_country=eval_res.get("by_country_macro_f05"), t=time.time())
    save_json("reports/gate1.json", g)
    log(f"==== GATE: {'PASSED' if g['passed'] else 'FAILED ' + str(reasons)} ====  union recall={g['union_recall_at_cap']}  "
        f"oracle F0.5={g['oracle_macro_f05']}  train-eval F0.5={g['train_eval_macro_f05']}  by country={g['by_country']}")
    return g


def gate1_status():
    g = _read_report("gate1")
    if g is None:
        return False, "the gate has not been evaluated (run train_model and train_eval)"
    if g.get("retr_cfg_fp") != retr_cfg_fp():
        return False, "the gate was evaluated with different retrieval settings -- run train_eval again"
    if not g["passed"]:
        return False, f"gate failed: {g['reasons']}"
    return True, "passed"


def scale_phase_allowed(stage_name, quiet=False):
    if cfg.ENABLE_SCALE_PHASE:
        return True
    if not quiet:
        skip(stage_name, "ENABLE_SCALE_PHASE=False")
    return False


def _final_model():
    m = load_model("train")
    if m is None:
        raise RuntimeError("no trained model found -- run train_model first")
    return "train", m


def _train_countries():
    inv = REPORT.get("country_invariant") or _read_report("country_invariant")
    if inv:
        return {c for k, d in inv["records_per_country"].items() if k.startswith("train") for c in d}
    return set(FR.s1["country_norm"].values[:FR.n_tr_s1])


def _test_country_breakdown(s1_local, p, keep, n_test):
    """Per test country: candidates, predicted matches, top-1 probability distribution (no labels needed)."""
    country = FR.s1["country_norm"].values[FR.n_tr_s1:]
    counts = np.bincount(s1_local, minlength=n_test)
    mcounts = np.bincount(s1_local[keep], minlength=n_test)
    pmax = np.zeros(n_test, np.float32)
    np.maximum.at(pmax, s1_local, p.astype(np.float32))
    seen = _train_countries()
    out = {}
    for c in sorted(set(country)):
        m = country == c
        out[str(c)] = dict(s1=int(m.sum()), seen_in_train=c in seen, cands_mean=round(float(counts[m].mean()), 2),
                           zero_cand_frac=round(float((counts[m] == 0).mean()), 4),
                           matches_per_entity=round(float(mcounts[m].mean()), 3),
                           entities_with_match=round(float((mcounts[m] > 0).mean()), 4),
                           top1_p_q10=round(float(np.quantile(pmax[m], 0.1)), 4),
                           top1_p_q50=round(float(np.quantile(pmax[m], 0.5)), 4),
                           top1_p_q90=round(float(np.quantile(pmax[m], 0.9)), 4))
    print("  per test country (no labels -- compare unseen countries with the seen ones):")
    print(pd.DataFrame(out).T.to_string())
    return out


def _raw_by_id(split, s, ids, cols):
    """Raw columns for the given entity ids of one source file (reads only those rows)."""
    _, ids_int = read_ids(split, s)
    df = read_source(split, s, ["entity_id"] + cols, np.isin(ids_int, ids_to_int(np.asarray(ids, dtype=object))))
    return df.set_index("entity_id")


def export_country_audit(preds, keep):
    """Hand-labelling files for test countries unseen in training (France): a hash sample of AUDIT_N_ENTITIES
    entities with their top AUDIT_TOP_N candidates (+ any predicted match), raw text side by side, and an empty
    is_match column. Fill it with 1/0, attach the file and set AUDIT_LABELS_PATH to get a precision estimate."""
    te_country = FR.s1["country_norm"].values[FR.n_tr_s1:]
    countries = cfg.AUDIT_COUNTRIES if cfg.AUDIT_COUNTRIES is not None else sorted(set(te_country) - _train_countries())
    if not countries or cfg.AUDIT_N_ENTITIES <= 0:
        return {}
    s1p = preds["s1_pos"].values.astype(np.int64)
    out_dir = os.path.join(OUTPUT_DIR, "audit")
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    raw_cols = ["business_name", "business_address"]
    for c in countries:
        rows_c = FR.n_tr_s1 + np.flatnonzero(te_country == c)
        if not len(rows_c):
            continue
        ids_c = FR.s1["entity_id"].values[rows_c]
        pick = rows_c[np.argsort(hash_u01(ids_to_int(ids_c), SALT["audit"]), kind="stable")[:cfg.AUDIT_N_ENTITIES]]
        m = np.isin(s1p, pick)
        sub = pd.DataFrame({"s1_pos": s1p[m], "pool_pos": preds["pool_pos"].values[m].astype(np.int64),
                            "p": preds["p"].values[m], "predicted_match": keep[m].astype(np.int8)})
        sub["cand_rank"] = within_group_rank(sub["s1_pos"].values, sub["p"].values).astype(np.int32)
        sub = sub[(sub["cand_rank"] < cfg.AUDIT_TOP_N) | (sub["predicted_match"] == 1)].copy()
        sub["s1_entity_id"] = FR.s1["entity_id"].values[sub["s1_pos"].values]
        sub["cand_entity_id"] = FR.pool["entity_id"].values[sub["pool_pos"].values]
        sub["cand_source"] = FR.pool["pool_source"].values[sub["pool_pos"].values]
        r1 = _raw_by_id("test", 1, sub["s1_entity_id"].unique(), raw_cols)
        rp = pd.concat([_raw_by_id("test", s, sub.loc[sub["cand_source"] == f"S{s}", "cand_entity_id"].unique(), raw_cols)
                        for s in (2, 3)])
        for col in raw_cols:
            sub[f"s1_{col}"] = r1[col].reindex(sub["s1_entity_id"].values).values
            sub[f"cand_{col}"] = rp[col].reindex(sub["cand_entity_id"].values).values
        sub["is_match"] = ""
        cols = (["s1_entity_id"] + [f"s1_{c_}" for c_ in raw_cols] + ["cand_rank", "cand_entity_id", "cand_source"]
                + [f"cand_{c_}" for c_ in raw_cols] + ["p", "predicted_match", "is_match"])
        sub = sub.sort_values(["s1_pos", "cand_rank"])[cols]
        path = os.path.join(out_dir, f"audit_{_slug(c)}.tsv")
        sub.to_csv(path, sep="\t", index=False)
        no_cand = int(len(pick) - len(np.unique(s1p[m])))
        written[str(c)] = dict(path=path, entities=int(len(pick)), rows=int(len(sub)), entities_without_candidates=no_cand)
        print(f"  audit file for {c}: {path} ({len(pick)} entities, {len(sub)} rows; {no_cand} entities had no candidate)")
    return written


def score_audit(path):
    """Precision of predicted matches (and matches the rule missed) from a hand-labelled audit file."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    lab = df[df["is_match"].str.strip().isin(["0", "1"])]
    if not len(lab):
        return None
    y, pm = (lab["is_match"].str.strip() == "1").values, (lab["predicted_match"] == "1").values
    res = dict(path=path, labelled_rows=int(len(lab)), labelled_entities=int(lab["s1_entity_id"].nunique()),
               predicted_matches=int(pm.sum()), precision=round(float(y[pm].mean()), 4) if pm.any() else None,
               true_matches_listed=int(y.sum()), true_matches_missed_by_rule=int((y & ~pm).sum()),
               recall_within_listed_candidates=round(float((y & pm).sum() / max(int(y.sum()), 1)), 4))
    print(f"  audit {os.path.basename(path)}: {res}")
    return res


def stage_test_inference():
    if not scale_phase_allowed("test_inference"):
        return
    resolve_emb_weight()
    if load_model("train") is None:
        return skip("test_inference", "no trained model yet")
    if cfg.TEST_FRACTION >= 1.0 and not cfg.FORCE_PAST_GATE:
        ok, why = gate1_status()
        if not ok:
            raise RuntimeError(f"test inference refused: {why}. Fix it, or set FORCE_PAST_GATE=True")
    with stage("test_inference") as st:
        mname, model = _final_model()
        rule = model.meta["rule"]
        print(f"final model: {mname}  rule={rule}  holdout macro F0.5={model.meta['holdout']['macro_f05']}")
        ctx = make_ctx("test")
        sdir = f"scores_{model.meta['model_fp']}"
        q = ctx.q
        B = cfg.QUERY_BLOCK_ROWS
        blocks = [(int(q[a]), int(q[min(a + B, len(q)) - 1]) + 1) for a in range(0, len(q), B)]
        t_feat = t_pred = 0.0
        if is_done(ctx_rel(ctx, sdir, "final_preds.parquet")):
            log("test predictions for this model already exist -- reusing (summary and audit files are rebuilt)")
        else:
            run_retrievers(ctx)
            t0, n_new = time.time(), 0
            n_cached = sum(1 for bi in range(len(blocks)) if is_done(ctx_rel(ctx, sdir, f"block_{bi:05d}.parquet")))
            log(f"test inference: {n_cached}/{len(blocks)} blocks already checkpointed")
            for bi, (lo, hi) in enumerate(blocks):
                if is_done(ctx_rel(ctx, sdir, f"block_{bi:05d}.parquet")):
                    continue
                if hours_left() <= 0:
                    raise RuntimeError("MAX_SESSION_HOURS reached during test inference -- completed blocks are saved; "
                                       "attach this output and run again to continue")
                C = build_union(truncate_retr(load_retr(ctx, lo, hi)), len(FR.pool), cap=cfg.MAX_CANDIDATES)
                xc = cross_country_count(C["s1_pos"].values, C["pool_pos"].values)
                if xc and not cfg.ENABLE_GLOBAL_FALLBACK:
                    raise RuntimeError(f"country partition violated in test block {bi}: {xc} cross-country candidates")
                ta = time.time()
                feats = features_in_chunks(C, ctx, label=f"test block {bi + 1}/{len(blocks)}")
                tb = time.time()
                p = gbdt_predict(model.booster, feats[FEATURE_COLS])
                tc = time.time()
                t_feat += tb - ta
                t_pred += tc - tb
                out = pd.DataFrame({"s1_pos": C["s1_pos"].values.astype(np.int32), "pool_pos": C["pool_pos"].values.astype(np.int32),
                                    "p": p.astype(np.float32), "pos_in_entity": C["pos_in_entity"].values.astype(np.int16)})
                ctx_save_parquet(ctx, out, sdir, f"block_{bi:05d}.parquet")
                n_new += 1
                eta = (time.time() - t0) / n_new * (len(blocks) - n_cached - n_new)
                log(f"  test block {bi + 1}/{len(blocks)}: {len(C):,} pairs  features {tb - ta:.0f}s  score {tc - tb:.1f}s  "
                    f"ETA {_fmt_duration(eta)}")
            preds = pd.concat([ctx_read_parquet(ctx, sdir, f"block_{bi:05d}.parquet") for bi in range(len(blocks))],
                              ignore_index=True)
            mem_mark("test predictions (all blocks)", preds)
            s1p, pp = preds["s1_pos"].values.astype(np.int64), preds["pool_pos"].values.astype(np.int64)
            preds["match"] = apply_rule_v3(s1p, pp, FR.pool_is_s3[pp], preds["p"].values, rule)
            ctx_save_parquet(ctx, preds, sdir, "final_preds.parquet")
            save_json(f"reports/test_timing_{model.meta['model_fp']}.json",
                      dict(feature_seconds=round(t_feat, 1), rerank_seconds=round(t_pred, 1)))
        preds = ctx_read_parquet(ctx, sdir, "final_preds.parquet")
        s1p, pp = preds["s1_pos"].values.astype(np.int64), preds["pool_pos"].values.astype(np.int64)
        keep = preds["match"].values.astype(bool)
        xc = cross_country_count(s1p, pp)
        if xc and not cfg.ENABLE_GLOBAL_FALLBACK:
            raise RuntimeError(f"country partition violated: {xc} cross-country test candidates")
        n_test = len(FR.s1) - FR.n_tr_s1
        s1_local = s1p - FR.n_tr_s1
        counts = np.bincount(s1_local, minlength=n_test)
        mcounts = np.bincount(s1_local[keep], minlength=n_test)
        timing = _read_report(f"test_timing_{model.meta['model_fp']}") or {}
        info = dict(model=mname, rule=rule, test_s1=n_test, test_pool=len(ctx.pool), candidate_pairs=len(preds),
                    predicted_matches=int(keep.sum()), cross_country_candidates=xc,
                    cands_per_query=dict(mean=round(float(counts.mean()), 2), p50=float(np.percentile(counts, 50)),
                                         p95=float(np.percentile(counts, 95)), zero_frac=round(float((counts == 0).mean()), 4)),
                    matches_per_entity=round(float(mcounts.mean()), 3), entities_with_match=round(float((mcounts > 0).mean()), 4),
                    feature_seconds=timing.get("feature_seconds"), rerank_seconds=timing.get("rerank_seconds"),
                    preds_path=ctx_rel(ctx, sdir, "final_preds.parquet"))
        print(f"test: {info}")
        print("  (train ground truth averages ~3.46 matches/entity with ~5.6% singletons -- a large deviation is worth "
              "investigating before submitting)")
        info["by_country"] = _test_country_breakdown(s1_local, preds["p"].values, keep, n_test)
        info["audit_files"] = export_country_audit(preds, keep)
        REPORT["test_inference"] = info
        save_json("reports/test_inference.json", info)
        st.add(**{k: v for k, v in info.items() if isinstance(v, (int, float, str))})