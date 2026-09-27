import xgboost as xgb

_XGB_V2 = _vt(xgb.__version__) >= (2, 0, 0)


def xgb_params(params):   # copied (device handling extended for xgboost<2)
    p = dict(objective="binary:logistic", eval_metric="logloss", max_depth=params["max_depth"],
             learning_rate=params["learning_rate"], subsample=0.8, colsample_bytree=0.8,
             min_child_weight=5, reg_lambda=1.0, max_bin=256, seed=cfg.RANDOM_SEED, nthread=cfg.NUM_WORKERS)
    if _XGB_V2:
        p.update(tree_method="hist", device=("cuda" if DEVICE == "cuda" else "cpu"))
    else:
        p.update(tree_method="gpu_hist" if DEVICE == "cuda" else "hist")
    return p


def fit_weights(row_mask, y, seed):   # copied (globals -> args)
    """1.0 for rows the model trains on (all positives + sampled negatives), 0.0 otherwise."""
    w = row_mask.astype(np.float32)
    if cfg.GBDT_TRAIN_NEG_FRAC < 1.0:
        neg = np.flatnonzero(row_mask & ~y)
        drop = np.random.default_rng(seed).choice(neg, size=int(len(neg) * (1 - cfg.GBDT_TRAIN_NEG_FRAC)), replace=False)
        w[drop] = 0.0
    return w


def fit_gbdt(dmat, params, row_mask, y, seed):   # copied
    dmat.set_weight(fit_weights(row_mask, y, seed))
    return xgb.train(xgb_params(params), dmat, num_boost_round=params["n_estimators"])


def gbdt_predict(booster, X, chunk=5_000_000):   # copied
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        part = X.iloc[s:s + chunk].values if hasattr(X, "iloc") else X[s:s + chunk]
        out[s:s + chunk] = booster.inplace_predict(np.ascontiguousarray(part, dtype=np.float32))
    return out


def entity_folds():
    """Per S1 row: 0..N_FOLDS-1 = CV fold, -1 = nested holdout (both = TRAINING entities, a
    GBDT_TRAIN_ENTITY_FRACTION hash sample), -3 = train EVALUATION entity (never trained on), -2 = test row."""
    ids = ids_to_int(FR.s1["entity_id"].values)
    u = hash_u01(ids, SALT["fold"])
    h = cfg.NESTED_HOLDOUT_FRAC
    fold = np.where(u < h, -1, np.minimum(((u - h) / (1 - h) * cfg.N_FOLDS).astype(np.int64), cfg.N_FOLDS - 1))
    fold[hash_u01(ids, SALT["gbdt"]) >= cfg.GBDT_TRAIN_ENTITY_FRACTION] = -3
    fold[FR.n_tr_s1:] = -2
    return fold


MODEL_KEYS = ["NESTED_HOLDOUT_FRAC", "N_FOLDS", "GBDT_TRAIN_NEG_FRAC", "GBDT_PARAM_GRID", "RANDOM_SEED", "MAX_CANDIDATES",
              "RULE_SIMPLICITY_TOL", "GBDT_TRAIN_ENTITY_FRACTION", "GBDT_GRID_SELECT_FRACTION"]


def model_fp(ctx):
    return cfg_fp(MODEL_KEYS, extra=[ctx.fp, FEATURE_COLS, MODEL_CODE_VERSION])


def leakage_checks(s1p, pp, y, fold):
    """Label-leakage audit of a training set (entity-level split, exclusivity, evaluation entities, test rows)."""
    tr = np.arange(FR.n_tr_s1)
    txt = pd.Series(FR.s1["name_clean"].values[tr] + "|" + FR.s1["addr_clean"].values[tr])
    f_ = fold[tr]
    dup_groups = pd.DataFrame({"t": txt.values[f_ >= -1], "f": f_[f_ >= -1]}).groupby("t")["f"].nunique()
    pos_per_pool = np.bincount(pp[y], minlength=len(FR.pool))
    res = dict(
        s1_entities_split_across_folds=0,   # by construction: every row of an S1 entity gets that entity's fold
        s1_exact_text_duplicates_straddling_folds=int((dup_groups > 1).sum()),
        pool_records_positive_for_more_than_one_s1=int((pos_per_pool > 1).sum()),
        evaluation_entity_rows_in_training_set=int((fold[s1p] == -3).sum()),
        test_s1_rows_in_training_set=int((s1p >= FR.n_tr_s1).sum()))
    assert res["test_s1_rows_in_training_set"] == 0, "test Source-1 rows in a training set"
    assert res["evaluation_entity_rows_in_training_set"] == 0, "evaluation entities in the training set"
    assert res["pool_records_positive_for_more_than_one_s1"] == 0, "a pool record is labelled positive for two S1"
    log(f"  leakage checks: {res}")
    return res


def _cv_oof(dmat, params, row_fold, rows, y, rel):
    """Out-of-fold predictions over `rows` (a boolean row mask), checkpointed at `rel` (one .npy per fold)."""
    oof = np.full(len(y), np.nan, dtype=np.float32)
    p_ = locate(rel)
    if p_:
        return np.load(p_)
    for f in range(cfg.N_FOLDS):
        frel = rel.replace(".npy", f"_fold{f}.npy")
        va = rows & (row_fold == f)
        fp_ = locate(frel)
        if fp_:
            oof[va] = np.load(fp_)[va]
            continue
        tr = rows & (row_fold >= 0) & (row_fold != f)
        if not tr.any() or not va.any():
            continue
        t0 = time.time()
        booster = fit_gbdt(dmat, params, tr, y, cfg.RANDOM_SEED + f)
        oof[va] = booster.predict(dmat)[va]
        save_npy(frel, oof)   # checkpoint per fold
        log(f"    fold {f + 1}/{cfg.N_FOLDS}: {int(tr.sum()):,} train rows, {int(va.sum()):,} scored in {time.time() - t0:.0f}s")
        del booster
        gc.collect()
    save_npy(rel, oof)
    return oof


def _best_global(s1p, y, p, n_true, mask_ent, rows):
    return max(macro_f05(s1p[rows][p[rows] >= t], y[rows][p[rows] >= t], n_true, mask_ent) for t in GLOBAL_THRESHOLD_GRID)


class TrainBlocks:
    """The training feature matrix as its checkpointed parquet blocks (rows in block order = the order of y)."""
    def __init__(self, paths, n):
        self.paths, self.n = list(paths), int(n)

    def __len__(self):
        return self.n


class _BlockIter(xgb.DataIter):
    """Feeds TrainBlocks to xgb.QuantileDMatrix one block at a time (optionally only the rows in `mask`)."""
    def __init__(self, blocks, y, mask=None):
        self.blocks, self.y, self.mask, self.i, self.a = blocks, y, mask, 0, 0
        super().__init__(release_data=True)

    def next(self, input_data):
        if self.i >= len(self.blocks.paths):
            return 0
        t = pq.read_table(self.blocks.paths[self.i], columns=FEATURE_COLS)
        b = self.a + t.num_rows
        Xb = np.empty((t.num_rows, len(FEATURE_COLS)), np.float32)
        for j, c in enumerate(FEATURE_COLS):
            Xb[:, j] = t.column(c).to_numpy()
        yb = self.y[self.a:b].astype(np.float32)
        if self.mask is not None:
            m = self.mask[self.a:b]
            Xb, yb = Xb[m], yb[m]
        self.i, self.a = self.i + 1, b
        input_data(data=Xb, label=yb)
        return 1

    def reset(self):
        self.i, self.a = 0, 0


def _qdm(X, y, mask=None, ref=None):
    """QuantileDMatrix from an in-memory matrix or, for TrainBlocks, streamed block by block."""
    if isinstance(X, TrainBlocks):
        return xgb.QuantileDMatrix(_BlockIter(X, y, mask), max_bin=256, ref=ref)
    Xm, ym = (X, y) if mask is None else (X[mask], y[mask])
    return xgb.QuantileDMatrix(Xm, label=ym.astype(np.float32), max_bin=256, ref=ref)


def train_and_calibrate(name, ctx, C, X, y, n_true):
    """C: (s1_pos, pool_pos) of the training entities' candidates; X: float32 feature matrix (FEATURE_COLS)."""
    fold = entity_folds()
    s1p, pp = C["s1_pos"].values.astype(np.int64), C["pool_pos"].values.astype(np.int64)
    is_s3 = FR.pool_is_s3[pp]
    row_fold = fold[s1p]
    cv_rows, hold_rows = row_fold >= 0, row_fold == -1
    cv_mask, hold_mask = fold >= 0, fold == -1
    mfp = model_fp(ctx)
    mrel = f"models/{name}_{mfp}"
    leak = leakage_checks(s1p, pp, y, fold)
    rec_cv = blocking_ceiling(s1p, y, n_true, cv_mask)
    rec_ho = blocking_ceiling(s1p, y, n_true, hold_mask)
    log(f"[{name}] rows={len(y):,} positives={int(y.sum()):,}  CV entities={int(cv_mask.sum()):,} "
        f"(pair recall {rec_cv[0]:.4f}, oracle F0.5 {rec_cv[1]:.4f})  holdout entities={int(hold_mask.sum()):,} "
        f"(pair recall {rec_ho[0]:.4f}, oracle F0.5 {rec_ho[1]:.4f})")
    t_train = time.time()
    dmat = _qdm(X, y)
    mem_mark("XGBoost QuantileDMatrix built")
    # ---- parameter selection on a subset of the training entities ----
    grid = cfg.GBDT_PARAM_GRID
    grid_scores = []
    if len(grid) > 1:
        g_ent = hash_u01(ids_to_int(FR.s1["entity_id"].values), SALT["grid"]) < cfg.GBDT_GRID_SELECT_FRACTION
        g_rows = cv_rows & g_ent[s1p]
        g_mask = cv_mask & g_ent
        dsub = _qdm(X, y, g_rows)
        for gi, params in enumerate(grid):
            log(f"  grid {gi} {params} on {int(g_mask.sum()):,} entities / {int(g_rows.sum()):,} rows")
            oof_s = _cv_oof(dsub, params, row_fold[g_rows], np.ones(int(g_rows.sum()), bool), y[g_rows],
                            f"{mrel}/oof_grid_{gi}.npy")
            oof_full = np.full(len(y), np.nan, np.float32)
            oof_full[g_rows] = oof_s
            grid_scores.append(_best_global(s1p, y, oof_full, n_true, g_mask, g_rows))
            log(f"  grid {gi}: OOF macro F0.5 @ best global thr = {grid_scores[-1]:.4f}")
        del dsub
        gc.collect()
        gi = int(np.argmax(grid_scores))
        track_f05("2a train: grid-select OOF (subset, best global thr)", grid_scores[gi],
                  entities=int(g_mask.sum()), params=str(grid[gi]))
    else:
        gi = 0
    params = grid[gi]
    # ---- full 5-fold CV with the chosen parameters ----
    log(f"  CV with {params} on {int(cv_mask.sum()):,} entities / {int(cv_rows.sum()):,} rows")
    oof = _cv_oof(dmat, params, row_fold, cv_rows, y, f"{mrel}/oof_{gi}.npy")
    rule, cv_score, rule_rows = search_rule_v3(s1p[cv_rows], pp[cv_rows], is_s3[cv_rows], oof[cv_rows], y[cv_rows],
                                               n_true, cv_mask, label=f"{name} OOF")
    plain = max(((r, s) for r, s in rule_rows if r["kind"] == "global" and not r.get("exclusive")), key=lambda t: t[1])
    track_f05("2b train: OOF with the chosen rule (CV)", cv_score, entities=int(cv_mask.sum()), rule=str(rule))
    cpath = cache_path(f"{mrel}/cv_model.json")
    if os.path.exists(cpath):
        cv_model = xgb.Booster()
        cv_model.load_model(cpath)
    else:
        cv_model = fit_gbdt(dmat, params, cv_rows, y, cfg.RANDOM_SEED)
        cv_model.save_model(cpath + ".tmp.json")
        os.replace(cpath + ".tmp.json", cpath)
    t_pred = time.time()
    p_hold = cv_model.predict(dmat)[hold_rows]
    rerank_s = time.time() - t_pred
    hold = eval_rule(s1p[hold_rows], pp[hold_rows], is_s3[hold_rows], p_hold, y[hold_rows], rule, n_true, hold_mask,
                     label=f"{name} NESTED HOLDOUT")
    hold_plain = hold if plain[0] == rule else eval_rule(
        s1p[hold_rows], pp[hold_rows], is_s3[hold_rows], p_hold, y[hold_rows], plain[0], n_true, hold_mask,
        label=f"{name} holdout, plain global threshold (comparison only)")
    track_f05("2c train: nested holdout (CV model + rule)", hold["macro_f05"], entities=int(hold_mask.sum()),
              precision=hold["micro_precision"], recall=hold["micro_recall"])
    gain = cv_model.get_score(importance_type="gain")
    imp = pd.Series({FEATURE_COLS[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: v for k, v in gain.items()})
    imp = (imp / imp.sum()).sort_values(ascending=False)
    print("  top feature importances (gain):\n" + imp.head(15).round(4).to_string())
    mpath = cache_path(f"{mrel}/final.json")
    if os.path.exists(mpath):
        final = xgb.Booster()
        final.load_model(mpath)
    else:
        final = fit_gbdt(dmat, params, cv_rows | hold_rows, y, cfg.RANDOM_SEED)
        final.save_model(mpath + ".tmp.json")
        os.replace(mpath + ".tmp.json", mpath)
    # out-of-sample predictions of every training row (OOF for CV rows, CV model for the holdout): train_eval
    # combines them with the evaluation entities' scores so exclusivity sees every competing S1
    oos = oof.copy()
    oos[hold_rows] = p_hold
    ctx_save_parquet(ctx, pd.DataFrame({"s1_pos": s1p.astype(np.int32), "pool_pos": pp.astype(np.int32),
                                        "p": oos.astype(np.float32)}), f"model_{mfp}", "train_entities_oos.parquet")
    meta = dict(name=name, params=params, rule=rule, cv_macro_f05=cv_score, holdout=hold, feature_cols=FEATURE_COLS,
                grid_scores=grid_scores, plain_global_rule=plain[0], plain_global_cv_macro_f05=plain[1],
                plain_global_holdout_macro_f05=hold_plain["macro_f05"], leakage_checks=leak,
                ctx_fp=ctx.fp, model_fp=mfp, blocking_cv=rec_cv, blocking_holdout=rec_ho,
                train_seconds=round(time.time() - t_train, 1), holdout_rerank_seconds=round(rerank_s, 2),
                holdout_pairs=int(hold_rows.sum()), training_pairs=int(len(y)), top_features=imp.head(25).round(5).to_dict())
    save_json(f"{mrel}/meta.json", meta)
    save_json(f"models/latest_{name}.json", dict(rel=mrel))
    save_json(f"reports/model_{name}.json", meta)
    del dmat
    gc.collect()
    return SimpleNamespace(booster=final, meta=meta)


def load_model(name, which="final"):
    """which='final' = refit on all train entities (used for test); 'cv' = CV folds only (scores the holdout)."""
    p = locate(f"models/latest_{name}.json")
    if p is None:
        return None
    with open(p) as f:
        mrel = json.load(f)["rel"]
    mp, jp = locate(f"{mrel}/{'final' if which == 'final' else 'cv_model'}.json"), locate(f"{mrel}/meta.json")
    if mp is None or jp is None:
        return None
    b = xgb.Booster()
    b.load_model(mp)
    with open(jp) as f:
        meta = json.load(f)
    if meta["feature_cols"] != FEATURE_COLS:
        raise RuntimeError(f"model {mrel} was trained with a different feature schema")
    return SimpleNamespace(booster=b, meta=meta)