def build_truth(gt_df, s1_df, pool_df):   # copied
    s1_index = pd.Index(s1_df["entity_id"])
    pool_index = pd.Index(pool_df["entity_id"])
    n_pool = np.int64(len(pool_df))
    rows = gt_df[gt_df["matched_entity_ids"] != ""]
    sid = np.repeat(rows["source1_entity_id"].values, rows["matched_entity_ids"].str.count(",").values + 1)
    cid = np.array([x.strip() for s in rows["matched_entity_ids"].values for x in s.split(",")], dtype=object)
    s1p = s1_index.get_indexer(sid)
    pp = pool_index.get_indexer(cid)
    ok = (s1p >= 0) & (pp >= 0)
    true_keys = np.unique(s1p[ok].astype(np.int64) * n_pool + pp[ok])
    n_true = np.bincount(s1p[ok], minlength=len(s1_df)).astype(np.int32)
    return true_keys, n_true


def isin_keys(s1_pos, pool_pos, n_pool, true_keys, chunk=20_000_000):
    """np.isin(s1_pos * n_pool + pool_pos, true_keys) in chunks (a single call on ~265M keys needs ~8 GB)."""
    out = np.empty(len(s1_pos), bool)
    tk = np.unique(true_keys)
    for a in range(0, len(s1_pos), chunk):
        k = np.asarray(s1_pos[a:a + chunk], np.int64) * np.int64(n_pool) + np.asarray(pool_pos[a:a + chunk], np.int64)
        out[a:a + chunk] = np.isin(k, tk, assume_unique=False)
    return out


def macro_f05(s1_pos_pred, tp_pred, n_true, eval_mask):   # copied
    """s1_pos_pred: entity positions of PREDICTED pairs; tp_pred: whether each predicted pair is
    true; n_true: true-match count per entity (all entities); eval_mask: entities being scored."""
    n = len(n_true)
    n_pred = np.bincount(s1_pos_pred, minlength=n).astype(np.float64)
    n_tp = np.bincount(s1_pos_pred, weights=tp_pred.astype(np.float64), minlength=n)
    t = n_true.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(n_pred > 0, n_tp / n_pred, 0.0)
        rec = np.where(t > 0, n_tp / t, 0.0)
        f = np.where(n_tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    f = np.where(t == 0, (n_pred == 0).astype(np.float64), f)
    sel = eval_mask
    return float(f[sel].mean()) if sel.any() else float("nan")


def blocking_ceiling(cands_s1_pos, y, n_true, eval_mask):   # copied
    """Pair recall of the candidate set, and the macro F_0.5 an oracle matcher would get."""
    in_eval = eval_mask[cands_s1_pos]
    found = y[in_eval].sum()
    total = n_true[eval_mask].sum()
    oracle = macro_f05(cands_s1_pos[in_eval & y], np.ones(int((in_eval & y).sum()), bool), n_true, eval_mask)
    return (found / total if total else 1.0), oracle


GLOBAL_THRESHOLD_GRID = np.round(np.concatenate([np.arange(0.10, 0.90, 0.02), np.arange(0.90, 0.996, 0.005)]), 3)
ADAPTIVE_PROB_FLOOR_GRID = np.round(np.arange(0.20, 0.96, 0.05), 3)
ADAPTIVE_GAP_MULT_GRID = [1.5, 2.0, 3.0]


class _RuleCache(dict):
    """Per-prediction-set quantities reused by every rule in a grid search, computed on first use with O(n)
    scatter-max (no groupby / lexsort over the whole set: those need ~8 GB at 265M pairs). Same values as the
    copied version: ent_max = max p of the pair's S1; pool_is_max = the pair is its pool record's best, ties to
    the lowest index (= rank 0 of within_group_rank's stable lexsort)."""
    def __init__(self, s1_pos, pool_pos, p):
        super().__init__()
        self.s1, self.pool, self.p = s1_pos, pool_pos, p

    def __missing__(self, key):
        if key == "ent_max":
            m = np.full(int(self.s1.max()) + 1 if len(self.s1) else 0, -np.inf, self.p.dtype)
            np.maximum.at(m, self.s1, self.p)
            v = m[self.s1]
        elif key == "pool_is_max":
            m = np.full(int(self.pool.max()) + 1 if len(self.pool) else 0, -np.inf, self.p.dtype)
            np.maximum.at(m, self.pool, self.p)
            idx = np.flatnonzero(self.p == m[self.pool])
            _, first = np.unique(self.pool[idx], return_index=True)
            v = np.zeros(len(self.p), bool)
            v[idx[first]] = True
        else:
            raise KeyError(key)
        self[key] = v
        return v


def rule_cache(s1_pos, pool_pos, p):
    return _RuleCache(s1_pos, pool_pos, p)


def apply_rule(s1_pos, pool_pos, p, rule, cache=None):   # copied
    """Returns a boolean mask over pairs. rule: dict(kind, thr | floor+gap_mult, exclusive)."""
    cache = cache if cache is not None else rule_cache(s1_pos, pool_pos, p)
    if rule["kind"] == "global":
        keep = p >= rule["thr"]
        if rule.get("exclusive"):
            # thresholding is monotone in p, so a pool record's best kept pair is its best pair overall
            return keep & cache["pool_is_max"]
        return keep
    keep = (p >= rule["floor"]) & (p * rule["gap_mult"] >= cache["ent_max"])
    if rule.get("exclusive"):
        idx = np.flatnonzero(keep)
        if len(idx):
            # among kept pairs, each pool record keeps only its highest-probability S1 entity
            order = idx[np.lexsort((-p[idx], pool_pos[idx]))]
            first = np.r_[True, pool_pos[order][1:] != pool_pos[order][:-1]]
            keep = np.zeros_like(keep)
            keep[order[first]] = True
    return keep


def candidate_rules():   # copied
    rules = [dict(kind="global", thr=float(t)) for t in GLOBAL_THRESHOLD_GRID]
    rules += [dict(kind="relative", floor=float(fl), gap_mult=float(g))
              for fl in ADAPTIVE_PROB_FLOOR_GRID for g in ADAPTIVE_GAP_MULT_GRID]
    return rules + [dict(r, exclusive=True) for r in rules]


def search_rule(s1_pos, pool_pos, p, y, n_true, eval_mask, label=""):   # copied
    """Grid-search decision rules on (held-out) predictions; returns (best_rule, best_score, table)."""
    rows = []
    cache = rule_cache(s1_pos, pool_pos, p)
    for r in candidate_rules():
        keep = apply_rule(s1_pos, pool_pos, p, r, cache)
        rows.append((r, macro_f05(s1_pos[keep], y[keep], n_true, eval_mask)))
    rows.sort(key=lambda x: -x[1])
    best_rule, best = rows[0]
    best_global = max((s for r, s in rows if r["kind"] == "global" and not r.get("exclusive")), default=float("nan"))
    print(f"  [{label}] best rule {best_rule} -> F0.5={best:.4f}   (best plain global threshold: {best_global:.4f})")
    return best_rule, best, rows


def apply_rule_v3(s1_pos, pool_pos, is_s3, p, rule, cache=None):
    if rule["kind"] == "per_source":
        keep = p >= np.where(is_s3, rule["thr_s3"], rule["thr_s2"])
        if rule.get("exclusive"):   # a pool record has one source -> still monotone per record
            keep &= (cache if cache is not None else rule_cache(s1_pos, pool_pos, p))["pool_is_max"]
        return keep
    return apply_rule(s1_pos, pool_pos, p, rule, cache)


_RULE_COMPLEXITY = {"global": 0, "per_source": 2, "relative": 4}


def rule_complexity(rule):
    return _RULE_COMPLEXITY[rule["kind"]] + int(bool(rule.get("exclusive")))


def select_rule(rows, tol):
    """The simplest rule whose score is within `tol` of the best; ties broken by score."""
    best = max(s for _, s in rows)
    eligible = [(r, s) for r, s in rows if s >= best - tol]
    return min(eligible, key=lambda t: (rule_complexity(t[0]), -t[1]))


def search_rule_v3(s1_pos, pool_pos, is_s3, p, y, n_true, eval_mask, label=""):
    """Copied rule families plus per-source thresholds; the simplest rule within RULE_SIMPLICITY_TOL wins."""
    best_rule, best, rows = search_rule(s1_pos, pool_pos, p, y, n_true, eval_mask, label=label)
    cache = rule_cache(s1_pos, pool_pos, p)

    def _score(r):
        keep = apply_rule_v3(s1_pos, pool_pos, is_s3, p, r, cache)
        return macro_f05(s1_pos[keep], y[keep], n_true, eval_mask)

    for excl in (False, True):
        g = [(r, s) for r, s in rows if r["kind"] == "global" and bool(r.get("exclusive")) == excl]
        r0, s0 = max(g, key=lambda t: t[1])
        cur, cur_s = dict(kind="per_source", thr_s2=r0["thr"], thr_s3=r0["thr"], exclusive=excl), s0
        for _ in range(2):
            for key in ("thr_s2", "thr_s3"):
                for t in GLOBAL_THRESHOLD_GRID:
                    cand = dict(cur, **{key: float(t)})
                    sc = _score(cand)
                    if sc > cur_s + 1e-9:
                        cur, cur_s = cand, sc
        rows.append((cur, cur_s))
    rows.sort(key=lambda x: -x[1])
    chosen, chosen_s = select_rule(rows, cfg.RULE_SIMPLICITY_TOL)
    print(f"  [{label}] best-scoring rule {rows[0][0]} -> CV macro F0.5={rows[0][1]:.4f}\n"
          f"  [{label}] SELECTED (simplest within {cfg.RULE_SIMPLICITY_TOL}) {chosen} -> CV macro F0.5={chosen_s:.4f}")
    return chosen, chosen_s, rows


def eval_rule(s1_pos, pool_pos, is_s3, p, y, rule, n_true, eval_mask, label=""):
    """Competition macro F0.5 plus micro P/R/F1, TP/FP/FN and breakdowns (P/R/F1 reported, F0.5 optimized)."""
    keep = apply_rule_v3(s1_pos, pool_pos, is_s3, p, rule)
    in_eval = eval_mask[s1_pos]
    tp = int((keep & y & in_eval).sum())
    fp = int((keep & ~y & in_eval).sum())
    fn = int(n_true[eval_mask].sum()) - tp
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    res = dict(label=label, rule=rule, entities=int(eval_mask.sum()),
               macro_f05=round(macro_f05(s1_pos[keep], y[keep], n_true, eval_mask), 5),
               micro_precision=round(P, 5), micro_recall=round(R, 5),
               micro_f1=round(2 * P * R / max(P + R, 1e-12), 5), micro_f05=round(1.25 * P * R / max(0.25 * P + R, 1e-12), 5),
               tp=tp, fp=fp, fn=fn, pred_matches_per_entity=round(float(keep[in_eval].sum()) / max(int(eval_mask.sum()), 1), 3),
               true_matches_per_entity=round(float(n_true[eval_mask].mean()), 3))
    country = FR.s1["country_norm"].values
    res["by_country_macro_f05"] = {str(c): round(macro_f05(s1_pos[keep], y[keep], n_true, eval_mask & (country == c)), 4)
                                   for c in sorted(set(country[eval_mask]))}
    nl = FR.pool["non_latin"].values[pool_pos] == 1
    for name, groups in [("by_source", (("S2", ~is_s3), ("S3", is_s3))),
                         ("by_cand_script", (("latin", ~nl), ("non-latin", nl)))]:   # boolean masks, not string arrays
        d = {}
        for gv, gm in groups:
            if not gm.any():
                continue
            m = in_eval & gm
            tpg, fpg = int((keep & y & m).sum()), int((keep & ~y & m).sum())
            pos_g = int((y & m).sum())
            d[gv] = dict(precision=round(tpg / max(tpg + fpg, 1), 4), recall_of_candidates=round(tpg / max(pos_g, 1), 4))
        res[name] = d
    print(f"  [{label}] macro F0.5={res['macro_f05']:.4f}  micro P={P:.4f} R={R:.4f} F1={res['micro_f1']:.4f}  "
          f"TP={tp:,} FP={fp:,} FN={fn:,}  pred/entity={res['pred_matches_per_entity']} (true {res['true_matches_per_entity']})")
    return res