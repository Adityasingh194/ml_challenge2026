def _dedupe_best_rank(frames, n_pool):
    """Concatenate retriever frames and keep each (s1, pool) pair once, at its best rank."""
    E = pd.concat(frames, ignore_index=True)
    E["key"] = E["s1_pos"].values.astype(np.int64) * n_pool + E["pool_pos"].values
    E = E.sort_values(["key", "rank"], kind="stable").drop_duplicates("key")
    return E


def _count_stats(counts):
    return dict(mean=round(float(counts.mean()), 2) if len(counts) else 0.0,
                p50=float(np.percentile(counts, 50)) if len(counts) else 0.0,
                p95=float(np.percentile(counts, 95)) if len(counts) else 0.0)


RECALL_BLOCK_PAIRS = 5_000_000   # retriever rows per S1 block when the diagnostic-depth union is evaluated


def _deep_union_stats(retr, n_pool, true_keys, eval_mask, Ks):
    """Everything recall_report needs from the diagnostic-depth union, accumulated over contiguous S1 blocks,
    so the full union (45M rows x 32 columns, ~17 GB while being built) is never materialized. Union rows and
    their pos_in_entity are per-S1 quantities, so each block's rows are exactly the full union's rows for it."""
    n_s1 = len(eval_mask)
    hits = {K: 0 for K in Ks}
    hits_all = xc = 0
    found = {K: np.zeros(n_s1, np.int32) for K in Ks}   # true in-cap candidates per S1 (eval entities only)
    cnt = np.zeros(n_s1, np.int64)                     # union candidates per S1
    q = np.flatnonzero(eval_mask)
    n_rows = sum(len(d) for d in retr.values())
    n_blocks = max(1, int(np.ceil(n_rows / RECALL_BLOCK_PAIRS)))
    edges = np.unique(np.r_[q[np.linspace(0, len(q), n_blocks, endpoint=False).astype(np.int64)], n_s1]) if len(q) else [0, n_s1]
    edges[0] = 0
    for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        part = {}
        for r, d in retr.items():
            s = d["s1_pos"].values
            part[r] = d[(s >= lo) & (s < hi)]
        W = build_union(part, n_pool)
        del part
        w_s1, w_pos = W["s1_pos"].values, W["pos_in_entity"].values
        t = np.isin(w_s1 * n_pool + W["pool_pos"].values, true_keys) & eval_mask[w_s1]
        for K in Ks:
            m = t & (w_pos < K)
            hits[K] += int(m.sum())
            found[K] += np.bincount(w_s1[m], minlength=n_s1).astype(np.int32)
        hits_all += int(t.sum())
        xc += cross_country_count(w_s1, W["pool_pos"].values)
        cnt += np.bincount(w_s1, minlength=n_s1)
        if bi == 0:
            mem_mark(f"recall_report: diagnostic-depth union block 1/{len(edges) - 1}", W)
        del W, w_s1, w_pos, t
    gc.collect()
    return dict(hits=hits, hits_all=hits_all, xc=xc, found=found, cnt=cnt, n_rows=int(cnt.sum()))


def recall_report(retr, n_pool, true_keys, n_true, eval_mask, label):
    Ks = cfg.RECALL_KS
    total = int(n_true[eval_mask].sum())
    n_pool = np.int64(n_pool)
    rows = []
    mem_mark("recall_report: retriever lists", retr)

    def _row(name, s1, keys, pos, pool):
        t = np.isin(keys, true_keys) & eval_mask[s1]
        row = dict(retriever=name)
        for K in Ks:
            row[f"R@{K}"] = round(float(t[pos < K].sum()) / max(total, 1), 4)
        row["R@all"] = round(float(t.sum()) / max(total, 1), 4)
        row["cross_country"] = cross_country_count(s1, pool)
        return row

    for r in RETRIEVERS:
        d = retr.get(r)
        if d is None or not len(d):
            continue
        s1 = d["s1_pos"].values
        keys = s1.astype(np.int64) * n_pool + d["pool_pos"].values
        ordr = d["order_rank"].values if "order_rank" in d.columns else d["rank"].values
        rows.append(_row(r, s1, keys, group_positions(s1, ordr, d["score"].values), d["pool_pos"].values))
    ex = [retr[k] for k in EXACT_KEYS if k in retr and len(retr[k])]
    if ex:
        E = _dedupe_best_rank(ex, n_pool)
        rows.append(_row("exact (any key)", E["s1_pos"].values, E["key"].values,
                         group_positions(E["s1_pos"].values, E["rank"].values, E["score"].values), E["pool_pos"].values))
        del E
    deep = _deep_union_stats(retr, n_pool, true_keys, eval_mask, Ks)
    rows.append(dict(retriever="UNION (diagnostic depth)",
                     **{f"R@{K}": round(float(deep["hits"][K]) / max(total, 1), 4) for K in Ks},
                     **{"R@all": round(float(deep["hits_all"]) / max(total, 1), 4), "cross_country": deep["xc"]}))
    prod = truncate_retr(retr)
    P = build_union(prod, n_pool)
    mem_mark("recall_report: configured-top-K union", P)
    pkeys = P["s1_pos"].values * n_pool + P["pool_pos"].values
    rows.append(_row("UNION (configured top-Ks)", P["s1_pos"].values, pkeys, P["pos_in_entity"].values, P["pool_pos"].values))
    table = pd.DataFrame(rows)
    n_cross = int(table.loc[table["retriever"].isin(RETRIEVERS), "cross_country"].sum())
    if n_cross and not cfg.ENABLE_GLOBAL_FALLBACK:
        raise RuntimeError(f"country partition violated: {n_cross:,} retrieved candidates are from another country "
                           f"({table[['retriever', 'cross_country']].to_dict('records')})")
    # ---- candidate caps: recall + candidates/query at every cap, configured top-Ks and diagnostic depth ----
    cap_rows = []
    w_s1 = P["s1_pos"].values
    tw = np.isin(pkeys, true_keys)
    cnt = np.bincount(w_s1, minlength=len(n_true))[eval_mask]
    for K in Ks:
        m = P["pos_in_entity"].values < K
        _, orc = blocking_ceiling(w_s1[m], tw[m], n_true, eval_mask)
        cap_rows.append(dict(union="configured top-Ks", cap=K, recall=round(float((tw[m] & eval_mask[w_s1[m]]).sum()) / max(total, 1), 4),
                             oracle_macro_f05=round(orc, 4), **{f"cands_{k}": v for k, v in _count_stats(np.minimum(cnt, K)).items()}))
    cnt = deep["cnt"][eval_mask]
    for K in Ks:   # oracle = every true in-cap candidate predicted: macro_f05 needs only the per-S1 counts
        s1_found = np.repeat(np.arange(len(n_true)), deep["found"][K])
        orc = macro_f05(s1_found, np.ones(len(s1_found), bool), n_true, eval_mask)
        cap_rows.append(dict(union="diagnostic depth", cap=K, recall=round(float(deep["hits"][K]) / max(total, 1), 4),
                             oracle_macro_f05=round(orc, 4), **{f"cands_{k}": v for k, v in _count_stats(np.minimum(cnt, K)).items()}))
    del deep, w_s1, tw
    gc.collect()
    # ---- reverse K: is REV_TOP_K deep enough? ----
    rev_rows = []
    d = retr.get("dense_rev")
    if d is not None and len(d):
        dr = d["rank"].values
        t_rev = np.isin(d["s1_pos"].values.astype(np.int64) * n_pool + d["pool_pos"].values, true_keys) & eval_mask[d["s1_pos"].values]
        searched = int(dr.max()) + 1
        for k in sorted({1, 3, 5, 10, 20, 50, 100, cfg.REV_TOP_K}):
            if k > searched:
                continue
            Pk = build_union(truncate_retr(retr, {"dense_rev": k}), n_pool, cap=cfg.MAX_CANDIDATES)
            yk = np.isin(Pk["s1_pos"].values * n_pool + Pk["pool_pos"].values, true_keys) & eval_mask[Pk["s1_pos"].values]
            rev_rows.append(dict(rev_k=k, reverse_recall=round(float(t_rev[dr < k].sum()) / max(total, 1), 4),
                                 union_recall_at_cap=round(float(yk.sum()) / max(total, 1), 4),
                                 configured=k == cfg.REV_TOP_K))
            del Pk
    capm = P["pos_in_entity"].values < cfg.MAX_CANDIDATES
    Pc = P[capm]
    yc = np.isin(pkeys[capm], true_keys)
    rec_cap = float((yc & eval_mask[Pc["s1_pos"].values]).sum()) / max(total, 1)
    pair_rec, oracle = blocking_ceiling(Pc["s1_pos"].values, yc, n_true, eval_mask)
    counts = np.bincount(Pc["s1_pos"].values, minlength=len(n_true))[eval_mask]
    found = np.bincount(Pc["s1_pos"].values[yc], minlength=len(n_true))
    has = eval_mask & (n_true > 0)
    loo = {}
    for r in list(prod):
        Pr = build_union({k: v for k, v in prod.items() if k != r}, n_pool, cap=cfg.MAX_CANDIDATES)
        yr = np.isin(Pr["s1_pos"].values * n_pool + Pr["pool_pos"].values, true_keys) & eval_mask[Pr["s1_pos"].values]
        loo[r] = round(rec_cap - float(yr.sum()) / max(total, 1), 5)
    # breakdowns over TRUE pairs (denominators include matches blocking missed)
    t_s1, t_pool = true_keys // n_pool, true_keys % n_pool
    t_eval = eval_mask[t_s1]
    t_found = np.isin(true_keys, pkeys[capm])
    same_country = float((FR.s1_cc[t_s1] == FR.pool_cc[t_pool])[t_eval].mean()) if t_eval.any() else float("nan")
    brk = {}
    for name, grp in [("country", FR.s1["country_norm"].values[t_s1]), ("source", FR.pool["pool_source"].values[t_pool]),
                      ("cand_script", np.where(FR.pool["non_latin"].values[t_pool] == 1, "non-latin", "latin"))]:
        brk[name] = {str(g): dict(true_pairs=int((t_eval & (grp == g)).sum()),
                                  recall=round(float((t_found & t_eval & (grp == g)).sum()) / max(int((t_eval & (grp == g)).sum()), 1), 4))
                     for g in sorted(set(grp[t_eval]))}
    res = dict(label=label, eval_entities=int(eval_mask.sum()), true_pairs=total, table=table.to_dict("records"),
               union_recall_at_cap=round(rec_cap, 4), max_candidates=cfg.MAX_CANDIDATES,
               oracle_macro_f05=round(oracle, 4), entity_all_found_rate=round(float((found[has] >= n_true[has]).mean()) if has.any() else float("nan"), 4),
               cands_per_query=dict(_count_stats(counts), max=int(counts.max()) if len(counts) else 0,
                                    zero_frac=round(float((counts == 0).mean()), 4)),
               cap_table=cap_rows, reverse_k_table=rev_rows, cross_country_candidates=n_cross,
               same_country_true_pair_frac=round(same_country, 6),
               leave_one_out_recall_drop=loo, breakdown=brk, retr_cfg_fp=retr_cfg_fp())
    print(f"\n==== candidate recall [{label}]  eval entities={res['eval_entities']:,}  true pairs={total:,} ====")
    print(table.to_string(index=False))
    print(f"configured union @ MAX_CANDIDATES={cfg.MAX_CANDIDATES}: recall={rec_cap:.4f}  oracle macro F0.5={oracle:.4f}  "
          f"entities with all matches found={res['entity_all_found_rate']}")
    print(f"candidates/query: {res['cands_per_query']}")
    print(f"country buckets: cross-country candidates={n_cross:,}   true pairs inside the same country bucket="
          f"{same_country:.4%}")
    print("candidate caps (recall, oracle F0.5 and candidates/query at each cap):\n"
          + pd.DataFrame(cap_rows).to_string(index=False))
    if rev_rows:
        print(f"reverse K (searched depth {searched}; configured REV_TOP_K={cfg.REV_TOP_K}):\n"
              + pd.DataFrame(rev_rows).to_string(index=False))
    print(f"leave-one-out recall drop (~0 -> retriever could be disabled at test time): {loo}")
    for k, v in brk.items():
        print(f"  by {k}: {v}")
    return res