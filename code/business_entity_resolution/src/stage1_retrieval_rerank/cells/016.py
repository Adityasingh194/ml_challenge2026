RETRIEVERS = ["dense", "dense_global", "dense_rev", "bm25", "tfidf"] + EXACT_KEYS
RANK_SENTINEL = 9999
# FR holds the normalized frames of ONE split at a time (train or test), never both: 12.5M rows each.
# n_tr_s1 / n_tr_pool = number of TRAIN rows at the front of the frames (all rows for train, 0 for test).
FR = SimpleNamespace(split=None, s1=None, pool=None, n_tr_s1=0, n_tr_pool=0, s1_cc=None, pool_cc=None, pool_is_s3=None)


def country_codes(s1_country, pool_country):
    """Integer country codes shared by both sides (fast bucket checks on millions of pairs)."""
    cats = pd.Index(sorted(set(s1_country) | set(pool_country)))
    return cats.get_indexer(s1_country).astype(np.int16), cats.get_indexer(pool_country).astype(np.int16)


def cross_country_count(s1_pos, pool_pos):
    """Number of (S1, pool) pairs whose countries differ -- must be 0 while retrieval is country-partitioned."""
    return int((FR.s1_cc[np.asarray(s1_pos, np.int64)] != FR.pool_cc[np.asarray(pool_pos, np.int64)]).sum())


SHARED_STR_COLS = ["country_norm", "state_norm", "city_norm", "name_suffix", "postcode", "house_no", "pool_source"]
SHARED_STR_MAX_UNIQUE = 0.05   # share only where values repeat a lot; high-cardinality columns would not shrink


def to_object_strings(df):
    """String columns as plain object arrays ("" for missing), as arrow_to_frame returns them. A frame that was
    just normalized (not read back from its parquet cache) has pandas>=3 Arrow-backed "str" columns instead:
    different types reaching the feature code than at training time, and slow to slice/pickle for the workers."""
    for c in df.columns:
        if df[c].dtype != object and pd.api.types.is_string_dtype(df[c].dtype):
            df[c] = df[c].to_numpy(dtype=object, na_value="")
    return df


def share_low_cardinality_strings(df, label):
    """Point every row of a low-cardinality object column at ONE str object per distinct value (pd.factorize).
    Same dtype (object) and identical values, so no downstream code changes; saves ~50 bytes per cell."""
    shared = []
    for c in SHARED_STR_COLS:
        if c not in df.columns or df[c].dtype != object or not len(df):
            continue
        codes, uniques = pd.factorize(df[c].values)
        if (codes < 0).any() or len(uniques) > SHARED_STR_MAX_UNIQUE * len(df):
            continue
        df[c] = np.asarray(uniques, dtype=object)[codes]
        shared.append(f"{c}({len(uniques):,})")
    gc.collect()
    mem_mark(f"frames {label}: shared string objects for {shared}", df)


def ensure_frames(split):
    assert split in ("train", "test"), split
    if FR.split == split and FR.s1 is not None:
        return FR
    with stage("frames", split=split):
        FR.s1 = FR.pool = FR.s1_cc = FR.pool_cc = FR.pool_is_s3 = None
        FR.split = None
        gc.collect()
        mem_mark("frames released")
        FR.s1 = to_object_strings(normalized_frame(split, "s1"))
        mem_mark(f"frames {split} S1 loaded", FR.s1)
        share_low_cardinality_strings(FR.s1, f"{split} S1")
        FR.pool = to_object_strings(normalized_frame(split, "pool"))
        mem_mark(f"frames {split} pool loaded", FR.pool)
        share_low_cardinality_strings(FR.pool, f"{split} pool")
        FR.n_tr_s1, FR.n_tr_pool = (len(FR.s1), len(FR.pool)) if split == "train" else (0, 0)
        for c in ("samp_match", "samp_dist", "samp_hard"):
            FR.pool[c] = (FR.pool[c].fillna(0) if c in FR.pool.columns else 0)
            FR.pool[c] = FR.pool[c].astype(np.int8)
        FR.pool_is_s3 = FR.pool["pool_source"].values == "S3"
        FR.s1_cc, FR.pool_cc = country_codes(FR.s1["country_norm"].values, FR.pool["country_norm"].values)
        FR.split = split
        assert FR.s1["entity_id"].is_unique and FR.pool["entity_id"].is_unique, "entity ids must be unique"
        log(f"frames[{split}]: S1 rows={len(FR.s1):,}  pool rows={len(FR.pool):,}  RAM={_ram_gb()}")
    return FR


def get_views(kind, split):
    stores = {r: get_store(kind, split, r) for r in ("s1", "pool")}
    for r, st_ in stores.items():
        if st_.complete():
            continue
        if kind == "tfidf":
            with stage(f"embed_tfidf.{split}.{r}"):
                run_embedding(st_, store_texts("tfidf", split, r), "tfidf_ret", f"tfidf {split}/{r}")
        if not st_.complete():
            raise RuntimeError(f"{kind} vectors for {split}/{r} incomplete ({len(st_.missing())} chunks) -- run again "
                               "to continue (MAX_SESSION_HOURS reached?)")
    return EmbView([stores["s1"]], stores["s1"].dim), EmbView([stores["pool"]], stores["pool"].dim)


RETR_KEYS = ["DENSE_TOP_K", "REV_TOP_K", "REV_DIAG_TOP_K", "ENABLE_GLOBAL_FALLBACK", "GLOBAL_FALLBACK_TOP_K",
             "BM25_TOP_K", "BM25_K1", "BM25_B", "BM25_FIELD_WEIGHTS", "BM25_MAX_DF_FRAC", "BM25_MIN_DF_CAP",
             "ENABLE_TFIDF_RETRIEVER", "TFIDF_TOP_K", "EXACT_TOP_K", "EXACT_MAX_BLOCK", "EXACT_MAX_PAIRS", "DIAG_TOP_K",
             "DIAG_QUERY_FRACTION", "EMB_NAME_WEIGHT"]


def retr_cfg_fp():
    """Fingerprint of everything a recall / gate number depends on (checked before a gate is trusted)."""
    return cfg_fp(RETR_KEYS + ["MAX_CANDIDATES"], extra=[RETR_CODE_VERSION, FP_SAMPLE, embed_model_tag()])


def diag_query_mask():
    """Train S1 rows searched at diagnostic depth (hash sample; the full train pool is always searched)."""
    return hash_u01(ids_to_int(FR.s1["entity_id"].values[:FR.n_tr_s1]), SALT["diag"]) < cfg.DIAG_QUERY_FRACTION


def make_ctx(name):
    """'diag'  = DIAG_QUERY_FRACTION of train S1 vs the FULL train pool at diagnostic depth (recall curves, choose K);
       'train' = all train S1 vs the full train pool at production depth (training set + train evaluation);
       'test'  = all test S1 vs the full test pool at production depth (submission)."""
    split = "test" if name == "test" else "train"
    ensure_frames(split)
    ctx = SimpleNamespace(name=name, split=split, depth="diag" if name == "diag" else "prod")
    ctx.dense = get_views("dense", split)
    ctx.emb_name, ctx.emb_addr = get_views("name", split), get_views("addr", split)   # reranker features
    ctx.tfidf = get_views("tfidf", split) if cfg.ENABLE_TFIDF_RETRIEVER else None
    n1, npool = len(FR.s1), len(FR.pool)
    ctx.pool, ctx.rev = np.arange(npool), np.arange(n1)   # the reverse index always holds every S1 of the split
    if name == "diag":
        ctx.q = np.flatnonzero(diag_query_mask())
    elif name in ("train", "test"):
        ctx.q = np.arange(n1)
    else:
        raise ValueError(name)
    tfp = get_tfidf_model("ret")[2] if cfg.ENABLE_TFIDF_RETRIEVER else None
    ctx.fp = cfg_fp(RETR_KEYS, extra=[name, FP_SAMPLE if split == "train" else FP_TEST, embed_model_tag(), tfp, n1, npool,
                                      len(ctx.q), RETR_CODE_VERSION])
    ctx.rrel = f"retrieval/{name}_{ctx.fp}"   # persisted in CACHE_DIR; a re-run finds finished partitions there
    log(f"context {name}: {len(ctx.q):,} queries vs {npool:,} pool records ({ctx.depth} depth)  fp={ctx.fp}")
    return ctx




def ctx_rel(ctx, *parts):
    return "/".join([ctx.rrel] + [str(p) for p in parts])


def ctx_save_parquet(ctx, df, *parts, row_group_size=None):
    rel = ctx_rel(ctx, *parts)
    save_parquet(rel, df, row_group_size=row_group_size)
    mark_done(rel, rows=len(df))
    return rel


def ctx_read_parquet(ctx, *parts, filters=None):
    p = locate(ctx_rel(ctx, *parts))
    if p is None:
        raise FileNotFoundError(ctx_rel(ctx, *parts))
    return pq.read_table(p, filters=filters).to_pandas()


def ctx_list_done(ctx, *parts):
    """Names of completed parquet parts under ctx/<parts>/ across this session and earlier sessions."""
    names = set()
    for base in [CACHE_DIR] + CACHE_SEARCH_PATHS:
        names |= {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(base, ctx_rel(ctx, *parts), "*.parquet.done"))}
    return sorted(names)


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)) or "none"


def depth_k(ctx, prod_k):
    return max(prod_k, cfg.DIAG_TOP_K) if ctx.depth == "diag" else prod_k


def partitions(ctx):
    qc = FR.s1["country_norm"].values[ctx.q]
    pcn = FR.pool["country_norm"].values[ctx.pool]
    psrc = FR.pool["pool_source"].values[ctx.pool]
    rc = FR.s1["country_norm"].values[ctx.rev]
    out = []
    for c in sorted(set(qc)):
        for s in ("S2", "S3"):
            out.append((c, s, ctx.q[qc == c], ctx.pool[(pcn == c) & (psrc == s)], ctx.rev[rc == c]))
    return out


def _dense_fwd(views, qr, pr, k):
    s, i = dense_search(views[0], qr, views[1], pr, k)
    return _topk_frame(qr, pr, s, i)


def _dense_rev(ctx, pr, rev_rows):
    k = max(cfg.REV_TOP_K, cfg.REV_DIAG_TOP_K) if ctx.depth == "diag" else cfg.REV_TOP_K
    s, i = dense_search(ctx.dense[1], pr, ctx.dense[0], rev_rows, k)
    # keep hits on this context's queries only (the scale context's reverse index also holds test S1). Masking the
    # index array first keeps memory bounded -- building the frame first would materialize len(pr) * k rows.
    in_q = np.zeros(len(FR.s1), bool)
    in_q[ctx.q] = True
    i = np.where((i >= 0) & in_q[rev_rows[np.maximum(i, 0)]], i, -1)
    return _topk_frame(pr, rev_rows, s, i, reverse=True)


def _dense_global(ctx, src):
    pr = ctx.pool[FR.pool["pool_source"].values[ctx.pool] == src]
    pool_countries = list(set(FR.pool["country_norm"].values[pr]))
    orphan = ~np.isin(FR.s1["country_norm"].values[ctx.q].astype(str), np.array(pool_countries, dtype=str))
    parts = []
    for mask, k in ((~orphan, cfg.GLOBAL_FALLBACK_TOP_K), (orphan, depth_k(ctx, cfg.DENSE_TOP_K))):
        qr = ctx.q[mask]
        if len(qr) and len(pr):
            parts.append(_dense_fwd(ctx.dense, qr, pr, k))
    REPORT.setdefault("orphan_queries", {})[f"{ctx.name}_{src}"] = int(orphan.sum())
    return pd.concat(parts, ignore_index=True) if parts else _topk_frame(np.zeros(0, np.int64), pr, np.zeros((0, 1), np.float32), np.zeros((0, 1), np.int64))


def _bm25(qr, pr, k):
    qi, di, s, rank = bm25_search(FR.pool, pr, FR.s1, qr, k)
    return pd.DataFrame({"s1_pos": qr[qi], "pool_pos": pr[di], "score": s, "rank": rank})


def _sorted_s1_rank(df):
    """df ordered by (s1_pos, rank). Forward retrievers already come out in that order: the stable sort would
    return the same rows, so it is skipped there (it copies the whole partition, up to 66M rows)."""
    s1, rk = df["s1_pos"].values, df["rank"].values
    d = np.diff(s1)
    if len(df) < 2 or ((d >= 0).all() and (np.diff(rk)[d == 0] >= 0).all()):
        return df
    return df.sort_values(["s1_pos", "rank"], kind="stable")


def run_retrievers(ctx):
    jobs = []
    for c, s, qr, pr, rr in partitions(ctx):
        tag = f"{_slug(c)}_{s}"
        if not len(qr) or not len(pr):
            continue
        jobs.append(("dense", tag, lambda qr=qr, pr=pr: _dense_fwd(ctx.dense, qr, pr, depth_k(ctx, cfg.DENSE_TOP_K))))
        if len(rr):
            jobs.append(("dense_rev", tag, lambda pr=pr, rr=rr: _dense_rev(ctx, pr, rr)))
        jobs.append(("bm25", tag, lambda qr=qr, pr=pr: _bm25(qr, pr, depth_k(ctx, cfg.BM25_TOP_K))))
        if ctx.tfidf is not None:
            jobs.append(("tfidf", tag, lambda qr=qr, pr=pr: _dense_fwd(ctx.tfidf, qr, pr, depth_k(ctx, cfg.TFIDF_TOP_K))))
    if cfg.ENABLE_GLOBAL_FALLBACK:
        for s in ("S2", "S3"):
            jobs.append(("dense_global", s, lambda s=s: _dense_global(ctx, s)))
    for key in EXACT_KEYS:
        jobs.append((key, "all", lambda key=key: exact_block(FR.s1, ctx.q, FR.pool, ctx.pool, key,
                                                           REPORT.setdefault("exact_stats", {}).setdefault(f"{ctx.name}_{key}", {}))))
    times = defaultdict(float)
    todo = [j for j in jobs if not is_done(ctx_rel(ctx, j[0], f"{j[1]}.parquet"))]
    log(f"[{ctx.name}] retrieval: {len(jobs) - len(todo)}/{len(jobs)} partition jobs already checkpointed, {len(todo)} to run")
    for n_done, (r, part, fn) in enumerate(todo, 1):
        if hours_left() <= 0:
            raise RuntimeError("MAX_SESSION_HOURS reached during retrieval -- finished partitions are saved; run again")
        t0 = time.time()
        df = _sorted_s1_rank(fn())
        ctx_save_parquet(ctx, df, r, f"{part}.parquet", row_group_size=500_000)   # checkpoint per partition
        times[r] += time.time() - t0
        log(f"  [{ctx.name}] {n_done}/{len(todo)} {r:17s} {part:14s} {len(df):>11,} pairs  {time.time() - t0:7.1f}s  "
            f"RSS {rss_now_gb()[0]} GB")
        del df
    rel = f"reports/retrieval_seconds_{ctx.name}_{ctx.fp}.json"
    prev_p = locate(rel)
    prev = {}
    if prev_p:
        with open(prev_p) as f:
            prev = {k: v for k, v in json.load(f).items() if k != "fp_sample"}
    merged = {k: round(float(prev.get(k, 0)) + times.get(k, 0.0), 2) for k in set(prev) | set(times)}
    if times:   # a fully cached (resumed) run must not overwrite earlier sessions' timings with zeros
        save_json(rel, dict(merged, fp_sample=FP_SAMPLE))
    REPORT.setdefault("retrieval_seconds", {})[ctx.name + REPORT_SUFFIX] = merged


def load_retr(ctx, lo=None, hi=None):
    out = {}
    filters = [("s1_pos", ">=", int(lo)), ("s1_pos", "<", int(hi))] if lo is not None else None
    for r in RETRIEVERS:
        dfs = [ctx_read_parquet(ctx, r, name, filters=filters) for name in ctx_list_done(ctx, r)]
        dfs = [d for d in dfs if len(d)]
        if not dfs:
            continue
        d = pd.concat(dfs, ignore_index=True)
        if r == "dense_rev":   # union ordering: rank among this S1's reverse hits (rev_rank is the pool's view)
            d["order_rank"] = within_group_rank(d["s1_pos"].values, d["score"].values).astype(np.int32)
        out[r] = d
    return out


PROD_TOPK = lambda: {"dense": cfg.DENSE_TOP_K, "dense_global": cfg.DENSE_TOP_K, "dense_rev": cfg.REV_TOP_K,
                     "bm25": cfg.BM25_TOP_K, "tfidf": cfg.TFIDF_TOP_K, **{k: cfg.EXACT_TOP_K for k in EXACT_KEYS}}


def truncate_retr(retr, override=None):
    """Production view of diagnostic-depth lists. The reverse list's per-S1 order_rank is recomputed on the
    truncated hits, exactly as production computes it (otherwise train and test see different best_rank)."""
    tk = dict(PROD_TOPK(), **(override or {}))
    out = {}
    for r, d in retr.items():
        d = d[d["rank"].values < tk[r]]
        if r == "dense_rev" and len(d):
            d = d.assign(order_rank=within_group_rank(d["s1_pos"].values, d["score"].values).astype(np.int32))
        out[r] = d
    return out


def group_positions(group, rank_key, score):
    """0-based position within each group when sorted by (rank_key asc, score desc)."""
    group = np.asarray(group)
    if len(group) == 0:
        return np.zeros(0, np.int64)
    order = np.lexsort((-np.asarray(score, dtype=np.float64), np.asarray(rank_key), group))
    g = group[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    pos_sorted = np.arange(len(g)) - np.repeat(starts, np.diff(np.r_[starts, len(g)]))
    pos = np.empty(len(g), dtype=np.int64)
    pos[order] = pos_sorted
    return pos


def build_union(retr, n_pool, cap=None):
    """Dedupe on (s1, pool); keep every retriever's score/rank/flag; cap per S1 by best rank (round-robin)."""
    names = [r for r in RETRIEVERS if r in retr and len(retr[r])]
    n_pool = np.int64(n_pool)
    keys = [retr[r]["s1_pos"].values.astype(np.int64) * n_pool + retr[r]["pool_pos"].values.astype(np.int64) for r in names]
    uniq, inv = (np.unique(np.concatenate(keys), return_inverse=True) if names else (np.zeros(0, np.int64), np.zeros(0, np.int64)))
    n = len(uniq)
    out = {"s1_pos": (uniq // n_pool).astype(np.int64), "pool_pos": (uniq % n_pool).astype(np.int64)}
    best = np.full(n, RANK_SENTINEL, np.int32)
    nret = np.zeros(n, np.int8)
    off = 0
    for r, k in zip(names, keys):
        d = retr[r]
        idx = inv[off:off + len(k)]
        off += len(k)
        o = np.lexsort((d["rank"].values, idx))
        first = np.r_[True, idx[o][1:] != idx[o][:-1]] if len(o) else np.zeros(0, bool)
        sel = o[first]
        sc = np.zeros(n, np.float32)
        rk = np.full(n, RANK_SENTINEL, np.int32)
        fl = np.zeros(n, np.int8)
        sc[idx[sel]] = d["score"].values[sel]
        rk[idx[sel]] = d["rank"].values[sel]
        fl[idx[sel]] = 1
        ordr = (d["order_rank"].values if "order_rank" in d.columns else d["rank"].values)[sel]
        best[idx[sel]] = np.minimum(best[idx[sel]], ordr)
        out[f"{r}_score"], out[f"{r}_rank"], out[f"by_{r}"] = sc, rk, fl
        nret += fl
    for r in RETRIEVERS:
        if r not in names:
            out[f"{r}_score"] = np.zeros(n, np.float32)
            out[f"{r}_rank"] = np.full(n, RANK_SENTINEL, np.int32)
            out[f"by_{r}"] = np.zeros(n, np.int8)
    out["n_retrievers"], out["best_rank"] = nret, best
    tie = np.where(out["by_dense"] == 1, out["dense_score"],
                   np.where(out["by_dense_global"] == 1, out["dense_global_score"], -1.0))
    order = np.lexsort((-tie, -nret.astype(np.int16), best, out["s1_pos"]))
    g = out["s1_pos"][order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1] if n else np.zeros(0, np.int64)
    pos = np.empty(n, np.int64)
    pos[order] = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n])) if n else pos
    wide = pd.DataFrame(out)
    wide["pos_in_entity"] = pos
    if cap is not None:
        wide = wide[wide["pos_in_entity"].values < cap].reset_index(drop=True)
    return wide