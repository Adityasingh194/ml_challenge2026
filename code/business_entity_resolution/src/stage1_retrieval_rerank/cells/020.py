try:
    from rapidfuzz.process import cpdist as _cpdist
    from rapidfuzz.utils import default_process as _rf_default_process
    from rapidfuzz.distance import JaroWinkler as _JW
    CPDIST_AVAILABLE = True
except ImportError:
    CPDIST_AVAILABLE = False
    from rapidfuzz.utils import default_process as _rf_default_process
    from rapidfuzz.distance import JaroWinkler as _JW

_RF_WORKERS = -1   # rapidfuzz threads per call; set to 1 inside feature worker processes (no oversubscription)


# ------------------------- copied from er_crossencoder_v2 §6 -------------------------
def _pairwise(scorer, a, b, processor=None):
    if CPDIST_AVAILABLE:
        return _cpdist(a, b, scorer=scorer, processor=processor, workers=_RF_WORKERS, dtype=np.float32)
    if processor is not None:
        return np.fromiter((scorer(processor(x), processor(y)) for x, y in zip(a, b)), dtype=np.float32, count=len(a))
    return np.fromiter((scorer(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))


def _soft_equal(a, b):
    """1.0 if both present and equal, 0.0 if both present and different, 0.5 if either missing.
    Works on object arrays directly (no astype(str): that makes a huge fixed-width unicode copy)."""
    present = (a != "") & (b != "")
    return np.where(present, (a == b).astype(np.float32), np.float32(0.5)).astype(np.float32)


def _set_jaccard(sa, sb):
    return np.fromiter(
        ((len(x & y) / len(x | y)) if (x or y) else 0.0 for x, y in zip(sa, sb)),
        dtype=np.float32, count=len(sa))


def _features_chunk(c, s1_df, pool_df):
    i1 = c["s1_pos"].values
    i2 = c["pool_pos"].values
    g1 = lambda col: s1_df[col].values[i1]
    g2 = lambda col: pool_df[col].values[i2]
    f = {}
    n1, n2 = g1("name_clean"), g2("name_clean")
    f["name_ratio"] = _pairwise(rf_fuzz.ratio, n1, n2, _rf_default_process) / 100
    f["name_partial"] = _pairwise(rf_fuzz.partial_ratio, n1, n2, _rf_default_process) / 100
    f["name_token_sort"] = _pairwise(rf_fuzz.token_sort_ratio, n1, n2, _rf_default_process) / 100
    f["name_token_set"] = _pairwise(rf_fuzz.token_set_ratio, n1, n2, _rf_default_process) / 100
    c1, c2 = g1("name_core"), g2("name_core")
    f["core_jw"] = _pairwise(_JW.normalized_similarity, c1, c2)
    f["core_exact"] = ((c1 == c2) & (c1 != "")).astype(np.float32)
    f["sorted_exact"] = ((g1("name_sorted") == g2("name_sorted")) & (c1 != "")).astype(np.float32)
    a1, a2 = g1("name_acronym"), g2("name_acronym")
    c1_compact = np.array([s.replace(" ", "") for s in c1], dtype=object)
    c2_compact = np.array([s.replace(" ", "") for s in c2], dtype=object)
    f["acronym_match"] = (((a1 != "") & (a1 == c2_compact)) | ((a2 != "") & (a2 == c1_compact))).astype(np.float32)
    p1, p2 = g1("name_phonetic"), g2("name_phonetic")
    f["phonetic_match"] = ((p1 == p2) & (p1 != "")).astype(np.float32)
    f["suffix_agree"] = _soft_equal(g1("name_suffix"), g2("name_suffix"))
    f["name_jaccard"] = _set_jaccard(g1("name_tokens"), g2("name_tokens"))
    l1 = np.fromiter((len(s) for s in n1), dtype=np.float32, count=len(n1))
    l2 = np.fromiter((len(s) for s in n2), dtype=np.float32, count=len(n2))
    f["name_len_diff"] = np.abs(l1 - l2) / np.maximum(np.maximum(l1, l2), 1)
    ad1, ad2 = g1("addr_clean"), g2("addr_clean")
    f["addr_ratio"] = _pairwise(rf_fuzz.ratio, ad1, ad2, _rf_default_process) / 100
    f["addr_token_sort"] = _pairwise(rf_fuzz.token_sort_ratio, ad1, ad2, _rf_default_process) / 100
    f["addr_token_set"] = _pairwise(rf_fuzz.token_set_ratio, ad1, ad2, _rf_default_process) / 100
    d1, d2 = g1("addr_digits"), g2("addr_digits")
    f["digits_jaccard"] = _set_jaccard(d1, d2)
    f["digits_conflict"] = np.fromiter(
        (1.0 if (x and y and not (x & y)) else 0.0 for x, y in zip(d1, d2)), dtype=np.float32, count=len(d1))
    f["city_match"] = _soft_equal(g1("city_norm"), g2("city_norm"))
    f["state_match"] = _soft_equal(g1("state_norm"), g2("state_norm"))
    f["country_match"] = _soft_equal(g1("country_norm"), g2("country_norm"))
    f["city_ratio"] = _pairwise(rf_fuzz.ratio, g1("city_norm"), g2("city_norm")) / 100
    f["cand_is_s3"] = (g2("pool_source") == "S3").astype(np.float32)
    return pd.DataFrame({k: v.astype(np.float32) for k, v in f.items()}, index=c.index)


def within_group_rank(group, score):
    """0-based rank of each element within its group, by descending score (numpy, O(n log n))."""
    if len(group) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((-score, group))
    g_sorted = group[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1]
    sizes = np.diff(np.r_[starts, len(g_sorted)])
    rank_sorted = np.arange(len(g_sorted)) - np.repeat(starts, sizes)
    rank = np.empty(len(group), dtype=np.int64)
    rank[order] = rank_sorted
    return rank

# ------------------------- new features -------------------------
BASE_FEATURE_COLS = [
    "name_ratio", "name_partial", "name_token_sort", "name_token_set", "core_jw", "core_exact",
    "sorted_exact", "acronym_match", "phonetic_match", "suffix_agree", "name_jaccard", "name_len_diff",
    "addr_ratio", "addr_token_sort", "addr_token_set", "digits_jaccard", "digits_conflict",
    "city_match", "state_match", "country_match", "city_ratio", "cand_is_s3",
]
NEW_FEATURE_COLS = (
    ["emb_cos", "emb_name_cos", "emb_addr_cos", "tfidf_cos"]
    + [f"{r}_{k}" for r in RETRIEVERS for k in ("score", "rank")] + [f"by_{r}" for r in RETRIEVERS]
    + ["n_retrievers", "best_rank", "mutual_top5",
       "house_agree", "house_present_s1", "house_present_cand", "postcode_agree", "addr_word_jaccard",
       "s1_city_missing", "cand_city_missing", "cand_state_missing", "cand_addr_missing", "cand_non_latin",
       "emb_rank_in_entity", "emb_gap_to_best", "name_gap_to_best", "bm25_rel_to_best", "n_cands_entity"]
)
FEATURE_COLS = BASE_FEATURE_COLS + NEW_FEATURE_COLS
DENSE_LIKE = ("dense", "dense_global", "dense_rev")


def _pair_cos(views, s1_rows, pool_rows):
    out = np.empty(len(s1_rows), np.float32)
    order = np.argsort(pool_rows, kind="stable")   # sequential reads of the pool cache
    for a in range(0, len(order), cfg.PAIR_BATCH):
        o = order[a:a + cfg.PAIR_BATCH]
        out[o] = (views[0].gather(s1_rows[o], np.float32) * views[1].gather(pool_rows[o], np.float32)).sum(1)
    return out


def _chunk_rows(C):
    """The S1 / pool frame rows a candidate chunk needs (sorted unique positions, as pair_features uses them)."""
    return (FR.s1.iloc[np.unique(C["s1_pos"].values)].reset_index(drop=True),
            FR.pool.iloc[np.unique(C["pool_pos"].values)].reset_index(drop=True))


def pair_features(C, ctx, rows=None):
    """Features for candidate pairs C (complete entities only -- entity-context features need them).
    rows = _chunk_rows(C) when computed by the caller (forked workers must not touch FR, see features_in_chunks)."""
    i1, i2 = C["s1_pos"].values, C["pool_pos"].values
    u1, inv1 = np.unique(i1, return_inverse=True)
    u2, inv2 = np.unique(i2, return_inverse=True)
    r1, r2 = rows if rows is not None else _chunk_rows(C)
    sub1 = add_set_columns(r1)
    sub2 = add_set_columns(r2)
    base = _features_chunk(pd.DataFrame({"s1_pos": inv1, "pool_pos": inv2}), sub1, sub2)
    f = {c: base[c].values for c in BASE_FEATURE_COLS}
    n = len(C)
    qcos = np.full(n, np.nan, np.float32)
    for r in DENSE_LIKE:
        m = C[f"by_{r}"].values == 1
        qcos[m] = C[f"{r}_score"].values[m]
    miss = np.isnan(qcos)
    if miss.any():
        qcos[miss] = _pair_cos(ctx.dense, i1[miss], i2[miss])
    f["emb_cos"] = qcos
    f["emb_name_cos"] = _pair_cos(ctx.emb_name, i1, i2)
    f["emb_addr_cos"] = _pair_cos(ctx.emb_addr, i1, i2)
    if ctx.tfidf is not None:
        tcos = np.where(C["by_tfidf"].values == 1, C["tfidf_score"].values, np.nan).astype(np.float32)
        tm = np.isnan(tcos)
        if tm.any():
            tcos[tm] = _pair_cos(ctx.tfidf, i1[tm], i2[tm])
        f["tfidf_cos"] = tcos
    else:
        f["tfidf_cos"] = np.zeros(n, np.float32)
    for r in RETRIEVERS:
        f[f"{r}_score"] = C[f"{r}_score"].values.astype(np.float32)
        f[f"{r}_rank"] = np.minimum(C[f"{r}_rank"].values, RANK_SENTINEL).astype(np.float32)
        f[f"by_{r}"] = C[f"by_{r}"].values.astype(np.float32)
    f["n_retrievers"] = C["n_retrievers"].values.astype(np.float32)
    f["best_rank"] = C["best_rank"].values.astype(np.float32)
    f["mutual_top5"] = ((C["dense_rank"].values < 5) & (C["dense_rev_rank"].values < 5)).astype(np.float32)
    h1, h2 = sub1["house_no"].values[inv1], sub2["house_no"].values[inv2]
    f["house_agree"] = _soft_equal(h1, h2)
    f["house_present_s1"] = (h1 != "").astype(np.float32)
    f["house_present_cand"] = (h2 != "").astype(np.float32)
    f["postcode_agree"] = _soft_equal(sub1["postcode"].values[inv1], sub2["postcode"].values[inv2])
    f["addr_word_jaccard"] = _set_jaccard(sub1["addr_words"].values[inv1], sub2["addr_words"].values[inv2])
    f["s1_city_missing"] = (sub1["city_norm"].values[inv1] == "").astype(np.float32)
    f["cand_city_missing"] = (sub2["city_norm"].values[inv2] == "").astype(np.float32)
    f["cand_state_missing"] = (sub2["state_norm"].values[inv2] == "").astype(np.float32)
    f["cand_addr_missing"] = (sub2["addr_clean"].values[inv2] == "").astype(np.float32)
    f["cand_non_latin"] = sub2["non_latin"].values[inv2].astype(np.float32)
    g = pd.Series(i1)
    f["emb_rank_in_entity"] = within_group_rank(i1, qcos).astype(np.float32)
    f["emb_gap_to_best"] = (pd.Series(qcos).groupby(g).transform("max").values - qcos).astype(np.float32)
    nts = f["name_token_set"]
    f["name_gap_to_best"] = (pd.Series(nts).groupby(g).transform("max").values - nts).astype(np.float32)
    b = f["bm25_score"]
    bmax = pd.Series(b).groupby(g).transform("max").values
    f["bm25_rel_to_best"] = np.where(bmax > 0, b / np.maximum(bmax, 1e-9), 0.0).astype(np.float32)
    f["n_cands_entity"] = pd.Series(i1).groupby(g).transform("size").values.astype(np.float32)
    return pd.DataFrame({k: f[k] for k in FEATURE_COLS})


# ------------------------- parallel feature computation (fork: workers share FR and the memmaps) -------------------------
_FEAT_JOB = None


def _feature_worker(a, b, rows):
    global _RF_WORKERS
    _RF_WORKERS = 1
    C, ctx = _FEAT_JOB
    return a, pair_features(C.iloc[a:b], ctx, rows).to_numpy(np.float32)


def _private_gb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            return sum(int(l.split()[1]) for l in f if l.startswith(("Private_Clean", "Private_Dirty", "Swap:"))) / 1e6
    except OSError:
        return 0.0


def feature_workers():
    """Worker count under RAM_BUDGET_GB: parent anonymous RSS + FEATURE_WORKER_GB per worker (measured and
    updated after every parallel run)."""
    anon = rss_now_gb()[1] or 0.0
    room = cfg.RAM_BUDGET_GB - anon - 3.0   # 3 GB headroom for the chunk's feature matrix and results
    return max(1, min(cfg.NUM_WORKERS, int(room // max(_WORKER_GB[0], 0.1))))


_WORKER_GB = [1.0]   # running estimate of one feature worker's private memory
FEATURE_CHUNK_PAIRS = 20_000   # pairs per worker task: fixed, so a worker's memory does not grow when fewer run


def cpu_temp_c():
    """Hottest CPU core (coretemp / k10temp), or None."""
    temps = []
    for t in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
        try:
            if open(os.path.join(os.path.dirname(t), "name")).read().strip() in ("coretemp", "k10temp", "zenpower"):
                temps.append(int(open(t).read()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def _entity_cuts(s1, target):
    starts = np.r_[0, np.flatnonzero(np.diff(s1)) + 1]
    cuts, last = [0], 0
    for st_ in starts:
        if st_ - last >= target:
            cuts.append(int(st_))
            last = st_
    cuts.append(len(s1))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def features_in_chunks(C, ctx, label=""):
    """Entity-aligned chunks (C is sorted by s1_pos), computed in NUM_WORKERS forked processes. Forked workers
    inherit the frames and memmaps (nothing large is pickled) and run CPU code only (no CUDA in children)."""
    global _FEAT_JOB
    t0 = time.time()
    n_proc = min(feature_workers(), max(1, len(C) // 50_000))
    target = max(min(FEATURE_CHUNK_PAIRS, math.ceil(len(C) / max(4 * n_proc, 1))), 1)
    bounds = _entity_cuts(C["s1_pos"].values, target)
    X = np.empty((len(C), len(FEATURE_COLS)), np.float32)   # results land here directly (no parts + concat copy)
    worker_gb = 0.0
    if n_proc > 1 and len(bounds) > 1:
        # The parent slices each chunk's frame rows and sends them (a few MB); workers never index FR, so they do
        # not touch (and copy-on-write) the pages of its ~10 GB of Python strings. gc.freeze keeps the children's
        # garbage collector off the inherited objects. At most 2 chunks per worker are in flight.
        import multiprocessing as mp
        _FEAT_JOB = (C, ctx)
        gc.collect()
        gc.freeze()
        try:
            with mp.get_context("fork").Pool(n_proc) as pool:
                pending, it = [], iter(bounds)
                for a, b in it:
                    pending.append(pool.apply_async(_feature_worker, (a, b, _chunk_rows(C.iloc[a:b]))))
                    while len(pending) >= 2 * n_proc or (pending and pending[0].ready()):
                        a_, arr = pending.pop(0).get()
                        X[a_:a_ + len(arr)] = arr
                    if len(pending) == 2 * n_proc - 1:
                        worker_gb = max(worker_gb, max(_private_gb(p.pid) for p in pool._pool))
                for r in pending:
                    a_, arr = r.get()
                    X[a_:a_ + len(arr)] = arr
                worker_gb = max(worker_gb, max(_private_gb(p.pid) for p in pool._pool))
        finally:
            _FEAT_JOB = None
            gc.unfreeze()
        _WORKER_GB[0] = max(worker_gb, 0.3)
    else:
        for a, b in bounds:
            X[a:b] = pair_features(C.iloc[a:b], ctx).to_numpy(np.float32)
        _WORKER_GB[0] = min(_WORKER_GB[0], 1.0)   # re-measure next time instead of staying serial
    out = pd.DataFrame(X, columns=FEATURE_COLS, copy=False)
    dt = time.time() - t0
    REPORT.setdefault("feature_seconds", {}).setdefault(ctx.name, 0.0)
    REPORT["feature_seconds"][ctx.name] += dt
    if label:
        log(f"  features {label}: {len(C):,} pairs in {dt:.1f}s ({len(C) / max(dt, 1e-9):,.0f} pairs/s, {n_proc} processes, "
            f"worker private peak {worker_gb:.2f} GB, parent RSS {rss_now_gb()[0]} GB, CPU {cpu_temp_c()} C)")
    return out
