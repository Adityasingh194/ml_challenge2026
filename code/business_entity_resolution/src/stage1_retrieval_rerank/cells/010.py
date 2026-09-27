# SRC_FILE / GT_PATH are set by §2b (parquet copies of the raw TSVs)
RAW_COLS = ["entity_id", "business_name", "business_address", "country", "name_clean", "addr_clean",
            "extracted_city", "extracted_state"]
# Only RAW_FIELDS exist in the data. The other names are views computed at read time from the raw fields:
# name_clean / addr_clean = lower-cased raw name / address; extracted_city / extracted_state = "" (not in the data).
_VIEW_OF = {"name_clean": "business_name", "addr_clean": "business_address",
            "extracted_city": None, "extracted_state": None}
ID_BASE = 10 ** 10
SALT = dict(s1=1, pool=2, hard=3, test_s1=4, test_pool=5, fold=6, fit=7, audit=8, diag=9, gbdt=10, grid=11)


def ids_to_int(ids):
    """'S2-681193310' -> 2*10^10 + 681193310 (int64). Raises on any unexpected id format."""
    a = ids if isinstance(ids, (pa.Array, pa.ChunkedArray)) else pa.array(np.asarray(ids, dtype=object), type=pa.string())
    if len(a) == 0:
        return np.zeros(0, np.int64)
    if not pc.all(pc.match_substring_regex(a, r"^S[123]-[0-9]{1,10}$")).as_py():
        raise ValueError("unexpected entity_id format (expected S<1|2|3>-<digits>)")
    src = pc.cast(pc.utf8_slice_codeunits(a, 1, 2), pa.int64())
    num = pc.cast(pc.utf8_slice_codeunits(a, 3, 20), pa.int64())
    out = pc.add(pc.multiply(src, ID_BASE), num)
    return np.asarray(out.to_numpy() if isinstance(out, pa.Array) else out.to_numpy(), dtype=np.int64)


def read_ids(split, s):
    col = pq.read_table(SRC_FILE[(split, s)], columns=["entity_id"]).column(0)
    return col, ids_to_int(col)


def read_raw_table(path, columns):
    """Read `columns` from a source parquet; the derived views (_VIEW_OF) are built from the raw fields."""
    need = list(dict.fromkeys(_VIEW_OF.get(c) or c for c in columns if _VIEW_OF.get(c, c) is not None))
    t = pq.read_table(path, columns=need or ["entity_id"])
    out = {}
    for c in columns:
        if c not in _VIEW_OF:
            out[c] = t.column(c)
        elif _VIEW_OF[c] is None:
            out[c] = pa.array(np.full(t.num_rows, "", dtype=object), type=pa.string())
        else:
            out[c] = pc.utf8_lower(pc.fill_null(t.column(_VIEW_OF[c]), ""))
    return pa.table(out)


def read_source(split, s, columns=RAW_COLS, mask=None):
    t = read_raw_table(SRC_FILE[(split, s)], columns)
    if mask is not None:
        t = t.filter(pa.array(np.asarray(mask, dtype=bool)))
    return arrow_to_frame(t)


def splitmix64(x):
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def hash_u01(ids_int, salt):
    """Deterministic uniform [0,1) per id (independent of machine / Python hash seed)."""
    key = np.uint64((cfg.RANDOM_SEED * 0x100000001B3 + salt * 0x9E3779B1 + 12345) & ((1 << 64) - 1))
    u = splitmix64(np.asarray(ids_int, dtype=np.int64).astype(np.uint64) ^ key)
    return (u >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def load_ground_truth():
    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False)
    assert list(gt.columns) == ["source1_entity_id", "matched_entity_ids"], gt.columns
    return gt


def explode_gt(gt):
    rows = gt[gt["matched_entity_ids"] != ""]
    counts = rows["matched_entity_ids"].str.count(",").values + 1
    s1 = np.repeat(rows["source1_entity_id"].values, counts)
    pool = np.array([x.strip() for s in rows["matched_entity_ids"].values for x in s.split(",")], dtype=object)
    return s1, pool


def sample_fps():
    """(FP_SAMPLE, FP_TEST) for the current cfg.RANDOM_SEED (recomputed by the seed sweep)."""
    return (cfg_fp(["DATA_FRACTION", "POOL_DISTRACTOR_FRACTION", "HARD_NEIGHBOR_K", "HARD_NEIGHBOR_POOL_FRACTION",
                    "RANDOM_SEED", "BM25_K1", "BM25_B", "BM25_MAX_DF_FRAC", "BM25_MIN_DF_CAP"]),
            cfg_fp(["TEST_FRACTION", "RANDOM_SEED"]))


FP_SAMPLE, FP_TEST = sample_fps()
SAMPLE_REL = f"samples/{FP_SAMPLE}"


def _canonical_country_column(split, s):
    """(int64 ids, canonical country per row) for one source file; only the unique values are canonicalized."""
    t = pq.read_table(SRC_FILE[(split, s)], columns=["entity_id", "country"])
    codes, uniq = pd.factorize(t.column(1).to_pandas(), use_na_sentinel=True)
    names = np.array([canonicalize_country(x) for x in uniq] + [""], dtype=object)   # code -1 (null) -> ""
    return ids_to_int(t.column(0)), names[codes]


def _lookup_rows(sorted_ids, order, query):
    j = order[np.clip(np.searchsorted(sorted_ids, query), 0, len(sorted_ids) - 1)]
    return j


def check_country_invariant():
    """Hard country blocking is valid only if NO labelled positive pair crosses countries. Checked on the FULL
    train ground truth (every pair, not the sample), with the same canonical country the partitions use.
    Raises (STOP) when the violation count exceeds COUNTRY_INVARIANT_MAX_VIOLATIONS."""
    rel = "reports/country_invariant.json"
    p = locate(rel)
    if p:
        with open(p) as f:
            res = json.load(f)
        print(f"country invariant (cached): {res['positive_pairs_checked']:,} positive pairs checked, "
              f"{res['cross_country_pairs']:,} cross-country")
    else:
        with stage("country_invariant") as st:
            cols, per_country = {}, {}
            for split in ("train", "test"):
                for s in (1, 2, 3):
                    ids, cn = _canonical_country_column(split, s)
                    per_country[f"{split}_S{s}"] = pd.Series(cn).value_counts().to_dict()
                    if split == "train":
                        cols[s] = (ids, cn)
            s1_ids, s1_cn = cols[1]
            pool_ids = np.concatenate([cols[2][0], cols[3][0]])
            pool_cn = np.concatenate([cols[2][1], cols[3][1]])
            a, b = explode_gt(load_ground_truth())
            ai, bi = ids_to_int(a), ids_to_int(b)
            o1, o2 = np.argsort(s1_ids, kind="stable"), np.argsort(pool_ids, kind="stable")
            j1 = _lookup_rows(s1_ids[o1], o1, ai)
            j2 = _lookup_rows(pool_ids[o2], o2, bi)
            if not ((s1_ids[j1] == ai).all() and (pool_ids[j2] == bi).all()):
                raise RuntimeError("ground-truth ids missing from the source files")
            c1, c2 = s1_cn[j1], pool_cn[j2]
            viol = c1 != c2
            combos = pd.DataFrame({"s1": c1, "pool": c2}).value_counts()
            te_s1 = set(per_country["test_S1"])
            te_pool = set(per_country["test_S2"]) | set(per_country["test_S3"])
            tr_all = set(per_country["train_S1"]) | set(per_country["train_S2"]) | set(per_country["train_S3"])
            res = dict(positive_pairs_checked=int(len(a)), cross_country_pairs=int(viol.sum()),
                       cross_country_pct=round(100 * float(viol.mean()) if len(a) else 0.0, 6),
                       pairs_with_empty_country=int(((c1 == "") | (c2 == "")).sum()),
                       pair_country_combos={f"{k[0]}->{k[1]}": int(v) for k, v in combos.items()},
                       violation_examples=[dict(s1=str(x), pool=str(y), s1_country=str(u), pool_country=str(v))
                                           for x, y, u, v in list(zip(a[viol], b[viol], c1[viol], c2[viol]))[:10]],
                       records_per_country=per_country,
                       test_s1_countries_without_test_pool=sorted(te_s1 - te_pool),
                       test_countries_unseen_in_train=sorted(te_s1 - tr_all))
            save_json(rel, res)
            st.add(pairs=res["positive_pairs_checked"], violations=res["cross_country_pairs"])
    REPORT["country_invariant"] = res
    print(f"  positive pairs checked: {res['positive_pairs_checked']:,}   cross-country: {res['cross_country_pairs']:,} "
          f"({res['cross_country_pct']}%)   empty country: {res['pairs_with_empty_country']:,}\n"
          f"  pair country combinations: {res['pair_country_combos']}\n"
          f"  test countries unseen in train: {res['test_countries_unseen_in_train']}   "
          f"test S1 countries with no test-pool record: {res['test_s1_countries_without_test_pool']}")
    if res["cross_country_pairs"] > cfg.COUNTRY_INVARIANT_MAX_VIOLATIONS:
        raise RuntimeError(
            f"STOP: the country invariant is violated ({res['cross_country_pairs']:,} labelled positive pairs cross "
            f"countries, e.g. {res['violation_examples'][:3]}). Hard country blocking would lose these matches. Review "
            "the data before changing the architecture (ENABLE_GLOBAL_FALLBACK is the optional cross-country pass).")
    if res["test_s1_countries_without_test_pool"] and not cfg.ENABLE_GLOBAL_FALLBACK:
        print(f"  NOTE: test S1 countries {res['test_s1_countries_without_test_pool']} have no pool records, so they get "
              "no candidates (correct under the invariant: they cannot have matches)")
    print("  -> country hard partitioning is valid and is kept" if res["cross_country_pairs"] == 0 else
          "  -> violations within the configured tolerance")
    return res


def _light_norm(df):
    df["name_core"] = [strip_legal_suffix(x)[0] for x in df["name_clean"]]
    df["city_norm"] = [_strip_accents(str(c)).strip().lower() for c in df["extracted_city"]]
    df["country_norm"] = df["country"].map(canonicalize_country)
    df["addr_clean"] = ""
    df["state_norm"] = ""
    return df


def mine_hard_neighbours(q, own_matches, s, ids_int, in_m, k):
    """Top-k BM25 (name+city) non-matches per sampled S1 from the (sub-sampled) FULL train pool of source s."""
    mine_mask = hash_u01(ids_int, SALT["hard"]) < cfg.HARD_NEIGHBOR_POOL_FRACTION
    mine_rows = np.flatnonzero(mine_mask)
    d = _light_norm(read_source("train", s, ["entity_id", "name_clean", "extracted_city", "country"], mine_mask))
    hard = np.zeros(len(ids_int), dtype=bool)
    q_country, d_country = q["country_norm"].values, d["country_norm"].values
    for c in sorted(set(q_country)):
        qr = np.flatnonzero(q_country == c)
        dr = np.flatnonzero(d_country == c)
        if len(qr) == 0 or len(dr) == 0:
            continue
        qi, di, _, rk = bm25_search(d, dr, q, qr, k + 12, fields=("n", "c"))
        q_ids = q["entity_id"].values[qr[qi]]
        d_ids = d["entity_id"].values[dr[di]]
        not_own = np.fromiter((did not in own_matches.get(qid, ()) for qid, did in zip(q_ids, d_ids)),
                              dtype=bool, count=len(qi))
        qi, di, rk = qi[not_own], di[not_own], rk[not_own]
        pos = within_group_rank(qi.astype(np.int64), -rk.astype(np.float32))
        sel = pos < k
        hard[mine_rows[dr[di[sel]]]] = True
    return hard & ~in_m


def build_train_sample():
    if is_done(f"{SAMPLE_REL}/sample"):
        print(f"train sample {SAMPLE_REL} already built -- reusing")
        return
    with stage("sample", fraction=cfg.DATA_FRACTION) as st:
        s1_col, s1_int = read_ids("train", 1)
        keep_s1 = hash_u01(s1_int, SALT["s1"]) < cfg.DATA_FRACTION
        gt = load_ground_truth()
        assert len(gt) == len(s1_int) and gt["source1_entity_id"].is_unique, "GT must have one row per train S1"
        all_s1, all_pool = explode_gt(gt)
        assert pd.Index(all_pool).is_unique, "a pool record is matched to more than one S1 -- exclusivity assumption broken"
        kept_ids = set(s1_col.filter(pa.array(keep_s1)).to_pylist())
        gt_s = gt[gt["source1_entity_id"].isin(kept_ids)].reset_index(drop=True)
        gs1, gpool = explode_gt(gt_s)
        own = defaultdict(set)
        for a, b in zip(gs1, gpool):
            own[a].add(b)
        m_int = np.unique(ids_to_int(gpool))
        masks = {}
        for s in (2, 3):
            _, ids_int = read_ids("train", s)
            in_m = np.isin(ids_int, m_int)
            dist = ~in_m & (hash_u01(ids_int, SALT["pool"]) < cfg.POOL_DISTRACTOR_FRACTION)
            masks[s] = dict(ids_int=ids_int, in_m=in_m, dist=dist, hard=np.zeros(len(ids_int), bool))
        assert sum(int(m["in_m"].sum()) for m in masks.values()) == len(m_int), "some GT matches missing from pool files"
        if cfg.HARD_NEIGHBOR_K > 0:
            q = _light_norm(read_source("train", 1, ["entity_id", "name_clean", "extracted_city", "country"], keep_s1))
            for s in (2, 3):
                with stage(f"sample.hard_neighbours.S{s}"):
                    m = masks[s]
                    m["hard"] = mine_hard_neighbours(q, own, s, m["ids_int"], m["in_m"], cfg.HARD_NEIGHBOR_K) & ~m["dist"]
        s1_raw = read_source("train", 1, RAW_COLS, keep_s1)
        parts = []
        for s in (2, 3):
            m = masks[s]
            keep = m["in_m"] | m["dist"] | m["hard"]
            df = read_source("train", s, RAW_COLS, keep)
            df["pool_source"] = f"S{s}"
            df["samp_match"] = m["in_m"][keep].astype(np.int8)
            df["samp_dist"] = m["dist"][keep].astype(np.int8)
            df["samp_hard"] = m["hard"][keep].astype(np.int8)
            parts.append(df)
        pool_raw = pd.concat(parts, ignore_index=True)
        # ---- integrity checks (fail loudly) ----
        assert s1_raw["entity_id"].is_unique and set(s1_raw["entity_id"]) == kept_ids
        assert pool_raw["entity_id"].is_unique
        assert set(gpool) <= set(pool_raw["entity_id"]), "GT closure broken"
        comp = pool_raw.groupby("pool_source")[["samp_match", "samp_dist", "samp_hard"]].sum()
        print(f"sampled S1 entities: {len(s1_raw):,} of {len(s1_int):,} ({len(s1_raw) / len(s1_int):.4%})  "
              f"per country: {s1_raw['country'].value_counts().to_dict()}")
        print(f"pool records: {len(pool_raw):,}  (true matches={len(gpool):,})\n{comp.to_string()}")
        print(f"GT pairs: {len(gpool):,}  singletons among sampled S1: {int((gt_s['matched_entity_ids'] == '').sum()):,}")
        save_parquet(f"{SAMPLE_REL}/train_s1_raw.parquet", s1_raw)
        save_parquet(f"{SAMPLE_REL}/train_pool_raw.parquet", pool_raw)
        save_parquet(f"{SAMPLE_REL}/gt.parquet", gt_s)
        info = dict(n_s1=len(s1_raw), n_pool=len(pool_raw), n_true_pairs=len(gpool),
                    composition=comp.to_dict(), s1_per_country=s1_raw["country"].value_counts().to_dict())
        st.add(**{k: v for k, v in info.items() if k != "composition"})
        save_json(f"reports/sample_{FP_SAMPLE}.json", info)
        mark_done(f"{SAMPLE_REL}/sample", **{k: v for k, v in info.items() if isinstance(v, int)})


def test_masks(s):
    _, ids_int = read_ids("test", s)
    if cfg.TEST_FRACTION >= 1.0:
        return np.ones(len(ids_int), dtype=bool)
    return hash_u01(ids_int, SALT["test_s1" if s == 1 else "test_pool"]) < cfg.TEST_FRACTION


def raw_frame(split, role, columns):
    """Raw columns for a frame, in the canonical order (S1 file order; pool = S2 rows then S3 rows)."""
    if split == "train":
        rel = f"{SAMPLE_REL}/train_{role}_raw.parquet"
        p = locate(rel)
        if p is None:
            raise FileNotFoundError(f"{rel} missing -- run the 'sample' stage first")
        cols = columns + [c for c in ("pool_source", "samp_match", "samp_dist", "samp_hard") if role == "pool"]
        return load_parquet(p, columns=cols)
    if role == "s1":
        return read_source("test", 1, columns, test_masks(1))
    parts = []
    for s in (2, 3):
        df = read_source("test", s, columns, test_masks(s))
        df["pool_source"] = f"S{s}"
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


_IDS_CACHE = {}


def frame_ids(split, role):
    key = (split, role)
    if key not in _IDS_CACHE:
        _IDS_CACHE[key] = ids_to_int(raw_frame(split, role, ["entity_id"])["entity_id"].values)
    return _IDS_CACHE[key]


def normalized_frame(split, role):
    """Cached normalized frame: train sample in CACHE_DIR (small), test in SCRATCH_DIR (rebuilt per session)."""
    base = CACHE_DIR if split == "train" else SCRATCH_DIR
    rel = f"frames/{FP_SAMPLE if split == 'train' else FP_TEST}/{split}_{role}_norm.parquet"
    p = os.path.join(base, rel)
    if split == "train" and is_done(rel):   # also found in earlier sessions' outputs
        return load_parquet(locate(rel))
    if os.path.exists(p + ".done"):
        return load_parquet(p)
    with stage(f"normalize.{split}.{role}"):
        if split == "train" or role == "s1":
            df = normalize_frame(raw_frame(split, role, RAW_COLS))
        else:   # test pool: one source file at a time to bound peak RAM
            parts = []
            for s in (2, 3):
                raw = read_source("test", s, RAW_COLS, test_masks(s))
                raw["pool_source"] = f"S{s}"
                parts.append(normalize_frame(raw))
                del raw
                gc.collect()
            df = pd.concat(parts, ignore_index=True)
        save_parquet(rel, df, base=base)
        save_json(rel + ".done", dict(rows=len(df)), base=base)
    return df