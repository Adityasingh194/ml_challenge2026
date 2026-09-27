def run_self_tests():
    with stage("selftest"):
        tmp = os.path.join(SCRATCH_DIR, "selftest")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        # 1. metric edge cases
        n_true = np.array([0, 0, 2, 1])
        s1p = np.array([1, 2])          # entity 0: empty/no-truth -> 1; 1: FP with no truth -> 0;
        tp = np.array([False, True])    # entity 2: 1 of 2 correct -> P=1,R=.5 -> 0.8333; entity 3: missed -> 0
        got = macro_f05(s1p, tp, n_true, np.ones(4, bool))
        assert abs(got - (1 + 0 + 1.25 * 0.5 / (0.25 + 0.5) + 0) / 4) < 1e-9, got
        # 2. BM25 vs brute force
        docs = pd.DataFrame({"name_core": ["alpha beta", "beta gamma", "alpha alpha delta", "omega"],
                             "addr_clean": ["", "", "", ""], "city_norm": ["x", "x", "y", "x"], "state_norm": ["", "", "", ""]})
        qs = pd.DataFrame({"name_core": ["alpha delta"], "addr_clean": [""], "city_norm": ["y"], "state_norm": [""]})
        old = (cfg.BM25_MIN_DF_CAP, cfg.BM25_MAX_DF_FRAC)
        cfg.BM25_MIN_DF_CAP, cfg.BM25_MAX_DF_FRAC = 10, 1.0
        qi, di, sc, rk = bm25_search(docs, np.arange(4), qs, np.arange(1), 4, fields=("n", "c"))
        toks = _bm25_tokens(docs, range(4), ("n", "c"))
        qt = _bm25_tokens(qs, range(1), ("n",))[0]
        qtc = _bm25_tokens(qs, range(1), ("c",))[0]
        N, avgdl = 4, np.mean([len(t) for t in toks])
        brute = []
        for d in toks:
            s_ = 0.0
            for t, w in [(t, cfg.BM25_FIELD_WEIGHTS["n"]) for t in set(qt)] + [(t, cfg.BM25_FIELD_WEIGHTS["c"]) for t in set(qtc)]:
                df_ = sum(t in x for x in toks)
                tf = d.count(t)
                if tf:
                    idf = math.log1p((N - df_ + 0.5) / (df_ + 0.5))
                    s_ += w * idf * tf * (cfg.BM25_K1 + 1) / (tf + cfg.BM25_K1 * (1 - cfg.BM25_B + cfg.BM25_B * len(d) / avgdl))
            brute.append(s_)
        cfg.BM25_MIN_DF_CAP, cfg.BM25_MAX_DF_FRAC = old
        got = dict(zip(di.tolist(), sc.tolist()))
        for j, b in enumerate(brute):
            assert abs(got.get(j, 0.0) - b) < 1e-4, (j, got, brute)
        assert di[np.argmax(sc)] == 2
        # 3. union dedupe / provenance / cap
        ra = pd.DataFrame({"s1_pos": [0, 0, 0, 1], "pool_pos": [5, 6, 7, 5], "score": [.9, .8, .7, .6], "rank": [0, 1, 2, 0]})
        rb = pd.DataFrame({"s1_pos": [0, 0], "pool_pos": [7, 8], "score": [3.0, 2.0], "rank": [0, 1]})
        U = build_union({"dense": ra, "bm25": rb}, 10, cap=3)
        u0 = U[U.s1_pos == 0].set_index("pool_pos")
        # best ranks: 5->0, 7->0 (bm25 rank 0, found twice), 6->1, 8->1; ties broken by n_retrievers then dense
        # score, so the cap of 3 keeps {7, 5, 6} and drops 8
        assert set(u0.index) == {5, 6, 7}, set(u0.index)
        assert u0.loc[7, "pos_in_entity"] == 0 and u0.loc[5, "pos_in_entity"] == 1
        assert u0.loc[7, "by_dense"] == 1 and u0.loc[7, "by_bm25"] == 1 and u0.loc[7, "n_retrievers"] == 2
        assert u0.loc[7, "dense_rank"] == 2 and u0.loc[7, "bm25_rank"] == 0 and u0.loc[5, "bm25_rank"] == RANK_SENTINEL
        assert len(U[U.s1_pos == 1]) == 1
        # 4. sharded == unsharded dense search
        rng = np.random.default_rng(0)
        A = rng.standard_normal((50, 16)).astype(np.float32)
        Bm = rng.standard_normal((300, 16)).astype(np.float32)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        Bm /= np.linalg.norm(Bm, axis=1, keepdims=True)

        class _Arr:
            def __init__(self, M):
                self.M, self.dim = M, M.shape[1]

            def gather(self, rows, out_dtype=np.float16):
                return self.M[rows].astype(out_dtype)

        old_shard = cfg.SEARCH_SHARD_ROWS
        cfg.SEARCH_SHARD_ROWS = 70
        s_sh, i_sh = dense_search(_Arr(A), np.arange(50), _Arr(Bm), np.arange(300), 10)
        cfg.SEARCH_SHARD_ROWS = old_shard
        ref = np.argsort(-(A @ Bm.T), axis=1)[:, :10]
        assert (np.sort(i_sh, 1) == np.sort(ref, 1)).mean() > 0.97, "sharded search disagrees with brute force"
        # 5. embedding store resume + int8 round trip
        ids = np.arange(10, 35, dtype=np.int64)
        V = rng.standard_normal((25, 8)).astype(np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        tag = f"selftest_{int(time.time() * 1000)}"
        stx = EmbStore(tag, ids, 8, "int8", 10)
        stx.write(0, V[0:10])
        stx.write(1, V[10:20])                       # "crash" before chunk 2
        assert stx.missing() == [2]
        stx2 = EmbStore(tag, ids, 8, "int8", 10)     # restart: only chunk 2 is missing
        assert stx2.missing() == [2]
        stx2.write(2, V[20:25])
        G = stx2.gather(np.array([24, 3, 11]), 8, np.float32)
        assert np.min((G * V[[24, 3, 11]]).sum(1)) > 0.999, "int8 round trip too lossy"
        G4 = stx2.gather(np.array([5]), 4, np.float32)
        ref4 = V[5, :4] / np.linalg.norm(V[5, :4])
        assert np.allclose(G4[0], ref4, atol=2e-2), "MRL slice + renormalize wrong"
        stx._mm.clear()
        stx2._mm.clear()   # release memmaps first (Windows cannot delete mapped files)
        del stx, stx2, G, G4
        gc.collect()
        shutil.rmtree(os.path.join(CACHE_DIR, "embeddings", tag), ignore_errors=True)
        # 6. nested sampling
        idsn = np.arange(1, 200_001, dtype=np.int64) + 2 * ID_BASE
        a1, a2 = hash_u01(idsn, 1) < 0.01, hash_u01(idsn, 1) < 0.02
        assert (a1 <= a2).all() and 0.008 < a1.mean() < 0.012
        # 7. strict validator rejects bad files
        exp = ["S1-1", "S1-2"]
        pool_ok = {"S2-10", "S3-11", "S2-12"}

        def _w(path, rows, col):
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(f"source1_entity_id\t{col}\n" + "".join(f"{a}\t{b}\n" for a, b in rows))

        mp, cp = os.path.join(tmp, "m.tsv"), os.path.join(tmp, "c.tsv")
        _w(mp, [("S1-1", "S2-10"), ("S1-2", "")], "matched_entity_ids")
        _w(cp, [("S1-1", "S2-10,S3-11"), ("S1-2", "S2-12")], "candidate_entity_ids")
        validate_submission_strict(mp, cp, exp, pool_ok, True, 2)
        bad_cases = {
            "dup s1": ([("S1-1", "S2-10"), ("S1-1", "")], [("S1-1", "S2-10"), ("S1-1", "")]),
            "missing row": ([("S1-1", "S2-10")], [("S1-1", "S2-10")]),
            "invalid id": ([("S1-1", "S2-99"), ("S1-2", "")], [("S1-1", "S2-99"), ("S1-2", "")]),
            "not subset": ([("S1-1", "S3-11"), ("S1-2", "")], [("S1-1", "S2-10"), ("S1-2", "")]),
            "order": ([("S1-2", ""), ("S1-1", "S2-10")], [("S1-2", ""), ("S1-1", "S2-10")]),
            "exclusive": ([("S1-1", "S2-10"), ("S1-2", "S2-10")], [("S1-1", "S2-10"), ("S1-2", "S2-10")]),
            "nan": ([("S1-1", "nan"), ("S1-2", "")], [("S1-1", "nan"), ("S1-2", "")]),
        }
        for name, (mr_, cr_) in bad_cases.items():
            _w(mp, mr_, "matched_entity_ids")
            _w(cp, cr_, "candidate_entity_ids")
            try:
                validate_submission_strict(mp, cp, exp, pool_ok, True, 2)
            except SubmissionValidationError:
                continue
            raise AssertionError(f"validator accepted a bad file: {name}")
        # 8. reverse retrieval: each pool record lists its nearest S1 in rank order, and truncating the list
        #    recomputes the per-S1 order_rank the union uses (same as production)
        S1v = rng.standard_normal((6, 8)).astype(np.float32)
        S1v /= np.linalg.norm(S1v, axis=1, keepdims=True)
        PLv = rng.standard_normal((9, 8)).astype(np.float32)
        PLv /= np.linalg.norm(PLv, axis=1, keepdims=True)
        s_r, i_r = dense_search(_Arr(PLv), np.arange(9), _Arr(S1v), np.arange(6), 3)
        Rv = _topk_frame(np.arange(9), np.arange(6), s_r, i_r, reverse=True)
        sims = PLv @ S1v.T
        assert len(Rv) == 27 and all(r.s1_pos in np.argsort(-sims[r.pool_pos])[:3] for r in Rv.itertuples())
        assert all(abs(r.score - sims[r.pool_pos, r.s1_pos]) < 1e-2 for r in Rv.itertuples())
        Rv["order_rank"] = within_group_rank(Rv["s1_pos"].values, Rv["score"].values).astype(np.int32)
        Tv = truncate_retr({"dense_rev": Rv}, {"dense_rev": 1})["dense_rev"]
        assert (Tv["rank"] == 0).all() and len(Tv) == 9
        assert all(sorted(g) == list(range(len(g))) for g in Tv.groupby("s1_pos")["order_rank"].apply(list))
        # 9. candidate cap: never more than `cap` per S1, and the survivors are the best-ranked ones
        rr_ = pd.DataFrame({"s1_pos": np.repeat(np.arange(20), 30), "pool_pos": rng.integers(0, 500, 600),
                            "score": rng.random(600).astype(np.float32)}).drop_duplicates(["s1_pos", "pool_pos"])
        rr_["rank"] = within_group_rank(rr_["s1_pos"].values, rr_["score"].values).astype(np.int32)
        Uc = build_union({"dense": rr_}, 500, cap=7)
        assert Uc.groupby("s1_pos").size().max() == 7 and (Uc["dense_rank"] < 7).all() and Uc["s1_pos"].nunique() == 20
        # 10. country buckets: partitions never mix countries or sources; the cross-country counter is exact
        saved = dict(FR.__dict__)
        try:
            FR.s1 = pd.DataFrame({"country_norm": np.array(["US", "India", "US", "France"], dtype=object)})
            FR.pool = pd.DataFrame({"country_norm": np.array(["India", "US", "US", "France", "France"], dtype=object),
                                    "pool_source": np.array(["S2", "S2", "S3", "S2", "S3"], dtype=object)})
            FR.s1_cc, FR.pool_cc = country_codes(FR.s1["country_norm"].values, FR.pool["country_norm"].values)
            for c_, s_, qr_, pr_, rv_ in partitions(SimpleNamespace(q=np.arange(4), pool=np.arange(5), rev=np.arange(4))):
                assert set(FR.s1["country_norm"].values[qr_]) <= {c_} and set(FR.pool["country_norm"].values[pr_]) <= {c_}
                assert set(FR.s1["country_norm"].values[rv_]) <= {c_} and (FR.pool["pool_source"].values[pr_] == s_).all()
            assert cross_country_count(np.array([0, 1, 3, 3]), np.array([1, 0, 3, 4])) == 0
            assert cross_country_count(np.array([0, 1]), np.array([0, 1])) == 2
        finally:
            FR.__dict__.clear()
            FR.__dict__.update(saved)
        # 11. embedding dimension consistency: a view never reads more dims than stored, and a cache written with
        #     other settings is refused instead of silently reused
        tag2 = f"selftest_dim_{int(time.time() * 1000)}"
        st3 = EmbStore(tag2, ids, 8, "float16", 10)
        for bad_fn, exc in ((lambda: EmbView([st3], 16), AssertionError), (lambda: EmbStore(tag2, ids, 16, "float16", 10), RuntimeError)):
            try:
                bad_fn()
            except exc:
                continue
            raise AssertionError("embedding dimension mismatch was not detected")
        shutil.rmtree(os.path.join(CACHE_DIR, "embeddings", tag2), ignore_errors=True)
        # 12. scale-phase gating: with ENABLE_SCALE_PHASE=False no scale stage may start
        old_sp = cfg.ENABLE_SCALE_PHASE
        cfg.ENABLE_SCALE_PHASE = False
        try:
            assert not any(scale_phase_allowed(s_, quiet=True) for s_ in SCALE_STAGES)
        finally:
            cfg.ENABLE_SCALE_PHASE = old_sp
        # 13. decision rule: the simplest rule wins unless a complex one is better by more than the tolerance
        rows_t = [(dict(kind="relative", floor=.5, gap_mult=2., exclusive=True), 0.9005), (dict(kind="global", thr=.5), 0.9000),
                  (dict(kind="global", thr=.6, exclusive=True), 0.8990)]
        assert select_rule(rows_t, 0.001)[0] == dict(kind="global", thr=.5)
        assert select_rule(rows_t, 0.0)[0]["kind"] == "relative"
        # 14. GPU BM25 == CPU BM25 (same pairs and scores) on a random partition
        if DEVICE == "cuda":
            words = np.array([f"w{i}" for i in range(400)], dtype=object)
            mk = lambda n: pd.DataFrame({"name_core": [" ".join(rng.choice(words, 3)) for _ in range(n)],
                                         "addr_clean": [" ".join(rng.choice(words, 5)) for _ in range(n)],
                                         "city_norm": [""] * n, "state_norm": [""] * n})
            dd, qq = mk(3000), mk(700)
            old = (cfg.BM25_MIN_DF_CAP, cfg.BM25_MAX_DF_FRAC)
            cfg.BM25_MIN_DF_CAP, cfg.BM25_MAX_DF_FRAC = 50, 0.02
            rc = bm25_search(dd, np.arange(3000), qq, np.arange(700), 20, use_gpu=False)
            rg = bm25_search(dd, np.arange(3000), qq, np.arange(700), 20, use_gpu=True)
            cfg.BM25_MIN_DF_CAP, cfg.BM25_MAX_DF_FRAC = old
            kc = pd.Series(rc[2], index=rc[0] * 10_000 + rc[1])
            kg = pd.Series(rg[2], index=rg[0] * 10_000 + rg[1])
            assert len(kc) == len(kg), (len(kc), len(kg))
            # identical per-query score lists (ties at the k-th place may pick different documents)
            sc_c = pd.DataFrame({"q": rc[0], "s": rc[2]}).sort_values(["q", "s"]).reset_index(drop=True)
            sc_g = pd.DataFrame({"q": rg[0], "s": rg[2]}).sort_values(["q", "s"]).reset_index(drop=True)
            assert (sc_c["q"] == sc_g["q"]).all() and np.allclose(sc_c["s"], sc_g["s"], atol=1e-4), "GPU BM25 != CPU BM25"
            # documents scoring strictly above a query's k-th score must be identical (ties at the k-th may differ)
            kth = sc_c.groupby("q")["s"].min()
            above_c = {int(k) for k, s in kc.items() if s > kth[int(k) // 10_000] + 1e-4}
            above_g = {int(k) for k, s in kg.items() if s > kth[int(k) // 10_000] + 1e-4}
            assert above_c == above_g, "GPU BM25 returns different documents than CPU BM25"
            common = kc.index.intersection(kg.index)
            assert np.allclose(kc[common], kg[common], atol=1e-4)
        # 15. multi-GPU sharded dense search == brute force (pool large enough to be split over the GPUs)
        A2 = rng.standard_normal((300, 32)).astype(np.float32)
        B2 = rng.standard_normal((MULTI_GPU_MIN_POOL + 7000, 32)).astype(np.float32)
        A2 /= np.linalg.norm(A2, axis=1, keepdims=True)
        B2 /= np.linalg.norm(B2, axis=1, keepdims=True)
        old_shard = cfg.SEARCH_SHARD_ROWS
        cfg.SEARCH_SHARD_ROWS = 9000
        s_m, i_m = dense_search(_Arr(A2), np.arange(300), _Arr(B2), np.arange(len(B2)), 10)
        cfg.SEARCH_SHARD_ROWS = old_shard
        ref2 = np.argsort(-(A2 @ B2.T), axis=1)[:, :10]
        assert (np.sort(i_m, 1) == np.sort(ref2, 1)).mean() > 0.97, "multi-GPU dense search disagrees with brute force"
        # 16. combined vector: <u, v> = w cos(name) + (1 - w) cos(addr)
        class _FakeView:
            def __init__(self, M):
                self.M, self.dim, self.n = M, M.shape[1], len(M)

            def gather(self, rows, use_dim=None, out_dtype=np.float16):
                X = self.M[rows]
                return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(out_dtype)
        Nm, Ad = rng.standard_normal((5, 6)), rng.standard_normal((5, 10))
        cs = CombinedStore.__new__(CombinedStore)
        cs.name, cs.addr, cs.w = _FakeView(Nm), _FakeView(Ad), 0.3
        cs.dim, cs.n, cs._a, cs._b = 16, 5, math.sqrt(0.3), math.sqrt(0.7)
        G = cs.gather(np.arange(5), None, np.float32)
        nn_, aa_ = _FakeView(Nm).gather(np.arange(5), None, np.float32), _FakeView(Ad).gather(np.arange(5), None, np.float32)
        assert np.allclose(G @ G.T, 0.3 * nn_ @ nn_.T + 0.7 * aa_ @ aa_.T, atol=1e-5) and np.allclose(np.linalg.norm(G, axis=1), 1)
        # 17. raw-field views: name_clean / addr_clean = lower-cased raw text, city / state empty
        tv = arrow_to_frame(read_raw_table(SRC_FILE[("train", 1)], ["business_name", "name_clean", "addr_clean",
                                                                     "business_address", "extracted_city"]).slice(0, 50))
        assert (tv["name_clean"] == tv["business_name"].str.lower()).all()
        assert (tv["addr_clean"] == tv["business_address"].str.lower()).all() and (tv["extracted_city"] == "").all()
        shutil.rmtree(tmp, ignore_errors=True)
        log("self-tests passed (17)")