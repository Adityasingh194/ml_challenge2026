class SubmissionValidationError(RuntimeError):
    pass


def build_id_lists(s1_pos, ids, score, n_s1):   # copied from er_crossencoder_v2 §15
    order = np.lexsort((-score, s1_pos))
    s_sorted, ids_sorted = s1_pos[order], ids[order]
    bounds = np.searchsorted(s_sorted, np.arange(n_s1 + 1))
    return [",".join(ids_sorted[bounds[i]:bounds[i + 1]]) for i in range(n_s1)]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def validate_submission_strict(match_path, cand_path, expected_ids, valid_pool_ids, exclusive, expected_rows):
    errors = []

    def err(msg):
        if len(errors) < 50:
            errors.append(msg)

    seen_matched = set()
    with open(match_path, newline="", encoding="utf-8") as fm, open(cand_path, newline="", encoding="utf-8") as fc:
        rm, rc = csv.reader(fm, delimiter="\t"), csv.reader(fc, delimiter="\t")
        hm, hc = next(rm, None), next(rc, None)
        if hm != ["source1_entity_id", "matched_entity_ids"]:
            err(f"matching_results: bad header {hm}")
        if hc != ["source1_entity_id", "candidate_entity_ids"]:
            err(f"candidate_pairs: bad header {hc}")
        n = 0
        seen_s1 = set()
        for n, (row_m, row_c) in enumerate(itertools.zip_longest(rm, rc), start=1):
            ln = n + 1
            if row_m is None or row_c is None:
                err(f"line {ln}: the two files have different row counts")
                break
            if len(row_m) != 2 or len(row_c) != 2:
                err(f"line {ln}: expected 2 tab-separated fields (got {len(row_m)} / {len(row_c)})")
                continue
            (sid, mlist), (sid_c, clist) = row_m, row_c
            if n > len(expected_ids) or sid != expected_ids[n - 1] or sid_c != expected_ids[n - 1]:
                err(f"line {ln}: source1 id order mismatch ({sid} / {sid_c} vs expected "
                    f"{expected_ids[n - 1] if n <= len(expected_ids) else 'EOF'})")
            if sid in seen_s1:
                err(f"line {ln}: duplicate source1_entity_id {sid}")
            seen_s1.add(sid)
            m_ids = mlist.split(",") if mlist else []
            c_ids = clist.split(",") if clist else []
            c_set = set(c_ids)
            for name, lst, st_ in (("matched", m_ids, set(m_ids)), ("candidate", c_ids, c_set)):
                if len(lst) != len(st_):
                    err(f"line {ln}: duplicate ids within {name} list of {sid}")
                bad = [x for x in lst if x in ("nan", "None", "NaN", "") or not (x.startswith("S2-") or x.startswith("S3-"))
                       or x not in valid_pool_ids]
                if bad:
                    err(f"line {ln}: {len(bad)} invalid {name} ids for {sid}, e.g. {bad[:3]}")
            if not set(m_ids) <= c_set:
                err(f"line {ln}: matches not a subset of candidates for {sid}")
            if exclusive:
                for x in m_ids:
                    if x in seen_matched:
                        err(f"line {ln}: pool id {x} matched to more than one source1 entity under an exclusive rule")
                    seen_matched.add(x)
    if n != expected_rows or n != len(expected_ids):
        err(f"row count {n:,} != expected {expected_rows:,}")
    missing = set(expected_ids) - seen_s1
    if missing:
        err(f"{len(missing):,} test source1 ids missing, e.g. {sorted(missing)[:3]}")
    if errors:
        raise SubmissionValidationError("submission validation FAILED:\n  - " + "\n  - ".join(errors))
    return dict(rows=n, matching_sha256=_sha256(match_path), candidate_sha256=_sha256(cand_path))


def _run_official_validator(match_path, cand_path):
    cands = [p for p in ["utils/validate_submission.py", "../utils/validate_submission.py"] if os.path.exists(p)]
    cands += [os.path.join(LOCAL_ROOT, p_) for p_ in ("utils/validate_submission.py", "validate_submission.py")
              if os.path.exists(os.path.join(LOCAL_ROOT, p_))]
    test_dirs = [d for d in ["dataset/test", "../dataset/test"] if os.path.isdir(d)]
    test_dirs += [os.path.join(DATA_DIR, "test")] if os.path.isdir(os.path.join(DATA_DIR, "test")) else []
    if not cands or not test_dirs or cfg.TEST_FRACTION < 1.0:
        print("official validator not found (or dry run) -- the strict in-notebook checks above apply the same rules")
        return None
    r = subprocess.run([sys.executable, cands[0], "--matching", match_path, "--candidate", cand_path,
                        "--test-dir", test_dirs[0]], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    if r.returncode != 0:
        raise SubmissionValidationError(f"official validator failed (exit {r.returncode})")
    return r.returncode


def stage_submit():
    if not scale_phase_allowed("submit"):
        return
    resolve_emb_weight()
    if load_model("train") is None:
        return skip("submit", "no trained model yet")
    with stage("submit") as st:
        mname, model = _final_model()
        ctx = make_ctx("test")
        sdir = f"scores_{model.meta['model_fp']}"
        if not is_done(ctx_rel(ctx, sdir, "final_preds.parquet")):
            return skip("submit", "no test predictions for the final model yet -- run test_inference")
        preds = ctx_read_parquet(ctx, sdir, "final_preds.parquet")
        n_test = len(FR.s1) - FR.n_tr_s1
        s1_local = preds["s1_pos"].values - FR.n_tr_s1
        assert s1_local.min() >= 0 and s1_local.max() < n_test, "prediction rows outside the test S1 range"
        assert (preds["pool_pos"].values >= FR.n_tr_pool).all(), "a train record leaked into test predictions"
        pool_ids = FR.pool["entity_id"].values
        pool_pos = preds["pool_pos"].values
        p, keep = preds["p"].values, preds["match"].values.astype(bool)
        mem_mark("submit: final test predictions", preds)
        test_ids = FR.s1["entity_id"].values[FR.n_tr_s1:]
        expected_rows = pq.ParquetFile(SRC_FILE[("test", 1)]).metadata.num_rows if cfg.TEST_FRACTION >= 1.0 else n_test
        if cfg.TEST_FRACTION >= 1.0:
            file_order = pq.read_table(SRC_FILE[("test", 1)], columns=["entity_id"]).column(0).to_pylist()
            if file_order != list(test_ids):
                raise SubmissionValidationError("test S1 frame order differs from test_source1_final.parquet order")
        out_dir = os.path.join(OUTPUT_DIR, "submission_1" if cfg.TEST_FRACTION >= 1.0 else "dryrun_NOT_SUBMITTABLE")
        os.makedirs(out_dir, exist_ok=True)
        match_path = os.path.join(out_dir, "matching_results.tsv")
        cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
        # written in S1 chunks (same bytes as one to_csv call): only one chunk's id lists are in memory at a time
        if not (np.diff(s1_local) >= 0).all():
            o = np.argsort(s1_local, kind="stable")
            s1_local, pool_pos, p, keep = s1_local[o], pool_pos[o], p[o], keep[o]
        B = cfg.QUERY_BLOCK_ROWS
        bounds = np.searchsorted(s1_local, np.arange(0, n_test + B, B).clip(max=n_test))
        for bi, a in enumerate(range(0, n_test, B)):
            b = min(a + B, n_test)
            r0, r1 = bounds[bi], bounds[bi + 1]
            s_, ids_, p_, k_ = s1_local[r0:r1] - a, pool_ids[pool_pos[r0:r1]], p[r0:r1], keep[r0:r1]
            mode, header = ("w", True) if a == 0 else ("a", False)
            pd.DataFrame({"source1_entity_id": test_ids[a:b], "matched_entity_ids": build_id_lists(s_[k_], ids_[k_], p_[k_], b - a)}
                         ).to_csv(match_path + ".tmp", sep="\t", index=False, quoting=csv.QUOTE_NONE, mode=mode, header=header)
            pd.DataFrame({"source1_entity_id": test_ids[a:b], "candidate_entity_ids": build_id_lists(s_, ids_, p_, b - a)}
                         ).to_csv(cand_path + ".tmp", sep="\t", index=False, quoting=csv.QUOTE_NONE, mode=mode, header=header)
            if bi == 0:
                mem_mark(f"submit: first TSV chunk written ({b - a:,} S1)")
        os.replace(match_path + ".tmp", match_path)
        os.replace(cand_path + ".tmp", cand_path)
        gc.collect()
        valid_pool = set(FR.pool["entity_id"].values[FR.n_tr_pool:])
        res = validate_submission_strict(match_path, cand_path, list(test_ids), valid_pool,
                                         bool(model.meta["rule"].get("exclusive")), expected_rows)
        _run_official_validator(match_path, cand_path)
        res.update(matching_path=match_path, candidate_path=cand_path, submittable=cfg.TEST_FRACTION >= 1.0,
                   config_fp=cfg_fp(sorted(CFG)), model=mname)
        log(f"VALIDATION PASSED: {res}")
        if cfg.TEST_FRACTION < 1.0:
            print("NOTE: TEST_FRACTION < 1 -- these files are a dry run and are NOT a valid submission.")
        REPORT["submission"] = res
        save_json("reports/submission.json", res)
        st.add(rows=res["rows"])