def _load_reports():
    out = {}
    for base in CACHE_SEARCH_PATHS[::-1] + [CACHE_DIR]:   # later sessions override earlier ones
        for p in glob.glob(os.path.join(base, "reports", "*.json")):
            try:
                with open(p) as f:
                    out[os.path.basename(p)[:-5]] = json.load(f)
            except (OSError, ValueError):
                pass
    return out


def _all_stage_logs():
    recs = []
    paths = [RUN_LOG_PATH]
    for p in dict.fromkeys(paths):
        if os.path.exists(p):
            with open(p) as f:
                recs += [json.loads(l) for l in f if l.strip()]
    return recs


def stage_report():
    reps = _load_reports()
    logs = _all_stage_logs()
    samp = reps.get(f"sample_{FP_SAMPLE}", {})
    rd, rf = reps.get("recall_diag") or next((v for k, v in reps.items() if k.startswith("recall_diag_")), {}), \
        reps.get("recall_train_full", {})
    mt, te = reps.get("model_train", {}), next((v for k, v in reps.items() if k.startswith("train_eval_")), {})
    ti, sub = reps.get("test_inference", {}), reps.get("submission", {})
    audits = {}
    if cfg.AUDIT_LABELS_PATH:
        for pth in sorted(glob.glob(cfg.AUDIT_LABELS_PATH, recursive=True)):
            audits[pth] = score_audit(pth)

    def _retr_secs(ctx_name, keys):
        tot = 0.0
        for k, v in reps.items():
            if k.startswith(f"retrieval_seconds_{ctx_name}_"):
                tot += sum(float(v.get(x, 0)) for x in keys)
        return round(tot, 1)

    def _recall_row(rep, name):
        for row in rep.get("table", []):
            if row["retriever"] == name:
                return {k: v for k, v in row.items() if k.startswith("R@")}
        return None

    stage_secs = defaultdict(float)
    for r in logs:
        stage_secs[r["stage"]] += float(r.get("seconds", 0))
    report = {
        "profile": PROFILE,
        "f05_progress": reps.get("f05_progress"),
        "country_invariant": {k: (reps.get("country_invariant") or {}).get(k) for k in
                              ("positive_pairs_checked", "cross_country_pairs", "pairs_with_empty_country",
                               "test_countries_unseen_in_train")},
        "gate": reps.get("gate1"),
        "records": dict(train_s1=samp.get("n_s1"), train_pool=samp.get("n_pool"), train_true_pairs=samp.get("n_true_pairs"),
                        test_s1=ti.get("test_s1"), test_pool=ti.get("test_pool")),
        "embeddings": dict(dims=EMB_DIMS, name_weight=cfg.EMB_NAME_WEIGHT, weight_sweep=reps.get("weight_sweep") or next(
            (v for k, v in reps.items() if k.startswith("weight_sweep_")), None), sanity=next(
            (v for k, v in reps.items() if k.startswith("embedding_sanity_")), None)),
        "diagnostic_recall": dict(per_retriever={rn: _recall_row(rd, rn) for rn in RETRIEVERS + ["exact (any key)",
                                  "UNION (diagnostic depth)", "UNION (configured top-Ks)"] if _recall_row(rd, rn)},
                                  union_recall_at_cap=rd.get("union_recall_at_cap"), oracle_macro_f05=rd.get("oracle_macro_f05"),
                                  cap_table=rd.get("cap_table"), reverse_k_table=rd.get("reverse_k_table"),
                                  leave_one_out_recall_drop=rd.get("leave_one_out_recall_drop")),
        "full_train_recall": rf or None,
        "model": dict(params=mt.get("params"), rule=mt.get("rule"), grid_scores=mt.get("grid_scores"),
                      cv_macro_f05=mt.get("cv_macro_f05"), training_pairs=mt.get("training_pairs"),
                      holdout={kk: (mt.get("holdout") or {}).get(kk) for kk in
                               ("macro_f05", "micro_precision", "micro_recall", "micro_f1", "tp", "fp", "fn")},
                      plain_global_threshold_holdout_f05=mt.get("plain_global_holdout_macro_f05"),
                      leakage_checks=mt.get("leakage_checks"), top_features=mt.get("top_features")),
        "train_evaluation": {kk: te.get(kk) for kk in ("entities", "macro_f05", "micro_precision", "micro_recall", "micro_f1",
                                                       "tp", "fp", "fn", "by_country_macro_f05", "by_source",
                                                       "entities_all_train_macro_f05")} if te else None,
        "test": {kk: ti.get(kk) for kk in ("rule", "candidate_pairs", "predicted_matches", "cands_per_query",
                                          "matches_per_entity", "entities_with_match", "by_country", "audit_files",
                                          "cross_country_candidates")} if ti else None,
        "audit_scores": audits or None,
        "seconds": dict(dense_retrieval=_retr_secs("train", ["dense", "dense_rev", "dense_global"]) + _retr_secs("test", ["dense", "dense_rev", "dense_global"]),
                        bm25=_retr_secs("train", ["bm25"]) + _retr_secs("test", ["bm25"]),
                        tfidf=_retr_secs("train", ["tfidf"]) + _retr_secs("test", ["tfidf"]),
                        reranker_train=mt.get("train_seconds"), test_features=ti.get("feature_seconds"),
                        test_rerank=ti.get("rerank_seconds"), per_stage={k: round(v, 1) for k, v in stage_secs.items()}),
        "total_runtime_seconds_all_sessions": round(sum(
            max(r.get("t_session", 0) for r in logs if r.get("session") == sid) for sid in {r.get("session") for r in logs}), 1),
        "sessions": len({r.get("session") for r in logs}),
        "this_session_seconds": round(time.time() - T0, 1),
        "peak_gpu_memory_gb": max([r["gpu_peak_gb"] for r in logs if r.get("gpu_peak_gb") is not None], default=None),
        "peak_ram_gb": max([r["peak_rss_gb"] for r in logs if r.get("peak_rss_gb") is not None], default=None),
        "disk_usage_gb": dict(cache=dir_size_gb(CACHE_DIR), scratch=dir_size_gb(SCRATCH_DIR), output=dir_size_gb(OUTPUT_DIR)),
        "submission": sub or None,
    }
    path = os.path.join(OUTPUT_DIR, "run_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))
    show_f05_progress()
    log(f"run report written to {path}")
    return report
