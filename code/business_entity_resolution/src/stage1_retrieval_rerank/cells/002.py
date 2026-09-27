import os

LOCAL_ROOT = os.environ.get("ER_ROOT", "/home/parth/Desktop/trial work")   # same as cell 0a
PROFILE = os.environ.get("ER_PROFILE", "full")        # "full" = 100% train + 100% test, "smoke" = tiny end-to-end check
STAGES_TO_RUN = os.environ.get("ER_STAGES", "all")    # "all" or comma list of stage names (see ALL_STAGES in §3)

CFG = dict(
    # ---- data volume (deterministic hash sampling; 1.0 = everything) ----
    DATA_FRACTION=1.0,              # share of TRAIN Source-1 entities (all their true matches always kept)
    TEST_FRACTION=1.0,              # 1.0 = full test = a submittable file; <1 = dry run, never named a submission
    POOL_DISTRACTOR_FRACTION=None,  # share of the other train S2/S3 records kept (None = DATA_FRACTION -> 1.0 = all)
    HARD_NEIGHBOR_K=0,              # only useful when DATA_FRACTION < 1 (the full pool already holds every neighbour)
    HARD_NEIGHBOR_POOL_FRACTION=1.0,
    RANDOM_SEED=42,

    # ---- phases and gates ----
    ENABLE_SCALE_PHASE=True,        # test inference + submission (still refused when the train-evaluation gate fails)
    GATE1_MIN_UNION_RECALL=0.99,    # configured-union recall (full train pool) needed before the test phase may start
    COUNTRY_INVARIANT_MAX_VIOLATIONS=0,  # cross-country positive pairs tolerated before the run STOPS
    FORCE_PAST_GATE=False,

    # ---- precomputed embeddings (Qwen3-Embedding-0.6B, computed elsewhere; no model runs here) ----
    EMB_DIR=os.path.join(LOCAL_ROOT, "embedding_store"),   # {split}/{s1,s2,s3}/{name,addr}/ (cell 0c)
    EMB_NAME_WEIGHT="auto",         # dense score = w * cos(name) + (1 - w) * cos(address); "auto" = best w of the sweep
    EMB_WEIGHT_SWEEP=[0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0],   # dense Recall@K per w on the diagnostic queries
    EMBED_CHUNK_ROWS=500_000,       # rows per resumable chunk of the char TF-IDF vector cache

    # ---- hardware ----
    NUM_GPUS="auto",                # "auto" = every visible GPU (2 here)
    NUM_WORKERS=20,                 # CPU processes (normalization, features, tf-idf); 24 cores, 4 left for the parent + OS
    RAM_BUDGET_GB=28,               # feature workers are capped so parent RSS + workers stay under this
    MAX_SESSION_HOURS=4.0,          # long stages stop cleanly after this; run again to continue from the cache
    DISK_MARGIN_GB=1.0,
    BM25_USE_GPU=True,              # sparse x sparse BM25 products + top-k on the GPUs (False = CPU scipy path)

    # ---- retrieval ----
    DENSE_TOP_K=50,                 # per pool source (S2 and S3 each), forward S1->pool
    REV_TOP_K=3,                    # reverse pool->S1 (diag table: 1-3 keep more true matches in the cap than 10)
    REV_DIAG_TOP_K=50,              # reverse depth on the diagnostic context, to check that REV_TOP_K suffices
    ENABLE_GLOBAL_FALLBACK=False,   # cross-country dense pass. OFF: 0 of 7,638,365 train positive pairs cross a country
    GLOBAL_FALLBACK_TOP_K=5,
    BM25_TOP_K=50,                  # per pool source
    BM25_K1=1.2, BM25_B=0.75,
    BM25_FIELD_WEIGHTS={"n": 2.0, "a": 1.0, "c": 0.5, "s": 0.25},  # name / address / city / state
    BM25_MAX_DF_FRAC=0.002,         # tokens in more than this share of a partition's docs are pruned
    BM25_MIN_DF_CAP=50,
    BM25_QUERY_CHUNK=2000,          # queries per sparse product (CPU path); the GPU path sizes its own blocks
    ENABLE_TFIDF_RETRIEVER=False,   # char TF-IDF+SVD retriever: OFF -- full-pool leave-one-out drop was negative (-0.0017)
    TFIDF_TOP_K=20, TFIDF_SVD_DIM=128, TFIDF_MAX_FEATURES=60_000, TFIDF_NGRAM_RANGE=(2, 4),
    TFIDF_FIT_SAMPLE=400_000,
    EXACT_TOP_K=5, EXACT_MAX_BLOCK=200, EXACT_MAX_PAIRS=20_000,
    MAX_CANDIDATES=120,             # final per-S1 cap after the union (diag: 100 -> oracle F0.5 0.989, 200 -> 0.992)
    RECALL_KS=[25, 50, 100, 200, 500],
    DIAG_QUERY_FRACTION=0.01,       # train S1 share (~22k) searched at diagnostic depth against the FULL train pool
    DIAG_TOP_K=500,                 # per-source retrieval depth on the diagnostic queries (recall curves only)
    SEARCH_SHARD_ROWS=1_000_000,    # pool rows per GPU search shard (bounds host RAM of the gathered fp16 shard)
    QUERY_BLOCK_ROWS=50_000,        # Source-1 rows per union/feature/predict block (bounds per-block RAM)
    PAIR_BATCH=200_000,

    # ---- model ----
    GBDT_TRAIN_ENTITY_FRACTION=0.20,  # train S1 entities the reranker is fit + calibrated on; the rest is evaluation
    GBDT_GRID_SELECT_FRACTION=0.25,   # share of the training entities used to pick GBDT_PARAM_GRID (faster)
    NESTED_HOLDOUT_FRAC=0.12, N_FOLDS=5, GBDT_TRAIN_NEG_FRAC=1.0,
    GBDT_PARAM_GRID=[dict(max_depth=6, learning_rate=0.10, n_estimators=300),
                     dict(max_depth=8, learning_rate=0.07, n_estimators=400)],
    RULE_SIMPLICITY_TOL=0.001,      # a more complex decision rule must beat the simplest one by this much (CV macro F0.5)

    # ---- manual audit of test countries unseen in training (France) ----
    AUDIT_COUNTRIES=None,           # None = test countries absent from train; [] = off
    AUDIT_N_ENTITIES=200, AUDIT_TOP_N=5,
    AUDIT_LABELS_PATH=None,         # path/glob of audit TSVs with the is_match column filled in (scored in the report)

    # ---- paths (None = derived from WORK_DIR) ----
    DATA_DIR=os.path.join(LOCAL_ROOT, "data"),   # raw TSVs: {train,test}/{split}_source{1,2,3}.tsv
    WORK_DIR=os.path.join(LOCAL_ROOT, "pipeline_run"),
    INPUT_CACHE_DIR=os.path.join(LOCAL_ROOT, "pipeline_run", "aligned_inputs"),   # parquet copies + aligned embeddings (all profiles)
    CACHE_DIR=None, SCRATCH_DIR=None, OUTPUT_DIR=None,
    RUN_SELF_TESTS=True,
)

PROFILES = {
    "full": {},
    "smoke": dict(
        DATA_FRACTION=0.002, TEST_FRACTION=0.002, DIAG_QUERY_FRACTION=0.5, GBDT_TRAIN_ENTITY_FRACTION=0.5,
        TFIDF_FIT_SAMPLE=30_000, TFIDF_SVD_DIM=48, TFIDF_MAX_FEATURES=20_000, EMBED_CHUNK_ROWS=3000,
        DIAG_TOP_K=200, REV_DIAG_TOP_K=20, SEARCH_SHARD_ROWS=2500, QUERY_BLOCK_ROWS=600, PAIR_BATCH=5000,
        N_FOLDS=3, GBDT_PARAM_GRID=[dict(max_depth=4, learning_rate=0.2, n_estimators=60)],
        GATE1_MIN_UNION_RECALL=0.0, AUDIT_N_ENTITIES=20, WORK_DIR=os.path.join(LOCAL_ROOT, "pipeline_run_smoke"),
    ),
}
CFG.update(PROFILES[PROFILE])
for _kv in filter(None, os.environ.get("ER_SET", "").split(";")):
    _k, _v = _kv.split("=", 1)
    assert _k in CFG, f"ER_SET: unknown config key {_k}"
    import json as _json
    CFG[_k] = _json.loads(_v)
if CFG["POOL_DISTRACTOR_FRACTION"] is None:
    CFG["POOL_DISTRACTOR_FRACTION"] = CFG["DATA_FRACTION"]
