CODE_VERSION = "pe1"   # bump when a stage's logic changes so stale caches are not reused (invalidates samples + frames)
RETR_CODE_VERSION = "r3"   # retrieval / union logic only (retrieval caches, training sets, models)
MODEL_CODE_VERSION = "m3"  # reranker / decision-rule logic only
REPORT_SUFFIX = ""


def fp_of(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def cfg_fp(keys, extra=None):
    return fp_of({"v": CODE_VERSION, "cfg": {k: getattr(cfg, k) for k in keys}, "extra": extra})


def cache_path(rel, base=None):
    p = os.path.join(base or CACHE_DIR, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def locate(rel):
    """First existing copy of a cache artifact: this session's CACHE_DIR, then earlier sessions' outputs."""
    for base in [CACHE_DIR] + CACHE_SEARCH_PATHS:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None


def is_done(rel):
    return locate(rel + ".done") is not None


def read_done(rel):
    p = locate(rel + ".done")
    if p is None:
        return None
    with open(p) as f:
        return json.load(f)


def _atomic_write_bytes(path, write_fn):
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def save_json(rel, obj, base=None):
    p = cache_path(rel, base)
    def _w(t):
        with open(t, "w") as f:
            json.dump(obj, f, default=str, indent=1)
    _atomic_write_bytes(p, _w)
    return p


def mark_done(rel, **meta):
    save_json(rel + ".done", dict(meta, t=time.time()))


def save_npy(rel, arr, base=None):
    p = cache_path(rel, base)
    def _w(t):
        with open(t, "wb") as f:
            np.save(f, arr)
    _atomic_write_bytes(p, _w)
    return p


PARQUET_STREAM_ROWS = 5_000_000   # numeric frames above this are converted to Arrow and written slice by slice


def save_parquet(rel, df, base=None, row_group_size=None):
    p = cache_path(rel, base)
    numeric = all(t.kind in "biuf" for t in df.dtypes)
    if len(df) <= PARQUET_STREAM_ROWS or not numeric:
        _atomic_write_bytes(p, lambda t: pq.write_table(pa.Table.from_pandas(df, preserve_index=False), t,
                                                         compression="zstd", row_group_size=row_group_size))
        return p
    # same file content (schema, row groups), without an Arrow copy of the whole frame
    rg = row_group_size or PARQUET_STREAM_ROWS
    step = max(rg, PARQUET_STREAM_ROWS // rg * rg)

    def _w(t):
        with pq.ParquetWriter(t, pa.Schema.from_pandas(df.iloc[:0], preserve_index=False), compression="zstd") as w:
            for a in range(0, len(df), step):
                w.write_table(pa.Table.from_pandas(df.iloc[a:a + step], preserve_index=False), row_group_size=rg)
    _atomic_write_bytes(p, _w)
    return p


def arrow_to_frame(table):
    """pyarrow Table -> DataFrame with string columns as plain object arrays (pandas-version independent)."""
    cols = {}
    for name in table.column_names:
        col = table.column(name)
        if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
            arr = col.to_numpy(zero_copy_only=False)
            cols[name] = np.where(pd.isna(arr), "", arr).astype(object) if col.null_count else arr.astype(object)
        else:
            cols[name] = col.to_numpy(zero_copy_only=False)
    return pd.DataFrame(cols)


def load_parquet(path, columns=None, filters=None):
    return arrow_to_frame(pq.read_table(path, columns=columns, filters=filters))


def dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return round(total / 1e9, 3)


def _fmt_duration(seconds):   # copied from er_crossencoder_v2 §2c
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _proc_status_gb(*keys):
    try:
        with open("/proc/self/status") as f:
            st_ = {l.split(":")[0]: int(l.split()[1]) for l in f if l.split(":")[0] in keys}
        return [round(st_[k] / 1e6, 2) for k in keys]
    except (OSError, KeyError, ValueError):
        return [None] * len(keys)


def rss_now_gb():
    """(RSS, anonymous, file-backed) GB. File-backed = memmapped embeddings: page cache, reclaimable."""
    return tuple(_proc_status_gb("VmRSS", "RssAnon", "RssFile"))


def _reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as f:   # Linux >= 4.0: resets VmHWM to the current RSS
            f.write("5")
    except OSError:
        pass


def peak_rss_gb():
    """RSS high-water mark since the last reset (VmHWM; per stage, see `stage`)."""
    return _proc_status_gb("VmHWM")[0]


def gpu_peaks_gb():
    """Per-GPU peak allocated GB since the last reset."""
    if not (TORCH_AVAILABLE and N_GPUS):
        return None
    return [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(N_GPUS)]


def gpu_peak_gb():
    p = gpu_peaks_gb()
    return max(p) if p else None


def _obj_gb(obj):
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):   # object columns count their 8-byte pointers only (strings may be shared)
        return obj.memory_usage(deep=False, index=False).sum() / 1e9
    if isinstance(obj, dict):
        return sum(_obj_gb(v) or 0 for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_obj_gb(v) or 0 for v in obj)
    return getattr(obj, "nbytes", None) and obj.nbytes / 1e9


def mem_mark(label, obj=None, rows=None):
    """Log current RSS (anonymous / file-backed) and the size of `obj` right after a large object is built."""
    rss, anon, fil = rss_now_gb()
    size = _obj_gb(obj)
    if rows is None and obj is not None:
        rows = sum(len(v) for v in obj.values()) if isinstance(obj, dict) else (len(obj) if hasattr(obj, "__len__") else None)
    log(f"   [mem] {label}: " + (f"{rows:,} rows, " if rows is not None else "")
        + (f"{size:.2f} GB, " if size is not None else "") + f"RSS {rss} GB (anon {anon}, file {fil})")


REPORT = {}
RUN_LOG_PATH = os.path.join(OUTPUT_DIR, "run_log.jsonl")
PROGRESS_LOG_PATH = os.path.join(OUTPUT_DIR, "progress.log")
SESSION_ID = f"{int(T0)}-{os.getpid()}"


def log(msg):
    """Print and append a timestamped line to output/progress.log (survives a crash / kernel restart)."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{_fmt_duration(time.time() - T0)}] {msg}"
    print(line)
    with open(PROGRESS_LOG_PATH, "a") as f:
        f.write(line + "\n")


F05_PROGRESS_REL = "reports/f05_progress.json"


def track_f05(step, macro_f05, **extra):
    """Record a macro F0.5 checkpoint of the pipeline (retrieval ceiling, CV, holdout, running eval, final)
    in reports/f05_progress.json and the progress log, and print the table so far."""
    p = locate(F05_PROGRESS_REL)
    rows = []
    if p:
        with open(p) as f:
            rows = json.load(f)
    rows = [r for r in rows if r["step"] != step] + [dict(step=step, macro_f05=round(float(macro_f05), 5),
                                                          t=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)]
    save_json(F05_PROGRESS_REL, rows)
    REPORT["f05_progress"] = rows
    log(f"F0.5 checkpoint  {step}: {float(macro_f05):.5f}  {extra if extra else ''}")
    return rows


def show_f05_progress():
    p = locate(F05_PROGRESS_REL)
    if p:
        with open(p) as f:
            rows = json.load(f)
        print("\n==== macro F0.5 at each pipeline step ====")
        print(pd.DataFrame(rows).to_string(index=False))


class stage:
    """Context manager: timing + peak RAM / GPU memory per stage, appended to run_log.jsonl."""
    def __init__(self, name, **info):
        self.name, self.info, self.extra = name, info, {}

    def add(self, **kv):
        self.extra.update(kv)

    _open = []   # enclosing stages: the peak counters are reset per stage, so parents fold in their children's peaks

    @staticmethod
    def _fold():
        rss, gpu = peak_rss_gb(), gpu_peaks_gb()
        for s in stage._open:
            s.rss_peak = max(s.rss_peak or 0, rss or 0)
            s.gpu_peaks = [max(a, b) for a, b in zip(s.gpu_peaks, gpu)] if gpu else s.gpu_peaks

    def __enter__(self):
        stage._fold()
        _reset_peak_rss()
        if TORCH_AVAILABLE and N_GPUS:
            for i in range(N_GPUS):
                torch.cuda.reset_peak_memory_stats(i)
        self.rss_peak, self.gpu_peaks = peak_rss_gb(), [0.0] * N_GPUS
        stage._open.append(self)
        self.t0 = time.time()
        print()
        log(f">> {self.name} {self.info or ''}  RSS={rss_now_gb()[0]}GB")
        return self

    def __exit__(self, exc_type, exc, tb):
        stage._fold()
        stage._open.remove(self)
        rec = dict(stage=self.name, seconds=round(time.time() - self.t0, 2), ok=exc_type is None,
                   peak_rss_gb=self.rss_peak, gpu_peak_gb=max(self.gpu_peaks) if self.gpu_peaks else None,
                   gpu_peaks_gb=self.gpu_peaks, session=SESSION_ID,
                   t_session=round(time.time() - T0, 1), **self.info, **self.extra)
        REPORT.setdefault("stages", []).append(rec)
        with open(RUN_LOG_PATH, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        log(f"<< {self.name} {'done' if exc_type is None else 'FAILED: ' + repr(exc)} "
            f"in {_fmt_duration(rec['seconds'])}  peak_rss={rec['peak_rss_gb']}GB (this stage)  "
            f"gpu_peaks={rec['gpu_peaks_gb']}GB  RSS now={rss_now_gb()[0]}GB")
        return False


def hours_left():
    return cfg.MAX_SESSION_HOURS - (time.time() - T0) / 3600


ALL_STAGES = ["selftest", "sample", "embed_check", "weight_sweep", "diag_recall", "train_model", "train_eval",
              "test_inference", "submit", "report"]
SCALE_STAGES = ("test_inference", "submit")   # gated by ENABLE_SCALE_PHASE and the train-evaluation gate
SELECTED = set(ALL_STAGES) if STAGES_TO_RUN.strip() == "all" else {s.strip() for s in STAGES_TO_RUN.split(",") if s.strip()}
assert SELECTED <= set(ALL_STAGES), f"unknown stages: {SELECTED - set(ALL_STAGES)}"
if not cfg.RUN_SELF_TESTS:
    SELECTED.discard("selftest")


def want(name):
    return name in SELECTED