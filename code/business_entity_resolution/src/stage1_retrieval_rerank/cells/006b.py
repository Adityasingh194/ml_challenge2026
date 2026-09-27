import pyarrow.csv as pacsv

RAW_FIELDS = ["entity_id", "business_name", "business_address", "country"]
SRC_DIR = os.path.join(cfg.INPUT_CACHE_DIR, "data_parquet")
EMB_ALIGNED_DIR = os.path.join(cfg.INPUT_CACHE_DIR, "emb_aligned")
EMB_VIEWS = ("name", "addr")


def _raw_tsv(split, s):
    for p in (os.path.join(DATA_DIR, split, f"{split}_source{s}.tsv"), os.path.join(DATA_DIR, f"{split}_source{s}.tsv")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{split}_source{s}.tsv not found under {DATA_DIR}")


def _gt_tsv():
    for p in (os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), os.path.join(DATA_DIR, "train_ground_truth.tsv")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"train_ground_truth.tsv not found under {DATA_DIR}")


def _count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 24), b""):
            n += buf.count(b"\n")
    return n


def _tsv_to_parquet(tsv, out):
    """Every field read as a plain string, no quoting (names contain quote characters), '' kept as ''."""
    t = pacsv.read_csv(
        tsv,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in RAW_FIELDS},
                                             strings_can_be_null=False, include_columns=RAW_FIELDS))
    n_lines = _count_lines(tsv)
    if t.num_rows != n_lines - 1:
        raise RuntimeError(f"{tsv}: parsed {t.num_rows:,} rows but the file has {n_lines - 1:,} data lines")
    tmp = out + ".tmp"
    pq.write_table(t, tmp, compression="zstd", row_group_size=500_000)
    os.replace(tmp, out)
    return t.num_rows


def _file_fp(path):
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"


def prepare_raw_inputs():
    os.makedirs(SRC_DIR, exist_ok=True)
    for sp_ in ("train", "test"):
        for s in (1, 2, 3):
            tsv = _raw_tsv(sp_, s)
            out = os.path.join(SRC_DIR, f"{sp_}_source{s}_final.parquet")
            marker = out + ".done"
            fp = _file_fp(tsv)
            if os.path.exists(out) and os.path.exists(marker) and open(marker).read() == fp:
                print(f"  {os.path.basename(out)}: up to date ({pq.ParquetFile(out).metadata.num_rows:,} rows)")
                continue
            t0 = time.time()
            n = _tsv_to_parquet(tsv, out)
            with open(marker, "w") as f:
                f.write(fp)
            print(f"  {os.path.basename(tsv)} -> {os.path.basename(out)}: {n:,} rows in {time.time() - t0:.1f}s")


def _emb_src_dir(split, s, view):
    return os.path.join(cfg.EMB_DIR, split, f"s{s}", view)


def aligned_path(split, s, view):
    return os.path.join(EMB_ALIGNED_DIR, f"{split}_s{s}_{view}.npy")


def _align_one(split, s, view, n_rows):
    """Write the (split, source, view) vectors as one int8 (n, dim) memmap in TSV row order."""
    src = _emb_src_dir(split, s, view)
    with open(os.path.join(src, "meta.json")) as f:
        meta = json.load(f)
    out = aligned_path(split, s, view)
    fp = dict(meta=meta, rows_npy=_file_fp(os.path.join(src, "rows.npy")), n_rows=n_rows)
    marker = out + ".done"
    if os.path.exists(out) and os.path.exists(marker):
        with open(marker) as f:
            if json.load(f) == json.loads(json.dumps(fp)):
                return meta, False
    n, dim, R = int(meta["n"]), int(meta["dim"]), int(meta["chunk_rows"])
    if meta.get("store_dtype") != "int8":
        raise RuntimeError(f"{src}: store_dtype {meta.get('store_dtype')} (expected int8)")
    if meta.get("file_split") not in (None, split) or meta.get("source") not in (None, f"s{s}") or meta.get("view") not in (None, view):
        raise RuntimeError(f"{src}: meta.json describes {meta.get('file_split')}/{meta.get('source')}/{meta.get('view')}")
    if n != n_rows:
        raise RuntimeError(f"{src}: meta n={n:,} but {split}_source{s} has {n_rows:,} rows")
    rows = np.load(os.path.join(src, "rows.npy"))
    if rows.shape != (n,) or rows.min() != 0 or rows.max() != n - 1 or np.bincount(rows, minlength=n).max() != 1:
        raise RuntimeError(f"{src}/rows.npy is not a permutation of 0..{n - 1}")
    tmp = out + ".tmp.npy"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int8, shape=(n, dim))
    n_chunks = math.ceil(n / R)
    for c in range(n_chunks):
        E = np.load(os.path.join(src, f"emb_{c:05d}.npy"))
        lo, hi = c * R, min(n, (c + 1) * R)
        if E.shape != (hi - lo, dim) or E.dtype != np.int8:
            raise RuntimeError(f"{src}/emb_{c:05d}.npy has shape {E.shape} {E.dtype}, expected {(hi - lo, dim)} int8")
        mm[rows[lo:hi]] = E        # embedding row i belongs to TSV row rows[i]
    mm.flush()
    del mm
    os.replace(tmp, out)
    with open(marker, "w") as f:
        json.dump(fp, f)
    return meta, True


def _check_alignment(split, s, sample=4000):
    """Records sharing a business_name must share the name vector (int8 rounding aside)."""
    names = pq.read_table(SRC_FILE[(split, s)], columns=["business_name"]).column(0).to_numpy(zero_copy_only=False)
    codes, uniq = pd.factorize(names)
    cnt = np.bincount(codes)
    dup_codes = np.flatnonzero(cnt > 1)
    if not len(dup_codes):
        return None
    rng = np.random.default_rng(0)
    pick = rng.choice(dup_codes, size=min(sample, len(dup_codes)), replace=False)
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.cumsum(cnt)[:-1]]
    a, b = order[starts[pick]], order[starts[pick] + 1]
    E = np.load(aligned_path(split, s, "name"), mmap_mode="r")
    va, vb = np.asarray(E[np.sort(a)], np.float32), np.asarray(E[np.sort(b)], np.float32)
    va, vb = va[np.argsort(np.argsort(a))], vb[np.argsort(np.argsort(b))]
    cos = (va * vb).sum(1) / np.maximum(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-9)
    frac = float((cos > 0.999).mean())
    if frac < 0.99:
        raise RuntimeError(f"embedding alignment check FAILED for {split}/S{s}: only {frac:.3f} of duplicate-name pairs "
                           "have identical name vectors -- rows.npy mapping or files are wrong")
    return round(frac, 4)


def align_embeddings():
    os.makedirs(EMB_ALIGNED_DIR, exist_ok=True)
    info = {}
    for sp_ in ("train", "test"):
        for s in (1, 2, 3):
            n_rows = pq.ParquetFile(SRC_FILE[(sp_, s)]).metadata.num_rows
            for view in EMB_VIEWS:
                t0 = time.time()
                meta, built = _align_one(sp_, s, view, n_rows)
                info[f"{sp_}_s{s}_{view}"] = dict(n=int(meta["n"]), dim=int(meta["dim"]), model=os.path.basename(meta.get("model", "")))
                if built:
                    print(f"  aligned {sp_}/S{s}/{view}: {meta['n']:,} x {meta['dim']} int8 in {time.time() - t0:.1f}s")
            if not os.path.exists(aligned_path(sp_, s, "name") + ".checked"):
                frac = _check_alignment(sp_, s)
                with open(aligned_path(sp_, s, "name") + ".checked", "w") as f:
                    f.write(str(frac))
                print(f"  alignment check {sp_}/S{s}: duplicate-name pairs with identical name vectors = {frac}")
    dims = {v: {d["dim"] for k, d in info.items() if k.endswith(v)} for v in EMB_VIEWS}
    assert all(len(x) == 1 for x in dims.values()), f"inconsistent embedding dims across files: {dims}"
    return {v: next(iter(x)) for v, x in dims.items()}, info


with stage("inputs"):
    prepare_raw_inputs()
    GT_PATH = _gt_tsv()
    SRC_FILE = {(split, s): os.path.join(SRC_DIR, f"{split}_source{s}_final.parquet")
                for split in ("train", "test") for s in (1, 2, 3)}
    EMB_DIMS, EMB_INFO = align_embeddings()
print(f"SRC_DIR={SRC_DIR}\nGT_PATH={GT_PATH}\nEMB_ALIGNED_DIR={EMB_ALIGNED_DIR}  dims={EMB_DIMS}  "
      f"model={sorted({d['model'] for d in EMB_INFO.values()})}")
