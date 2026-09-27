from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def combined_text(df):   # copied from er_crossencoder_v2 §5 (char TF-IDF text)
    return (df["name_clean"].astype(str) + " | " + df["addr_clean"].astype(str)).tolist()


# ------------------------- precomputed Qwen3-Embedding-0.6B vectors (name / address views) -------------------------
_SRC_LOOKUP = {}


def _src_lookup(split, s):
    """(sorted int ids, their TSV rows) for one source file -- maps entity ids to rows of the aligned memmaps."""
    key = (split, s)
    if key not in _SRC_LOOKUP:
        _, ids_int = read_ids(split, s)
        order = np.argsort(ids_int, kind="stable")
        _SRC_LOOKUP[key] = (ids_int[order], order.astype(np.int64))
    return _SRC_LOOKUP[key]


class PrecomputedStore:
    """One embedding view ('name' 256-d or 'addr' 512-d) for the rows of one frame (split x role), read from the
    aligned int8 memmaps of §2b. Same read interface as EmbStore: n, dim, gather(rows, use_dim, out_dtype)."""

    def __init__(self, split, role, view):
        self.split, self.role, self.view = split, role, view
        self.dim, self.dtype = int(EMB_DIMS[view]), "int8"
        ids = frame_ids(split, role)
        self.n = len(ids)
        self.tag = f"pre-{view}/{frame_fp(split)}_{split}_{role}"
        self.src = (ids // ID_BASE).astype(np.int8)          # 1, 2 or 3 (the id prefix is the source)
        self.tsv_row = np.empty(self.n, np.int64)
        self._mm = {}
        for s in np.unique(self.src):
            s = int(s)
            m = self.src == s
            sid, srow = _src_lookup(split, s)
            j = np.clip(np.searchsorted(sid, ids[m]), 0, len(sid) - 1)
            if not np.array_equal(sid[j], ids[m]):
                raise RuntimeError(f"{self.tag}: frame ids missing from {split}_source{s}")
            self.tsv_row[m] = srow[j]
            self._mm[s] = np.load(aligned_path(split, s, view), mmap_mode="r")
            if self._mm[s].shape[1] != self.dim:
                raise RuntimeError(f"{aligned_path(split, s, view)}: dim {self._mm[s].shape[1]} != {self.dim}")

    def complete(self):
        return True

    def missing(self):
        return []

    def bytes_per_row(self):
        return self.dim

    def gather(self, rows, use_dim=None, out_dtype=np.float16):
        """(len(rows), use_dim) L2-normalized float vectors; memmap reads in ascending TSV-row order per source."""
        use_dim = self.dim if use_dim is None else int(use_dim)
        assert use_dim <= self.dim
        rows = np.asarray(rows, dtype=np.int64)
        res = np.empty((len(rows), use_dim), dtype=out_dtype)
        if not len(rows):
            return res
        src, tr = self.src[rows], self.tsv_row[rows]
        for s, mm in self._mm.items():
            m = np.flatnonzero(src == s)
            if not len(m):
                continue
            o = np.argsort(tr[m], kind="stable")
            block = np.asarray(mm[tr[m][o], :use_dim], dtype=np.float32)
            nrm = np.linalg.norm(block, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            res[m[o]] = block / nrm
        return res


class CombinedStore:
    """The dense retrieval vector: [sqrt(w) * name, sqrt(1-w) * addr] (unit norm), so that
    <u, v> = w * cos(name_u, name_v) + (1 - w) * cos(addr_u, addr_v)."""

    def __init__(self, split, role, w):
        self.name, self.addr = get_store("name", split, role), get_store("addr", split, role)
        self.w = float(w)
        self.dim, self.dtype, self.n = self.name.dim + self.addr.dim, "float16", self.name.n
        self.tag = f"combined-w{self.w:g}/{self.name.tag}"
        self._a, self._b = math.sqrt(self.w), math.sqrt(1.0 - self.w)

    def complete(self):
        return True

    def missing(self):
        return []

    def bytes_per_row(self):
        return self.name.dim + self.addr.dim

    def gather(self, rows, use_dim=None, out_dtype=np.float16):
        assert use_dim in (None, self.dim), f"the combined vector is used at its full dim {self.dim}"
        res = np.empty((len(rows), self.dim), dtype=out_dtype)
        res[:, :self.name.dim] = self.name.gather(rows, None, np.float32) * self._a
        res[:, self.name.dim:] = self.addr.gather(rows, None, np.float32) * self._b
        return res


# ------------------------- chunked vector cache (char TF-IDF retriever) -------------------------
class EmbStore:
    """Row-aligned chunked embedding store for one frame. Chunk c = frame rows [c*R, (c+1)*R)."""

    def __init__(self, tag, ids_int, dim, dtype, chunk_rows):
        assert dtype in ("float16", "int8"), dtype
        self.tag, self.dim, self.dtype, self.R = tag, int(dim), dtype, int(chunk_rows)
        self.ids = np.asarray(ids_int, dtype=np.int64)
        self.n = len(self.ids)
        self.rel = f"embeddings/{tag}"
        self.n_chunks = math.ceil(self.n / self.R) if self.n else 0
        self.meta = dict(tag=tag, n=self.n, dim=self.dim, dtype=dtype, chunk_rows=self.R,
                         ids_sha=hashlib.sha1(self.ids.tobytes()).hexdigest()[:16])
        man = locate(self.rel + "/manifest.json")
        if man:
            with open(man) as f:
                old = json.load(f)
            diff = {k: (old.get(k), v) for k, v in self.meta.items() if old.get(k) != v}
            if diff:
                raise RuntimeError(f"embedding cache {self.rel} exists with different settings {diff}")
        else:
            save_json(self.rel + "/manifest.json", self.meta)
        self._mm = {}

    def _crel(self, c, kind):
        return f"{self.rel}/chunk_{c:05d}.{kind}.npy"

    def rng(self, c):
        lo = c * self.R
        return lo, min(self.n, lo + self.R)

    def chunk_done(self, c):
        d = read_done(self._crel(c, "vec"))
        lo, hi = self.rng(c)
        return d is not None and d.get("rows") == hi - lo

    def missing(self):
        return [c for c in range(self.n_chunks) if not self.chunk_done(c)]

    def complete(self):
        return not self.missing()

    def bytes_per_row(self):
        return self.dim * (1 if self.dtype == "int8" else 2) + 8

    def write(self, c, vecs):
        lo, hi = self.rng(c)
        assert vecs.shape == (hi - lo, self.dim), (vecs.shape, (hi - lo, self.dim))
        if self.dtype == "int8":   # row-scaled; the per-row scale cancels after re-normalization
            m = np.abs(vecs).max(axis=1, keepdims=True)
            m[m == 0] = 1.0
            arr = np.rint(vecs / m * 127.0).astype(np.int8)
        else:
            arr = vecs.astype(np.float16)
        save_npy(self._crel(c, "ids"), self.ids[lo:hi])
        save_npy(self._crel(c, "vec"), arr)
        mark_done(self._crel(c, "vec"), rows=hi - lo, dim=self.dim, dtype=self.dtype)

    def _chunk(self, c):
        if c not in self._mm:
            p, ip = locate(self._crel(c, "vec")), locate(self._crel(c, "ids"))
            if p is None or ip is None:
                raise FileNotFoundError(f"embedding chunk {self._crel(c, 'vec')} not found in {CACHE_DIR}")
            lo, hi = self.rng(c)
            if not np.array_equal(np.load(ip), self.ids[lo:hi]):
                raise RuntimeError(f"id mismatch in {self._crel(c, 'ids')} -- cache does not belong to this frame")
            self._mm[c] = np.load(p, mmap_mode="r")
        return self._mm[c]

    def gather(self, rows, use_dim, out_dtype=np.float16):
        """(len(rows), use_dim) L2-normalized vectors, read chunk by chunk."""
        rows = np.asarray(rows, dtype=np.int64)
        assert use_dim <= self.dim
        res = np.empty((len(rows), use_dim), dtype=out_dtype)
        if len(rows) == 0:
            return res
        order = np.argsort(rows, kind="stable")
        rs = rows[order]
        ch = rs // self.R
        cut = np.flatnonzero(np.diff(ch)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(rs)]):
            c = int(ch[a])
            block = np.asarray(self._chunk(c)[rs[a:b] - c * self.R, :use_dim], dtype=np.float32)
            nrm = np.linalg.norm(block, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            res[order[a:b]] = block / nrm
        return res


class EmbView:
    """Concatenation of stores (e.g. train + test) addressed by combined row position."""

    def __init__(self, stores, use_dim):
        self.stores = list(stores)
        self.dim = int(use_dim)
        self.offsets = np.cumsum([0] + [s.n for s in self.stores])
        self.n = int(self.offsets[-1])
        for s in self.stores:
            assert use_dim <= s.dim, f"use_dim={use_dim} > stored dim {s.dim} of {s.tag}"

    def gather(self, rows, out_dtype=np.float16):
        rows = np.asarray(rows, dtype=np.int64)
        res = np.empty((len(rows), self.dim), dtype=out_dtype)
        which = np.searchsorted(self.offsets, rows, side="right") - 1
        for i, s in enumerate(self.stores):
            m = which == i
            if m.any():
                res[m] = s.gather(rows[m] - self.offsets[i], self.dim, out_dtype)
        return res


# ------------------------- char TF-IDF + SVD (copied from er_crossencoder_v2 §5) -------------------------
def _char_embed_batch(texts, vec, svd):   # copied from er_crossencoder_v2 §5
    Z = svd.transform(vec.transform(texts)).astype(np.float32)
    return _l2_normalize_rows(Z).astype(np.float16)


def _l2_normalize_rows(mat):   # copied from er_crossencoder_v2 §5
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def fit_char_tfidf_svd(text_sample, n_components, max_features, ngram_range, seed):
    """copied from er_crossencoder_v2 §5 (config globals -> arguments)"""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(ngram_range),
                          max_features=max_features, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(text_sample)
    n_comp = min(n_components, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, algorithm="randomized", n_iter=4, random_state=seed)
    svd.fit(X)
    print(f"  char TF-IDF vocab={len(vec.vocabulary_):,}  SVD dim={n_comp}  "
          f"explained_var={svd.explained_variance_ratio_.sum():.3f}")
    return vec, svd


class TfidfSvdEmbedder:
    """Char TF-IDF + SVD vectors for the `tfidf` retriever, transformed in NUM_WORKERS processes."""

    def __init__(self, vec, svd):
        self.vec, self.svd = vec, svd
        self.native_dim = int(svd.n_components)

    def embed(self, texts, out_dim):
        texts = list(texts)
        step = max(5000, math.ceil(len(texts) / max(cfg.NUM_WORKERS, 1)))
        batches = [texts[i:i + step] for i in range(0, len(texts), step)]
        if len(batches) > 1 and cfg.NUM_WORKERS > 1:
            parts = joblib.Parallel(n_jobs=cfg.NUM_WORKERS, backend="loky")(
                joblib.delayed(_char_embed_batch)(b, self.vec, self.svd) for b in batches)
        else:
            parts = [_char_embed_batch(b, self.vec, self.svd) for b in batches]
        Z = np.vstack(parts).astype(np.float32)[:, :out_dim] if parts else np.zeros((0, out_dim), np.float32)
        return _l2_normalize_rows(Z), 0


def _fit_sample_texts(text_fn, columns, n_total):
    """Unsupervised fit text: a hash sample of train + of the three TEST files (so test-only
    n-grams, e.g. France, are in the vocabulary -- same idea as er_crossencoder_v2 §5)."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    texts = []
    for split, share in (("train", n_total // 2), ("test", n_total // 2)):
        for s in (1, 2, 3):
            n = pq.ParquetFile(SRC_FILE[(split, s)]).metadata.num_rows
            idx = np.sort(rng.choice(n, size=min(n, share // 3), replace=False))
            texts += text_fn(arrow_to_frame(read_raw_table(SRC_FILE[(split, s)], columns).take(pa.array(idx))))
    return texts


_TFIDF_MODELS = {}


def get_tfidf_model(kind="ret"):
    """The char TF-IDF retriever model (name_clean | addr_clean)."""
    if kind in _TFIDF_MODELS:
        return _TFIDF_MODELS[kind]
    assert kind == "ret", kind
    dim, fn, cols = cfg.TFIDF_SVD_DIM, combined_text, ["name_clean", "addr_clean"]
    fpk = fp_of([kind, dim, cfg.TFIDF_MAX_FEATURES, cfg.TFIDF_NGRAM_RANGE, cfg.TFIDF_FIT_SAMPLE, cfg.RANDOM_SEED, CODE_VERSION])
    rel = f"models/tfidf_{kind}_{fpk}.joblib"
    p = locate(rel)
    if p:
        vec, svd = joblib.load(p)
    else:
        with stage(f"fit_tfidf_{kind}"):
            vec, svd = fit_char_tfidf_svd(_fit_sample_texts(fn, cols, cfg.TFIDF_FIT_SAMPLE), dim,
                                          cfg.TFIDF_MAX_FEATURES, cfg.TFIDF_NGRAM_RANGE, cfg.RANDOM_SEED)
            tmp = cache_path(rel) + ".tmp"
            joblib.dump((vec, svd), tmp)
            os.replace(tmp, cache_path(rel))
    _TFIDF_MODELS[kind] = (vec, svd, fpk)
    return _TFIDF_MODELS[kind]


_EMBEDDERS = {}


def get_embedder(kind="tfidf_ret"):
    assert kind == "tfidf_ret", kind
    if kind not in _EMBEDDERS:
        vec, svd, _ = get_tfidf_model("ret")
        _EMBEDDERS[kind] = TfidfSvdEmbedder(vec, svd)
    return _EMBEDDERS[kind]


def release_embedders():
    _EMBEDDERS.clear()
    gc.collect()
    if N_GPUS:
        torch.cuda.empty_cache()


def embed_model_tag():
    """Identifies the dense vectors: the precomputed model files + the name/address weight."""
    models = sorted({d["model"] for d in EMB_INFO.values()})
    return f"pre-{fp_of([models, EMB_DIMS])[:6]}-w{float(resolve_emb_weight()):g}"


def frame_fp(split):
    return FP_SAMPLE if split == "train" else FP_TEST


_STORES = {}


def get_store(kind, split, role):
    """kind 'dense' = combined name+addr vector; 'name' / 'addr' = one precomputed view; 'tfidf' = char TF-IDF."""
    key = (kind, split, role, float(resolve_emb_weight()) if kind == "dense" else None)
    if key not in _STORES:
        if kind in EMB_VIEWS:
            _STORES[key] = PrecomputedStore(split, role, kind)
        elif kind == "dense":
            _STORES[key] = CombinedStore(split, role, cfg.EMB_NAME_WEIGHT)
        elif kind == "tfidf":
            dim, dtype = int(get_tfidf_model("ret")[1].n_components), "float16"
            tag = f"tfidf-{get_tfidf_model('ret')[2]}/{frame_fp(split)}_{split}_{role}_d{dim}_{dtype}"
            _STORES[key] = EmbStore(tag, frame_ids(split, role), dim, dtype, cfg.EMBED_CHUNK_ROWS)
        else:
            raise ValueError(kind)
    return _STORES[key]


def store_texts(kind, split, role):
    assert kind == "tfidf", kind
    return combined_text(raw_frame(split, role, ["name_clean", "addr_clean"]))


def run_embedding(store, texts, kind, label):
    """Fill the TF-IDF store's missing chunks (CPU processes inside the embedder). Returns True when complete."""
    todo = store.missing()
    if not todo:
        print(f"[embed {label}] all {store.n_chunks} chunks already done")
        return True
    assert len(texts) == store.n
    emb = get_embedder(kind)
    t_start, rows_done = time.time(), 0
    remaining_rows = sum(store.rng(c)[1] - store.rng(c)[0] for c in todo)
    print(f"[embed {label}] {len(todo)}/{store.n_chunks} chunks to do ({remaining_rows:,} rows, d={store.dim} {store.dtype})")
    for c in todo:
        if hours_left() <= 0:
            print(f"  [embed {label}] MAX_SESSION_HOURS reached -- stopping cleanly")
            return False
        lo, hi = store.rng(c)
        codes, uniques = pd.factorize(pd.Series(texts[lo:hi], dtype=object))
        vu, _ = emb.embed(list(uniques), store.dim)
        store.write(c, vu[codes])
        rows_done += hi - lo
    secs = time.time() - t_start
    print(f"  [embed {label}] {rows_done:,} rows in {secs:.1f}s ({rows_done / max(secs, 1e-9):,.0f} rows/s)")
    return store.complete()


# ------------------------- sanity check of the precomputed vectors -------------------------
def embedding_sanity_train():
    """True pairs must be closer than random pairs, per view and combined (catches a wrong row mapping)."""
    rel = f"reports/embedding_sanity_{FP_SAMPLE}_{fp_of(sorted(EMB_INFO.items()))}.json"
    if locate(rel):
        with open(locate(rel)) as f:
            REPORT["embedding_sanity"] = json.load(f)
        print(f"embedding sanity (cached): {REPORT['embedding_sanity']}")
        return REPORT["embedding_sanity"]
    with stage("embed_check"):
        s1_ids = raw_frame("train", "s1", ["entity_id"])["entity_id"].values
        pool_ids = raw_frame("train", "pool", ["entity_id"])["entity_id"].values
        gt = load_parquet(locate(f"{SAMPLE_REL}/gt.parquet"))
        a, b = explode_gt(gt)
        i1 = pd.Index(s1_ids).get_indexer(a)
        i2 = pd.Index(pool_ids).get_indexer(b)
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        sel = rng.choice(len(i1), size=min(len(i1), 20000), replace=False)
        rnd = rng.integers(0, len(pool_ids), len(sel))
        res = {}
        for kind in ("name", "addr", "combined_w0.5"):
            if kind in EMB_VIEWS:
                v1, v2 = get_store(kind, "train", "s1"), get_store(kind, "train", "pool")
            else:
                v1, v2 = CombinedStore("train", "s1", 0.5), CombinedStore("train", "pool", 0.5)
            q = v1.gather(i1[sel], None, np.float32)
            pos = (q * v2.gather(i2[sel], None, np.float32)).sum(1)
            neg = (q * v2.gather(rnd, None, np.float32)).sum(1)
            res[kind] = dict(true_pair_cos_mean=round(float(pos.mean()), 4), random_pair_cos_mean=round(float(neg.mean()), 4),
                             p_true_gt_random=round(float((pos > neg).mean()), 4))
            log(f"  {kind:13s}: mean cos true pairs={res[kind]['true_pair_cos_mean']:.4f}  random pairs="
                  f"{res[kind]['random_pair_cos_mean']:.4f}  P(true > random)={res[kind]['p_true_gt_random']:.4f}")
            if pos.mean() <= neg.mean():
                raise RuntimeError(f"{kind} embeddings do not separate true from random pairs -- check the row mapping")
        REPORT["embedding_sanity"] = res
        save_json(rel, res)
    return res
