# ------------------------- exact inner-product search (copied from er_crossencoder_v2 §5, multi-GPU) -------------------------
SCORE_BUDGET_BYTES = 4e9   # score matrix per search step on one GPU (24 GB cards)


def build_ann_index(vecs, device=None):
    """vecs: (n, d) L2-normalized. Exact search: fp16 torch tensor on a GPU, else faiss/numpy on the CPU."""
    device = device or DEVICES[0]
    if device.startswith("cuda"):
        return ("torch", torch.as_tensor(np.asarray(vecs), dtype=torch.float16, device=device))
    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(vecs.shape[1])
        for start in range(0, len(vecs), 500_000):
            index.add(np.ascontiguousarray(vecs[start:start + 500_000], dtype=np.float32))
        return ("faiss", index)
    return ("numpy", np.asarray(vecs, dtype=np.float32))


def ann_search(index_handle, queries, k, chunk=8192):
    """Returns (scores, idx), each (n_queries, k). idx == -1 where fewer than k results exist.
    (copied; changes: score-matrix budget is SCORE_BUDGET_BYTES, top-k runs on the fp16 scores directly,
    and the queries go to the index's own device)"""
    kind, index = index_handle
    nq = queries.shape[0]
    out_s = np.full((nq, k), -1.0, dtype=np.float32)
    out_i = np.full((nq, k), -1, dtype=np.int64)
    if nq == 0 or k == 0:
        return out_s, out_i
    if kind == "torch":
        chunk = int(max(64, min(chunk, SCORE_BUDGET_BYTES // max(1, index.shape[0] * index.element_size()))))
    for start in range(0, nq, chunk):
        q = np.ascontiguousarray(queries[start:start + chunk], dtype=np.float32)
        if kind == "faiss":
            s, i = index.search(q, k)
            out_s[start:start + len(q)] = s
            out_i[start:start + len(q)] = i
        elif kind == "torch":
            qt = torch.as_tensor(q, dtype=index.dtype, device=index.device)
            sims = qt @ index.T
            kk = min(k, sims.shape[1])
            ts, ti = torch.topk(sims, kk, dim=1)
            out_s[start:start + len(q), :kk] = ts.float().cpu().numpy()
            out_i[start:start + len(q), :kk] = ti.cpu().numpy()
            del qt, sims, ts, ti
        else:
            sims = q @ index.T
            kk = min(k, sims.shape[1])
            ti = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            ts = np.take_along_axis(sims, ti, axis=1)
            order = np.argsort(-ts, axis=1)
            out_s[start:start + len(q), :kk] = np.take_along_axis(ts, order, axis=1)
            out_i[start:start + len(q), :kk] = np.take_along_axis(ti, order, axis=1)
    out_i[out_s < -0.5] = -1  # FAISS pads missing results with very negative scores
    return out_s, out_i


def free_index(index_handle):
    kind, index = index_handle
    dev = index.device if kind == "torch" else None
    del index
    gc.collect()
    if dev is not None:
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

# ------------------------- sharded exact search over all GPUs (new) -------------------------
MULTI_GPU_MIN_POOL = 50_000   # smaller pools are searched on one GPU (a second copy would cost more than it saves)


def _merge_topk(best_s, best_i, a, s, i, step=200_000):
    k = best_s.shape[1]
    for r0 in range(0, len(s), step):
        sl = slice(a + r0, a + r0 + min(step, len(s) - r0))
        cs = np.concatenate([best_s[sl], s[r0:r0 + step]], axis=1)
        ci = np.concatenate([best_i[sl], i[r0:r0 + step]], axis=1)
        if cs.shape[1] > k:
            top = np.argpartition(-cs, k - 1, axis=1)[:, :k]
            cs, ci = np.take_along_axis(cs, top, 1), np.take_along_axis(ci, top, 1)
        o = np.argsort(-cs, axis=1, kind="stable")
        best_s[sl], best_i[sl] = np.take_along_axis(cs, o, 1), np.take_along_axis(ci, o, 1)


def _dense_search_cpu(q_view, q_rows, p_view, p_rows, k):
    """CPU fallback (no GPU): shards + numpy running top-k."""
    nq = len(q_rows)
    best_s = np.full((nq, k), -np.inf, np.float32)
    best_i = np.full((nq, k), -1, np.int64)
    Q = q_view.gather(q_rows)
    for s0 in range(0, len(p_rows), cfg.SEARCH_SHARD_ROWS):
        pr = p_rows[s0:s0 + cfg.SEARCH_SHARD_ROWS]
        handle = build_ann_index(p_view.gather(pr), "cpu")
        s, i = ann_search(handle, Q, min(k, len(pr)))
        ok = i >= 0
        _merge_topk(best_s, best_i, 0, np.where(ok, s, -np.inf).astype(np.float32), np.where(ok, i + s0, -1))
    return best_s, best_i


def dense_search(q_view, q_rows, p_view, p_rows, k):
    """Exact top-k inner product of q_rows vs p_rows. Returns (scores, idx) with idx = position into p_rows
    (-1 = none). The pool is cut into shards spread round-robin over all GPUs (one thread per GPU). Each GPU keeps
    the queries and its running top-k ON the GPU (fp16 matmul, torch.topk merge), so the CPU only gathers vectors;
    the per-GPU lists are merged on cuda:0 at the end."""
    q_rows, p_rows = np.asarray(q_rows, np.int64), np.asarray(p_rows, np.int64)
    nq, k = len(q_rows), int(min(k, len(p_rows)))
    if nq == 0 or k <= 0:
        return np.full((nq, max(k, 0)), -np.inf, np.float32), np.full((nq, max(k, 0)), -1, np.int64)
    if DEVICE != "cuda":
        s, i = _dense_search_cpu(q_view, q_rows, p_view, p_rows, k)
        i[~np.isfinite(s)] = -1
        return s, i
    devices = DEVICES if len(p_rows) >= MULTI_GPU_MIN_POOL else DEVICES[:1]
    shard = int(min(cfg.SEARCH_SHARD_ROWS, math.ceil(len(p_rows) / len(devices))))
    shards = [(s0, p_rows[s0:s0 + shard]) for s0 in range(0, len(p_rows), shard)]
    used = devices[:len(shards)]
    Qh = q_view.gather(q_rows)                       # fp16 host copy, uploaded once per GPU
    results, errors = {}, []

    def worker(di, dev):
        try:
            with torch.cuda.device(dev):
                Q = torch.as_tensor(Qh, dtype=torch.float16, device=dev)
                bs = torch.full((nq, k), -float("inf"), dtype=torch.float32, device=dev)
                bi = torch.full((nq, k), -1, dtype=torch.int64, device=dev)
                for s0, pr in shards[di::len(used)]:
                    P = torch.as_tensor(p_view.gather(pr), dtype=torch.float16, device=dev)
                    kk = min(k, len(pr))
                    step = int(max(64, min(65536, SCORE_BUDGET_BYTES // max(1, len(pr) * 2))))
                    for a in range(0, nq, step):
                        ts, ti = torch.topk(Q[a:a + step] @ P.T, kk, dim=1)
                        cs = torch.cat([bs[a:a + step], ts.float()], 1)
                        ci = torch.cat([bi[a:a + step], ti + s0], 1)
                        ms, mi = torch.topk(cs, k, dim=1)
                        bs[a:a + step], bi[a:a + step] = ms, torch.gather(ci, 1, mi)
                        del ts, ti, cs, ci, ms, mi
                    del P
                results[di] = (bs, bi)
                del Q
        except Exception as e:   # surfaced after join
            errors.append(e)

    if len(used) == 1:
        worker(0, used[0])
    else:
        threads = [threading.Thread(target=worker, args=(di, dev), daemon=True) for di, dev in enumerate(used)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    d0 = used[0]
    bs, bi = results.pop(0)
    for di in sorted(results):                       # merge the other GPUs' lists on the first GPU, in row blocks
        s2, i2 = results.pop(di)
        for a in range(0, nq, 1_000_000):
            # via host memory: direct GPU->GPU (peer) copies silently return zeros on this machine (IOMMU / PHB)
            cs = torch.cat([bs[a:a + 1_000_000], s2[a:a + 1_000_000].cpu().to(d0)], 1)
            ci = torch.cat([bi[a:a + 1_000_000], i2[a:a + 1_000_000].cpu().to(d0)], 1)
            ms, mi = torch.topk(cs, k, dim=1)
            bs[a:a + 1_000_000], bi[a:a + 1_000_000] = ms, torch.gather(ci, 1, mi)
        del s2, i2
    best_s, best_i = bs.cpu().numpy(), bi.cpu().numpy()
    del bs, bi
    for dev in used:
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    best_i[~np.isfinite(best_s)] = -1
    return best_s, best_i


def _topk_frame(q_rows, p_rows, s, i, reverse=False):
    """Long format. Forward: q_rows are S1 and i indexes p_rows (pool). Reverse: swapped."""
    valid = i >= 0
    rank = np.broadcast_to(np.arange(i.shape[1], dtype=np.int32), i.shape)[valid]
    a = np.broadcast_to(q_rows[:, None], i.shape)[valid]
    b = p_rows[i[valid]]
    s1, pool = (b, a) if reverse else (a, b)
    return pd.DataFrame({"s1_pos": s1.astype(np.int64), "pool_pos": pool.astype(np.int64),
                         "score": s[valid].astype(np.float32), "rank": rank})

# ------------------------- BM25 (hashed sparse product; GPU sparse x sparse on all GPUs) -------------------------
from sklearn.feature_extraction.text import HashingVectorizer

BM25_N_FEATURES = 2 ** 22
_TOK_RE = re.compile(r"[a-z0-9]+")
_BM25_COLS = {"n": "name_core", "a": "addr_clean", "c": "city_norm", "s": "state_norm"}
BM25_GPU_QUERY_CHUNK = 8192   # queries per GPU sparse product (halved automatically on OOM)


def _identity_analyzer(tokens):
    return tokens


_HV = HashingVectorizer(analyzer=_identity_analyzer, n_features=BM25_N_FEATURES, alternate_sign=False,
                        norm=None, dtype=np.float32)


def _bm25_tokens(frame, rows, fields):
    cols = [(f + ":", frame[_BM25_COLS[f]].values) for f in fields]
    out = []
    for r in rows:
        toks = []
        for pre, col in cols:
            toks.extend(pre + t for t in _TOK_RE.findall(col[r]) if len(t) >= 2)
        out.append(toks)
    return out


def _hash_token_block(col_values, prefixes):
    """Hashed term-count matrix for one block of rows (col_values: one object array per field)."""
    docs = []
    for r in range(len(col_values[0]) if col_values else 0):
        toks = []
        for pre, col in zip(prefixes, col_values):
            toks.extend(pre + t for t in _TOK_RE.findall(col[r]) if len(t) >= 2)
        docs.append(toks)
    return _HV.transform(docs).tocsr()


def _bm25_matrix(frame, rows, fields, block=200_000):
    """Hashed term counts for frame[rows] over `fields`, blocks hashed in NUM_WORKERS processes."""
    rows = np.asarray(rows, np.int64)
    cols = [frame[_BM25_COLS[f]].values for f in fields]
    prefixes = [f + ":" for f in fields]
    blocks = [rows[i:i + block] for i in range(0, len(rows), block)]
    if len(blocks) > 1 and cfg.NUM_WORKERS > 1:
        mats = joblib.Parallel(n_jobs=min(cfg.NUM_WORKERS, len(blocks)), backend="loky")(
            joblib.delayed(_hash_token_block)([c[b] for c in cols], prefixes) for b in blocks)
    else:
        mats = [_hash_token_block([c[b] for c in cols], prefixes) for b in blocks]
    return sp.vstack(mats).tocsr() if mats else sp.csr_matrix((0, BM25_N_FEATURES), dtype=np.float32)


def bm25_index(frame, rows, fields):
    X = _bm25_matrix(frame, rows, fields)
    N = X.shape[0]
    df = np.bincount(X.indices, minlength=BM25_N_FEATURES)
    cap = max(cfg.BM25_MIN_DF_CAP, cfg.BM25_MAX_DF_FRAC * N)
    keep = (df > 0) & (df <= cap)
    idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
    dl = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    avgdl = max(float(dl.mean()) if N else 1.0, 1e-6)
    row_of = np.repeat(np.arange(N, dtype=np.int32), np.diff(X.indptr))
    tf = X.data
    k1, b = cfg.BM25_K1, cfg.BM25_B
    w = idf[X.indices] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[row_of] / avgdl))
    w[~keep[X.indices]] = 0.0
    W = sp.csr_matrix((w.astype(np.float32), X.indices, X.indptr), shape=X.shape)
    W.eliminate_zeros()
    return dict(WT=W.T.tocsr(), keep=keep, n_docs=N, pruned=int((df > cap).sum()), vocab=int((df > 0).sum()))


def bm25_query_matrix(frame, rows, fields, keep):
    Q = None
    for f in fields:
        M = _bm25_matrix(frame, rows, (f,))
        M.data[:] = cfg.BM25_FIELD_WEIGHTS.get(f, 1.0)   # binary presence x field weight
        Q = M if Q is None else Q + M
    Q = Q.tocsr()
    Q.data *= keep[Q.indices]
    Q.eliminate_zeros()
    return Q


def _bm25_topk_cpu(Q, WT, k, out):
    for start in range(0, Q.shape[0], cfg.BM25_QUERY_CHUNK):
        C = (Q[start:start + cfg.BM25_QUERY_CHUNK] @ WT).tocsr()
        for r in range(C.shape[0]):   # loop copied from er_crossencoder_v2 name_token_block
            lo, hi = C.indptr[r], C.indptr[r + 1]
            if hi == lo:
                continue
            data, cols = C.data[lo:hi], C.indices[lo:hi]
            if hi - lo > k:
                top = np.argpartition(-data, k - 1)[:k]
                data, cols = data[top], cols[top]
            out.append((np.full(len(cols), start + r, dtype=np.int64), cols.astype(np.int64), data.astype(np.float32)))


def _csr_to_torch(M, device):
    M = M.tocsr()
    M.sort_indices()
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int64)), torch.from_numpy(M.indices.astype(np.int64)),
                                   torch.from_numpy(M.data.astype(np.float32)), size=M.shape, device=device)


def _gpu_rowwise_topk(C, k):
    """Top-k columns per row of a CUDA CSR matrix -> (row, col, value) tensors, rows ascending, values descending."""
    crow, col, val = C.crow_indices(), C.col_indices(), C.values()
    if val.numel() == 0:
        z = torch.zeros(0, dtype=torch.int64, device=val.device)
        return z, z, val
    row = torch.repeat_interleave(torch.arange(C.shape[0], device=val.device), crow[1:] - crow[:-1])
    o = torch.argsort(val, descending=True, stable=True)
    o = o[torch.argsort(row[o], stable=True)]
    row, col, val = row[o], col[o], val[o]
    pos = torch.arange(val.numel(), device=val.device) - crow[row]
    keep = pos < k
    return row[keep], col[keep], val[keep]


def _bm25_topk_gpu(Q, WT, k, out):
    """Q @ WT on every GPU (query chunks from a shared queue, one thread per GPU), top-k per query on the GPU."""
    chunks = queue.Queue()
    for start in range(0, Q.shape[0], BM25_GPU_QUERY_CHUNK):
        chunks.put((start, min(Q.shape[0], start + BM25_GPU_QUERY_CHUNK)))
    lock, errors = threading.Lock(), []

    def run(dev, WTg, lo, hi):
        try:
            A = _csr_to_torch(Q[lo:hi], dev)
            r, c, v = _gpu_rowwise_topk(torch.sparse.mm(A, WTg), k)
            res = (r.cpu().numpy() + lo, c.cpu().numpy(), v.cpu().numpy())
            del A, r, c, v
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if hi - lo <= 64:
                raise
            mid = (lo + hi) // 2
            run(dev, WTg, lo, mid)
            run(dev, WTg, mid, hi)
            return
        with lock:
            out.append(res)

    def worker(dev):
        try:
            WTg = _csr_to_torch(WT, dev)
            while True:
                try:
                    lo, hi = chunks.get_nowait()
                except queue.Empty:
                    break
                run(dev, WTg, lo, hi)
            del WTg
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
        except Exception as e:   # surfaced after join
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(d,), daemon=True) for d in DEVICES]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def bm25_search(doc_frame, doc_rows, q_frame, q_rows, k, fields=("n", "a", "c", "s"), use_gpu=None):
    """Returns (qi, di, score, rank): positions into q_rows / doc_rows, rank within query by score."""
    use_gpu = (cfg.BM25_USE_GPU and DEVICE == "cuda") if use_gpu is None else use_gpu
    idx = bm25_index(doc_frame, doc_rows, fields)
    Q = bm25_query_matrix(q_frame, q_rows, fields, idx["keep"])
    out = []
    (_bm25_topk_gpu if use_gpu else _bm25_topk_cpu)(Q, idx["WT"], k, out)
    if not out:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32), np.zeros(0, np.int32)
    qi = np.concatenate([o[0] for o in out]).astype(np.int64)
    di = np.concatenate([o[1] for o in out]).astype(np.int64)
    s = np.concatenate([o[2] for o in out]).astype(np.float32)
    o = np.lexsort((di, -s, qi))   # deterministic order regardless of which GPU finished first
    qi, di, s = qi[o], di[o], s[o]
    return qi, di, s, within_group_rank(qi, s).astype(np.int32)

# ------------------------- exact / structured keys (new) -------------------------
EXACT_KEYS = ["exact_name", "exact_sorted_city", "exact_postcode", "exact_house_city"]


def exact_key_values(frame, rows, kind):
    c = frame["country_norm"].values[rows].astype(object)
    if kind == "exact_name":
        v = frame["name_core"].values[rows]
        ok = np.fromiter((len(x) >= 3 for x in v), dtype=bool, count=len(v))
        key = c + "|" + v
    elif kind == "exact_sorted_city":
        v, city = frame["name_sorted"].values[rows], frame["city_norm"].values[rows]
        ok = (v != "") & (city != "")
        key = c + "|" + city + "|" + v
    elif kind == "exact_postcode":
        pcode = frame["postcode"].values[rows]
        first = np.array([x.split(" ", 1)[0] for x in frame["name_core"].values[rows]], dtype=object)
        ok = (pcode != "") & (first != "")
        key = c + "|" + pcode + "|" + first
    elif kind == "exact_house_city":
        h, city = frame["house_no"].values[rows], frame["city_norm"].values[rows]
        ok = (h != "") & (city != "")
        key = c + "|" + city + "|" + h
    else:
        raise ValueError(kind)
    return np.where(ok, key, "").astype(object)


def exact_block(s1_frame, q_rows, pool_frame, pool_rows, kind, stats=None):
    qk = exact_key_values(s1_frame, q_rows, kind)
    pk = exact_key_values(pool_frame, pool_rows, kind)
    qm, pm = qk != "", pk != ""
    empty = pd.DataFrame({"s1_pos": np.zeros(0, np.int64), "pool_pos": np.zeros(0, np.int64),
                          "score": np.zeros(0, np.float32), "rank": np.zeros(0, np.int32)})
    if not qm.any() or not pm.any():
        return empty
    codes, uniq = pd.factorize(np.concatenate([qk[qm], pk[pm]]))
    nqm = int(qm.sum())
    qc, pcd = codes[:nqm], codes[nqm:]
    nq = np.bincount(qc, minlength=len(uniq))
    npl = np.bincount(pcd, minlength=len(uniq))
    both = (nq > 0) & (npl > 0)
    allowed = both & (npl <= cfg.EXACT_MAX_BLOCK) & (nq * npl <= cfg.EXACT_MAX_PAIRS)
    if stats is not None:
        stats.update(shared_keys=int(both.sum()), skipped_blocks=int((both & ~allowed).sum()),
                     queries_with_key=nqm, queries_in_skipped=int((both & ~allowed)[qc].sum()))
    dq = pd.DataFrame({"code": qc, "s1_pos": q_rows[qm]})
    dp = pd.DataFrame({"code": pcd, "pool_pos": pool_rows[pm]})
    m = dq[allowed[dq["code"].values]].merge(dp[allowed[dp["code"].values]], on="code")
    if not len(m):
        return empty
    s1p, pp = m["s1_pos"].values.astype(np.int64), m["pool_pos"].values.astype(np.int64)
    col = "addr_clean" if kind in ("exact_name", "exact_sorted_city") else "name_clean"
    score = np.empty(len(m), np.float32)
    for a in range(0, len(m), cfg.PAIR_BATCH):
        sl = slice(a, a + cfg.PAIR_BATCH)
        score[sl] = _pairwise(rf_fuzz.token_set_ratio, s1_frame[col].values[s1p[sl]],
                              pool_frame[col].values[pp[sl]], _rf_default_process) / 100
    grp = s1p * 2 + (pool_frame["pool_source"].values[pp] == "S3")
    rank = within_group_rank(grp, score).astype(np.int32)
    keep = rank < cfg.EXACT_TOP_K
    if stats is not None:
        stats["pairs_before_topk"] = int(len(m))
    return pd.DataFrame({"s1_pos": s1p[keep], "pool_pos": pp[keep], "score": score[keep], "rank": rank[keep]})