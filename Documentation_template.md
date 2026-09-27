# Methodology: Business Entity Resolution

**Team:** BlowTheGPUS!!
**Final submission:** submission_3 — **the files in `output/` are this version** (leaderboard macro F0.5 **0.975**, the highest of the three we produced; see §6 Results for submission_1 / submission_2 comparisons)
**Code:** `code/business_entity_resolution/` (run instructions in its `README.md`)

---

## 0. Submission package

```
BlowTheGPUS!!_submission.zip
├── output/
│   ├── matching_results.tsv              # final matches — submission_3 (highest-scoring)
│   └── candidate_pairs.tsv               # blocking candidate set — submission_3
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── embeddings/               # Qwen3-Embedding-0.6B: model + store builder
│       │   ├── stage1_retrieval_rerank/  # normalisation, blocking, 74-feature XGBoost reranker
│       │   ├── stage2_cross_encoder/     # XLM-R cross-encoder + 2nd stage  -> submission_2
│       │   ├── stage3_adapter_clustering/# adapter retrieval + clustering  -> submission_3 (final)
│       │   ├── scripts/                  # run_all.sh (end-to-end) + individual launchers
│       │   └── analysis/                 # APPROACH.md, experiments.tsv, decision files
│       ├── README.md                     # reproduction steps: data -> blocking -> matching -> output
│       └── requirements.txt              # pinned dependencies
└── Documentation_template.md             # this file
```

## 1. Problem summary

For each Source-1 business record, we find every Source-2 and Source-3 record that describes the same business. Records have only `entity_id, business_name, business_address, country` — **no phone, email or city column**. Postcode and house number are parsed out of the address text.

| | S1 records | Pool (S2 + S3) | Countries |
|---|---|---|---|
| Train | 2,206,821 | 10,320,219 | US, India |
| Test | 1,732,544 | 9,969,589 | India, US, **France (test only, no labels)** |

What the training data shows:
- 3.46 true matches per S1 on average, 11 at most;
- 5.6% of S1 records have no match;
- a pool record matches at most one S1 (used later as the exclusivity rule);
- no true match crosses countries (used later to block per country).

The metric is macro F0.5 per S1 entity, so precision is weighted twice as much as recall.

## 2. Methodology (complete pipeline)

```
Preprocessing
      │   name/address cleaning, core name, legal-suffix class, phonetic + acronym keys,
      │   postcode + house-number extraction  (src/stage1_retrieval_rerank/er_pipeline.ipynb)
      ▼
Normalisation
      │   per-field cleaned/canonical columns used by every retriever and feature below
      ▼
Embedding generation
      │   Qwen3-Embedding-0.6B: name view 256-d, address view 512-d, int8
      │   (src/embeddings/qwen_embed.py + build_embedding_store.py)
      ▼
┌─────────────────┬─────────────────┬──────────────────────────┐
│  Dense retrieval │      BM25       │ Exact / structured        │
│  forward top-50  │  top-50/source  │ blocking: name key,       │
│  + reverse top-3 │  (hashed sparse)│ postcode key               │
│  per source, GPU │                 │                           │
└─────────────────┴─────────────────┴──────────────────────────┘
      ▼
Candidate union
      │   union of all retrievers, capped at 120/S1, ordered by best rank
      │   + adapter fused top-10 (submission_3 only)           → candidate_pairs.tsv
      ▼
Feature engineering
      │   74 pairwise features: string similarity, embedding cosine, postcode/
      │   house-number agreement, retrieval ranks/scores, entity-context features
      ▼
XGBoost / reranking
      │   stage-1 XGBoost  →  XLM-R cross-encoder on the uncertain band  →
      │   second-stage XGBoost (stage-1 p, CE logit, ranks, clustering features)
      ▼
Threshold
      │   p ≥ 0.66, exclusive (each pool record assigned to its best-scoring S1 only)
      ▼
Final matches                                                    → matching_results.tsv
```

## 3. Candidate generation / blocking

Blocking runs per **country × source** partition (safe: no true pair crosses countries).

| Retriever | What it is |
|---|---|
| **Qwen embeddings** | Qwen3-Embedding-0.6B vectors, name (256-d) + address (512-d), int8, cosine similarity |
| **Dense retrieval** | forward search, top-50 per source, on the combined vector `[√w·name, √(1−w)·addr]` (w = 0.4, chosen by a sweep); exact fp16 matmul, both GPUs, merged on host |
| **Reverse retrieval** | pool → S1 dense search, top-3, to recover S1 records that a forward search alone misses |
| **BM25** | sparse hashed BM25 on GPU, top-50 per source |
| **Exact matching** | exact-key blocking on the normalised name key and the postcode key |
| **Postcode / house-number** | postcode is used as an **exact blocking key** (above); house number is **not** used to block (there is no reliable exact house-number key — matches deliberately have altered house numbers) — it is instead a **feature** (§5) fed to the reranker, which learns how much a number mismatch should cost |
| **Candidate union** | the union of every retriever above, deduplicated, ordered by each candidate's best rank across retrievers |
| **Candidate limits** | capped at **120 candidates per S1**. Recall at this cap is 0.9742 on train (US 0.983, India 0.962); oracle macro F0.5 at that recall is 0.9913 |
| **Adapter retrieval** (submission_3 only) | a learned two-tower MLP over the same Qwen vectors (see §4) adds its fused top-10 per S1 (~4.5 new candidates/S1 not already in the union). Candidate recall rose from 0.9745 → 0.9875, India gaining the most |

`candidate_pairs.tsv` = the stage-1 union ∪ the new adapter pairs.

## 4. Model architecture

| Component | Detail |
|---|---|
| **Embedding model** | Qwen3-Embedding-0.6B (open weights, 0.6B params, under the competition's 8B limit), last-token pooling |
| **Embedding dimensionality / precision** | name view 256-d, address view 512-d (Matryoshka truncation of a wider representation), fp16 compute, L2-normalised and **quantised to int8** for storage (cosine similarity survives quantisation since every vector is unit-norm) |
| **Retrieval** | dense (forward + reverse) + BM25 + exact keys, see §3; union capped at 120/S1 |
| **Pairwise features** | 74 features per (S1, candidate) pair, see §5 |
| **XGBoost (stage 1)** | GPU, depth 8, learning rate 0.07, 400 trees; trained on 20% of train entities (5-fold entity-grouped CV + a nested holdout); reads training blocks through a streaming `DataIter`/`QuantileDMatrix` so the ~15 GB feature matrix is never fully materialised |
| **Cross-encoder (2nd stage)** | `xlm-roberta-base` (multilingual — covers French, which has no training labels), input `"name \| address"` as a sentence pair, max length 128; scores the **uncertain band** (stage-1 p ∈ [0.05, 0.95]: 1.15% of pairs, 88.6% of the missed positives) plus all new adapter pairs; band AUC rose from 0.88 to 0.98 |
| **Second-stage model** | a second, small XGBoost (depth 6, 600 trees) over [stage-1 p, cross-encoder logit, rank in entity, top-1/top-2 p, gap to top-1, number of band candidates] (submission_2), extended in submission_3 with the adapter rank/cosine, an `is_new` flag, and clustering features (§5) |
| **Adapter model** (submission_3) | two MLP towers (768 → 1024 → 256, linear skip, L2-normalised), one for S1 records and one for the pool, trained with InfoNCE (τ = 0.05, 6 hard negatives from stage-1 p, multi-positive mask, batch 2048, 4 epochs, ~50 s) |
| **Thresholding** | a single decision rule, chosen on CV/FIT data only, never on the reported split: **p ≥ 0.66, exclusive** — each pool record kept only for its highest-scoring S1 (a pool record matches at most one S1 in the data, so this directly reflects the label structure) |

## 5. Feature engineering

74 pairwise features, computed in 50k-S1 blocks by forked workers (parent sends each worker its chunk's rows, so frames are never copied):

- **Name similarity:** RapidFuzz string similarity on raw and cleaned names, core name (legal suffix stripped), legal-suffix-class agreement, phonetic key match, acronym key match.
- **Address similarity:** RapidFuzz string similarity on raw and cleaned addresses.
- **Embedding cosine:** Qwen3 name-view cosine and address-view cosine between the S1 record and the candidate.
- **Phone / email:** **not applicable** — the dataset has no phone or email column (§1); no such features exist.
- **Postcode:** exact/partial postcode agreement (also used as a blocking key, §3).
- **House number:** house-number agreement, treated as a soft feature rather than a blocking key, because matches deliberately corrupt house numbers (about 31% of "different number" top candidates are still true matches).
- **Source-specific features:** which pool source (S2 vs S3) the candidate came from, and per-source retrieval behaviour, since S2 and S3 differ in size and field quality.
- **Retrieval features:** rank and score from every retriever (dense forward, dense reverse, BM25, exact), and whether a candidate is new-from-the-adapter (`is_new`, submission_3).
- **Contextual / entity-level features:** candidate count and score-gap-to-best within the entity's candidate set; and (submission_3 second stage) **clustering features** — a candidate's similarity to the entity's own top-3 confident candidates (adapter cosine, raw name/address cosine, p-weighted max, near-duplicate count), which lets the model use an entity's already-confident matches as context for its uncertain ones.

## 6. Validation methodology (leakage-safe)

- Train entities are split by **entity** (never by row, so no record of one entity leaks across a split):
  - 20% train the stage-1 reranker (5-fold entity-grouped CV + a nested 12% holdout);
  - the other 80% are never used to train it.
- The cross-encoder trains only on the CV entities.
- The second stage and the decision rule are fit on one part of the untouched 80% (FIT); results are **reported on a disjoint part (REP)** and on the nested holdout, both never seen by any model or threshold.
- Rules that depend on competition between S1 records (exclusivity) are evaluated with **all** train S1 competing, matching how test actually runs — a rule that looked useless with only the CV entities competing was worth +0.0026 once all competitors were present.
- **France** has no labels at all; we rely on the multilingual cross-encoder and on features that don't depend on language (postcode, house number, embedding cosine). The France contribution is only inferred from the leaderboard delta.

## 7. Results

| Version | What changed | Held-out macro F0.5 | Leaderboard |
|---|---|---|---|
| submission_1 | Stage-1 blocking + XGBoost, threshold 0.66 | 0.9483 (80% eval) | 0.934 |
| (phase 2, not a separate submission) | + exclusive rule | 0.9508 | — |
| submission_2 | + cross-encoder second stage, threshold 0.50 exclusive | 0.9782 (75% eval); P 0.993, R 0.954 | 0.971 |
| **submission_3 (final, in `output/`)** | + adapter candidates + clustering features, threshold 0.66 exclusive | 0.9796 (REP2, vs 0.9765 for the sub2 pipeline on the same entities) | **0.975** |

Things tried that gave no measured gain: per-country thresholds, margin rules, top-1 rescue (all within ±0.0001); an adapter top-20 instead of top-10 (recall +0.18 pt, precision −0.06 pt, net +0.00001 — not shipped).

## 8. Error analysis

Baseline OOF losses break down as: 44% a true match retrieved but scored below threshold; 17% false positives from look-alike businesses; 14% never retrieved; the rest mixed. Labels are deliberately noisy (changed house numbers, altered streets, renamed businesses), so every decision above was validated on held-out labelled data rather than by rule/inspection. India is harder than the US; entities with exactly one true match are the hardest group (F0.5 0.871 at baseline).

## 9. Other relevant information

- **Hardware:** 2 × NVIDIA RTX PRO 4000 Blackwell (24 GB each), 24 CPU cores, 62 GB RAM, Linux.
- **Multi-GPU processing:** GPU peer-to-peer is **broken** on this hardware (a direct `cuda:1 → cuda:0` copy silently returns zeros). We never use `DataParallel`/`DDP`; every multi-GPU stage runs as independent per-GPU workers (e.g. cross-encoder scoring, retrieval search) merged back on the host, with a self-test that checks the merge against brute force. Moving a model between GPUs always goes `.cpu().to(dev)`, never a direct device-to-device `.to()`.
- **Batching:** length-bucketed dynamic batching for embedding generation (a token budget, not a fixed batch size, so short/long records don't waste VRAM); cross-encoder training batch 64, adapter training batch 2048; cross-encoder scoring in length-sorted, 50k-pair chunks.
- **Checkpointing:** every large stage is checkpointed and resumable — retrieval per country/source partition, cross-encoder training every 1,000 steps, cross-encoder/adapter scoring per chunk, the reranker's CV folds and OOF predictions, and the final decision rule/second-stage model files. Rerunning any stage picks up where it left off instead of recomputing.
- **Memory optimizations:** feature extraction runs in forked workers that receive only their chunk's rows from the parent (avoids copy-on-write blow-up, which caused the first OOM); worker count is capped from a measured per-worker RSS against a 28 GB budget; recall/report stages run block-wise instead of loading the full frame; the test-time scoring path uses numpy arrays and scores only the rows that need it (~25 GB peak) instead of assembling a 200M+-row DataFrame; XGBoost reads training data through a streaming `DataIter` so the full feature matrix is never materialised in RAM.
- **Reproducibility:** pinned versions in `requirements.txt`; the full run is `src/scripts/run_all.sh` (embeddings → stage 1 → cross-encoder → adapter/clustering → validated output), parameterised by `$ER_ROOT` so it runs on any machine with the same data; only open-weight local models are used (Qwen3-Embedding-0.6B, XLM-R base) — no external APIs or services.
- **Validation methodology:** see §6.
