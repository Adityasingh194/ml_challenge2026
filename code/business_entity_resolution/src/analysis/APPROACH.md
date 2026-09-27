# Entity-resolution pipeline: current approach and state

Last updated 2026-09-27, ~19:20 IST. This file explains how the pipeline works, what has been measured, where everything lives, and what comes next. It gives a later session (or another person) the full context.

---

## 1. The task and the metric

- **Task:** for every Source-1 (S1) business record, find all records in Source-2 and Source-3 (the "pool") that describe the same real-world business.
  - Train: S1 2,206,821 records (US 1.32M, India 0.88M); pool 10,320,219 (S2 5.03M + S3 5.29M). Ground truth is in `aksfinal_clean/train/train_ground_truth.tsv`.
  - Test: S1 1,732,544 records (India 0.81M, US 0.66M, **France 0.26M**); pool 9,969,589. **France appears only in test, so there are no French labels.**
- **Fields:** only `entity_id, business_name, business_address, country`. There is no phone, email or city column. Postcode and house number are parsed from the address.
- **Match statistics (train):**
  - 3.46 true matches per S1 on average, 11 at most (≤5 from S2, ≤6 from S3);
  - 5.6% of S1 have no match;
  - a pool record matches at most one S1;
  - no true pair crosses countries.
- **Metric (as implemented, `macro_f05` in cell 28):** macro F0.5 per S1 entity.
  - Per S1: F0.5 = 1.25PR / (0.25P + R) over its predicted set.
  - An S1 with no true matches scores 1 for an empty prediction and 0 otherwise.
  - The score is the mean over all S1.
  - No official scorer ships with the data. The leaderboard score (0.934) fits macro better than micro (our micro F0.5 was 0.963).
- **Output files:**
  - `matching_results.tsv` (`source1_entity_id \t matched_entity_ids`, comma-separated, may be empty);
  - `candidate_pairs.tsv` (`source1_entity_id \t candidate_entity_ids`).
  - Both have one row per test S1, in `test_source1` order.

## 2. Machine and constraints

- 2 × NVIDIA RTX PRO 4000 Blackwell (**24 GB** each), 24 CPU cores, 62 GB RAM, venv at `trial work/venv` (torch 2.14 + CUDA 13, xgboost 3.4.1, pandas 3.0.6).
- **Broken GPU peer-to-peer:** a direct `cuda:1 → cuda:0` copy returns zeros (PHB topology / IOMMU). Every cross-GPU merge goes through host memory. **Never use DataParallel or DDP here.** Multi-GPU work runs as independent per-GPU workers merged on the host. Cell 0b reports "P2P BROKEN".
- Limits set by the user:
  - host RAM ≤ 28 GB per job, GPU ≤ 22 GB each;
  - one job at a time;
  - no smoke runs unless asked;
  - every large stage checkpointed and resumable;
  - logs at every step;
  - never overwrite the test submission files.

## 3. Code layout

- Notebook: `/home/parth/Downloads/kaagle.ipynb` (60 cells), built from sources in the session scratchpad `cells/*.py|md` with `build_nb.py`. Original Kaggle version: `kaagle.kaggle_original.ipynb`.
- Runner: `scratchpad/nbrun.py`, a persistent Jupyter kernel that executes cells by index (`start` / `run <conn> <nb> <idx…|code:…>`).
- Experiments (outside the notebook): `er_work/experiments/`:
  - `rules_phase2.py`, `ce_pilot.py`, `ce_full.py`;
  - logs: `experiments.log` (shared, timestamped) and `*_run.log`;
  - tracker: `experiments.tsv`;
  - reference: `baseline_v1.json` (read-only).

| Notebook cell | Content |
|---|---|
| 2 / 3 / 4 | 0a install, 0b GPU check (incl. P2P test), 0c embeddings check / S3 sync |
| 6 | config (`CFG`, profiles `full` / `smoke`, `ER_SET` overrides) |
| 8 / 10 | setup; helpers (`log`, `stage` with per-stage RSS/GPU peaks, `mem_mark`, `track_f05`, `save_parquet` streaming) |
| 12 | TSV → parquet copies, embedding alignment (`INPUT_CACHE_DIR = er_work/inputs`) |
| 14 / 16 | normalisation (parallel), raw-table loading |
| 18 | embedding stores (precomputed Qwen3-0.6B name 256-d + addr 512-d, int8) |
| 20 | search: `dense_search` (fp16, both GPUs, host merge), `bm25_search` (sparse, GPU), `exact_block` |
| 22 | frames `FR`, contexts, `run_retrievers`, `load_retr`, `truncate_retr`, **`build_union`** |
| 24 | `recall_report` (block-wise, memory-safe) |
| 26 | `pair_features` (74 features), `features_in_chunks` (forked workers, RAM-budgeted) |
| 28 | truth, `macro_f05`, rules (`apply_rule_v3`, `search_rule_v3`, `eval_rule`, lazy `_RuleCache`) |
| 30 | XGBoost trainer (`entity_folds`, `_cv_oof`, `train_and_calibrate`, streamed `QuantileDMatrix`) |
| 32 | stages: weight sweep, diag recall, `train_model`, `train_eval`, gate, `test_inference` |
| 34 | submission writer and strict validation |
| 41–59 | run cells: selftest, sample, embed_check, weight_sweep, diag_recall, train_model, train_eval, test_inference, submit, report |

## 4. Pipeline (baseline, submission_1)

1. **Inputs:** TSV → parquet (row counts verified). The precomputed Qwen3-Embedding-0.6B vectors are re-ordered once into TSV-order memmaps.
2. **Normalisation:** name and address cleaning, core name, legal suffix, phonetic and acronym keys, postcode, house number.
3. **Retrieval, per country × source partition** (0 cross-country positives, so buckets are safe):
   - dense forward top-50 per source, with the combined vector `[√w·name, √(1−w)·addr]`, where `w=0.4` was chosen by a weight sweep;
   - dense reverse (pool → S1) top-3;
   - BM25 top-50 per source (hashed sparse, on GPU);
   - exact keys (name, postcode);
   - TF-IDF off (no recall gain).
   - The union is capped at **120 candidates per S1**, ordered by best rank. It is checkpointed per partition under `er_cache/retrieval/{ctx}_{fp}/`.
4. **Features:** 74 per pair: name, address, number, postcode agreement, embedding cosines, retrieval ranks and scores, and entity context. Computed in 50k-S1 blocks by forked workers that never touch the frames: the parent sends each chunk's rows. This avoids copy-on-write blow-up.
5. **Reranker:** XGBoost on GPU (depth 8, learning rate 0.07, 400 trees) on **20% of train entities**:
   - 5-fold entity-grouped CV on 88% of them, plus a 12% nested holdout;
   - the other **80% of entities are never trained on** (the evaluation split).
6. **Decision rule:** chosen on CV OOF only (simplest within 0.001). Baseline: global threshold 0.66, non-exclusive.
7. **Test:** the same retrieval, features, model and rule, then strict validation (header, row order, ids, matches ⊆ candidates).

## 5. Results so far (all measured)

| Stage | Macro F0.5 | Notes |
|---|---|---|
| Retrieval ceiling (perfect reranker) @120, all train | **0.9913** | union recall 0.9742 (US 0.983, India 0.962) |
| Baseline CV OOF | 0.9481 | |
| Baseline nested holdout | 0.9493 | P 0.980, R 0.902 |
| Baseline 80% eval (1.77M entities) | **0.9483** | India 0.9396, US 0.9540 |
| **Leaderboard, submission_1** | **0.934** | implies France ≈ 0.873 if US/India transfer |
| Phase 2: exclusivity rule (80% eval) | 0.9508 (+0.0026) | pool record to its best S1 only; other rule variants gave nothing |
| **Cross-encoder + second stage, 75% eval (1.33M never-used entities)** | **0.9782 (+0.0272)** | P 0.9926, R 0.9539, India 0.9743, US 0.9808; holdout 0.9789 |
| **Leaderboard, submission_2** | **0.971** | +0.037; implied France ≈ 0.942 (from ≈0.873) |

### Where F0.5 was lost (baseline OOF analysis)
- 44%: true match retrieved but scored below the threshold (2nd–5th variant records of the same business);
- 17%: false positive only (70% look-alike distractors, 30% another S1's record);
- 14%: never retrieved;
- 9%: false positive plus a missed match;
- 9%: a zero-match entity got a false positive;
- 7%: both kinds of miss.

India is weaker than the US, and entities with exactly one true match are the hardest (0.871).

### Key lessons
- **Exclusivity looked useless on CV (+0.0005)** because only the CV entities competed there. With all S1 competing, as on test, it is worth +0.0026. Rules that depend on competition must be evaluated with all competitors present.
- **Rule-based or judgement labels are unreliable on this data.** True matches are deliberately corrupted (changed house numbers, altered streets, renamed businesses). Among the model's top candidates, 31% of "different number" pairs are true matches. A rule labeller estimated precision at 0.58 when the truth was 0.987. France can't be labelled by surface cues.
- **The cross-encoder is the big lever:** uncertain-band AUC rose from 0.88 (stage-1) to 0.98.

## 6. Cross-encoder design (`experiments/ce_full.py`)

- **Band:** pairs with stage-1 p in [0.05, 0.95]: 1.15% of pairs, holding 88.6% of the below-threshold missed positives. Test band: 3.29M pairs; eval band 2.43M.
- **Model:** `FacebookAI/xlm-roberta-base` (multilingual, including French). Input `"name | address"` for S1 and candidate as a sequence pair, max length 128.
- **Training:** all 532k band pairs of the **CV training entities** (41% positive), 1 epoch, batch 64, lr 2e-5, bf16, **single GPU** (P2P). About 460 pairs/s, 19 min. Checkpoint every 1,000 steps.
- **Scoring:** both GPUs, each with its own model copy, 50k-pair chunks saved to disk (resumable), length-sorted batches, about 6.4k pairs/s.
- **Second stage:** XGBoost (depth 6, 600 trees) on [stage-1 p, cross-encoder logit, rank in entity, top-1 p, top-2 p, number of band candidates, gap to top-1]. Applied only to band pairs; other pairs keep stage-1 p.
- **Leakage-safe protocol:**
  - the cross-encoder is trained on CV entities only;
  - the second stage and rule are fitted on **25% of the eval entities**;
  - results are reported on the other **75%** and on the nested holdout, both never used;
  - exclusivity is evaluated with all train S1 competing.
- **Chosen rule:** threshold 0.50, exclusive.
- **Outputs:** `experiments/ce_full/` (model, score chunks, `stage2.json`, `result.tsv`) → **`output/submission_2/`** (+ `submission_info.json` with hashes).

## 7. Files and checkpoints

| What | Where |
|---|---|
| Baseline test submission (read-only copy) | `er_work/output/baseline_v1/` (SHA-256 `a1c8d3d5…` / `6c2e163f…`) |
| submission_1 (baseline, never rewritten) | `er_work/output/submission_1/` |
| submission_2 (cross-encoder) | `er_work/output/submission_2/` |
| Retrieval partitions | `er_cache/retrieval/{train_a3b06eb911, test_5120fd2099, diag_609a049b5a}/{retriever}/*.parquet` |
| Training set (features + labels, 52.8M rows) | `er_cache/retrieval/train_a3b06eb911/trainset_68403d2268/block_*.parquet` |
| Stage-1 models, OOF | `er_cache/models/train_4331827724/` (`final.json`, `cv_model.json`, `oof_1.npy`, per-fold files) |
| Train-entity out-of-sample p | `…/model_4331827724/train_entities_oos.parquet` |
| 80% eval predictions (p, y) | `…/eval_4331827724/block_*.parquet` |
| Test stage-1 predictions | `er_cache/retrieval/test_5120fd2099/scores_*/final_preds.parquet` |
| Reports / F0.5 checkpoints | `er_cache/reports/*.json`, `reports/f05_progress.json`, `output/progress.log` |
| Experiment tracker | `er_work/experiments/{experiments.tsv, experiments.log, baseline_v1.json}` |

## 8. Operational fixes made along the way (don't undo)

1. **P2P:** host-mediated merges; a self-test compares against brute force.
2. **Memory:**
   - `recall_report` works block-wise (it used to peak at 17 GB);
   - low-cardinality string columns share objects;
   - numeric parquet is written as it streams;
   - `stage()` records per-stage `VmHWM`;
   - `mem_mark` lines are logged.
3. **Forked-worker copy-on-write** caused the first OOM crash. Workers now receive chunk rows from the parent, `gc.freeze()` is used, chunks are fixed at 20k pairs, and the worker count is capped by `RAM_BUDGET_GB=28` from the measured per-worker memory.
4. **Rule step over 265M pairs:** O(n) scatter-max rule cache, boolean-mask breakdowns, chunked `isin`.
5. **XGBoost reads the training blocks through a `DataIter`**, so the 15 GB X matrix is never assembled.
6. **Freshly normalised frames become `object` strings.** pandas 3 builds Arrow `str` columns there, which slowed features 2.5× and made test feature inputs differ from train.
7. `QUERY_BLOCK_ROWS = 50_000`, `NUM_WORKERS = 20`, and the gate is overridden (`FORCE_PAST_GATE`) for the baseline, since union recall 0.974 is below the 0.99 gate.

## 9. Next steps (planned order, one at a time, each measured on the 75% eval split + holdout)

1. ~~Submit `submission_2`~~: done, leaderboard **0.971**.
2. **More from the cross-encoder:**
   - a wider band or top-k scoring (pairs with p > 0.95 or < 0.05 that are wrong);
   - a second epoch or more training pairs;
   - adding base features to the second stage.
3. **Context / cluster features:** similarity of a candidate to the entity's confident matches.
4. **Look-alike features:** name and token frequency, IDF overlap, address specificity.
5. **Retrieval** (never-retrieved bucket, about 0.7 points; union ceiling 0.991):
   - transliterated BM25 for non-Latin candidates;
   - a larger cap only where it helps (120 → 200 gives +0.3 recall points).
6. **France:** no labels. Human labels on the audit file (`output/audit/audit_France.tsv`) are the only reliable measurement. Self-training with confident pseudo-labels is possible but can't be validated on France.

Reaching 0.99 on test needs near-perfect reranking **and** better retrieval (the current ceiling is 0.991) **and** France at US level. Report only measured gains.

## 10. Adapter + clustering (2026-09-27 evening)

- **Adapter** (`adapter_pilot.py`): two MLP towers (768→1024→256 + linear skip, L2-normalised) over the int8 Qwen name+addr vectors. InfoNCE, tau 0.05, batch 2048, 6 hard negatives (the highest stage-1-p non-matches), multi-positive mask, 4 epochs. It trains in **50 s**.
  - Training entities: 441,535 (the 25% eval "fit" split), disjoint from the reranker's entities.
  - Held-out recall against the **full 10.3M pool** (60k report entities):

    | Fused top-K | Fused alone | Union with existing 120 |
    |---|---|---|
    | 10 | 0.960 | **0.9872** (was 0.9736) |
    | 20 | 0.972 | 0.9899 |
    | 30 | 0.977 | 0.9912 |
    | 50 | 0.981 | 0.9925 |

  - India goes 0.961 → 0.989 at K=30.
- **Integration** (`adapter_fast.py`, `finish_test_lean.py`) skips the stage-1 retrain:
  - new candidates = fused top-10 pairs not already in the union (4.25–4.7 per S1);
  - they are scored by the existing cross-encoder;
  - second stage v3 adds fused rank and cosine, an `is_new` flag, and **clustering features**: similarity of each candidate to the entity's top-3 confident candidates (adapter cosine, raw name and address cosine, p-weighted max, top-3 membership, near-duplicate count).
  - It is fitted on FIT2 (250k report entities) and measured on **REP2 (250k held-out entities)**.
- **Result on REP2** (rule chosen on FIT2):

  | Pipeline | Macro F0.5 | Delta |
  |---|---|---|
  | sub2 pipeline | 0.97646 | — |
  | + adapter | 0.97933 | **+0.0029** |
  | **+ adapter + clustering** | **0.97961** | **+0.0032** (clustering itself +0.0003) |

  - Candidate recall on REP2 went 0.9745 → 0.9875; misses fell 44.3k → 36.1k; India +0.0069.
- **Test:** 7.37M new pairs, 10.66M cross-encoder pairs, rule threshold 0.66 exclusive → `output/submission_3/`.
- **Operational lessons:**
  - **(a)** Moving a model between GPUs with `.to("cuda:1")` also hits the broken P2P and returns zeros. Always use `.cpu().to(dev)`.
  - **(b)** Entity ids load as int32, so `s1 * n_pool` overflows. Cast to int64 first.
  - **(c)** Wide 216M-row DataFrames reached 45 GB RSS. The test path now uses numpy arrays and computes features only for the scored rows (about 25 GB).
  - **(d)** Cluster batches are capped at 250k rows (1M caused a GPU OOM).

### K=20 follow-up (22:41)
- On REP2, K=20 adapter + clustering scored **0.97962**, against 0.97961 at K=10: **no gain**.
  - Candidate recall rose 0.9875 → 0.9901 and final recall rose by 0.0018.
  - Precision fell by 0.0006 (+536 false positives), cancelling the recall gain.
  - The test path was not run; **submission_3 (leaderboard 0.975) remains the best submission**.
- **Lesson:** deeper adapter pairs need a better scorer before more candidates help. The next step is cross-encoder epoch 2, trained on adapter-only pairs plus band pairs; K=50 with a cheap pre-filter can follow after that.
- Export: `output/adapter_candidates_test.tsv` (34.65M rows, top-20) and `adapter_candidates_test_France.tsv`.
- **Tooling fixes applied:**
  - top-K prefix locking (fp16 ties reorder about 10% of rows);
  - cross-encoder chunk reuse only for an identical pair order (K=15 chunks were misaligned for K=20 and had to be deleted);
  - never run two fused searches on the same GPUs at once (OOM).
