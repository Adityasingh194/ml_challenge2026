# Business entity resolution: reproduction guide

This folder regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the training and test TSVs. The final submission is **submission_3**: stage-1 reranker, then the cross-encoder second stage, then adapter candidates with clustering features (leaderboard 0.975).

## 1. Environment

- Linux, Python 3.12.3, CUDA 13.0.
- Tested on 2 × NVIDIA RTX PRO 4000 Blackwell (24 GB each), 24 CPU cores, 62 GB RAM.
- Peak usage is about 32 GB RAM and 22 GB per GPU.
- Disk: about 80 GB free for embeddings, caches and outputs.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Two models are downloaded from the Hugging Face Hub on first use, both open weights and under 8B parameters:

- `Qwen/Qwen3-Embedding-0.6B`: set `QWEN_MODEL` to use a local copy.
- `FacebookAI/xlm-roberta-base`: the cross-encoder.

## 2. Data layout

Set `ER_ROOT` to a working directory that contains the competition data:

```
$ER_ROOT/data/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
$ER_ROOT/data/test/{test_source1,test_source2,test_source3}.tsv
```

All intermediate files are written under `$ER_ROOT`. What each folder is for:

| Folder | Contents |
|---|---|
| `data/` | the raw TSVs you provide (see above) |
| `embedding_store/` | Qwen3-Embedding-0.6B vectors for every record, int8, chunked (built by step 1) |
| `pipeline_run/` | everything the pipeline produces |
| `pipeline_run/aligned_inputs/` | the TSVs copied to parquet, and the embeddings re-ordered to match TSV row order (a one-time alignment step, so later stages can index by row) |
| `pipeline_run/cache/` | checkpoints: blocking/retrieval results, extracted features, trained models, F0.5 reports — kept so a rerun skips finished stages instead of recomputing them |
| `pipeline_run/scratch/` | temporary per-run working files; safe to delete between runs |
| `pipeline_run/experiments/` | cross-encoder and adapter training runs, decision-rule files, logs |
| `pipeline_run/output/submission_N/` | each stage's `matching_results.tsv` / `candidate_pairs.tsv` (N = 1, 2, 3; **submission_3 is final**) |

Run with a fresh, empty `$ER_ROOT` (other than `data/`) the first time — the folders above are created automatically.

## 3. Run end to end

```bash
export ER_ROOT=/path/to/workdir
PY=$(which python) bash src/scripts/run_all.sh
```

`run_all.sh` runs the six steps below in order. Each step is checkpointed, so rerunning the script resumes where it stopped. It copies the two final files into `../../output/`.

| # | Step | Code | Output |
|---|------|------|--------|
| 1 | Qwen3-Embedding-0.6B vectors: name 256-d, address 512-d, int8, raw text | `src/embeddings/build_embedding_store.py` (uses `qwen_embed.py`) | `$ER_ROOT/embedding_store/` |
| 2 | **Stage 1**: normalisation, **blocking** (dense forward/reverse, BM25, exact keys, union capped at 120 per S1), 74 pair features, XGBoost reranker, decision rule | `src/stage1_retrieval_rerank/er_pipeline.ipynb`, run by `run_notebook.py` | `pipeline_run/output/submission_1/` |
| 3 | **Cross-encoder** (XLM-R base) on the uncertain band, plus an XGBoost second stage | `src/stage2_cross_encoder/ce_full.py` | `pipeline_run/output/submission_2/` |
| 4 | **Adapter**: two-tower InfoNCE fusion of the Qwen vectors | `src/stage3_adapter_clustering/adapter_pilot.py` | `pipeline_run/experiments/adapter/adapter.pt` |
| 5 | Adapter top-10 candidates, cross-encoder scoring, clustering features, second stage v3 and rule (fitted on FIT2, measured on REP2) | `adapter_fast.py` (`K_NEW=10 SKIP_TEST=1`) | `pipeline_run/experiments/adapter_fast/` |
| 6 | Test path and strict validation | `finish_test_lean.py` (`K_NEW=10`) | `pipeline_run/output/submission_3/` |

**Blocking candidate set** (`candidate_pairs.tsv`): the stage-1 union (up to 120 per S1) plus the adapter's fused top-10 pairs that were not already in it. **Matches** (`matching_results.tsv`): candidates scoring at least 0.66 under the exclusive rule, where each pool record goes only to its best S1.

To run the notebook interactively instead of through `run_notebook.py`, open `er_pipeline.ipynb`, set `ER_ROOT`, and run all cells. Its behaviour can be changed with these environment variables:

- `ER_PROFILE=full|smoke`: `smoke` runs a small end-to-end check.
- `ER_STAGES=all` or a comma-separated list of stages.
- `ER_SET="KEY=json;..."`: overrides config values.

## 4. Source map

```
src/
├── embeddings/
│   ├── qwen_embed.py                 # Qwen3-Embedding: load, last-token pooling, length-bucketed encode, int8, cosine top-k
│   └── build_embedding_store.py      # writes the chunked store the notebook reads
├── stage1_retrieval_rerank/
│   ├── er_pipeline.ipynb             # stage-1 pipeline (60 cells)
│   ├── cells/ + assemble_notebook.py # notebook sources, one file per cell; assemble_notebook.py rebuilds the notebook.
│   │                                 # Later scripts exec the metric code from cells/022.py.
│   ├── run_notebook.py               # runs notebook cells headless in a persistent kernel
│   └── rules_phase2.py               # decision-rule study on saved predictions (analysis only)
├── stage2_cross_encoder/
│   ├── ce_pilot.py                   # go/no-go pilot
│   └── ce_full.py                    # full cross-encoder, second stage, submission_2
├── stage3_adapter_clustering/
│   ├── adapter_pilot.py              # adapter training and recall check
│   ├── adapter_fast.py               # adapter candidates, clustering, second stage v3 (REP2 measurement)
│   ├── finish_test_lean.py           # memory-lean test path, submission_3
│   ├── export_adapter_candidates.py  # exports the adapter top-20 candidates (analysis)
│   ├── test_fused20.py               # fused top-20 search on test (K=20 experiment)
│   └── ce2_train.py                  # cross-encoder epoch 2 (experiment, not in submission_3)
├── scripts/
│   ├── run_all.sh                    # end-to-end reproduction
│   └── launch_*.sh, queue_*.sh       # launchers used for the K=20 and cross-encoder epoch 2 experiments
└── analysis/                         # APPROACH.md (full notes), experiments.tsv (every measured run), decision files
```

## 5. Notes

- **Do not use `DataParallel` or `DDP`.** GPU peer-to-peer is broken on the development machine, so all multi-GPU work runs as independent workers per GPU and is merged on the host. Move models between GPUs with `.cpu().to(dev)`.
- Memory limits (28 GB RAM per job, 22 GB per GPU) are enforced by chunking. Run one job at a time.
- The leakage-safe evaluation protocol and all measured scores are in `src/analysis/APPROACH.md` and `src/analysis/experiments.tsv`.
