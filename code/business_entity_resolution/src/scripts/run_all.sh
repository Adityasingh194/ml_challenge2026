#!/bin/bash
# End-to-end reproduction: data -> embeddings -> blocking + stage-1 reranker -> cross-encoder -> adapter + clustering
# -> output/matching_results.tsv + output/candidate_pairs.tsv (the final submission, = submission_3).
#
#   ER_ROOT=/path/to/workdir PY=/path/to/venv/bin/python bash run_all.sh
#
# ER_ROOT must contain data/{train,test}/*.tsv. Everything else (embedding_store/, pipeline_run/) is created under it.
# Every stage is checkpointed and resumes on rerun. Use a fresh ER_ROOT so no earlier submission folder is overwritten.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
export ER_ROOT="${ER_ROOT:-/home/parth/Desktop/trial work}"
PY="${PY:-python}"
SUBDIR="${SUBDIR:-submission_3}"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [run_all] $*"; }
[ -d "$ER_ROOT/data/train" ] || { echo "no data at $ER_ROOT/data"; exit 1; }
mkdir -p "$ER_ROOT/pipeline_run/experiments/logs"

log "1/6 Qwen3-Embedding-0.6B store -> $ER_ROOT/embedding_store"
$PY "$SRC/embeddings/build_embedding_store.py"

log "2/6 stage 1 notebook (normalise, blocking/retrieval, 74 features, XGBoost, rule) -> pipeline_run/output/submission_1"
CONN="$ER_ROOT/pipeline_run/kernel.json"
ER_PROFILE=full ER_STAGES=all $PY "$SRC/stage1_retrieval_rerank/run_notebook.py" start "$CONN"
trap 'pkill -f "ipykernel_launcher -f $CONN" || true' EXIT
$PY "$SRC/stage1_retrieval_rerank/run_notebook.py" run "$CONN" "$SRC/stage1_retrieval_rerank/er_pipeline.ipynb" $(seq 0 59)

cd "$SRC/stage2_cross_encoder"
log "3/6 cross-encoder (XLM-R base) + 2nd stage -> pipeline_run/output/submission_2"
$PY -u ce_full.py

cd "$SRC/stage3_adapter_clustering"
log "4/6 adapter (two-tower over Qwen vectors)"
$PY -u adapter_pilot.py
log "5/6 adapter candidates + clustering features + 2nd stage v3, measured on REP2"
K_NEW=10 SKIP_TEST=1 $PY -u adapter_fast.py
log "6/6 test path -> pipeline_run/output/$SUBDIR"
K_NEW=10 SUBDIR="$SUBDIR" $PY -u finish_test_lean.py

OUT="$SRC/../../../output"
mkdir -p "$OUT"
cp "$ER_ROOT/pipeline_run/output/$SUBDIR/matching_results.tsv" "$ER_ROOT/pipeline_run/output/$SUBDIR/candidate_pairs.tsv" "$OUT/"
log "done: $(cd "$OUT" && pwd)/{matching_results.tsv,candidate_pairs.tsv}"
