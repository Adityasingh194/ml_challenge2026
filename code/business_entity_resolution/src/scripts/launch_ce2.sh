#!/bin/bash
# CE epoch 2: train -> REP2 measurement at K=20 -> REP2 at K=10 (reuses the K<=10 CE2 chunks). Test NOT run here.
cd "$(dirname "$0")/../stage3_adapter_clustering"; ER_ROOT="${ER_ROOT:-/home/parth/Desktop/trial work}"; export ER_ROOT
PY="${PY:-python}"; M="$ER_ROOT/pipeline_run/experiments/ce2/model"
echo "$(date '+%Y-%m-%d %H:%M:%S') [queue] CE2 start" >> "$ER_ROOT/pipeline_run/experiments/experiments.log"
$PY -u ce2_train.py > ce2_train.log 2>&1 || { echo "$(date '+%Y-%m-%d %H:%M:%S') [queue] ce2_train FAILED" >> "$ER_ROOT/pipeline_run/experiments/experiments.log"; exit 1; }
CE_MDIR="$M" CE_TAG=_ce2 K_NEW=20 SKIP_TEST=1 $PY -u adapter_fast.py > ce2_k20.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') [queue] CE2 K=20 REP2 exited ($?)" >> "$ER_ROOT/pipeline_run/experiments/experiments.log"
CE_MDIR="$M" CE_TAG=_ce2 K_NEW=10 SKIP_TEST=1 $PY -u adapter_fast.py > ce2_k10.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') [queue] CE2 K=10 REP2 exited ($?)" >> "$ER_ROOT/pipeline_run/experiments/experiments.log"
