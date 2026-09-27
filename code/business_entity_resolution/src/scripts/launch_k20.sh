#!/bin/bash
# K=20: train-side measurement (REP2) and the test fused search, in parallel
cd "$(dirname "$0")/../stage3_adapter_clustering"; ER_ROOT="${ER_ROOT:-/home/parth/Desktop/trial work}"; export ER_ROOT
PY="${PY:-python}"
rm -f "$ER_ROOT/pipeline_run/experiments"/adapter_fast/ce_train_sample_chunks/*.tmp.npy
K_NEW=20 SKIP_TEST=1 nohup $PY -u adapter_fast.py > k20_train.log 2>&1 &
nohup $PY -u test_fused20.py > k20_testfused.log 2>&1 &
echo "launched at $(date +%H:%M:%S)"
