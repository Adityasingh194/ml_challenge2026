#!/bin/bash
# start the K=20 train side only after the test fused search has released the GPUs
cd "$(dirname "$0")/../stage3_adapter_clustering"; ER_ROOT="${ER_ROOT:-/home/parth/Desktop/trial work}"; export ER_ROOT
until [ -f "$ER_ROOT/pipeline_run/experiments"/adapter_fast/fused_test_top20.npz ] && ! ps -eo args | grep -q "[t]est_fused20.py"; do sleep 5; done
echo "$(date '+%Y-%m-%d %H:%M:%S') [queue] test fused top-20 done; starting K=20 train side" >> "$ER_ROOT/pipeline_run/experiments/experiments.log"
rm -f "$ER_ROOT/pipeline_run/experiments"/adapter_fast/ce_train_sample_chunks/*.tmp.npy
K_NEW=20 SKIP_TEST=1 ${PY:-python} -u adapter_fast.py > k20_train.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') [queue] K=20 train side exited ($?)" >> "$ER_ROOT/pipeline_run/experiments/experiments.log"
