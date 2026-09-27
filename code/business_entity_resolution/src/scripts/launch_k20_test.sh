#!/bin/bash
# K=20 test path -> output/submission_4 (submission_1/2/3 untouched). Reuses: fused top-20 (cached), CE scores of the
# K=10 test run (first 213 full chunks), so only the new rank-11..20 pairs are scored.
cd "$(dirname "$0")/../stage3_adapter_clustering"; ER_ROOT="${ER_ROOT:-/home/parth/Desktop/trial work}"; export ER_ROOT
rm -f "$ER_ROOT/pipeline_run/experiments"/adapter_fast/ce_test_chunks/*.tmp.npy "$ER_ROOT/pipeline_run/experiments"/adapter_fast/ce_test.npy
K_NEW=20 SUBDIR=submission_4 nohup ${PY:-python} -u finish_test_lean.py > k20_test.log 2>&1 &
echo "K=20 test launched at $(date +%H:%M:%S)"
