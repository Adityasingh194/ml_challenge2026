"""Test fused top-20 search only (runs in parallel with the K=20 train side); cached for finish_test_lean.py."""
import sys, glob
import numpy as np, pyarrow.parquet as pq
sys.argv = [sys.argv[0]]
import adapter_fast as AF
from adapter_fast import emb, W

fr = glob.glob(f"{W}/scratch/frames/*")[0]
cq = pq.read_table(f"{fr}/test_s1_norm.parquet", columns=["country_norm"]).column(0).to_numpy(zero_copy_only=False)
cp = pq.read_table(f"{fr}/test_pool_norm.parquet", columns=["country_norm"]).column(0).to_numpy(zero_copy_only=False)
X1 = emb("test", 1)
XP = np.concatenate([emb("test", 2), emb("test", 3)], axis=0)
AF.fused_search(X1, XP, np.arange(len(X1)), cq, cp, 20, "test")
AF.log("test fused top-20 search cached", "fused")
