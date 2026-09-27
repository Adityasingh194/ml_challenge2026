import csv
import gc
import glob
import functools
import hashlib
import importlib
import itertools
import json
import math
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import warnings
from collections import defaultdict
from importlib import metadata as _md
from types import SimpleNamespace

warnings.filterwarnings("ignore")
cfg = SimpleNamespace(**CFG)
T0 = time.time()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def _vt(v):
    return tuple(int(x) for x in re.findall(r"\d+", str(v))[:3])


def _dist_version(dist):
    try:
        return _md.version(dist)
    except _md.PackageNotFoundError:
        return None


_REQUIRED = [("numpy", None), ("pandas", None), ("pyarrow", None), ("scipy", None), ("scikit-learn", None),
             ("xgboost", "2.0.0"), ("rapidfuzz", "3.0.0"), ("joblib", None), ("torch", None)]
_missing = [(d, m, _dist_version(d)) for d, m in _REQUIRED
            if _dist_version(d) is None or (m and _vt(_dist_version(d)) < _vt(m))]
if _missing:
    raise RuntimeError("Missing/outdated packages: "
                       + ", ".join(f"{d} (need >={m}, have {v})" for d, m, v in _missing)
                       + "\nFix: run cell 0a (it installs them into this kernel's environment), then restart the kernel.")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
import joblib
import torch
from rapidfuzz import fuzz as rf_fuzz

try:  # pandas>=3 infers Arrow-backed "str" columns; the copied feature code expects object arrays
    pd.set_option("future.infer_string", False)
except Exception:
    pass

TORCH_AVAILABLE = True
N_GPU_AVAIL = torch.cuda.device_count()
N_GPUS = N_GPU_AVAIL if cfg.NUM_GPUS == "auto" else min(int(cfg.NUM_GPUS), N_GPU_AVAIL)
DEVICE = "cuda" if N_GPUS > 0 else "cpu"
DEVICES = [f"cuda:{i}" for i in range(N_GPUS)] or ["cpu"]   # dense search / BM25 are split across these
GPU_INFO = []
for _i in range(N_GPUS):
    _p = torch.cuda.get_device_properties(_i)
    GPU_INFO.append(dict(idx=_i, name=_p.name, cc=f"{_p.major}.{_p.minor}", mem_gb=round(_p.total_memory / 1e9, 1)))

try:   # optional; only used for CPU-only runs. On GPUs, exact fp16 torch search runs on every device.
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

DATA_DIR = cfg.DATA_DIR
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"DATA_DIR {DATA_DIR} not found (set DATA_DIR / LOCAL_ROOT in the config cell)")
WORK_DIR = cfg.WORK_DIR
CACHE_DIR = cfg.CACHE_DIR or os.path.join(WORK_DIR, "cache")
SCRATCH_DIR = cfg.SCRATCH_DIR or os.path.join(WORK_DIR, "scratch")
OUTPUT_DIR = cfg.OUTPUT_DIR or os.path.join(WORK_DIR, "output")
CACHE_SEARCH_PATHS = []   # extra read-only cache roots (none locally: everything is in CACHE_DIR)
for _d in (CACHE_DIR, SCRATCH_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)


def _ram_gb():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 1), round(vm.available / 1e9, 1)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            info = {l.split(":")[0]: int(l.split()[1]) for l in f}
        return round(info["MemTotal"] / 1e6, 1), round(info.get("MemAvailable", 0) / 1e6, 1)
    except OSError:
        return None, None


def free_gb(path):
    return round(shutil.disk_usage(path).free / 1e9, 2)


print(f"profile={PROFILE}  stages={STAGES_TO_RUN}  python={platform.python_version()}  platform={platform.platform()}")
print(f"torch={torch.__version__} (CUDA {torch.version.cuda})  xgboost={_dist_version('xgboost')}  "
      f"rapidfuzz={_dist_version('rapidfuzz')}  faiss={FAISS_AVAILABLE}")
print(f"GPUs used={N_GPUS} {GPU_INFO}  devices={DEVICES}")
print(f"RAM (total, available) GB={_ram_gb()}  CPUs={os.cpu_count()}  NUM_WORKERS={cfg.NUM_WORKERS}")
print(f"DATA_DIR={DATA_DIR}\nEMB_DIR={cfg.EMB_DIR}\nCACHE_DIR={CACHE_DIR} (free {free_gb(CACHE_DIR)} GB)\n"
      f"SCRATCH_DIR={SCRATCH_DIR}\nOUTPUT_DIR={OUTPUT_DIR}")
