# 0a. Install / verify the packages in THIS kernel's Python environment (the venv kernel)
import os
import subprocess
import sys
from importlib import metadata

LOCAL_ROOT = os.environ.get("ER_ROOT", "/home/parth/Desktop/trial work")   # data, embeddings and all outputs
print("python:", sys.executable)


def _pip(*args):
    print("pip install", " ".join(args))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


# Blackwell (RTX PRO 4000, compute capability 12.0 = sm_120) needs a PyTorch build for CUDA >= 12.8.
# The installed torch is kept when it already has sm_120 kernels; otherwise the CUDA 12.8 build is installed.
_probe = subprocess.run([sys.executable, "-c", "import torch; print('sm_120' in torch.cuda.get_arch_list())"],
                        capture_output=True, text=True)
if _probe.returncode != 0 or _probe.stdout.strip() != "True":
    _pip("--upgrade", "torch", "--index-url", "https://download.pytorch.org/whl/cu128")
    print("torch (re)installed -> restart the kernel before continuing")

_pip("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost>=2.0", "rapidfuzz>=3.0", "joblib", "psutil")
# faiss is optional and not installed: dense search uses exact fp16 torch matmul on the GPUs instead.

for _d in ["torch", "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "rapidfuzz", "joblib", "psutil"]:
    print(f"  {_d:14s} {metadata.version(_d)}")
