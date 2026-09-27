# 0b. GPU check: expects 2 x 24 GB NVIDIA GPUs
import subprocess

import torch

EXPECTED_GPUS = 2
print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version",
                      "--format=csv"], capture_output=True, text=True).stdout)
print(f"torch {torch.__version__}  CUDA build {torch.version.cuda}  compiled archs {torch.cuda.get_arch_list()}")
assert torch.cuda.is_available(), "torch sees no GPU: check the NVIDIA driver and that torch is a CUDA build"
n_gpu = torch.cuda.device_count()
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    arch = f"sm_{p.major}{p.minor}"
    assert arch in torch.cuda.get_arch_list(), f"this torch build has no kernels for {arch} ({p.name}): rerun 0a"
    x = torch.randn(2048, 2048, device=f"cuda:{i}", dtype=torch.float16)
    ok = bool(torch.isfinite((x @ x).float()).all())
    free, total = torch.cuda.mem_get_info(i)
    print(f"cuda:{i}  {p.name}  cc {p.major}.{p.minor}  {total / 1e9:.1f} GB (free {free / 1e9:.1f})  "
          f"fp16 matmul {'OK' if ok else 'FAILED'}")
    assert ok, f"fp16 matmul failed on cuda:{i}"
    del x
torch.cuda.empty_cache()
if n_gpu >= 2:   # direct GPU->GPU copies: broken peer-to-peer (e.g. IOMMU on) silently returns zeros
    _x = torch.arange(1, 9, dtype=torch.float32, device="cuda:1")
    _p2p_ok = bool((_x.to("cuda:0").cpu() == _x.cpu()).all())
    print(f"direct GPU1->GPU0 copy: {'OK' if _p2p_ok else 'BROKEN (returns wrong data)'} -- the notebook always copies "
          "between GPUs through host memory, so results are correct either way")
    del _x
if n_gpu != EXPECTED_GPUS:
    print(f"WARNING: expected {EXPECTED_GPUS} GPUs but torch sees {n_gpu} (check CUDA_VISIBLE_DEVICES)")
else:
    print(f"OK: {n_gpu} GPUs -- dense search and BM25 are split across all of them")
