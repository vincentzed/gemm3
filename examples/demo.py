"""Run one NVFP4 GEMM with its tuned preset and check it against an FP32 reference.

python examples/demo.py 4096 4096 4096
"""

import sys

import torch

from nvfp4_gemm_cute import config_for, gemm, persisting_l2, supported
from nvfp4_gemm_cute.testing import check, make_inputs, reference

m, n, k = (int(x) for x in (sys.argv[1:4] if len(sys.argv) > 3 else (4096, 4096, 4096)))
assert supported(), "needs an sm_103 GPU (B300/GB300) and nvidia-cutlass-dsl>=4.8"
inp = make_inputs(m, n, k)
cfg = config_for(m, n, k)
with persisting_l2(cfg.persist_mb):
    out = gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], config=cfg)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], out=out, config=cfg)
    end.record()
    torch.cuda.synchronize()
us = start.elapsed_time(end) / 20 * 1e3
ok, err = check(out, reference(inp))
print(f"{m}x{n}x{k}: {us:.1f} us/iter (warm, events)  {2 * m * n * k / us / 1e6:.0f} TFLOPS")
print(f"config: cluster={cfg.cluster} fallback={cfg.fallback} attrs={cfg.attr_dict()}")
print(f"check vs FP32 reference: {'PASS' if ok else 'FAIL'} (rel L2 {err:.2e})")
