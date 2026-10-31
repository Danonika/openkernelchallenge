"""Compile to explicit SM75 target without loading/executing GPU code.

Local Triton is NOT the pinned target compiler; no PTX/cubin is shipped.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("TRITON_INTERPRET", None)
import tempfile
from pathlib import Path
import sys
import json
import re
import importlib.metadata
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from solution.packed_nf4 import soup_nf4_blockpair_decode

rows = []
with tempfile.TemporaryDirectory(prefix="nf4-compile-only-") as cache:
    os.environ["TRITON_CACHE_DIR"] = cache
    for dtype, packed_half, n in (("fp16", True, 1536*8960), ("fp32", False, 1536*8960),
                                   ("fp16", False, 65), ("fp16", True, 65)):
        source = ASTSource(soup_nf4_blockpair_decode,
            signature={"PACKED": "*u8", "QABS": "*u8", "CODE2": "*fp32", "ABS2": "*fp32", "OFFSET": "*fp32", "OUT": "*"+dtype},
            constexprs={"N": n, "SCALES": 16, "PACK_HALF": packed_half})
        compiled = triton.compile(source, target=GPUTarget("cuda", 75, 32),
                                  options={"num_warps": 4, "enable_fp_fusion": False})
        ptx = compiled.asm["ptx"]
        # Metadata/property access here does not call _init_handles or launch.
        rows.append({"dtype": dtype, "packed_half": packed_half, "n": n,
                     "shared_bytes": compiled.metadata.shared,
                     "separate_scale_f32_mul_add": bool(re.search(r"\bmul(?:\.[a-z]+)*\.f32\b", ptx)) and bool(re.search(r"\badd(?:\.[a-z]+)*\.f32\b", ptx)),
                     "contains_fma": bool(re.search(r"\bfma(?:\.[a-z]+)*\.f32\b", ptx)),
                     "ptx_store_b32_present": "st.global.b32" in ptx,
                     "compiled": True})
report = {"status": "OFFLINE_COMPILE_ONLY", "gpu_executed": False,
          "actual_triton": importlib.metadata.version("triton"), "target_triton": "3.6.0",
          "target": "cuda_sm75", "cases": rows, "binaries_shipped": False,
          "limitations": "Triton3.9 offline compilation is not pinned3.6 compilation or T4 execution"}
Path(__file__).with_name("offline_compile_results.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
