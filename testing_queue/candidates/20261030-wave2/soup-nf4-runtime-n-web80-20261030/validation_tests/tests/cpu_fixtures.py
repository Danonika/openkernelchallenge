"""No GPU/model execution. Actual Triton interpreter + independent NumPy oracle.

Interpreter version may differ from target. These are not CUDA compilation,
SM75 numerics, full Soup gradients, useful-share or performance evidence.
"""
import os
os.environ["TRITON_INTERPRET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from solution.packed_nf4 import launch_into, state_supported, raw_supported
from solution.routes import forward_wrapper, functional_wrapper, forward_supported

TABLE = np.asarray([-1., -.6961928009986877, -.5250730514526367,
    -.39491748809814453, -.28444138169288635, -.18477343022823334,
    -.09105003625154495, 0., .07958029955625534, .16093020141124725,
    .24611230194568634, .33791524171829224, .44070982933044434,
    .5626170039176941, .7229568362236023, 1.], dtype=np.float32)


def oracle(packed, qabs, code, abs2, offset, n, dtype):
    # Intentionally per-weight flat oracle, unlike implementation's block-pair grid.
    pos = np.arange(n, dtype=np.int64)
    byte = packed.numpy().reshape(-1)[pos // 2]
    nibble = np.where(pos % 2 == 0, byte >> 4, byte & 15)
    block = pos // 64
    with np.errstate(all="ignore"):
        centered = np.multiply(code.numpy()[qabs.numpy()[block]], abs2.numpy()[block // 256], dtype=np.float32)
        scale = np.add(centered, np.float32(offset.item()), dtype=np.float32)
        value = np.multiply(TABLE[nibble], scale, dtype=np.float32)
        return value.astype(np.float16 if dtype == torch.float16 else np.float32)


def check_values(actual, expected):
    a = actual.detach().numpy().reshape(-1)
    assert np.array_equal(np.isnan(a), np.isnan(expected))
    assert np.array_equal(np.isposinf(a), np.isposinf(expected))
    assert np.array_equal(np.isneginf(a), np.isneginf(expected))
    finite = np.isfinite(expected)
    item = np.uint16 if a.dtype == np.float16 else np.uint32
    assert np.array_equal(a[finite].view(item), expected[finite].view(item)), "finite bit mismatch"


def main():
    records = []
    # Includes all 256 byte values, tail, 64-weight boundaries, secondary256 boundary.
    for n in (1, 2, 63, 64, 65, 195, 512, 16383, 16384, 16385):
        for dtype in (torch.float16, torch.float32):
            for start in (0, 1):
                count = (n + 1) // 2
                packed = (torch.arange(count, dtype=torch.int64) % 256).to(torch.uint8).reshape(-1, 1)
                outer = (n + 63) // 64
                qabs = (torch.arange(outer, dtype=torch.int64) % 256).to(torch.uint8)
                code = torch.linspace(-.98123, 1.01234, 256, dtype=torch.float32)
                abs2 = torch.linspace(.314159, .719231, (outer + 255) // 256, dtype=torch.float32)
                offset = torch.tensor(.5112345, dtype=torch.float32)
                storage = torch.full((n + 3,), -11., dtype=dtype)
                out = storage[start:start + n]
                launch_into(packed, qabs, code, abs2, offset, out)
                expected = oracle(packed, qabs, code, abs2, offset, n, dtype)
                check_values(out, expected)
                assert torch.all(storage[:start] == -11.)
                assert torch.all(storage[start + n:] == -11.)
                records.append({"n": n, "dtype": str(dtype), "start": start, "passed": True})
    for scale in (float("inf"), -float("inf"), float("nan"), -0.):
        for dtype in (torch.float16, torch.float32):
            packed = torch.arange(256, dtype=torch.uint8).reshape(-1, 1)
            qabs = torch.zeros(8, dtype=torch.uint8)
            code = torch.ones(256, dtype=torch.float32)
            abs2 = torch.tensor([scale], dtype=torch.float32)
            offset = torch.tensor(0., dtype=torch.float32)
            out = torch.empty(512, dtype=dtype)
            launch_into(packed, qabs, code, abs2, offset, out)
            check_values(out, oracle(packed, qabs, code, abs2, offset, 512, dtype))
            records.append({"nonfinite_or_zero_scale": str(scale), "dtype": str(dtype), "passed": True})
    # Direct wrappers must forward unsupported arguments exactly once.
    calls = []
    token = object()
    def original(*args, **kwargs):
        calls.append((args, kwargs)); return token
    f = functional_wrapper(original)
    A = torch.zeros((5, 1), dtype=torch.uint8)
    supplied_out = torch.empty(9)
    assert f(A, None, out=supplied_out, blocksize=128, quant_type="fp4") is token
    assert len(calls) == 1 and calls[-1][1]["out"] is supplied_out
    assert calls[-1][1]["blocksize"] == 128
    f = forward_wrapper(original)
    x = torch.zeros(3, 2)
    assert f(x, A, (5, 2), None, 64, "nf4") is token
    assert len(calls) == 2 and calls[-1][0][0] is x
    assert not state_supported(A, None)
    assert not raw_supported(A, (5, 2), None, None, None, None, torch.float16)
    assert not forward_supported(x, A, (5, 2), None, 64, "nf4")
    # No GPU API queried here; CUDA_VISIBLE_DEVICES intentionally empty.
    report = {"status": "CPU_INTERPRETER_FIXTURES_PASSED", "gpu_tested": False,
              "target_triton": "3.6.0", "actual_triton": importlib.metadata.version("triton"),
              "actual_torch": torch.__version__, "interpreter_cases": records,
              "cases": len(records), "fallback_contract_checks": 5,
              "finite_bits_and_nonfinite_classes": True, "nan_payload_identity_claimed": False,
              "limitations": ["No CUDA compilation or execution", "No actual T4 speed/share/peak/quality",
                              "No full Soup/LoRA integration", "No final admission or score"]}
    Path(__file__).with_name("cpu_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "interpreter_cases"}, indent=2))

if __name__ == "__main__":
    main()
