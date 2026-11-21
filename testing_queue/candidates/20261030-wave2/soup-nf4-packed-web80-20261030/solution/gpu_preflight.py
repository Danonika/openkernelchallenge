"""Future authorized exact T4 preflight. Never imported by solution.apply()."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F
import bitsandbytes.backends.cuda.ops as ops
from bitsandbytes.autograd._functions import MatMul4Bit
from solution import routes
from solution.packed_nf4 import decode, state_supported


@contextmanager
def boundaries(forward, functional):
    old = ops._dequant_linear_fallback, F.dequantize_4bit
    ops._dequant_linear_fallback, F.dequantize_4bit = forward, functional
    try:
        yield
    finally:
        ops._dequant_linear_fallback, F.dequantize_4bit = old


def exact(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    assert actual.stride() == expected.stride(), label + ": stride"
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected)), label + ": NaN"
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected)), label + ": +Inf"
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected)), label + ": -Inf"
    bits = torch.int16 if actual.dtype == torch.float16 else torch.int32
    assert torch.equal(actual[finite].view(bits), expected[finite].view(bits)), label + ": bits"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assert bnb.__version__ == "0.50.2"
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "Tesla T4"
    assert torch.cuda.get_device_capability(0) == (7, 5)
    torch.manual_seed(20261030)
    native_fwd, native_fun, native_back = ops._dequant_linear_fallback, F.dequantize_4bit, MatMul4Bit.backward
    own_fwd, own_fun = routes.forward_wrapper(native_fwd), routes.functional_wrapper(native_fun)
    results = []
    shapes = {(1536, 1536): (4, 8, 65, 356), (256, 1536): (4, 356),
              (8960, 1536): (128, 129, 356), (1536, 8960): (64, 65, 356)}
    for shape, ms in shapes.items():
        for dtype in (torch.float16, torch.float32):
            dense = torch.randn(shape, device="cuda", dtype=dtype) * .03
            packed, state = F.quantize_4bit(dense, compress_statistics=True, quant_type="nf4", blocksize=64)
            assert state_supported(packed, state)
            before = [t.clone() for t in (packed, state.absmax, state.state2.code, state.state2.absmax, state.offset)]
            actual = own_fun(packed, state)
            reference = native_fun(packed, state)
            exact(actual, reference, "decode")
            del actual, reference, dense
            for m in ms:
                x = torch.randn((m, shape[1]), device="cuda", dtype=torch.float16)
                cotangent = torch.randn((m, shape[0]), device="cuda", dtype=torch.float16) * 8
                bias = torch.randn(shape[0], device="cuda", dtype=torch.float16) * .02
                observations = []
                # Native operator decides fused-vs-fallback independently each time.
                for fwd, fun in ((native_fwd, native_fun), (own_fwd, own_fun)):
                    a = x.clone().requires_grad_(True)
                    b = bias.clone().requires_grad_(True)
                    with boundaries(fwd, fun):
                        y = bnb.matmul_4bit(a, packed.t(), quant_state=state, bias=b)
                        grads = torch.autograd.grad(y, (a, b), cotangent)
                    observations.append((y.detach(), grads[0], grads[1]))
                for i, label in enumerate(("output", "input_gradient", "bias_gradient")):
                    exact(observations[1][i], observations[0][i], label)
                results.append({"shape": list(shape), "state_dtype": str(dtype), "M": m,
                                "decode_output_input_gradient_bias_gradient_exact": True})
                del observations, x, cotangent, bias, y, grads, a, b
            for original, saved in zip((packed, state.absmax, state.state2.code, state.state2.absmax, state.offset), before):
                assert torch.equal(original.reshape(-1).view(torch.uint8), saved.reshape(-1).view(torch.uint8))
            del packed, state, before
    assert MatMul4Bit.backward is native_back
    assert routes._COUNTS["forward"] > 0 and routes._COUNTS["functional"] > 0
    result = {"status": "GPU_PREFLIGHT_PASSED", "cases": results, "native_backward_identity_preserved": True,
              "routes": dict(routes._COUNTS), "official_admission": "UNKNOWN", "score": None,
              "limits": "Fixture parity only: no full-task speed/share/quality result"}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
