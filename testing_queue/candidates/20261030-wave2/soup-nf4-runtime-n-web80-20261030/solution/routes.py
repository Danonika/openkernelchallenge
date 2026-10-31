"""Two narrow decode boundaries; native dispatch, GEMM and autograd stay intact."""
from functools import wraps
import inspect
import json
import os
import torch
from .packed_nf4 import decode, raw_supported, state_supported

_MARKER = "_soup_nf4_runtime_n_v2"
_COUNTS = {"forward": 0, "functional": 0}


def _activated(route, shape, dtype):
    _COUNTS[route] += 1
    n = _COUNTS[route]
    # Metadata-only geometric receipts bypass Rich's stdout proxy; this does not
    # alter the trainer display or native logs. CUDA event timing remains the
    # only own-kernel time evidence. Leading newline makes one parseable record.
    if n == 1 or n & (n - 1) == 0:
        record={"candidate":"nf4_runtime_n_v2","route":route,"calls":n,
                "shape":list(shape),"dtype":str(dtype),
                "kernel":"soup_nf4_blockpair_runtime_n"}
        data=("\nSOUP_NF4_ROUTE "+json.dumps(record,sort_keys=True,separators=(",",":"))+"\n").encode()
        while data:
            count=os.write(1,data)
            if count<=0:raise OSError("NF4 metadata receipt write made no progress")
            data=data[count:]



def forward_supported(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                      absmax_8bit=None, absmax_code=None, absmax_offset=None):
    if (type(A) is not torch.Tensor or not A.is_cuda or A.layout != torch.strided
            or A.dtype != torch.float16 or A.ndim not in (2, 3)
            or not A.is_contiguous() or any(d <= 0 for d in A.shape)
            or type(blocksize) is not int or blocksize != 64 or quant_type != "nf4"):
        return False
    if not raw_supported(B, shapeB, absmax_8bit, absmax_code, absmax, absmax_offset, A.dtype):
        return False
    if B.device != A.device or A.shape[-1] != shapeB[1]:
        return False
    # Bias participates in unchanged F.linear. Native MatMul4Bit owns bias grad.
    if bias is not None and (type(bias) is not torch.Tensor or bias.device != A.device
                            or bias.dtype != A.dtype or bias.shape != (shapeB[0],)
                            or bias.layout != torch.strided or not bias.is_contiguous()):
        return False
    return True


def forward_wrapper(original):
    @wraps(original)
    def wrapped(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                absmax_8bit=None, absmax_code=None, absmax_offset=None):
        if not forward_supported(A, B, shapeB, absmax, blocksize, quant_type, bias,
                                 absmax_8bit, absmax_code, absmax_offset):
            return original(A, B, shapeB, absmax, blocksize, quant_type, bias,
                            absmax_8bit=absmax_8bit, absmax_code=absmax_code,
                            absmax_offset=absmax_offset)
        weight = decode(B, shapeB, absmax_8bit, absmax_code, absmax, absmax_offset, A.dtype)
        _activated("forward", shapeB, A.dtype)
        return torch.nn.functional.linear(A, weight, bias)
    setattr(wrapped, _MARKER, True)
    return wrapped


def functional_wrapper(original):
    @wraps(original)
    def wrapped(A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4"):
        if out is not None or not state_supported(A, quant_state):
            return original(A, quant_state=quant_state, absmax=absmax, out=out,
                            blocksize=blocksize, quant_type=quant_type)
        q = quant_state
        result = decode(A, q.shape, q.absmax, q.state2.code, q.state2.absmax, q.offset, q.dtype)
        _activated("functional", q.shape, q.dtype)
        return result
    setattr(wrapped, _MARKER, True)
    return wrapped


def install():
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    if bnb.__version__ != "0.50.2":
        return {"patched": False, "reason": "requires_bitsandbytes_0.50.2"}
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return {"patched": False, "reason": "requires_sm75_cuda"}
    import bitsandbytes.backends.cuda.ops as cuda_ops
    fwd, fun = cuda_ops._dequant_linear_fallback, functional.dequantize_4bit
    if getattr(fwd, _MARKER, False) and getattr(fun, _MARKER, False):
        return {"patched": True, "already_patched": True}
    expected = ((fwd, "bitsandbytes.backends.cuda.ops", "_dequant_linear_fallback",
                 ("A", "B", "shapeB", "absmax", "blocksize", "quant_type", "bias", "absmax_8bit", "absmax_code", "absmax_offset")),
                (fun, "bitsandbytes.functional", "dequantize_4bit",
                 ("A", "quant_state", "absmax", "out", "blocksize", "quant_type")))
    for fn, module, name, params in expected:
        if (getattr(fn, "__module__", None) != module or getattr(fn, "__name__", None) != name
                or hasattr(fn, "__wrapped__") or tuple(inspect.signature(fn).parameters) != params):
            return {"patched": False, "reason": "unreviewed_decode_boundary"}
    # Check both before changing either. Do not replace MatMul4Bit or tiny-M routing.
    cuda_ops._dequant_linear_fallback = forward_wrapper(fwd)
    functional.dequantize_4bit = functional_wrapper(fun)
    return {"patched": True, "already_patched": False, "routes": ["forward", "functional"]}
