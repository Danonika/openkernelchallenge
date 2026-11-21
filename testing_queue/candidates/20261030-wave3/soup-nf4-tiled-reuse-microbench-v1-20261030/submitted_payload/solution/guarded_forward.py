"""Research-only fallback replacement; native dispatcher/autograd stay owned.

No persistent installation and no custom backward. Direct grad-enabled use is
ineligible: only native MatMul4Bit.forward's no-grad context may route trainable
inputs here. This helper is not an admission or full-task candidate.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math
import torch
import triton
from .kernel import soup_nf4_mma_tiled_reuse
from .parameter_bias import normalized_native_bias

COUNTS = {"custom": 0, "native": 0}
COMPILED = {}
FALLBACK_REASONS = {}


def reason(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
           absmax_8bit=None, absmax_code=None, absmax_offset=None):
    if type(blocksize) is not int or blocksize != 64 or quant_type != "nf4":
        return "format"
    if type(shapeB) not in (list, tuple, torch.Size) or len(shapeB) != 2:
        return "shape_metadata"
    if any(type(x) is not int for x in shapeB) or tuple(shapeB) not in ((1536,1536),(256,1536)):
        return "shape_scope"
    n,k = shapeB
    operands = (A,B,absmax,absmax_8bit,absmax_code,absmax_offset)
    if any(type(x) is not torch.Tensor for x in operands) or (bias is not None and type(bias) is not torch.Tensor):
        return "tensor_type"
    if A.ndim not in (2,3) or A.shape[-1] != k or (A.ndim == 3 and A.shape[0] != 4):
        return "activation_geometry"
    m = math.prod(A.shape[:-1])
    if not 65 <= m <= 2048:
        return "row_scope"
    all_tensors = operands + (() if bias is None else (bias,))
    if any(x.is_neg() or x.is_conj() or not x.is_contiguous() for x in all_tensors):
        return "lazy_or_strided"
    if any(x.device != A.device for x in all_tensors) or A.device.type != "cuda":
        return "device"
    if A.dtype != torch.float16 or B.dtype != torch.uint8 or B.numel() != n*k//2 or B.ndim not in (1,2):
        return "activation_or_packed_dtype_geometry"
    if A.data_ptr()%4:
        return "activation_half2_alignment"
    blocks=n*k//64
    if (absmax_8bit.dtype != torch.uint8 or absmax_8bit.ndim != 1 or absmax_8bit.numel() != blocks
            or absmax.dtype != torch.float32 or absmax.ndim != 1 or absmax.numel() != triton.cdiv(blocks,256)
            or absmax_code.dtype != torch.float32 or absmax_code.ndim != 1 or absmax_code.numel() != 256
            or absmax_offset.dtype != torch.float32 or absmax_offset.numel() != 1):
        return "nested_scale_geometry"
    if any(x.requires_grad for x in operands[1:]):
        return "differentiable_metadata"
    if bias is not None and (bias.dtype != torch.float16 or bias.shape != (n,)):
        return "bias_geometry"
    if torch.is_grad_enabled() and (A.requires_grad or (bias is not None and bias.requires_grad)):
        return "direct_autograd_use"
    prop=torch.cuda.get_device_properties(A.device)
    if (prop.major,prop.minor,prop.multi_processor_count) != (7,5,40):
        return "architecture"
    return None


def make_replacement(native):
    def replacement(A,B,shapeB,absmax,blocksize,quant_type,bias=None,
                    absmax_8bit=None,absmax_code=None,absmax_offset=None):
        args=(A,B,shapeB,absmax,blocksize,quant_type,bias,absmax_8bit,absmax_code,absmax_offset)
        raw_bias,why=normalized_native_bias(bias)
        if why is None:
            normalized=(A,B,shapeB,absmax,blocksize,quant_type,raw_bias,absmax_8bit,absmax_code,absmax_offset)
            why=reason(*normalized)
        if why is not None:
            COUNTS["native"]+=1
            FALLBACK_REASONS[why]=FALLBACK_REASONS.get(why,0)+1
            return native(*args)
        n,k=shapeB;m=A.numel()//k
        out=torch.empty((*A.shape[:-1],n),device=A.device,dtype=torch.float16)
        compiled=soup_nf4_mma_tiled_reuse[(triton.cdiv(m,64),triton.cdiv(n,64))](
            A,B,absmax_8bit,absmax_code,absmax,absmax_offset,A if raw_bias is None else raw_bias,out,
            m,n,k,bias is not None,num_warps=4,num_stages=1,enable_fp_fusion=False)
        COMPILED[(n,k,bias is not None)]=compiled
        COUNTS["custom"]+=1
        return out
    return replacement


def verify_native_owner(key,value,expected):
    owners={'fallback':('bitsandbytes.backends.cuda.ops','_dequant_linear_fallback'),
            'backward':('bitsandbytes.autograd._functions','MatMul4Bit.backward')}
    if (not inspect.isfunction(value) or hasattr(value,'__wrapped__')
            or (value.__module__,value.__qualname__)!=owners[key]):
        raise RuntimeError('Foreign native callable owner: '+key)
    if hashlib.sha256(inspect.getsource(value).encode()).hexdigest()!=expected[key]:
        raise RuntimeError('Native ownership source mismatch: '+key)


@contextmanager
def temporary_native_fallback(expected):
    """Only for controlled microbench children; fail closed on foreign owners."""
    import bitsandbytes.backends.cuda.ops as ops
    from bitsandbytes.autograd._functions import MatMul4Bit
    native=ops._dequant_linear_fallback;backward=MatMul4Bit.backward
    for key,value in (("fallback",native),("backward",backward)):
        verify_native_owner(key,value,expected)
    replacement=make_replacement(native)
    ops._dequant_linear_fallback=replacement
    try:
        yield replacement
        if MatMul4Bit.backward is not backward:
            raise RuntimeError("Native backward owner changed")
        if ops._dequant_linear_fallback is not replacement:
            raise RuntimeError("Fallback owner changed")
    finally:
        if ops._dequant_linear_fallback is replacement:
            ops._dequant_linear_fallback=native
