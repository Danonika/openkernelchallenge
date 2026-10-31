"""Optional installation of the bounded frozen NF4 cache.

The ordinary Linear4bit module establishes training intent before its custom
autograd forward disables grad mode. The supported CUDA fallback uses a
retained matrix. Backward reuse is a separate, disabled-by-default option:
only native MatMul4Bit.backward's exact packed tensor and QuantState can use
that same entry. Public calls, evaluation, and supplied out buffers bypass.
Every linear/matmul and bias reduction still executes the native operation.
"""

from contextvars import ContextVar
from functools import wraps
from importlib.metadata import version

import torch

from .nf4_forward_cache import FrozenNF4ForwardCache


_TRAINING_FORWARD = ContextVar("soup_nf4_cache_training_forward", default=False)
_NATIVE_BACKWARD = ContextVar("soup_nf4_cache_native_backward", default=None)


def _training_wrapper(original):
    @wraps(original)
    def forward(self, *args, **kwargs):
        token = _TRAINING_FORWARD.set(
            bool(self.training and torch.is_grad_enabled()
                 and not torch.is_inference_mode_enabled()))
        try:
            return original(self, *args, **kwargs)
        finally:
            _TRAINING_FORWARD.reset(token)
    forward._soup_nf4_cache_training_context = True
    return forward


def _canonical_components(components):
    """Normalize an already-validated contiguous packed view without copying.

    The public API uses [1, bytes] to request a transposed output. Orientation
    is restored at that boundary; it is not a different decoded cache value.
    The persistent source root, storage and version counter are unchanged.
    """
    return dict(components, packed=components["packed"].reshape(-1, 1))


def _matmul_forward_wrapper(original):
    @wraps(original)
    def forward(ctx, *args, **kwargs):
        # Capture module intent while it is still in scope. Grad mode is
        # disabled inside Function.forward even for a training call.
        ctx._soup_nf4_cache_training = _TRAINING_FORWARD.get() is True
        return original(ctx, *args, **kwargs)
    forward._soup_nf4_cache_native_forward_context = True
    return forward


def _matmul_backward_wrapper(original):
    @wraps(original)
    def backward(ctx, grad_output):
        admitted = (getattr(ctx, "_soup_nf4_cache_training", False) is True
                    and not ctx.is_empty and ctx.needs_input_grad[0]
                    and ctx.dtype_A == torch.float16
                    and isinstance(grad_output, torch.Tensor)
                    and grad_output.dtype == torch.float16
                    and not torch.is_inference_mode_enabled())
        token = _NATIVE_BACKWARD.set(ctx if admitted else None)
        try:
            return original(ctx, grad_output)
        finally:
            _NATIVE_BACKWARD.reset(token)
    backward._soup_nf4_cache_native_backward_context = True
    return backward


def _backward_dequantize_wrapper(original, cache, nf4):
    @wraps(original)
    def dequantize_4bit(A, quant_state=None, absmax=None, out=None,
                       blocksize=None, quant_type="fp4"):
        ctx = _NATIVE_BACKWARD.get()
        if (ctx is None or out is not None or quant_state is not ctx.state
                or A is not ctx.tensors[1]):
            return original(A, quant_state, absmax, out, blocksize, quant_type)
        components = nf4._state_components(A, quant_state, out, cuda_only=True)
        if components is None or components["dtype"] != torch.float16:
            return original(A, quant_state, absmax, out, blocksize, quant_type)
        decoded = cache.decode(_canonical_components(components), nf4._execute,
                               training=True)
        return decoded.t() if A.shape[0] == 1 else decoded

    dequantize_4bit._soup_nf4_backward_cache = cache
    return dequantize_4bit


def patch_nf4_forward_cache(byte_budget, *, cache_backward=False):
    """Return the installed cache, or False for an unsupported installation.

    The budget is a fixed source setting. It never depends on measurements,
    available device memory, batch contents, or the test environment.
    ``cache_backward`` is also fixed at installation; it never changes the
    native backward math or admits an unrelated public dequantization call.
    """
    if type(cache_backward) is not bool:
        raise ValueError("cache_backward must be a bool")
    if type(byte_budget) is not int or byte_budget <= 0:
        return False
    if (version("bitsandbytes") != "0.50.2" or torch.version.cuda is None
            or torch.version.hip is not None):
        return False
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    from bitsandbytes.backends.cuda import ops
    from . import nf4

    original_forward = bnb.nn.Linear4bit.forward
    original_fallback = ops._dequant_linear_fallback
    existing = getattr(original_fallback, "_soup_nf4_forward_cache", None)
    if existing is not None:
        if existing.byte_budget != byte_budget:
            raise ValueError("NF4 forward cache already has a different fixed budget")
        if getattr(original_fallback, "_soup_nf4_cache_backward", False) != cache_backward:
            raise ValueError("NF4 cache already has a different fixed backward scope")
        return existing
    if (not getattr(functional.dequantize_4bit, "_soup_candidate_nf4", False)
            or not getattr(original_fallback, "_soup_candidate_nf4", False)):
        return False
    cache = FrozenNF4ForwardCache(byte_budget)

    @wraps(original_fallback)
    def fallback(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                 absmax_8bit=None, absmax_code=None, absmax_offset=None):
        if not _TRAINING_FORWARD.get():
            return original_fallback(A, B, shapeB, absmax, blocksize, quant_type,
                                     bias, absmax_8bit, absmax_code, absmax_offset)
        nested = absmax_8bit is not None
        components = dict(
            packed=B, shape=shapeB, dtype=A.dtype,
            scales=absmax_8bit if nested else absmax, blocksize=blocksize,
            nested_scales=absmax if nested else None, nested_code=absmax_code,
            offset=absmax_offset, nested_blocksize=256,
        )
        if (quant_type != "nf4" or A.device != B.device
                or not nf4._components_supported(**components, cuda_only=True)):
            return original_fallback(A, B, shapeB, absmax, blocksize, quant_type,
                                     bias, absmax_8bit, absmax_code, absmax_offset)
        decoded = cache.decode(_canonical_components(components), nf4._execute,
                               training=True)
        return torch.nn.functional.linear(A, decoded, bias)

    fallback._soup_nf4_forward_cache = cache
    fallback._soup_nf4_cache_backward = cache_backward
    if cache_backward:
        from bitsandbytes.autograd._functions import MatMul4Bit

        functional.dequantize_4bit = _backward_dequantize_wrapper(
            functional.dequantize_4bit, cache, nf4)
        MatMul4Bit.forward = staticmethod(_matmul_forward_wrapper(MatMul4Bit.forward))
        MatMul4Bit.backward = staticmethod(_matmul_backward_wrapper(MatMul4Bit.backward))
    ops._dequant_linear_fallback = fallback
    bnb.nn.Linear4bit.forward = _training_wrapper(original_forward)
    return cache
