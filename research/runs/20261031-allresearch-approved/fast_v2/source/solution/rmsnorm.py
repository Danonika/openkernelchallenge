"""Original fused Qwen2 RMSNorm for frozen FP32 scale parameters.

The implementation follows the RMS normalization equation, not another kernel's
code.  The reference semantics are transformers 5.16.1 Qwen2RMSNorm.forward:
normalize in FP32, round to the input dtype, then multiply the FP32 weight.
That last multiplication returns FP32 even when the input is FP16.  Its
backward also rounds the weighted output gradient to the input dtype.

One Triton program handles one token in both directions.  Saving only the
original input, the frozen weight, and one inverse RMS per token avoids keeping
the reference's FP32 intermediate tensors for backward.  No autotuning or
compiler-cache warmup is performed by this module.
"""

from functools import wraps

import torch
import triton
import triton.language as tl


@triton.jit
def soup_rmsnorm_forward(
    X, W, Y, INV_RMS,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    x = tl.load(X + row * WIDTH + col, valid, 0).to(tl.float32)
    w = tl.load(W + col, valid, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / WIDTH + EPS)
    # A required reference cast: moving this after weight multiplication would
    # change the FP16 path and its derivative.
    normalized = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * WIDTH + col, normalized * w, valid)
    tl.store(INV_RMS + row, inv)


@triton.jit
def soup_rmsnorm_backward(
    DY, X, W, INV_RMS, DX,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    x = tl.load(X + row * WIDTH + col, valid, 0).to(tl.float32)
    w = tl.load(W + col, valid, 0).to(tl.float32)
    dy = tl.load(DY + row * WIDTH + col, valid, 0).to(tl.float32)
    inv = tl.load(INV_RMS + row)
    # Autograd through weight * normalized.to(input_dtype) casts this gradient
    # to input_dtype before passing it back through FP32 normalization.
    g = (dy * w).to(X.dtype.element_ty).to(tl.float32)
    # Follow eager autograd's arithmetic boundaries instead of factoring the
    # mathematically equivalent expression into (g - x*c)*inv. The user's
    # Nanochat transfer audit found that this factoring changes FP32 rounding.
    # Reduction trees and device rsqrt still need target-GPU tolerance checks.
    direct = g * inv
    dr = tl.sum(g * x, axis=0)
    inv_squared = inv * inv
    inv_cubed = inv_squared * inv
    dv = (dr * -0.5) * inv_cubed
    dpow = dv * (1.0 / WIDTH)
    indirect = dpow * (2.0 * x)
    dx = direct + indirect
    tl.store(DX + row * WIDTH + col, dx, valid)


class _FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        width = x.shape[-1]
        rows = x.numel() // width
        y = torch.empty_like(x, dtype=torch.float32)
        inverse_rms = torch.empty((rows,), device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(width)
        soup_rmsnorm_forward[(rows,)](
            x, weight, y, inverse_rms, width, float(eps), block,
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, weight, inverse_rms)
        ctx.width = width
        ctx.block = block
        return y

    @staticmethod
    def backward(ctx, dy):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        x, weight, inverse_rms = ctx.saved_tensors
        dx = torch.empty_like(x)
        # Upstream autograd may supply an expanded or otherwise strided dy.
        dy = dy.contiguous()
        soup_rmsnorm_backward[(x.numel() // ctx.width,)](
            dy, x, weight, inverse_rms, dx, ctx.width, ctx.block,
            num_warps=4, enable_fp_fusion=False,
        )
        return dx, None, None


def rmsnorm(x, weight, eps=1e-6):
    """Fused RMSNorm for contiguous FP16/FP32 inputs and frozen FP32 weights.

    Normally called on CUDA.  Direct invocation also permits Triton's official
    CPU interpreter for development; the model patch below only selects CUDA.
    """
    if x.ndim < 1 or x.numel() == 0 or not x.is_contiguous():
        raise ValueError("rmsnorm requires a nonempty contiguous input")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError("rmsnorm supports FP16 and FP32 activations")
    if (
        weight.ndim != 1 or weight.numel() != x.shape[-1]
        or weight.dtype != torch.float32 or not weight.is_contiguous()
        or weight.device != x.device or weight.requires_grad
    ):
        raise ValueError("rmsnorm requires matching frozen contiguous FP32 weights")
    if x.shape[-1] > 8192 or eps <= 0:
        raise ValueError("rmsnorm requires width <= 8192 and positive epsilon")
    return _FrozenRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Idempotently install the narrow fast path; preserve every other call."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    if not isinstance(Qwen2RMSNorm, type) or not issubclass(Qwen2RMSNorm, torch.nn.Module):
        return False
    original = Qwen2RMSNorm.forward
    if getattr(original, "_soup_kernel_rmsnorm", False):
        return True

    @wraps(original)
    def forward(self, hidden_states):
        weight = self.weight
        supported = (
            isinstance(hidden_states, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and hidden_states.is_cuda
            and hidden_states.device == weight.device
            and hidden_states.dtype in (torch.float16, torch.float32)
            and weight.dtype == torch.float32
            and not weight.requires_grad
            and hidden_states.ndim >= 1
            and hidden_states.numel() > 0
            and 0 < hidden_states.shape[-1] <= 8192
            and weight.ndim == 1
            and hidden_states.shape[-1] == weight.numel()
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and self.variance_epsilon > 0
        )
        if not supported:
            return original(self, hidden_states)
        return rmsnorm(hidden_states, weight, self.variance_epsilon)

    forward._soup_kernel_rmsnorm = True
    Qwen2RMSNorm.forward = forward
    return True
