"""Original FP32 Qwen2 RMSNorm kernels for the fixed Soup task.

The reference formula and derivative come from Transformers 5.16.1's
Qwen2RMSNorm and PyTorch 2.11's constituent autograd rules. The Triton code
below was written independently for this solution. It does not import or
copy an optimization-kernel package.

Only the observed task layout is specialized: contiguous CUDA float32
residuals of width 1536, with frozen contiguous float32 gamma. All other
cases use the original class method. Forward reductions can differ in
rounding from eager PyTorch; device comparisons and end-to-end gates remain
necessary. No weight gradient or higher-order derivative is implemented.
"""

from __future__ import annotations

import functools
import math

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def soup_rms_forward_fp32(
    x_ptr,
    weight_ptr,
    y_ptr,
    rstd_ptr,
    WIDTH: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    active = column < WIDTH
    x = tl.load(x_ptr + row * WIDTH + column, mask=active, other=0.0)
    weight = tl.load(weight_ptr + column, mask=active, other=0.0)
    squared = x * x
    variance = tl.sum(squared, axis=0) * (1.0 / WIDTH)
    rstd = tl.rsqrt(variance + EPSILON)
    normalized = x * rstd
    output = weight * normalized
    tl.store(y_ptr + row * WIDTH + column, output, mask=active)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def soup_rms_backward_fp32(
    dy_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    dx_ptr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    active = column < WIDTH
    x = tl.load(x_ptr + row * WIDTH + column, mask=active, other=0.0)
    weight = tl.load(weight_ptr + column, mask=active, other=0.0)
    dy = tl.load(dy_ptr + row * WIDTH + column, mask=active, other=0.0)
    rstd = tl.load(rstd_ptr + row)

    # Retain the eager derivative's FP32 stages. In particular, do not
    # rewrite this as rstd * (h - x * mean(h*x) * rstd*rstd).
    h = dy * weight
    dot = tl.sum(h * x, axis=0)
    rstd2 = rstd * rstd
    rstd3 = rstd2 * rstd
    variance_grad = (-0.5 * dot) * rstd3
    square_grad = variance_grad * (1.0 / WIDTH)
    indirect = square_grad * (2.0 * x)
    direct = h * rstd
    dx = direct + indirect
    tl.store(dx_ptr + row * WIDTH + column, dx, mask=active)


class _TaskRMSNorm(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, epsilon):
        width = x.shape[-1]
        rows = x.numel() // width
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
        rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
        soup_rms_forward_fp32[(rows,)](
            x,
            weight,
            y,
            rstd,
            WIDTH=width,
            EPSILON=float(epsilon),
            BLOCK=triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        x, weight, rstd = ctx.saved_tensors
        dy = grad_output.contiguous()
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        width = x.shape[-1]
        rows = x.numel() // width
        soup_rms_backward_fp32[(rows,)](
            dy,
            x,
            weight,
            rstd,
            dx,
            WIDTH=width,
            BLOCK=triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return dx, None, None


def _supported(module, x):
    # This function is also reached by Soup regression tests containing mock
    # modules. Do not assume any model-specific state before checking it.
    if not isinstance(x, torch.Tensor):
        return False
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    if (
        not x.is_cuda
        or x.dtype != torch.float32
        or x.layout != torch.strided
        or x.ndim < 2
        or x.shape[-1] != 1536
        or x.numel() == 0
        or not x.is_contiguous()
        or weight.device != x.device
        or weight.dtype != torch.float32
        or weight.ndim != 1
        or weight.numel() != 1536
        or not weight.is_contiguous()
        or weight.requires_grad
    ):
        return False
    epsilon = getattr(module, "variance_epsilon", None)
    return isinstance(epsilon, (int, float)) and math.isfinite(epsilon) and epsilon > 0


def install():
    """Install the narrow class patch once; never construct a model or train."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    except (ImportError, AttributeError):
        # Some Soup regression tests provide minimal fake dependencies.
        return
    original = getattr(Qwen2RMSNorm, "forward", None)
    if original is None or getattr(original, "_soup_task_rmsnorm", False):
        return

    @functools.wraps(original)
    def forward(module, hidden_states):
        if not _supported(module, hidden_states):
            return original(module, hidden_states)
        return _TaskRMSNorm.apply(hidden_states, module.weight, module.variance_epsilon)

    forward._soup_task_rmsnorm = True
    forward._soup_original_forward = original
    Qwen2RMSNorm.forward = forward
