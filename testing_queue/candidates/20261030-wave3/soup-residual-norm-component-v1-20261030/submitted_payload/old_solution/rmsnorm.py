"""Original FP32 frozen-affine RMSNorm trial; no residual fusion or borrowed kernel.

The derivative retains the source graph's individual FP32 arithmetic stages.
It is not a claim of bitwise equivalence to every parallel reduction or external
autograd accumulation order. Unsupported module contracts use the original method.
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def soup_rms32_forward(X, W, Y, R, H: tl.constexpr, EPS: tl.constexpr,
                       KEEP_R: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < H
    x = tl.load(X + row * H + col, valid, other=0).to(tl.float32)
    w = tl.load(W + col, valid, other=0).to(tl.float32)
    squares = x * x
    variance = tl.div_rn(tl.sum(squares, axis=0), H)
    inv = tl.rsqrt(variance + EPS)
    normalized = x * inv
    y = w * normalized
    tl.store(Y + row * H + col, y, valid)
    if KEEP_R:
        tl.store(R + row, inv)


@triton.jit
def soup_rms32_backward(DY, X, W, R, DX, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < H
    x = tl.load(X + row * H + col, valid, other=0).to(tl.float32)
    w = tl.load(W + col, valid, other=0).to(tl.float32)
    dy = tl.load(DY + row * H + col, valid, other=0).to(tl.float32)
    inv = tl.load(R + row).to(tl.float32)
    g = dy * w
    direct = g * inv
    dr = tl.sum(g * x, axis=0)
    inv_cubed = (inv * inv) * inv
    dv = (-0.5 * dr) * inv_cubed
    scaled_variance = tl.div_rn(dv, H)
    variance_path = scaled_variance * (2.0 * x)
    dx = direct + variance_path
    tl.store(DX + row * H + col, dx, valid)


class FrozenRMS32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, need_backward):
        # The public helper checks the contract; no input, gradient or weight mutation.
        h = x.shape[-1]
        rows = x.numel() // h
        y = torch.empty_like(x)
        keep = bool(need_backward)
        inv = torch.empty((rows,) if keep else (0,), device=x.device, dtype=torch.float32)
        if rows:
            soup_rms32_forward[(rows,)](
                x, weight, y, inv, h, float(eps), keep,
                BLOCK=triton.next_power_of_2(h), num_warps=4, enable_fp_fusion=False)
        if keep:
            ctx.save_for_backward(x, weight, inv)
        ctx.h = h
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight, inv = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        rows = x.numel() // ctx.h
        if rows:
            soup_rms32_backward[(rows,)](
                dy, x, weight, inv, dx, ctx.h,
                BLOCK=triton.next_power_of_2(ctx.h), num_warps=4, enable_fp_fusion=False)
        return dx, None, None, None


def supported(x, weight):
    return (isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor)
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.dtype == torch.float32 and weight.dtype == torch.float32
            and x.ndim >= 1 and 0 < x.shape[-1] <= 4096
            and weight.shape == (x.shape[-1],) and not weight.requires_grad
            and x.is_contiguous() and weight.is_contiguous())


def frozen_rms32(x, weight, eps=1e-6):
    if not supported(x, weight):
        raise ValueError('frozen_rms32 needs contiguous CUDA FP32 x and frozen FP32 weight')
    need_backward = torch.is_grad_enabled() and x.requires_grad
    return FrozenRMS32.apply(x, weight, eps, need_backward)


def patch_qwen2_norm():
    from transformers.models.qwen2 import modeling_qwen2
    cls = modeling_qwen2.Qwen2RMSNorm
    original = cls.forward
    if getattr(original, '_soup_original_rms_trial001', False):
        return
    if (cls.__module__ != 'transformers.models.qwen2.modeling_qwen2'
            or getattr(original, '__module__', None) != cls.__module__):
        return

    def forward(self, hidden_states):
        weight = getattr(self, 'weight', None)
        if (type(self) is cls and supported(hidden_states, weight) and hidden_states.shape[-1] == 1536
                and isinstance(getattr(self, 'variance_epsilon', None), (int, float))):
            return frozen_rms32(hidden_states, weight, self.variance_epsilon)
        return original(self, hidden_states)

    forward._soup_original_rms_trial001 = True
    forward._soup_unpatched_forward = original
    cls.forward = forward
