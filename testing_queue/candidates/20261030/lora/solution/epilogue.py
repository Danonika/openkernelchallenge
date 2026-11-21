"""Original narrow LoRA forward epilogue; not a GEMM/dropout replacement.

Contract: native half adapter B output -> half-rounded multiply by 2 ->
add with half/float base. The defensive PEFT clone is deliberately retained.
"""
import atexit
import collections
import json

import torch
import triton
import triton.language as tl

_COUNTS = collections.Counter()


def note(event, **details):
    _COUNTS[event] += 1
    if _COUNTS[event] == 1:
        print('SOUP_LORA_EPILOGUE ' + json.dumps({'event': event, **details}, sort_keys=True), flush=True)


@atexit.register
def _summary():
    if _COUNTS:
        print('SOUP_LORA_EPILOGUE ' + json.dumps({'event': 'process_exit_counts', 'counts': dict(_COUNTS)}, sort_keys=True), flush=True)


@triton.jit
def lora_half_scale_add(Base, Delta, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = i < N
    b = tl.load(Base + i, keep, other=0).to(tl.float32)
    d = tl.load(Delta + i, keep, other=0).to(tl.float32)
    # Native `delta * 2.0` writes a half tensor BEFORE the add. In particular,
    # half overflow may not be canceled by an opposite-sign float base value.
    scaled = (d * 2.0).to(tl.float16).to(tl.float32)
    tl.store(Out + i, b + scaled, keep)


class _ScaleAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, delta):
        ctx.base_dtype, ctx.delta_dtype = base.dtype, delta.dtype
        out = torch.empty_like(base, dtype=torch.promote_types(base.dtype, delta.dtype))
        note('launch_attempt', kernel='lora_half_scale_add', base_dtype=str(base.dtype),
             delta_dtype=str(delta.dtype), shape=list(delta.shape))
        lora_half_scale_add[(triton.cdiv(delta.numel(), 1024),)](
            base, delta, out, delta.numel(), BLOCK=1024, num_warps=4,
            enable_fp_fusion=False,
        )
        # This means host launch submission, not synchronized GPU completion.
        note('launch_submitted', kernel='lora_half_scale_add')
        return out

    @staticmethod
    def backward(ctx, grad):
        # Native AddBackward casts to each input dtype first. Native MulBackward
        # then multiplies the half delta adjoint by 2. Do not scale in float and
        # cast only afterwards: GradScaler overflow/rounding can differ.
        g_base = grad.to(ctx.base_dtype) if ctx.needs_input_grad[0] else None
        g_delta = grad.to(ctx.delta_dtype) * 2.0 if ctx.needs_input_grad[1] else None
        return g_base, g_delta


def scale_add(base, delta, scaling):
    """Conservative runtime dispatch; fallback has the original two operations."""
    okay = (
        type(base) is torch.Tensor and type(delta) is torch.Tensor
        and type(scaling) in (int, float) and scaling == 2.0
        and base.is_cuda and delta.is_cuda and base.device == delta.device
        and delta.dtype == torch.float16 and base.dtype in (torch.float16, torch.float32)
        and base.shape == delta.shape and base.ndim in (2, 3)
        and base.is_contiguous() and delta.is_contiguous()
        and delta.numel() > 0 and delta.shape[-1] in (1536, 256)
        and torch.cuda.get_device_capability(base.device) == (7, 5)
        and not torch.jit.is_tracing() and not torch.jit.is_scripting()
        and not torch.compiler.is_compiling()
    )
    if not okay:
        note('native_epilogue_fallback')
        return base + delta * scaling
    return _ScaleAdd.apply(base, delta)
