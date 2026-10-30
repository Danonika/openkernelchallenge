"""Original FP32 split-half RoPE fusion for the pinned Qwen2 attention path.

Reference arithmetic and layout:
https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py

The task's FP32 residual stream makes bitsandbytes Linear4bit return FP32;
Q/V LoRA then adds its result in that dtype.  Qwen2 also supplies FP32 rotary
coefficients.  This module therefore keeps the multiply and add operations in
FP32, including their separate rounding boundaries.  It does not move the
later SDPA autocast or grouped-key/value replication boundary.  Moving a cast
before replication can make its backward reduction overflow in FP16.

Each program handles a block of split-half pairs from either Q or K.  One
launch covers both tensors, with disjoint output writes and no reductions,
atomics, random numbers, input mutation, or retained Q/K activation copies.
The backward supports arbitrary upstream strides and saves only the frozen
cosine/sine tensors.  Both kernels are original implementations of the
equations; no third-party rotary kernel is imported or copied.

This is an optional, first-order training experiment.  Importing this module
does not install a patch, compile a kernel, or run a warmup.  Boundary arithmetic
parity does not establish whole-model bitwise parity: a combined autograd node
can change the scheduling of Q/K/V contributions to shared upstream inputs.
"""

from functools import wraps

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


_BLOCK = 512


@triton.jit
def _negate_fp32(x):
    # Triton unary minus lowers to zero-minus, whose +0 result for a +0 input
    # differs from torch.neg.  A sign-bit flip also preserves signed zeros.
    return (x.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)


@triton.jit
def _rotary_forward_block(
    X, COS, SIN, Y, block, batch, LENGTH,
    HEADS: tl.constexpr, WIDTH: tl.constexpr,
    BATCHED_COEFFICIENTS: tl.constexpr, BLOCK: tl.constexpr,
):
    half: tl.constexpr = WIDTH // 2
    pair = block * BLOCK + tl.arange(0, BLOCK)
    valid = pair < LENGTH * HEADS * half
    row, dim = pair // half, pair % half
    token = row // HEADS
    # Input and output are BTHD-dense allocations exposed as [B,H,T,D].
    offset = (batch.to(tl.int64) * LENGTH * HEADS + row) * WIDTH + dim
    coefficient_batch = batch if BATCHED_COEFFICIENTS else 0
    coefficient = (coefficient_batch * LENGTH + token) * WIDTH + dim
    u = tl.load(X + offset, valid, 0)
    v = tl.load(X + offset + half, valid, 0)
    c0 = tl.load(COS + coefficient, valid, 0)
    c1 = tl.load(COS + coefficient + half, valid, 0)
    s0 = tl.load(SIN + coefficient, valid, 0)
    s1 = tl.load(SIN + coefficient + half, valid, 0)
    # enable_fp_fusion=False at every launch preserves the reference's three
    # rounded FP32 operations for each result.  Do not contract these to FMA.
    y0 = u * c0 + _negate_fp32(v) * s0
    y1 = v * c1 + u * s1
    tl.store(Y + offset, y0, valid)
    tl.store(Y + offset + half, y1, valid)


@triton.jit
def soup_rotary_forward(
    Q, K, COS, SIN, Q_OUT, K_OUT, LENGTH, Q_BLOCKS,
    Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr, WIDTH: tl.constexpr,
    BATCHED_COEFFICIENTS: tl.constexpr, BLOCK: tl.constexpr,
):
    block, batch = tl.program_id(0), tl.program_id(1)
    # This branch is uniform within a program, and no program is launched for
    # padding beyond the combined Q/K block counts.
    if block < Q_BLOCKS:
        _rotary_forward_block(
            Q, COS, SIN, Q_OUT, block, batch, LENGTH,
            Q_HEADS, WIDTH, BATCHED_COEFFICIENTS, BLOCK,
        )
    else:
        _rotary_forward_block(
            K, COS, SIN, K_OUT, block - Q_BLOCKS, batch, LENGTH,
            K_HEADS, WIDTH, BATCHED_COEFFICIENTS, BLOCK,
        )


@triton.jit
def _rotary_backward_block(
    DY, COS, SIN, DX, block, batch, LENGTH, SB, SH, ST, SD,
    HEADS: tl.constexpr, WIDTH: tl.constexpr,
    BATCHED_COEFFICIENTS: tl.constexpr, BLOCK: tl.constexpr,
):
    half: tl.constexpr = WIDTH // 2
    pair = block * BLOCK + tl.arange(0, BLOCK)
    valid = pair < LENGTH * HEADS * half
    row, dim = pair // half, pair % half
    token, head = row // HEADS, row % HEADS
    offset = (batch.to(tl.int64) * LENGTH * HEADS + row) * WIDTH + dim
    gradient_offset = (
        batch.to(tl.int64) * SB + head.to(tl.int64) * SH
        + token.to(tl.int64) * ST + dim.to(tl.int64) * SD
    )
    coefficient_batch = batch if BATCHED_COEFFICIENTS else 0
    coefficient = (coefficient_batch * LENGTH + token) * WIDTH + dim
    g0 = tl.load(DY + gradient_offset, valid, 0)
    g1 = tl.load(DY + gradient_offset + half * SD, valid, 0)
    c0 = tl.load(COS + coefficient, valid, 0)
    c1 = tl.load(COS + coefficient + half, valid, 0)
    s0 = tl.load(SIN + coefficient, valid, 0)
    s1 = tl.load(SIN + coefficient + half, valid, 0)
    # Reference autograd first rounds each product, then negates the rotated
    # contribution to the second half, then adds the two FP32 contributions.
    rotate_u = g1 * s1
    rotate_v = _negate_fp32(g0 * s0)
    # SliceBackward pads each disjoint half with +0; adding those gradients
    # turns a -0 rotated contribution into +0.  Preserve this observable eager
    # behavior explicitly instead of relying on an optimizable "value + 0".
    rotate_u = tl.where(rotate_u == 0.0, 0.0, rotate_u)
    rotate_v = tl.where(rotate_v == 0.0, 0.0, rotate_v)
    du = g0 * c0 + rotate_u
    dv = g1 * c1 + rotate_v
    tl.store(DX + offset, du, valid)
    tl.store(DX + offset + half, dv, valid)


@triton.jit
def soup_rotary_backward(
    DQ_OUT, DK_OUT, COS, SIN, DQ, DK, LENGTH, Q_BLOCKS,
    Q_SB, Q_SH, Q_ST, Q_SD, K_SB, K_SH, K_ST, K_SD,
    Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr, WIDTH: tl.constexpr,
    BATCHED_COEFFICIENTS: tl.constexpr,
    DO_Q: tl.constexpr, DO_K: tl.constexpr, BLOCK: tl.constexpr,
):
    block, batch = tl.program_id(0), tl.program_id(1)
    if DO_Q and DO_K:
        if block < Q_BLOCKS:
            _rotary_backward_block(
                DQ_OUT, COS, SIN, DQ, block, batch, LENGTH, Q_SB, Q_SH, Q_ST, Q_SD,
                Q_HEADS, WIDTH, BATCHED_COEFFICIENTS, BLOCK,
            )
        else:
            _rotary_backward_block(
                DK_OUT, COS, SIN, DK, block - Q_BLOCKS, batch, LENGTH, K_SB, K_SH, K_ST, K_SD,
                K_HEADS, WIDTH, BATCHED_COEFFICIENTS, BLOCK,
            )
    elif DO_Q:
        _rotary_backward_block(
            DQ_OUT, COS, SIN, DQ, block, batch, LENGTH, Q_SB, Q_SH, Q_ST, Q_SD,
            Q_HEADS, WIDTH, BATCHED_COEFFICIENTS, BLOCK,
        )
    elif DO_K:
        _rotary_backward_block(
            DK_OUT, COS, SIN, DK, block, batch, LENGTH, K_SB, K_SH, K_ST, K_SD,
            K_HEADS, WIDTH, BATCHED_COEFFICIENTS, BLOCK,
        )


def _standard_strides(shape):
    _, heads, length, width = shape
    return (length * heads * width, width, heads * width, 1)


def _supported(q, k, cos, sin, unsqueeze_dim=1, *, cuda_only=False):
    tensors = (q, k, cos, sin)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        return False
    if (
        unsqueeze_dim != 1
        or any(x.dtype != torch.float32 or x.layout != torch.strided for x in tensors)
        or any(x.device != q.device or x.is_conj() or x.is_neg() for x in tensors)
        or (cuda_only and not q.is_cuda)
        or q.device.type not in ("cpu", "cuda")
        or q.ndim != 4 or k.ndim != 4 or cos.ndim != 3 or sin.ndim != 3
        or any(size <= 0 for size in (*q.shape, *k.shape))
    ):
        return False
    batch, _, length, width = q.shape
    if (
        k.shape[0] != batch or k.shape[2:] != (length, width)
        or width % 2 or width > 256
        or q.stride() != _standard_strides(q.shape)
        or k.stride() != _standard_strides(k.shape)
        or cos.shape != sin.shape or cos.shape[0] not in (1, batch)
        or cos.shape[1:] != (length, width)
        or not cos.is_contiguous() or not sin.is_contiguous()
        or cos.requires_grad or sin.requires_grad
        or max(q.numel(), k.numel(), cos.numel()) >= 2**31
        # Joining two paths from the same storage can change the reference's
        # upstream gradient accumulation order.  The task has separate Q/K
        # projections; conservatively delegate every shared-storage case.
        or q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
    ):
        return False
    return True


def _new_output(shape, device):
    return torch.empty_strided(shape, _standard_strides(shape), dtype=torch.float32, device=device)


class _FrozenRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        batch, q_heads, length, width = q.shape
        k_heads = k.shape[1]
        q_blocks = triton.cdiv(length * q_heads * (width // 2), _BLOCK)
        k_blocks = triton.cdiv(length * k_heads * (width // 2), _BLOCK)
        q_out, k_out = _new_output(q.shape, q.device), _new_output(k.shape, k.device)
        batched_coefficients = cos.shape[0] != 1
        soup_rotary_forward[(q_blocks + k_blocks, batch)](
            q, k, cos, sin, q_out, k_out, length, q_blocks,
            q_heads, k_heads, width, batched_coefficients, _BLOCK,
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(cos, sin)
        ctx.q_shape, ctx.k_shape = q.shape, k.shape
        ctx.batched_coefficients = batched_coefficients
        ctx.set_materialize_grads(False)
        non_differentiable = []
        if not ctx.needs_input_grad[0]:
            non_differentiable.append(q_out)
        if not ctx.needs_input_grad[1]:
            non_differentiable.append(k_out)
        if non_differentiable:
            ctx.mark_non_differentiable(*non_differentiable)
        return q_out, k_out

    @staticmethod
    @once_differentiable
    def backward(ctx, dq_out, dk_out):
        do_q = ctx.needs_input_grad[0] and dq_out is not None
        do_k = ctx.needs_input_grad[1] and dk_out is not None
        if not (do_q or do_k):
            return None, None, None, None
        if do_q and (dq_out.is_neg() or dq_out.is_conj()):
            dq_out = dq_out.resolve_neg().resolve_conj()
        if do_k and (dk_out.is_neg() or dk_out.is_conj()):
            dk_out = dk_out.resolve_neg().resolve_conj()
        cos, sin = ctx.saved_tensors
        batch, q_heads, length, width = ctx.q_shape
        k_heads = ctx.k_shape[1]
        dq = _new_output(ctx.q_shape, cos.device) if do_q else None
        dk = _new_output(ctx.k_shape, cos.device) if do_k else None
        q_blocks = triton.cdiv(length * q_heads * (width // 2), _BLOCK) if do_q else 0
        k_blocks = triton.cdiv(length * k_heads * (width // 2), _BLOCK) if do_k else 0
        # A pointer of the right dtype is required for the unused specialization
        # arguments.  Compile-time DO_Q/DO_K removes every access to it.
        q_pointer, k_pointer = dq_out if do_q else cos, dk_out if do_k else cos
        q_strides = dq_out.stride() if do_q else (0, 0, 0, 0)
        k_strides = dk_out.stride() if do_k else (0, 0, 0, 0)
        soup_rotary_backward[(q_blocks + k_blocks, batch)](
            q_pointer, k_pointer, cos, sin, dq if do_q else cos, dk if do_k else cos,
            length, q_blocks, *q_strides, *k_strides,
            q_heads, k_heads, width, ctx.batched_coefficients, do_q, do_k, _BLOCK,
            num_warps=4, enable_fp_fusion=False,
        )
        return dq, dk, None, None


def fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply FP32 Qwen split-half RoPE with frozen coefficients.

    Q/K must be [B,H,T,D] views of separate dense [B,T,H,D] storage, with positive
    dimensions and even D <= 256.  Cos/sin must be contiguous [1 or B,T,D]
    FP32 tensors on the same device, without gradients.  Each result keeps its
    input's shape, dtype, and strides and has independent storage.

    Normally called on CUDA; direct calls also support Triton's official CPU
    interpreter when TRITON_INTERPRET=1.  The optional patch selects CUDA only.
    Higher-order derivatives are deliberately outside this narrow fast path.
    """
    if not _supported(q, k, cos, sin, unsqueeze_dim):
        raise ValueError("rotary requires standard Q/K strides and frozen matching FP32 coefficients")
    return _FrozenRotary.apply(q, k, cos, sin)


def patch_qwen2_rotary():
    """Idempotently install the guarded fast path on Transformers 5.16.1."""
    import transformers
    from transformers.models.qwen2 import modeling_qwen2

    if transformers.__version__ != "5.16.1":
        return False
    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_soup_kernel_rotary", False):
        return True

    @wraps(original)
    def forward(q, k, cos, sin, unsqueeze_dim=1):
        # Decide before running any custom work.  Fallback never repeats a
        # projection or dropout operation and consumes no additional RNG.
        if not _supported(q, k, cos, sin, unsqueeze_dim, cuda_only=True):
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)

    forward._soup_kernel_rotary = True
    modeling_qwen2.apply_rotary_pos_emb = forward
    return True
