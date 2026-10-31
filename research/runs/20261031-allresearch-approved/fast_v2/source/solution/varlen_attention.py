"""Uninstalled experiment: native Torch variable-length causal attention.

This module does not patch a model or alter the selected solution profile.
Its caller must own a finite, private training path whose padded query values
are unobserved and whose upstream derivatives at padding are zero. Padded
outputs are zero here; arbitrary NaN/Inf propagation and public full outputs
are deliberately outside this primitive's contract.

Inputs are FP32 BHTD tensors AFTER the original rectangular q/v LoRA dropout,
projection, rotary and native FP32 repeat_kv. Packing performs the same FP16
conversion as native SDPA autocast. Backward scatters each expanded head's
native FP16 derivative to FP32 BEFORE repeat_kv's unchanged FP32 adjoint sum.
The native efficient-attention operators do all attention arithmetic. The
four row-copy kernels are original implementations, not attention kernels.

Pinned primary-source audit (PyTorch v2.11.0):
* attention.cu, _efficient_attention_forward: mode 1MHK accepts [1,total,H,D],
  CUDA int32 cumulative lengths, max lengths and custom_mask_type=1 (top-left
  causal). Sequence starts/ends are applied independently inside the kernel.
* attention_backward.cu: cumulative lengths and bias cannot be combined.
  Both max lengths must be positive. Forward's returned maxK is zero unless
  its max_seqlen_k argument is supplied, so this wrapper supplies both.
* Generated cutlassF.h and cutlassB.h include FP16 SM75 head-dimension-128
  variants without a block-length-alignment requirement. Their presence is
  source evidence, not a substitute for executing the physical T4 tests.
* dropout_p=0 never acquires the CUDA generator in the native forward.

Source URLs begin https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/
aten/src/ATen/native/transformers/cuda/ . Relevant SHA256 values:
attention.cu: fd736fba414a1846d3e588ff15b279c105d9c03995abc7931b900e88ae15e46d
attention_backward.cu: e1258ad7d15fa9633da8e66d92827b9c202b498e9d232346745ed1b70bc811f4
mem_eff_attention/kernels/cutlassF.h:
e38db7b7ff3fa98eb212da5740b4251258bf1940bca12ebc6bb89f80c932d398
mem_eff_attention/kernels/cutlassB.h:
9e44ca5b38b05897720f5ef20496406accf5cb9c9738c57009ad59267d19d1e7

There is no emulated native-attention fallback or higher-order derivative.
Unsupported callers must retain their existing native SDPA path.
"""

from dataclasses import dataclass
import math
import os

import torch
import triton
import triton.language as tl


_BLOCK = 512
_CAUSAL_FROM_TOP_LEFT = 1


@dataclass(frozen=True)
class VarlenPlan:
    """One forward's validated right-padding plan; do not mutate its tensors."""

    batch: int
    sequence: int
    lengths: tuple
    total: int
    maximum: int
    rows: torch.Tensor
    inverse: torch.Tensor
    cumulative: torch.Tensor
    versions: tuple


def plan_from_row_tensors(rows, inverse, cumulative, *, lengths, sequence):
    """Reuse one validated CPU row plan's device tensors without copying.

    ``lengths`` and ``sequence`` are ordinary Python metadata obtained while
    validating that same CPU right-padding snapshot. ``rows``/``inverse`` may
    be the existing int64 MLP maps; only ``cumulative`` must be native int32.
    This factory checks metadata and versions, never synchronizes or reads
    GPU values. The caller must supply maps from the same validated mask.
    """
    if (type(lengths) is not tuple or not lengths or type(sequence) is not int
            or sequence < 1 or any(type(length) is not int or not 1 <= length <= sequence
                                   for length in lengths)
            or len(lengths) * sequence > torch.iinfo(torch.int32).max):
        raise ValueError("requires CPU-validated nonempty right-padding lengths")
    tensors = (rows, inverse, cumulative)
    if any(type(tensor) is not torch.Tensor for tensor in tensors):
        raise ValueError("row maps and cumulative lengths must be tensors")
    plan = VarlenPlan(len(lengths), sequence, lengths, sum(lengths), max(lengths),
                      rows, inverse, cumulative, tuple(tensor._version for tensor in tensors))
    _check_plan(plan, rows.device)
    return plan


def make_varlen_plan(host_mask, *, device):
    """Build a reusable per-forward plan from an existing CPU mask snapshot.

    This function never copies a CUDA mask back to the host. A future owner
    can share its existing one-copy MLP row plan. Nonempty full and short
    sequences are supported. Left padding, holes and empty sequences raise;
    the owning caller must choose its original attention for those inputs.
    CPU plans exist for isolated row-copy tests; the attention entry rejects
    CPU execution and the Triton interpreter.
    """
    if (type(host_mask) is not torch.Tensor or host_mask.device.type != "cpu"
            or host_mask.ndim != 2 or min(host_mask.shape) < 1
            or host_mask.dtype not in (torch.bool, torch.int32, torch.int64)):
        raise ValueError("plan requires a nonempty CPU binary mask with shape (B,T)")
    if not bool(((host_mask == 0) | (host_mask == 1)).all()):
        raise ValueError("attention mask must contain only zero and one")
    valid = host_mask == 1
    if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("only nonempty right-padded sequences are supported")
    batch, sequence = valid.shape
    lengths = tuple(int(value) for value in valid.sum(dim=1).tolist())
    total = sum(lengths)
    if valid.numel() > torch.iinfo(torch.int32).max:
        raise ValueError("row plan exceeds native int32 cumulative-length range")
    rows = valid.flatten().nonzero(as_tuple=False).flatten().to(torch.int32)
    inverse = torch.full((valid.numel(),), -1, dtype=torch.int32)
    inverse.index_copy_(0, rows.long(), torch.arange(total, dtype=torch.int32))
    cumulative = torch.tensor((0, *lengths), dtype=torch.int32).cumsum(0, dtype=torch.int32)
    rows, inverse, cumulative = (value.to(device=device).contiguous()
                                for value in (rows, inverse, cumulative))
    return plan_from_row_tensors(rows, inverse, cumulative, lengths=lengths, sequence=sequence)


@triton.jit(do_not_specialize=["N", "T", "Q_B", "Q_H", "Q_T",
                             "K_B", "K_H", "K_T", "V_B", "V_H", "V_T"])
def soup_varlen_pack_qkv(
    Q, K, V, ROWS, Q_OUT, K_OUT, V_OUT, N, T,
    Q_B, Q_H, Q_T, K_B, K_H, K_T, V_B, V_H, V_T,
    H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < N
    packed_row = offset // (H * D)
    head = (offset // D) % H
    column = offset % D
    source = tl.load(ROWS + packed_row, active, other=0).to(tl.int64)
    batch, token = source // T, source % T
    q = tl.load(Q + batch * Q_B + head * Q_H + token * Q_T + column, active, other=0)
    k = tl.load(K + batch * K_B + head * K_H + token * K_T + column, active, other=0)
    v = tl.load(V + batch * V_B + head * V_H + token * V_T + column, active, other=0)
    # Output pointers are FP16: this is the native SDPA autocast boundary.
    tl.store(Q_OUT + offset, q, active)
    tl.store(K_OUT + offset, k, active)
    tl.store(V_OUT + offset, v, active)


@triton.jit(do_not_specialize=["N"])
def soup_varlen_scatter_qkv(
    DQ, DK, DV, INVERSE, Q_OUT, K_OUT, V_OUT, N,
    WIDTH: tl.constexpr, NEED_Q: tl.constexpr, NEED_K: tl.constexpr,
    NEED_V: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < N
    row, column = offset // WIDTH, offset % WIDTH
    source = tl.load(INVERSE + row, active, other=-1).to(tl.int64)
    location = tl.maximum(source, 0) * WIDTH + column
    valid = active & (source >= 0)
    # FP32 destinations restore the native cast's adjoint before grouped-KV
    # reduction. Each location has one writer; padding needs no extra memset.
    if NEED_Q:
        tl.store(Q_OUT + offset, tl.load(DQ + location, valid, other=0), active)
    if NEED_K:
        tl.store(K_OUT + offset, tl.load(DK + location, valid, other=0), active)
    if NEED_V:
        tl.store(V_OUT + offset, tl.load(DV + location, valid, other=0), active)


@triton.jit(do_not_specialize=["N", "T", "X_B", "X_H", "X_T"])
def soup_varlen_gather_output_gradient(
    X, ROWS, OUT, N, T, X_B, X_H, X_T,
    H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < N
    packed_row, head, column = offset // (H * D), (offset // D) % H, offset % D
    source = tl.load(ROWS + packed_row, active, other=0).to(tl.int64)
    location = (source // T) * X_B + head * X_H + (source % T) * X_T + column
    tl.store(OUT + offset, tl.load(X + location, active, other=0), active)


@triton.jit(do_not_specialize=["N"])
def soup_varlen_scatter_output(X, INVERSE, OUT, N, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < N
    row, column = offset // WIDTH, offset % WIDTH
    source = tl.load(INVERSE + row, active, other=-1).to(tl.int64)
    value = tl.load(X + tl.maximum(source, 0) * WIDTH + column,
                    active & (source >= 0), other=0)
    tl.store(OUT + offset, value, active)


def _pack_qkv(q, k, v, plan):
    heads, width = q.shape[1], q.shape[3]
    outputs = tuple(torch.empty((1, plan.total, heads, width),
                                dtype=torch.float16, device=q.device) for _ in range(3))
    size = outputs[0].numel()
    soup_varlen_pack_qkv[(triton.cdiv(size, _BLOCK),)](
        q, k, v, plan.rows, *outputs, size, plan.sequence,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        heads, width, _BLOCK, num_warps=4)
    return outputs


def _scatter_qkv(gradients, plan):
    present = next((value for value in gradients if value is not None), None)
    if present is None:
        return (None, None, None)
    heads, width = present.shape[-2:]
    gradients = tuple(value.resolve_conj().resolve_neg().contiguous()
                      if value is not None else None for value in gradients)
    shape = (plan.batch, plan.sequence, heads, width)
    outputs = tuple(torch.empty(shape, dtype=torch.float32, device=present.device)
                    if value is not None else None for value in gradients)
    dummy_input = next(value for value in gradients if value is not None)
    dummy_output = next(value for value in outputs if value is not None)
    size = dummy_output.numel()
    soup_varlen_scatter_qkv[(triton.cdiv(size, _BLOCK),)](
        *(value if value is not None else dummy_input for value in gradients), plan.inverse,
        *(value if value is not None else dummy_output for value in outputs), size,
        heads * width, *(value is not None for value in gradients), _BLOCK, num_warps=4)
    return tuple(value.permute(0, 2, 1, 3) if value is not None else None for value in outputs)


def _gather_output_gradient(grad, plan):
    grad = grad.resolve_conj().resolve_neg()
    if grad.stride(-1) != 1:
        grad = grad.contiguous()
    heads, width = grad.shape[1], grad.shape[3]
    output = torch.empty((1, plan.total, heads, width), dtype=torch.float16, device=grad.device)
    soup_varlen_gather_output_gradient[(triton.cdiv(output.numel(), _BLOCK),)](
        grad, plan.rows, output, output.numel(), plan.sequence,
        *grad.stride()[:3], heads, width, _BLOCK, num_warps=4)
    return output


def _scatter_output(output, plan):
    heads, width = output.shape[-2:]
    output = output.resolve_conj().resolve_neg().contiguous()
    rectangular = torch.empty((plan.batch, plan.sequence, heads, width),
                              dtype=torch.float16, device=output.device)
    soup_varlen_scatter_output[(triton.cdiv(rectangular.numel(), _BLOCK),)](
        output, plan.inverse, rectangular, rectangular.numel(), heads * width,
        _BLOCK, num_warps=4)
    # The underlying storage is BTHD, matching native efficient SDPA output.
    return rectangular.permute(0, 2, 1, 3)


class _PackQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, plan):
        ctx.plan = plan
        ctx.set_materialize_grads(False)
        return _pack_qkv(q, k, v, plan)

    @staticmethod
    def backward(ctx, dq, dk, dv):
        if torch.is_grad_enabled():
            raise RuntimeError("varlen packing currently supports first-order training only")
        gradients = tuple(value if needed else None
                          for value, needed in zip((dq, dk, dv), ctx.needs_input_grad[:3]))
        present = next((value for value in gradients if value is not None), None)
        if present is not None:
            _check_plan(ctx.plan, present.device)
        return (*_scatter_qkv(gradients, ctx.plan), None)


def _native_forward(q, k, v, cumulative, maximum, scale):
    return torch.ops.aten._efficient_attention_forward.default(
        q, k, v, None, cumulative, cumulative, maximum, maximum, 0.0,
        _CAUSAL_FROM_TOP_LEFT, True, scale=scale)


def _native_backward(grad, q, k, v, output, cumulative, maximum_q, maximum_k,
                     logsumexp, seed, offset, scale):
    return torch.ops.aten._efficient_attention_backward.default(
        grad, q, k, v, None, output, cumulative, cumulative,
        maximum_q, maximum_k, logsumexp, 0.0, seed, offset,
        _CAUSAL_FROM_TOP_LEFT, False, scale=scale, shared_storage_dqdkdv=False)


class _PackedNativeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cumulative, maximum, scale):
        with torch.autocast("cuda", enabled=False):
            output, lse, seed, offset, maximum_q, maximum_k = _native_forward(
                q, k, v, cumulative, maximum, scale)
        ctx.save_for_backward(q, k, v, output, lse, seed, offset, cumulative)
        ctx.maximum_q, ctx.maximum_k, ctx.scale = maximum_q, maximum_k, scale
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return (None,) * 6
        if torch.is_grad_enabled():
            raise RuntimeError("native varlen attention supports first-order training only")
        q, k, v, output, lse, seed, offset, cumulative = ctx.saved_tensors
        grad = grad.resolve_conj().resolve_neg().contiguous()
        with torch.autocast("cuda", enabled=False):
            dq, dk, dv, _ = _native_backward(
                grad, q, k, v, output, cumulative, ctx.maximum_q, ctx.maximum_k,
                lse, seed, offset, ctx.scale)
        return dq, dk, dv, None, None, None


class _ScatterOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, plan):
        ctx.plan = plan
        ctx.set_materialize_grads(False)
        return _scatter_output(output, plan)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None
        if torch.is_grad_enabled():
            raise RuntimeError("varlen scattering currently supports first-order training only")
        _check_plan(ctx.plan, grad.device)
        return _gather_output_gradient(grad, ctx.plan), None


def _check_plan(plan, device):
    if type(plan) is not VarlenPlan:
        raise ValueError("use make_varlen_plan for this forward's CPU mask snapshot")
    tensors = (plan.rows, plan.inverse, plan.cumulative)
    expected_shapes = ((plan.total,), (plan.batch * plan.sequence,), (plan.batch + 1,))
    allowed_dtypes = ((torch.int32, torch.int64), (torch.int32, torch.int64), (torch.int32,))
    if (plan.batch < 1 or plan.sequence < 1 or len(plan.lengths) != plan.batch
            or min(plan.lengths) < 1 or max(plan.lengths) > plan.sequence
            or plan.total != sum(plan.lengths) or plan.maximum != max(plan.lengths)
            or any(type(tensor) is not torch.Tensor or tensor.device != device
                   or tensor.dtype not in dtypes or tuple(tensor.shape) != shape
                   or not tensor.is_contiguous() or tensor.requires_grad
                   for tensor, shape, dtypes in zip(tensors, expected_shapes, allowed_dtypes))
            or tuple(tensor._version for tensor in tensors) != plan.versions):
        raise ValueError("invalid, mutated or wrong-device variable-length plan")


def varlen_attention(query, key, value, plan, *, scale=None,
                     private_training=False, finite_training=False):
    """Return private FP16 BHTD attention with zero-filled padded queries.

    All Q/K/V heads must already be repeated using the original FP32 graph.
    Only ordinary first-order training on Torch2.11, physical SM75 and native
    FP16 autocast is admitted. The two explicit caller assertions describe
    semantic preconditions; they are not inferred by scanning tensor values.
    There is no causal/dropout/bias option: this is self-attention, causal
    within each original nonempty sequence, with native attention dropout0.
    """
    if private_training is not True or finite_training is not True:
        raise ValueError("requires an explicitly private, finite training path")
    if (torch.__version__.split("+")[0] != "2.11.0" or torch.version.hip is not None
            or os.environ.get("TRITON_INTERPRET") == "1"
            or not torch.is_grad_enabled() or torch.is_inference_mode_enabled()
            or not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.float16):
        raise RuntimeError("requires Torch2.11 physical CUDA FP16-autocast first-order training")
    tensors = (query, key, value)
    if (any(type(tensor) is not torch.Tensor or tensor.device.type != "cuda"
            or tensor.dtype != torch.float32 or tensor.ndim != 4
            or tensor.stride(-1) != 1 or min(tensor.stride()) < 1
            or tensor.is_conj() or tensor.is_neg() for tensor in tensors)
            or not any(tensor.requires_grad for tensor in tensors)):
        raise ValueError("requires native FP32 post-repeat Q/K/V with last dimension contiguous")
    if torch.cuda.get_device_capability(query.device) != (7, 5):
        raise RuntimeError("this experimental native path is guarded to SM75 T4")
    _check_plan(plan, query.device)
    if (key.shape != query.shape or value.shape != query.shape
            or key.device != query.device or value.device != query.device
            or tuple(query.shape[::2]) != (plan.batch, plan.sequence)
            or min(query.shape) < 1 or query.shape[-1] % 8 or query.shape[-1] > 128
            or query.numel() > torch.iinfo(torch.int32).max):
        raise ValueError("Q/K/V must share the plan's B,T and expanded H,D with D<=128 divisible by8")
    if scale is not None and (type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0):
        raise ValueError("scale must be a finite positive number or None")
    packed = _PackQKV.apply(query, key, value, plan)
    output = _PackedNativeAttention.apply(*packed, plan.cumulative, plan.maximum, scale)
    return _ScatterOutput.apply(output, plan)
