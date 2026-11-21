"""Original packed frozen-MLP rematerialization for the Soup T4 experiment.

This module replaces the *team's own* input-only replay after it is installed.
It uses the existing per-layer qgroup.span() view to compute gate+up in a
one unchanged reference forward and a new grouped backward GEMM. During
backward it reconstructs both frozen projections once, evaluates both
derivatives in a single kernel, and computes the input gradient in one GEMM per bounded row tile.

The packed matrices already live in qgroup's decoded per-pass buffer; no
global fp16 weight cache, no changed parameter, no saved wide activations.
A fixed 768-token backward tile bounds the transient FP16 [rows, 2*8960]\nscratch tensor while retaining the same original NF4 group. This trades a few\nextra GEMM launches only for longer sequences and targets real peak memory.\nThe runtime TOTAL scalar avoids recompiling the original VJP kernel for
each different number of packed tokens. BLOCK and WIDTH remain compile-time
constants; no prebuilt cache is used.
Split/dequant boundary layers retain the tested input-only replay.
Idea references: grouped linear algebra, QLoRA, SiLU derivative. The source
was independently authored. The surrounding external contributor research control is
third-party and cannot be submitted as original code.
"""
from functools import wraps

import torch
import triton
import triton.language as tl

_DIM = 8960
_INPUT = 1536
_GATE = 4
_STOP = 6
_TILE = 512
_VJP_ROW_CAP = 768  # fixed shape-only scratch bound; no runtime allocation monitoring


@triton.jit
def soup_packed_gated_activation(
    PREACT, OUTPUT, TOTAL: tl.constexpr, WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = linear < TOTAL
    token = linear // WIDTH
    feature = linear % WIDTH
    offset = token * (2 * WIDTH) + feature
    gate = tl.load(PREACT + offset, keep, other=0).to(tl.float32)
    up = tl.load(PREACT + offset + WIDTH, keep, other=0).to(tl.float32)
    value = (gate * tl.sigmoid(gate)) * up
    tl.store(OUTPUT + linear, value, keep)


@triton.jit(do_not_specialize=["TOTAL"])
def soup_packed_gated_input_vjp(
    PREACT, DACT, TOTAL, WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = linear < TOTAL
    token = linear // WIDTH
    feature = linear % WIDTH
    offset = token * (2 * WIDTH) + feature
    gate = tl.load(PREACT + offset, keep, other=0).to(tl.float32)
    up = tl.load(PREACT + offset + WIDTH, keep, other=0).to(tl.float32)
    grad = tl.load(DACT + linear, keep, other=0).to(tl.float32)
    sig = tl.sigmoid(gate)
    # Same mathematical SiLU/GLU derivative as the native op. This node
    # writes both halves in-place, directly matching the stacked weight GEMM.
    dg = ((grad * up) * sig) * (1.0 + gate * (1.0 - sig))
    du = (grad * gate) * sig
    tl.store(PREACT + offset, dg, keep)
    tl.store(PREACT + offset + WIDTH, du, keep)


class _SingleGemmReplay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        in_width = x.shape[-1]
        if in_width != _INPUT or x.dtype != torch.float16:
            raise ValueError("Unsupported packed grouped MLP input")
        # Preserve the exact already-measured forward's separate gate/up
        # GEMM and starter SwiGLU, and optimize only the input VJP GEMM.
        from .swiglu import BLOCK, example_swiglu_fwd_kernel
        wg = group.weight("fwd", _GATE)
        gate = torch.nn.functional.linear(x, wg, group.biases[_GATE]).contiguous()
        del wg
        wu = group.weight("fwd", _GATE + 1)
        up = torch.nn.functional.linear(x, wu, group.biases[_GATE + 1]).contiguous()
        del wu
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate, up, gate, n, BLOCK=BLOCK,
        )
        ctx.group = group
        ctx.input_shape = tuple(x.shape)
        ctx.save_for_backward(x)
        ctx.set_materialize_grads(False)
        return gate

    @staticmethod
    def backward(ctx, dact):
        if dact is None:
            return None, None
        (x,) = ctx.saved_tensors
        group = ctx.group
        # Fetch the two adjacent frozen weights once per backward pass;
        # qgroup owns their lifetime, unchanged from the measured parent.
        weights = group.span("bwd", _GATE, _STOP)
        if weights is None or tuple(weights.shape) != (2 * _DIM, _INPUT):
            raise RuntimeError("Grouped backward span unavailable")
        bias = group.span_bias(_GATE, _STOP)
        rows = x.numel() // _INPUT
        if dact.numel() != rows * _DIM:
            raise RuntimeError("Frozen MLP gradient geometry changed")
        xflat = x.reshape(rows, _INPUT)
        gradflat = dact.reshape(rows, _DIM)
        dx = torch.empty((rows, _INPUT), device=x.device, dtype=x.dtype)
        # Explicit bounded buffer lifetime: no persistent/reusable cache,
        # no hidden-row or memory-monitor branch. Only actual tensor shape.
        # For ordinary shorter batches the existing single GEMM remains.
        for left in range(0, rows, _VJP_ROW_CAP):
            right = min(left + _VJP_ROW_CAP, rows)
            with torch.autocast(device_type="cuda", enabled=False):
                scratch = torch.nn.functional.linear(
                    xflat[left:right], weights, bias
                )
            cotangent = gradflat[left:right].to(scratch.dtype).contiguous()
            soup_packed_gated_input_vjp[(triton.cdiv(cotangent.numel(), _TILE),)](
                scratch, cotangent, TOTAL=cotangent.numel(),
                WIDTH=_DIM, BLOCK=_TILE, num_warps=4,
                enable_fp_fusion=False,
            )
            with torch.autocast(device_type="cuda", enabled=False):
                torch.mm(scratch, weights, out=dx[left:right])
            del scratch, cotangent
        return dx.reshape(ctx.input_shape), None


def install_packed_mlp_replay():
    from transformers.models.qwen2 import modeling_qwen2
    from .mlp_save_up import save_up_group
    from .state import DISABLED

    layer = modeling_qwen2.Qwen2MLP
    old = layer.forward
    if getattr(old, "_soup_packed_mlp_replay", False):
        return True

    @wraps(old)
    def forward(self, x):
        if (not torch.is_grad_enabled()
                or not self.training
                or not isinstance(x, torch.Tensor)
                or not x.is_cuda
                or x.dtype != torch.float16
                or not x.requires_grad
                or not x.is_contiguous()
                or x.shape[-1] != _INPUT
                or "qgroup" in DISABLED):
            return old(self, x)
        group = save_up_group(self, x)
        if (group is None or getattr(group, "checkpointed", False)
                or group.split_now("fwd") or group.split_now("bwd")
                or tuple(group.shapes[_GATE]) != (_DIM, _INPUT)
                or tuple(group.shapes[_GATE + 1]) != (_DIM, _INPUT)):
            return old(self, x)
        # The installed parent input-only path uses exactly two additional
        # grouped backward accesses for the gate/up projections.
        hidden = _SingleGemmReplay.apply(x, group)
        group.expected_bwd += 2
        return self.down_proj(hidden)

    forward._soup_packed_mlp_replay = True
    forward._previous = old
    layer.forward = forward
    return True
