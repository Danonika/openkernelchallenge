"""Hero1 original grouped frozen q|k|v base-projection fusion for Soup Qwen2.

Idea credit: an external contributor fused q/k/v GEMM with packed frozen weights. Our
implementation is independent; no third-party kernel or source is copied.
PEFT q/v LoRA forward, dropout RNG, adapters, parameter ownership and
optimizer remain native. This replaces only frozen bitsandbytes base matmuls.

The q/k/v weights are decoded with the team's existing original NF4 patch;
the three dense weights are concatenated only in this one invocation, not
retained across layers or microbatches. Backward reconstructs the dense
weights and differentiates the shared input. This is a T4 source-reviewed optimization; full task and numerical parity must be measured.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import torch
import triton
import triton.language as tl


_CURRENT = ContextVar("_hero1_fused_qkv", default=None)


def _base(module):
    return getattr(module, "base_layer", module)


def _supported_base(module, dtype):
    import bitsandbytes as bnb
    base = _base(module)
    weight = getattr(base, "weight", None)
    state = getattr(weight, "quant_state", None)
    bias = getattr(base, "bias", None)
    return (
        type(base) is bnb.nn.Linear4bit
        and weight is not None and weight.dtype == torch.uint8
        and not weight.requires_grad
        and state is not None and getattr(state, "quant_type", None) == "nf4"
        and getattr(state, "dtype", None) == torch.float16
        and _components(weight) is not None
        and getattr(base, "compute_dtype", None) == torch.float16
        and (bias is None or (not bias.requires_grad and bias.dtype in (torch.float16, torch.float32)))
        and not base._forward_hooks and not base._forward_pre_hooks
        and not hasattr(base, "_hf_hook")
    )


QKV_DECODE_BLOCK = 1024


@triton.jit
def soup_qkv_nf4_dequantize(
    Q_PACKED, Q_SCALES, Q_NESTED_SCALES, Q_NESTED_CODE, Q_OFFSET,
    K_PACKED, K_SCALES, K_NESTED_SCALES, K_NESTED_CODE, K_OFFSET,
    V_PACKED, V_SCALES, V_NESTED_SCALES, V_NESTED_CODE, V_OFFSET,
    NF4_CODE, OUTPUT, Q_BYTES, K_BYTES, TOTAL_BYTES,
    QUANT_BLOCK: tl.constexpr, NESTED_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Decode canonical Q/K/V nested-NF4 buffers in one launch."""
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = byte < TOTAL_BYTES
    q_mask = valid & (byte < Q_BYTES)
    k_mask = valid & (byte >= Q_BYTES) & (byte < Q_BYTES + K_BYTES)
    v_mask = valid & (byte >= Q_BYTES + K_BYTES)

    q_byte = byte
    k_byte = byte - Q_BYTES
    v_byte = byte - Q_BYTES - K_BYTES

    q_packed = tl.load(Q_PACKED + q_byte, mask=q_mask, other=0).to(tl.int32)
    k_packed = tl.load(K_PACKED + k_byte, mask=k_mask, other=0).to(tl.int32)
    v_packed = tl.load(V_PACKED + v_byte, mask=v_mask, other=0).to(tl.int32)

    q_scale_i = (2 * q_byte) // QUANT_BLOCK
    k_scale_i = (2 * k_byte) // QUANT_BLOCK
    v_scale_i = (2 * v_byte) // QUANT_BLOCK

    q_scale_code = tl.load(
        Q_NESTED_CODE + tl.load(Q_SCALES + q_scale_i, mask=q_mask, other=0).to(tl.int32),
        mask=q_mask, other=0.0,
    )
    k_scale_code = tl.load(
        K_NESTED_CODE + tl.load(K_SCALES + k_scale_i, mask=k_mask, other=0).to(tl.int32),
        mask=k_mask, other=0.0,
    )
    v_scale_code = tl.load(
        V_NESTED_CODE + tl.load(V_SCALES + v_scale_i, mask=v_mask, other=0).to(tl.int32),
        mask=v_mask, other=0.0,
    )

    q_scale = (
        q_scale_code
        * tl.load(Q_NESTED_SCALES + q_scale_i // NESTED_BLOCK, mask=q_mask, other=0.0)
        + tl.load(Q_OFFSET, mask=q_mask, other=0.0)
    )
    k_scale = (
        k_scale_code
        * tl.load(K_NESTED_SCALES + k_scale_i // NESTED_BLOCK, mask=k_mask, other=0.0)
        + tl.load(K_OFFSET, mask=k_mask, other=0.0)
    )
    v_scale = (
        v_scale_code
        * tl.load(V_NESTED_SCALES + v_scale_i // NESTED_BLOCK, mask=v_mask, other=0.0)
        + tl.load(V_OFFSET, mask=v_mask, other=0.0)
    )

    q_hi = tl.load(NF4_CODE + (q_packed >> 4), mask=q_mask, other=0.0) * q_scale
    q_lo = tl.load(NF4_CODE + (q_packed & 15), mask=q_mask, other=0.0) * q_scale
    k_hi = tl.load(NF4_CODE + (k_packed >> 4), mask=k_mask, other=0.0) * k_scale
    k_lo = tl.load(NF4_CODE + (k_packed & 15), mask=k_mask, other=0.0) * k_scale
    v_hi = tl.load(NF4_CODE + (v_packed >> 4), mask=v_mask, other=0.0) * v_scale
    v_lo = tl.load(NF4_CODE + (v_packed & 15), mask=v_mask, other=0.0) * v_scale

    hi = tl.where(q_mask, q_hi, tl.where(k_mask, k_hi, v_hi))
    lo = tl.where(q_mask, q_lo, tl.where(k_mask, k_lo, v_lo))
    hi_bits = hi.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    lo_bits = lo.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    tl.store(
        OUTPUT.to(tl.pointer_type(tl.uint32)) + byte,
        hi_bits | (lo_bits << 16),
        mask=valid,
    )


def _components(weight):
    from .nf4 import _state_components
    if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.uint8
            or weight.ndim != 2 or weight.shape[1] != 1
            or not weight.is_cuda or not weight.is_contiguous()):
        return None
    components = _state_components(
        weight.data, getattr(weight, "quant_state", None), None, cuda_only=True
    )
    if components is None:
        return None
    if (components["dtype"] != torch.float16
            or components["blocksize"] != 64
            or components["nested_scales"] is None
            or components["nested_blocksize"] != 256):
        return None
    return components


def _decode_three(weights):
    from .nf4 import _nf4_table
    components = [_components(weight) for weight in weights]
    if any(c is None for c in components):
        raise RuntimeError("Grouped QKV received unsupported NF4 state")

    widths = [int(c["shape"][1]) for c in components]
    if len(set(widths)) != 1:
        raise RuntimeError("Grouped QKV input widths differ")
    sizes = [int(c["shape"][0]) * int(c["shape"][1]) for c in components]
    if any(size % 2 for size in sizes):
        raise RuntimeError("Grouped QKV requires even packed matrices")
    byte_counts = [size // 2 for size in sizes]
    total_bytes = sum(byte_counts)

    output = torch.empty(sum(sizes), device=weights[0].device, dtype=torch.float16)
    q, k, v = components
    soup_qkv_nf4_dequantize[(triton.cdiv(total_bytes, QKV_DECODE_BLOCK),)](
        q["packed"], q["scales"], q["nested_scales"], q["nested_code"], q["offset"],
        k["packed"], k["scales"], k["nested_scales"], k["nested_code"], k["offset"],
        v["packed"], v["scales"], v["nested_scales"], v["nested_code"], v["offset"],
        _nf4_table(weights[0].device), output,
        byte_counts[0], byte_counts[1], total_bytes,
        QUANT_BLOCK=64, NESTED_BLOCK=256, BLOCK=QKV_DECODE_BLOCK,
        num_warps=4, enable_fp_fusion=False,
    )
    rows = [int(c["shape"][0]) for c in components]
    return output.view(sum(rows), widths[0]), rows


class _FrozenQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias):
        weights = [q_weight, k_weight, v_weight]
        full, sizes = _decode_three(weights)
        width = x.shape[-1]
        xh = x.reshape(-1, width).to(torch.float16)
        with torch.autocast("cuda", enabled=False):
            y = torch.mm(xh, full.t())
        offset = 0
        for size, bias in zip(sizes, (q_bias, k_bias, v_bias)):
            if bias is not None:
                y[:, offset:offset + size].add_(bias.to(torch.float16))
            offset += size
        result = y.reshape(*x.shape[:-1], sum(sizes))
        ctx.weights = weights
        ctx.sizes = sizes
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        ctx.versions = tuple((id(w), w._version) for w in weights)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        if tuple((id(w), w._version) for w in ctx.weights) != ctx.versions:
            raise RuntimeError("Frozen QKV weight version changed during backward")
        weight, sizes = _decode_three(ctx.weights)
        if sizes != ctx.sizes:
            raise RuntimeError("Grouped QKV decoded sizes changed during backward")
        with torch.autocast("cuda", enabled=False):
            dxh = torch.mm(grad_out.reshape(-1, sum(ctx.sizes)).to(torch.float16), weight)
        dx = dxh.reshape(ctx.input_shape).to(ctx.input_dtype)
        return dx, None, None, None, None, None, None


def patch_grouped_qkv():
    """Patch one native Qwen2 attention boundary and its frozen base GEMMs."""
    import bitsandbytes as bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    original_attention = Qwen2Attention.forward
    if getattr(original_attention, "_hero1_group_qkv", False):
        return True

    original_linear = bnb.nn.Linear4bit.forward

    @wraps(original_linear)
    def intercepted_base(self, x, *args, **kwargs):
        frame = _CURRENT.get()
        if frame is not None and not args and not kwargs:
            match = frame.get(id(self))
            if match is not None:
                out, pointer, expected_shape = match
                if (isinstance(x, torch.Tensor)
                        and x.data_ptr() == pointer
                        and tuple(x.shape) == expected_shape):
                    return out
        return original_linear(self, x, *args, **kwargs)

    @wraps(original_attention)
    def attention(self, hidden_states, *args, **kwargs):
        if (not isinstance(hidden_states, torch.Tensor)
                or not hidden_states.is_cuda
                or hidden_states.ndim != 3
                or not self.training or not torch.is_grad_enabled()
                or not hidden_states.requires_grad
                or hidden_states.dtype not in (torch.float16, torch.float32)
                or getattr(self.q_proj, "_soup_query_padding_plan", None) is not None
                or _CURRENT.get() is not None):
            return original_attention(self, hidden_states, *args, **kwargs)
        projections = (self.q_proj, self.k_proj, self.v_proj)
        bases = [_base(layer) for layer in projections]
        if not all(_supported_base(layer, hidden_states.dtype) for layer in projections):
            return original_attention(self, hidden_states, *args, **kwargs)
        for layer in (self.q_proj, self.v_proj):
            if not getattr(layer, "active_adapters", None):
                return original_attention(self, hidden_states, *args, **kwargs)
            if getattr(layer, "disable_adapters", False) or getattr(layer, "merged", False):
                return original_attention(self, hidden_states, *args, **kwargs)

        frozen = _FrozenQKV.apply(
            hidden_states,
            *(base.weight for base in bases),
            *(base.bias for base in bases)
        )
        widths = [base.out_features for base in bases]
        q, k, v = frozen.split(widths, dim=-1)
        table = {
            id(base): (out, hidden_states.data_ptr(), tuple(hidden_states.shape))
            for base, out in zip(bases, (q, k, v))
        }
        token = _CURRENT.set(table)
        try:
            return original_attention(self, hidden_states, *args, **kwargs)
        finally:
            _CURRENT.reset(token)

    intercepted_base._hero1_group_qkv = True
    attention._hero1_group_qkv = True
    bnb.nn.Linear4bit.forward = intercepted_base
    Qwen2Attention.forward = attention
    return True
