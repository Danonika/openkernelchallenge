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
        and getattr(base, "compute_dtype", None) == torch.float16
        and (bias is None or (not bias.requires_grad and bias.dtype in (torch.float16, torch.float32)))
        and not base._forward_hooks and not base._forward_pre_hooks
        and not hasattr(base, "_hf_hook")
    )


def _decode(weight):
    import bitsandbytes.functional as bnbf
    dense = bnbf.dequantize_4bit(weight.data, weight.quant_state)
    return dense.to(torch.float16).contiguous()


class _FrozenQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias):
        weights = [q_weight, k_weight, v_weight]
        full = torch.cat([_decode(w) for w in weights], dim=0)
        sizes = [int(w.quant_state.shape[0]) for w in weights]
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
        weight = torch.cat([_decode(w) for w in ctx.weights], dim=0)
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
