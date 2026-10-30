"""Lossless storage for the fixed task's frozen NF4 MLP outputs.

Original integration. The arithmetic kernels imported from ``swiglu`` are the
official starter example, kept under their original ``example_`` names. Neither
the example nor this wrapper claims own-kernel credit.

Source of the representation argument: bitsandbytes 0.50.2 Linear4bit.forward
computes a bias-free projection in its explicit FP16 compute dtype, then widens
the output to its FP32 input dtype. Such outputs can be retained in FP16 without
losing their effective values. The original projections and their autograd
boundaries stay intact.
"""

import torch
import triton

from .swiglu import (
    BLOCK,
    example_swiglu_bwd_kernel,
    example_swiglu_fwd_kernel,
)


class HalfSavedSwiGLU(torch.autograd.Function):
    """Starter arithmetic, half saved operands, explicit FP32 boundaries.

Call only on verified half-origin FP32 gate/up values. In particular, this is
not a general-purpose compression operation for arbitrary FP32 activations.
"""

    @staticmethod
    def forward(ctx, gate, up):
        # Conversion is lossless under the module guard below. No original
        # FP32 activation tensor is retained in the context or a closure.
        gate_half = gate.to(dtype=torch.float16).contiguous()
        up_half = up.to(dtype=torch.float16).contiguous()
        out = torch.empty(gate.shape, dtype=torch.float32, device=gate.device)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate_half, up_half, out, n, BLOCK=BLOCK
        )
        ctx.save_for_backward(gate_half, up_half)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate_half, up_half = ctx.saved_tensors
        # The unchanged starter kernel widens both half operands in registers.
        # Keep the incoming gradient and both destinations FP32: using the
        # saved operand dtype for these buffers would move a rounding boundary.
        grad = grad.to(dtype=torch.float32).contiguous()
        dgate = torch.empty(gate_half.shape, dtype=torch.float32, device=gate_half.device)
        dup = torch.empty_like(dgate)
        n = gate_half.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            grad, gate_half, up_half, dgate, dup, n, BLOCK=BLOCK
        )
        return dgate, dup


def _plain_frozen_half_nf4(projection, linear4bit_type):
    """Check module provenance; avoid device synchronization/value scans."""
    if not isinstance(projection, linear4bit_type):
        return False
    # Exclude subclasses that change the projection's numerical computation.
    if type(projection).forward is not linear4bit_type.forward:
        return False
    # Hooks or an instance override can change the value/dtype returned by
    # Linear4bit after its FP16 computation, invalidating half-origin storage.
    # PEFT's attention adapters and embedding hooks are not inspected here.
    if "forward" in vars(projection):
        return False
    if projection._forward_hooks or projection._forward_pre_hooks:
        return False
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.requires_grad:
        return False
    if weight.device.type != "cuda" or getattr(projection, "bias", None) is not None:
        return False
    if getattr(projection, "compute_dtype", None) != torch.float16:
        return False
    if not getattr(projection, "compute_type_is_set", False):
        return False
    state = getattr(weight, "quant_state", None)
    if state is None or getattr(state, "quant_type", None) != "nf4":
        return False
    return True


def _eligible_mlp(module, x, linear4bit_type):
    if not isinstance(x, torch.Tensor) or x.device.type != "cuda":
        return False
    if x.dtype != torch.float32 or x.ndim < 2 or x.numel() == 0:
        return False
    if "forward" in vars(module):
        return False
    config = getattr(module, "config", None)
    if getattr(config, "hidden_act", None) != "silu":
        return False
    if getattr(module, "hidden_size", None) != 1536:
        return False
    if getattr(module, "intermediate_size", None) != 8960:
        return False
    if x.shape[-1] != 1536:
        return False
    activation = getattr(module, "act_fn", None)
    activation_type = torch.nn.SiLU
    if not isinstance(activation, torch.nn.SiLU):
        # In Transformers 5.16.1 the "silu" entry is SiLUActivation rather
        # than nn.SiLU; the "swish" alias still uses nn.SiLU.
        try:
            from transformers.activations import SiLUActivation
        except (ImportError, AttributeError):
            return False
        if not isinstance(activation, SiLUActivation):
            return False
        activation_type = SiLUActivation
    # The fused activation replaces this module call, so preserve arbitrary
    # user hooks/overrides by falling back instead of silently bypassing them.
    if type(activation).forward is not activation_type.forward:
        return False
    if "forward" in vars(activation) or getattr(activation, "inplace", False):
        return False
    if any(getattr(activation, name, None) for name in (
        "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"
    )):
        return False
    return all(
        _plain_frozen_half_nf4(getattr(module, name, None), linear4bit_type)
        for name in ("gate_proj", "up_proj", "down_proj")
    )


def _packed_mlp_forward(module, x):
    # The two original projections still own separate FP32->FP16 casts and
    # separate widened input gradients. Do not hoist x.half() across them.
    gate = module.gate_proj(x)
    up = module.up_proj(x)
    return module.down_proj(HalfSavedSwiGLU.apply(gate, up))


def install():
    """Patch only eligible real Qwen2 MLPs; safe to call more than once."""
    try:
        from bitsandbytes.nn import Linear4bit
        from transformers.models.qwen2 import modeling_qwen2
    except (ImportError, AttributeError):
        return False

    cls = getattr(modeling_qwen2, "Qwen2MLP", None)
    if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
        return False
    original = getattr(cls, "forward", None)
    if original is None:
        return False
    if getattr(original, "_soup_half_saved_mlp", False):
        return True

    def forward(self, x):
        if not _eligible_mlp(self, x, Linear4bit):
            return original(self, x)
        return _packed_mlp_forward(self, x)

    forward._soup_half_saved_mlp = True
    forward._soup_original_forward = original
    cls.forward = forward
    return True
