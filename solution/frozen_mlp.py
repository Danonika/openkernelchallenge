"""Original compact frozen NF4 MLP, saving only up + existing FP16 gate input.

Idea-level study of public Soup research, implemented independently. The
unchanged starter SwiGLU is *not* our own countable Triton kernel. The frozen
gate is recomputed once in backward (same NF4 linear), while original gate/up
autograd graphs retain correct independently rounded input gradients.

Supports FP16 (direct-output RMSNorm) and FP32 (older normalized residual)
training without touching Q/V LoRA, model parameters or optimizer.
"""
from functools import wraps
from importlib.metadata import version

import torch
import triton

from .swiglu import BLOCK, example_swiglu_fwd_kernel, example_swiglu_bwd_kernel

_COUNTS = {'recomputed_gate': 0, 'compact_forward': 0, 'fallback': 0}


class _SaveUp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, gate_input_half, gate_projection, output_dtype):
        output = torch.empty_like(gate, dtype=output_dtype)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate, up, output, n, BLOCK=BLOCK)
        ctx.save_for_backward(up, gate_input_half)
        ctx.module = gate_projection
        ctx.n = n
        return output

    @staticmethod
    def backward(ctx, upstream):
        if upstream is None:
            return None, None, None, None, None
        up, gate_input_half = ctx.saved_tensors
        # Frozen gate: recomputing under no_grad creates no extra graph nor
        # retained MLP activations. The original gate projection autograd graph
        # receives the returned dgate below, just as in native SwiGLU.
        with torch.no_grad():
            gate = ctx.module(gate_input_half)
        if gate.shape != up.shape or gate.dtype != torch.float16:
            raise RuntimeError('Recomputed frozen NF4 gate violated FP16 contract')
        grad = upstream.float().contiguous()
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        example_swiglu_bwd_kernel[(triton.cdiv(ctx.n, BLOCK),)](
            grad, gate.contiguous(), up, dgate, dup, ctx.n, BLOCK=BLOCK)
        _COUNTS['recomputed_gate'] += 1
        return dgate, dup, None, None, None


def _supported(module, x, linear_class):
    if (not isinstance(x, torch.Tensor) or not x.is_cuda
            or x.dtype not in (torch.float16, torch.float32)
            or not x.is_contiguous() or not x.requires_grad
            or x.ndim not in (2, 3) or x.numel() == 0
            or not module.training or not torch.is_grad_enabled()
            or getattr(module.config, 'hidden_act', 'silu') != 'silu'):
        return False
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        layer = getattr(module, name, None)
        if (type(layer) is not linear_class or 'forward' in layer.__dict__
                or getattr(layer, '_hf_hook', None) is not None
                or layer.bias is not None or layer.weight.requires_grad
                or layer.compute_dtype != torch.float16
                or not layer.weight.is_cuda or layer.weight.device != x.device
                or layer.weight.dtype != torch.uint8
                or layer._forward_hooks or layer._forward_pre_hooks
                or layer._backward_hooks or layer._backward_pre_hooks):
            return False
        state = getattr(layer.weight, 'quant_state', None)
        if state is None or state.quant_type != 'nf4' or state.dtype != torch.float16:
            return False
    return True


def patch_frozen_mlp():
    """Install after the starter and existing compact activation patch."""
    if version('bitsandbytes') != '0.50.2':
        return False
    import bitsandbytes as bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    previous = Qwen2MLP.forward
    if getattr(previous, '_soup_independent_mlp', False):
        return True
    linear_class = bnb.nn.Linear4bit

    @wraps(previous)
    def forward(self, x):
        if not _supported(self, x, linear_class):
            _COUNTS['fallback'] += 1
            return previous(self, x)
        # Gate and up FP32->FP16 casts are intentionally separate: their
        # individual autograd branches must round before summing in FP32.
        gate_input = x.to(torch.float16)
        up_input = x.to(torch.float16)
        gate = self.gate_proj(gate_input)
        up = self.up_proj(up_input)
        if (gate.dtype != torch.float16 or up.dtype != torch.float16
                or gate.shape != up.shape or not gate.is_contiguous()
                or not up.is_contiguous()):
            raise RuntimeError('Pinned frozen MLP GEMMs did not return FP16')
        # Our FP32 residual branch's compact SwiGLU output is FP32; under
        # direct FP16 normalized input the starter SwiGLU returns FP16.
        result_dtype = torch.float16 if x.dtype == torch.float16 else torch.float32
        activated = _SaveUp.apply(gate, up, gate_input, self.gate_proj, result_dtype)
        _COUNTS['compact_forward'] += 1
        return self.down_proj(activated)

    forward._soup_independent_mlp = True
    Qwen2MLP.forward = forward
    return True


def counters():
    return dict(_COUNTS)
