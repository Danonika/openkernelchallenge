"""Optional lossless storage of frozen NF4 MLP intermediates for backward.

The pinned bitsandbytes Linear4bit consumes its configured FP16 compute input,
returns FP16 GEMM output, and casts that output back to the incoming FP32 dtype.
For ordinary frozen Qwen2 gate/up projections this proves their FP32 outputs
are exactly FP16-representable. Passing two separately cast FP16 inputs keeps
those outputs in FP16 and avoids the round trip through global FP32 storage.
The starter kernels still perform activation arithmetic in FP32 registers;
the activation output is FP32, and each projection gradient is rounded to
FP16 at the same point as in the original Linear4bit casting graph.

Both Triton functions used here are the UNCHANGED, CREDITED STARTER SWIGLU
KERNELS. This storage optimization introduces NO original/countable kernel.
In particular, a new mixed-pointer backward specialization is still the
starter example and must not count toward the 5% original-kernel threshold.
"""

from functools import wraps
from importlib import metadata

import torch
import triton

from .swiglu import (
    BLOCK,
    example_swiglu_fwd_kernel,
    example_swiglu_bwd_kernel,
    swiglu,
)


class _CompactSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate, up = gate.contiguous(), up.contiguous()
        ctx.input_dtype = gate.dtype
        # Even when the projection outputs remain FP16, the original FP32
        # SwiGLU result must not be rounded before the down projection's cast.
        output = torch.empty_like(gate, dtype=torch.float32)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate, up, output, n, BLOCK=BLOCK,
        )
        # FP16 inputs are saved by reference. For the compatibility FP32 path,
        # these casts are lossless only under the checked projection dataflow.
        # No original FP32 gate/up tensor is retained by this autograd context.
        ctx.save_for_backward(gate.to(torch.float16), up.to(torch.float16))
        return output

    @staticmethod
    def backward(ctx, grad):
        gate_half, up_half = ctx.saved_tensors
        # Preserve the original FP32 upstream gradient. Casting to the saved
        # tensor dtype here would round before differentiation and be wrong.
        grad = grad.to(torch.float32).contiguous()
        dgate = torch.empty_like(gate_half, dtype=ctx.input_dtype)
        dup = torch.empty_like(up_half, dtype=ctx.input_dtype)
        n = gate_half.numel()
        # The credited kernel evaluates each derivative in FP32 registers.
        # With FP16 projection outputs, storing that derivative as FP16 fuses
        # the original output-upcast's backward conversion. For FP32 inputs,
        # retain FP32 derivative outputs as the earlier storage wrapper did.
        # Neither mode reconstructs global FP32 gate/up activations.
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            grad, gate_half, up_half, dgate, dup, n, BLOCK=BLOCK,
        )
        return dgate, dup


def compact_swiglu(gate, up):
    """Internal primitive: matching FP16 inputs or exact FP16 values in FP32.

    Always returns an FP32 activation. The production patch uses FP16 input
    tensors directly. The compatibility FP32 path does not scan values;
    arbitrary FP32 inputs violate its exact-representability precondition.
    """
    if (gate.dtype not in (torch.float16, torch.float32) or up.dtype != gate.dtype
            or gate.shape != up.shape or gate.device != up.device
            or gate.numel() == 0):
        raise ValueError("requires matching nonempty FP16 values or their exact FP32 representations")
    return _CompactSwiGLU.apply(gate, up)


def _is_cuda_tensor(tensor):
    return tensor.is_cuda


def _eligible_mlp(module, x, linear_class, linear_forward):
    if (not isinstance(x, torch.Tensor) or not _is_cuda_tensor(x)
            or x.dtype != torch.float32 or x.ndim < 2 or not x.is_contiguous()
            or not x.numel() or not x.requires_grad or not module.training or not torch.is_grad_enabled()
            or getattr(getattr(module, "config", None), "hidden_act", None) != "silu"):
        return False
    # Refuse output-changing hooks or overrides: the representability proof
    # depends on the documented, unmodified Linear4bit forward formula.
    hooks = torch.nn.modules.module
    if (hooks._global_forward_hooks or hooks._global_forward_pre_hooks
            or hooks._global_backward_hooks or hooks._global_backward_pre_hooks):
        return False
    projections = [getattr(module, name, None) for name in ("gate_proj", "up_proj", "down_proj")]
    for layer in projections:
        if (type(layer) is not linear_class or type(layer).forward is not linear_forward
                or "forward" in layer.__dict__ or hasattr(layer, "_hf_hook")
                or layer._forward_hooks or layer._forward_pre_hooks
                or layer._backward_hooks or layer._backward_pre_hooks
                or layer.compute_dtype != torch.float16 or not layer.compute_type_is_set
                or layer.bias is not None or layer.weight.requires_grad
                or layer.weight.dtype != torch.uint8
                or not _is_cuda_tensor(layer.weight) or layer.weight.device != x.device):
            return False
        state = getattr(layer.weight, "quant_state", None)
        if (state is None or state.quant_type != "nf4" or state.dtype != torch.float16
                or tuple(state.shape) != (layer.out_features, layer.in_features)):
            return False
    gate, up, down = projections
    return (gate.in_features == up.in_features == x.shape[-1]
            and gate.out_features == up.out_features == down.in_features
            and down.out_features == x.shape[-1])


def patch_compact_mlp_activations():
    """Install after the starter MLP patch; unsupported calls keep that path."""
    if metadata.version("bitsandbytes") != "0.50.2" or metadata.version("transformers") != "5.16.1":
        return False
    import bitsandbytes as bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    original = Qwen2MLP.forward
    if getattr(original, "_soup_candidate_compact_activations", False):
        return True
    linear_class = bnb.nn.Linear4bit
    linear_forward = linear_class.forward

    @wraps(original)
    def forward(self, x):
        if not _eligible_mlp(self, x, linear_class, linear_forward):
            return original(self, x)
        # Keep these casts SEPARATE. Sharing one FP16 input would sum the two
        # input-gradient branches in FP16 before upcasting to FP32, unlike the
        # original graph, which upcasts each branch first and then adds them.
        gate = self.gate_proj(x.to(torch.float16))
        up = self.up_proj(x.to(torch.float16))
        if (gate.dtype == up.dtype == torch.float16 and gate.shape == up.shape
                and gate.device == up.device and gate.numel()):
            activation = compact_swiglu(gate, up)
        else:
            # Unmodified guarded projections produce FP16. If that contract
            # is violated, do not rerun projections; preserve FP32 arithmetic.
            activation = swiglu(gate.float(), up.float())
        return self.down_proj(activation)

    forward._soup_candidate_compact_activations = True
    Qwen2MLP.forward = forward
    return True
