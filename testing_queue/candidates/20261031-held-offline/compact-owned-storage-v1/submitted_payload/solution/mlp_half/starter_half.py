"""Producer-owned half buffers using the EXACT official starter JIT functions.

No new Triton mathematics or original kernel credit. This wrapper deliberately
does NOT call starter.SwiGLU with half inputs: that wrapper allocates half output
and casts incoming dy to half. Both are incorrect at this FP32 consumer boundary.
"""
import torch
import triton
from . import starter_swiglu as starter
from .native_owner import NativeMLPHalfOwner


class OwnedStarterHalf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if (type(gate) is not torch.Tensor or type(up) is not torch.Tensor
                or gate.dtype != torch.float16 or up.dtype != torch.float16
                or gate.device != up.device or gate.shape != up.shape
                or not gate.is_contiguous() or not up.is_contiguous()):
            raise RuntimeError('Actual contiguous half producer contract required')
        out = torch.empty(gate.shape, device=gate.device, dtype=torch.float32)
        n = gate.numel()
        starter.example_swiglu_fwd_kernel[(triton.cdiv(n, starter.BLOCK),)](
            gate, up, out, n, BLOCK=starter.BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError('Explicit first-order contract violated')
        gate, up = ctx.saved_tensors  # Keep normal version checks.
        if grad.dtype != torch.float32:
            raise RuntimeError('FP32 incoming activation cotangent required')
        grad = grad.contiguous()  # NO half cast; its small/scaled values matter.
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        starter.example_swiglu_bwd_kernel[(triton.cdiv(n, starter.BLOCK),)](
            grad, gate, up, dgate, dup, n, BLOCK=starter.BLOCK)
        # Exact unchanged JIT computes FP32 then RN-stores each half branch.
        # Separate preceding source->half edges preserve FP32 fan-in upstream.
        return dgate, dup


def starter_widened_activation(gate, up):
    """Frozen official starter on independent native widening edges."""
    return starter.swiglu(gate.float(), up.float())


def starter_wide_mlp(module, x):
    """Unmodified native producer outputs + the unchanged FP32 starter wrapper."""
    gate, up = module.gate_proj(x), module.up_proj(x)
    if gate.dtype != torch.float32 or up.dtype != torch.float32:
        raise RuntimeError('Standalone reference requires actual FP32 producer return')
    return module.down_proj(starter.swiglu(gate, up))


def owned_starter_mlp(module, x):
    # Do not share these casts: shared half fan-in changes the native graph.
    gate16 = module.gate_proj(x.to(torch.float16))
    up16 = module.up_proj(x.to(torch.float16))
    return module.down_proj(OwnedStarterHalf.apply(gate16, up16))


class StarterMLPHalfOwner(NativeMLPHalfOwner):
    """All existing native producer/source/dispatch eligibility reused unchanged."""
    def __call__(self, x):
        why = self.reason(x)
        if why is not None:
            self.counts['native'] += 1
            self.reasons[why] = self.reasons.get(why, 0) + 1
            return self.module(x)
        self.counts['owned'] += 1
        return owned_starter_mlp(self.module, x)
