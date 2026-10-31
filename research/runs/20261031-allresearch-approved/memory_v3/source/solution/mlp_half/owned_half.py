"""Original CPU integration reference for SOUP-MEM-001, not a model patch.

The accepted producer boundary supplies two actual FP16 tensors. The caller
explicitly transfers non-mutating ownership and promises first-order training.
The baseline widens each branch independently before native FP32 SiLU/multiply.
This reference retains only the original half operands, and replays that same
native operation graph during backward. No starter kernel/code is used.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OwnershipContract:
    """Caller promises, not facts inferred from a tensor's values.

    owned_producer_outputs: both operands are producer-owned temporaries with
    no writes/escaping mutable aliases through the end of backward. Sharing
    read-only storage is allowed. first_order_only: no create_graph backward.
    This contract covers ordinary reverse-mode AD only. These preconditions
    must be established by a future integration owner, not inferred from
    exact value roundtrips. Opt out before forward for higher-order AD.
    """
    owned_producer_outputs: bool = False
    first_order_only: bool = False


def native_widened_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Unmodified native graph for this pre-widening producer boundary.

    A half producer is widened independently; other dtypes retain their native
    behavior. This is not a replacement definition for arbitrary model layers.
    """
    gate_wide = gate.float() if gate.dtype == torch.float16 else gate
    up_wide = up.float() if up.dtype == torch.float16 else up
    return F.silu(gate_wide) * up_wide


def exact_half_origin_cpu(value: torch.Tensor) -> bool:
    """Conservative value audit, not an alias/provenance certificate.

    Finite CPU float32 values must survive float32->half->float32 bit exactly,
    including signed zero. Nonfinite, non-CPU, non-FP32 inputs decline.
    """
    if type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.float32:
        return False
    with torch.no_grad():
        if not bool(torch.isfinite(value).all()):
            return False
        original = value.detach().contiguous()
        restored = original.half().float()
        return bool(torch.equal(original.view(torch.int32), restored.view(torch.int32)))


def eligibility(gate: torch.Tensor, up: torch.Tensor, contract: OwnershipContract) -> tuple[bool, str]:
    if type(gate) is not torch.Tensor or type(up) is not torch.Tensor:
        return False, "tensor_subclass"
    if not contract.owned_producer_outputs:
        return False, "ownership_not_transferred"
    if not contract.first_order_only:
        return False, "first_order_contract_absent"
    if gate.device.type != "cpu" or up.device.type != "cpu":
        return False, "cpu_reference_only"
    if (torch.autograd.forward_ad.unpack_dual(gate).tangent is not None or
            torch.autograd.forward_ad.unpack_dual(up).tangent is not None):
        return False, "forward_ad_not_supported_by_reference"
    if gate.dtype != torch.float16 or up.dtype != torch.float16:
        return False, "actual_half_producers_required"
    if gate.shape != up.shape or gate.ndim not in (2, 3) or gate.numel() == 0:
        return False, "shape"
    if not gate.is_contiguous() or not up.is_contiguous():
        return False, "layout"
    if torch.is_inference_mode_enabled() or not torch.is_grad_enabled() or not (gate.requires_grad or up.requires_grad):
        return False, "no_training_gradient"
    if torch.is_autocast_enabled("cpu"):
        return False, "autocast_not_supported_by_reference"
    with torch.no_grad():
        if not bool(torch.isfinite(gate).all()) or not bool(torch.isfinite(up).all()):
            return False, "nonfinite_producer"
    return True, "owned_finite_half_producers"


class _OwnedHalfActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        # Ordinary save_for_backward keeps native version checks. Never save
        # widened tensors, callbacks, producer modules, or a parent graph.
        ctx.save_for_backward(gate, up)
        ctx.need_gate = ctx.needs_input_grad[0]
        ctx.need_up = ctx.needs_input_grad[1]
        return native_widened_activation(gate, up)

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("MEM001 explicit first-order ownership contract was violated; use the original fallback before forward for higher-order AD")
        gate_half, up_half = ctx.saved_tensors  # native version validation
        with torch.enable_grad():
            # Recreate the native FP32 graph, rather than invent/reassociate
            # the SiLU derivative or borrow the supplied starter's kernel.
            gate = gate_half.detach().float().requires_grad_(True)
            up = up_half.detach().float().requires_grad_(True)
            output = F.silu(gate) * up
            dgate, dup = torch.autograd.grad(output, (gate, up), grad_output,
                                           create_graph=False, retain_graph=False)
        # Each native widening edge rounds its own gradient to FP16. The
        # projection autograd nodes and their later FP32 join stay upstream.
        return (dgate.to(torch.float16) if ctx.need_gate else None,
                dup.to(torch.float16) if ctx.need_up else None)


def owned_half_activation(gate: torch.Tensor, up: torch.Tensor, *,
                          contract: Optional[OwnershipContract] = None,
                          diagnostics: Optional[dict] = None) -> torch.Tensor:
    contract = contract or OwnershipContract()
    accepted, reason = eligibility(gate, up, contract)
    if diagnostics is not None:
        diagnostics.update(route="owned_half_replay" if accepted else "native_fallback",
                           reason=reason, logical_half_payload_bytes=(gate.numel() + up.numel()) * 2 if accepted else 0)
    if not accepted:
        return native_widened_activation(gate, up)
    return _OwnedHalfActivation.apply(gate, up)
