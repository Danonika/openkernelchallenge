"""Input-only rematerialization for the fixed frozen Qwen2 MLP.

Independent team implementation.  The idea is activation rematerialization:
the frozen gate/up/down MLP has no trainable parameters and no RNG.  Forward
executes the exact currently-installed MLP path under the existing autocast,
returns its value, and drops its interior autograd graph while retaining only
the MLP input.  Backward recreates that same path and differentiates only the
input.

This composes with our FP16 RMS output, NF4 decoder and starter SwiGLU path.
No model parameter, optimizer state, data, batch, step, dropout or Soup switch
is changed.  Frozen parameter identities/versions are checked before replay.
"""
from __future__ import annotations

from functools import wraps
import torch


def _frozen_mlp(module):
    layers = tuple(getattr(module, name, None)
                   for name in ("gate_proj", "up_proj", "down_proj"))
    if any(layer is None for layer in layers):
        return False
    return all(not any(p.requires_grad for p in layer.parameters())
               for layer in layers)


class _ReplayFrozenMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, original):
        ctx.module = module
        ctx.original = original
        ctx.autocast_enabled = torch.is_autocast_enabled(x.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(x.device.type)
        ctx.param_stamps = tuple((id(p), p._version) for p in module.parameters())
        ctx.save_for_backward(x)
        ctx.set_materialize_grads(False)

        # Keep grad mode enabled so every nested version-specific MLP fast path
        # takes the same branch as normal training; detach only the returned
        # value so its transient interior graph can be released immediately.
        with torch.enable_grad(), torch.autocast(
            device_type=x.device.type,
            enabled=ctx.autocast_enabled,
            dtype=ctx.autocast_dtype,
        ):
            replay_input = x.detach().requires_grad_(True)
            output = original(module, replay_input)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("frozen MLP replay expected one tensor output")
        return output.detach()

    @staticmethod
    def backward(ctx, upstream):
        if upstream is None:
            return None, None, None
        module = ctx.module
        stamps = tuple((id(p), p._version) for p in module.parameters())
        if stamps != ctx.param_stamps:
            raise RuntimeError("frozen MLP parameters changed before replay")

        (saved_x,) = ctx.saved_tensors
        x = saved_x.detach().requires_grad_(True)
        with torch.enable_grad(), torch.autocast(
            device_type=x.device.type,
            enabled=ctx.autocast_enabled,
            dtype=ctx.autocast_dtype,
        ):
            output = ctx.original(module, x)
        (dx,) = torch.autograd.grad(
            output, x, upstream,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        return dx, None, None


def patch_all_frozen_mlp_replay():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    previous = Qwen2MLP.forward
    if getattr(previous, "_soup_own_all_mlp_replay", False):
        return True

    @wraps(previous)
    def forward(self, x):
        eligible = (
            self.training
            and torch.is_grad_enabled()
            and isinstance(x, torch.Tensor)
            and x.is_cuda
            and x.requires_grad
            and x.ndim in (2, 3)
            and x.numel() > 0
            and _frozen_mlp(self)
        )
        if not eligible:
            return previous(self, x)
        return _ReplayFrozenMLP.apply(x, self, previous)

    forward._soup_own_all_mlp_replay = True
    forward._soup_own_previous = previous
    Qwen2MLP.forward = forward
    return True
