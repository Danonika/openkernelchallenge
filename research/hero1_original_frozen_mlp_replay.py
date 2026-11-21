"""Hero1: original input-only frozen-MLP rematerialization for Soup/Qwen2.

Concept inspiration: publicly reported compact frozen MLP approach from
external-contributor/soup-kernel-challenge plus our previous selective-remat work.
Source implementation is independent: no competitor kernel code copied.

Forward: execute the exact existing frozen MLP in native grad-enabled mode,
release its transient interior graph, and retain no interior activations.
autograd intermediates. Backward: recreate its differentiable operations,
restoring the original CUDA autocast context, then differentiate the input.
The original Soup training configuration, model, 64 steps and LoRA optimizer
are unchanged. Activation lifetime is shortened; compute is intentionally
traded for peak VRAM and must be benchmarked under the official harness.
"""
from __future__ import annotations

from functools import wraps
import torch


def _frozen_three_projection_mlp(module):
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    if gate is None or up is None or down is None:
        return False
    return all(not any(p.requires_grad for p in layer.parameters())
               for layer in (gate, up, down))


class _Hero1Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, original_fn):
        ctx.module = module
        ctx.original_fn = original_fn
        ctx.use_amp = torch.is_autocast_enabled(x.device.type)
        ctx.amp_dtype = torch.get_autocast_dtype(x.device.type)
        ctx.param_stamps = tuple((id(p), p._version) for p in module.parameters())
        ctx.save_for_backward(x)
        # Some valid frozen-MLP patches select their exact native implementation
        # only when grad mode is enabled. Replay must execute the same branch
        # as ordinary training, without retaining that transient inner graph.
        with torch.enable_grad(), torch.autocast(
            device_type=x.device.type,
            enabled=ctx.use_amp,
            dtype=ctx.amp_dtype,
        ):
            transient_input = x.detach().requires_grad_(True)
            output = original_fn(module, transient_input)
        return output.detach()

    @staticmethod
    def backward(ctx, upstream):
        (saved_x,) = ctx.saved_tensors
        module = ctx.module
        current_stamps = tuple((id(p), p._version) for p in module.parameters())
        if current_stamps != ctx.param_stamps:
            raise RuntimeError("Frozen MLP parameters changed during Hero1 replay")
        x = saved_x.detach().requires_grad_(True)
        with torch.enable_grad(), torch.autocast(
            device_type=x.device.type,
            enabled=ctx.use_amp,
            dtype=ctx.amp_dtype,
        ):
            replay_output = ctx.original_fn(module, x)
        (dx,) = torch.autograd.grad(
            outputs=replay_output,
            inputs=(x,),
            grad_outputs=upstream,
            allow_unused=False,
        )
        return dx, None, None


def replay_frozen_mlp_if_selected(module, x, original_fn):
    """No-gate differences for inference, fake regressions, non-CUDA or LoRA MLP."""
    eligible = (
        getattr(module, "_hero1_replay_selected", False)
        and module.training and torch.is_grad_enabled()
        and isinstance(x, torch.Tensor) and x.is_cuda
        and x.requires_grad and x.ndim == 2
        and _frozen_three_projection_mlp(module)
    )
    if not eligible:
        return original_fn(module, x)
    return _Hero1Replay.apply(x, module, original_fn)


def patch_soup_sparse_frozen_mlp_replay(*, selected_layers=(20, 22, 24, 26),
                                        expected_layers=28):
    """Mark a deterministic subset of real Qwen2 MLPs when Soup loads the model.

    Attaches only Python attributes to MLP modules. Does not replace or create
    parameters, buffers, data, hooks, config switches, or optimizer state.
    """
    from soup_cli.trainer import sft
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    original_setup = sft.SFTTrainerWrapper._setup_transformers
    if getattr(original_setup, "_hero1_replay_installed", False):
        return True
    chosen = frozenset(int(i) for i in selected_layers)
    assert chosen and min(chosen) >= 0 and max(chosen) < expected_layers

    @wraps(original_setup)
    def wrapped_setup(self, *args, **kwargs):
        value = original_setup(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            mlps = [mod for mod in model.modules() if type(mod) is Qwen2MLP]
            if len(mlps) == expected_layers:
                for index, mlp in enumerate(mlps):
                    if index in chosen and _frozen_three_projection_mlp(mlp):
                        mlp._hero1_replay_selected = True
        return value

    wrapped_setup._hero1_replay_installed = True
    sft.SFTTrainerWrapper._setup_transformers = wrapped_setup
    return True
