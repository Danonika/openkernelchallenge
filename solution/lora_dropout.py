"""Native LoRA GEMMs with one original fused dropout-gradient epilogue.

Derived from the user's Round 7 ``l1e_bridge.py`` / Round 5 epilogue study.
No third-party kernel body is used. The forward uses the genuine PEFT layer
and genuine Dropout exactly once, then bridges the existing FP32 -> FP16
autocast boundary. Its backward reuses PyTorch's actual saved bool mask.

Only the pinned CUDA FP16 training path is admitted by the installed patch.
Unknown private saved-mask layouts retain the complete native autograd graph.
CPU interpreter tests explicitly override that device admission boundary;
passing them is neither CUDA correctness nor a speed measurement.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from importlib import metadata
import os

import torch
import triton
import triton.language as tl


_ACTIVE_DROPOUT = ContextVar("soup_lora_dropout_boundary", default=None)
_PEFT_CLASSES = ()
_KEEP_SCALE = 1.0 / (1.0 - 0.05)
_KEEP_SCALE_FP32 = 1.0526316165924072
_BLOCK = 256


@triton.jit
def soup_dropout_half_gradient_epilogue(GRAD, MASK, OUT, N,
                                        SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    grad = tl.load(GRAD + offset, mask=valid, other=0).to(tl.float32)
    keep = tl.load(MASK + offset, mask=valid, other=0).to(tl.float32)
    # Multiplication, rather than tl.where, preserves native 0 * Inf -> NaN.
    value = (keep * grad) * SCALE
    tl.store(OUT + offset, value, mask=valid)


class DropoutHalfBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, native_output_detached, mask):
        if native_output_detached.requires_grad:
            raise ValueError("the bridge must not retain the old dropout gradient edge")
        ctx.save_for_backward(mask)
        ctx.set_materialize_grads(False)
        return native_output_detached.to(torch.float16)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None, None
        (mask,) = ctx.saved_tensors
        if (torch.is_grad_enabled() or grad.dtype != torch.float16
                or grad.shape != mask.shape or grad.device != mask.device
                or not grad.is_contiguous() or grad.is_neg() or grad.is_conj()):
            # Native arithmetic preserves higher-order differentiation and
            # metadata-only negative views without raw-pointer assumptions.
            dx = torch.ops.aten.native_dropout_backward(
                grad.to(torch.float32), mask, _KEEP_SCALE)
        else:
            dx = torch.empty(grad.shape, device=grad.device, dtype=torch.float32)
            soup_dropout_half_gradient_epilogue[(triton.cdiv(grad.numel(), _BLOCK),)](
                grad, mask, dx, grad.numel(), _KEEP_SCALE_FP32, _BLOCK,
                num_warps=4, enable_fp_fusion=False)
        return dx, None, None


def _supported_device(x):
    return (os.environ.get("TRITON_INTERPRET") != "1" and x.device.type == "cuda"
            and torch.cuda.get_device_capability(x.device) == (7, 5)
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16)


def _plain_module(module):
    return ("forward" not in module.__dict__ and not hasattr(module, "_hf_hook")
            and not module._forward_hooks and not module._forward_pre_hooks
            and not module._backward_hooks and not module._backward_pre_hooks)


def _eligible_dropout(layer, x, args=(), kwargs=None):
    """Metadata-only eligibility; return the one active ordinary dropout."""
    if (args or kwargs or type(layer) not in _PEFT_CLASSES
            or type(x) is not torch.Tensor or x.dtype != torch.float32
            or x.ndim not in (2, 3) or not x.numel() or not x.is_contiguous()
            or x.is_neg() or x.is_conj() or not x.requires_grad
            or not torch.is_grad_enabled() or torch.is_inference_mode_enabled()
            or not layer.training or not _plain_module(layer)
            or layer.disable_adapters or layer.merged
            or len(layer.active_adapters) != 1 or not _supported_device(x)
            or torch.jit.is_tracing() or torch.jit.is_scripting()
            or torch.compiler.is_compiling()):
        return None
    hooks = torch.nn.modules.module
    if (hooks._global_forward_hooks or hooks._global_forward_pre_hooks
            or hooks._global_backward_hooks or hooks._global_backward_pre_hooks):
        return None
    name = layer.active_adapters[0]
    if (name not in layer.lora_A or name not in layer.lora_B
            or name not in layer.lora_dropout or name in layer.lora_variant
            or layer.r.get(name) != 16 or layer.lora_alpha.get(name) != 32
            or layer.scaling.get(name) != 2.0
            or layer.use_dora.get(name, False) or layer.use_rslora.get(name, False)):
        return None
    a, b = layer.lora_A[name], layer.lora_B[name]
    for module in (a, b):
        if (type(module) is not torch.nn.Linear or not _plain_module(module)
                or not module.training or module.bias is not None
                or not module.weight.requires_grad or module.weight.dtype != torch.float32
                or module.weight.device != x.device or not module.weight.is_contiguous()):
            return None
    dropout = layer.lora_dropout[name]
    if (type(dropout) is not torch.nn.Dropout or not _plain_module(dropout)
            or not dropout.training or dropout.p != 0.05 or dropout.inplace
            or a.in_features != x.shape[-1] or a.out_features != b.in_features
            or a.out_features != 16):
        return None
    return dropout


def _saved_native_mask(dropped, x):
    """Read the genuine native mask, never infer it from tensor values."""
    fn = dropped.grad_fn
    if type(fn).__name__ != "NativeDropoutBackward0":
        return None
    try:
        mask, p = fn._saved_result1, fn._saved_p
        if (float(p) != 0.05 or not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool or mask.requires_grad
                or mask.shape != x.shape or mask.device != x.device
                or not mask.is_contiguous() or mask.is_neg() or mask.is_conj()):
            return None
        return mask
    except (AttributeError, TypeError, RuntimeError):
        return None


@contextmanager
def lora_dropout_context(layer, x, args=(), kwargs=None):
    """Scope the bridge to one genuine PEFT invocation.

    The optional scale/add epilogue calls this around its existing A/B branch
    when it bypasses the native PEFT forward. It still calls Dropout once.
    Unsupported calls clear rather than inherit a surrounding layer context.
    """
    dropout = _eligible_dropout(layer, x, args, kwargs)
    token = _ACTIVE_DROPOUT.set((dropout, x) if dropout is not None else None)
    try:
        yield
    finally:
        _ACTIVE_DROPOUT.reset(token)


def _make_layer_forward(original):
    @wraps(original)
    def forward(self, x, *args, **kwargs):
        with lora_dropout_context(self, x, args, kwargs):
            return original(self, x, *args, **kwargs)
    forward._soup_candidate_lora_dropout = True
    return forward


def patch_lora_dropout():
    """Install before model construction; preserve every unsupported call.

    For joint scale/add fusion, install this before ``patch_lora_epilogue``
    and enter ``lora_dropout_context`` in that patch's adapter branch.
    No model instances, seeds, training data or benchmark state are inspected.
    """
    global _PEFT_CLASSES
    try:
        if (metadata.version("peft") != "0.20.0"
                or metadata.version("torch").split("+")[0] != "2.11.0"):
            return False
    except metadata.PackageNotFoundError:
        return False
    try:
        from peft.tuners.lora.layer import Linear
    except (ImportError, AttributeError):
        return False
    if not isinstance(Linear, type) or not issubclass(Linear, torch.nn.Module):
        return False
    classes = [Linear]
    try:
        from peft.tuners.lora.bnb import Linear4bit
        if isinstance(Linear4bit, type) and issubclass(Linear4bit, torch.nn.Module):
            classes.append(Linear4bit)
    except (ImportError, AttributeError):
        pass
    _PEFT_CLASSES = tuple(classes)

    original_dropout = torch.nn.Dropout.forward
    if not getattr(original_dropout, "_soup_candidate_lora_dropout", False):
        @wraps(original_dropout)
        def dropout_forward(self, x):
            # Native forward executes exactly once even if the mask contract
            # is unknown. Falling back never repeats RNG-consuming work.
            dropped = original_dropout(self, x)
            call = _ACTIVE_DROPOUT.get()
            if call is None or self is not call[0] or x is not call[1]:
                return dropped
            mask = _saved_native_mask(dropped, x)
            if mask is None:
                return dropped
            return DropoutHalfBridge.apply(x, dropped.detach(), mask)
        dropout_forward._soup_candidate_lora_dropout = True
        torch.nn.Dropout.forward = dropout_forward

    for cls in classes:
        if not getattr(cls.forward, "_soup_candidate_lora_dropout", False):
            cls.forward = _make_layer_forward(cls.forward)
    return True
