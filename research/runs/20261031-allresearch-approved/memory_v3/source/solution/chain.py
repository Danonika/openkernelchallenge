"""Pinned native ownership helpers. No cast or module hook is installed."""
from dataclasses import dataclass
import inspect
import weakref

import torch
from torch.nn.modules import module as module_globals

_HOOK_NAMES = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_GLOBAL_HOOK_NAMES = tuple("_global" + name for name in _HOOK_NAMES)


def no_hooks(module, *, own_head_hook=None):
    for name in _HOOK_NAMES:
        hooks = getattr(module, name, {})
        if name == "_forward_hooks" and own_head_hook is not None:
            if set(hooks) != {own_head_hook}:
                return False
        elif hooks:
            return False
    return True


def no_global_hooks():
    return not any(getattr(module_globals, name, {}) for name in _GLOBAL_HOOK_NAMES)


def raw_callable(value):
    return getattr(value, "__func__", value)


@dataclass(frozen=True)
class Owners:
    """Explicit dependency injection for CPU fixtures; production is pinned."""
    rewind_class: type
    sft_class: type
    hf_class: type
    peft_class: type
    qwen_class: type
    tuner_class: type
    rewind_loss: object
    sft_loss: object
    hf_loss: object
    peft_forward: object
    qwen_forward: object
    tuner_forward: object
    native_loss: object
    entropy_function: object
    rewind_rows: object
    get_base_model: object
    linear_forward: object
    converter_class: type
    converter_call: object
    converter_function: object
    converter_recursive: object
    converter_code: object
    autocast_code: object
    device_type: str = "cuda"
    vocabulary: int = 151936
    batch_size: int = 4


def prepared_chain_reason(model, owners):
    """Prove the normal Accelerate->autocast->original PEFT callable chain."""
    original = getattr(model, "_original_forward", None)
    if raw_callable(original) is not owners.peft_forward:
        return "original_peft_forward"
    outer = raw_callable(model.forward)
    if getattr(outer, "__code__", None) is not owners.converter_code:
        return "outer_conversion_code"
    converter = getattr(outer, "__wrapped__", None)
    if (type(converter) is not owners.converter_class
            or type(converter).__call__ is not owners.converter_call):
        return "outer_conversion_owner"
    try:
        closed = inspect.getclosurevars(outer).nonlocals
    except (TypeError, ValueError):
        return "outer_conversion_closure"
    if closed != {"model_forward": converter}:
        return "outer_conversion_closure"
    amp = converter.model_forward
    if (getattr(converter, "__wrapped__", None) is not amp
            or getattr(amp, "__code__", None) is not owners.autocast_code
            or getattr(amp, "__wrapped__", None) is not owners.peft_forward):
        return "autocast_callable_chain"
    try:
        amp_closure = inspect.getclosurevars(amp).nonlocals
    except (TypeError, ValueError):
        return "autocast_closure"
    context = amp_closure.get("autocast_instance")
    if (set(amp_closure) != {"autocast_instance", "func"}
            or amp_closure["func"] is not owners.peft_forward
            or type(context) is not torch.amp.autocast_mode.autocast
            or context.device != owners.device_type
            or context.fast_dtype != torch.float16 or not context._enabled):
        return "autocast_context"
    # Converter's global must still be the captured native recursive function.
    if (owners.converter_call.__globals__.get("convert_to_fp32") is not owners.converter_function
            or owners.converter_function.__globals__.get("recursively_apply") is not owners.converter_recursive):
        return "converter_global_changed"
    return None
