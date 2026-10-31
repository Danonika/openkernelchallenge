"""Avoid PEFT's transient full-size FP32 tied embedding in Qwen2 NF4 preparation.

The model starts with an exact FP16 tied frozen base embedding. The original
PEFT preparation freezes parameters and upcasts nonquantized tensors. For this
one guarded parameter, run PEFT with an empty temporary tensor, then restore
the original FP16 bytes, the same Parameter identity and tying, and the
existing exact FP32 lookup bridge. All other PEFT operations run normally.
"""
from functools import wraps
import inspect
import torch
from . import frozen_storage

_STATS = {'installed': 0, 'fallback': 0}

def counters():
    return dict(_STATS)

def patch_early_storage():
    import peft
    parent = peft.prepare_model_for_kbit_training
    if getattr(parent, '_soup_early_tied_half', False):
        return True
    signature = inspect.signature(parent)

    @wraps(parent)
    def prepare(model, *args, **kwargs):
        bound = signature.bind(model, *args, **kwargs)
        bound.apply_defaults()
        candidate = frozen_storage._capture_candidate(
            model, bound.arguments.get('use_gradient_checkpointing', True))
        if candidate is None:
            _STATS['fallback'] += 1
            return parent(model, *args, **kwargs)
        param = candidate.parameter
        original_half = candidate.half_storage
        # Original PEFT sees the exact same Parameter and module tree but an
        # empty FP16 data allocation for the one frozen tied weight only.
        # Its parameter-freezing and optional hooks still run unchanged.
        param.data = original_half.new_empty((0,))
        try:
            prepared = parent(model, *args, **kwargs)
        finally:
            # Restore even if PEFT raises; no invalid tied weights escape.
            param.data = original_half
        if (prepared is not candidate.model
            or candidate.embedding.weight is not param
            or candidate.head.weight is not param
            or param.requires_grad
            or tuple(param.shape) != tuple(original_half.shape)
            or param.dtype != torch.float16):
            raise RuntimeError('Early tied-embedding preparation violated model identity')
        frozen_storage._install_storage(candidate)
        _STATS['installed'] += 1
        return prepared

    prepare._soup_early_tied_half = True
    peft.prepare_model_for_kbit_training = prepare
    return True
