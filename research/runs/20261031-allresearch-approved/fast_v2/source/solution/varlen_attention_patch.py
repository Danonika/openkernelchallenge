"""Optional Qwen registry integration for the standalone native varlen path.

No default setting or model class is changed by importing this module. The
installer replaces only the shared AttentionInterface instance's SDPA entry.
Original Qwen forward, projection, rectangular LoRA dropout and RoPE still
execute. Only a validated private MLP-padding context can admit a call.

One exact comparison per context verifies that the original boolean mask is
causal AND the original two-dimensional key-padding mask. A changed mask
identity/version disables this path for the rest of that context. A single
GPU int32 cumulative-length tensor and the existing row maps are then shared
across layers. No persistent model cache or cross-forward mask cache exists.

The context owner should call clear_varlen_attention_call(call) in its
finally block. This releases cached forward mask/plan references immediately;
the native attention autograd nodes retain the row tensors they need.

As with the underlying padding experiment, only finite private training is
supported: padded query outputs are not public and have zero derivatives.
The wrapper does not scan activations for finiteness or claim equivalence for
arbitrary NaN/Inf propagation through unused padding. Native attention and
the row-copy experiment require physical T4 validation before activation.
"""

from dataclasses import dataclass
from functools import wraps
from importlib import metadata
from itertools import accumulate
import math
import os

import torch

from . import mlp_padding
from . import varlen_attention as primitive


_STATE_ATTRIBUTE = "_soup_varlen_attention_state"


@dataclass
class _AttentionState:
    mask: object
    version: int
    plan: object = None
    blocked: bool = False


def clear_varlen_attention_call(call):
    """Release this context's forward-only references, including on failure."""
    state = getattr(call, _STATE_ATTRIBUTE, None)
    if state is not None:
        state.mask = None
        state.plan = None
        state.blocked = True
        delattr(call, _STATE_ATTRIBUTE)


def _supported_device(query):
    return (query.device.type == "cuda" and os.environ.get("TRITON_INTERPRET") != "1"
            and torch.cuda.get_device_capability(query.device) == (7, 5)
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16)


def _plain_attention(module):
    hooks = torch.nn.modules.module
    return ("forward" not in module.__dict__ and not hasattr(module, "_hf_hook")
            and not module._forward_hooks and not module._forward_pre_hooks
            and not module._backward_hooks and not module._backward_pre_hooks
            and not hooks._global_forward_hooks and not hooks._global_forward_pre_hooks
            and not hooks._global_backward_hooks and not hooks._global_backward_pre_hooks)


def _eligible_call(module, query, key, value, attention_mask, dropout, scaling,
                   is_causal, position_bias, kwargs, attention_class):
    call = mlp_padding._ACTIVE_MLP_PADDING.get()
    configured_dropout = getattr(module, "attention_dropout", None)
    if (type(call) is not mlp_padding._PaddingCall or type(module) is not attention_class
            or id(module) not in call.attention_ids or not module.training
            or not _plain_attention(module) or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled() or torch.jit.is_tracing()
            or torch.jit.is_scripting() or torch.compiler.is_compiling()
            or getattr(module.config, "_attn_implementation", None) != "sdpa"
            or getattr(module, "is_causal", None) is not True
            or (is_causal is not None and is_causal is not True)
            or type(dropout) not in (int, float) or dropout != 0
            or type(configured_dropout) not in (int, float) or configured_dropout != 0
            or getattr(module, "sliding_window", None) is not None
            or position_bias is not None or kwargs.get("sliding_window") is not None
            or kwargs.get("past_key_values") is not None
            or kwargs.get("past_key_value") is not None
            or (kwargs.get("use_cache") is not None and kwargs.get("use_cache") is not False)
            or kwargs.get("output_attentions", False)):
        return None
    if scaling is not None and (type(scaling) not in (int, float)
                                or not math.isfinite(scaling) or scaling <= 0):
        return None
    tensors = (query, key, value)
    if (any(type(tensor) is not torch.Tensor or tensor.dtype != torch.float32
            or tensor.ndim != 4 or tensor.stride(-1) != 1
            or min(tensor.stride()) < 1 or tensor.is_neg() or tensor.is_conj()
            or tensor.is_inference() for tensor in tensors)
            or not any(tensor.requires_grad for tensor in tensors)
            or not _supported_device(query)):
        return None
    batch, heads, sequence, width = query.shape
    groups = getattr(module, "num_key_value_groups", None)
    if (min(query.shape) < 1 or width % 8 or width > 128
            or type(groups) is not int or groups < 1
            or key.shape != value.shape or key.shape[0] != batch
            or key.shape[1] * groups != heads or tuple(key.shape[2:]) != (sequence, width)
            or key.device != query.device or value.device != query.device
            or tuple(call.shape) != (batch, sequence)
            or type(call.mask) is not torch.Tensor or call.mask.device != query.device
            or tuple(call.mask.shape) != (batch, sequence)
            or call.mask.dtype not in (torch.bool, torch.int32, torch.int64)
            or call.mask.is_inference() or call.mask.requires_grad
            or call.mask._version != call.mask_version
            or type(call.lengths) is not tuple or not call.lengths or len(call.lengths) != batch
            or any(type(length) is not int or not 1 <= length <= sequence for length in call.lengths)
            or type(attention_mask) is not torch.Tensor or attention_mask.dtype != torch.bool
            or tuple(attention_mask.shape) != (batch, 1, sequence, sequence)
            or attention_mask.device != query.device or attention_mask.requires_grad
            or attention_mask.is_inference() or attention_mask.is_neg() or attention_mask.is_conj()):
        return None
    return call


def _verify_mask(attention_mask, call):
    """Exactly one real GPU equality/synchronization for an admitted context."""
    sequence = call.shape[1]
    position = torch.arange(sequence, device=attention_mask.device)
    causal = position[:, None] >= position[None, :]
    expected = causal[None, None, :, :] & call.mask.bool()[:, None, None, :]
    return torch.equal(attention_mask, expected)


def _plan_for_call(call, attention_mask):
    state = getattr(call, _STATE_ATTRIBUTE, None)
    if state is not None:
        if (state.blocked or state.mask is not attention_mask
                or state.version != attention_mask._version):
            state.blocked = True
            state.plan = None
            return None
        try:
            primitive._check_plan(state.plan, attention_mask.device)
        except ValueError:
            state.blocked = True
            state.plan = None
            return None
        return state.plan
    # Retain identity even on failure so there is no second verification in
    # this decoder call. The strong mask reference prevents Python id reuse.
    state = _AttentionState(attention_mask, attention_mask._version, blocked=True)
    setattr(call, _STATE_ATTRIBUTE, state)
    if not _verify_mask(attention_mask, call):
        return None
    cumulative = torch.tensor(tuple(accumulate(call.lengths, initial=0)),
                              dtype=torch.int32, device=attention_mask.device)
    try:
        state.plan = primitive.plan_from_row_tensors(
            call.rows, call.inverse, cumulative,
            lengths=call.lengths, sequence=call.shape[1])
    except ValueError:
        return None
    state.blocked = False
    return state.plan


def _make_dispatcher(original, attention_class, native_repeat_kv):
    @wraps(original)
    def sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
             is_causal=None, position_bias=None, **kwargs):
        call = _eligible_call(module, query, key, value, attention_mask, dropout,
                              scaling, is_causal, position_bias, kwargs, attention_class)
        plan = _plan_for_call(call, attention_mask) if call is not None else None
        if plan is None:
            return original(module, query, key, value, attention_mask,
                dropout=dropout, scaling=scaling, is_causal=is_causal,
                position_bias=position_bias, **kwargs)
        # The original boolean mask disables native GQA. Retain precisely its
        # FP32 repeat operation, then let the primitive apply the FP16 cast.
        if module.num_key_value_groups > 1:
            key = native_repeat_kv(key, module.num_key_value_groups)
            value = native_repeat_kv(value, module.num_key_value_groups)
        output = primitive.varlen_attention(query, key, value, plan, scale=scaling,
                                             private_training=True, finite_training=True)
        # Native registry callers expect BTHD, not the SDPA primitive's BHTD.
        return output.transpose(1, 2).contiguous(), None

    sdpa._soup_candidate_varlen_attention = True
    sdpa._soup_varlen_native_sdpa = original
    return sdpa


def patch_varlen_attention():
    """Install only a version-pinned, idempotent SDPA registry override."""
    try:
        if (metadata.version("transformers") != "5.16.1"
                or metadata.version("torch").split("+")[0] != "2.11.0"
                or metadata.version("triton") != "3.6.0"):
            return False
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward, repeat_kv
        from transformers.models.qwen2 import modeling_qwen2
    except (ImportError, AttributeError, metadata.PackageNotFoundError):
        return False
    attention_class = getattr(modeling_qwen2, "Qwen2Attention", None)
    if (not isinstance(attention_class, type) or not issubclass(attention_class, torch.nn.Module)
            or type(ALL_ATTENTION_FUNCTIONS) is not AttentionInterface
            or getattr(modeling_qwen2, "ALL_ATTENTION_FUNCTIONS", None) is not ALL_ATTENTION_FUNCTIONS):
        return False
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(original, "_soup_candidate_varlen_attention", False):
        return True
    if original is not sdpa_attention_forward:
        return False  # Do not bypass another user's registered attention behavior.
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _make_dispatcher(original, attention_class, repeat_kv)
    return True
