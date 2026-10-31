"""Share one additive SDPA mask within one ordinary Qwen2 decoder call.

Port of the user's original Round 7 shared-mask integration. The original
mask factory, Q/K/V calls, attention backend and RNG are unchanged. A boolean
mask is converted once to FP16 with an eight-element row pitch; later layers
reuse it only while the original tensor identity and version remain equal.

The installed path selects CUDA SM75 / FP16 only. No GPU result is asserted.
CPU tests use explicit admission overrides and Triton's interpreter.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from importlib import metadata
import os

import torch
import triton
import triton.language as tl


_ACTIVE_MASK = ContextVar("soup_shared_attention_mask", default=None)


@triton.jit
def soup_aligned_additive_mask(BOOL, OUT, B, T, PITCH, SB, SR, SC,
                               BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < B * T * T
    col = offset % T
    row = (offset // T) % T
    batch = offset // (T * T)
    allow = tl.load(BOOL + batch * SB + row * SR + col * SC,
                    mask=valid, other=0)
    value = tl.where(allow, 0.0, -float("inf"))
    # Padding is not visible to SDPA and is deliberately untouched.
    tl.store(OUT + (batch * T + row) * PITCH + col, value, mask=valid)


def convert_mask(mask, *, sentinel=None):
    """Direct original kernel entry; CPU requires Triton's explicit interpreter."""
    if (type(mask) is not torch.Tensor or mask.dtype != torch.bool or mask.ndim != 4
            or mask.requires_grad or mask.is_neg() or mask.is_conj()):
        raise ValueError("expected a four-dimensional nondifferentiable bool mask")
    b, heads, q, k = mask.shape
    if b <= 0 or heads != 1 or q <= 0 or q != k or any(s < 0 for s in mask.stride()):
        raise ValueError("expected nonempty square [B,1,T,T] mask")
    pitch = triton.cdiv(k, 8) * 8
    owner = torch.empty((b, 1, q, pitch), dtype=torch.float16, device=mask.device)
    if sentinel is not None:
        owner.fill_(sentinel)
    soup_aligned_additive_mask[(triton.cdiv(b * q * k, 256),)](
        mask, owner, b, q, pitch, mask.stride(0), mask.stride(2), mask.stride(3),
        256, num_warps=4)
    return owner[..., :k], owner


@dataclass
class _MaskCall:
    decoder: object
    attention_ids: set
    cache: dict = field(default_factory=dict)


def _autocast_fp16():
    return (torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16)


def _supported_device(hidden):
    return (os.environ.get("TRITON_INTERPRET") != "1" and hidden.device.type == "cuda"
            and torch.cuda.get_device_capability(hidden.device) == (7, 5))


def _decoder_supported(decoder, args, kwargs):
    return (not args and decoder.training and torch.is_grad_enabled()
            and not torch.is_inference_mode_enabled() and _autocast_fp16()
            and decoder.config._attn_implementation == "sdpa"
            and not decoder.has_sliding_layers
            and not bool(getattr(decoder, "gradient_checkpointing", False))
            and kwargs.get("past_key_values") is None
            and kwargs.get("use_cache") is False
            and not isinstance(kwargs.get("attention_mask"), dict)
            and not torch.jit.is_tracing() and not torch.jit.is_scripting()
            and not torch.compiler.is_compiling())


def _attention_mask(module, args, kwargs, call):
    if args or id(module) not in call.attention_ids:
        return None
    hidden, mask = kwargs.get("hidden_states"), kwargs.get("attention_mask")
    if (type(mask) is not torch.Tensor or mask.dtype != torch.bool
            or mask.ndim != 4 or mask.is_inference() or mask.requires_grad
            or type(hidden) is not torch.Tensor or hidden.ndim != 3
            or hidden.dtype != torch.float32 or not _supported_device(hidden)
            or mask.device != hidden.device
            or mask.shape != (hidden.shape[0], 1, hidden.shape[1], hidden.shape[1])
            or module.config._attn_implementation != "sdpa" or not module.training
            or kwargs.get("past_key_values") is not None):
        return None
    key = (id(mask), mask._version, tuple(mask.shape), tuple(mask.stride()), mask.device)
    found = call.cache.get(key)
    if found is None:
        converted, owner = convert_mask(mask)
        # A strong reference prevents Python identity reuse within this call.
        call.cache[key] = (mask, converted, owner)
    else:
        _, converted, owner = found
    return converted


def patch_shared_attention_mask():
    """Idempotent, before-construction patch with no retained cross-call state."""
    try:
        if (metadata.version("transformers") != "5.16.1"
                or metadata.version("torch").split("+")[0] != "2.11.0"):
            return False
    except metadata.PackageNotFoundError:
        return False
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2Attention
    except (ImportError, AttributeError):
        return False
    if not all(isinstance(cls, type) and issubclass(cls, torch.nn.Module)
               for cls in (Qwen2Model, Qwen2Attention)):
        return False

    if getattr(Qwen2Model.forward, "_soup_candidate_shared_mask", False):
        return True
    original_decoder, original_attention = Qwen2Model.forward, Qwen2Attention.forward

    @wraps(original_decoder)
    def decoder_forward(self, *args, **kwargs):
        call = None
        if _decoder_supported(self, args, kwargs):
            call = _MaskCall(self, {id(layer.self_attn) for layer in self.layers})
        token = _ACTIVE_MASK.set(call)
        try:
            return original_decoder(self, *args, **kwargs)
        finally:
            if call is not None:
                call.cache.clear()
            _ACTIVE_MASK.reset(token)

    @wraps(original_attention)
    def attention_forward(self, *args, **kwargs):
        call = _ACTIVE_MASK.get()
        mask = _attention_mask(self, args, kwargs, call) if call is not None else None
        if mask is not None:
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = mask
        return original_attention(self, *args, **kwargs)

    decoder_forward._soup_candidate_shared_mask = True
    attention_forward._soup_candidate_shared_mask = True
    Qwen2Model.forward = decoder_forward
    Qwen2Attention.forward = attention_forward
    return True
