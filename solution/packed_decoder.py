"""Independent, batch-local Qwen2 decoder token packing.

Only algorithmic ideas from public challenge submissions are used. This module
contains original implementation, not copied kernels. Retains the input batch,
labels, model, and native LoRA dropout RNG stream. Attention is computed inside
each original example, with no cross-example attention.

The optimized call is engaged by `integration.py` only for labelled training;
evaluation, unsupported layouts, tests, and other models retain their native
behavior. No changes to Soup feature switches, sampling, or optimizer.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch
import torch.nn.functional as F

_LABELS = ContextVar('soup_fusion_labels', default=None)
_SEGMENTS = ContextVar('soup_fusion_segments', default=None)
_DROPOUT = ContextVar('soup_fusion_dropout', default=None)
_INSTALLED = False
_COUNTS = {'packed_calls': 0, 'native_calls': 0, 'native_dropout_calls': 0, 'sdpa_calls': 0}


@contextmanager
def labelled_training(labels):
    """Establish labels from the real SFTTrainer.compute_loss, not model guessing."""
    token = _LABELS.set(labels)
    try:
        yield
    finally:
        _LABELS.reset(token)


def _valid_right_padded(mask):
    if mask.dtype not in (torch.int64, torch.int32, torch.bool, torch.uint8):
        return False
    live = mask != 0
    if live.ndim != 2 or live.shape[0] < 2 or live.shape[1] < 2:
        return False
    if not bool(live.any(dim=-1).all().item()):
        return False
    return not bool((~live[:, :-1] & live[:, 1:]).any().item())


def _batch_plan(mask):
    b, t = mask.shape
    live = mask != 0
    indices = torch.nonzero(live.flatten(), as_tuple=False).flatten()
    offsets = indices.remainder(t)
    lengths = live.sum(dim=1).tolist()
    segments = []
    start = 0
    for n in lengths:
        end = start + int(n)
        segments.append((start, end))
        start = end
    assert start == indices.numel()
    return indices, offsets, tuple(segments)


class _OriginalGeometryDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, reciprocal_keep):
        ctx.save_for_backward(mask)
        ctx.reciprocal_keep = reciprocal_keep
        return x * (mask.to(x.dtype) * reciprocal_keep)

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        return grad * (mask.to(grad.dtype) * ctx.reciprocal_keep), None, None


def _native_geometry_dropout(x, plan, p):
    b, t, indices, expected_ids, counters = plan
    # The padded B*T*D native allocation consumes the identical Philox stream
    # that stock torch.nn.Dropout on uncompressed B,T,D would consume.
    dummy = torch.ones((b, t, x.shape[-1]), device=x.device, dtype=x.dtype)
    _, bool_mask = torch.ops.aten.native_dropout(dummy, float(p), True)
    selected = bool_mask.reshape(b * t, -1).index_select(0, indices)
    selected = selected.reshape_as(x)
    counters['dropout'] += 1
    return _OriginalGeometryDropout.apply(x, selected, 1.0 / (1.0 - p))


def install_decoder_packing():
    """Install *after* the original streamed-head and normalization patches."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    global _INSTALLED
    if _INSTALLED:
        return True

    model_forward = Qwen2Model.forward
    original_dropout = torch.nn.Dropout.forward
    original_sdpa = F.scaled_dot_product_attention

    @wraps(original_sdpa)
    def sliced_sdpa(q, k, v, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, enable_gqa=False):
        state = _SEGMENTS.get()
        if state is None:
            return original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                 is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        bounds, count, counters = state
        if (attn_mask is not None or dropout_p != 0.0 or not is_causal
                or q.ndim != 4 or k.ndim != 4 or v.ndim != 4
                or q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1
                or q.shape[-2] != count or k.shape[-2] != count
                or v.shape[-2] != count):
            # Never silently use a global causal mask on concatenated examples.
            raise RuntimeError('Packed SDPA invariant broken: cross-row attention prohibited')
        pieces = []
        for low, high in bounds:
            pieces.append(original_sdpa(
                q[:, :, low:high, :], k[:, :, low:high, :],
                v[:, :, low:high, :], attn_mask=None, dropout_p=0.0,
                is_causal=True, scale=scale, enable_gqa=enable_gqa))
        counters['attention'] += 1
        return torch.cat(pieces, dim=-2)

    @wraps(original_dropout)
    def dropout_forward(module, x):
        plan = _DROPOUT.get()
        if (plan is None or id(module) not in plan[3]
                or type(module) is not torch.nn.Dropout
                or not module.training or module.inplace or module.p != 0.05
                or not isinstance(x, torch.Tensor) or not x.is_cuda
                or x.ndim != 3 or x.dtype not in (torch.float16, torch.float32)
                or x.shape[0] != 1 or x.shape[1] != plan[2].numel()):
            return original_dropout(module, x)
        return _native_geometry_dropout(x, plan, module.p)

    @wraps(model_forward)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        def native():
            _COUNTS['native_calls'] += 1
            return model_forward(self, input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, use_cache=use_cache,
                                 **kwargs)

        labels = _LABELS.get()
        if (not self.training or not torch.is_grad_enabled()
                or not isinstance(input_ids, torch.Tensor) or not input_ids.is_cuda
                or input_ids.ndim != 2
                or not isinstance(attention_mask, torch.Tensor) or not attention_mask.is_cuda
                or attention_mask.shape != input_ids.shape
                or not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape
                or labels.device != input_ids.device
                or inputs_embeds is not None or past_key_values is not None or use_cache
                or self.gradient_checkpointing or self.has_sliding_layers
                or float(getattr(self.config, 'attention_dropout', 0.0)) != 0.0
                or getattr(self.config, '_attn_implementation', 'sdpa') != 'sdpa'
                or kwargs.get('output_attentions') or kwargs.get('output_hidden_states')
                or not _valid_right_padded(attention_mask)):
            return native()

        if not bool(((labels == -100) | (attention_mask != 0)).all().item()):
            return native()
        b, t = input_ids.shape
        indices, positions, bounds = _batch_plan(attention_mask)
        if indices.numel() == b*t:
            return native()
        if position_ids is not None:
            if (position_ids.ndim != 2 or position_ids.shape[-1] != t
                    or position_ids.shape[0] not in (1, b)):
                return native()
            pos_flat = position_ids.expand(b, t).reshape(-1)
            positions = pos_flat.index_select(0, indices)
        positions = positions.reshape(1, -1)
        packed_ids = input_ids.reshape(-1).index_select(0, indices).reshape(1, -1)

        dropout_ids = {
            id(layer) for name, layer in self.named_modules()
            if '.lora_dropout.' in name and type(layer) is torch.nn.Dropout
            and layer.p == 0.05 and layer.training
        }
        # Verify exactly the two trainable adapter dropout sites per decoder layer.
        if len(dropout_ids) != 2 * self.config.num_hidden_layers:
            return native()
        counters = {'dropout': 0, 'attention': 0}
        drop_token = _DROPOUT.set((b, t, indices, dropout_ids, counters))
        sdpa_token = _SEGMENTS.set((bounds, indices.numel(), counters))
        try:
            packed = model_forward(
                self, input_ids=packed_ids,
                attention_mask={'full_attention': None},
                position_ids=positions,
                past_key_values=None, inputs_embeds=None,
                use_cache=False, **kwargs)
        finally:
            _SEGMENTS.reset(sdpa_token)
            _DROPOUT.reset(drop_token)

        if (counters['dropout'] != len(dropout_ids)
                or counters['attention'] != self.config.num_hidden_layers):
            raise RuntimeError('Packed Qwen path did not fully engage dropout and attention')

        from transformers.modeling_outputs import BaseModelOutputWithPast
        states = packed.last_hidden_state
        restored = torch.zeros((b*t, states.shape[-1]),
                               dtype=states.dtype, device=states.device)
        restored.index_copy_(0, indices, states.reshape(-1, states.shape[-1]))
        _COUNTS['packed_calls'] += 1
        _COUNTS['native_dropout_calls'] += counters['dropout']
        _COUNTS['sdpa_calls'] += counters['attention']
        return BaseModelOutputWithPast(
            last_hidden_state=restored.reshape(b, t, -1),
            past_key_values=None, hidden_states=None, attentions=None)

    Qwen2Model.forward = forward
    F.scaled_dot_product_attention = sliced_sdpa
    torch.nn.Dropout.forward = dropout_forward
    forward._soup_original_fullpack = True
    _INSTALLED = True
    return True


def counters():
    return dict(_COUNTS)
