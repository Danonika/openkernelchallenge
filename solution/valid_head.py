"""Original padding-aware vocabulary head, fused into the existing Soup pipeline.

Our packed decoder restores B*T hidden rows for Soup's metric contract, but
the head only needs logits for real (attention_mask=1) rows. Repack those real
rows before the 151936-vocab projection; restore aligned token_loss, entropy,
and argmax afterward. The scalar causal loss and its input gradient stay on
the compact graph. No change to optimizer, LoRA/dropout RNG or dataset.
"""
from contextvars import ContextVar
from functools import wraps
import torch

_MASK = ContextVar('own_soup_head_mask', default=None)
_COUNTS = {'engaged': 0, 'fallback': 0, 'saved_head_rows': 0}

def counters():
    return dict(_COUNTS)

def _geometry(mask, labels):
    if (not isinstance(mask, torch.Tensor) or not isinstance(labels, torch.Tensor)
        or mask.shape != labels.shape or mask.ndim != 2 or mask.shape[0] < 2
        or mask.shape[1] < 2 or mask.dtype not in
        (torch.bool, torch.uint8, torch.int32, torch.int64)):
        return None
    valid = mask != 0
    lengths = valid.sum(-1)
    if (not bool(lengths.gt(0).all().item())
        or not bool((valid[:, :-1] >= valid[:, 1:]).all().item())
        or not bool(((labels == -100) | valid).all().item())):
        return None
    selected = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
    if selected.numel() == valid.numel():
        return None
    start = torch.cat([lengths.new_zeros((1,)), lengths.cumsum(0)[:-1]], dim=0)
    return selected, start

def _expand(compact, indices, batch, steps):
    shape = (batch * steps,) + tuple(compact.shape[2:])
    out = compact.new_zeros(shape)
    out.index_copy_(0, indices, compact.reshape((indices.numel(),) + tuple(compact.shape[2:])))
    return out.view(batch, steps, *compact.shape[2:])

def patch_valid_head():
    from . import integration, streamed_head
    old_forward = integration._forward_streamed
    if getattr(old_forward, '_soup_own_valid_head_context', False):
        return True
    old_ce = streamed_head.streamed_linear_ce

    @wraps(old_forward)
    def forward(causal_lm, *args, **kwargs):
        from .hero3_pack import _LABEL_CONTEXT
        mask_token = _MASK.set(kwargs.get('attention_mask'))
        # Our integrated loss calls causal_lm.model directly, bypassing
        # Qwen2ForCausalLM.forward. Explicitly activate the packed decoder.
        label_token = _LABEL_CONTEXT.set(kwargs.get('labels'))
        try:
            return old_forward(causal_lm, *args, **kwargs)
        finally:
            _LABEL_CONTEXT.reset(label_token)
            _MASK.reset(mask_token)

    @wraps(old_ce)
    def reduced(hidden, weight, labels, num_items_in_batch=None,
                chunk_size=256, projection='torch', sparse_backward=True, **kwargs):
        mask = _MASK.get()
        eligible = (
            isinstance(hidden, torch.Tensor) and hidden.is_cuda and hidden.requires_grad
            and hidden.ndim == 3 and hidden.dtype in (torch.float16, torch.float32)
            and isinstance(labels, torch.Tensor) and labels.dtype == torch.int64
            and hidden.shape[:2] == labels.shape and labels.device == hidden.device
            and isinstance(mask, torch.Tensor) and mask.device == labels.device
            and weight.ndim == 2 and not weight.requires_grad
            and projection == 'torch' and not kwargs
        )
        if not eligible:
            _COUNTS['fallback'] += 1
            return old_ce(hidden, weight, labels, num_items_in_batch,
                          chunk_size, projection, sparse_backward, **kwargs)
        packed = _geometry(mask, labels)
        if packed is None:
            _COUNTS['fallback'] += 1
            return old_ce(hidden, weight, labels, num_items_in_batch,
                          chunk_size, projection, sparse_backward, **kwargs)
        idx, starts = packed
        batch, steps, width = hidden.shape
        compact_hidden = hidden.reshape(-1, width).index_select(0, idx)[None, ...]
        compact_labels = labels.flatten().index_select(0, idx).clone()
        # Original row b's last token must NOT predict row b+1's first
        # label. Each packed segment's first label is therefore ignored.
        compact_labels.index_fill_(0, starts.long(), -100)
        small = old_ce(compact_hidden, weight, compact_labels[None, :],
                       num_items_in_batch, chunk_size, projection,
                       sparse_backward)
        out = streamed_head.HeadResult(
            loss=small.loss,
            token_loss=_expand(small.token_loss, idx, batch, steps),
            entropy=_expand(small.entropy, idx, batch, steps),
            predictions=_expand(small.predictions, idx, batch, steps),
        )
        _COUNTS['engaged'] += 1
        _COUNTS['saved_head_rows'] += batch * steps - idx.numel()
        return out

    forward._soup_own_valid_head_context = True
    reduced._soup_own_valid_head = True
    integration._forward_streamed = forward
    streamed_head.streamed_linear_ce = reduced
    return True
