"""Bound native exceptional-row replay while CE scheduling uses full-M grids.

These are ordinary native PyTorch operations, not original numerical kernels.
The 128-row ceiling preserves the prior tile's full-vocabulary transient bound.
No selected-row subset totals are combined: every row sees the real global q.
"""
import torch
import torch.nn.functional as F

NATIVE_ROWS = 128


def _summaries(logits, target):
    public = logits.float()
    nll = F.cross_entropy(public, target, ignore_index=-100, reduction='none')
    logp = F.log_softmax(public, -1)
    entropy = -(logp.exp() * logp).sum(-1)
    predictions = public.argmax(-1)
    return nll, entropy, predictions


def summaries(logits, target):
    """Preserve native NaN/ignored-target/first-index semantics at <=128 rows."""
    with torch.no_grad():
        if logits.shape[0] <= NATIVE_ROWS:
            return _summaries(logits, target)
        n = logits.shape[0]
        nll = torch.empty(n, dtype=torch.float32, device=logits.device)
        entropy = torch.empty_like(nll)
        predictions = torch.empty(n, dtype=torch.int64, device=logits.device)
        for start in range(0, n, NATIVE_ROWS):
            stop = min(start + NATIVE_ROWS, n)
            values = _summaries(logits[start:stop], target[start:stop])
            nll[start:stop], entropy[start:stop], predictions[start:stop] = values
        return nll, entropy, predictions


def _gradient(logits, target, q):
    with torch.enable_grad():
        work = logits.detach().requires_grad_(True)
        total = F.cross_entropy(work.float(), target, ignore_index=-100, reduction='sum')
        gradient, = torch.autograd.grad(total, work, grad_outputs=q)
    return gradient


def gradient(logits, target, q):
    """True normalized/scaled upstream enters CE before the native half cast."""
    if logits.shape[0] <= NATIVE_ROWS:
        return _gradient(logits, target, q)
    result = torch.empty_like(logits)
    for start in range(0, logits.shape[0], NATIVE_ROWS):
        stop = min(start + NATIVE_ROWS, logits.shape[0])
        result[start:stop] = _gradient(logits[start:stop], target[start:stop], q)
    return result


def selected_values(logits, target, selected, native_rows):
    """Disjoint native row replacement; never allocate all selected logits."""
    values = selected.clone()
    for start in range(0, native_rows.numel(), NATIVE_ROWS):
        rows = native_rows[start:start + NATIVE_ROWS]
        native_logits = logits.index_select(0, rows)
        native_target = target.index_select(0, rows)
        native_loss = F.cross_entropy(native_logits.float(), native_target,
                                      reduction='none', ignore_index=-100)
        values.index_copy_(0, rows, -native_loss)
    return values


def scatter_gradient(logits, target, q, native_rows, destination):
    """Fill only native-selected rows in the final dZ; custom rows stay intact."""
    for start in range(0, native_rows.numel(), NATIVE_ROWS):
        rows = native_rows[start:start + NATIVE_ROWS]
        native_logits = logits.index_select(0, rows)
        native_target = target.index_select(0, rows)
        native_gradient = _gradient(native_logits, native_target, q)
        destination.index_copy_(0, rows, native_gradient)
