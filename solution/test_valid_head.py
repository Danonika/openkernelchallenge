"""CPU mathematical regression oracle for the original valid-row Soup head.

This is NOT GPU perf/admission evidence. It checks causal boundaries, Rewind
loss geometry, masked entropy/accuracy and input gradients.
"""
import torch
import torch.nn.functional as F
from .valid_head import _geometry, _expand

def _statistics(h, w, labels):
    b, t, d = h.shape
    logits = h @ w.t()
    target = torch.cat((labels[:, 1:], labels.new_full((b, 1), -100)), 1)
    ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                         ignore_index=-100, reduction='none').reshape(b, t)
    p = logits.softmax(-1)
    entropy = -(p * p.log()).sum(-1)
    return ce.sum() / (target != -100).sum(), ce, entropy, logits.argmax(-1)

def test_valid_row_projection_matches_full_50_random_batches():
    for seed in range(50):
        torch.manual_seed(seed)
        b, t, d, v = 4, 12, 16, 48
        h = torch.randn(b, t, d, requires_grad=True)
        w = torch.randn(v, d)
        mask = torch.zeros((b, t), dtype=torch.int64)
        for i in range(b):
            mask[i, :torch.randint(2, t+1, ()).item()] = 1
        labels = torch.randint(v, (b, t))
        labels[mask == 0] = -100
        for i in range(b):
            labels[i, :max(1, mask[i].sum().item() // 2)] = -100
        full = _statistics(h, w, labels)
        grad_full = torch.autograd.grad(full[0], h)[0]
        idx, starts = _geometry(mask, labels)
        compact_labels = labels.flatten().index_select(0, idx).clone()
        compact_labels.index_fill_(0, starts.long(), -100)
        packed = h.reshape(b*t, d).index_select(0, idx).unsqueeze(0)
        small = _statistics(packed, w, compact_labels.unsqueeze(0))
        restored = [small[0]] + [
            _expand(y, idx, b, t) for y in small[1:]
        ]
        grad_small = torch.autograd.grad(restored[0], h)[0]
        torch.testing.assert_close(restored[0], full[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(restored[1], full[1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close((restored[2]*mask).sum(),
                                   (full[2]*mask).sum(), rtol=1e-5, atol=1e-5)
        assert torch.equal(restored[3][mask.bool()], full[3][mask.bool()])
        torch.testing.assert_close(grad_small, grad_full, rtol=1e-5, atol=2e-6)
