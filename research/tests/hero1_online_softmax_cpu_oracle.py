"""Pure CPU mathematical oracle for original Hero1 online vocabulary statistics.

This validates the stable online max/sum/centered-entropy recurrence against
PyTorch's FP32 log_softmax on actual rounded FP16 logits. It does NOT execute
Triton and does not establish a GPU benchmark, original-kernel share, or
challenge admission.
"""
import numpy as np
import torch
import torch.nn.functional as F


def online_reference(scores: np.ndarray, targets: np.ndarray, block: int = 2048):
    rows, vocab = scores.shape
    out = np.empty((rows, 5), dtype=np.float64)
    for row in range(rows):
        max_seen, total, centered, winner = -float("inf"), 0.0, 0.0, 2**31 - 1
        for left in range(0, vocab, block):
            values = scores[row, left:left + block].astype(np.float32)
            blockmax = float(values.max())
            new_max = max(max_seen, blockmax)
            factor = float(np.exp(max_seen - new_max)) if np.isfinite(max_seen) else 0.0
            weights = np.exp(values - new_max)
            block_total = float(weights.sum(dtype=np.float32))
            block_centered = float((weights * (values - new_max)).sum(dtype=np.float32))
            if np.isfinite(max_seen):
                centered = factor * (centered + (max_seen - new_max) * total) + block_centered
            else:
                centered = block_centered
            total = factor * total + block_total
            candidate = left + int(np.flatnonzero(values == blockmax)[0])
            if blockmax > max_seen:
                winner = candidate
            elif blockmax == max_seen:
                winner = min(winner, candidate)
            max_seen = new_max
        log_norm = float(np.log(total))
        target = int(targets[row])
        target_logit = float(np.float32(scores[row, target])) if target != -100 else 0.0
        ce = -((target_logit - max_seen) - log_norm) if target != -100 else 0.0
        h = log_norm - centered / total
        out[row] = (ce, h, winner, max_seen, log_norm)
    return out


def run_case(shape, scale=1.0, shift=0.0, ties=False):
    torch.manual_seed(20261031)
    x = (torch.randn(shape) * scale + shift).half()
    if ties:
        x[:, 0] = x[:, 0] + 35
        x[:, 11] = x[:, 0]   # exact first-index tie
    y = torch.randint(0, shape[1], (shape[0],), dtype=torch.long)
    y[::7] = -100
    got = online_reference(x.numpy(), y.numpy())
    logp = F.log_softmax(x.float(), dim=-1)
    ce = F.nll_loss(logp, y, reduction="none", ignore_index=-100)
    entropy = -(logp.exp() * logp).sum(-1)
    predicted = x.argmax(-1)
    ce_err = float(np.max(np.abs(got[:, 0] - ce.numpy())))
    h_err = float(np.max(np.abs(got[:, 1] - entropy.numpy())))
    ties_err = int(np.sum(got[:, 2] != predicted.numpy()))
    assert ce_err < 2.5e-3 and h_err < 5e-3 and ties_err == 0
    assert np.all(got[y.numpy() == -100, 0] == 0)
    return {"shape": shape, "ce_max_abs": ce_err, "entropy_max_abs": h_err,
            "argmax_mismatches": ties_err}


if __name__ == "__main__":
    for configuration in [
        ((8, 4096), 1.0, 0.0, False),
        ((32, 151936), 0.8, 0.0, False),
        ((16, 151936), 2.0, 10.0, False),
        ((8, 151936), 1.0, 0.0, True),
    ]:
        print("PASS", run_case(*configuration))
    print("CPU recurrence passes, GPU original Triton kernel untested.")
