"""Hero1 independent one-pass FP16-logit vocabulary statistics.

Algorithmic inspiration: public online-softmax and fused CE/entropy/argmax designs;
see the project research references. This is an original implementation, not
copied from a competitor kernel or the challenge starter. This module is an
UNINTEGRATED COMPONENT PROTOTYPE, not a qualifying training solution.

For each row we maintain stable (max, sum_exp, centered_weighted_sum),
and the first-index argmax. The centered accumulation avoids subtracting
large nearly equal terms when logits have common offsets. The operation
reads the ORIGINAL FP16 logits and respects ignore_index=-100.

No weight or optimizer changes, no changes to training data or metrics.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def hero1_online_vocab_stats(
    SCORES, TARGETS,
    CE_OUT, ENTROPY_OUT, ARGMAX_OUT, MAX_OUT, LOGNORM_OUT,
    VOCAB: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mx = tl.full((), -float("inf"), tl.float32)
    total = tl.full((), 0.0, tl.float32)
    centered = tl.full((), 0.0, tl.float32)
    winner = tl.full((), 2147483647, tl.int32)



    for tile in range(tl.cdiv(VOCAB, BLOCK)):
        idx = tile * BLOCK + lane
        inside = idx < VOCAB
        x = tl.load(SCORES + row * VOCAB + idx, inside, other=-float("inf")).to(tl.float32)
        batch_max = tl.max(x, 0)
        new_mx = tl.maximum(mx, batch_max)
        has_old = mx != -float("inf")
        delta_old = tl.where(has_old, mx - new_mx, 0.0)
        factor = tl.where(has_old, tl.exp(delta_old), 0.0)
        expval = tl.where(inside, tl.exp(x - new_mx), 0.0)
        part_total = tl.sum(expval, 0)
        part_centered = tl.sum(tl.where(inside, expval * (x - new_mx), 0.0), 0)
        new_centered = factor * (centered + delta_old * total) + part_centered
        new_total = factor * total + part_total

        this_winner = tl.min(tl.where(inside & (x == batch_max), idx, 2147483647), 0)
        winner = tl.where(batch_max > mx, this_winner,
                          tl.where(batch_max == mx, tl.minimum(winner, this_winner), winner))
        mx = new_mx
        total = new_total
        centered = new_centered

    log_norm = tl.log(total)
    entropy = log_norm - centered / total
    label = tl.load(TARGETS + row)
    supervised = label != -100
    safe_label = tl.where(supervised, label, 0)
    target_val = tl.load(SCORES + row * VOCAB + safe_label, supervised, other=0.).to(tl.float32)
    ce = tl.where(supervised, -((target_val - mx) - log_norm), 0.0)
    tl.store(CE_OUT + row, ce)
    tl.store(ENTROPY_OUT + row, entropy)
    tl.store(ARGMAX_OUT + row, winner)
    tl.store(MAX_OUT + row, mx)
    tl.store(LOGNORM_OUT + row, log_norm)


def online_vocab_stats(logits, labels, *, block=2048):
    if not (isinstance(logits, torch.Tensor) and logits.is_cuda and
            logits.dtype == torch.float16 and logits.ndim == 2 and
            logits.is_contiguous() and isinstance(labels, torch.Tensor) and
            labels.is_cuda and labels.dtype == torch.long and
            labels.ndim == 1 and labels.numel() == logits.shape[0] and
            logits.shape[1] > 0 and block in (1024, 2048, 4096)):
        raise ValueError("Hero1 diagnostic supports contiguous CUDA FP16 [rows,vocab] and int64 targets only")
    rows, vocab = logits.shape
    ce = torch.empty(rows, device=logits.device, dtype=torch.float32)
    h = torch.empty_like(ce)
    argmax = torch.empty(rows, device=logits.device, dtype=torch.long)
    mx = torch.empty_like(ce)
    lognorm = torch.empty_like(ce)
    hero1_online_vocab_stats[(rows,)](
        logits, labels, ce, h, argmax, mx, lognorm,
        VOCAB=vocab, BLOCK=block, num_warps=4, enable_fp_fusion=False,
    )
    return ce, h, argmax, mx, lognorm
