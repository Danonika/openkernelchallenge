"""Optional native log-softmax with original fused statistics consumers.

This module is not installed by solution.apply() and changes no source
settings. It is an alternative forward-statistics helper for the retained
native head. Projection and scaled CE backward remain the caller's native
operations; this helper deliberately creates no autograd graph.

Native FP32 log-softmax fixes the numerical normalization. Original tiled
kernels gather NLL directly from those values, compute -sum(exp(logp)*logp),
and find the first argmax of the ORIGINAL scores. They avoid full exp/product
intermediate tensors. Entropy reduction order can still differ from Torch.

These kernels are original implementations of those elementary consumers,
not copied ready-made kernels. Physical CUDA validation and timing are
required before selecting this optional path.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


VOCAB_TILE = 2048


@triton.jit
def soup_native_logp_consumers_partial(
    logp, scores, partial_entropy, partial_max, partial_argmax,
    VOCAB: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    valid = col < VOCAB
    address = row.to(tl.int64) * VOCAB + col
    # Zero is a safe tail value: exp(0)*0 contributes exactly zero. A valid
    # -Inf logp must retain 0*(-Inf)=NaN, including on ignored-label rows.
    lp = tl.load(logp + address, valid, other=0.0)
    contribution = -(libdevice.exp(lp) * lp)
    entropy = tl.sum(tl.where(valid, contribution, 0.0), 0)

    value = tl.load(scores + address, valid, other=-float("inf")).to(tl.float32)
    is_nan = valid & (value != value)
    first_nan = tl.min(tl.where(is_nan, col, 2147483647), 0)
    maximum = tl.max(tl.where(valid & ~is_nan, value, -float("inf")), 0)
    first_max = tl.min(tl.where(valid & ~is_nan & (value == maximum), col, 2147483647), 0)
    # A NaN wins argmax even if an earlier column contains +Inf. Retain the
    # first NaN index separately from ordinary max ties in the merged state.
    has_nan = first_nan != 2147483647
    chosen_max = tl.where(has_nan, float("nan"), maximum)
    chosen_index = tl.where(has_nan, first_nan, first_max)
    output = row.to(tl.int64) * TILES + tile
    tl.store(partial_entropy + output, entropy)
    tl.store(partial_max + output, chosen_max)
    tl.store(partial_argmax + output, chosen_index)


@triton.jit
def soup_native_logp_consumers_finish(
    logp, targets, partial_entropy, partial_max, partial_argmax,
    token_loss, token_entropy, predictions,
    VOCAB: tl.constexpr, TILES: tl.constexpr, MERGE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, MERGE_BLOCK)
    active = tile < TILES
    offset = row.to(tl.int64) * TILES + tile
    parts = tl.load(partial_entropy + offset, active, other=0.0)
    entropy = tl.sum(parts, 0)
    maximums = tl.load(partial_max + offset, active, other=-float("inf"))
    indices = tl.load(partial_argmax + offset, active, other=2147483647)
    nan_tile = active & (maximums != maximums)
    first_nan = tl.min(tl.where(nan_tile, indices, 2147483647), 0)
    maximum = tl.max(tl.where(active & ~nan_tile, maximums, -float("inf")), 0)
    first_max = tl.min(tl.where(active & ~nan_tile & (maximums == maximum), indices, 2147483647), 0)
    prediction = tl.where(first_nan != 2147483647, first_nan, first_max)

    target = tl.load(targets + row)
    supervised = target != -100
    in_bounds = (target >= 0) & (target < VOCAB)
    # This makes the kernel memory-safe independently of the asynchronous
    # target-range assertion in the public wrapper.
    safe_target = tl.minimum(tl.maximum(target, 0), VOCAB - 1)
    selected = tl.load(logp + row.to(tl.int64) * VOCAB + safe_target,
                       supervised & in_bounds, other=0.0)
    loss = tl.where(supervised, -selected, 0.0)
    loss = tl.where(supervised & ~in_bounds, float("nan"), loss)
    tl.store(token_loss + row, loss)
    tl.store(token_entropy + row, entropy)
    tl.store(predictions + row, prediction)


def _consume_native_logp(logp, scores, targets, loss, entropy, predictions):
    """Internal kernel boundary; the public helper validates all metadata."""
    rows, vocab = scores.shape
    tiles = triton.cdiv(vocab, VOCAB_TILE)
    partial_entropy = torch.empty((rows, tiles), device=scores.device, dtype=torch.float32)
    partial_max = torch.empty_like(partial_entropy)
    partial_argmax = torch.empty((rows, tiles), device=scores.device, dtype=torch.int32)
    soup_native_logp_consumers_partial[(rows, tiles)](
        logp, scores, partial_entropy, partial_max, partial_argmax,
        VOCAB=vocab, TILES=tiles, BLOCK=VOCAB_TILE,
        num_warps=4, enable_fp_fusion=False)
    soup_native_logp_consumers_finish[(rows,)](
        logp, targets, partial_entropy, partial_max, partial_argmax,
        loss, entropy, predictions,
        VOCAB=vocab, TILES=tiles, MERGE_BLOCK=triton.next_power_of_2(tiles),
        num_warps=4, enable_fp_fusion=False)


def native_head_statistics(scores, targets, chunk_size=128):
    """Return native NLL, fused entropy and first original-score argmax.

    Input is a contiguous (rows, vocabulary) FP16/FP32 score tensor and int64
    target vector. Ignore index is -100. Every row contributes entropy and an
    argmax, including ignored rows. NaN/Inf behavior is preserved rather than
    masked by supervision. Target errors are asserted asynchronously on CUDA.

    FP32 cast/log-softmax workspaces are bounded by chunk_size. No logits,
    exponential, or entropy-product matrix is retained in an autograd graph.
    CUDA uses libdevice.exp; the CPU-interpreter test fixture explicitly
    substitutes its supported exp primitive because externs are unsupported.
    """
    if scores.ndim != 2 or targets.ndim != 1 or scores.shape[0] != targets.shape[0]:
        raise ValueError("expected scores (rows,vocabulary) and one target per row")
    if min(scores.shape) < 1 or scores.shape[1] > 2147483647:
        raise ValueError("expected nonempty scores and int32-addressable vocabulary")
    if scores.dtype not in (torch.float16, torch.float32) or targets.dtype != torch.long:
        raise TypeError("expected FP16/FP32 scores and int64 targets")
    if scores.device != targets.device or not scores.is_contiguous() or not targets.is_contiguous():
        raise ValueError("scores and targets must be contiguous on the same device")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not scores.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("requires CUDA; CPU diagnostics need TRITON_INTERPRET=1")
    rows, vocab = scores.shape
    torch._assert_async(((targets == -100) | ((targets >= 0) & (targets < vocab))).all(),
                        "native head target index is out of bounds")
    with torch.no_grad():
        loss = torch.empty(rows, device=scores.device, dtype=torch.float32)
        entropy = torch.empty_like(loss)
        predictions = torch.empty(rows, device=scores.device, dtype=torch.long)
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            original_scores = scores[start:stop]
            logp = F.log_softmax(original_scores.float(), dim=-1)
            _consume_native_logp(logp, original_scores, targets[start:stop],
                                 loss[start:stop], entropy[start:stop], predictions[start:stop])
            del logp
    return loss, entropy, predictions
