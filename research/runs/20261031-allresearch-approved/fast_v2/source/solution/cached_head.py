"""Private native FP16 logits and bounded native statistics/CE backward.

This composes the user's Round 7 native-head/control-gradient result with the
original tiled statistics from their build candidate.  It does NOT replay the
head projection, change its GEMM geometry, or reconstruct its scaled gradient
with an approximate exponential.  The ordinary head autograd graph remains
responsible for the single full-shape hidden-gradient GEMM.

Native forward statistics are the default because custom reduction trees can
differ outside a fixed near-saturation guard. The earlier original Triton
summaries remain an explicit "guarded" experiment; the native path does not
execute or discard them. No GPU runtime or peak-memory result is implied.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .streamed_head import (
    HeadResult,
    NATIVE_NORMALIZER_THRESHOLD,
    VOCAB_TILE,
    _native_row_statistics,
    soup_head_partial_statistics,
)


@triton.jit
def soup_cached_head_finish(
    scores, targets, partial_max, partial_sum, partial_centered_sum,
    partial_argmax, token_loss, token_entropy, predictions, exceptional,
    VOCAB: tl.constexpr, TILES: tl.constexpr, MERGE_BLOCK: tl.constexpr,
    SATURATION_THRESHOLD: tl.constexpr,
):
    """Merge our tile summaries; store only consumer outputs and a guard."""
    row = tl.program_id(0)
    tile = tl.arange(0, MERGE_BLOCK)
    active = tile < TILES
    offset = row * TILES + tile
    local_max = tl.load(partial_max + offset, active, other=-float("inf"))
    maximum = tl.max(local_max, 0)
    delta = tl.where(active, local_max - maximum, 0.0)
    factor = tl.exp(delta)
    local_sum = tl.load(partial_sum + offset, active, other=0.0)
    local_centered = tl.load(partial_centered_sum + offset, active, other=0.0)
    total = tl.sum(factor * local_sum, 0)
    centered = tl.sum(factor * (local_centered + delta * local_sum), 0)
    log_total = tl.log(total)
    entropy = log_total - centered / total
    local_argmax = tl.load(partial_argmax + offset, active, other=2147483647)
    prediction = tl.min(tl.where(active & (local_max == maximum),
                                 local_argmax, 2147483647), 0)
    target = tl.load(targets + row)
    supervised = target != -100
    in_bounds = (target >= 0) & (target < VOCAB)
    # Invalid labels are rejected by the asynchronous host wrapper assertion;
    # this bound also makes the pointer safe independently of its scheduling.
    safe_target = tl.minimum(tl.maximum(target, 0), VOCAB - 1)
    target_score = tl.load(scores + row * VOCAB + safe_target,
                           supervised & in_bounds, other=0.0).to(tl.float32)
    ce = tl.where(supervised, -((target_score - maximum) - log_total), 0.0)
    ce = tl.where(supervised & ~in_bounds, float("nan"), ce)
    finite_max = (maximum == maximum) & (tl.abs(maximum) != float("inf"))
    finite_log = (log_total == log_total) & (tl.abs(log_total) != float("inf"))
    finite_entropy = (entropy == entropy) & (tl.abs(entropy) != float("inf"))
    guarded = ((log_total <= SATURATION_THRESHOLD) | ~finite_max
               | ~finite_log | ~finite_entropy)
    tl.store(token_loss + row, ce)
    tl.store(token_entropy + row, entropy)
    tl.store(predictions + row, prediction)
    tl.store(exceptional + row, guarded)


def _guarded_summaries(flat, targets, chunk_size):
    """Experimental custom reductions; the guard cannot prove native parity."""
    rows, vocab = flat.shape
    tiles = triton.cdiv(vocab, VOCAB_TILE)
    maximum = torch.empty((rows, tiles), device=flat.device, dtype=torch.float32)
    total = torch.empty_like(maximum)
    centered = torch.empty_like(maximum)
    winner = torch.empty((rows, tiles), device=flat.device, dtype=torch.int32)
    loss = torch.empty(rows, device=flat.device, dtype=torch.float32)
    entropy = torch.empty_like(loss)
    predictions = torch.empty(rows, device=flat.device, dtype=torch.long)
    exceptional = torch.empty(rows, device=flat.device, dtype=torch.bool)
    soup_head_partial_statistics[(rows, tiles)](
        flat, maximum, total, centered, winner,
        VOCAB=vocab, TILES=tiles, BLOCK=VOCAB_TILE, num_warps=4,
    )
    soup_cached_head_finish[(rows,)](
        flat, targets, maximum, total, centered, winner,
        loss, entropy, predictions, exceptional,
        VOCAB=vocab, TILES=tiles, MERGE_BLOCK=triton.next_power_of_2(tiles),
        SATURATION_THRESHOLD=NATIVE_NORMALIZER_THRESHOLD, num_warps=4,
    )
    # Native entropy deliberately preserves NaN from e.g. 0*log(0), including
    # masked/ignored rows.  Near-saturation falls back for forward accuracy.
    # This size-discovery synchronization is a real cost to measure on T4.
    exceptional_rows = exceptional.nonzero(as_tuple=False).flatten()
    for start in range(0, exceptional_rows.numel(), chunk_size):
        selected = exceptional_rows[start:start + chunk_size]
        native = _native_row_statistics(flat.index_select(0, selected),
                                         targets.index_select(0, selected))
        for output, values in zip((loss, entropy, predictions), native):
            output.index_copy_(0, selected, values)
    return loss, entropy, predictions


def _summaries(flat, targets, chunk_size, numerics="native"):
    """Compute every consumer from native row statistics in bounded chunks.

    Keep the original half logits in place. FP32 log-softmax and entropy
    intermediates exist only for one row chunk, with no projection replay,
    exceptional-row size discovery, or discarded custom kernel work.
    """
    if numerics == "guarded":
        return _guarded_summaries(flat, targets, chunk_size)
    rows = flat.shape[0]
    loss = torch.empty(rows, device=flat.device, dtype=torch.float32)
    entropy = torch.empty_like(loss)
    predictions = torch.empty(rows, device=flat.device, dtype=torch.long)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        native = _native_row_statistics(flat[start:stop], targets[start:stop])
        for output, values in zip((loss, entropy, predictions), native):
            output[start:stop].copy_(values)
    return loss, entropy, predictions


class _CachedLogitsCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, num_items_in_batch, chunk_size, numerics, backward_mode):
        batch, length, vocab = logits.shape
        flat = logits.view(-1, vocab)
        shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
        targets = shifted.view(-1)
        # Value checking stays asynchronous on CUDA; no host .item() here.
        torch._assert_async(((targets == -100) | ((targets >= 0) & (targets < vocab))).all(),
                            "causal target index is out of bounds")
        token_loss, entropy, predictions = _summaries(flat, targets, chunk_size, numerics)
        if num_items_in_batch is None:
            divisor = (targets != -100).sum().to(torch.float32)
        else:
            divisor = torch.as_tensor(num_items_in_batch, device=logits.device,
                                      dtype=torch.float32).detach().reshape(())
        # Never clamp: native all-ignored mean CE is NaN with zero finite-row
        # gradients; explicit zero/negative divisors preserve their semantics.
        loss = token_loss.sum() / divisor
        ctx.save_for_backward(logits, targets, divisor)
        ctx.chunk_size = chunk_size
        ctx.backward_mode = backward_mode
        ctx.set_materialize_grads(False)
        token_loss = token_loss.reshape(batch, length)
        entropy = entropy.reshape(batch, length)
        predictions = predictions.reshape(batch, length)
        ctx.mark_non_differentiable(token_loss, entropy, predictions)
        return loss, token_loss, entropy, predictions

    @staticmethod
    def backward(ctx, grad_loss, _token_loss, _entropy, _predictions):
        if grad_loss is None:
            return None, None, None, None, None, None
        logits, targets, divisor = ctx.saved_tensors
        flat = logits.reshape(-1, logits.shape[-1])
        higher_order = torch.is_grad_enabled()
        if ctx.backward_mode == "native_fused" and not higher_order:
            from .cached_ce_backward import native_logsoftmax_ce_backward
            dz = native_logsoftmax_ce_backward(flat, targets, divisor,
                                               grad_loss.float(), ctx.chunk_size)
            return dz.reshape(logits.shape), None, None, None, None, None
        # Ordinary training writes one full FP16 dZ; native FP32 CE workspaces
        # exist for just one chunk.  The original head receives this tensor and
        # executes its unmodified full-shape backward GEMM once.
        dz = None if higher_order else torch.empty_like(flat)
        pieces = [] if higher_order else None
        with torch.enable_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
            for start in range(0, flat.shape[0], ctx.chunk_size):
                stop = min(start + ctx.chunk_size, flat.shape[0])
                part = flat[start:stop]
                leaf = part if higher_order else part.detach().requires_grad_(True)
                native_loss = F.cross_entropy(leaf.float(), targets[start:stop],
                                               ignore_index=-100, reduction="sum") / divisor
                gradient, = torch.autograd.grad(native_loss, leaf,
                                                 grad_outputs=grad_loss.float(),
                                                 create_graph=higher_order)
                if higher_order:
                    pieces.append(gradient)
                else:
                    dz[start:stop].copy_(gradient)
                # In the ordinary path no chunk graph survives to the next
                # iteration.  Ignored nonfinite rows are never skipped.
        if higher_order:
            dz = torch.cat(pieces, dim=0)
        return dz.reshape(logits.shape), None, None, None, None, None


def cached_logits_ce(logits, labels, num_items_in_batch=None, chunk_size=128,
                     numerics="native", backward_mode="native"):
    """Return differentiable scalar loss and detached true per-token metrics.

    ``logits`` must be the unchanged native head's contiguous FP16 (B,T,V)
    output.  No copy of that tensor is retained.  ``labels`` use causal shift
    and ignore index -100; the last position contributes entropy but no CE.
    Explicit denominators are global, not per chunk.  Actual upstream scaling
    is applied by native autograd before the FP32-to-FP16 gradient cast.
    Native statistics are the default; "guarded" explicitly selects the
    custom-forward experiment, whose threshold cannot establish parity.
    The optional "native_fused" backward retains native FP32 log-softmax and
    replaces its sparse-NLL gradient consumer with an original Triton kernel.
    Higher-order differentiation always retains the native autograd path.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or min(logits.shape) < 1:
        raise ValueError("expected nonempty logits (B,T,V) and labels (B,T)")
    if logits.dtype != torch.float16 or labels.dtype != torch.long:
        raise TypeError("cached head requires FP16 logits and int64 labels")
    if logits.device != labels.device or not logits.is_contiguous():
        raise ValueError("cached logits must be contiguous and share the labels' device")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(num_items_in_batch, torch.Tensor) and num_items_in_batch.numel() != 1:
        raise ValueError("num_items_in_batch must be scalar")
    if numerics not in ("native", "guarded"):
        raise ValueError("numerics must be 'native' or 'guarded'")
    if backward_mode not in ("native", "native_fused"):
        raise ValueError("backward_mode must be 'native' or 'native_fused'")
    return HeadResult(*_CachedLogitsCrossEntropy.apply(
        logits, labels, num_items_in_batch, chunk_size, numerics, backward_mode))
