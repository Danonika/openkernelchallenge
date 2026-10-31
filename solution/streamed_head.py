"""Bounded-memory causal language-model loss and required token statistics.

The output projection has explicit PyTorch and original Triton implementations.
Original Triton kernels reduce its FP16 output in vocabulary tiles, merge the
reductions, and form the logit gradient. No whole-batch vocabulary tensor is retained. Backward recomputes
one token chunk at a time so the actual upstream GradScaler scale is applied
*before* converting the logit gradient to FP16.

This implementation is derived from the softmax / cross-entropy identities, not
copied from another kernel implementation. The general motivation of chunking
the linear loss comes from the papers and projects cited in the solution README.
Only a frozen output weight and ordinary causal cross-entropy are supported.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


VOCAB_TILE = 4096
GRAD_BLOCK = 4096
DEFAULT_CHUNK_SIZE = 256
# Different FP32 reduction trees are relatively inaccurate when p(max) is
# almost one. Such rows use the backend's native log-softmax in bounded chunks.
# This fixed numerical guard is independent of datasets, row identities or runs.
NATIVE_NORMALIZER_THRESHOLD = 0.01


@triton.jit
def soup_head_partial_statistics(
    scores,
    partial_max,
    partial_sum,
    partial_centered_sum,
    partial_argmax,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One token / vocabulary tile: stable sufficient softmax statistics."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    active = column < VOCAB
    value = tl.load(scores + row * VOCAB + column, active, other=-float("inf")).to(tl.float32)
    maximum = tl.max(value, axis=0)
    centered = tl.where(active, value - maximum, 0.0)
    exponential = tl.where(active, tl.exp(centered), 0.0)
    total = tl.sum(exponential, axis=0)
    centered_total = tl.sum(exponential * centered, axis=0)
    # PyTorch argmax breaks equal-score ties by the first vocabulary index.
    winner = tl.min(tl.where(active & (value == maximum), column, 2147483647), axis=0)
    offset = row * TILES + tile
    tl.store(partial_max + offset, maximum)
    tl.store(partial_sum + offset, total)
    tl.store(partial_centered_sum + offset, centered_total)
    tl.store(partial_argmax + offset, winner)


@triton.jit
def soup_head_merge_statistics(
    scores,
    targets,
    partial_max,
    partial_sum,
    partial_centered_sum,
    partial_argmax,
    token_loss,
    token_entropy,
    predictions,
    logsumexp,
    row_max,
    log_normalizer,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
):
    """Merge tiles without constructing vocabulary-sized FP32 tensors."""
    row = tl.program_id(0)
    tile = tl.arange(0, MERGE_BLOCK)
    active = tile < TILES
    offset = row * TILES + tile
    local_max = tl.load(partial_max + offset, active, other=-float("inf"))
    maximum = tl.max(local_max, axis=0)
    delta = tl.where(active, local_max - maximum, 0.0)
    rescale = tl.exp(delta)
    local_sum = tl.load(partial_sum + offset, active, other=0.0)
    local_centered_sum = tl.load(partial_centered_sum + offset, active, other=0.0)
    total = tl.sum(rescale * local_sum, axis=0)
    centered_total = tl.sum(rescale * (local_centered_sum + delta * local_sum), axis=0)
    log_total = tl.log(total)
    lse = maximum + log_total
    # H = log(sum(exp(z - m))) - E[z - m]. This form avoids subtracting
    # two large, nearly equal values when a common offset is added to logits.
    entropy = log_total - centered_total / total
    local_winner = tl.load(partial_argmax + offset, active, other=2147483647)
    winner = tl.min(tl.where(active & (local_max == maximum), local_winner, 2147483647), axis=0)
    target = tl.load(targets + row)
    supervised = target != -100
    safe_target = tl.where(supervised, target, 0)
    target_score = tl.load(scores + row * VOCAB + safe_target, supervised, other=0.0).to(tl.float32)
    # Do not subtract target_score from the rounded (maximum + log_total).
    # Native log_softmax retains the centered difference before normalization.
    ce = tl.where(supervised, -((target_score - maximum) - log_total), 0.0)
    tl.store(token_loss + row, ce)
    tl.store(token_entropy + row, entropy)
    tl.store(predictions + row, winner)
    tl.store(logsumexp + row, lse)
    tl.store(row_max + row, maximum)
    tl.store(log_normalizer + row, log_total)


@triton.jit(do_not_specialize=["n_elements"])
def soup_head_logit_gradient(
    scores_and_gradient,
    targets,
    row_max,
    log_normalizer,
    loss_gradient_over_denominator,
    n_elements,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Overwrite a recomputed FP16 score chunk with its FP16 gradient."""
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = position < n_elements
    row = position // VOCAB
    column = position % VOCAB
    score = tl.load(scores_and_gradient + position, active, other=0.0).to(tl.float32)
    target = tl.load(targets + row, active, other=-100)
    maximum = tl.load(row_max + row, active, other=0.0)
    log_total = tl.load(log_normalizer + row, active, other=0.0)
    scale = tl.load(loss_gradient_over_denominator)
    probability = tl.exp((score - maximum) - log_total)
    derivative = tl.fma(probability, scale, tl.where(column == target, -scale, 0.0))
    derivative = tl.where(target != -100, derivative, 0.0)
    # The FP16 cast occurs here, after both loss normalization and the actual
    # upstream scale, matching the autocast linear -> FP32 CE backward path.
    tl.store(scores_and_gradient + position, derivative, active)


def _summarize_logits(scores, targets, token_loss, entropy, predictions, logsumexp,
                      row_max=None, log_normalizer=None):
    """Fill small per-token outputs; scores must be contiguous FP16 (rows,V)."""
    rows, vocab = scores.shape
    if row_max is None:
        row_max = torch.empty(rows, device=scores.device, dtype=torch.float32)
    if log_normalizer is None:
        log_normalizer = torch.empty_like(row_max)
    tiles = triton.cdiv(vocab, VOCAB_TILE)
    partial_max = torch.empty((rows, tiles), device=scores.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    partial_centered_sum = torch.empty_like(partial_max)
    partial_argmax = torch.empty((rows, tiles), device=scores.device, dtype=torch.int32)
    soup_head_partial_statistics[(rows, tiles)](
        scores, partial_max, partial_sum, partial_centered_sum, partial_argmax,
        VOCAB=vocab, TILES=tiles, BLOCK=VOCAB_TILE, num_warps=4,
    )
    soup_head_merge_statistics[(rows,)](
        scores, targets, partial_max, partial_sum, partial_centered_sum, partial_argmax,
        token_loss, entropy, predictions, logsumexp, row_max, log_normalizer,
        VOCAB=vocab, TILES=tiles, MERGE_BLOCK=triton.next_power_of_2(tiles), num_warps=4,
    )
    return row_max, log_normalizer


def _native_row_statistics(scores, targets):
    """Native exceptional-row oracle, with the same FP16 projected values.

    In particular, ignored NaN/Inf logits still affect entropy and backward.
    This helper never changes labels, masks, clipping or optimizer behavior.
    """
    logp = F.log_softmax(scores.float(), dim=-1)
    return (F.nll_loss(logp, targets, reduction="none"),
            -(logp.exp() * logp).sum(-1), scores.argmax(-1))


def _native_logit_gradient(scores, targets, scale):
    """Use the backend's real CE backward for numerically exceptional rows."""
    with torch.enable_grad():
        logits = scores.float().detach().requires_grad_()
        loss = F.cross_entropy(logits, targets, reduction="sum")
        gradient, = torch.autograd.grad(loss, logits, grad_outputs=scale)
    return gradient.to(torch.float16)


def _project(hidden, weight, projection):
    if projection == "triton":
        from .head_matmul import project_fp16

        return project_fp16(hidden, weight)
    if projection == "t4_mma":
        from .head_matmul_t4 import project_fp16

        return project_fp16(hidden, weight)
    if projection == "torch":
        return torch.mm(hidden, weight.t())
    raise ValueError("the selected projection does not materialize ordinary logits")


class _StreamedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, num_items_in_batch, chunk_size, projection, sparse_backward):
        batch, length, width = hidden.shape
        token_count = batch * length
        # Autocast's output projection consumes FP16 activations and weights.
        # A caller can supply a cached, frozen FP16 weight to avoid recasting.
        flat_hidden = hidden.detach().reshape(token_count, width).to(torch.float16).contiguous()
        weight_half = weight.detach().to(torch.float16).contiguous()
        shifted_labels = torch.full_like(labels, -100)
        shifted_labels[:, :-1] = labels[:, 1:]
        targets = shifted_labels.reshape(-1)
        # Explicit FP16 operands preserve the target T4 path even when the
        # caller's surrounding autocast context has already ended.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            if projection == "t4_fused":
                from .head_fused_t4 import fused_head_statistics_fp16

                # All rows, including prompts and final positions, contribute
                # to required entropy. The fused kernel retains only tiled
                # sufficient statistics, so forward does not need token chunks.
                token_loss, entropy, predictions, logsumexp, row_max, log_normalizer = fused_head_statistics_fp16(
                    flat_hidden, weight_half, targets,
                )
            else:
                token_loss = torch.empty(token_count, device=hidden.device, dtype=torch.float32)
                entropy = torch.empty_like(token_loss)
                predictions = torch.empty(token_count, device=hidden.device, dtype=torch.long)
                logsumexp = torch.empty_like(token_loss)
                row_max = torch.empty_like(token_loss)
                log_normalizer = torch.empty_like(token_loss)
                for start in range(0, token_count, chunk_size):
                    end = min(start + chunk_size, token_count)
                    scores = _project(flat_hidden[start:end], weight_half, projection)
                    _summarize_logits(
                        scores, targets[start:end], token_loss[start:end], entropy[start:end],
                        predictions[start:end], logsumexp[start:end],
                        row_max[start:end], log_normalizer[start:end],
                    )
                    del scores

            exceptional = ((log_normalizer <= NATIVE_NORMALIZER_THRESHOLD)
                           | ~torch.isfinite(row_max) | ~torch.isfinite(log_normalizer)
                           | ~torch.isfinite(entropy))
            # One size-discovery synchronization per forward. The extra native
            # projection/CE work occurs only on guarded rows, still in chunks.
            exceptional_rows = torch.nonzero(exceptional, as_tuple=False).flatten()
            for start in range(0, exceptional_rows.numel(), chunk_size):
                rows = exceptional_rows[start:start + chunk_size]
                scores = torch.mm(flat_hidden.index_select(0, rows), weight_half.t())
                native = _native_row_statistics(scores, targets.index_select(0, rows))
                for destination, replacement in zip((token_loss, entropy, predictions), native):
                    destination.index_copy_(0, rows, replacement)
                del scores

        if num_items_in_batch is None:
            denominator = (targets != -100).sum().to(torch.float32)
        else:
            denominator = torch.as_tensor(num_items_in_batch, device=hidden.device, dtype=torch.float32).detach().reshape(())
        # With no supervised tokens and no explicit normalization, PyTorch's
        # mean CE is NaN. Deliberately preserve that behavior instead of hiding it.
        loss = token_loss.sum() / denominator
        ctx.hidden_shape = hidden.shape
        ctx.hidden_dtype = hidden.dtype
        ctx.chunk_size = chunk_size
        ctx.projection = projection
        ctx.sparse_backward = sparse_backward
        ctx.save_for_backward(flat_hidden, weight_half, targets, row_max, log_normalizer,
                              denominator, exceptional, exceptional_rows)
        ctx.set_materialize_grads(False)
        token_loss = token_loss.reshape(batch, length)
        entropy = entropy.reshape(batch, length)
        predictions = predictions.reshape(batch, length)
        ctx.mark_non_differentiable(token_loss, entropy, predictions)
        return loss, token_loss, entropy, predictions

    @staticmethod
    def backward(ctx, grad_loss, _grad_token_loss, _grad_entropy, _grad_predictions):
        if grad_loss is None:
            return None, None, None, None, None, None, None
        (flat_hidden, weight_half, targets, row_max, log_normalizer,
         denominator, exceptional, exceptional_rows) = ctx.saved_tensors
        if ctx.sparse_backward or exceptional_rows.numel():
            # Each projected row contributes independently to causal CE. Rows
            # with finite logits and ignored labels have zero output-hidden gradient;
            # the transformer's own backward still propagates attention paths
            # from supervised positions into earlier input tokens normally.
            # CUDA nonzero synchronizes to discover the compacted size. Its
            # cost must be compared with the saved GEMMs on the target T4.
            selected = ~exceptional
            if ctx.sparse_backward:
                selected = selected & (targets != -100)
            supervised_rows = torch.nonzero(selected, as_tuple=False).flatten()
            grad_hidden = torch.zeros_like(flat_hidden)
            work_hidden = flat_hidden.index_select(0, supervised_rows)
            work_targets = targets.index_select(0, supervised_rows)
            work_max = row_max.index_select(0, supervised_rows)
            work_log_normalizer = log_normalizer.index_select(0, supervised_rows)
        else:
            supervised_rows = None
            grad_hidden = None
            work_hidden, work_targets = flat_hidden, targets
            work_max, work_log_normalizer = row_max, log_normalizer
        token_count, _ = work_hidden.shape
        vocab = weight_half.shape[0]
        work_gradient = torch.empty_like(work_hidden)
        scale = (grad_loss.float() / denominator).contiguous()
        if ctx.projection == "t4_fused":
            from .head_fused_t4 import fused_head_logit_gradient_fp16

        with torch.autocast(device_type=flat_hidden.device.type, enabled=False):
            for start in range(0, token_count, ctx.chunk_size):
                end = min(start + ctx.chunk_size, token_count)
                if ctx.projection == "t4_fused":
                    # Recompute with exactly the fused forward's MMA order and
                    # FP16 logit rounding, using ITS centered normalization. A different
                    # GEMM here could make the softmax probabilities inconsistent.
                    logit_gradient = fused_head_logit_gradient_fp16(
                        work_hidden[start:end], weight_half,
                        work_targets[start:end], work_max[start:end],
                        work_log_normalizer[start:end], scale,
                    )
                else:
                    # Recompute scores rather than retaining N*V elements. This
                    # extra GEMM is a real tradeoff to measure on the target GPU.
                    logit_gradient = _project(work_hidden[start:end], weight_half, ctx.projection)
                    n_elements = logit_gradient.numel()
                    soup_head_logit_gradient[(triton.cdiv(n_elements, GRAD_BLOCK),)](
                        logit_gradient, work_targets[start:end], work_max[start:end],
                        work_log_normalizer[start:end], scale, n_elements,
                        VOCAB=vocab, BLOCK=GRAD_BLOCK, num_warps=4,
                    )
                torch.mm(logit_gradient, weight_half, out=work_gradient[start:end])
                del logit_gradient
        if supervised_rows is None:
            grad_hidden = work_gradient
        else:
            grad_hidden.index_copy_(0, supervised_rows, work_gradient)
        # These rows deliberately use the same native projection in forward
        # and backward. Never hide a NaN in an ignored row by omitting it:
        # native log_softmax backward can propagate exp(NaN) * 0 to dH.
        with torch.autocast(device_type=flat_hidden.device.type, enabled=False):
            for start in range(0, exceptional_rows.numel(), ctx.chunk_size):
                rows = exceptional_rows[start:start + ctx.chunk_size]
                scores = torch.mm(flat_hidden.index_select(0, rows), weight_half.t())
                logit_gradient = _native_logit_gradient(scores, targets.index_select(0, rows), scale)
                grad_hidden.index_copy_(0, rows, torch.mm(logit_gradient, weight_half))
                del scores, logit_gradient
        grad_hidden = grad_hidden.reshape(ctx.hidden_shape).to(ctx.hidden_dtype)
        return grad_hidden, None, None, None, None, None, None


@dataclass(frozen=True)
class HeadResult:
    loss: torch.Tensor
    token_loss: torch.Tensor
    entropy: torch.Tensor
    predictions: torch.Tensor


def streamed_linear_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch=None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    projection: str = "torch",
    sparse_backward: bool = True,
) -> HeadResult:
    """Project causal LM hidden states with bounded logits storage.

    Args:
        hidden: (batch, sequence, hidden_size), FP16 or FP32.
        weight: (vocabulary, hidden_size), frozen FP16 or FP32. A cached FP16
            tensor avoids repeated full-weight conversions.
        labels: (batch, sequence), int64 with valid vocabulary IDs or -100.
        num_items_in_batch: Optional scalar denominator from HF Trainer. If
            omitted, divide by the number of nonignored *shifted* labels.
        chunk_size: Maximum number of backward rows projected at once, fixed
            before measuring. Ordinary projection modes use the same chunks
            in forward; "t4_fused" summarizes all forward rows in one call.
            The default uses no autotuning or compiled caches.
        projection: Explicit implementation choice: "torch", generic original
            "triton" GEMM, experimental explicit-PTX "t4_mma" GEMM, or
            "t4_fused" projection/statistics and projection/gradient kernels.
            The fused mode uses the same MMA arithmetic in both directions
            and requires width divisible by 32 and aligned CUDA FP16 operands.
            Generic Triton 3.6 dot emits scalar FMA on sm75; all original MMA
            alternatives remain experiments pending physical GPU validation.
        sparse_backward: If true, skip finite, ordinary ignored-label rows;
            guarded nonfinite or nearly saturated rows always preserve the
            native path. All forward statistics cover the complete batch.
            Compaction and exceptional-row detection synchronize their sizes.

    The loss is differentiable with respect to hidden only. token_loss is
    causal-aligned: position t predicts labels[t+1], and the last position is
    zero. Entropy and predictions cover ALL positions, including the last one;
    the caller applies the attention mask / supervised-label mask as required.
    """
    if hidden.ndim != 3 or weight.ndim != 2 or labels.shape != hidden.shape[:2]:
        raise ValueError("expected hidden (B,T,D), weight (V,D), and labels (B,T)")
    if hidden.shape[-1] != weight.shape[-1] or min(*hidden.shape, weight.shape[0]) < 1:
        raise ValueError("nonempty compatible hidden and vocabulary dimensions are required")
    if hidden.dtype not in (torch.float16, torch.float32) or weight.dtype not in (torch.float16, torch.float32):
        raise TypeError("the T4 head supports FP16/FP32 activations and weights")
    if labels.dtype != torch.long:
        raise TypeError("labels must be int64")
    if hidden.device != weight.device or hidden.device != labels.device:
        raise ValueError("hidden, weight and labels must be on the same device")
    if weight.requires_grad:
        raise ValueError("streamed_linear_ce requires a frozen output weight")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(num_items_in_batch, torch.Tensor) and num_items_in_batch.numel() != 1:
        raise ValueError("num_items_in_batch must be a scalar")
    if projection not in ("torch", "triton", "t4_mma", "t4_fused"):
        raise ValueError("projection must be 'torch', 'triton', 't4_mma', or 't4_fused'")
    if projection == "t4_fused" and hidden.shape[-1] % 32:
        raise ValueError("t4_fused requires hidden width divisible by 32")
    if not isinstance(sparse_backward, bool):
        raise TypeError("sparse_backward must be a bool")
    return HeadResult(*_StreamedLinearCrossEntropy.apply(
        hidden, weight, labels, num_items_in_batch, chunk_size, projection, sparse_backward,
    ))
