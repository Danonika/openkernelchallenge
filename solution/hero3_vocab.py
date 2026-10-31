"""Original Hero1 vocabulary-sliced head for the fixed Soup T4 Qwen2 task.

Algorithmic references (not source-code copies): online normalizer, streamed
cross-entropy, an external contributor's reported vocabulary-axis scheduling. The equations
and implementation below are independent. For FP32 tied weights, this streams the fp16 cast by vocabulary slice.
The full fixed task, TRL metrics
and Soup rewind are preserved; no change to model/optimizer/dataset/steps.
"""
from functools import wraps
import torch
import triton
import triton.language as tl


@triton.jit
def hero1_slice_stats_inplace(
    SCORES, TARGETS, RUN_MAX, RUN_SUM, RUN_CENTERED,
    BEST_VAL, BEST_IDX, TARGET_VAL, SCALE_OLD,
    SLICE_COLS, COL_OFFSET,
    FIRST: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    used = j < SLICE_COLS
    z = tl.load(SCORES + row * SLICE_COLS + j, used, other=-float("inf")).to(tl.float32)
    slice_max = tl.max(z, axis=0)
    if FIRST:
        prev_max = slice_max
        prev_sum = tl.full((), 0.0, tl.float32)
        prev_centered = tl.full((), 0.0, tl.float32)
        best = tl.full((), -float("inf"), tl.float32)
        best_index = tl.full((), 2147483647, tl.int64)
    else:
        prev_max = tl.load(RUN_MAX + row)
        prev_sum = tl.load(RUN_SUM + row)
        prev_centered = tl.load(RUN_CENTERED + row)
        best = tl.load(BEST_VAL + row)
        best_index = tl.load(BEST_IDX + row)

    new_max = tl.maximum(prev_max, slice_max)
    old_factor = tl.exp(prev_max - new_max)
    centered_z = tl.where(used, z - new_max, 0.0)
    expval = tl.where(used, tl.exp(centered_z), 0.0)
    local_sum = tl.sum(expval, axis=0)
    local_center = tl.sum(expval * centered_z, axis=0)
    new_sum = prev_sum * old_factor + local_sum
    new_center = old_factor * (prev_centered + (prev_max - new_max) * prev_sum) + local_center

    chosen_col = tl.min(tl.where(used & (z == slice_max), j, 2147483647), 0)
    chosen_idx = COL_OFFSET + chosen_col
    next_idx = tl.where(slice_max > best, chosen_idx,
                        tl.where(slice_max == best, tl.minimum(best_index, chosen_idx), best_index))
    next_best = tl.maximum(best, slice_max)

    label = tl.load(TARGETS + row)
    belongs = (label >= COL_OFFSET) & (label < COL_OFFSET + SLICE_COLS)
    label_col = tl.maximum(label - COL_OFFSET, 0)
    label_value = tl.load(SCORES + row * SLICE_COLS + label_col,
                          mask=belongs, other=0.0).to(tl.float32)

    tl.store(RUN_MAX + row, new_max)
    tl.store(RUN_SUM + row, new_sum)
    tl.store(RUN_CENTERED + row, new_center)
    tl.store(BEST_VAL + row, next_best)
    tl.store(BEST_IDX + row, next_idx)
    tl.store(TARGET_VAL + row, label_value, mask=belongs)
    tl.store(SCALE_OLD + row, old_factor)
    # Reuse this exact FP16 buffer for an actual weighted-gradient GEMM;
    # never allocate the full N x vocabulary probabilities.
    tl.store(SCORES + row * SLICE_COLS + j, expval.to(tl.float16), used)


@triton.jit
def hero1_finalize_token_stats(
    RUN_MAX, RUN_SUM, RUN_CENTERED, BEST_IDX, TARGET_VAL, TARGETS,
    LOSS_OUT, ENTROPY_OUT, PRED_OUT, N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < N
    mx = tl.load(RUN_MAX + row, valid, other=0.0)
    total = tl.load(RUN_SUM + row, valid, other=1.0)
    centered = tl.load(RUN_CENTERED + row, valid, other=0.0)
    y = tl.load(TARGETS + row, valid, other=-100)
    target = tl.load(TARGET_VAL + row, valid, other=0.0)
    lse_centered = tl.log(total)
    ce = tl.where(y == -100, 0.0, (mx - target) + lse_centered)
    entropy = lse_centered - centered / total
    argmax = tl.load(BEST_IDX + row, valid, other=0)
    tl.store(LOSS_OUT + row, ce, valid)
    tl.store(ENTROPY_OUT + row, entropy, valid)
    tl.store(PRED_OUT + row, argmax, valid)


class _VocabStream(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, num_items, vocab_tile):
        batch, steps, width = hidden.shape
        vocab = weight.shape[0]
        n = batch * steps
        half_h = hidden.reshape(n, width).detach().to(torch.float16).contiguous()
        # Keep FP16 tied storage as a view; otherwise cast ONE vocabulary
        # slice at a time, never a permanent 445 MiB full-head copy.
        frozen_w = weight.detach()
        shifted = torch.empty_like(labels)
        shifted[:, :-1] = labels[:, 1:]
        shifted[:, -1] = -100
        targets = shifted.reshape(-1).contiguous()
        selected = torch.nonzero(targets != -100, as_tuple=False).flatten()
        live_count = selected.numel()
        oldmax = torch.empty(n, device=hidden.device, dtype=torch.float32)
        mass = torch.empty_like(oldmax)
        centered = torch.empty_like(oldmax)
        best = torch.empty_like(oldmax)
        bestid = torch.empty(n, device=hidden.device, dtype=torch.int64)
        target_logits = torch.zeros_like(oldmax)
        factor = torch.empty_like(oldmax)
        weighted = torch.zeros((live_count, width), device=hidden.device, dtype=torch.float32)
        with torch.autocast("cuda", enabled=False):
            for left in range(0, vocab, vocab_tile):
                right = min(left + vocab_tile, vocab)
                w = frozen_w[left:right].to(torch.float16).contiguous()
                scores = torch.mm(half_h, w.t())
                hero1_slice_stats_inplace[(n,)](
                    scores, targets, oldmax, mass, centered, best, bestid,
                    target_logits, factor, right-left, left,
                    FIRST=(left == 0), BLOCK=triton.next_power_of_2(vocab_tile),
                    num_warps=8, enable_fp_fusion=False,
                )
                if live_count:
                    prior = factor.index_select(0, selected)
                    weighted.mul_(prior[:, None])
                    weighted.add_(
                        torch.mm(scores.index_select(0, selected), w).float()
                    )
                del scores
            token_loss = torch.empty(n, device=hidden.device, dtype=torch.float32)
            entropy = torch.empty_like(token_loss)
            predictions = torch.empty(n, device=hidden.device, dtype=torch.int64)
            hero1_finalize_token_stats[(triton.cdiv(n, 128),)](
                oldmax, mass, centered, bestid, target_logits, targets,
                token_loss, entropy, predictions, N=n, BLOCK=128,
                num_warps=4, enable_fp_fusion=False
            )
            if live_count:
                normalized = weighted / mass.index_select(0, selected)[:, None]
                truth = frozen_w.index_select(0, targets.index_select(0, selected)).to(torch.float16).float()
                unscaled_dx = normalized - truth
            else:
                unscaled_dx = weighted
        if num_items is None:
            denom = (targets != -100).sum().float()
        else:
            denom = torch.as_tensor(
                num_items, dtype=torch.float32, device=hidden.device
            ).detach().reshape(())
        ctx.save_for_backward(unscaled_dx, selected, denom)
        ctx.output_shape = hidden.shape
        ctx.output_dtype = hidden.dtype
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(token_loss, entropy, predictions)
        return (token_loss.sum() / denom, token_loss.reshape(batch, steps),
                entropy.reshape(batch, steps), predictions.reshape(batch, steps))

    @staticmethod
    def backward(ctx, grad_loss, _loss, _entropy, _predictions):
        if grad_loss is None:
            return None, None, None, None, None
        live_dx, live_index, denom = ctx.saved_tensors
        rows = ctx.output_shape[0] * ctx.output_shape[1]
        width = ctx.output_shape[2]
        full_dx = torch.zeros((rows, width), device=live_dx.device, dtype=torch.float32)
        if live_index.numel():
            full_dx.index_copy_(
                0, live_index, live_dx * (grad_loss.float() / denom)
            )
        return full_dx.reshape(ctx.output_shape).to(ctx.output_dtype), None, None, None, None



class _PackedVocabStream(torch.autograd.Function):
    """Vocabulary-axis head over the decoder's real-token [1,N,H] output.

    Dense [B,T] tensors are restored only for Soup/TRL's small metric surfaces.
    The hidden-state gradient stays packed and returns directly to the packed
    decoder, eliminating a dense B*T*H scatter/gather in both directions.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, active, num_items, vocab_tile):
        if (hidden.ndim != 3 or hidden.shape[0] != 1
                or labels.ndim != 2 or active.ndim != 1
                or active.dtype != torch.long or active.device != hidden.device
                or active.numel() != hidden.shape[1]):
            raise ValueError("packed vocabulary head geometry mismatch")

        batch, steps = labels.shape
        width = hidden.shape[-1]
        vocab = weight.shape[0]
        n = active.numel()
        half_h = hidden.reshape(n, width).detach().to(torch.float16).contiguous()
        frozen_w = weight.detach()

        shifted = torch.empty_like(labels)
        shifted[:, :-1] = labels[:, 1:]
        shifted[:, -1] = -100
        dense_targets = shifted.reshape(-1)
        targets = dense_targets.index_select(0, active).contiguous()

        selected = torch.nonzero(targets != -100, as_tuple=False).flatten()
        live_count = selected.numel()
        oldmax = torch.empty(n, device=hidden.device, dtype=torch.float32)
        mass = torch.empty_like(oldmax)
        centered = torch.empty_like(oldmax)
        best = torch.empty_like(oldmax)
        bestid = torch.empty(n, device=hidden.device, dtype=torch.int64)
        target_logits = torch.zeros_like(oldmax)
        factor = torch.empty_like(oldmax)
        weighted = torch.zeros((live_count, width), device=hidden.device, dtype=torch.float32)

        with torch.autocast("cuda", enabled=False):
            for left in range(0, vocab, vocab_tile):
                right = min(left + vocab_tile, vocab)
                w = frozen_w[left:right].to(torch.float16).contiguous()
                scores = torch.mm(half_h, w.t())
                hero1_slice_stats_inplace[(n,)](
                    scores, targets, oldmax, mass, centered, best, bestid,
                    target_logits, factor, right-left, left,
                    FIRST=(left == 0), BLOCK=triton.next_power_of_2(vocab_tile),
                    num_warps=8, enable_fp_fusion=False,
                )
                if live_count:
                    prior = factor.index_select(0, selected)
                    weighted.mul_(prior[:, None])
                    weighted.add_(
                        torch.mm(scores.index_select(0, selected), w).float()
                    )
                del scores

            packed_loss = torch.empty(n, device=hidden.device, dtype=torch.float32)
            packed_entropy = torch.empty_like(packed_loss)
            packed_predictions = torch.empty(n, device=hidden.device, dtype=torch.int64)
            hero1_finalize_token_stats[(triton.cdiv(n, 128),)](
                oldmax, mass, centered, bestid, target_logits, targets,
                packed_loss, packed_entropy, packed_predictions,
                N=n, BLOCK=128, num_warps=4, enable_fp_fusion=False,
            )
            if live_count:
                normalized = weighted / mass.index_select(0, selected)[:, None]
                truth = frozen_w.index_select(
                    0, targets.index_select(0, selected)
                ).to(torch.float16).float()
                unscaled_dx = normalized - truth
            else:
                unscaled_dx = weighted

        if num_items is None:
            denom = (dense_targets != -100).sum().float()
        else:
            denom = torch.as_tensor(
                num_items, dtype=torch.float32, device=hidden.device
            ).detach().reshape(())

        dense_n = batch * steps
        token_loss = torch.zeros(dense_n, device=hidden.device, dtype=torch.float32)
        entropy = torch.zeros_like(token_loss)
        predictions = torch.zeros(dense_n, device=hidden.device, dtype=torch.int64)
        token_loss.index_copy_(0, active, packed_loss)
        entropy.index_copy_(0, active, packed_entropy)
        predictions.index_copy_(0, active, packed_predictions)

        ctx.save_for_backward(unscaled_dx, selected, denom)
        ctx.output_shape = hidden.shape
        ctx.output_dtype = hidden.dtype
        ctx.set_materialize_grads(False)
        token_loss = token_loss.view(batch, steps)
        entropy = entropy.view(batch, steps)
        predictions = predictions.view(batch, steps)
        ctx.mark_non_differentiable(token_loss, entropy, predictions)
        return packed_loss.sum() / denom, token_loss, entropy, predictions

    @staticmethod
    def backward(ctx, grad_loss, _loss, _entropy, _predictions):
        if grad_loss is None:
            return None, None, None, None, None, None
        live_dx, live_index, denom = ctx.saved_tensors
        rows = ctx.output_shape[0] * ctx.output_shape[1]
        width = ctx.output_shape[2]
        packed_dx = torch.zeros((rows, width), device=live_dx.device, dtype=torch.float32)
        if live_index.numel():
            packed_dx.index_copy_(
                0, live_index, live_dx * (grad_loss.float() / denom)
            )
        return (
            packed_dx.reshape(ctx.output_shape).to(ctx.output_dtype),
            None, None, None, None, None,
        )


def packed_vocab_stream(hidden, weight, labels, active, num_items_in_batch=None):
    from .streamed_head import HeadResult
    return HeadResult(*_PackedVocabStream.apply(
        hidden, weight, labels, active, num_items_in_batch, 8192
    ))

def install_vocab_head():
    """Runtime-only patch; the fixed Soup HEAD_MODE must be 'streamed'."""
    from . import streamed_head
    if getattr(streamed_head.streamed_linear_ce, "_hero1_vocab_stream", False):
        return True
    previous = streamed_head.streamed_linear_ce

    @wraps(previous)
    def fast_stream(hidden, weight, labels, num_items_in_batch=None,
                    chunk_size=256, projection="torch", sparse_backward=True):
        if (not isinstance(hidden, torch.Tensor) or not hidden.is_cuda
                or not hidden.requires_grad or hidden.ndim != 3
                or not isinstance(weight, torch.Tensor) or weight.device != hidden.device
                or weight.requires_grad or weight.dtype not in (torch.float16, torch.float32)
                or labels.dtype != torch.long or labels.shape != hidden.shape[:2]
                or hidden.shape[-1] != weight.shape[-1]
                or weight.shape[0] == 0
                or projection != "torch"):
            return previous(hidden, weight, labels, num_items_in_batch,
                            chunk_size, projection, sparse_backward)
        return streamed_head.HeadResult(*_VocabStream.apply(
            hidden, weight, labels, num_items_in_batch, 8192
        ))

    fast_stream._hero1_vocab_stream = True
    streamed_head.streamed_linear_ce = fast_stream
    return True
