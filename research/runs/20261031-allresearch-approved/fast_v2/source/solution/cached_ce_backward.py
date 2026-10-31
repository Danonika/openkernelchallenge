"""Inactive first-order CE backward experiment using native log-softmax.

The original full-shape head forward/backward GEMMs remain the caller's work.
This helper replaces only bounded native cross-entropy/autograd reconstruction
with native FP32 log-softmax and an original elementwise gradient consumer.
Higher-order differentiation must select the existing native fallback BEFORE
calling this helper; this module deliberately returns no autograd graph.

For a sum NLL with an external divisor, native autograd first divides the
upstream FP32 gradient by that divisor. The gradient into log-softmax has one
negative scaled value at the label and zeros elsewhere. Its row sum is that
same value (positive zero for an all-zero row), so no dense NLL gradient or
row-reduction workspace is necessary. The CUDA consumer evaluates the native
log-softmax derivative in FP32, including its exponential and fused multiply
add, before the final FP16 store. Ignored rows are not skipped: NaN logp still
produces NaN through zero times NaN, as in native backward.

Arithmetic was checked against the defining operations in PyTorch v2.11.0
aten/src/ATen/native/cuda/{Loss.cu,SoftMax.cu}; no library kernel was copied.
Exact CUDA parity is a physical-test requirement, not established by the CPU
interpreter or offline compilation. No production path imports this module.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


GRADIENT_TILE = 1024


@triton.jit
def soup_native_logp_ce_gradient(
    logp, targets, scaled_gradient, output,
    VOCAB: tl.constexpr, BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    valid = column < VOCAB
    offset = row.to(tl.int64) * VOCAB + column
    target = tl.load(targets + row)
    in_bounds = (target >= 0) & (target < VOCAB)
    ignored = target == -100
    scaled = tl.load(scaled_gradient)
    # Triton's unary minus can lower to +0-x, which loses the sign of -0.
    # Native scalar negation flips the sign bit, including for zero scale.
    label_gradient = (scaled.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    row_sum = tl.where(in_bounds, label_gradient, 0.0)
    # A native reduction starts from +0; adding a single -0 yields +0.
    row_sum = tl.where(row_sum == 0.0, 0.0, row_sum)
    sparse_gradient = tl.where(in_bounds & (column == target), label_gradient, 0.0)
    lp = tl.load(logp + offset, valid, other=0.0)
    probability = libdevice.exp(lp)
    negative_probability = (probability.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    gradient = tl.fma(negative_probability, row_sum, sparse_gradient)
    # The public helper asserts valid labels. This independent defensive
    # branch makes even a direct invalid-target launch memory-safe and loud.
    gradient = tl.where(in_bounds | ignored, gradient, float("nan"))
    tl.store(output + offset, gradient, valid)


def _consume_logp_gradient(logp, targets, scaled_gradient, output):
    rows, vocab = logp.shape
    soup_native_logp_ce_gradient[(rows, triton.cdiv(vocab, GRADIENT_TILE))](
        logp, targets, scaled_gradient, output,
        VOCAB=vocab, BLOCK=GRADIENT_TILE, num_warps=4,
        # The explicit fma remains fused. Prevent unrelated algebraic
        # contraction from changing the source's stated rounding boundaries.
        enable_fp_fusion=False,
    )


def native_logsoftmax_ce_backward(logits, targets, divisor, grad_loss,
                                 chunk_size=128):
    """Return a first-order FP16 gradient for contiguous (rows, vocabulary).

    Targets are already causally shifted and contain int64 labels or -100.
    divisor and grad_loss are scalar FP32 tensors on the logits' device.
    Neither divisor nor upstream scaling is clamped or converted to FP16.
    Zero, negative, NaN and infinite divisors preserve the native arithmetic.
    All full-vocabulary temporary workspaces are bounded by chunk_size.
    """
    if (logits.ndim != 2 or targets.ndim != 1
            or logits.shape[0] != targets.numel() or min(logits.shape) < 1):
        raise ValueError("expected nonempty logits (rows,vocabulary) and one target per row")
    if logits.dtype != torch.float16 or targets.dtype != torch.long:
        raise TypeError("expected FP16 logits and int64 targets")
    if (not logits.is_contiguous() or not targets.is_contiguous()
            or logits.device != targets.device):
        raise ValueError("logits and targets must be contiguous on the same device")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    for name, value in (("divisor", divisor), ("grad_loss", grad_loss)):
        if (not isinstance(value, torch.Tensor) or value.numel() != 1
                or value.dtype != torch.float32 or value.device != logits.device):
            raise ValueError(name + " must be a scalar FP32 tensor on the logits device")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (logits, divisor, grad_loss)):
        raise ValueError("first-order helper requires native higher-order fallback before dispatch")
    if not logits.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("requires CUDA; CPU diagnostics need TRITON_INTERPRET=1")
    rows, vocab = logits.shape
    torch._assert_async(((targets == -100) | ((targets >= 0) & (targets < vocab))).all(),
                        "causal target index is out of bounds")
    with torch.no_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
        # Keep this native scalar division: it is DivBackward's exact input
        # to sum-NLL backward, shared unchanged by every row chunk.
        scaled_gradient = grad_loss.reshape(()) / divisor.reshape(())
        result = torch.empty_like(logits)
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            logp = F.log_softmax(logits[start:stop].float(), dim=-1)
            _consume_logp_gradient(logp, targets[start:stop], scaled_gradient,
                                   result[start:stop])
            del logp
    return result
