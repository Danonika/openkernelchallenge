"""Original FP16 vocabulary projection for the streamed frozen output head.

Computes Y[m, v] = sum_d X[m, d] * W[v, d], accumulating in FP32 and
rounding the completed dot product to FP16.  This is an independent tiled
implementation of matrix multiplication; no external kernel code is used.

The caller owns autograd.  In this solution the streamed head's custom
autograd function calls this operation in its forward/recomputation path.
The weight must be frozen.  This experimental kernel is a configurable
alternative to PyTorch GEMM, with no claim of a measured T4 speed advantage.

Compiler limitation: stock Triton 3.6 lowers tl.dot to scalar FMA on SM75,
not to Turing tensor cores.  Offline PTX inspection confirmed this for this
kernel.  Keep this variant disabled for the competition candidate unless
official T4 measurements demonstrate otherwise; correctness is insufficient.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def soup_head_projection(
    X, W, Y,
    TOKENS,
    DEPTH: tl.constexpr,
    VOCAB: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_V: tl.constexpr,
    TILE_D: tl.constexpr,
):
    token = tl.program_id(0) * TILE_M + tl.arange(0, TILE_M)
    word = tl.program_id(1) * TILE_V + tl.arange(0, TILE_V)
    within_depth = tl.arange(0, TILE_D)
    sums = tl.full((TILE_M, TILE_V), 0, tl.float32)

    # W is stored [vocabulary, hidden], so the right operand is a transposed
    # view during loading; no full weight transpose is materialized.
    for depth_tile in range(tl.cdiv(DEPTH, TILE_D)):
        channel = depth_tile * TILE_D + within_depth
        a = tl.load(
            X + token[:, None] * DEPTH + channel[None, :],
            (token[:, None] < TOKENS) & (channel[None, :] < DEPTH),
            0,
        )
        b = tl.load(
            W + word[None, :] * DEPTH + channel[:, None],
            (word[None, :] < VOCAB) & (channel[:, None] < DEPTH),
            0,
        )
        # Both inputs are FP16, with FP32 accumulation and no TF32/BF16.
        # Stock Triton 3.6 emits scalar FMA here for SM75 (see module note).
        sums = tl.dot(a, b, sums)

    tl.store(
        Y + token[:, None] * VOCAB + word[None, :],
        sums.to(tl.float16),
        (token[:, None] < TOKENS) & (word[None, :] < VOCAB),
    )


def project_fp16(x, weight):
    """Project contiguous [M,D] and frozen [V,D] FP16 tensors to [M,V].

    This function intentionally has no autograd rule.  A gradient-requiring
    input must be used under a caller-owned autograd Function or no_grad().
    Direct CPU execution is supported only by Triton's development interpreter.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("project_fp16 expects tensors")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("project_fp16 expects [M,D] and [V,D] with matching D")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("project_fp16 requires FP16 inputs")
    if x.device != weight.device or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("project_fp16 requires contiguous inputs on the same device")
    if weight.requires_grad:
        raise ValueError("project_fp16 requires a frozen weight")
    if torch.is_grad_enabled() and x.requires_grad:
        raise ValueError("project_fp16 requires caller-owned autograd or no_grad()")

    tokens, depth = x.shape
    vocab = weight.shape[0]
    y = torch.empty((tokens, vocab), dtype=torch.float16, device=x.device)
    if tokens == 0 or vocab == 0:
        return y
    # A fixed configuration keeps cold compilation bounded.  The first grid
    # dimension traverses token tiles for the same vocabulary tile, providing
    # neighboring programs with the opportunity to reuse weight-cache lines.
    tile_m, tile_v, tile_d = 32, 128, 64
    soup_head_projection[(triton.cdiv(tokens, tile_m), triton.cdiv(vocab, tile_v))](
        x, weight, y, tokens, depth, vocab, tile_m, tile_v, tile_d,
        num_warps=4, num_stages=2,
    )
    return y
