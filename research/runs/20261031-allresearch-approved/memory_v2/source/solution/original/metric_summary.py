"""Original detached materialized CE metric summaries, research proposal.

One extra FP16 vocabulary read uses the already computed CE max/logsum. It
replaces native full-vocabulary entropy and first-wins argmax ONLY after a
consumer-owned per-call sink is accepted. It does not change CE arithmetic,
normalization or backward. Finite logits are an upstream global invariant.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def soup_ce_metric_tiles(Z, Maximum, LogSum, PartialEntropy, PartialArgmax,
                         V: tl.constexpr, NT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    real = column < V
    value = tl.load(Z + row * V + column, mask=real, other=0.0).to(tl.float32)
    maximum = tl.load(Maximum + row)
    logsum = tl.load(LogSum + row)
    # Padding is neutral BEFORE exp/product, not an unsafe 0 * infinity mask.
    logp = tl.where(real, (value - maximum) - logsum, 0.0)
    contribution = tl.exp(logp) * logp
    entropy = -tl.sum(contribution, axis=0)
    first = tl.min(tl.where(real & (value == maximum), column, 2147483647), axis=0)
    tl.store(PartialEntropy + row * NT + tile, entropy)
    tl.store(PartialArgmax + row * NT + tile, first)


@triton.jit
def soup_ce_metric_merge(PartialEntropy, PartialArgmax, Entropy, Argmax,
                         NT: tl.constexpr, MERGE: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, MERGE)
    entropy = tl.load(PartialEntropy + row * NT + index, mask=index < NT, other=0.0)
    first = tl.load(PartialArgmax + row * NT + index, mask=index < NT, other=2147483647)
    tl.store(Entropy + row, tl.sum(entropy, axis=0))
    tl.store(Argmax + row, tl.min(first, axis=0).to(tl.int64))


def detached_metrics(logits, maximum, logsum):
    """Private finite-route helper; no host value reads or autograd retention."""
    n, v = logits.numel() // logits.shape[-1], logits.shape[-1]
    block = 2048
    nt = triton.cdiv(v, block)
    shape = tuple(logits.shape[:-1])
    with torch.no_grad():
        partial_entropy = torch.empty((n, nt), device=logits.device, dtype=torch.float32)
        partial_argmax = torch.empty((n, nt), device=logits.device, dtype=torch.int32)
        entropy = torch.empty(shape, device=logits.device, dtype=torch.float32)
        argmax = torch.empty(shape, device=logits.device, dtype=torch.int64)
        soup_ce_metric_tiles[(n, nt)](logits, maximum, logsum, partial_entropy, partial_argmax,
            V=v, NT=nt, BLOCK=block, num_warps=4, enable_fp_fusion=False)
        soup_ce_metric_merge[(n,)](partial_entropy, partial_argmax, entropy, argmax,
            NT=nt, MERGE=triton.next_power_of_2(nt), num_warps=4, enable_fp_fusion=False)
    return entropy, argmax


def publish(sink, selected, target, entropy, argmax, *, route):
    """Fresh small NLL allocation protects training-selected storage from sink."""
    if sink is None:
        return
    with torch.no_grad():
        token_nll = torch.where(target.view(-1) != -100, -selected.detach(), 0.0)
        token_nll = token_nll.view(target.shape).contiguous()
        sink(token_nll=token_nll, token_entropy=entropy, first_argmax=argmax, route=route)
