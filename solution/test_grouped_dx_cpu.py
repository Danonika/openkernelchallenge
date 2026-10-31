"""CPU-only arithmetic oracle for the grouped frozen-MLP VJP.

This checks reduction-order drift; it does not prove the CUDA kernel,
bitsandbytes nested NF4 parity, speed or official Soup admission.
"""
import math
import torch


def _check(rows, hidden, inter, seed):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, hidden, dtype=torch.float16, generator=rng) * 0.11
    wg = torch.randn(inter, hidden, dtype=torch.float16, generator=rng) / math.sqrt(hidden)
    wu = torch.randn(inter, hidden, dtype=torch.float16, generator=rng) / math.sqrt(hidden)
    wd = torch.randn(hidden, inter, dtype=torch.float16, generator=rng) / math.sqrt(inter)
    upstream = torch.randn(rows, hidden, dtype=torch.float16, generator=rng) * 0.17
    dh = upstream @ wd
    gate, up = x @ wg.T, x @ wu.T
    sig = torch.sigmoid(gate.float())
    dg = ((dh.float()*up.float())*sig*(1.0+gate.float()*(1.0-sig))).half()
    du = (dh.float()*gate.float()*sig).half()
    separated = dg @ wg
    separated.add_(du @ wu)
    combined = torch.cat([dg,du],dim=-1) @ torch.cat([wg,wu],dim=0)
    error = (combined.float() - separated.float()).norm()
    denominator = separated.float().norm().clamp_min(1e-8)
    assert (error / denominator).item() < 0.006


def test_20_small_cases():
    for seed in range(20):
        _check(24, 64, 160, seed)


def test_qwen_shape():
    _check(16, 1536, 8960, 123)
