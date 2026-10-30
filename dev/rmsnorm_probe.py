"""GPU comparison for the original task RMSNorm kernels.

Run from the candidate's root after writing the real solution files:
    python dev/rmsnorm_probe.py

This probe does not modify the task, harness, or test suite. It reports
numerical errors rather than a performance claim. Cold measured training
must still use the harness's fresh compiler caches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import rmsnorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


def reference(x, weight, epsilon):
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    y = x32 * torch.rsqrt(variance + epsilon)
    return weight * y.to(input_dtype)


def error(a, b):
    difference = (a.float() - b.float()).abs()
    return {
        "max_abs": float(difference.max().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(b.float()).clamp_min(1e-30)
        ),
    }


def run():
    assert torch.cuda.is_available(), "A CUDA GPU is required"
    torch.manual_seed(7319)
    torch.cuda.manual_seed_all(7319)
    rmsnorm.install()
    installed_forward = Qwen2RMSNorm.forward
    rmsnorm.install()
    assert Qwen2RMSNorm.forward is installed_forward, "Installation is not idempotent"

    results = []
    cases = [(1, 1536), (17, 1536), (4, 127, 1536), (4, 512, 1536)]
    for shape in cases:
        module = Qwen2RMSNorm(1536, eps=1e-6).cuda().float()
        module.weight.requires_grad_(False)
        with torch.no_grad():
            module.weight.copy_(torch.empty_like(module.weight).uniform_(0.5, 1.5))
        for scale in (0.01, 1.0, 10.0):
            x = (torch.randn(shape, device="cuda") * scale).requires_grad_()
            xr = x.detach().clone().requires_grad_()
            assert rmsnorm._supported(module, x)
            # A transposed upstream gradient checks noncontiguous dY handling.
            flat = x.numel() // x.shape[-1]
            dy = torch.randn((1536, flat), device="cuda").t().reshape(shape)
            with torch.autocast("cuda", dtype=torch.float16):
                y = module(x)
                yr = reference(xr, module.weight, module.variance_epsilon)
            dx, = torch.autograd.grad(y, x, dy)
            dxr, = torch.autograd.grad(yr, xr, dy)
            f_error, b_error = error(y, yr), error(dx, dxr)
            torch.testing.assert_close(y, yr, rtol=3e-6, atol=3e-6)
            torch.testing.assert_close(dx, dxr, rtol=3e-5, atol=3e-5 / scale)
            assert f_error["relative_l2"] < 3e-6
            assert b_error["relative_l2"] < 5e-6
            results.append({"shape": list(shape), "scale": scale,
                            "forward": f_error, "backward": b_error})

    # Explicit fallback coverage includes trainable gamma, unsupported width,
    # half residuals, noncontiguous residuals, empty rows, and CPU tensors.
    fallback = []
    for kind in ("trainable_gamma", "width", "half", "stride", "empty", "cpu"):
        width = 64 if kind == "width" else 1536
        device = "cpu" if kind == "cpu" else "cuda"
        module = Qwen2RMSNorm(width, eps=1e-6).to(device=device, dtype=torch.float32)
        module.weight.requires_grad_(kind == "trainable_gamma")
        dtype = torch.float16 if kind == "half" else torch.float32
        if kind == "stride":
            x = torch.randn((7, width * 2), device=device)[:, ::2].requires_grad_()
        else:
            x = torch.randn((0 if kind == "empty" else 7, width),
                            device=device, dtype=dtype).requires_grad_()
        assert not rmsnorm._supported(module, x), kind
        original = Qwen2RMSNorm.forward._soup_original_forward
        actual = module(x)
        expected = original(module, x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        fallback.append(kind)

    # The first frozen prefix may not participate in autograd at all.
    module = Qwen2RMSNorm(1536).cuda().float()
    module.weight.requires_grad_(False)
    x = torch.randn((4, 11, 1536), device="cuda")
    y = module(x)
    assert not y.requires_grad
    torch.testing.assert_close(y, reference(x, module.weight, module.variance_epsilon),
                               rtol=3e-6, atol=3e-6)
    torch.cuda.synchronize()
    print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                      "cases": results, "fallbacks": fallback,
                      "frozen_prefix": "passed", "status": "passed"}, indent=2))


if __name__ == "__main__":
    run()
