"""RMSNorm numerical checks; CPU execution uses Triton's official interpreter.

GPU: python -m pytest solution/test_rmsnorm.py -q
CPU math check: TRITON_INTERPRET=1 python -m pytest solution/test_rmsnorm.py -q
The latter checks code and arithmetic, not GPU compilation or performance.
"""

import os

import pytest
import torch


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CAN_EXECUTE = torch.cuda.is_available() or INTERPRETING
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reference(x, weight, eps):
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return weight * (xf * inv).to(x.dtype)


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 5, 1536), (3, 1001), (1, 17)])
def test_rmsnorm_forward_backward(dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(193)
    # Non-unit, non-power-of-two weights catch moving the FP16 rounding point.
    weight = torch.randn(shape[-1], dtype=torch.float32, device=DEVICE) * 0.3 + 0.8
    x = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    expected = reference(x, weight, 1e-6)
    expected.backward(grad)
    expected_dx = x.grad.detach().clone()
    x.grad = None

    actual = rmsnorm(x, weight, 1e-6)
    actual.backward(grad)
    assert actual.dtype == torch.float32
    assert x.grad.dtype == dtype
    tolerance = 2e-3 if dtype == torch.float16 else 4e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x.grad, expected_dx, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rmsnorm_zero_input_and_expanded_gradient(dtype):
    from solution.rmsnorm import rmsnorm

    weight = torch.linspace(0.2, 1.4, 1536, device=DEVICE, dtype=torch.float32)
    x = torch.zeros((3, 1536), device=DEVICE, dtype=dtype, requires_grad=True)
    expected = reference(x, weight, 1e-5)
    expected.sum().backward()
    expected_dx = x.grad.detach().clone()
    x.grad = None
    actual = rmsnorm(x, weight, 1e-5)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    tolerance = 2e-3 if dtype == torch.float16 else 4e-6
    torch.testing.assert_close(x.grad, expected_dx, rtol=tolerance, atol=tolerance)


def test_rmsnorm_rejects_trainable_weight():
    from solution.rmsnorm import rmsnorm

    x = torch.randn(2, 17)
    weight = torch.ones(17, requires_grad=True)
    with pytest.raises(ValueError, match="frozen"):
        rmsnorm(x, weight)


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [1.0, 1000.0, 65536.0])
def test_rmsnorm_preserves_scaled_native_gradient(dtype, scale):
    """Exercise the actual eager Qwen module and GradScaler-sized gradients."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(719)
    module = Qwen2RMSNorm(1536, eps=1e-6).to(DEVICE)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.31, 1.19, 1536, device=DEVICE))
    module.weight.requires_grad_(False)
    x = (torch.randn(4, 1536, device=DEVICE) * 0.4).to(dtype).requires_grad_()
    upstream = torch.randn(4, 1536, device=DEVICE) * (scale * 0.07)
    expected = module(x)
    expected_dx, = torch.autograd.grad(expected, x, upstream)
    actual = rmsnorm(x, module.weight, module.variance_epsilon)
    actual_dx, = torch.autograd.grad(actual, x, upstream)
    tolerance = 2e-3 if dtype == torch.float16 else 4e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=tolerance,
                               atol=tolerance * max(1, scale), equal_nan=True)


def test_qwen2_patch_is_idempotent_and_keeps_cpu_reference(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from solution.rmsnorm import patch_qwen2_rmsnorm

    original = Qwen2RMSNorm.forward
    monkeypatch.setattr(Qwen2RMSNorm, "forward", original)
    assert patch_qwen2_rmsnorm()
    installed = Qwen2RMSNorm.forward
    assert patch_qwen2_rmsnorm()
    assert Qwen2RMSNorm.forward is installed
    module = Qwen2RMSNorm(17)
    module.weight.requires_grad_(False)
    x = torch.randn(2, 17, dtype=torch.float16, requires_grad=True)
    actual = module(x)
    expected = reference(x, module.weight, module.variance_epsilon)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
