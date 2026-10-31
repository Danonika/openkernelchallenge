"""Numerical checks for the optional original vocabulary projection.

Run on CUDA or with TRITON_INTERPRET=1 for small CPU arithmetic checks.
Interpreter checks do not measure CUDA performance.
"""

import os

import pytest
import torch


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CAN_EXECUTE = torch.cuda.is_available() or INTERPRETING
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("shape", [(3, 7, 11), (33, 65, 67), (32, 128, 128), (7, 257, 129), (3, 11, 1536)])
def test_projection_matches_torch(shape):
    from solution.head_matmul import project_fp16

    tokens, vocab, depth = shape
    torch.manual_seed(947)
    x = torch.randn((tokens, depth), dtype=torch.float16, device=DEVICE)
    weight = torch.randn((vocab, depth), dtype=torch.float16, device=DEVICE) / depth**0.5
    expected = torch.mm(x, weight.T)
    actual = project_fp16(x, weight)
    assert actual.shape == (tokens, vocab)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or INTERPRETING, reason="full shape requires CUDA execution")
def test_projection_qwen_full_vocabulary():
    from solution.head_matmul import project_fp16

    torch.manual_seed(947)
    x = torch.randn((256, 1536), dtype=torch.float16, device="cuda")
    weight = torch.randn((151936, 1536), dtype=torch.float16, device="cuda") / 1536**0.5
    expected = torch.mm(x, weight.T)
    actual = project_fp16(x, weight)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


def test_projection_rejects_trainable_weight():
    from solution.head_matmul import project_fp16

    x = torch.zeros((2, 17), dtype=torch.float16)
    weight = torch.ones((3, 17), dtype=torch.float16, requires_grad=True)
    with pytest.raises(ValueError, match="frozen"):
        project_fp16(x, weight)


def test_projection_rejects_unowned_autograd():
    from solution.head_matmul import project_fp16

    x = torch.zeros((2, 17), dtype=torch.float16, requires_grad=True)
    weight = torch.ones((3, 17), dtype=torch.float16)
    with pytest.raises(ValueError, match="caller-owned autograd"):
        project_fp16(x, weight)


@pytest.mark.parametrize("shape", [(0, 3, 17), (2, 0, 17)])
def test_projection_empty_output(shape):
    from solution.head_matmul import project_fp16

    tokens, vocab, depth = shape
    x = torch.empty((tokens, depth), dtype=torch.float16)
    weight = torch.empty((vocab, depth), dtype=torch.float16)
    result = project_fp16(x, weight)
    assert result.shape == (tokens, vocab)
    assert result.dtype == torch.float16
