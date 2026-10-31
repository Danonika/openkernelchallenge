"""Optional native-logp consumer checks; no default path is changed.

CPU interpreter tests substitute tl.exp for libdevice.exp because Triton 3.6
cannot interpret extern_elementwise. This verifies indexing/reduction logic,
not CUDA libdevice rounding or performance. Physical tests use the unchanged
libdevice primitive. FP64 entropy references evaluate the SAME native FP32
logp values: they do not silently replace native normalization with FP64.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
import triton.language as tl

from . import native_head_stats as stats


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
PHYSICAL_CUDA = torch.cuda.is_available() and not INTERPRETING
CAN_RUN = PHYSICAL_CUDA or INTERPRETING
DEVICE = "cuda" if PHYSICAL_CUDA else "cpu"
needs_device = pytest.mark.skipif(not CAN_RUN, reason="requires physical CUDA or explicit interpreter")


@pytest.fixture(autouse=True)
def interpreter_exp_substitute(monkeypatch):
    if INTERPRETING:
        # Resolve tl.exp during kernel interpretation, after Triton installs
        # its semantic wrapper; capturing the original builtin is invalid.
        monkeypatch.setattr(stats.libdevice, "exp", lambda value: tl.exp(value))


def _reference(scores, targets):
    lp = F.log_softmax(scores.float(), dim=-1)
    native = (F.nll_loss(lp, targets, reduction="none"),
              -(lp.exp() * lp).sum(-1), scores.argmax(-1))
    lp64 = lp.double()
    entropy64 = -(lp64.exp() * lp64).sum(-1)
    return native, entropy64


def _check(scores, targets, chunk=2):
    expected, entropy64 = _reference(scores, targets)
    actual = stats.native_head_statistics(scores, targets, chunk_size=chunk)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(actual[2], expected[2])
    # Predeclared low-ppm consumer bounds. The FP64 check evaluates the exact
    # native logp input, separating consumer accuracy from normalizer drift.
    torch.testing.assert_close(actual[1].double(), entropy64,
                               rtol=2e-6, atol=2e-7, equal_nan=True)
    torch.testing.assert_close(actual[1], expected[1],
                               rtol=3e-6, atol=3e-7, equal_nan=True)
    assert all(not value.requires_grad for value in actual)
    return actual


@needs_device
@pytest.mark.parametrize("vocab", [17, 2048, 2051, 4097, 151936])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native_head_consumers_loss_entropy_argmax_and_tails(vocab, dtype):
    torch.manual_seed(733)
    scores = (torch.randn(5, vocab, device=DEVICE) * .8).to(dtype).requires_grad_()
    targets = torch.tensor([0, vocab - 1, -100, 3, -100], device=DEVICE)
    _check(scores, targets)
    assert scores.grad is None


@needs_device
@pytest.mark.parametrize("gap", [16.53125, 16.5625, 24.0, 110.0])
def test_native_head_consumers_full_vocab_peak_positions(gap):
    vocab = 151936
    peaks = [0, 31, 4095, 4096, vocab - 1, vocab - 1]
    scores = torch.full((len(peaks), vocab), 32.0-gap,
                        device=DEVICE, dtype=torch.float16)
    scores[torch.arange(len(peaks), device=DEVICE), peaks] = 32
    targets = torch.tensor(peaks, device=DEVICE)
    targets[-1] = -100
    _check(scores, targets, chunk=3)


@needs_device
def test_native_head_consumers_uniform_and_cross_tile_first_ties():
    vocab = 4103
    scores = torch.zeros(5, vocab, device=DEVICE, dtype=torch.float16)
    scores[1, [3, 2047, 2048, 4096, vocab - 1]] = 3
    scores[2, [2048, vocab - 1]] = 7
    scores[3, [4096, vocab - 1]] = 9
    # Normalization rounds this real original-score difference away. Taking
    # argmax from logp would incorrectly choose zero instead of column 4096.
    scores[4, 4096] = 2**-24
    normalized = F.log_softmax(scores[4].float(), dim=-1)
    assert normalized[0].item() == normalized[4096].item()
    targets = torch.tensor([0, 2048, vocab - 1, -100, -100], device=DEVICE)
    actual = _check(scores, targets)
    assert actual[2].tolist() == [0, 3, 2048, 4096, 4096]


@needs_device
def test_native_head_consumers_nonfinite_ignored_and_first_nan_argmax():
    vocab = 4103
    scores = torch.zeros(8, vocab, device=DEVICE, dtype=torch.float16)
    scores[0, [31, 4096]] = float("inf")
    scores[1, 0] = float("inf")
    scores[1, [31, 4096]] = float("nan")
    scores[2, :] = -float("inf")
    scores[3, 4096] = -float("inf")
    scores[4, vocab - 1] = float("nan")
    scores[5, [3, 4096]] = float("nan")
    scores[6, :] = float("nan")
    targets = torch.tensor([31, 0, 1, -100, -100, -100, -100, -100], device=DEVICE)
    actual = _check(scores, targets)
    assert actual[2].tolist() == [31, 31, 0, 0, vocab - 1, 3, 0, 0]
    assert torch.equal(actual[0][3:], torch.zeros_like(actual[0][3:]))
    assert torch.isnan(actual[1][:7]).all()


@needs_device
def test_native_head_consumers_invalid_kernel_targets_never_read_out_of_bounds():
    # Exercise the kernel's independent memory bounds without launching a
    # device assertion, which would poison a CUDA pytest process. The public
    # wrapper's error path is checked separately on the CPU interpreter.
    scores = torch.zeros(4, 17, device=DEVICE, dtype=torch.float16)
    logp = F.log_softmax(scores.float(), dim=-1)
    targets = torch.tensor([-101, 17, 2**40, -100], device=DEVICE)
    loss = torch.empty(4, device=DEVICE)
    entropy = torch.empty_like(loss)
    predictions = torch.empty(4, device=DEVICE, dtype=torch.long)
    stats._consume_native_logp(logp, scores, targets, loss, entropy, predictions)
    assert torch.isnan(loss[:3]).all() and loss[-1].item() == 0
    assert torch.isfinite(entropy).all() and torch.equal(predictions, torch.zeros_like(predictions))


class _NoEntropyMatrices(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten.exp.default:
            raise AssertionError("consumer must not create an ATen exponential matrix")
        if function is torch.ops.aten.mul.Tensor and any(
                isinstance(value, torch.Tensor) and value.ndim == 2 and value.numel() > 32
                for value in args):
            raise AssertionError("consumer must not create an ATen entropy-product matrix")
        return function(*args, **(kwargs or {}))


@needs_device
def test_native_head_consumers_bounded_log_softmax_no_entropy_matrices(monkeypatch):
    scores = torch.zeros(5, 2051, device=DEVICE, dtype=torch.float16, requires_grad=True)
    targets = torch.tensor([0, 1, -100, 3, -100], device=DEVICE)
    original = stats.F.log_softmax
    calls = []
    def observed(value, *args, **kwargs):
        calls.append((tuple(value.shape), value.dtype, torch.is_grad_enabled()))
        return original(value, *args, **kwargs)
    monkeypatch.setattr(stats.F, "log_softmax", observed)
    with _NoEntropyMatrices():
        actual = stats.native_head_statistics(scores, targets, chunk_size=2)
    assert calls == [((2,2051),torch.float32,False),
                     ((2,2051),torch.float32,False),
                     ((1,2051),torch.float32,False)]
    assert all(not value.requires_grad for value in actual)


def test_native_head_consumers_reject_metadata_before_launch():
    scores = torch.zeros(2, 3, dtype=torch.float16)
    targets = torch.tensor([0, 1])
    with pytest.raises(TypeError, match="FP16/FP32"):
        stats.native_head_statistics(scores.double(), targets)
    with pytest.raises(TypeError, match="int64"):
        stats.native_head_statistics(scores, targets.int())
    with pytest.raises(ValueError, match="positive"):
        stats.native_head_statistics(scores, targets, chunk_size=0)
    with pytest.raises(ValueError, match="contiguous"):
        stats.native_head_statistics(scores.t(), torch.zeros(3,dtype=torch.long))
    with pytest.raises(ValueError, match="one target"):
        stats.native_head_statistics(scores, torch.tensor([0]))


@pytest.mark.skipif(not INTERPRETING, reason="CPU-only invalid-label assertion; CUDA assert would poison process")
def test_native_head_consumers_public_invalid_labels_are_rejected():
    scores = torch.zeros(2, 3, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="out of bounds"):
        stats.native_head_statistics(scores, torch.tensor([0, 3]))


@pytest.mark.skipif(not PHYSICAL_CUDA, reason="requires physical CUDA with real libdevice.exp")
def test_native_head_consumers_physical_full_vocab_and_original_math():
    assert stats.libdevice.exp is not tl.exp
    scores = torch.randn(8, 151936, device="cuda", dtype=torch.float16)
    targets = torch.tensor([0, 31, 4095, -100, 4096, 151935, 7, -100], device="cuda")
    _check(scores, targets, chunk=3)
