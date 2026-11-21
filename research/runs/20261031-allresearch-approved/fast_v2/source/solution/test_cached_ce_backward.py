"""Inactive gradient consumer: CPU formula screen and exact physical gate.

Triton 3.6's interpreter cannot execute libdevice.exp and implements fma as
an unfused NumPy expression. The explicit CPU-only fixture substitutes
NumPy exp through interpreted tl.exp and simulates FP32 fused arithmetic.
The independent CPU reference uses that same exponential input and C99
fmaf for the sparse target term. CPU success is NOT native CUDA parity.
Physical CUDA runs use unmodified libdevice.exp / hardware FMA and require
exact native FP16 autograd gradients, including upstream scale 65536.
"""

import ctypes
import ctypes.util
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
import triton.language as tl

from . import cached_ce_backward as backward


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
PHYSICAL = torch.cuda.is_available() and not INTERPRETING
DEVICE = "cuda" if PHYSICAL else "cpu"
needs_device = pytest.mark.skipif(not (PHYSICAL or INTERPRETING),
                                  reason="requires physical CUDA or explicit CPU interpreter")


def _c99_fmaf():
    library = ctypes.CDLL(ctypes.util.find_library("m"))
    operation = library.fmaf
    operation.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    operation.restype = ctypes.c_float
    return operation


@pytest.fixture(autouse=True)
def explicit_interpreter_math_substitutes(monkeypatch):
    if not INTERPRETING:
        return
    from triton.runtime.interpreter import TensorHandle, interpreter_builder
    fmaf = _c99_fmaf()
    monkeypatch.setattr(backward.libdevice, "exp", lambda value: tl.exp(value))

    def fma_fp32(x, y, z):
        a, b, c = np.broadcast_arrays(x.data, y.data, z.data)
        assert a.dtype == b.dtype == c.dtype == np.float32
        # Off-target z is +0, so FP64 holds the full FP32 product exactly.
        # Use actual C99 fmaf for every nonzero/negative-zero sparse addend,
        # including NaN/Inf; this avoids double-rounding at the label.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = (a.astype(np.float64) * b.astype(np.float64)
                      + c.astype(np.float64)).astype(np.float32)
        flat_result = result.reshape(-1)
        fa, fb, fc = a.reshape(-1), b.reshape(-1), c.reshape(-1)
        for i in np.flatnonzero((fc != 0) | np.signbit(fc)):
            flat_result[i] = fmaf(float(fa[i]), float(fb[i]), float(fc[i]))
        return TensorHandle(result, z.dtype.scalar)

    monkeypatch.setattr(interpreter_builder, "create_fma", fma_fp32)


def _native(logits, targets, divisor, upstream):
    leaf = logits.detach().clone().requires_grad_()
    loss = F.cross_entropy(leaf.float(), targets, ignore_index=-100,
                           reduction="sum") / divisor
    gradient, = torch.autograd.grad(loss, leaf, grad_outputs=upstream)
    return gradient


def _interpreter_formula(logits, targets, divisor, upstream):
    # Native normalization remains an actual Torch tensor result. This is
    # the consumer's defining sparse formula, not a replacement normalizer.
    lp = F.log_softmax(logits.float(), dim=-1).numpy()
    scaled = np.float32((upstream / divisor).item())
    labels = targets.numpy()
    row_sum = np.where(labels == -100, np.float32(0), -scaled).astype(np.float32)
    row_sum[row_sum == 0] = np.float32(0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        probability = np.exp(lp)
        output = (-probability.astype(np.float64) * row_sum[:, None].astype(np.float64)
                  + np.float64(0)).astype(np.float32)
    fmaf = _c99_fmaf()
    for row, target in enumerate(labels):
        if target != -100:
            output[row, target] = fmaf(-float(probability[row, target]),
                                       float(row_sum[row]), float(-scaled))
    return torch.from_numpy(output).half()


def _check(logits, targets, *, denominator=13., scale=65536., chunk=2):
    divisor = torch.tensor(denominator, device=DEVICE, dtype=torch.float32)
    upstream = torch.tensor(scale, device=DEVICE, dtype=torch.float32)
    expected_native = _native(logits, targets, divisor, upstream)
    actual = backward.native_logsoftmax_ce_backward(logits, targets, divisor,
                                                     upstream, chunk_size=chunk)
    if INTERPRETING:
        expected = _interpreter_formula(logits, targets, divisor, upstream)
    else:
        expected = expected_native
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zero = (actual == 0) & (expected == 0)
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))
    assert actual.dtype == torch.float16 and actual.shape == logits.shape
    assert not actual.requires_grad and actual.grad_fn is None
    assert actual.data_ptr() != logits.data_ptr()
    return actual, expected_native


@needs_device
@pytest.mark.parametrize("vocab", [1, 17, 1024, 1025, 4099, 151936])
@pytest.mark.parametrize("scale", [1., 65536.])
def test_cached_ce_consumer_random_native_gradient_and_tails(vocab, scale):
    torch.manual_seed(934)
    logits = (torch.randn(5, vocab, device=DEVICE) * .8).half()
    targets = torch.tensor([0, vocab-1, -100, min(3, vocab-1), -100], device=DEVICE)
    _check(logits, targets, scale=scale)


@needs_device
@pytest.mark.parametrize("gap", [16.53125, 16.5625, 24., 110.])
@pytest.mark.parametrize("scale", [1., 65536.])
def test_cached_ce_consumer_full_vocab_saturated_peak_columns(gap, scale):
    vocab = 151936
    peaks = [0, 31, 4095, 4096, vocab-1, vocab-1]
    logits = torch.full((len(peaks), vocab), 32-gap, dtype=torch.float16, device=DEVICE)
    logits[torch.arange(len(peaks), device=DEVICE), peaks] = 32
    targets = torch.tensor(peaks, device=DEVICE)
    targets[-1] = -100
    _check(logits, targets, denominator=2048., scale=scale, chunk=3)


@needs_device
@pytest.mark.parametrize("denominator", [0., -7., 7., float("inf"), float("nan")])
@pytest.mark.parametrize("scale", [1., 65536.])
def test_cached_ce_consumer_nonfinite_ignored_and_explicit_divisors(denominator, scale):
    logits = torch.zeros(8, 259, device=DEVICE, dtype=torch.float16)
    logits[0, 17] = float("nan")
    logits[1, 7] = float("nan")
    logits[2, 31] = float("inf")
    logits[3, :] = -float("inf")
    logits[4, 17] = -float("inf")
    logits[5, 31] = -float("inf")
    targets = torch.tensor([17, -100, 31, -100, 0, -100, -100, 7], device=DEVICE)
    _check(logits, targets, denominator=denominator, scale=scale, chunk=3)


@needs_device
@pytest.mark.parametrize("denominator,scale", [(0.,65536.), (0.,0.), (7.,float("inf")),
                                               (7.,float("nan")), (float("nan"),65536.)])
def test_cached_ce_consumer_all_ignored_has_zero_finite_row_gradients(denominator, scale):
    logits = torch.zeros(4, 131, device=DEVICE, dtype=torch.float16)
    targets = torch.full((4,), -100, device=DEVICE, dtype=torch.long)
    actual, native = _check(logits, targets, denominator=denominator, scale=scale)
    assert torch.equal(actual, torch.zeros_like(actual))
    assert torch.equal(native, torch.zeros_like(native))


@needs_device
@pytest.mark.parametrize("scale", [0., -0., -65536., float("inf"), float("nan")])
def test_cached_ce_consumer_upstream_sign_zero_and_nonfinite(scale):
    logits = torch.tensor([[0.,-110.,-1.], [1.,2.,3.], [0.,0.,0.]],
                          device=DEVICE, dtype=torch.float16)
    targets = torch.tensor([0, 2, -100], device=DEVICE)
    _check(logits, targets, denominator=2048., scale=scale, chunk=1)


@needs_device
def test_cached_ce_consumer_scale_is_applied_before_half_cast():
    logits = torch.zeros(3, 4103, device=DEVICE, dtype=torch.float16)
    targets = torch.tensor([0, 4102, -100], device=DEVICE)
    actual, _ = _check(logits, targets, denominator=2048., scale=65536.)
    assert torch.isfinite(actual).all()
    assert actual[0,1].item() > 0
    assert -33 < actual[0,0].item() < -31


class _NoNativeBackwardMatrices(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.divisions = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        assert function not in (
            torch.ops.aten.nll_loss_forward.default,
            torch.ops.aten.nll_loss_backward.default,
            torch.ops.aten._log_softmax_backward_data.default,
            torch.ops.aten.exp.default,
        ), "consumer must not reconstruct native NLL/backward/exp matrices"
        if function is torch.ops.aten.div.Tensor:
            self.divisions.append(tuple(tuple(x.shape) for x in args[:2]))
        return function(*args, **(kwargs or {}))


@needs_device
def test_cached_ce_consumer_native_work_is_bounded_and_scalar_division_shared(monkeypatch):
    logits = torch.zeros(5, 4103, device=DEVICE, dtype=torch.float16)
    targets = torch.tensor([0, 4102, -100, 7, -100], device=DEVICE)
    divisor = torch.tensor(13., device=DEVICE)
    upstream = torch.tensor(65536., device=DEVICE)
    original, calls = backward.F.log_softmax, []
    def observed(value, *args, **kwargs):
        calls.append((tuple(value.shape), value.dtype, torch.is_grad_enabled()))
        return original(value, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("helper must not call native autograd.grad")
    monkeypatch.setattr(backward.F, "log_softmax", observed)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    dispatch = _NoNativeBackwardMatrices()
    with dispatch:
        result = backward.native_logsoftmax_ce_backward(logits, targets, divisor, upstream, 2)
    assert calls == [((2,4103),torch.float32,False), ((2,4103),torch.float32,False),
                     ((1,4103),torch.float32,False)]
    assert dispatch.divisions == [((), ())]
    assert not result.requires_grad


@needs_device
def test_cached_ce_consumer_invalid_private_targets_are_memory_safe():
    logits = torch.zeros(3,17,device=DEVICE,dtype=torch.float16)
    lp = F.log_softmax(logits.float(),dim=-1)
    targets = torch.tensor([-101,2**40,-100],device=DEVICE)
    output = torch.empty_like(logits)
    backward._consume_logp_gradient(lp,targets,torch.tensor(32.,device=DEVICE),output)
    assert torch.isnan(output[:2]).all() and torch.equal(output[2],torch.zeros_like(output[2]))


def test_cached_ce_consumer_metadata_and_higher_order_guards():
    z, t, d, g = torch.zeros(2,3,dtype=torch.float16), torch.tensor([0,1]), torch.tensor(2.), torch.tensor(1.)
    with pytest.raises(TypeError,match="FP16"):
        backward.native_logsoftmax_ce_backward(z.float(),t,d,g)
    with pytest.raises(ValueError,match="positive"):
        backward.native_logsoftmax_ce_backward(z,t,d,g,0)
    with pytest.raises(ValueError,match="scalar FP32"):
        backward.native_logsoftmax_ce_backward(z,t,d.half(),g)
    with pytest.raises(ValueError,match="one target"):
        backward.native_logsoftmax_ce_backward(z,t[:1],d,g)
    with pytest.raises(ValueError,match="higher-order"):
        backward.native_logsoftmax_ce_backward(z.requires_grad_(),t,d,g)


@pytest.mark.skipif(not INTERPRETING,reason="CPU-only invalid label assertion avoids poisoning CUDA process")
def test_cached_ce_consumer_public_invalid_labels_are_rejected():
    with pytest.raises(RuntimeError,match="out of bounds"):
        backward.native_logsoftmax_ce_backward(torch.zeros(1,3,dtype=torch.float16),
                                               torch.tensor([3]),torch.tensor(1.),torch.tensor(65536.))


@pytest.mark.skipif(not PHYSICAL,reason="requires physical CUDA with actual libdevice.exp and hardware FMA")
def test_cached_ce_consumer_physical_full_vocab_exact_native_scaled_gradients():
    assert backward.libdevice.exp is not tl.exp
    torch.manual_seed(848)
    logits = torch.randn(8,151936,device="cuda",dtype=torch.float16)
    targets = torch.tensor([0,31,4095,4096,151935,-100,7,-100],device="cuda")
    _check(logits,targets,denominator=2048.,scale=65536.,chunk=3)
