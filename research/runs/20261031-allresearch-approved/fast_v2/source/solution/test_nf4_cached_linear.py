"""Policy and physical native-parity gates for the optional 2D linear route.

CPU tests use an explicit dummy decoder and injected device admission. They
establish metadata, call routing, casting, and native Torch autograd behavior,
not CUDA NF4 math or speed. GPU cases require bitwise native parity; no relaxed
tolerance or performance inference is used to admit this optional route.
"""

from functools import wraps
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from . import nf4_cached_linear as direct
from .nf4_forward_cache import FrozenNF4ForwardCache
from .nf4_forward_cache_patch import _canonical_components, patch_nf4_forward_cache
from .test_nf4_backward_cache import _state, _rng, _assert_rng_equal
from .test_nf4_forward_cache import _components, _DummyDecoder


class _CpuLayer(torch.nn.Module):
    def __init__(self, c, *, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(c["packed"].clone(), requires_grad=False)
        c["packed"] = self.weight
        self.weight.quant_state = _state(c)
        self.out_features, self.in_features = c["shape"]
        self.bias = (torch.nn.Parameter(torch.linspace(-.1, .1, self.out_features,
                                                       dtype=torch.float16),
                                        requires_grad=False) if bias else None)
        self.compute_dtype, self.compute_type_is_set = torch.float16, True


def _cpu_fixture(monkeypatch, *, bias=False, nested=True):
    from . import nf4

    c = _components(nested=nested)
    layer = _CpuLayer(c, bias=bias)
    execute, cache, dispatch_calls = _DummyDecoder(), FrozenNF4ForwardCache(1024), []
    def heuristic(*args):
        dispatch_calls.append(args)
        return False
    ops = SimpleNamespace(_gemm_4bit_custom_max_m=1536, _gemm_4bit_use_custom_fn=heuristic)
    proxy = SimpleNamespace(_state_components=lambda A, state, out, *, cuda_only:
                            nf4._state_components(A, state, out, cuda_only=False),
                            _execute=execute)
    monkeypatch.setattr(direct, "_is_cuda_tensor", lambda tensor: tensor.device.type == "cpu")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return c, layer, cache, execute, ops, proxy, dispatch_calls


class _NativeFrozen2D(torch.autograd.Function):
    """Test-only pinned native formulas, with no quantizer/cache substitutions."""
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(weight)
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, gradient):
        (weight,) = ctx.saved_tensors
        return torch.matmul(gradient, weight), None, None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_nf4_cached_linear_cpu_casts_native_2d_gradient_and_cache(monkeypatch, dtype, bias, nested):
    c, layer, cache, execute, ops, proxy, dispatch_calls = _cpu_fixture(
        monkeypatch, bias=bias, nested=nested)
    delegated = []
    def original(module, x):
        delegated.append(True)
        return _NativeFrozen2D.apply(x.half(), _DummyDecoder()(**c), module.bias).to(x.dtype)
    forward = direct._make_forward(original, cache, _CpuLayer, ops, proxy)
    torch.manual_seed(76)
    x = torch.randn(7, 32, dtype=dtype, requires_grad=True)
    upstream = [torch.randn(7, 4, dtype=dtype) * .01,
                (torch.randn(4, 7, dtype=dtype) * .01).t()]
    assert not upstream[1].is_contiguous()
    before_rng = torch.get_rng_state().clone()
    for gradient in upstream:
        expected = original(layer, x)
        expected_dx, = torch.autograd.grad(expected, x, gradient)
        delegated.clear()
        actual = forward(layer, x)
        dx, = torch.autograd.grad(actual, x, gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
        assert actual.dtype == dx.dtype == dtype and delegated == []
        assert torch.equal(torch.get_rng_state(), before_rng)
        assert layer.weight.grad is None
        if layer.bias is not None:
            assert layer.bias.dtype == torch.float16 and layer.bias.grad is None
    assert execute.calls == 1 and cache.info()["entries"] == 1 and cache.info()["hits"] == 1
    assert dispatch_calls == [(None, torch.float16, 7, 4, 32)] * 2


def test_nf4_cached_linear_cpu_reuses_existing_canonical_entry(monkeypatch):
    c, layer, cache, execute, ops, proxy, _ = _cpu_fixture(monkeypatch)
    first = cache.decode(_canonical_components(dict(c, packed=layer.weight.t())),
                         execute, training=True)
    forward = direct._make_forward(lambda *_: pytest.fail("unexpected native fallback"),
                                   cache, _CpuLayer, ops, proxy)
    x = torch.ones(7, 32, dtype=torch.float32, requires_grad=True)
    result = forward(layer, x)
    result.sum().backward()
    assert execute.calls == 1 and cache.info()["hits"] == 1 and cache.info()["entries"] == 1
    assert first.numel() * first.element_size() == cache.info()["cached_bytes"]


@pytest.mark.parametrize("mode", ["3d", "noncontiguous", "eval", "no_grad", "inference",
                                  "compute_unset", "compute_fp32", "bias_fp32", "bias_trainable",
                                  "missing_state", "state_fp32", "nested_block", "forward_hook",
                                  "hf_hook", "fused_gemm", "one_row", "bf16", "quant_type"])
def test_nf4_cached_linear_cpu_guards_delegate_once_without_decoding(monkeypatch, mode):
    c, layer, cache, execute, ops, proxy, _ = _cpu_fixture(monkeypatch, bias=True)
    x = torch.ones(7, 32, dtype=torch.float32, requires_grad=True)
    if mode == "3d":
        x = x.unsqueeze(0)
    elif mode == "noncontiguous":
        x = torch.ones(32, 7, dtype=torch.float32).t().requires_grad_()
    elif mode == "eval":
        layer.eval()
    elif mode == "compute_unset":
        layer.compute_type_is_set = False
    elif mode == "compute_fp32":
        layer.compute_dtype = torch.float32
    elif mode == "bias_fp32":
        layer.bias = torch.nn.Parameter(layer.bias.float(), requires_grad=False)
    elif mode == "bias_trainable":
        layer.bias.requires_grad_(True)
    elif mode == "missing_state":
        layer.weight.quant_state = None
    elif mode == "state_fp32":
        layer.weight.quant_state.dtype = torch.float32
    elif mode == "nested_block":
        layer.weight.quant_state.state2.blocksize = 64
    elif mode == "forward_hook":
        layer.register_forward_hook(lambda *_: None)
    elif mode == "hf_hook":
        layer._hf_hook = object()
    elif mode == "fused_gemm":
        ops._gemm_4bit_use_custom_fn = lambda *args: True
    elif mode == "one_row":
        x = x[:1]
    elif mode == "bf16":
        x = x.bfloat16()
    elif mode == "quant_type":
        layer.weight.quant_state.quant_type = "fp4"
    calls, sentinel = [], object()
    def original(module, value):
        calls.append((module, value))
        return sentinel
    forward = direct._make_forward(original, cache, _CpuLayer, ops, proxy)
    if mode == "no_grad":
        with torch.no_grad():
            actual = forward(layer, x)
    elif mode == "inference":
        with torch.inference_mode():
            actual = forward(layer, x)
    else:
        actual = forward(layer, x)
    assert actual is sentinel and len(calls) == 1
    assert calls[0][0] is layer and calls[0][1] is x
    assert execute.calls == 0 and cache.info()["entries"] == 0


def test_nf4_cached_linear_cpu_dispatch_uses_native_heuristic_and_cap(monkeypatch):
    c, layer, cache, execute, ops, proxy, calls = _cpu_fixture(monkeypatch)
    x = torch.empty(1537, 32)
    assert direct._eligible_components(layer, x, _CpuLayer, ops, proxy) is not None
    assert calls == []
    x = torch.empty(1536, 32)
    assert direct._eligible_components(layer, x, _CpuLayer, ops, proxy) is not None
    assert calls == [(None, torch.float16, 1536, 4, 32)]
    # A native alignment-warning case below the hard cap is delegated.
    assert direct._selects_native_linear(ops, torch.empty(100, 33), (4, 33), 32) is False
    assert len(calls) == 1
    assert execute.calls == 0 and cache.info()["entries"] == 0


def test_nf4_cached_linear_cpu_fallback_exception_is_not_replayed(monkeypatch):
    _, layer, cache, execute, ops, proxy, _ = _cpu_fixture(monkeypatch)
    calls = []
    def original(module, x):
        calls.append(True)
        raise RuntimeError("native sentinel")
    forward = direct._make_forward(original, cache, _CpuLayer, ops, proxy)
    with pytest.raises(RuntimeError, match="native sentinel"):
        forward(layer, torch.empty(1, 7, 32))
    assert calls == [True] and execute.calls == 0


def test_nf4_cached_linear_installer_requires_same_existing_cache(monkeypatch):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional

    cache, other = FrozenNF4ForwardCache(1024), FrozenNF4ForwardCache(1024)
    def original(module, x):
        return x
    original._soup_nf4_cache_training_context = True
    def public(*args, **kwargs):
        return None
    public._soup_candidate_nf4 = True
    def fallback(*args, **kwargs):
        return None
    fallback._soup_nf4_forward_cache = cache
    fake_ops = SimpleNamespace(_dequant_linear_fallback=fallback,
                               _gemm_4bit_custom_max_m=1536,
                               _gemm_4bit_use_custom_fn=lambda *args: False)
    fake_cuda = ModuleType("bitsandbytes.backends.cuda")
    fake_cuda.ops = fake_ops
    monkeypatch.setitem(sys.modules, "bitsandbytes.backends.cuda", fake_cuda)
    monkeypatch.setattr(torch.version, "cuda", "test-only-CPU-installation")
    monkeypatch.setattr(bnb.nn.Linear4bit, "forward", original)
    monkeypatch.setattr(functional, "dequantize_4bit", public)
    assert direct.patch_nf4_cached_linear(other) is False
    assert direct.patch_nf4_cached_linear(cache) is cache
    installed = bnb.nn.Linear4bit.forward
    assert direct.patch_nf4_cached_linear(cache) is cache
    assert bnb.nn.Linear4bit.forward is installed
    with pytest.raises(ValueError, match="different fixed cache"):
        direct.patch_nf4_cached_linear(other)


_PHYSICAL_CUDA = (torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"
                  and torch.cuda.get_device_capability() == (7, 5))
_DIRECT_CASES = [
    (1600, 64, 96, dtype, nested, bias)
    for dtype in (torch.float16, torch.float32)
    for nested in (False, True)
    for bias in (False, True)
] + [
    (640, width, out_features, dtype, True, False)
    for width, out_features in ((1536, 8960), (8960, 1536))
    for dtype in (torch.float16, torch.float32)
]


def _physical_install(monkeypatch, byte_budget):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    from bitsandbytes.autograd._functions import MatMul4Bit
    from bitsandbytes.backends.cuda import ops
    from . import nf4

    monkeypatch.setattr(functional, "dequantize_4bit", functional.dequantize_4bit)
    monkeypatch.setattr(ops, "_dequant_linear_fallback", ops._dequant_linear_fallback)
    monkeypatch.setattr(bnb.nn.Linear4bit, "forward", bnb.nn.Linear4bit.forward)
    monkeypatch.setattr(MatMul4Bit, "forward", staticmethod(MatMul4Bit.forward))
    monkeypatch.setattr(MatMul4Bit, "backward", staticmethod(MatMul4Bit.backward))
    assert nf4.patch_bitsandbytes_nf4()
    cache = patch_nf4_forward_cache(byte_budget, cache_backward=True)
    counters = dict(decodes=0, matmul_forward=0, matmul_backward=0,
                    public_dequant=0, backend_fallback=0, fused_gemm=0)
    def observed(function, key):
        @wraps(function)
        def wrapper(*args, **kwargs):
            counters[key] += 1
            return function(*args, **kwargs)
        return wrapper
    monkeypatch.setattr(nf4, "_execute", observed(nf4._execute, "decodes"))
    monkeypatch.setattr(MatMul4Bit, "forward", staticmethod(observed(MatMul4Bit.forward, "matmul_forward")))
    monkeypatch.setattr(MatMul4Bit, "backward", staticmethod(observed(MatMul4Bit.backward, "matmul_backward")))
    monkeypatch.setattr(functional, "dequantize_4bit", observed(functional.dequantize_4bit, "public_dequant"))
    monkeypatch.setattr(ops, "_dequant_linear_fallback", observed(ops._dequant_linear_fallback, "backend_fallback"))
    monkeypatch.setattr(ops, "_gemm_4bit_kernel_impl", observed(ops._gemm_4bit_kernel_impl, "fused_gemm"))
    assert direct.patch_nf4_cached_linear(cache) is cache
    return cache, counters


def _sources(layer):
    from . import nf4
    components = nf4._state_components(layer.weight, layer.weight.quant_state, None, cuda_only=True)
    values = [value for value in components.values() if isinstance(value, torch.Tensor)]
    if layer.bias is not None:
        values.append(layer.bias)
    return [(value, value.clone(), value._version, value.dtype) for value in values]


def _assert_sources_unchanged(snapshots):
    for value, snapshot, version, dtype in snapshots:
        assert value._version == version and value.dtype == dtype and torch.equal(value, snapshot)


@pytest.mark.skipif(not _PHYSICAL_CUDA, reason="requires physical T4 native NF4/linear autograd")
@pytest.mark.parametrize("rows,width,out_features,dtype,nested,bias", _DIRECT_CASES)
def test_nf4_cached_linear_physical_exact_native_two_steps(
        monkeypatch, record_property, rows, width, out_features, dtype, nested, bias):
    import bitsandbytes as bnb

    torch.manual_seed(204)
    layer = bnb.nn.Linear4bit(width, out_features, bias=bias, compute_dtype=torch.float16,
                             compress_statistics=nested, quant_type="nf4").half().cuda()
    if layer.bias is not None:
        layer.bias.requires_grad_(False)
    inputs = [torch.randn(rows, width, device="cuda", dtype=dtype, requires_grad=True)
              for _ in range(2)]
    upstream = [torch.randn(rows, out_features, device="cuda", dtype=dtype) * 128,
                (torch.randn(out_features, rows, device="cuda", dtype=dtype) * 128).t()]
    assert not upstream[1].is_contiguous()
    snapshots, before_rng, expected = _sources(layer), _rng(), []
    for x, dy in zip(inputs, upstream):
        y = layer(x)
        dx, = torch.autograd.grad(y, x, dy)
        expected.append((y.detach(), dx))
        _assert_rng_equal(_rng(), before_rng)
    cache, counters = _physical_install(monkeypatch, width * out_features * 2)
    observations = []
    for step, (x, dy, reference) in enumerate(zip(inputs, upstream, expected)):
        before = dict(counters)
        actual = layer(x)
        dx, = torch.autograd.grad(actual, x, dy)
        for value, native in zip((actual, dx), reference):
            torch.testing.assert_close(value, native, rtol=0, atol=0)
        assert actual.dtype == dx.dtype == dtype
        assert counters["decodes"] == 1
        assert all(counters[key] == 0 for key in counters if key != "decodes")
        assert layer.weight.grad is None
        if layer.bias is not None:
            assert layer.bias.grad is None and layer.bias.dtype == torch.float16
        _assert_rng_equal(_rng(), before_rng)
        observations.append(dict(step=step + 1, upstream_contiguous=dy.is_contiguous(),
                                 delta={key: counters[key] - before[key] for key in counters}))
    _assert_sources_unchanged(snapshots)
    assert cache.info()["entries"] == 1 and cache.info()["hits"] == 1
    assert cache.info()["cached_bytes"] == width * out_features * 2
    record_property("soup_nf4_cached_linear", json.dumps(dict(
        shape=[rows, width, out_features], input_dtype=str(dtype), nested=nested, bias=bias,
        steps=observations, cache=cache.info(), direct_route=True,
        exact_output_and_dx=True, frozen_bias_gradient_is_none=True,
        cpu_and_cuda_rng_unchanged=True, frozen_sources_unchanged=True,
        physical_cuda=True, performance_claim=False,
    ), sort_keys=True, allow_nan=False))


_FALLBACK_CASES = [
    (mode, dtype) for mode in ("3d", "noncontiguous", "bias_first_cast", "fused_gemm")
    for dtype in (torch.float16, torch.float32)
] + [("eval", torch.float32), ("trainable_bias", torch.float32)]


@pytest.mark.skipif(not _PHYSICAL_CUDA, reason="requires physical T4 native fallback parity")
@pytest.mark.parametrize("mode,dtype", _FALLBACK_CASES)
def test_nf4_cached_linear_physical_native_fallback_and_first_bias_cast(
        monkeypatch, record_property, mode, dtype):
    import bitsandbytes as bnb

    torch.manual_seed(407)
    layer = bnb.nn.Linear4bit(64, 96, bias=True, compute_dtype=torch.float16,
                             compress_statistics=True, quant_type="nf4").half().cuda()
    layer.bias.requires_grad_(mode == "trainable_bias")
    if mode == "bias_first_cast" or (mode == "3d" and dtype == torch.float32):
        layer.bias = torch.nn.Parameter(layer.bias.float(), requires_grad=False)
    if mode == "eval":
        layer.eval()
    initial_bias = layer.bias.detach().clone()
    initial_bias_trainable = layer.bias.requires_grad
    rows = 64 if mode == "fused_gemm" else 1600
    inputs, upstream = [], []
    for _ in range(2):
        if mode == "3d":
            x = torch.randn(2, 800, 64, device="cuda", dtype=dtype)
        elif mode == "noncontiguous":
            x = torch.randn(64, rows, device="cuda", dtype=dtype).t()
        else:
            x = torch.randn(rows, 64, device="cuda", dtype=dtype)
        inputs.append(x.requires_grad_())
        upstream.append(torch.randn(*x.shape[:-1], 96, device="cuda", dtype=dtype) * 128)
    before_rng, expected = _rng(), []
    for x, dy in zip(inputs, upstream):
        y = layer(x)
        targets = (x, layer.bias) if initial_bias_trainable else (x,)
        grads = torch.autograd.grad(y, targets, dy)
        expected.append((y.detach(), *grads))
        _assert_rng_equal(_rng(), before_rng)
    expected_bias = layer.bias.detach().clone()
    # Restore only the test layer's initial bias so the candidate must take
    # the same native first-call cast, rather than inherit baseline setup.
    layer.bias = torch.nn.Parameter(initial_bias, requires_grad=initial_bias_trainable)
    cache, counters = _physical_install(monkeypatch, 96 * 64 * 2)
    observations = []
    for step, (x, dy, reference) in enumerate(zip(inputs, upstream, expected)):
        before = dict(counters)
        actual = layer(x)
        targets = (x, layer.bias) if initial_bias_trainable else (x,)
        grads = torch.autograd.grad(actual, targets, dy)
        for value, native in zip((actual, *grads), reference):
            torch.testing.assert_close(value, native, rtol=0, atol=0)
        delta = {key: counters[key] - before[key] for key in counters}
        delegated = not (mode == "bias_first_cast" and step == 1)
        assert delta["matmul_forward"] == delta["matmul_backward"] == int(delegated)
        assert delta["public_dequant"] == int(delegated)
        if mode == "fused_gemm" and torch.cuda.get_device_capability() == (7, 5):
            assert delta["fused_gemm"] == 1 and delta["backend_fallback"] == 0
        elif mode != "fused_gemm":
            assert delta["fused_gemm"] == 0 and delta["backend_fallback"] == int(delegated)
        assert actual.dtype == grads[0].dtype == dtype
        _assert_rng_equal(_rng(), before_rng)
        observations.append(dict(step=step + 1, delegated=delegated, delta=delta,
                                 bias_dtype_after=str(layer.bias.dtype)))
    torch.testing.assert_close(layer.bias, expected_bias, rtol=0, atol=0)
    assert layer.bias.dtype == torch.float16
    record_property("soup_nf4_cached_linear_fallback", json.dumps(dict(
        mode=mode, input_dtype=str(dtype), steps=observations, cache=cache.info(),
        exact_output_and_gradients=True, native_bias_cast_preserved=True,
        cpu_and_cuda_rng_unchanged=True, physical_cuda=True, performance_claim=False,
    ), sort_keys=True, allow_nan=False))
