"""Scope/identity checks and physical gates for optional frozen backward reuse.

CPU tests inject a dummy decoder and establish policy only. Physical tests
compare real native Linear4bit with both fixed cache scopes for two training
steps, and require exact outputs, input/bias gradients and RNG state. No CPU
result is evidence of CUDA correctness, kernel credit, or end-to-end speed.
"""

import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from . import nf4_forward_cache as cache_module
from .nf4_forward_cache import FrozenNF4ForwardCache
from .nf4_forward_cache_patch import (
    _NATIVE_BACKWARD,
    _TRAINING_FORWARD,
    _backward_dequantize_wrapper,
    _canonical_components,
    _matmul_backward_wrapper,
    _matmul_forward_wrapper,
    _training_wrapper,
    patch_nf4_forward_cache,
)
from .test_nf4_forward_cache import _components, _DummyDecoder


def _state(c):
    nested = c["nested_scales"] is not None
    state2 = (SimpleNamespace(absmax=c["nested_scales"], code=c["nested_code"],
                              blocksize=c["nested_blocksize"], dtype=torch.float32,
                              nested=False) if nested else None)
    return SimpleNamespace(shape=c["shape"], dtype=c["dtype"], blocksize=c["blocksize"],
                           absmax=c["scales"], offset=c["offset"], quant_type="nf4",
                           nested=nested, state2=state2)


def _ctx(packed, state, *, training=True):
    return SimpleNamespace(_soup_nf4_cache_training=training, is_empty=False,
                           needs_input_grad=(True, False, False, True, False),
                           dtype_A=torch.float16, tensors=(None, packed), state=state)


def _cpu_public_fixture(c):
    """Inject CPU metadata admission, never a production CPU/GPU substitute."""
    from . import nf4

    execute, delegates = _DummyDecoder(), []
    proxy = SimpleNamespace(
        _state_components=lambda A, state, out, *, cuda_only:
            nf4._state_components(A, state, out, cuda_only=False),
        _execute=execute,
    )
    def original(A, quant_state=None, absmax=None, out=None,
                 blocksize=None, quant_type="fp4"):
        delegates.append((A, quant_state, absmax, out, blocksize, quant_type))
        components = nf4._state_components(A, quant_state, out, cuda_only=False)
        if components is None:
            return "unsupported native sentinel"
        value = _DummyDecoder()(**components)
        return value.t() if A.shape[0] == 1 else value
    cache = FrozenNF4ForwardCache(1024)
    public = _backward_dequantize_wrapper(original, cache, proxy)
    return cache, execute, delegates, public


def _run_public_in_backward(ctx, public, **kwargs):
    def native(context, gradient):
        assert _NATIVE_BACKWARD.get() is context
        return public(context.tensors[1], context.state, **kwargs)
    return _matmul_backward_wrapper(native)(ctx, torch.ones(1, dtype=torch.float16))


def test_nf4_backward_cache_root_metadata_is_stamped_once(monkeypatch):
    source = torch.nn.Parameter(torch.arange(64, dtype=torch.uint8).view(-1, 1),
                                requires_grad=False)
    stamp, calls = cache_module._stamp, []
    def counted(tensor):
        calls.append(tensor)
        return stamp(tensor)
    monkeypatch.setattr(cache_module, "_stamp", counted)
    owner, identity = cache_module._root_and_stamp(source)
    assert owner is source and len(calls) == 1 and calls[0] is source
    assert identity == (id(source), stamp(source), stamp(source))
    calls.clear()
    view = source.t()
    owner, identity = cache_module._root_and_stamp(view)
    assert owner is source and len(calls) == 2
    assert calls[0] is view and calls[1] is source
    assert identity == (id(source), stamp(view), stamp(source))


@pytest.mark.parametrize("nested", [False, True])
def test_nf4_backward_cache_canonical_transpose_shares_forward_entry(nested):
    c = _components(nested=nested)
    c["packed"] = torch.nn.Parameter(c["packed"].clone(), requires_grad=False)
    cache, execute, delegates, public = _cpu_public_fixture(c)
    version, original_shape = c["packed"]._version, c["packed"].shape
    forward = cache.decode(_canonical_components(c), execute, training=True)
    canonical_ctx = _ctx(c["packed"].reshape(-1, 1), _state(c))
    backward = _run_public_in_backward(canonical_ctx, public)
    assert backward is forward
    transposed_ctx = _ctx(c["packed"].t(), canonical_ctx.state)
    transposed = _run_public_in_backward(transposed_ctx, public)
    assert transposed.data_ptr() == forward.data_ptr()
    assert torch.equal(transposed, forward.t())
    assert execute.calls == 1 and delegates == []
    assert cache.info() == dict(byte_budget=1024, cached_bytes=256, entries=1,
                               hits=2, misses=1, bypasses=0)
    assert c["packed"]._version == version and c["packed"].shape == original_shape
    assert _NATIVE_BACKWARD.get() is None


def test_nf4_backward_cache_public_and_out_calls_preserve_original_arguments():
    c = _components()
    cache, execute, delegates, public = _cpu_public_fixture(c)
    state = _state(c)
    expected = _DummyDecoder()(**c)
    actual = public(c["packed"].t(), state, quant_type="nf4")
    assert torch.equal(actual, expected.t())
    assert execute.calls == 0 and cache.info()["entries"] == 0
    assert delegates[-1][0].data_ptr() == c["packed"].data_ptr()
    assert delegates[-1][1] is state and delegates[-1][-1] == "nf4"
    ctx = _ctx(c["packed"].t(), state)
    out = torch.empty(c["shape"], dtype=torch.float16)
    result = _run_public_in_backward(ctx, public, out=out, absmax=c["scales"],
                                     blocksize=32, quant_type="nf4")
    assert result.data_ptr() == out.data_ptr() and torch.equal(result, expected.t())
    args = delegates[-1]
    assert args[0] is ctx.tensors[1] and args[1] is state and args[2] is c["scales"]
    assert args[3] is out and args[4:] == (32, "nf4")
    assert execute.calls == 0 and cache.info()["entries"] == 0


@pytest.mark.parametrize("unrelated", ["packed", "state"])
def test_nf4_backward_cache_unrelated_public_call_in_context_delegates(unrelated):
    c = _components()
    cache, execute, delegates, public = _cpu_public_fixture(c)
    state = _state(c)
    ctx = _ctx(c["packed"], state)
    other_packed = c["packed"].clone() if unrelated == "packed" else c["packed"]
    other_state = _state(c) if unrelated == "state" else state
    def native(context, gradient):
        return public(other_packed, other_state)
    result = _matmul_backward_wrapper(native)(ctx, torch.ones(1, dtype=torch.float16))
    assert torch.equal(result, _DummyDecoder()(**c))
    assert len(delegates) == 1 and execute.calls == 0 and cache.info()["entries"] == 0


@pytest.mark.parametrize("mode", ["eval", "empty", "no_input_gradient", "fp32_input",
                                  "fp32_gradient", "inference"])
def test_nf4_backward_cache_native_scope_rejects_ineligible_backward(mode):
    c = _components()
    cache, execute, delegates, public = _cpu_public_fixture(c)
    ctx, gradient = _ctx(c["packed"], _state(c)), torch.ones(1, dtype=torch.float16)
    if mode == "eval":
        ctx._soup_nf4_cache_training = False
    elif mode == "empty":
        ctx.is_empty = True
    elif mode == "no_input_gradient":
        ctx.needs_input_grad = (False, False, False, True, False)
    elif mode == "fp32_input":
        ctx.dtype_A = torch.float32
    elif mode == "fp32_gradient":
        gradient = gradient.float()
    def native(context, grad):
        assert _NATIVE_BACKWARD.get() is None
        return public(context.tensors[1], context.state)
    wrapped = _matmul_backward_wrapper(native)
    if mode == "inference":
        with torch.inference_mode():
            result = wrapped(ctx, gradient)
    else:
        result = wrapped(ctx, gradient)
    assert torch.equal(result, _DummyDecoder()(**c))
    assert len(delegates) == 1 and execute.calls == 0 and cache.info()["entries"] == 0
    assert _NATIVE_BACKWARD.get() is None


def test_nf4_backward_cache_fp32_state_and_unsupported_quantization_delegate():
    c = _components()
    cache, execute, delegates, public = _cpu_public_fixture(c)
    state = _state(c)
    ctx = _ctx(c["packed"], state)
    state.dtype = torch.float32
    actual = _run_public_in_backward(ctx, public)
    assert actual.dtype == torch.float32
    state.dtype, state.quant_type = torch.float16, "fp4"
    assert _run_public_in_backward(ctx, public) == "unsupported native sentinel"
    assert len(delegates) == 2 and execute.calls == 0 and cache.info()["entries"] == 0


def test_nf4_backward_cache_forward_intent_and_context_restore_exactly():
    c = _components()
    ctx = _ctx(c["packed"], _state(c))
    module = SimpleNamespace(training=True)
    payload, calls = object(), []
    def native_forward(context, item):
        calls.append(("forward", torch.is_grad_enabled(), _TRAINING_FORWARD.get()))
        return item
    forward = _matmul_forward_wrapper(native_forward)
    def module_forward(layer, item):
        # Match Function.forward's grad-mode boundary in this policy test.
        with torch.no_grad():
            return forward(ctx, item)
    wrapped_module = _training_wrapper(module_forward)
    assert wrapped_module(module, payload) is payload
    assert ctx._soup_nf4_cache_training is True
    assert calls == [("forward", False, True)] and _TRAINING_FORWARD.get() is False
    def native_backward(context, gradient):
        calls.append(("backward", torch.is_grad_enabled(), _NATIVE_BACKWARD.get() is context))
        return payload
    backward = _matmul_backward_wrapper(native_backward)
    gradient = torch.ones(1, dtype=torch.float16)
    with torch.no_grad():
        assert backward(ctx, gradient) is payload
    assert calls[-1] == ("backward", False, True) and _NATIVE_BACKWARD.get() is None
    with torch.no_grad():
        assert wrapped_module(module, payload) is payload
    assert ctx._soup_nf4_cache_training is False
    module.training = False
    assert wrapped_module(module, payload) is payload
    assert ctx._soup_nf4_cache_training is False

    ctx._soup_nf4_cache_training = True
    outer = object()
    token = _NATIVE_BACKWARD.set(outer)
    def fail(context, grad):
        assert _NATIVE_BACKWARD.get() is context
        raise LookupError("native backward sentinel")
    try:
        with pytest.raises(LookupError, match="native backward sentinel"):
            _matmul_backward_wrapper(fail)(ctx, gradient)
        assert _NATIVE_BACKWARD.get() is outer
    finally:
        _NATIVE_BACKWARD.reset(token)


@pytest.mark.parametrize("cache_backward", [False, True])
def test_nf4_backward_cache_installation_keeps_one_fixed_scope(monkeypatch, cache_backward):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    from bitsandbytes.autograd._functions import MatMul4Bit

    native_forward, native_backward = MatMul4Bit.forward, MatMul4Bit.backward
    def public(*args, **kwargs):
        return "candidate public sentinel"
    def fallback(*args, **kwargs):
        return "candidate fallback sentinel"
    public._soup_candidate_nf4 = fallback._soup_candidate_nf4 = True
    fake_ops = SimpleNamespace(_dequant_linear_fallback=fallback)
    fake_cuda = ModuleType("bitsandbytes.backends.cuda")
    fake_cuda.ops = fake_ops
    monkeypatch.setitem(sys.modules, "bitsandbytes.backends.cuda", fake_cuda)
    monkeypatch.setattr(torch.version, "cuda", "test-only-CPU-installation")
    monkeypatch.setattr(functional, "dequantize_4bit", public)
    monkeypatch.setattr(bnb.nn.Linear4bit, "forward", bnb.nn.Linear4bit.forward)
    monkeypatch.setattr(MatMul4Bit, "forward", staticmethod(native_forward))
    monkeypatch.setattr(MatMul4Bit, "backward", staticmethod(native_backward))
    cache = patch_nf4_forward_cache(1024, cache_backward=cache_backward)
    assert isinstance(cache, FrozenNF4ForwardCache)
    assert patch_nf4_forward_cache(1024, cache_backward=cache_backward) is cache
    assert (MatMul4Bit.forward is native_forward) is (not cache_backward)
    assert (MatMul4Bit.backward is native_backward) is (not cache_backward)
    assert (functional.dequantize_4bit is public) is (not cache_backward)
    assert fake_ops._dequant_linear_fallback._soup_nf4_cache_backward is cache_backward
    with pytest.raises(ValueError, match="different fixed backward scope"):
        patch_nf4_forward_cache(1024, cache_backward=not cache_backward)
    with pytest.raises(ValueError, match="different fixed budget"):
        patch_nf4_forward_cache(2048, cache_backward=cache_backward)


@pytest.mark.parametrize("invalid", [None, 0, 1, "yes"])
def test_nf4_backward_cache_requires_explicit_boolean_scope(invalid):
    with pytest.raises(ValueError, match="must be a bool"):
        patch_nf4_forward_cache(1024, cache_backward=invalid)


_PHYSICAL_CUDA = torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"


def _rng():
    return (torch.get_rng_state().clone(),
            tuple(state.clone() for state in torch.cuda.get_rng_state_all()))


def _assert_rng_equal(actual, expected):
    assert torch.equal(actual[0], expected[0]) and len(actual[1]) == len(expected[1])
    assert all(torch.equal(a, b) for a, b in zip(actual[1], expected[1]))


@pytest.mark.skipif(not _PHYSICAL_CUDA, reason="requires physical CUDA native NF4 forward/backward")
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("cache_backward", [False, True])
def test_nf4_backward_cache_physical_linear4bit_exact_two_steps(
        monkeypatch, record_property, nested, cache_backward):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    from bitsandbytes.autograd._functions import MatMul4Bit
    from bitsandbytes.backends.cuda import ops
    from . import nf4

    torch.manual_seed(493)
    layer = bnb.nn.Linear4bit(64, 96, bias=True, compute_dtype=torch.float16,
                             compress_statistics=nested, quant_type="nf4").half().cuda()
    inputs = [torch.randn(2, 800, 64, device="cuda", dtype=torch.float16,
                          requires_grad=True) for _ in range(2)]
    upstream = [torch.randn(2, 800, 96, device="cuda", dtype=torch.float16) * 128
                for _ in range(2)]
    state = layer.weight.quant_state
    source = nf4._state_components(layer.weight, state, None, cuda_only=True)
    snapshots = {name: (value, value.clone(), value._version)
                 for name, value in source.items() if isinstance(value, torch.Tensor)}
    native_weight = functional.dequantize_4bit(layer.weight, state)
    before_rng = _rng()
    expected = []
    for x, dy in zip(inputs, upstream):
        y = layer(x)
        dx, db = torch.autograd.grad(y, (x, layer.bias), dy)
        expected.append((y.detach(), dx, db))
        _assert_rng_equal(_rng(), before_rng)

    # Restore all five native boundaries on exit; no wrapper survives into
    # another test or into a purported native reference run.
    monkeypatch.setattr(functional, "dequantize_4bit", functional.dequantize_4bit)
    monkeypatch.setattr(ops, "_dequant_linear_fallback", ops._dequant_linear_fallback)
    monkeypatch.setattr(bnb.nn.Linear4bit, "forward", bnb.nn.Linear4bit.forward)
    monkeypatch.setattr(MatMul4Bit, "forward", staticmethod(MatMul4Bit.forward))
    monkeypatch.setattr(MatMul4Bit, "backward", staticmethod(MatMul4Bit.backward))
    assert nf4.patch_bitsandbytes_nf4()
    execute, calls, phase = nf4._execute, [], ["forward"]
    def observed(**components):
        calls.append(phase[0])
        return execute(**components)
    monkeypatch.setattr(nf4, "_execute", observed)
    cache = patch_nf4_forward_cache(96 * 64 * 2, cache_backward=cache_backward)
    counts = []
    for step, (x, dy, reference) in enumerate(zip(inputs, upstream, expected)):
        phase[0] = "forward"
        before = len(calls)
        y = layer(x)
        forward_decodes = len(calls) - before
        assert forward_decodes == (1 if step == 0 else 0)
        phase[0] = "backward"
        before = len(calls)
        dx, db = torch.autograd.grad(y, (x, layer.bias), dy)
        backward_decodes = len(calls) - before
        assert backward_decodes == (0 if cache_backward else 1)
        counts.append(dict(step=step + 1, forward_decodes=forward_decodes,
                           backward_decodes=backward_decodes))
        for actual, native in zip((y, dx, db), reference):
            torch.testing.assert_close(actual, native, rtol=0, atol=0)
        _assert_rng_equal(_rng(), before_rng)
    training_info = cache.info()
    assert training_info["entries"] == 1 and training_info["cached_bytes"] == 96 * 64 * 2
    assert training_info["hits"] == (3 if cache_backward else 1)
    assert training_info["misses"] == 1
    assert calls == (["forward"] if cache_backward else ["forward", "backward", "backward"])

    # Public transpose/out behavior is untouched even after a cache is full.
    phase[0] = "public"
    actual = functional.dequantize_4bit(layer.weight.t(), state)
    torch.testing.assert_close(actual, native_weight.t(), rtol=0, atol=0)
    out = torch.empty(state.shape, dtype=state.dtype, device="cuda")
    actual = functional.dequantize_4bit(layer.weight.t(), state, out=out)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, native_weight.t(), rtol=0, atol=0)
    assert calls[-2:] == ["public", "public"]
    assert cache.info() == training_info

    # eval() with gradients still has a real backward; neither phase may use
    # the training cache, and native bias/input gradients must remain exact.
    layer.eval()
    phase[0] = "eval_forward"
    y = layer(inputs[0])
    phase[0] = "eval_backward"
    dx, db = torch.autograd.grad(y, (inputs[0], layer.bias), upstream[0])
    for actual, native in zip((y, dx, db), expected[0]):
        torch.testing.assert_close(actual, native, rtol=0, atol=0)
    assert calls[-2:] == ["eval_forward", "eval_backward"]
    assert cache.info() == training_info
    _assert_rng_equal(_rng(), before_rng)
    for value, snapshot, version in snapshots.values():
        assert value._version == version and torch.equal(value, snapshot)
    assert _NATIVE_BACKWARD.get() is None and _TRAINING_FORWARD.get() is False
    record_property("soup_nf4_backward_cache", json.dumps(dict(
        nested=nested, cache_backward=cache_backward, steps=counts,
        cache_after_training=training_info, exact_output_and_dx_and_db=True,
        cpu_and_cuda_rng_unchanged=True, frozen_sources_unchanged=True,
        public_transpose_and_out_bypass=True, eval_forward_and_backward_bypass=True,
        physical_cuda=True, performance_claim=False,
    ), sort_keys=True, allow_nan=False))
