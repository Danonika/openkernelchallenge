"""CPU identity/lifetime tests plus an unrun physical NF4 training gate.

The CPU decoder is injected test code; it establishes cache policy, not NF4
math or GPU speed. The physical test uses actual bitsandbytes Linear4bit,
native baseline forward/backward, and the existing original NF4 decoder.
"""

import gc
import math
import os
import weakref

import pytest
import torch

from .nf4_forward_cache import FrozenNF4ForwardCache


def _components(shape=(4, 32), nested=True, device="cpu"):
    count = math.prod(shape)
    scales = (count + 31) // 32
    return dict(
        packed=(torch.arange((count + 1) // 2, device=device) % 251).to(torch.uint8).reshape(-1, 1),
        shape=shape, dtype=torch.float16, blocksize=32,
        scales=torch.full((scales,), 3, device=device, dtype=torch.uint8 if nested else torch.float32),
        nested_scales=torch.full(((scales + 255) // 256,), .25, device=device) if nested else None,
        nested_code=torch.linspace(-1, 1, 256, device=device) if nested else None,
        offset=torch.tensor(.5, device=device) if nested else None,
        nested_blocksize=256,
    )


class _DummyDecoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, **components):
        self.calls += 1
        # An original deterministic dummy, deliberately independent of cache
        # implementation and unrelated to the real NF4 numeric formula.
        value = components["packed"].flatten().float().repeat_interleave(2)
        value = value[:math.prod(components["shape"])].reshape(components["shape"])
        value = value + components["scales"].flatten()[0].float()
        if components.get("nested_scales") is not None:
            value = value + components["nested_scales"].flatten()[0]
            value = value + components["nested_code"][0] + components["offset"]
        value = value.to(components["dtype"])
        if components.get("out") is not None:
            components["out"].copy_(value)
            return components["out"]
        return value


@pytest.mark.parametrize("nested", [False, True])
def test_nf4_forward_cache_hit_is_exact_and_sources_are_not_written(nested):
    cache, execute, c = FrozenNF4ForwardCache(1024), _DummyDecoder(), _components(nested=nested)
    before = {name: (value.clone(), value._version) for name, value in c.items()
              if isinstance(value, torch.Tensor)}
    first = cache.decode(c, execute, training=True)
    second = cache.decode(c, execute, training=True)
    assert second is first and execute.calls == 1
    assert cache.info() == dict(byte_budget=1024, cached_bytes=256, entries=1,
                               hits=1, misses=1, bypasses=0)
    for name, (value, version) in before.items():
        assert c[name]._version == version and torch.equal(c[name], value)


def test_nf4_forward_cache_reuses_ephemeral_transposed_views_of_parameter():
    c, execute, cache = _components(), _DummyDecoder(), FrozenNF4ForwardCache(1024)
    root = torch.nn.Parameter(c.pop("packed").clone(), requires_grad=False)
    first = cache.decode(dict(c, packed=root.t().view(-1, 1)), execute, training=True)
    gc.collect()
    second = cache.decode(dict(c, packed=root.t().view(-1, 1)), execute, training=True)
    assert first is second and execute.calls == 1
    assert cache.info()["entries"] == 1


@pytest.mark.parametrize("name", ["packed", "scales", "nested_scales", "nested_code", "offset"])
def test_nf4_forward_cache_invalidates_each_source_version(name):
    cache, execute, c = FrozenNF4ForwardCache(1024), _DummyDecoder(), _components()
    first = cache.decode(c, execute, training=True)
    c[name].add_(1)
    second = cache.decode(c, execute, training=True)
    assert second is not first and execute.calls == 2
    assert cache.info()["entries"] == 1 and cache.info()["cached_bytes"] == 256
    fresh = _DummyDecoder()(**c)
    assert torch.equal(second, fresh)


def test_nf4_forward_cache_alias_mutation_invalidates_root_version():
    c, execute, cache = _components(), _DummyDecoder(), FrozenNF4ForwardCache(1024)
    first = cache.decode(c, execute, training=True)
    c["packed"].t()[0, 0] = 17
    second = cache.decode(c, execute, training=True)
    assert first is not second and execute.calls == 2
    assert torch.equal(second, _DummyDecoder()(**c))


def test_nf4_forward_cache_storage_replacement_and_view_layout_are_keys():
    c, execute, cache = _components(), _DummyDecoder(), FrozenNF4ForwardCache(1024)
    root = torch.nn.Parameter(c["packed"].clone(), requires_grad=False)
    c["packed"] = root
    first = cache.decode(c, execute, training=True)
    # .data replacement can preserve _version, so pointer/StorageImpl identity
    # must invalidate independently. Raw alias writes remain unsupported.
    version = root._version
    root.data = torch.full_like(root, 7)
    assert root._version == version
    second = cache.decode(c, execute, training=True)
    assert second is not first and execute.calls == 2
    transposed = dict(c, packed=root.t())
    third = cache.decode(transposed, execute, training=True)
    assert third is not second and execute.calls == 3
    assert torch.equal(third, second)
    # Matrix output shape is also part of the signature, even at equal size.
    reshaped = dict(transposed, shape=(8, 16))
    fourth = cache.decode(reshaped, execute, training=True)
    assert fourth.shape == (8, 16) and execute.calls == 4
    assert cache.info()["entries"] == 1


@pytest.mark.parametrize("name", ["packed", "scales", "nested_scales", "nested_code", "offset"])
def test_nf4_forward_cache_replacement_identity_cannot_reuse_old_values(name):
    c, execute, cache = _components(), _DummyDecoder(), FrozenNF4ForwardCache(1024)
    first = cache.decode(c, execute, training=True)
    old = c[name]  # Keep the old identity alive to test identity, not GC.
    c[name] = old.clone()
    second = cache.decode(c, execute, training=True)
    assert second is not first and execute.calls == 2
    assert torch.equal(second, first)


def test_nf4_forward_cache_mutated_output_is_redecoded_without_writing_old_alias():
    c, execute, cache = _components(), _DummyDecoder(), FrozenNF4ForwardCache(1024)
    first = cache.decode(c, execute, training=True)
    first.add_(10)
    modified = first.clone()
    second = cache.decode(c, execute, training=True)
    assert second is not first and execute.calls == 2
    assert torch.equal(first, modified)
    assert torch.equal(second, _DummyDecoder()(**c))
    second.resize_(2, 64)
    third = cache.decode(c, execute, training=True)
    assert third.shape == (4, 32) and third is not second and execute.calls == 3


def test_nf4_forward_cache_first_fit_budget_never_evicts_valid_entries():
    execute, cache = _DummyDecoder(), FrozenNF4ForwardCache(256)
    a, b = _components(), _components()
    first = cache.decode(a, execute, training=True)
    assert cache.decode(b, execute, training=True) is not cache.decode(b, execute, training=True)
    assert cache.decode(a, execute, training=True) is first
    assert execute.calls == 3 and cache.info()["cached_bytes"] == 256
    assert cache.info()["entries"] == 1
    cache.clear()
    assert cache.info()["entries"] == 0 and cache.info()["cached_bytes"] == 0
    admitted_b = cache.decode(b, execute, training=True)
    assert cache.decode(b, execute, training=True) is admitted_b and execute.calls == 4


@pytest.mark.parametrize("name", ["packed", "scales", "nested_scales", "nested_code", "offset"])
def test_nf4_forward_cache_source_death_releases_entry_and_decoded_storage(name):
    c, execute, cache = _components(), _DummyDecoder(), FrozenNF4ForwardCache(1024)
    result = cache.decode(c, execute, training=True)
    output_ref = weakref.ref(result)
    del result
    del c[name]
    gc.collect()
    assert cache.info()["entries"] == 0 and cache.info()["cached_bytes"] == 0
    assert output_ref() is None


def test_nf4_forward_cache_training_admission_and_inference_bypass():
    c, execute, cache = _components(), _DummyDecoder(), FrozenNF4ForwardCache(1024)
    assert cache.decode(c, execute) is not cache.decode(c, execute)
    with torch.no_grad():  # This is also custom autograd Function.forward.
        first = cache.decode(c, execute, training=True)
        assert cache.decode(c, execute, training=True) is first
    with torch.inference_mode():
        assert cache.decode(c, execute, training=True) is not first
    assert execute.calls == 4 and cache.info()["entries"] == 1
    with torch.inference_mode():
        inference_components = _components()
    cache.decode(inference_components, execute, training=True)
    cache.decode(inference_components, execute, training=True)
    assert execute.calls == 6 and cache.info()["entries"] == 1


def test_nf4_forward_cache_out_dtype_trainable_and_noncontiguous_delegate():
    execute, cache = _DummyDecoder(), FrozenNF4ForwardCache(4096)
    c = _components(nested=False)
    out = torch.empty(c["shape"], dtype=torch.float16)
    for _ in range(2):
        assert cache.decode(dict(c, out=out), execute, training=True) is out
    cache.decode(dict(c, dtype=torch.float32), execute, training=True)
    trainable = dict(c, scales=c["scales"].clone().requires_grad_())
    cache.decode(trainable, execute, training=True)
    noncontiguous = torch.arange(c["packed"].numel()*2, dtype=torch.uint8)[::2].view(-1, 1)
    cache.decode(dict(c, packed=noncontiguous), execute, training=True)
    assert execute.calls == 5 and cache.info()["entries"] == 0


def test_nf4_forward_cache_unsafe_decoder_output_is_not_retained():
    c, cache = _components(nested=False), FrozenNF4ForwardCache(4096)
    # Source storage alias, even though dtype-view's _base may be None.
    source = torch.zeros(256, dtype=torch.uint8)
    c["packed"] = source[:64].view(-1, 1)
    def alias_decoder(**components):
        # set_ creates a non-view TensorImpl with the same storage. Rejecting
        # _base alone would not protect this case.
        return torch.empty(0, dtype=torch.float16).set_(
            source.untyped_storage(), 0, (4, 32), (32, 1))
    cache.decode(c, alias_decoder, training=True)
    assert cache.info()["entries"] == 0
    allocation = torch.zeros(129, dtype=torch.float16)
    def view_decoder(**components):
        return allocation[:128].view(4, 32)
    cache.decode(c, view_decoder, training=True)
    assert cache.info()["entries"] == 0


def test_nf4_forward_cache_mutating_decoder_and_exceptions_never_replay():
    c, cache = _components(), FrozenNF4ForwardCache(4096)
    calls = []
    def mutate(**components):
        calls.append(True)
        components["packed"].add_(1)
        return torch.zeros(components["shape"], dtype=torch.float16)
    cache.decode(c, mutate, training=True)
    assert len(calls) == 1 and cache.info()["entries"] == 0
    def fail(**components):
        calls.append(True)
        raise RuntimeError("decoder failure")
    with pytest.raises(RuntimeError, match="decoder failure"):
        cache.decode(c, fail, training=True)
    assert len(calls) == 2 and cache.info()["entries"] == 0


def test_nf4_forward_cache_budget_validation_and_zero_budget():
    for invalid in (-1, 2.5, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            FrozenNF4ForwardCache(invalid)
    c, execute, cache = _components(), _DummyDecoder(), FrozenNF4ForwardCache(0)
    assert cache.decode(c, execute, training=True) is not cache.decode(c, execute, training=True)
    assert execute.calls == 2 and cache.info()["cached_bytes"] == 0


_PHYSICAL_CUDA = torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"


def test_nf4_cache_training_context_preserves_native_flags_and_exception():
    from .nf4_forward_cache_patch import _TRAINING_FORWARD, _training_wrapper

    class Layer:
        training = True
    layer = Layer()
    observations = []
    def original(module, payload, *, fail=False):
        observations.append((_TRAINING_FORWARD.get(), torch.is_grad_enabled(), module.training))
        # Actual MatMul4Bit.forward disables grad mode after module entry.
        with torch.no_grad():
            observations.append((_TRAINING_FORWARD.get(), torch.is_grad_enabled(), module.training))
        if fail:
            raise LookupError("native sentinel")
        return payload
    wrapped = _training_wrapper(original)
    payload = object()
    assert wrapped(layer, payload) is payload
    assert observations[-2:] == [(True, True, True), (True, False, True)]
    assert _TRAINING_FORWARD.get() is False
    with torch.no_grad():
        assert wrapped(layer, payload) is payload
        assert observations[-2:] == [(False, False, True), (False, False, True)]
    layer.training = False
    assert wrapped(layer, payload) is payload
    assert observations[-2:] == [(False, True, False), (False, False, False)]
    layer.training = True
    with pytest.raises(LookupError, match="native sentinel"):
        wrapped(layer, payload, fail=True)
    assert _TRAINING_FORWARD.get() is False


@pytest.mark.skipif(not _PHYSICAL_CUDA, reason="requires actual CUDA NF4 dispatch and native gradients")
@pytest.mark.parametrize("nested", [False, True])
def test_nf4_forward_cache_physical_linear4bit_exact_forward_and_uncached_backward(monkeypatch, nested):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    from bitsandbytes.backends.cuda import ops
    from . import nf4

    torch.manual_seed(198)
    layer = bnb.nn.Linear4bit(64, 96, bias=True, compute_dtype=torch.float16,
                             compress_statistics=nested, quant_type="nf4").half().cuda()
    x = torch.randn(2, 800, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    upstream = torch.randn(2, 800, 96, device="cuda", dtype=torch.float16) * 128
    expected = layer(x)
    expected_dx, expected_db = torch.autograd.grad(expected, (x, layer.bias), upstream)

    # Register restorations before installing the existing decoder patch.
    monkeypatch.setattr(functional, "dequantize_4bit", functional.dequantize_4bit)
    monkeypatch.setattr(ops, "_dequant_linear_fallback", ops._dequant_linear_fallback)
    monkeypatch.setattr(bnb.nn.Linear4bit, "forward", bnb.nn.Linear4bit.forward)
    assert nf4.patch_bitsandbytes_nf4()
    patched_public = functional.dequantize_4bit
    execute = nf4._execute
    calls, phase = [], ["forward"]
    def observed(**components):
        calls.append(phase[0])
        return execute(**components)
    monkeypatch.setattr(nf4, "_execute", observed)
    # Exercise the actual production installer and its module training context.
    from .nf4_forward_cache_patch import patch_nf4_forward_cache
    cache = patch_nf4_forward_cache(96 * 64 * 2)
    assert isinstance(cache, FrozenNF4ForwardCache)
    for step in range(2):
        phase[0] = "forward"
        before = len(calls)
        actual = layer(x)
        assert len(calls) - before == (1 if step == 0 else 0)
        phase[0] = "backward"
        before = len(calls)
        dx, db = torch.autograd.grad(actual, (x, layer.bias), upstream)
        assert len(calls) - before == 1  # Useful original NF4 runs every step.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
        torch.testing.assert_close(db, expected_db, rtol=0, atol=0)
    assert calls == ["forward", "backward", "backward"]
    assert functional.dequantize_4bit is patched_public
    assert cache.info()["entries"] == 1 and cache.info()["hits"] == 1
