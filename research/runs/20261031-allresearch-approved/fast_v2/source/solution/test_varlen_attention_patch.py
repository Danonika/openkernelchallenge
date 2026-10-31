"""Registry/context plumbing plus real physical Qwen2 composition gates.

CPU dispatch fixtures explicitly replace only device admission and the
standalone varlen primitive. Their sentinels test routing, mask checks and
lifetime, not attention numerics. Physical tests have no guard or GEMM
substitution and use actual native nested-NF4, PEFT, Qwen2 and SDPA.
"""

from contextlib import contextmanager
import json
import os

import pytest
import torch

from . import mlp_padding
from . import varlen_attention as primitive
from . import varlen_attention_patch as patch


PHYSICAL = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
                             reason="requires physical CUDA, not the interpreter")


def _modules():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    config = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "sdpa"
    return (Qwen2Attention(config, layer_idx=0), Qwen2Attention(config, layer_idx=1))


def _inputs(modules):
    mask = torch.tensor([[1] * 9, [1] * 5 + [0] * 4], dtype=torch.long)
    plan = primitive.make_varlen_plan(mask, device="cpu")
    call = mlp_padding._PaddingCall(tuple(mask.shape), mask, mask._version,
        plan.rows.long(), plan.inverse.long(), frozenset(), plan.lengths,
        frozenset(id(module) for module in modules))
    generator = torch.Generator().manual_seed(73189)
    q = torch.randn((2, 9, 4, 8), generator=generator).permute(0, 2, 1, 3).requires_grad_(True)
    k = torch.randn((2, 9, 2, 8), generator=generator).permute(0, 2, 1, 3).requires_grad_(True)
    v = torch.randn((2, 9, 2, 8), generator=generator).permute(0, 2, 1, 3).requires_grad_(True)
    position = torch.arange(9)
    allowed = ((position[:, None] >= position[None, :])[None, None]
               & mask.bool()[:, None, None, :])
    return call, q, k, v, allowed


@contextmanager
def _active(call):
    token = mlp_padding._ACTIVE_MLP_PADDING.set(call)
    try:
        yield call
    finally:
        patch.clear_varlen_attention_call(call)
        mlp_padding._ACTIVE_MLP_PADDING.reset(token)


def _cpu_dispatch_fixture(monkeypatch, modules):
    from transformers.integrations.sdpa_attention import repeat_kv

    native_calls, fast_calls, verifications = [], [], []
    sentinel = (object(), None)
    def original(module, q, k, v, mask, **kwargs):
        native_calls.append((module, q, k, v, mask, kwargs))
        return sentinel
    def fast(q, k, v, plan, **kwargs):
        fast_calls.append((q, k, v, plan, kwargs))
        # Explicit CPU plumbing sentinel, never the native GPU primitive.
        return q.half()
    original_verify = patch._verify_mask
    def verify(mask, call):
        verifications.append((mask, call))
        return original_verify(mask, call)
    monkeypatch.setattr(patch, "_supported_device", lambda q: q.device.type == "cpu")
    monkeypatch.setattr(primitive, "varlen_attention", fast)
    monkeypatch.setattr(patch, "_verify_mask", verify)
    wrapped = patch._make_dispatcher(original, type(modules[0]), repeat_kv)
    return wrapped, native_calls, fast_calls, verifications, sentinel


def _restore_registry(monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # setitem on the local mapping preserves whether this key existed before
    # the test; mutating the class-global registry is unnecessary.
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._local_mapping, "sdpa", ALL_ATTENTION_FUNCTIONS["sdpa"])
    return ALL_ATTENTION_FUNCTIONS


def test_varlen_registry_install_idempotence_and_restoration():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    registry = ALL_ATTENTION_FUNCTIONS
    original_local = dict(registry._local_mapping)
    original = registry["sdpa"]
    with pytest.MonkeyPatch.context() as monkeypatch:
        _restore_registry(monkeypatch)
        assert patch.patch_varlen_attention()
        installed = registry["sdpa"]
        assert getattr(installed, "_soup_candidate_varlen_attention", False)
        assert patch.patch_varlen_attention() and registry["sdpa"] is installed
        assert registry.get_interface("sdpa", None) is installed
    assert registry["sdpa"] is original
    assert registry._local_mapping == original_local


@pytest.mark.parametrize("reason", ["other_registration", "fake_class", "wrong_version"])
def test_varlen_registry_refuses_unproved_installations(monkeypatch, reason):
    from transformers.models.qwen2 import modeling_qwen2

    registry = _restore_registry(monkeypatch)
    if reason == "other_registration":
        monkeypatch.setitem(registry._local_mapping, "sdpa", lambda *args, **kwargs: None)
    elif reason == "fake_class":
        monkeypatch.setattr(modeling_qwen2, "Qwen2Attention", object)
    else:
        version = patch.metadata.version
        monkeypatch.setattr(patch.metadata, "version", lambda name: "3.5.0" if name == "triton" else version(name))
    before = registry["sdpa"]
    assert not patch.patch_varlen_attention()
    assert registry["sdpa"] is before


def test_varlen_registry_cpu_production_path_stays_native(monkeypatch):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    modules = _modules()
    call, q, k, v, mask = _inputs(modules)
    registry = _restore_registry(monkeypatch)
    assert patch.patch_varlen_attention()
    def forbidden(*args, **kwargs):
        raise AssertionError("production primitive should not run on CPU")
    monkeypatch.setattr(primitive, "varlen_attention", forbidden)
    with _active(call):
        actual, weights = registry["sdpa"](modules[0], q, k, v, mask, dropout=0.0, scaling=8**-0.5)
    expected, _ = sdpa_attention_forward(modules[0], q, k, v, mask, dropout=0.0, scaling=8**-0.5)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert weights is None


def test_varlen_dispatch_reuses_one_plan_verification_and_native_fp32_repeat(monkeypatch):
    from transformers.integrations.sdpa_attention import repeat_kv

    modules = _modules()
    call, q, k, v, mask = _inputs(modules)
    dispatch, native_calls, fast_calls, verifications, _ = _cpu_dispatch_fixture(monkeypatch, modules)
    with _active(call):
        first, _ = dispatch(modules[0], q, k, v, mask, dropout=0.0, scaling=8**-0.5)
        second, _ = dispatch(modules[1], q, k, v, mask, dropout=0.0, scaling=8**-0.5)
        state = getattr(call, patch._STATE_ATTRIBUTE)
        assert len(verifications) == 1 and len(fast_calls) == 2 and native_calls == []
        assert fast_calls[0][3] is fast_calls[1][3] is state.plan
        assert state.plan.rows is call.rows and state.plan.inverse is call.inverse
        assert state.plan.cumulative.tolist() == [0, 9, 14]
        for _, expanded_k, expanded_v, _, flags in fast_calls:
            assert expanded_k.dtype == expanded_v.dtype == torch.float32
            torch.testing.assert_close(expanded_k, repeat_kv(k, 2), atol=0, rtol=0)
            torch.testing.assert_close(expanded_v, repeat_kv(v, 2), atol=0, rtol=0)
            assert flags == {"scale": 8**-0.5, "private_training": True, "finite_training": True}
        assert first.shape == second.shape == (2, 9, 4, 8)
        assert first.is_contiguous() and first.dtype == torch.float16
    assert state.mask is state.plan is None and state.blocked
    assert not hasattr(call, patch._STATE_ATTRIBUTE)
    assert mlp_padding._ACTIVE_MLP_PADDING.get() is None


@pytest.mark.parametrize("change", ["identity", "mask_version", "row_map_version"])
def test_varlen_dispatch_invalidates_cached_identity_and_versions(monkeypatch, change):
    modules = _modules()
    call, q, k, v, mask = _inputs(modules)
    dispatch, native_calls, fast_calls, verifications, sentinel = _cpu_dispatch_fixture(monkeypatch, modules)
    with _active(call):
        dispatch(modules[0], q, k, v, mask)
        changed = mask
        if change == "identity":
            changed = mask.clone()
        elif change == "mask_version":
            mask[0, 0, 0, 0] = False
        else:
            call.rows[0] = 1
        assert dispatch(modules[1], q, k, v, changed) is sentinel
        assert dispatch(modules[0], q, k, v, mask) is sentinel
        assert len(fast_calls) == 1 and len(native_calls) == 2 and len(verifications) == 1


def test_varlen_dispatch_rejects_wrong_causal_mask_once_per_context(monkeypatch):
    modules = _modules()
    call, q, k, v, mask = _inputs(modules)
    dispatch, native_calls, fast_calls, verifications, sentinel = _cpu_dispatch_fixture(monkeypatch, modules)
    wrong = mask.clone()
    wrong[0, 0, 0, 1] = True
    with _active(call):
        assert dispatch(modules[0], q, k, v, wrong) is sentinel
        assert dispatch(modules[1], q, k, v, wrong) is sentinel
        assert fast_calls == [] and len(native_calls) == 2 and len(verifications) == 1
    with _active(call):
        dispatch(modules[0], q, k, v, mask)
        assert len(fast_calls) == 1 and len(verifications) == 2


@pytest.mark.parametrize("reason", [
    "no_context", "foreign_id", "evaluation", "no_grad", "dropout", "configured_dropout",
    "half_q", "float_mask", "mask_shape", "mask_changed", "missing_lengths",
    "position_bias", "sliding_window", "past", "use_cache", "noncausal", "hook",
])
def test_varlen_dispatch_unsupported_calls_delegate_once_without_copies(monkeypatch, reason):
    modules = _modules()
    call, q, k, v, mask = _inputs(modules)
    dispatch, native_calls, fast_calls, verifications, sentinel = _cpu_dispatch_fixture(monkeypatch, modules)
    kwargs = {}
    active = call
    handle = None
    if reason == "no_context": active = None
    elif reason == "foreign_id": call.attention_ids = frozenset()
    elif reason == "evaluation": modules[0].eval()
    elif reason == "no_grad": q, k, v = (tensor.detach() for tensor in (q, k, v))
    elif reason == "dropout": kwargs["dropout"] = 0.1
    elif reason == "configured_dropout": modules[0].attention_dropout = 0.1
    elif reason == "half_q": q = q.half()
    elif reason == "float_mask": mask = mask.float()
    elif reason == "mask_shape": mask = mask[:, 0]
    elif reason == "mask_changed": call.mask[0, -1] = 0
    elif reason == "missing_lengths": call.lengths = ()
    elif reason == "position_bias": kwargs["position_bias"] = torch.zeros_like(mask, dtype=torch.float32)
    elif reason == "sliding_window": kwargs["sliding_window"] = 7
    elif reason == "past": kwargs["past_key_values"] = object()
    elif reason == "use_cache": kwargs["use_cache"] = True
    elif reason == "noncausal": kwargs["is_causal"] = False
    elif reason == "hook": handle = modules[0].register_forward_hook(lambda *args: None)
    try:
        with _active(active):
            assert dispatch(modules[0], q, k, v, mask, **kwargs) is sentinel
    finally:
        if handle is not None: handle.remove()
    assert len(native_calls) == 1 and not fast_calls and not verifications
    passed = native_calls[0]
    assert all(actual is expected for actual, expected in zip(passed[:5], (modules[0], q, k, v, mask)))
    assert all(passed[5][name] is value for name, value in kwargs.items())


def test_varlen_dispatch_does_not_retry_native_after_a_fast_path_error(monkeypatch):
    modules = _modules()
    call, q, k, v, mask = _inputs(modules)
    dispatch, native_calls, _, _, _ = _cpu_dispatch_fixture(monkeypatch, modules)
    def failure(*args, **kwargs):
        raise RuntimeError("intentional native-varlen failure")
    monkeypatch.setattr(primitive, "varlen_attention", failure)
    with pytest.raises(RuntimeError, match="intentional"):
        with _active(call):
            dispatch(modules[0], q, k, v, mask)
    assert native_calls == []
    assert mlp_padding._ACTIVE_MLP_PADDING.get() is None
    assert not hasattr(call, patch._STATE_ATTRIBUTE)


@PHYSICAL
@pytest.mark.parametrize("width,intermediate,layers,length", [(64, 128, 2, 13), (1536, 8960, 1, 128)])
def test_physical_t4_varlen_real_qwen_registry_composition(
        monkeypatch, record_property, width, intermediate, layers, length):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from . import apply
    from .test_fusion_cuda import _register_patch_restoration
    from .test_mlp_padding import (_restore, _model, _batch, _private_training,
                                   _gradient_pairs, _fp16_error_screen)

    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("native varlen composition requires physical SM75 T4")
    assert torch.cuda.device_count() == 1
    _register_patch_restoration(monkeypatch)
    _restore(monkeypatch)
    registry = _restore_registry(monkeypatch)
    apply()
    assert mlp_padding.patch_mlp_padding()
    model, _ = _model("cuda", width=width, intermediate=intermediate, layers=layers)
    batch = _batch("cuda", length=length)
    # Keep P1 and the entire selected body identical. Only the optional SDPA
    # registry dispatch differs between these two real private training calls.
    with pytest.MonkeyPatch.context() as native_registry:
        native_registry.setitem(registry._local_mapping, "sdpa", sdpa_attention_forward)
        expected = _private_training(model, batch, compact=True, device="cuda", scale=65536)
    assert patch.patch_varlen_attention()
    counts = {"forward": 0, "backward": 0, "mask_verifications": 0, "plans": 0}
    for owner, name, label in ((primitive, "_native_forward", "forward"),
                               (primitive, "_native_backward", "backward"),
                               (patch, "_verify_mask", "mask_verifications"),
                               (primitive, "plan_from_row_tensors", "plans")):
        original = getattr(owner, name)
        def observed(*args, _original=original, _label=label, **kwargs):
            result = _original(*args, **kwargs)
            counts[_label] += 1
            return result
        monkeypatch.setattr(owner, name, observed)
    actual = _private_training(model, batch, compact=True, device="cuda", scale=65536)
    pairs = _gradient_pairs(actual, expected)
    evidence = {"valid_logits": _fp16_error_screen(actual["valid_logits"], expected["valid_logits"], rows=True),
                "gradients": [dict(meta, **_fp16_error_screen(a/65536, b/65536)) for meta, a, b in pairs]}
    record_property("soup_varlen_qwen_fp16_evidence", json.dumps(evidence, allow_nan=False))
    record_property("soup_varlen_qwen_native_counts", json.dumps(counts))
    assert evidence["valid_logits"]["screen_passed"], evidence["valid_logits"]
    assert all(row["screen_passed"] for row in evidence["gradients"]), evidence["gradients"]
    assert counts == {"forward": layers, "backward": layers, "mask_verifications": 1, "plans": 1}
    for name in ("loss", "row_loss", "entropy"):
        torch.testing.assert_close(actual[name], expected[name], atol=3e-5, rtol=3e-3)
    for name in ("row_tokens", "correct", "tokens", "rng", "labels", "attention_mask"):
        assert torch.equal(actual[name], expected[name]), name
    assert mlp_padding._ACTIVE_MLP_PADDING.get() is None
    assert actual["call"] is not None
    assert not hasattr(actual["call"], patch._STATE_ATTRIBUTE), "context owner must release forward cache"
