"""FP32 rotary arithmetic, autograd, and attention-boundary checks.

CPU math: TRITON_INTERPRET=1 python -m pytest solution/test_rotary.py -q
CUDA:    python -m pytest solution/test_rotary.py -q
Interpreter checks are not evidence of GPU execution or performance.
"""

import itertools
import os
from types import SimpleNamespace

import pytest
import torch


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CAN_EXECUTE = torch.cuda.is_available() or INTERPRETING
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _reference(q, k, cos, sin):
    # The pinned eager source is the oracle, not a second handwritten kernel.
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    while getattr(apply_rotary_pos_emb, "_soup_kernel_rotary", False):
        apply_rotary_pos_emb = apply_rotary_pos_emb.__wrapped__
    return apply_rotary_pos_emb(q, k, cos, sin)


def _inputs(batch=2, length=7, q_heads=12, k_heads=2, width=128, coefficient_batch=1, device=DEVICE):
    torch.manual_seed(4713)

    def activation(heads):
        # FP32 sums of two half results model the task's base/LoRA addition;
        # these sums need not themselves be representable in FP16.
        shape = (batch, length, heads, width)
        base = torch.randn(shape, dtype=torch.float32, device=device).half().float()
        adapter = (torch.randn(shape, dtype=torch.float32, device=device) * 0.1).half().float()
        return (base + adapter).transpose(1, 2).requires_grad_()

    q, k = activation(q_heads), activation(k_heads)
    # Deliberately distinct coefficient halves catch hidden equality assumptions.
    cos = torch.randn((coefficient_batch, length, width), device=device)
    sin = torch.randn_like(cos)
    return q, k, cos, sin


def _assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("scale", [1.0, 65536.0])
@pytest.mark.parametrize("shape", [
    (2, 7, 12, 2, 128, 1),
    (2, 3, 3, 1, 10, 2),
    (1, 1, 12, 2, 128, 1),
    (1, 9, 1, 1, 2, 1),
    (2, 5, 12, 2, 128, 2),
])
def test_rotary_exact_forward_scaled_backward_and_tails(shape, scale):
    from solution.rotary import fused_apply_rotary_pos_emb

    q, k, cos, sin = _inputs(*shape)
    before = [x.detach().clone() for x in (q, k, cos, sin)]
    versions = [x._version for x in (q, k, cos, sin)]
    expected = _reference(q, k, cos, sin)
    actual = fused_apply_rotary_pos_emb(q, k, cos, sin)
    grads = tuple(torch.randn_like(y) * scale for y in expected)
    expected_grads = torch.autograd.grad(expected, (q, k), grads)
    actual_grads = torch.autograd.grad(actual, (q, k), grads)
    for output, oracle, input_tensor in zip(actual, expected, (q, k)):
        assert output.dtype == torch.float32
        assert output.stride() == input_tensor.stride()
        # TensorIterator may choose a different, immaterial singleton stride.
        assert all(a == b for size, a, b in zip(output.shape, output.stride(), oracle.stride()) if size > 1)
        assert output.data_ptr() not in (q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr())
        _assert_exact(output, oracle)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        _assert_exact(actual_grad, expected_grad)
    for tensor, snapshot, version in zip((q, k, cos, sin), before, versions):
        _assert_exact(tensor, snapshot)
        assert tensor._version == version


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("layout", ["bhtd", "sliced", "expanded", "negative_view"])
def test_rotary_arbitrary_upstream_gradient_strides(layout):
    from solution.rotary import fused_apply_rotary_pos_emb

    q, k, cos, sin = _inputs(batch=2, length=5, coefficient_batch=2)

    def gradient(x):
        batch, heads, length, width = x.shape
        if layout == "expanded":
            return torch.randn((batch, heads, 1, width), device=DEVICE).expand_as(x)
        if layout == "bhtd":
            return torch.randn(x.shape, device=DEVICE)
        result = torch.randn((batch, heads, length, width * 2), device=DEVICE)[..., ::2]
        return torch._neg_view(result) if layout == "negative_view" else result

    grads = tuple(gradient(x) for x in (q, k))
    expected = torch.autograd.grad(_reference(q, k, cos, sin), (q, k), grads)
    actual = torch.autograd.grad(fused_apply_rotary_pos_emb(q, k, cos, sin), (q, k), grads)
    for a, e in zip(actual, expected):
        _assert_exact(a, e)


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("requires_q,requires_k,used_output", [
    (True, True, 0), (True, True, 1), (True, False, 0), (False, True, 1),
])
def test_rotary_unused_and_frozen_branches(requires_q, requires_k, used_output):
    from solution.rotary import fused_apply_rotary_pos_emb

    q, k, cos, sin = _inputs(batch=1, length=3)
    q.requires_grad_(requires_q)
    k.requires_grad_(requires_k)
    expected = _reference(q, k, cos, sin)
    actual = fused_apply_rotary_pos_emb(q, k, cos, sin)
    assert actual[0].requires_grad == requires_q
    assert actual[1].requires_grad == requires_k
    variables = tuple(x for x in (q, k) if x.requires_grad)
    expected_grad = torch.autograd.grad(expected[used_output].sum(), variables, allow_unused=True)
    actual_grad = torch.autograd.grad(actual[used_output].sum(), variables, allow_unused=True)
    for a, e in zip(actual_grad, expected_grad):
        if e is None:
            assert a is None
        else:
            _assert_exact(a, e)


def test_rotary_shared_qk_storage_is_rejected_to_preserve_accumulation_order():
    from solution.rotary import _supported, fused_apply_rotary_pos_emb

    q, _, cos, sin = _inputs(batch=1, length=3, q_heads=2, k_heads=2, device="cpu")
    assert not _supported(q, q, cos, sin)
    assert not _supported(q, q.view_as(q), cos, sin)
    with pytest.raises(ValueError):
        fused_apply_rotary_pos_emb(q, q, cos, sin)


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
def test_rotary_signed_zero_backward_matches_slice_padding_arithmetic():
    from solution.rotary import fused_apply_rotary_pos_emb

    values = torch.tensor(list(itertools.product([0.0, -0.0, 1.0, -1.0], repeat=6)), device=DEVICE)
    count = values.shape[0]
    cos = values[:, :2].reshape(1, count, 2).contiguous()
    sin = values[:, 2:4].reshape(1, count, 2).contiguous()
    grad = values[:, 4:].reshape(1, 1, count, 2)
    q = torch.zeros((1, count, 1, 2), device=DEVICE).transpose(1, 2).requires_grad_()
    k = q.detach().clone(memory_format=torch.preserve_format).requires_grad_()
    expected = _reference(q, k, cos, sin)
    actual = fused_apply_rotary_pos_emb(q, k, cos, sin)
    e = torch.autograd.grad(expected, (q, k), (grad, grad))
    a = torch.autograd.grad(actual, (q, k), (grad, grad))
    for output, oracle in zip(actual, expected):
        _assert_exact(output, oracle)
    for output, oracle in zip(a, e):
        _assert_exact(output, oracle)


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
def test_rotary_nonfinite_values_and_zero_coefficients():
    from solution.rotary import fused_apply_rotary_pos_emb

    values = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan")], device=DEVICE)
    q = values.repeat(4).reshape(1, 7, 1, 4).transpose(1, 2).requires_grad_()
    k = q.detach().flip(-1).requires_grad_()
    cos = torch.tensor([0.0, -1.0, 1.0, 0.0], device=DEVICE).repeat(1, 7, 1)
    sin = torch.tensor([1.0, 0.0, -0.0, -1.0], device=DEVICE).repeat(1, 7, 1)
    expected = _reference(q, k, cos, sin)
    actual = fused_apply_rotary_pos_emb(q, k, cos, sin)
    grads = (q.detach(), k.detach())
    e = torch.autograd.grad(expected, (q, k), grads)
    a = torch.autograd.grad(actual, (q, k), grads)
    for output, oracle in zip((*actual, *a), (*expected, *e)):
        _assert_exact(output, oracle)


def test_cast_before_gqa_replication_is_a_real_gradient_overflow_counterexample():
    cos = sin = torch.full((1, 1, 2), 2**-0.5)
    grad = torch.zeros((1, 6, 1, 2), dtype=torch.float16)
    grad[:, :2, :, 0] = 40000
    grad[:, :2, :, 1] = -40000
    gradients = []
    for early_cast in (False, True):
        key = torch.zeros((1, 1, 1, 2), requires_grad=True)
        rotated = _reference(key, key, cos, sin)[1]
        repeated = rotated.half().repeat_interleave(6, 1) if early_cast else rotated.repeat_interleave(6, 1).half()
        gradients.append(torch.autograd.grad(repeated, key, grad)[0])
    assert torch.isfinite(gradients[0]).all()
    assert gradients[0].flatten()[0] == 0
    assert torch.isnan(gradients[1].flatten()[0])
    assert torch.isneginf(gradients[1].flatten()[1])


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
def test_rotary_preserves_fp32_gqa_reduction_for_large_scaled_gradients():
    from solution.rotary import fused_apply_rotary_pos_emb

    key = torch.zeros((1, 1, 1, 2), device=DEVICE, requires_grad=True)
    query = torch.zeros_like(key)
    cos = sin = torch.full((1, 1, 2), 2**-0.5, device=DEVICE)
    grad = torch.zeros((1, 6, 1, 2), dtype=torch.float16, device=DEVICE)
    grad[:, :2, :, 0] = 40000
    grad[:, :2, :, 1] = -40000
    expected = _reference(query, key, cos, sin)[1].repeat_interleave(6, 1).half()
    actual = fused_apply_rotary_pos_emb(query, key, cos, sin)[1].repeat_interleave(6, 1).half()
    e, = torch.autograd.grad(expected, key, grad)
    a, = torch.autograd.grad(actual, key, grad)
    assert torch.isfinite(a).all()
    _assert_exact(a, e)


def test_rotary_guards_reject_unsupported_metadata_without_device_work():
    from solution.rotary import _supported, fused_apply_rotary_pos_emb

    q, k, cos, sin = _inputs(device="cpu")
    assert _supported(q, k, cos, sin)
    assert not _supported(q, k, cos, sin, cuda_only=True)
    assert not _supported(q.contiguous(), k, cos, sin)
    assert not _supported(q.half(), k, cos, sin)
    assert not _supported(q, k, cos, sin, unsqueeze_dim=2)
    assert not _supported(q, k, cos[:, :-1], sin[:, :-1])
    assert not _supported(q, k, cos.expand(2, -1, -1), sin.expand(2, -1, -1))
    assert not _supported(torch._neg_view(q), k, cos, sin)
    cos.requires_grad_(True)
    assert not _supported(q, k, cos, sin)
    with pytest.raises(ValueError, match="frozen"):
        fused_apply_rotary_pos_emb(q, k, cos, sin)


def test_rotary_patch_is_idempotent_and_keeps_cpu_reference_and_rng(monkeypatch):
    from transformers.models.qwen2 import modeling_qwen2
    from solution.rotary import patch_qwen2_rotary

    original = modeling_qwen2.apply_rotary_pos_emb
    monkeypatch.setattr(modeling_qwen2, "apply_rotary_pos_emb", original)
    q, k, cos, sin = _inputs(device="cpu")
    expected = original(q, k, cos, sin)
    assert patch_qwen2_rotary()
    installed = modeling_qwen2.apply_rotary_pos_emb
    assert patch_qwen2_rotary() and modeling_qwen2.apply_rotary_pos_emb is installed
    assert modeling_qwen2.Qwen2Attention.forward.__globals__["apply_rotary_pos_emb"] is installed
    state = torch.get_rng_state().clone()
    actual = installed(q, k, cos, sin)
    assert torch.equal(state, torch.get_rng_state())
    for a, e in zip(actual, expected):
        _assert_exact(a, e)


@pytest.mark.skipif(not INTERPRETING, reason="test-only CPU interpreter boundary injection")
def test_rotary_patch_dispatch_and_trainable_coefficient_fallback(monkeypatch):
    from transformers.models.qwen2 import modeling_qwen2
    from solution import rotary

    original = modeling_qwen2.apply_rotary_pos_emb
    monkeypatch.setattr(modeling_qwen2, "apply_rotary_pos_emb", original)
    supported = rotary._supported
    monkeypatch.setattr(rotary, "_supported", lambda *a, cuda_only=False, **kw: supported(*a, **kw))
    assert rotary.patch_qwen2_rotary()
    q, k, cos, sin = _inputs(device="cpu")
    state = torch.get_rng_state().clone()
    result = modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)
    assert "FrozenRotary" in type(result[0].grad_fn).__name__
    assert torch.equal(state, torch.get_rng_state())
    cos.requires_grad_(True)
    fallback = modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)
    coefficient_grad, = torch.autograd.grad(sum(x.sum() for x in fallback), cos)
    assert torch.isfinite(coefficient_grad).all()
    assert "FrozenRotary" not in type(fallback[0].grad_fn).__name__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs physical CUDA execution")
@pytest.mark.parametrize("length,coefficient_batch", [(512, 1), (511, 4)])
def test_rotary_cuda_full_task_shape(length, coefficient_batch):
    from solution.rotary import fused_apply_rotary_pos_emb

    q, k, cos, sin = _inputs(batch=4, length=length, coefficient_batch=coefficient_batch, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        expected = _reference(q, k, cos, sin)
        actual = fused_apply_rotary_pos_emb(q, k, cos, sin)
    grads = tuple(torch.randn_like(x) * 65536 for x in expected)
    e = torch.autograd.grad(expected, (q, k), grads)
    a = torch.autograd.grad(actual, (q, k), grads)
    for output, oracle in zip((*actual, *a), (*expected, *e)):
        _assert_exact(output, oracle)


@pytest.mark.skipif(not INTERPRETING, reason="test-only real Qwen2 CPU interpreter integration")
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_rotary_patch_reaches_real_qwen2_attention_and_preserves_gradients(monkeypatch, scale):
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from solution import rotary

    torch.manual_seed(1701)
    config = Qwen2Config(
        hidden_size=96, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=12, num_key_value_heads=2,
        attention_dropout=0.0, max_position_embeddings=32,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
    )
    config._attn_implementation = "sdpa"
    attention = modeling_qwen2.Qwen2Attention(config, layer_idx=0).train()
    embedding = modeling_qwen2.Qwen2RotaryEmbedding(config)
    hidden = torch.randn((2, 5, 96), requires_grad=True)
    coefficients = embedding(hidden, torch.arange(5).unsqueeze(0))
    mask = torch.ones((2, 1, 5, 5), dtype=torch.bool).tril()
    mask[1, :, :, -1] = False
    grad = torch.randn_like(hidden) * scale
    variables = (hidden, *attention.parameters())
    rng = torch.get_rng_state().clone()
    expected, expected_attentions = attention(hidden, coefficients, mask)
    expected_grads = torch.autograd.grad(expected, variables, grad)
    assert torch.equal(rng, torch.get_rng_state())

    original = modeling_qwen2.apply_rotary_pos_emb
    monkeypatch.setattr(modeling_qwen2, "apply_rotary_pos_emb", original)
    supported = rotary._supported
    # Keep every shape, dtype, coefficient, and alias guard; override only the
    # CUDA selection boundary for this explicitly labelled interpreter test.
    monkeypatch.setattr(rotary, "_supported", lambda *a, cuda_only=False, **kw: supported(*a, **kw))
    calls = []
    fused = rotary.fused_apply_rotary_pos_emb

    def counted(q, k, cos, sin, unsqueeze_dim=1):
        assert q.dtype == k.dtype == torch.float32
        assert q.untyped_storage().data_ptr() != k.untyped_storage().data_ptr()
        calls.append((q.shape, k.shape))
        return fused(q, k, cos, sin, unsqueeze_dim)

    monkeypatch.setattr(rotary, "fused_apply_rotary_pos_emb", counted)
    assert rotary.patch_qwen2_rotary()
    actual, actual_attentions = attention(hidden, coefficients, mask)
    actual_grads = torch.autograd.grad(actual, variables, grad)
    assert calls == [(torch.Size((2, 12, 5, 8)), torch.Size((2, 2, 5, 8)))]
    assert actual_attentions is expected_attentions is None
    assert torch.equal(rng, torch.get_rng_state())
    _assert_exact(actual, expected)
    # Q/K/V share hidden ancestry: changing autograd node scheduling may change
    # FP32 upstream accumulation order.  This is a tight tolerance check of the
    # real CPU attention graph, not a claim of whole-model bitwise parity.
    for output, oracle in zip(actual_grads, expected_grads):
        torch.testing.assert_close(output / scale, oracle / scale, rtol=2e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs physical CUDA SDPA autocast")
@pytest.mark.parametrize("masked", [False, True])
def test_rotary_cuda_sdpa_cast_and_grouped_attention_boundary(masked):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from solution.rotary import fused_apply_rotary_pos_emb

    q, k, cos, sin = _inputs(batch=2, length=17, device="cuda")
    value = torch.randn((2, 17, 2, 128), device="cuda").transpose(1, 2).requires_grad_()
    module = SimpleNamespace(num_key_value_groups=6, is_causal=True)
    mask = torch.ones((2, 1, 17, 17), device="cuda", dtype=torch.bool).tril() if masked else None
    state = torch.cuda.get_rng_state().clone()

    def attention(rotary):
        with torch.autocast("cuda", dtype=torch.float16):
            qr, kr = rotary(q, k, cos, sin)
            return sdpa_attention_forward(module, qr, kr, value, mask, dropout=0.0, scaling=128**-0.5)[0]

    expected, actual = attention(_reference), attention(fused_apply_rotary_pos_emb)
    assert torch.equal(state, torch.cuda.get_rng_state())
    assert actual.dtype == expected.dtype == torch.float16
    _assert_exact(actual, expected)
    grad = torch.randn_like(expected) * 64
    e = torch.autograd.grad(expected, (q, k, value), grad)
    a = torch.autograd.grad(actual, (q, k, value), grad)
    # SDPA may select a parallel backend with non-deterministic accumulation;
    # the stand-alone tests above require exact rotary arithmetic itself.
    for output, oracle in zip(a, e):
        torch.testing.assert_close(output, oracle, rtol=2e-5, atol=2e-5, equal_nan=True)
