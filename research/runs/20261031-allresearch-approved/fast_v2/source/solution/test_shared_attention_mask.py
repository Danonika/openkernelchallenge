"""Real Qwen2/PEFT mask sharing with explicit interpreter-only overrides."""
import copy
import os

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2Attention
from peft import LoraConfig, get_peft_model

from . import shared_attention_mask as shared


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CPU_KERNEL = pytest.mark.skipif(not INTERPRETING, reason="requires TRITON_INTERPRET=1")
CUDA_KERNEL = pytest.mark.skipif(
    not torch.cuda.is_available() or INTERPRETING,
    reason="requires a physical CUDA device, not the interpreter")


def _model(device="cpu"):
    torch.manual_seed(173)
    cfg = Qwen2Config(vocab_size=37, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, max_position_embeddings=64, tie_word_embeddings=True,
                     attention_dropout=0.0, pad_token_id=0, eos_token_id=1)
    cfg._attn_implementation = "sdpa"
    model = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.02)
    return model.to(device).train()


def _batch(pattern="right", device="cpu"):
    ids = torch.tensor([[4, 7, 5, 8, 10, 11, 7, 6, 9, 12, 8, 7, 5],
                        [6, 8, 7, 9, 12, 14, 8, 7, 9, 13, 5, 6, 8]], device=device)
    mask = torch.ones_like(ids)
    if pattern == "right":
        mask[1, -3:] = 0
    elif pattern == "left":
        mask[1, :3] = 0
    elif pattern == "empty":
        mask[1] = 0
    elif pattern == "none":
        mask = None
    labels = ids.clone()
    labels[:, :2] = -100
    if mask is not None:
        labels[mask == 0] = -100
    return dict(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)


def _run(model, inputs, device="cpu"):
    torch.manual_seed(1739)
    if device == "cuda":
        torch.cuda.manual_seed_all(1739)
    with torch.autocast(device, dtype=torch.float16):
        out = model(**inputs)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    grads = torch.autograd.grad(out.loss, params)
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.random.get_rng_state()
    return out.loss.detach(), out.logits.detach(), grads, rng


def _patch(monkeypatch, cpu=False):
    monkeypatch.setattr(Qwen2Model, "forward", Qwen2Model.forward)
    monkeypatch.setattr(Qwen2Attention, "forward", Qwen2Attention.forward)
    if cpu:
        monkeypatch.setattr(shared, "_autocast_fp16", lambda:
            torch.is_autocast_enabled("cpu") and torch.get_autocast_dtype("cpu") == torch.float16)
        monkeypatch.setattr(shared, "_supported_device", lambda hidden: hidden.device.type == "cpu")
    original_convert = shared.convert_mask
    conversions = []
    def convert(mask, **kwargs):
        result = original_convert(mask, **kwargs)
        conversions.append((tuple(mask.shape), tuple(result[0].shape), result[0].stride()))
        return result
    monkeypatch.setattr(shared, "convert_mask", convert)
    assert shared.patch_shared_attention_mask()
    assert shared.patch_shared_attention_mask()
    return conversions


@CPU_KERNEL
@pytest.mark.parametrize("pattern", ["right", "left", "empty", "full", "none"])
def test_real_qwen_peft_loss_logits_gradients_rng_and_one_conversion(monkeypatch, pattern):
    base = _model()
    candidate = copy.deepcopy(base)
    inputs = _batch(pattern)
    expected = _run(base, inputs)
    conversions = _patch(monkeypatch, cpu=True)
    actual = _run(candidate, inputs)
    for got, want in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    for got, want in zip(actual[2], expected[2]):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert torch.equal(actual[3], expected[3])
    assert len(conversions) == (1 if pattern in {"right", "left", "empty"} else 0)
    assert shared._ACTIVE_MASK.get() is None
    assert all("forward" not in layer.self_attn.__dict__ for layer in candidate.get_base_model().model.layers)


@CPU_KERNEL
def test_noncontiguous_boolean_input_exact_values_and_unwritten_padding():
    source = torch.arange(2 * 13 * 26).reshape(2, 1, 13, 26) % 3 == 0
    mask = source[..., ::2]
    converted, owner = shared.convert_mask(mask, sentinel=17)
    expected = torch.where(mask, torch.tensor(0.0, dtype=torch.float16),
                           torch.tensor(-float("inf"), dtype=torch.float16))
    torch.testing.assert_close(converted, expected, atol=0, rtol=0)
    assert torch.equal(torch.signbit(converted), torch.signbit(expected))
    assert converted.shape == mask.shape
    assert converted.stride(-1) == 1 and converted.stride(-2) == 16
    assert torch.all(owner[..., 13:] == 17)


@CPU_KERNEL
def test_cache_identity_version_and_float_passthrough(monkeypatch):
    model = _model().get_base_model().model
    module = model.layers[0].self_attn
    monkeypatch.setattr(shared, "_supported_device", lambda hidden: True)
    call = shared._MaskCall(model, {id(module)})
    hidden = torch.randn(2, 9, 32)
    mask = torch.ones((2, 1, 9, 9), dtype=torch.bool)
    kwargs = dict(hidden_states=hidden, attention_mask=mask)
    one = shared._attention_mask(module, (), kwargs, call)
    two = shared._attention_mask(module, (), kwargs, call)
    assert one is two and len(call.cache) == 1
    mask[..., 0, 0] = False
    three = shared._attention_mask(module, (), kwargs, call)
    assert three is not one and len(call.cache) == 2
    assert torch.isneginf(three[..., 0, 0]).all()
    assert shared._attention_mask(module, (), dict(kwargs, attention_mask=one), call) is None
    assert shared._attention_mask(module, (), dict(kwargs, attention_mask=None), call) is None


def test_production_patch_installs_before_model_and_rejects_cpu(monkeypatch):
    conversions = _patch(monkeypatch, cpu=False)
    model = _model()
    _run(model, _batch())
    assert conversions == []
    assert shared._ACTIVE_MASK.get() is None


@CPU_KERNEL
def test_context_is_cleared_when_attention_raises(monkeypatch):
    model = _model()
    original = Qwen2Attention.forward
    def fail(*args, **kwargs):
        raise RuntimeError("intentional attention fixture failure")
    monkeypatch.setattr(Qwen2Attention, "forward", fail)
    _patch(monkeypatch, cpu=True)
    with pytest.raises(RuntimeError, match="intentional attention"):
        _run(model, _batch())
    assert shared._ACTIVE_MASK.get() is None


@CUDA_KERNEL
def test_physical_t4_qwen_attention_composition(monkeypatch):
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("production candidate is guarded to T4 SM75")
    base = _model("cuda")
    candidate = copy.deepcopy(base)
    inputs = _batch(device="cuda")
    expected = _run(base, inputs, device="cuda")
    conversions = _patch(monkeypatch, cpu=False)
    actual = _run(candidate, inputs, device="cuda")
    for got, want in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    for got, want in zip(actual[2], expected[2]):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert torch.equal(actual[3], expected[3])
    assert len(conversions) == 1
