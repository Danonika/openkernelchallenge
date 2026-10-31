"""Lossless storage invariants and real PEFT/AMP compatibility checks.

CPU tests override only the installer device capability predicate. They do
not assert physical CUDA kernel correctness or measured memory improvement.
"""

import copy
import json
import os

import pytest
import torch
import torch.nn.functional as F

from . import frozen_storage as storage
from .integration import _cached_weight_fp16


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(580)
    config = Qwen2Config(
        vocab_size=67, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, pad_token_id=0, tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config).half()
    # Quantized linears are not simulated: this exercises the real preparation
    # and PEFT contracts for the non-quantized tied embedding/head only.
    model.is_loaded_in_4bit = True
    return model


@pytest.fixture
def preparation(monkeypatch):
    import peft

    original = peft.prepare_model_for_kbit_training
    while getattr(original, "_soup_candidate_frozen_storage", False):
        original = original.__wrapped__
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", original)
    assert storage.patch_frozen_storage()
    monkeypatch.setattr(storage, "_supported_device", lambda weight: True)
    return original, peft.prepare_model_for_kbit_training


def _prepared_pair(preparation):
    original_prepare, packed_prepare = preparation
    baseline = _tiny_model()
    packed = copy.deepcopy(baseline)
    original_prepare(baseline, use_gradient_checkpointing=False)
    packed_prepare(packed, use_gradient_checkpointing=False)
    assert packed._candidate_frozen_storage is True
    return baseline, packed


def test_real_prepare_preserves_values_tie_parameter_names_and_half_cache_alias(preparation):
    baseline, packed = _prepared_pair(preparation)
    weight = packed.get_input_embeddings().weight
    assert weight is packed.get_output_embeddings().weight
    assert weight.dtype == torch.float16 and not weight.requires_grad
    assert baseline.get_input_embeddings().weight.dtype == torch.float32
    before, after = baseline.state_dict(), packed.state_dict()
    assert set(before) == set(after)
    for key in before:
        torch.testing.assert_close(before[key], after[key].to(before[key].dtype), rtol=0, atol=0)
    assert before["model.embed_tokens.weight"].shape == after["model.embed_tokens.weight"].shape
    assert after["model.embed_tokens.weight"].data_ptr() == after["lm_head.weight"].data_ptr()
    cached = _cached_weight_fp16(packed)
    assert cached.data_ptr() == weight.data_ptr()
    assert cached.untyped_storage().nbytes() == weight.untyped_storage().nbytes()
    assert len(list(baseline.parameters())) == len(list(packed.parameters()))
    baseline_buffers, packed_buffers = dict(baseline.named_buffers()), dict(packed.named_buffers())
    assert baseline_buffers.keys() == packed_buffers.keys()
    for name in baseline_buffers:
        torch.testing.assert_close(packed_buffers[name], baseline_buffers[name], rtol=0, atol=0)


@pytest.mark.parametrize("autocast_dtype", [None, torch.float16, torch.bfloat16])
def test_embedding_and_output_head_match_reference_in_all_precision_contexts(preparation, autocast_dtype):
    baseline, packed = _prepared_pair(preparation)
    indices = torch.tensor([[1, 4, 9, 2, 3], [1, 5, 8, 0, 0]])
    actual_embedding = packed.get_input_embeddings()(indices)
    expected_embedding = baseline.get_input_embeddings()(indices)
    assert actual_embedding.dtype == expected_embedding.dtype == torch.float32
    torch.testing.assert_close(actual_embedding, expected_embedding, rtol=0, atol=0)

    torch.manual_seed(91)
    expected_input = torch.randn(2, 5, 16, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=autocast_dtype or torch.float16, enabled=autocast_dtype is not None):
        expected = baseline.lm_head(expected_input)
        actual = packed.lm_head(actual_input)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)


def test_normal_module_hooks_and_input_require_grads_are_preserved(preparation):
    baseline, packed = _prepared_pair(preparation)
    calls = []
    packed.enable_input_require_grads()
    handle = packed.get_input_embeddings().register_forward_hook(
        lambda module, inputs, outputs: calls.append((inputs[0].clone(), outputs.dtype, outputs.requires_grad))
    )
    ids = torch.tensor([[1, 4, 9]])
    output = packed.get_input_embeddings()(ids)
    handle.remove()
    assert len(calls) == 1
    assert torch.equal(calls[0][0], ids)
    assert calls[0][1:] == (torch.float32, True)
    output.sum().backward()
    assert packed.get_input_embeddings().weight.grad is None
    torch.testing.assert_close(output, baseline.get_input_embeddings()(ids), rtol=0, atol=0)


def test_deepcopy_binds_packed_forwards_to_the_new_modules(preparation):
    _, packed = _prepared_pair(preparation)
    other = copy.deepcopy(packed)
    assert other.model.embed_tokens.weight is other.lm_head.weight
    assert other.lm_head.weight is not packed.lm_head.weight
    with torch.no_grad():
        other.model.embed_tokens.weight[1].add_(1)
    ids = torch.tensor([[1, 3]])
    actual = other.get_input_embeddings()(ids)
    expected = F.embedding(ids, other.model.embed_tokens.weight.float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, packed.get_input_embeddings()(ids))
    hidden = torch.ones(1, 2, 16)
    torch.testing.assert_close(other.lm_head(hidden), F.linear(hidden, other.lm_head.weight.float()), rtol=0, atol=0)


def _attach_lora(model):
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(581)
    result = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
    ))
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.03)
    return result.train()


def test_real_peft_dropout_gradients_and_saved_adapter_are_unchanged(preparation, tmp_path):
    from safetensors.torch import load_file

    baseline, packed = (_attach_lora(model) for model in _prepared_pair(preparation))
    baseline_trainable = {n: p for n, p in baseline.named_parameters() if p.requires_grad}
    packed_trainable = {n: p for n, p in packed.named_parameters() if p.requires_grad}
    assert baseline_trainable.keys() == packed_trainable.keys()
    for name in baseline_trainable:
        torch.testing.assert_close(packed_trainable[name], baseline_trainable[name], rtol=0, atol=0)
    inputs = {
        "input_ids": torch.tensor([[1, 4, 9, 2, 3], [1, 5, 8, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, -100, 9, 2, 3], [-100, 5, 8, -100, -100]]),
        "use_cache": False,
    }
    torch.manual_seed(591)
    with torch.autocast("cpu", dtype=torch.float16):
        expected = baseline(**inputs)
    expected.loss.backward()
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(591)
    with torch.autocast("cpu", dtype=torch.float16):
        actual = packed(**inputs)
    actual.loss.backward()
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    for name in baseline_trainable:
        torch.testing.assert_close(packed_trainable[name].grad, baseline_trainable[name].grad, rtol=0, atol=0)
    baseline.save_pretrained(tmp_path / "baseline")
    packed.save_pretrained(tmp_path / "packed")
    expected_state = load_file(tmp_path / "baseline" / "adapter_model.safetensors")
    actual_state = load_file(tmp_path / "packed" / "adapter_model.safetensors")
    assert expected_state.keys() == actual_state.keys()
    for key in expected_state:
        torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
    expected_config = json.loads((tmp_path / "baseline" / "adapter_config.json").read_text())
    actual_config = json.loads((tmp_path / "packed" / "adapter_config.json").read_text())
    assert actual_config == expected_config


def test_capture_refuses_arbitrary_fp32_untied_and_checkpointed_models(preparation):
    for mode in ("fp32", "untied", "checkpointed"):
        model = _tiny_model()
        if mode == "fp32":
            model.float()
        elif mode == "untied":
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        assert storage._capture_candidate(model, mode == "checkpointed") is None


def test_capture_refuses_cpu_without_test_override(monkeypatch):
    model = _tiny_model()
    monkeypatch.setattr(storage, "_supported_device", lambda weight: weight.is_cuda)
    assert storage._capture_candidate(model, False) is None


def test_changed_values_and_trainable_weights_are_not_repacked(preparation):
    original_prepare, _ = preparation
    model = _tiny_model()
    candidate = storage._capture_candidate(model, False)
    assert candidate is not None
    original_prepare(model, use_gradient_checkpointing=False)
    assert storage._unchanged_after_preparation(candidate, model)
    with torch.no_grad():
        model.lm_head.weight[0, 0] += 1e-5
    assert not storage._unchanged_after_preparation(candidate, model)
    with torch.no_grad():
        model.lm_head.weight.copy_(candidate.half_storage.float())
    model.lm_head.weight.requires_grad_(True)
    assert not storage._unchanged_after_preparation(candidate, model)


@pytest.mark.skipif(not INTERPRETING, reason="CPU kernel test requires TRITON_INTERPRET=1")
@pytest.mark.parametrize("width", [16, 1536])
def test_original_gather_kernel_outputs_exact_fp32_values(width):
    torch.manual_seed(33)
    weight = torch.randn(73, width).half()
    ids = torch.tensor([[0, 72, 8, 0], [7, 3, 4, 71]], dtype=torch.long).t()
    actual = storage.frozen_embedding(weight, ids)
    expected = F.embedding(ids, weight.float())
    assert actual.shape == expected.shape and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical CUDA device")
def test_cuda_gather_and_real_prepare_storage_reduction():
    import peft

    assert storage.patch_frozen_storage()
    model = _tiny_model().cuda()
    parameter = model.get_input_embeddings().weight
    original_bytes = parameter.numel() * 4
    peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    assert model._candidate_frozen_storage is True
    assert model.get_input_embeddings().weight is parameter
    assert parameter.untyped_storage().nbytes() == original_bytes // 2
    ids = torch.tensor([[1, 66, 0, 8]], device="cuda")
    actual = model.get_input_embeddings()(ids)
    torch.testing.assert_close(actual, F.embedding(ids, parameter.float()), rtol=0, atol=0)
