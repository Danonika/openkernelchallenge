"""Physical-T4 admission test for the selected merged candidate profile.

This test uses a locally initialized model and a tiny local dataset. Every
decoder projection is real nested-NF4 bitsandbytes; the full Soup factory,
TRL training loop, Accelerate FP16/GradScaler, AdamW and Rewind writer execute.
There are no device-guard overrides, fake GEMMs or interpreter substitutes.

Passing is a component-integration gate, not the official 64-step benchmark,
hidden-set quality check, peak-memory result, or five-percent kernel-share
qualification. Run this test in a fresh Python process before solution.apply.
"""

from collections import Counter
import os

import pytest
import torch


_PHYSICAL_CUDA = torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"


def _tiny_nested_nf4_model():
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2Config, Qwen2ForCausalLM
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    from transformers.utils.quantization_config import QuantizationMethod

    torch.manual_seed(173)
    cfg = Qwen2Config(vocab_size=257, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, max_position_embeddings=64, tie_word_embeddings=True,
                     attention_dropout=0.0, pad_token_id=0, eos_token_id=1)
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    base = Qwen2ForCausalLM(cfg).half()
    replacements = []
    # Preserve every native dense value before quantizing with the real BNB
    # quantizer. The tied head/embedding remain ordinary frozen parameters.
    for name, module in base.model.named_modules():
        if type(module) is torch.nn.Linear:
            replacements.append((name, module))
    assert len(replacements) == 14
    for name, module in replacements:
        quantized = bnb.nn.Linear4bit(module.in_features, module.out_features,
            bias=module.bias is not None, compute_dtype=torch.float16,
            compress_statistics=True, quant_type="nf4").half()
        with torch.no_grad():
            quantized.weight.copy_(module.weight)
            if module.bias is not None:
                quantized.bias.copy_(module.bias)
        quantized = quantized.to("cuda")
        parent_name, _, child = name.rpartition(".")
        parent = base.model.get_submodule(parent_name) if parent_name else base.model
        setattr(parent, child, quantized)
    base.to("cuda")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.float16)
    # These are truthful metadata for the native quantization above. Attach
    # the real Transformers quantizer so its training checks run unchanged.
    base.is_loaded_in_4bit = True
    base.is_quantized = True
    base.quantization_method = QuantizationMethod.BITS_AND_BYTES
    base.hf_quantizer = Bnb4BitHfQuantizer(quant_config)
    base.config.quantization_config = quant_config
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    generator = torch.Generator().manual_seed(271)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator).to(param.device) * 0.02)
    assert base.lm_head.weight is base.model.embed_tokens.weight
    expected_storage = torch.float16 if getattr(base, "_candidate_frozen_storage", False) else torch.float32
    assert base.lm_head.weight.dtype == expected_storage and not base.lm_head.weight.requires_grad
    native_quantized = [m for m in model.modules() if type(m) is bnb.nn.Linear4bit]
    assert len(native_quantized) == 14
    assert all(m.weight.dtype == torch.uint8 and m.weight.quant_state.nested
               and m.weight.quant_state.dtype == torch.float16 for m in native_quantized)
    return model.train()


def _local_data():
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"tok{i}": i for i in range(3, 257)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
        eos_token="<eos>", unk_token="<unk>", padding_side="right", model_max_length=64)
    generator = torch.Generator().manual_seed(731)
    rows = []
    for index, length in enumerate((9, 6, 13, 7, 11, 10, 8, 12)):
        ids = torch.randint(3, 257, (length,), generator=generator).tolist()
        ids[0] = index + 3
        prompt = length if index == 2 else 2 + index % 3
        rows.append({"input_ids": ids, "assistant_masks": [0] * prompt + [1] * (length - prompt)})
    return Dataset.from_list(rows), rows, tokenizer


def _run_native_training(root, model):
    from transformers import TrainerCallback
    from trl import SFTTrainer, SFTConfig
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.monitoring.rewind_log import read_rewind_log
    from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

    def params(module, gradients=False):
        result = {}
        for name, p in module.named_parameters():
            if p.requires_grad:
                value = p.grad if gradients else p
                assert value is not None, f"missing adapter gradient: {name}"
                assert torch.isfinite(value).all(), f"nonfinite adapter state: {name}"
                result[name] = value.detach().cpu().clone()
        return result

    class Snapshots(TrainerCallback):
        def __init__(self):
            self.gradients, self.parameters = [], []
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            self.gradients.append(params(model, gradients=True))
        def on_step_end(self, args, state, control, model=None, **kwargs):
            self.parameters.append(params(model))

    class ObservedSFT(SFTTrainer):
        def __init__(self, *args, **kwargs):
            self.observed_calls = []
            super().__init__(*args, **kwargs)
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
            self.observed_calls.append({
                "loss": float(loss.detach()),
                "rows": (inputs["input_ids"][:, 0] - 3).detach().cpu().tolist(),
                "metrics": {k: list(v) for k, v in self._metrics["train"].items()},
                "has_logits": outputs.logits is not None,
                "outer_cuda_autocast": torch.is_autocast_enabled("cuda"),
            })
            return (loss, outputs) if return_outputs else loss

    root.mkdir(parents=True)
    dataset, rows, tokenizer = _local_data()
    config = SoupConfig(base="local-fusion-fixture-qwen2",
        data={"train": "local-fusion-fixture.jsonl", "max_length": 64},
        training={"quantization": "4bit", "gradient_checkpointing": False}, output=str(root))
    wrapper = SFTTrainerWrapper(config, device="cuda")
    wrapper.model, wrapper.tokenizer = model, tokenizer
    args = SFTConfig(output_dir=str(root), fp16=True, bf16=False,
        report_to=[], disable_tqdm=True, per_device_train_batch_size=4,
        gradient_accumulation_steps=1, max_steps=2, learning_rate=2e-4,
        optim="adamw_torch", lr_scheduler_type="constant", max_grad_norm=1.0,
        logging_steps=1, save_strategy="no", seed=419, data_seed=419,
        gradient_checkpointing=False, packing=False, padding_free=False,
        activation_offloading=False, use_liger_kernel=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=0, dataloader_pin_memory=False)
    snapshots = Snapshots()
    trainer = wrapper._build_rewind_trainer(ObservedSFT,
        dict(model=model, args=args, train_dataset=dataset, processing_class=tokenizer,
             callbacks=[snapshots]), rows=rows, output_dir=root, batch_size=4, grad_accum=1)
    # This is exactly Soup's existing post-TRL, pre-optimizer correction. It
    # retains the preceding BF16 rounding instead of restoring old weights.
    align_trainable_dtype_for_fp16(trainer.model, fp16=True, bf16=False)
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters() if p.requires_grad)
    initial = params(trainer.model)
    head_calls, decoder_calls = [], []
    def head_forward(module, inputs, output):
        head_calls.append({"autocast": torch.is_autocast_enabled("cuda"),
                           "dtype": output.dtype, "shape": tuple(output.shape)})
    hook = model.get_base_model().lm_head.register_forward_hook(head_forward)
    decoder_hook = model.get_base_model().model.register_forward_pre_hook(
        lambda module, inputs: decoder_calls.append(torch.is_autocast_enabled("cuda")))
    torch.manual_seed(991)
    torch.cuda.manual_seed_all(991)
    try:
        result = trainer.train()
    finally:
        hook.remove()
        decoder_hook.remove()
        wrapper._rewind_log.close()
    torch.cuda.synchronize()
    state, writer = wrapper._rewind_state, wrapper._rewind_log
    assert not state.failed and state.dropped == 0 and writer.dropped == 0 and not writer.disabled
    assert trainer.accelerator.scaler is not None
    assert trainer.state.global_step == 2
    assert len(snapshots.gradients) == len(snapshots.parameters) == len(decoder_calls) == 2
    assert all(decoder_calls)
    assert all(call["autocast"] and call["dtype"] == torch.float16 for call in head_calls)
    assert all(not call["outer_cuda_autocast"] for call in trainer.observed_calls)
    return {"initial": initial, "gradients": snapshots.gradients,
        "parameters": snapshots.parameters, "calls": trainer.observed_calls,
        "rewind": read_rewind_log(writer.path), "head_calls": head_calls,
        "decoder_calls": decoder_calls,
        "cpu_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state().clone(),
        "scaler": trainer.accelerator.scaler.state_dict(),
        "training_loss": float(result.training_loss)}


def _register_patch_restoration(monkeypatch):
    import peft
    import bitsandbytes.functional as functional
    from bitsandbytes.backends.cuda import ops
    from peft.tuners.lora.layer import Linear
    from peft.tuners.lora.bnb import Linear4bit
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from trl import SFTTrainer
    from . import lora_dropout
    owners = [
        (qwen.Qwen2ForCausalLM, "forward"), (qwen.Qwen2MLP, "forward"),
        (qwen.Qwen2RMSNorm, "forward"), (qwen.Qwen2Model, "forward"),
        (qwen.Qwen2Attention, "forward"), (qwen, "apply_rotary_pos_emb"),
        (Linear, "forward"), (Linear4bit, "forward"), (torch.nn.Dropout, "forward"),
        (SFTTrainer, "compute_loss"), (functional, "dequantize_4bit"),
        (ops, "_dequant_linear_fallback"), (lora_dropout, "_PEFT_CLASSES"),
        (peft, "prepare_model_for_kbit_training"),
    ]
    for owner, name in owners:
        monkeypatch.setattr(owner, name, getattr(owner, name))


def _kernel_spies(monkeypatch):
    from . import (cached_head, streamed_head, nf4, rmsnorm, activation_storage,
                   lora_dropout, lora_epilogue, rotary, shared_attention_mask, frozen_storage)
    counts = Counter()
    class LaunchSpy:
        def __init__(self, native, label):
            self.native, self.label = native, label
        def __getitem__(self, grid):
            launch = self.native[grid]
            def wrapped(*args, **kwargs):
                result = launch(*args, **kwargs)
                counts[self.label] += 1
                return result
            return wrapped
    entries = [
        (cached_head, "soup_head_partial_statistics", "head_partials"),
        (cached_head, "soup_cached_head_finish", "head_finish"),
        (streamed_head, "soup_head_partial_statistics", "streamed_partials"),
        (streamed_head, "soup_head_merge_statistics", "streamed_merge"),
        (nf4, "soup_nf4_dequantize", "nf4"),
        (rmsnorm, "soup_rmsnorm_forward", "rms_forward"),
        (rmsnorm, "soup_rmsnorm_backward", "rms_backward"),
        (lora_dropout, "soup_dropout_half_gradient_epilogue", "l1e_backward"),
        (activation_storage, "example_swiglu_fwd_kernel", "compact_forward"),
        (activation_storage, "example_swiglu_bwd_kernel", "compact_backward"),
        (lora_epilogue, "soup_lora_epilogue_forward", "lora_scale_add_forward"),
        (lora_epilogue, "soup_lora_epilogue_backward", "lora_scale_add_backward"),
        (shared_attention_mask, "soup_aligned_additive_mask", "shared_mask"),
        (frozen_storage, "soup_frozen_embedding_forward", "frozen_embedding"),
    ]
    for owner, attr, label in entries:
        monkeypatch.setattr(owner, attr, LaunchSpy(getattr(owner, attr), label))
    original_streamed = streamed_head.streamed_linear_ce
    def streamed(*args, **kwargs):
        result = original_streamed(*args, **kwargs)
        counts["streamed_calls"] += 1
        return result
    monkeypatch.setattr(streamed_head, "streamed_linear_ce", streamed)
    # Observe the two launchable rotary kernels without wrapping their JIT
    # helper functions, which are compiled into a caller rather than launched.
    for attr in ("soup_rotary_forward", "soup_rotary_backward"):
        monkeypatch.setattr(rotary, attr, LaunchSpy(getattr(rotary, attr), attr))
    return counts


@pytest.mark.skipif(not _PHYSICAL_CUDA, reason="requires physical CUDA, not CPU or Triton interpreter")
def test_physical_t4_full_fusion_training(monkeypatch, tmp_path):
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from . import apply, settings

    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("the target integration gate requires an SM75 T4")
    assert torch.cuda.device_count() == 1, "expose exactly the single target GPU"
    assert settings.HEAD_MODE in {"cached", "streamed"}
    assert qwen.Qwen2MLP.forward.__module__.startswith("transformers."), (
        "run this physical gate in a fresh Python process before solution.apply")
    _register_patch_restoration(monkeypatch)
    baseline = _run_native_training(tmp_path / "baseline", _tiny_nested_nf4_model())
    counts = _kernel_spies(monkeypatch)
    apply()
    candidate = _run_native_training(tmp_path / "candidate", _tiny_nested_nf4_model())

    assert baseline["initial"].keys() == candidate["initial"].keys()
    for name in baseline["initial"]:
        torch.testing.assert_close(candidate["initial"][name], baseline["initial"][name], atol=0, rtol=0)
    for key, atol, rtol in (("gradients", 3e-5, 3e-3), ("parameters", 3e-6, 2e-3)):
        for actual, expected in zip(candidate[key], baseline[key]):
            assert actual.keys() == expected.keys()
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol,
                                           msg=lambda msg, n=name: f"{key} {n}: {msg}")
    assert torch.equal(candidate["cpu_rng"], baseline["cpu_rng"])
    assert torch.equal(candidate["cuda_rng"], baseline["cuda_rng"])
    assert candidate["scaler"] == baseline["scaler"]
    assert len(baseline["head_calls"]) == 2
    if settings.HEAD_MODE == "cached":
        assert candidate["head_calls"] == baseline["head_calls"]
    else:
        assert candidate["head_calls"] == []
        assert counts["streamed_calls"] == 2
    assert candidate["decoder_calls"] == baseline["decoder_calls"]
    for actual, expected in zip(candidate["calls"], baseline["calls"]):
        assert actual["rows"] == expected["rows"]
        assert not actual["has_logits"] and expected["has_logits"]
        assert actual["loss"] == pytest.approx(expected["loss"], rel=3e-4, abs=3e-5)
        assert actual["metrics"].keys() == expected["metrics"].keys()
        for name in actual["metrics"]:
            if name == "num_tokens":
                assert actual["metrics"][name] == expected["metrics"][name]
            else:
                assert actual["metrics"][name] == pytest.approx(expected["metrics"][name], rel=0.03, abs=1e-6)
    a, b = candidate["rewind"], baseline["rewind"]
    assert len(a.batches) == len(b.batches) == 2
    assert sorted(row for batch in a.batches for row in batch.rows) == list(range(8))
    for actual, expected in zip(a.batches, b.batches):
        assert (actual.step, actual.micro, actual.rows, actual.row_tokens) == (
            expected.step, expected.micro, expected.rows, expected.row_tokens)
        assert actual.row_loss == pytest.approx(expected.row_loss, rel=3e-4, abs=3e-5)
    assert {k: v for k, v in a.header.items() if k != "created"} == {
        k: v for k, v in b.header.items() if k != "created"}
    required = []
    if settings.HEAD_MODE == "cached":
        required.extend(("head_partials", "head_finish"))
        assert counts["head_partials"] == counts["head_finish"] == 2
    elif settings.HEAD_PROJECTION != "t4_fused":
        required.extend(("streamed_partials", "streamed_merge"))
    if settings.FUSED_NF4:
        required.append("nf4")
    if settings.FUSED_RMSNORM:
        required.extend(("rms_forward", "rms_backward"))
    if settings.FUSED_LORA_DROPOUT:
        required.append("l1e_backward")
    if settings.COMPACT_MLP_ACTIVATIONS:
        required.extend(("compact_forward", "compact_backward"))
    if settings.FUSED_LORA_EPILOGUE:
        required.extend(("lora_scale_add_forward", "lora_scale_add_backward"))
    if settings.FUSED_ROTARY:
        required.extend(("soup_rotary_forward", "soup_rotary_backward"))
    if settings.SHARED_ATTENTION_MASK:
        required.append("shared_mask")
    if settings.FROZEN_STORAGE:
        required.append("frozen_embedding")
    for name in required:
        assert counts[name] > 0, f"required component never executed: {name}; {dict(counts)}"
    # These counts prove execution only; they do not establish the official
    # original-kernel percentage. Credited compact SwiGLU is never original.
