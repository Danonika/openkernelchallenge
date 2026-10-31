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
import json
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


def _local_data(extra_tokens=0):
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
        length += extra_tokens
        ids = torch.randint(3, 257, (length,), generator=generator).tolist()
        ids[0] = index + 3
        prompt = length if index == 2 else 2 + index % 3
        rows.append({"input_ids": ids, "assistant_masks": [0] * prompt + [1] * (length - prompt)})
    return Dataset.from_list(rows), rows, tokenizer


def _run_native_training(root, model, extra_tokens=0):
    from bitsandbytes.backends.cuda import ops
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
            self.optimizer_lrs, self.optimizer_initial_state_empty, self.skipped_steps = [], None, []
        def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
            native = getattr(optimizer, "optimizer", optimizer)
            if self.optimizer_initial_state_empty is None:
                self.optimizer_initial_state_empty = not bool(native.state)
            self.optimizer_lrs.append([float(group["lr"]) for group in native.param_groups])
            self.gradients.append(params(model, gradients=True))
        def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
            self.skipped_steps.append(bool(getattr(optimizer, "step_was_skipped", False)))
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
                "token_positions": inputs["input_ids"].numel(),
            })
            return (loss, outputs) if return_outputs else loss

    root.mkdir(parents=True)
    dataset, rows, tokenizer = _local_data(extra_tokens=extra_tokens)
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
    native_optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    assert type(native_optimizer) is torch.optim.AdamW
    names = {id(parameter): name for name, parameter in trainer.model.named_parameters()
             if parameter.requires_grad}
    optimizer_keys = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
                      "foreach", "capturable", "differentiable", "fused")
    optimizer_groups = [dict(
        names=[names[id(parameter)] for parameter in group["params"]],
        **{key: group[key] for key in optimizer_keys if key in group})
        for group in native_optimizer.param_groups]
    # The installed cache owns weak references to frozen model weights. Take
    # its real counters while trainer/model still retain those weights, before
    # leaving this helper can release entries through their weakref callbacks.
    nf4_cache = getattr(ops._dequant_linear_fallback, "_soup_nf4_forward_cache", None)
    nf4_cache_info = nf4_cache.info() if nf4_cache is not None else None
    return {"initial": initial, "gradients": snapshots.gradients,
        "parameters": snapshots.parameters, "calls": trainer.observed_calls,
        "rewind": read_rewind_log(writer.path), "head_calls": head_calls,
        "decoder_calls": decoder_calls,
        "cpu_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state().clone(),
        "scaler": trainer.accelerator.scaler.state_dict(),
        "optimizer_groups": optimizer_groups,
        "optimizer_lrs": snapshots.optimizer_lrs,
        "optimizer_initial_state_empty": snapshots.optimizer_initial_state_empty,
        "optimizer_skipped_steps": snapshots.skipped_steps,
        "nf4_forward_cache_info": nf4_cache_info,
        "training_loss": float(result.training_loss)}


def _verify_native_optimizer_replay(record):
    """Replay actual post-clipping gradients through the same native AdamW.

    Adam's coordinate update is sensitive to tiny gradients: matching FP16
    derivatives does not imply elementwise matching parameters. This checks
    the optimizer itself without hiding those baseline/candidate differences.
    The fixture uses its original constant schedule and finite gradients.
    """
    parameters = {name: torch.nn.Parameter(value.to("cuda"))
                  for name, value in record["initial"].items()}
    assert record["optimizer_initial_state_empty"] is True
    groups = []
    for group in record["optimizer_groups"]:
        groups.append(dict(params=[parameters[name] for name in group["names"]],
                           **{key: value for key, value in group.items() if key != "names"}))
    assert sorted(name for group in record["optimizer_groups"] for name in group["names"]) == sorted(parameters)
    optimizer = torch.optim.AdamW(groups)
    largest = 0.0
    exact = True
    assert len(record["optimizer_lrs"]) == len(record["optimizer_skipped_steps"]) == len(record["parameters"])
    for step, (gradients, expected) in enumerate(zip(record["gradients"], record["parameters"])):
        assert len(record["optimizer_lrs"][step]) == len(optimizer.param_groups)
        for group, rate in zip(optimizer.param_groups, record["optimizer_lrs"][step]):
            group["lr"] = rate
        for name, parameter in parameters.items():
            parameter.grad = gradients[name].to(parameter.device)
        if not record["optimizer_skipped_steps"][step]:
            optimizer.step()
        for name, parameter in parameters.items():
            actual = parameter.detach().cpu()
            exact = exact and torch.equal(actual, expected[name])
            largest = max(largest, float((actual - expected[name]).abs().max()))
            torch.testing.assert_close(actual, expected[name], atol=3e-8, rtol=3e-7,
                msg=lambda message, n=name: f"native AdamW replay {n}: {message}")
    return {"verified": True, "max_abs_parameter_error": largest,
            "all_parameters_bitwise_equal": exact,
            "fp32_tolerance": {"atol": 3e-8, "rtol": 3e-7},
            "steps": len(record["parameters"]), "optimizer": "torch.optim.AdamW"}


def _rounding_evidence(candidate, baseline):
    """Retain the earlier per-element criteria alongside normwise evidence."""
    rows = []
    half_epsilon = torch.finfo(torch.float16).eps
    for key, atol, rtol in (("gradients", 3e-5, 3e-3), ("parameters", 3e-6, 2e-3)):
        for step, (actual, expected) in enumerate(zip(candidate[key], baseline[key]), 1):
            assert actual.keys() == expected.keys()
            for name, observed in actual.items():
                reference = expected[name]
                assert observed.shape == reference.shape and observed.dtype == reference.dtype
                assert torch.isfinite(observed).all() and torch.isfinite(reference).all()
                delta = observed.double() - reference.double()
                reference_norm = float(torch.linalg.vector_norm(reference.double()))
                reference_peak = float(reference.abs().max())
                error_norm = float(torch.linalg.vector_norm(delta))
                relative_l2 = error_norm / reference_norm if reference_norm else (0.0 if not error_norm else None)
                index = int(delta.abs().argmax())
                row = {"kind": key, "step": step, "name": name,
                       "legacy_mismatches": int((~torch.isclose(observed, reference, atol=atol, rtol=rtol)).sum()),
                       "relative_l2": relative_l2,
                       "reference_zero_norm": reference_norm == 0,
                       "max_abs_error": float(delta.abs().max()),
                       "reference_max_abs": reference_peak,
                       "worst_flat_index": index,
                       "worst_actual": float(observed.flatten()[index]),
                       "worst_reference": float(reference.flatten()[index])}
                if key == "gradients":
                    row.update(relative_l2_limit=2 * half_epsilon,
                               max_abs_limit=4 * half_epsilon * reference_peak + 1e-7)
                    # These are explicit FP16 integration screening budgets,
                    # not a proof or a substitute for whole-task quality.
                    row["screen_passed"] = (relative_l2 is not None and relative_l2 <= row["relative_l2_limit"]
                                             and row["max_abs_error"] <= row["max_abs_limit"])
                else:
                    prior_a = candidate["initial"][name] if step == 1 else candidate[key][step - 2][name]
                    prior_b = baseline["initial"][name] if step == 1 else baseline[key][step - 2][name]
                    update_a = observed.double() - prior_a.double()
                    update_b = reference.double() - prior_b.double()
                    row["update_relative_l2"] = float(torch.linalg.vector_norm(update_a - update_b)) / max(
                        float(torch.linalg.vector_norm(update_b)), 1e-30)
                    row["worst_actual_update"] = float(update_a.flatten()[index])
                    row["worst_reference_update"] = float(update_b.flatten()[index])
                    row["worst_actual_gradient"] = float(candidate["gradients"][step - 1][name].flatten()[index])
                    row["worst_reference_gradient"] = float(baseline["gradients"][step - 1][name].flatten()[index])
                rows.append(row)
    return rows


def _register_patch_restoration(monkeypatch):
    import peft
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    from bitsandbytes.autograd._functions import MatMul4Bit
    from bitsandbytes.backends.cuda import ops
    from peft.tuners.lora.layer import Linear
    from peft.tuners.lora.bnb import Linear4bit
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from trl import SFTTrainer
    from soup_cli.trainer import rewind_hf
    from . import lora_dropout, mlp_padding
    owners = [
        (qwen.Qwen2ForCausalLM, "forward"), (qwen.Qwen2MLP, "forward"),
        (qwen.Qwen2RMSNorm, "forward"), (qwen.Qwen2Model, "forward"),
        (qwen.Qwen2Attention, "forward"), (qwen, "apply_rotary_pos_emb"),
        (Linear, "forward"), (Linear4bit, "forward"), (torch.nn.Dropout, "forward"),
        (SFTTrainer, "compute_loss"), (functional, "dequantize_4bit"),
        (rewind_hf, "row_losses_from_token_losses"),
        (bnb.nn.Linear4bit, "forward"),
        (MatMul4Bit, "forward"), (MatMul4Bit, "backward"),
        (ops, "_dequant_linear_fallback"), (lora_dropout, "_PEFT_CLASSES"),
        (peft, "prepare_model_for_kbit_training"),
    ]
    owners.extend((mlp_padding, name) for name in (
        "_QWEN_CAUSAL_LM", "_QWEN_MLP", "_BNB_LINEAR", "_BNB_FORWARD", "_SILU_CLASS", "_SILU_FORWARD"))
    for owner, name in owners:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    # The instance-local registry entry may not exist before installation.
    # monkeypatch records that absence and removes our entry at teardown.
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._local_mapping, "sdpa",
                        ALL_ATTENTION_FUNCTIONS["sdpa"])


def _kernel_spies(monkeypatch):
    from . import (cached_head, streamed_head, nf4, rmsnorm, activation_storage,
                   lora_dropout, lora_epilogue, rotary, shared_attention_mask, frozen_storage,
                   mlp_padding, cached_ce_backward, sparse_cached_head, varlen_attention,
                   varlen_attention_patch, nf4_cached_linear)
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
        (cached_ce_backward, "soup_native_logp_ce_gradient", "native_ce_backward"),
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
        (frozen_storage, "soup_frozen_embedding_forward", "frozen_embedding_kernel"),
        (mlp_padding, "soup_mlp_gather_rows", "mlp_gather"),
        (mlp_padding, "soup_mlp_scatter_rows", "mlp_scatter"),
        (varlen_attention, "soup_varlen_pack_qkv", "varlen_pack_qkv"),
        (varlen_attention, "soup_varlen_scatter_qkv", "varlen_scatter_qkv"),
        (varlen_attention, "soup_varlen_scatter_output", "varlen_scatter_output"),
        (varlen_attention, "soup_varlen_gather_output_gradient", "varlen_gather_output_gradient"),
    ]
    for owner, attr, label in entries:
        monkeypatch.setattr(owner, attr, LaunchSpy(getattr(owner, attr), label))
    for owner, name, label in (
            (varlen_attention, "_native_forward", "varlen_native_forward"),
            (varlen_attention, "_native_backward", "varlen_native_backward"),
            (varlen_attention_patch, "_verify_mask", "varlen_mask_verifications")):
        native = getattr(owner, name)
        def observed(*args, _native=native, _label=label, **kwargs):
            result = _native(*args, **kwargs)
            counts[_label] += 1
            return result
        monkeypatch.setattr(owner, name, observed)
    # Storage-only eager retention deliberately uses native Embedding.forward.
    # Observe that route directly and check its exact values/dtype/storage.
    native_retained_embedding = frozen_storage._embedding_forward
    def observed_retained_embedding(module, token_ids):
        pointer = module.weight.data_ptr()
        assert module.weight.dtype == torch.float16 and not module.weight.requires_grad
        result = native_retained_embedding(module, token_ids)
        assert module.weight.data_ptr() == pointer
        assert result.dtype == torch.float32
        assert torch.equal(result, torch.nn.Embedding.forward(module, token_ids).float())
        counts["frozen_embedding"] += 1
        return result
    monkeypatch.setattr(frozen_storage, "_embedding_forward", observed_retained_embedding)
    native_mlp_call = mlp_padding._make_call
    def observed_mlp_call(*args, **kwargs):
        result = native_mlp_call(*args, **kwargs)
        if result is not None:
            counts["mlp_compacted_calls"] += 1
            counts["mlp_valid_rows"] += result.rows.numel()
            counts["mlp_original_rows"] += result.inverse.numel()
        return result
    monkeypatch.setattr(mlp_padding, "_make_call", observed_mlp_call)
    original_direct_components = nf4_cached_linear._eligible_components
    def direct_components(*args, **kwargs):
        result = original_direct_components(*args, **kwargs)
        if result is not None:
            counts["nf4_direct_linear_calls"] += 1
        return result
    monkeypatch.setattr(nf4_cached_linear, "_eligible_components", direct_components)
    original_streamed = streamed_head.streamed_linear_ce
    def streamed(*args, **kwargs):
        result = original_streamed(*args, **kwargs)
        counts["streamed_calls"] += 1
        return result
    monkeypatch.setattr(streamed_head, "streamed_linear_ce", streamed)
    original_sparse_cached = sparse_cached_head.sparse_cached_linear_ce
    def sparse_cached(*args, **kwargs):
        result = original_sparse_cached(*args, **kwargs)
        counts["sparse_cached_calls"] += 1
        return result
    monkeypatch.setattr(sparse_cached_head, "sparse_cached_linear_ce", sparse_cached)
    for owner, label in ((cached_head, "cached_native"), (streamed_head, "streamed_native")):
        native = owner._native_row_statistics
        def statistics(scores, targets, _native=native, _label=label):
            result = _native(scores, targets)
            counts[_label + "_calls"] += 1
            counts[_label + "_rows"] += scores.shape[0]
            return result
        monkeypatch.setattr(owner, "_native_row_statistics", statistics)
    # Observe the two launchable rotary kernels without wrapping their JIT
    # helper functions, which are compiled into a caller rather than launched.
    for attr in ("soup_rotary_forward", "soup_rotary_backward"):
        monkeypatch.setattr(rotary, attr, LaunchSpy(getattr(rotary, attr), attr))
    return counts


@pytest.mark.skipif(not _PHYSICAL_CUDA, reason="requires physical CUDA, not CPU or Triton interpreter")
@pytest.mark.parametrize("extra_tokens", (0, 16))
def test_physical_t4_full_fusion_training(monkeypatch, tmp_path, record_property, extra_tokens):
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from . import apply, settings

    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("the target integration gate requires an SM75 T4")
    assert torch.cuda.device_count() == 1, "expose exactly the single target GPU"
    assert settings.HEAD_MODE in {"cached", "streamed", "sparse_cached"}
    assert settings.HEAD_NUMERICS in {"native", "guarded"}
    assert qwen.Qwen2MLP.forward.__module__.startswith("transformers."), (
        "run this physical gate in a fresh Python process before solution.apply")
    _register_patch_restoration(monkeypatch)
    baseline = _run_native_training(tmp_path / "baseline", _tiny_nested_nf4_model(),
                                    extra_tokens=extra_tokens)
    counts = _kernel_spies(monkeypatch)
    apply()
    candidate = _run_native_training(tmp_path / "candidate", _tiny_nested_nf4_model(),
                                     extra_tokens=extra_tokens)
    record_property("soup_actual_kernel_counts", json.dumps(dict(counts)))
    record_property("soup_fixture_extra_tokens", extra_tokens)
    record_property("soup_nf4_forward_cache", json.dumps({
        "baseline": baseline["nf4_forward_cache_info"],
        "candidate": candidate["nf4_forward_cache_info"]}, allow_nan=False))

    assert baseline["initial"].keys() == candidate["initial"].keys()
    for name in baseline["initial"]:
        torch.testing.assert_close(candidate["initial"][name], baseline["initial"][name], atol=0, rtol=0)
    evidence = _rounding_evidence(candidate, baseline)
    record_property("soup_fp16_rounding_evidence", json.dumps(evidence, allow_nan=False))
    # The original elementwise gate rejected the unchanged credited example
    # itself on physical T4. Keep those residuals visible, require a bounded
    # FP16 gradient discrepancy, and verify actual optimizer arithmetic below.
    failed = [row for row in evidence if row["kind"] == "gradients" and not row["screen_passed"]]
    assert not failed, json.dumps(failed, indent=2)
    assert candidate["optimizer_groups"] == baseline["optimizer_groups"]
    replay = {"baseline": _verify_native_optimizer_replay(baseline),
              "candidate": _verify_native_optimizer_replay(candidate)}
    record_property("soup_native_optimizer_replay", json.dumps(replay, allow_nan=False))
    assert torch.equal(candidate["cpu_rng"], baseline["cpu_rng"])
    assert torch.equal(candidate["cuda_rng"], baseline["cuda_rng"])
    assert candidate["scaler"] == baseline["scaler"]
    assert len(baseline["head_calls"]) == 2
    if settings.HEAD_MODE == "cached":
        assert candidate["head_calls"] == baseline["head_calls"]
    else:
        assert candidate["head_calls"] == []
        label = "sparse_cached_calls" if settings.HEAD_MODE == "sparse_cached" else "streamed_calls"
        assert counts[label] == 2
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
    if settings.HEAD_NUMERICS == "native":
        label = "cached_native" if settings.HEAD_MODE in ("cached", "sparse_cached") else "streamed_native"
        assert counts[label + "_calls"] > 0
        assert counts[label + "_rows"] == sum(call["token_positions"] for call in candidate["calls"])
        assert all(counts[name] == 0 for name in
                   ("head_partials", "head_finish", "streamed_partials", "streamed_merge"))
    elif settings.HEAD_MODE in ("cached", "sparse_cached"):
        required.extend(("head_partials", "head_finish"))
        assert counts["head_partials"] == counts["head_finish"] == 2
    elif settings.HEAD_PROJECTION != "t4_fused":
        required.extend(("streamed_partials", "streamed_merge"))
    if settings.HEAD_MODE in ("cached", "sparse_cached") and settings.CACHED_CE_BACKWARD == "native_fused":
        required.append("native_ce_backward")
    if settings.FUSED_NF4:
        required.append("nf4")
    if extra_tokens > 0 and settings.NF4_FORWARD_CACHE_BYTES > 0:
        cache_info = candidate["nf4_forward_cache_info"]
        assert cache_info is not None, "enabled NF4 forward cache was not installed"
        assert cache_info["entries"] > 0, f"longer geometry retained no cached weights: {cache_info}"
        assert cache_info["hits"] > 0, f"longer geometry never reused cached weights: {cache_info}"
    if extra_tokens > 0 and settings.NF4_DIRECT_LINEAR:
        required.append("nf4_direct_linear_calls")
    if settings.FUSED_RMSNORM:
        required.extend(("rms_forward", "rms_backward"))
    if settings.FUSED_LORA_DROPOUT:
        required.append("l1e_backward")
    if settings.COMPACT_MLP_ACTIVATIONS:
        required.extend(("compact_forward", "compact_backward"))
    if settings.MLP_PADDING_COMPACTION:
        required.extend(("mlp_gather", "mlp_scatter", "mlp_compacted_calls"))
        assert 0 < counts["mlp_valid_rows"] < counts["mlp_original_rows"]
    if settings.FUSED_LORA_EPILOGUE:
        required.extend(("lora_scale_add_forward", "lora_scale_add_backward"))
    if settings.FUSED_ROTARY:
        required.extend(("soup_rotary_forward", "soup_rotary_backward"))
    if settings.SHARED_ATTENTION_MASK:
        required.append("shared_mask")
    if settings.VARLEN_ATTENTION:
        required.extend(("varlen_pack_qkv", "varlen_scatter_qkv", "varlen_scatter_output",
                         "varlen_gather_output_gradient", "varlen_native_forward", "varlen_native_backward"))
        assert counts["varlen_mask_verifications"] == 2
        assert counts["varlen_native_forward"] == counts["varlen_native_backward"] == 4
    if settings.FROZEN_STORAGE:
        required.append("frozen_embedding")
        assert counts["frozen_embedding"] == 2
        assert counts["frozen_embedding_kernel"] == 0
    for name in required:
        assert counts[name] > 0, f"required component never executed: {name}; {dict(counts)}"
    # These counts prove execution only; they do not establish the official
    # original-kernel percentage. Credited compact SwiGLU is never original.
