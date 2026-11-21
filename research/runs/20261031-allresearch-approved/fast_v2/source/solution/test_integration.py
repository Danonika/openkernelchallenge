"""Training integration checks, independent of any performance measurement.

The optional CPU interpreter checks deliberately replace the CUDA capability
guard inside the test only. Production still requires CUDA FP16 autocast.
"""

from collections import defaultdict
from functools import wraps
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from . import integration


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"


class _LocalAccelerator:
    num_processes = 1
    parallelism_config = None

    @staticmethod
    def gather_for_metrics(value):
        return value


def _trainer(model=None):
    state = SimpleNamespace(
        model=model,
        accelerator=_LocalAccelerator(),
        args=SimpleNamespace(
            average_tokens_across_devices=True, n_gpu=1,
            fp16=True, bf16=False, loss_type="nll", use_liger_kernel=False,
            packing=False,
        ),
        label_smoother=None,
        compute_loss_func=None,
        model_accepts_loss_kwargs=True,
        padding_free=False,
        num_virtual_tokens=0,
        aux_loss_enabled=False,
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens=0,
    )
    if model is None:
        return state
    # TRL's reference method uses zero-argument super(); use a real trainer
    # instance with only the state needed by compute_loss, without downloads.
    from trl import SFTTrainer

    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.__dict__.update(vars(state))
    return trainer


@pytest.fixture
def patched_classes(monkeypatch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from trl import SFTTrainer
    from soup_cli.trainer import rewind_hf

    def without_candidate(function):
        while getattr(function, "_soup_candidate_head", False):
            function = function.__wrapped__
        return function

    reference_forward = without_candidate(Qwen2ForCausalLM.forward)
    reference_compute = without_candidate(SFTTrainer.compute_loss)
    # Record the initial state with monkeypatch, so even direct class assignment
    # by the production installer is restored at the end of this test.
    monkeypatch.setattr(Qwen2ForCausalLM, "forward", reference_forward)
    monkeypatch.setattr(SFTTrainer, "compute_loss", reference_compute)
    original_rows = rewind_hf.row_losses_from_token_losses
    while getattr(original_rows, "_soup_candidate_metric_packet", False):
        original_rows = original_rows.__wrapped__
    monkeypatch.setattr(rewind_hf, "row_losses_from_token_losses", original_rows)
    assert integration.patch_streamed_training()
    return Qwen2ForCausalLM, SFTTrainer, reference_forward, reference_compute


def test_metric_masks_count_prompts_and_last_position_but_accuracy_is_shifted():
    trainer = _trainer()
    trainer._total_train_tokens = 10
    inputs = {
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]]),
        "labels": torch.tensor([[-100, -100, 4, 5, -100], [-100, 7, -100, -100, -100]]),
    }
    result = SimpleNamespace(
        entropy=torch.tensor([[1., 2., 3., 4., 100.], [10., 20., 100., 100., 100.]]),
        predictions=torch.tensor([[4, 4, 99, 5, 0], [7, 1, 0, 0, 0]]),
    )
    integration._log_statistics(trainer, inputs, result)
    assert trainer._metrics["train"]["entropy"] == pytest.approx([40 / 6])
    assert trainer._metrics["train"]["mean_token_accuracy"] == pytest.approx([2 / 3])
    assert trainer._metrics["train"]["num_tokens"] == [16]
    assert trainer._total_train_tokens == 16
    assert not trainer._metrics["eval"]


def test_metric_accuracy_handles_no_supervised_targets():
    trainer = _trainer()
    inputs = {"labels": torch.full((1, 3), -100), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    result = SimpleNamespace(entropy=torch.tensor([[2., 3., 4.]]), predictions=torch.zeros(1, 3, dtype=torch.long))
    integration._log_statistics(trainer, inputs, result)
    assert trainer._metrics["train"]["entropy"] == [3.0]
    assert trainer._metrics["train"]["mean_token_accuracy"] == [0.0]
    assert trainer._metrics["train"]["num_tokens"] == [3]


def test_frozen_weight_cache_invalidation_does_not_change_state_dict():
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(5, 11, bias=False)
    model.lm_head.weight.requires_grad_(False)
    parameter = model.lm_head.weight
    before = {name: value.clone() for name, value in model.state_dict().items()}
    first = integration._cached_weight_fp16(model)
    assert first.dtype == torch.float16
    assert integration._cached_weight_fp16(model) is first
    assert list(model.state_dict()) == list(before)
    assert list(model.named_buffers()) == []
    assert model.lm_head.weight is parameter
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])

    with torch.no_grad():
        parameter.add_(0.25)
    second = integration._cached_weight_fp16(model)
    assert second is not first
    assert torch.equal(second, parameter.half())
    assert integration._cached_weight_fp16(model) is second

    model.lm_head.weight = torch.nn.Parameter(torch.zeros_like(parameter), requires_grad=False)
    third = integration._cached_weight_fp16(model)
    assert third is not second
    assert torch.count_nonzero(third) == 0
    assert list(model.state_dict()) == list(before)
    assert list(model.named_buffers()) == []


def test_weight_cache_rejects_trainable_head():
    model = SimpleNamespace(lm_head=torch.nn.Linear(3, 7, bias=False))
    with pytest.raises(ValueError, match="frozen"):
        integration._cached_weight_fp16(model)


@pytest.mark.skipif(not INTERPRETING, reason="CPU test requires TRITON_INTERPRET=1")
def test_rewind_token_loss_protocol_matches_original_recorder():
    from soup_cli.trainer.rewind_hf import row_losses, row_losses_from_token_losses
    from .streamed_head import streamed_linear_ce

    torch.manual_seed(102)
    hidden = torch.randn(2, 5, 9) * 0.3
    weight = torch.randn(19, 9) * 0.2
    labels = torch.tensor([[8, -100, 4, 7, -100], [-100, -100, -100, -100, -100]])
    result = streamed_linear_ce(hidden, weight, labels, chunk_size=3)
    reference_logits = F.linear(hidden.half(), weight.half()).float()
    expected_losses, expected_counts = row_losses(reference_logits, labels)
    actual_losses, actual_counts = row_losses_from_token_losses(result.token_loss, labels)
    assert actual_counts == expected_counts == [2, 0]
    assert actual_losses == pytest.approx(expected_losses, rel=3e-6, abs=3e-6)
    assert result.token_loss.shape == labels.shape
    assert torch.equal(result.token_loss[:, -1], torch.zeros(2))
    assert torch.equal(result.token_loss[1], torch.zeros(5))


def test_patch_is_idempotent(patched_classes):
    qwen_class, trainer_class, _, _ = patched_classes
    forward, compute = qwen_class.forward, trainer_class.compute_loss
    assert integration.patch_streamed_training()
    assert qwen_class.forward is forward
    assert trainer_class.compute_loss is compute


def test_exception_restores_nested_context_without_retry(monkeypatch, patched_classes):
    _, trainer_class, _, _ = patched_classes
    expected_model = object()
    monkeypatch.setattr(integration, "_supported_causal_lm", lambda *args: expected_model)

    class FailingModel(torch.nn.Module):
        calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            assert integration._HEAD_CALL.get().causal_lm is expected_model
            assert kwargs["num_items_in_batch"] == 9
            assert kwargs["use_cache"] is False
            assert "_prediction_loss_only" not in kwargs
            raise RuntimeError("deliberate decoder failure")

    model = FailingModel()
    trainer = _trainer(model)
    inputs = {"labels": torch.tensor([[1, 2]]), "_prediction_loss_only": True}
    previous = integration._HeadCall(object())
    outer_token = integration._HEAD_CALL.set(previous)
    try:
        with pytest.raises(RuntimeError, match="deliberate decoder failure"):
            trainer_class.compute_loss(trainer, model, inputs, num_items_in_batch=9)
        assert integration._HEAD_CALL.get() is previous
        assert model.calls == 1
        assert inputs == {"labels": inputs["labels"], "_prediction_loss_only": True}
        assert not trainer._metrics["train"]
    finally:
        integration._HEAD_CALL.reset(outer_token)


def _tiny_peft_model(hidden_size=16):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(138)
    config = Qwen2Config(
        vocab_size=67, hidden_size=hidden_size, intermediate_size=hidden_size * 2,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, pad_token_id=0, tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    base = Qwen2ForCausalLM(config)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
    ))
    # Nonzero B matrices exercise gradients of both LoRA factors.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.03)
    model.train()
    return model, base


def _batch():
    return {
        "input_ids": torch.tensor([[1, 4, 9, 2, 3], [1, 5, 8, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, -100, 9, 2, 3], [-100, 5, 8, -100, -100]]),
    }


@pytest.mark.parametrize("projection", ["torch", "t4_fused"])
def test_normal_qwen_and_peft_cpu_paths_fall_back(monkeypatch, patched_classes, projection):
    _, trainer_class, reference_forward, reference_compute = patched_classes
    monkeypatch.setattr(integration, "HEAD_PROJECTION", projection)
    model, base = _tiny_peft_model()
    batch = _batch()
    trainer = _trainer(model)
    assert integration._supported_causal_lm(trainer, model, batch, type(base)) is None
    torch.manual_seed(518)
    expected_loss, expected = reference_compute(trainer, model, dict(batch), return_outputs=True, num_items_in_batch=7)
    other = _trainer(model)
    setattr(other, integration._REWIND_STEP, SimpleNamespace(token_loss=torch.ones(2, 5)))
    torch.manual_seed(518)
    loss, outputs = trainer_class.compute_loss(other, model, dict(batch), return_outputs=True, num_items_in_batch=7)
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(outputs.logits, expected.logits, rtol=0, atol=0)
    assert outputs.logits.shape == (2, 5, 67)
    assert dict(other._metrics["train"]) == dict(trainer._metrics["train"])
    assert getattr(other, integration._REWIND_STEP) is None
    assert integration._HEAD_CALL.get() is None

    base.eval()
    with torch.no_grad():
        regular = base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        expected = reference_forward(base, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    torch.testing.assert_close(regular.logits, expected.logits, rtol=0, atol=0)


@pytest.mark.skipif(not INTERPRETING, reason="CPU integration test requires TRITON_INTERPRET=1")
@pytest.mark.parametrize("projection", ["torch", "t4_fused"])
def test_streamed_path_keeps_peft_amp_wrapper_metrics_and_gradients(monkeypatch, patched_classes, projection):
    from accelerate.utils.operations import convert_outputs_to_fp32
    from soup_cli.trainer.rewind_hf import row_losses, row_losses_from_token_losses

    _, trainer_class, _, reference_compute = patched_classes
    monkeypatch.setattr(integration, "HEAD_MODE", "streamed")
    monkeypatch.setattr(integration, "HEAD_NUMERICS", "guarded" if projection == "t4_fused" else "native")
    model, base = _tiny_peft_model(hidden_size=32 if projection == "t4_fused" else 16)
    batch = _batch()
    reference_trainer = _trainer(model)
    candidate_trainer = _trainer(model)
    original_forward = model.forward
    visits = []

    @wraps(original_forward)
    def cpu_amp_forward(*args, **kwargs):
        visits.append("outer-peft-forward")
        with torch.autocast("cpu", dtype=torch.float16):
            return original_forward(*args, **kwargs)

    # This is the same output conversion that Accelerate installs, surrounding
    # the real PEFT forward. Only the device of the autocast context differs.
    monkeypatch.setattr(model, "forward", convert_outputs_to_fp32(cpu_amp_forward))
    torch.manual_seed(911)
    expected_loss, expected = reference_compute(
        reference_trainer, model, dict(batch), return_outputs=True, num_items_in_batch=torch.tensor(9),
    )
    (expected_loss * 65536.0).backward()
    expected_gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    model.zero_grad(set_to_none=True)
    expected_rewind = row_losses(expected.logits, batch["labels"])

    monkeypatch.setattr(integration, "_supported_causal_lm", lambda *args: base)
    is_enabled = torch.is_autocast_enabled
    get_dtype = torch.get_autocast_dtype
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device="cuda": True if device == "cuda" else is_enabled(device))
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: torch.float16 if device == "cuda" else get_dtype(device))
    monkeypatch.setattr(integration, "HEAD_PROJECTION", projection)
    monkeypatch.setattr(integration, "CHUNK_TOKENS", 3)
    monkeypatch.setattr(integration, "SPARSE_BACKWARD", True)
    fused_trace = None
    if projection == "t4_fused":
        from .test_streamed_head import _install_cpu_fused_substitute

        fused_trace = _install_cpu_fused_substitute(monkeypatch)
    torch.manual_seed(911)
    loss, outputs = trainer_class.compute_loss(
        candidate_trainer, model, dict(batch), return_outputs=True, num_items_in_batch=torch.tensor(9),
    )
    (loss * 65536.0).backward()

    assert visits == ["outer-peft-forward", "outer-peft-forward"]
    assert outputs.logits is None
    assert integration._HEAD_CALL.get() is None
    torch.testing.assert_close(loss, expected_loss, rtol=3e-6, atol=3e-6)
    for name, value in model.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(value.grad, expected_gradients[name], rtol=0.008, atol=0.015)
    for name in ("entropy", "mean_token_accuracy", "num_tokens"):
        assert candidate_trainer._metrics["train"][name] == pytest.approx(reference_trainer._metrics["train"][name], rel=3e-6, abs=3e-6)
    recorded = getattr(candidate_trainer, integration._REWIND_STEP)
    actual_rewind = row_losses_from_token_losses(recorded.token_loss, batch["labels"])
    assert actual_rewind[1] == expected_rewind[1]
    assert actual_rewind[0] == pytest.approx(expected_rewind[0], rel=3e-6, abs=3e-6)
    if fused_trace is not None:
        assert len(fused_trace["statistics"]) == 1
        assert fused_trace["statistics"][0]["hidden"].shape == (10, 32)
        assert len(fused_trace["gradient"]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical CUDA device")
def test_fused_capability_guard_rejects_before_decoder_cuda_only(monkeypatch, patched_classes):
    monkeypatch.setattr(integration, "HEAD_MODE", "streamed")
    monkeypatch.setattr(integration, "HEAD_NUMERICS", "guarded")
    model, base = _tiny_peft_model(hidden_size=16)
    model.cuda()
    inputs = {name: value.cuda() for name, value in _batch().items()}
    trainer = _trainer(model)
    monkeypatch.setattr(integration, "HEAD_PROJECTION", "torch")
    assert integration._supported_causal_lm(trainer, model, inputs, type(base)) is base
    monkeypatch.setattr(integration, "HEAD_PROJECTION", "t4_fused")
    assert integration._supported_causal_lm(trainer, model, inputs, type(base)) is None

    model, base = _tiny_peft_model(hidden_size=32)
    model.cuda()
    trainer = _trainer(model)
    assert integration._supported_causal_lm(trainer, model, inputs, type(base)) is base
    weight = base.lm_head.weight
    unaligned = torch.empty(weight.numel() + 1, device="cuda", dtype=torch.float16)[1:].view_as(weight)
    base.lm_head.weight = torch.nn.Parameter(unaligned, requires_grad=False)
    assert base.lm_head.weight.is_contiguous() and base.lm_head.weight.data_ptr() % 4 == 2
    assert integration._supported_causal_lm(trainer, model, inputs, type(base)) is None


# Exact metric-packet regressions from the supplied native control candidate.
import math
import struct
from torch.utils._python_dispatch import TorchDispatchMode


class _TransferTrace(TorchDispatchMode):
    """CPU dispatcher proxy, or actual dispatch counts when run on CUDA.

    CPU _local_scalar_dense events are sites that would require a host read on
    CUDA.  They are not observed CUDA synchronization events or GPU timings.
    """

    def __init__(self):
        self.scalar_reads = []
        self.host_packets = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = function(*args, **kwargs)
        if function is torch.ops.aten._local_scalar_dense.default:
            self.scalar_reads.append((str(args[0].dtype), str(args[0].device)))
        elif function is torch.ops.aten._to_copy.default:
            destination = kwargs.get("device")
            if destination is not None and torch.device(destination).type == "cpu":
                self.host_packets.append(args[0].numel() * args[0].element_size())
        return result


def _metric_case(device="cpu", all_ignored=False, empty_attention=False):
    labels = torch.tensor([
        [-100, 2, 3, 4, -100], [-100, 1, -100, -100, -100],
        [-100, -100, -100, -100, -100], [-100, 9, 8, -100, -100],
    ], device=device)
    if all_ignored:
        labels.fill_(-100)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], device=device)
    if empty_attention:
        mask.zero_()
    result = SimpleNamespace(
        entropy=torch.tensor([[1.1, 2.2, 3.3, 4.4, 10.], [0.2, 1.3, 2., 3., 4.],
                              [2., 3., 4., 5., 6.], [0.1, 0.3, 0.5, 0.7, 0.9]],
                             device=device, dtype=torch.float32),
        predictions=torch.tensor([[2, 3, 0, 5, 0], [1, 8, 7, 6, 5],
                                  [1, 2, 3, 4, 5], [9, 0, 0, 0, 0]], device=device),
    )
    return {"labels": labels, "attention_mask": mask}, result


def _assert_python_values_exact(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert struct.pack("<d", actual) == struct.pack("<d", expected)
    else:
        assert actual == expected


def _assert_metrics_exact(actual, expected):
    assert actual._metrics["train"].keys() == expected._metrics["train"].keys()
    for key in expected._metrics["train"]:
        assert len(actual._metrics["train"][key]) == len(expected._metrics["train"][key])
        for a, e in zip(actual._metrics["train"][key], expected._metrics["train"][key]):
            _assert_python_values_exact(a, e)
    _assert_python_values_exact(actual._total_train_tokens, expected._total_train_tokens)


@pytest.mark.parametrize("all_ignored,empty_attention", [(False, False), (True, False), (True, True)])
def test_metric_packet_exact_values_integer_accumulation_and_dispatch(monkeypatch, all_ignored, empty_attention):
    monkeypatch.setattr(integration, "PACKED_METRICS", True)
    inputs, result = _metric_case(all_ignored=all_ignored, empty_attention=empty_attention)
    reference, candidate = _trainer(), _trainer()
    reference._total_train_tokens = candidate._total_train_tokens = 2**63 + 7
    before = torch.get_rng_state().clone()
    native_trace, packed_trace = _TransferTrace(), _TransferTrace()
    with native_trace:
        integration._log_statistics_unpacked(reference, inputs, result)
    with packed_trace:
        integration._log_statistics(candidate, inputs, result)
    _assert_metrics_exact(candidate, reference)
    assert torch.equal(before, torch.get_rng_state())
    assert len(native_trace.scalar_reads) == (3 if all_ignored else 4)
    assert native_trace.host_packets == []
    assert packed_trace.scalar_reads == []
    assert packed_trace.host_packets == [16]


def test_metric_packet_keeps_scalar_gathers_at_dataloader_remainder(monkeypatch):
    monkeypatch.setattr(integration, "PACKED_METRICS", True)
    from accelerate import Accelerator

    class RemainderAccelerator(_LocalAccelerator):
        gradient_state = SimpleNamespace(end_of_dataloader=True, remainder=1)

        def __init__(self):
            self.calls = []

        @staticmethod
        def gather(value):
            return value

        def gather_for_metrics(self, value):
            self.calls.append((value.shape, value.dtype))
            return Accelerator.gather_for_metrics(self, value)

    inputs, result = _metric_case()
    reference, candidate = _trainer(), _trainer()
    reference.accelerator, candidate.accelerator = RemainderAccelerator(), RemainderAccelerator()
    integration._log_statistics_unpacked(reference, inputs, result)
    integration._log_statistics(candidate, inputs, result)
    _assert_metrics_exact(candidate, reference)
    assert candidate.accelerator.calls == reference.accelerator.calls == [
        (torch.Size([]), torch.float32), (torch.Size([]), torch.int64),
        (torch.Size([]), torch.int64), (torch.Size([]), torch.int64),
    ]
    # This is why the packet must be built AFTER all original scalar gathers.
    assert Accelerator.gather_for_metrics(candidate.accelerator, torch.arange(3)).shape == (1,)


@pytest.mark.parametrize("unsupported", ["processes", "dtype", "runtime", "gather_dtype"])
def test_metric_packet_unsupported_paths_preserve_original_values(monkeypatch, unsupported):
    monkeypatch.setattr(integration, "PACKED_METRICS", True)
    from . import metric_transfer

    inputs, result = _metric_case()
    reference, candidate = _trainer(), _trainer()
    if unsupported == "processes":
        reference.accelerator.num_processes = candidate.accelerator.num_processes = 2
    elif unsupported == "dtype":
        result.entropy = result.entropy.double()
    elif unsupported == "runtime":
        monkeypatch.setattr(metric_transfer, "metric_runtime_supported", lambda: False)
    else:
        def changed(value):
            return value.double() if value.is_floating_point() else value
        reference.accelerator.gather_for_metrics = candidate.accelerator.gather_for_metrics = changed
    integration._log_statistics_unpacked(reference, inputs, result)
    trace = _TransferTrace()
    with trace:
        integration._log_statistics(candidate, inputs, result)
    _assert_metrics_exact(candidate, reference)
    assert trace.host_packets == []
    assert len(trace.scalar_reads) == 4
