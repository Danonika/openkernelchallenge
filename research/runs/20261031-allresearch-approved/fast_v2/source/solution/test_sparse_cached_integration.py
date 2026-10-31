"""Real tiny Qwen/PEFT integration for the optional sparse cached head.

These are CPU control-flow and numerical tests, including when collected on
the physical service. They use actual native CPU FP16 autocast, the production
sparse helper and native CE consumer, real PEFT/Trainer calls, and Accelerate's
output conversion. Only CUDA admission/autocast metadata is adapted inside the
test. No fused CUDA consumer, CUDA timing, or official training quality is
established here; the separate physical full-vocabulary/full-factory gates
cover those requirements.
"""

from contextlib import nullcontext
from functools import wraps
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from . import integration, sparse_cached_head
from .test_integration import _batch, _trainer, patched_classes


def _model():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(751)
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=67, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, pad_token_id=0, tie_word_embeddings=True,
        attention_dropout=0.0,
    ))
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
    ))
    # Nonzero B makes gradients of both LoRA factors meaningful on one step.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.03)
    model.train()
    return model, base


class _HeadMatmuls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if function is torch.ops.aten.mm.default:
            shapes = (tuple(args[0].shape), tuple(args[1].shape))
            # V=67 occurs only in the frozen output head of this fixture.
            if shapes[1] in ((32, 67), (67, 32)):
                self.calls.append((*shapes, result.dtype))
        return result


def _cpu_amp_model(monkeypatch, model, visits):
    from accelerate.utils.operations import convert_outputs_to_fp32

    original = model.forward

    @wraps(original)
    def autocast_forward(*args, **kwargs):
        visits.append("real_peft_forward")
        with torch.autocast("cpu", dtype=torch.float16):
            return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", convert_outputs_to_fp32(autocast_forward))


def _admit_cpu_private_call(monkeypatch, base):
    # Production still rejects CPU. The wrapper's autocast check reflects the
    # actual CPU context, rather than reporting autocast active unconditionally.
    monkeypatch.setattr(integration, "_supported_causal_lm", lambda *args: base)
    is_enabled, get_dtype = torch.is_autocast_enabled, torch.get_autocast_dtype
    monkeypatch.setattr(torch, "is_autocast_enabled",
        lambda device="cuda": is_enabled("cpu" if device == "cuda" else device))
    monkeypatch.setattr(torch, "get_autocast_dtype",
        lambda device: get_dtype("cpu" if device == "cuda" else device))


def _assert_gradient_norms(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    a, r = actual.float(), expected.float()
    error = a - r
    reference_norm = torch.linalg.vector_norm(r).item()
    reference_max = r.abs().max().item()
    if reference_norm == 0:
        assert torch.equal(a, r)
        return {"relative_l2": 0.0, "relative_max": 0.0}
    relative_l2 = torch.linalg.vector_norm(error).item() / reference_norm
    relative_max = error.abs().max().item() / reference_max
    # Predeclared FP16 component bounds, matching the head diagnostic. They do
    # not replace the independent official aggregate training-curve gate.
    epsilon = torch.finfo(torch.float16).eps
    assert relative_l2 <= 2 * epsilon
    assert relative_max <= 4 * epsilon
    return {"relative_l2": relative_l2, "relative_max": relative_max}


@pytest.mark.parametrize("scale,denominator,sparse_backward", [
    (1.0, None, True), (65536.0, 2048, True), (65536.0, 2048, False),
])
def test_sparse_cached_cpu_peft_amp_metrics_rng_and_gradient_norms(
        monkeypatch, patched_classes, tmp_path, scale, denominator, sparse_backward):
    from soup_cli.trainer.rewind_hf import row_losses, row_losses_from_token_losses

    _, trainer_class, _, native_compute = patched_classes
    for name, value in {
        "HEAD_MODE": "sparse_cached", "HEAD_NUMERICS": "native",
        "HEAD_PROJECTION": "torch", "CACHED_CE_BACKWARD": "native",
        "CACHED_CE_CHUNK_TOKENS": 3, "SPARSE_BACKWARD": sparse_backward,
        "PACKED_METRICS": False, "MLP_PADDING_COMPACTION": False,
    }.items():
        monkeypatch.setattr(integration, name, value)
    model, base = _model()
    batch = _batch()
    native_trainer, candidate_trainer = _trainer(model), _trainer(model)
    visits, decoder_calls, native_head_calls = [], [], []
    _cpu_amp_model(monkeypatch, model, visits)
    decoder_hook = base.model.register_forward_pre_hook(
        lambda module, inputs: decoder_calls.append(True))
    head_hook = base.lm_head.register_forward_hook(
        lambda module, inputs, outputs: native_head_calls.append(tuple(outputs.shape)))
    try:
        torch.manual_seed(752)
        initial_rng = torch.get_rng_state().clone()
        assert not torch.is_autocast_enabled("cpu")
        expected_loss, expected_outputs = native_compute(
            native_trainer, model, dict(batch), return_outputs=True,
            num_items_in_batch=denominator)
        expected_loss.backward(torch.tensor(scale))
        expected_rng = torch.get_rng_state().clone()
        expected_gradients = {
            name: value.grad.detach().clone() for name, value in model.named_parameters()
            if value.requires_grad}
        assert expected_gradients and all("lora_" in name for name in expected_gradients)
        assert decoder_calls == [True] and native_head_calls == [(2, 5, 67)]
        expected_rewind = row_losses(expected_outputs.logits, batch["labels"])
        model.zero_grad(set_to_none=True)
        decoder_calls.clear()
        native_head_calls.clear()

        actual_helper = sparse_cached_head.sparse_cached_linear_ce
        helper_calls = []

        def observed_helper(hidden, weight, labels, **kwargs):
            result = actual_helper(hidden, weight, labels, **kwargs)
            helper_calls.append((tuple(hidden.shape), weight.dtype, dict(kwargs), result))
            return result

        monkeypatch.setattr(sparse_cached_head, "sparse_cached_linear_ce", observed_helper)
        _admit_cpu_private_call(monkeypatch, base)
        torch.set_rng_state(initial_rng)
        trace = _HeadMatmuls()
        assert not torch.is_autocast_enabled("cpu")
        with trace:
            loss, outputs = trainer_class.compute_loss(
                candidate_trainer, model, dict(batch), return_outputs=True,
                num_items_in_batch=denominator)
            loss.backward(torch.tensor(scale))
        actual_rng = torch.get_rng_state().clone()
    finally:
        decoder_hook.remove()
        head_hook.remove()

    assert visits == ["real_peft_forward", "real_peft_forward"]
    assert decoder_calls == [True]
    # The private helper directly executes the native torch.mm; it does not
    # invoke lm_head.forward or replay a forward projection in backward.
    assert native_head_calls == []
    assert trace.calls == [((10, 32), (32, 67), torch.float16),
                           ((5 if sparse_backward else 10, 67), (67, 32), torch.float16)]
    assert torch.equal(actual_rng, expected_rng)
    assert integration._HEAD_CALL.get() is None
    assert not torch.is_autocast_enabled("cpu")
    assert outputs.logits is None and "logits" not in outputs
    assert len(helper_calls) == 1
    hidden_shape, weight_dtype, kwargs, result = helper_calls[0]
    assert hidden_shape == (2, 5, 32) and weight_dtype == torch.float16
    assert kwargs == dict(num_items_in_batch=denominator, chunk_size=3,
                          projection="torch", sparse_backward=sparse_backward,
                          numerics="native", backward_mode="native")

    # The scalar sum reduction may round once differently from native NLL's
    # reduction; compact per-token statistics must retain exact native values.
    epsilon = torch.finfo(torch.float32).eps
    torch.testing.assert_close(loss, expected_loss, rtol=2 * epsilon, atol=0)
    targets = F.pad(batch["labels"], (0, 1), value=-100)[..., 1:]
    expected_lp = F.log_softmax(expected_outputs.logits, dim=-1)
    expected_token = F.nll_loss(expected_lp.reshape(-1, 67), targets.reshape(-1),
                                reduction="none").reshape_as(targets)
    torch.testing.assert_close(result.token_loss, expected_token, rtol=0, atol=0)
    torch.testing.assert_close(result.entropy, -(expected_lp.exp() * expected_lp).sum(-1),
                               rtol=0, atol=0)
    assert torch.equal(result.predictions, expected_outputs.logits.argmax(-1))
    assert dict(candidate_trainer._metrics["train"]) == dict(native_trainer._metrics["train"])
    assert candidate_trainer._total_train_tokens == native_trainer._total_train_tokens == 8
    recorded = getattr(candidate_trainer, integration._REWIND_STEP)
    assert recorded.token_loss is result.token_loss
    actual_rewind = row_losses_from_token_losses(recorded.token_loss, batch["labels"])
    assert actual_rewind == expected_rewind
    assert actual_rewind[1] == [3, 2]
    norms = {}
    for name, value in model.named_parameters():
        if value.requires_grad:
            assert value.grad is not None
            norms[name] = _assert_gradient_norms(value.grad, expected_gradients[name])
        else:
            assert value.grad is None
    (tmp_path / "sparse_cached_cpu_integration.json").write_text(json.dumps({
        "passed": True, "physical_gpu_execution": False,
        "consumer": "actual_native_cpu", "scale": scale,
        "denominator": denominator, "sparse_backward": sparse_backward,
        "decoder_calls": len(decoder_calls), "head_gemm_calls": len(trace.calls),
        "metrics_exact": True, "rewind_exact": True, "rng_exact": True,
        "lora_gradient_norm_errors": norms,
        "test_overrides": ["CPU FP16 model autocast", "private CPU device admission",
                           "CUDA autocast metadata reflects active CPU autocast",
                           "unpacked metric transfer", "no MLP padding compaction"],
    }, indent=2) + "\n")


@pytest.mark.parametrize("unsupported", [
    "backward_mode", "numerics", "projection", "compiling", "capture", "head_mode",
])
def test_sparse_cached_predecoder_guards_do_not_consume_rng(monkeypatch, unsupported):
    class CudaMetadataOnlyTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    class UnreachableDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(return_dict=True)
            self.unwraps = 0
            self.forwards = 0

        def get_base_model(self):
            self.unwraps += 1
            return self

        def forward(self, **kwargs):
            self.forwards += 1
            raise AssertionError("a capability check must not invoke the decoder")

    # CPU metadata-only tensors deliberately exercise guards that follow the
    # CUDA requirement. They never enter a real CUDA context or a CUDA kernel.
    tensor = torch.ones(1, 2, dtype=torch.long).as_subclass(CudaMetadataOnlyTensor)
    inputs = dict(input_ids=tensor, labels=tensor, attention_mask=tensor)
    model = UnreachableDecoder()
    trainer = _trainer(model)
    for name, value in {"HEAD_MODE": "sparse_cached", "HEAD_NUMERICS": "native",
                        "HEAD_PROJECTION": "torch", "CACHED_CE_BACKWARD": "native"}.items():
        monkeypatch.setattr(integration, name, value)
    contexts = []

    def metadata_device_context(device):
        contexts.append(str(device))
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "device", metadata_device_context)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: unsupported == "capture")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: unsupported == "compiling")
    if unsupported == "backward_mode":
        monkeypatch.setattr(integration, "CACHED_CE_BACKWARD", "unsupported")
    elif unsupported == "numerics":
        monkeypatch.setattr(integration, "HEAD_NUMERICS", "unsupported")
    elif unsupported == "projection":
        monkeypatch.setattr(integration, "HEAD_PROJECTION", "t4_fused")
    elif unsupported == "head_mode":
        monkeypatch.setattr(integration, "HEAD_MODE", "unsupported")
    before = torch.get_rng_state().clone()
    assert integration._supported_causal_lm(trainer, model, inputs, UnreachableDecoder) is None
    assert torch.equal(torch.get_rng_state(), before)
    assert model.forwards == 0
    assert model.unwraps == int(unsupported == "head_mode")
    assert contexts == (["cpu"] if unsupported == "capture" else [])
    assert dict(trainer._metrics["train"]) == {} and trainer._total_train_tokens == 0


@pytest.mark.parametrize("logits_to_keep", [1, torch.tensor([0])])
def test_sparse_cached_private_sequence_guard_precedes_decoder(logits_to_keep):
    class UnreachableDecoder:
        def model(self, **kwargs):
            raise AssertionError("unsupported token selection must precede the decoder")

    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="full token sequence"):
        integration._forward_sparse_cached(UnreachableDecoder(), logits_to_keep=logits_to_keep)
    assert torch.equal(torch.get_rng_state(), before)
