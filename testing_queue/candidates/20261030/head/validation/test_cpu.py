"""Focused CPU-only fixtures. Never invokes apply(), CUDA, model download or training."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solution.native_head import NativeHeadLoss, Telemetry
from solution.install import install_pinned
from transformers.loss.loss_utils import ForCausalLMLoss

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
GENERATOR = torch.Generator().manual_seed(20261030)
RECORDS = []


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(torch.isposinf(a), torch.isposinf(b))
    assert torch.equal(torch.isneginf(a), torch.isneginf(b))
    finite = torch.isfinite(a) & torch.isfinite(b)
    assert torch.equal(a[finite], b[finite])
    zero = (a == 0) & (b == 0)
    assert torch.equal(torch.signbit(a[zero]), torch.signbit(b[zero]))


def compare(shape=(2, 3, 23), scale=3.0, denominator=11, anomaly=None,
            all_ignored=False, fp32=False, expected_replay=True, extra=None):
    z = torch.randn(shape, generator=GENERATOR).half()
    if fp32:
        z = z.float()
    labels = torch.arange(shape[0] * shape[1]).reshape(shape[:2]).remainder(shape[-1])
    labels[:, 0] = -100
    if all_ignored:
        labels[:] = -100
    if anomaly is not None:
        z[0, -1, 0] = anomaly
    baseline, candidate = z.detach().clone().requires_grad_(), z.detach().clone().requires_grad_()
    metrics = Telemetry(stream=False)
    wrapper = NativeHeadLoss(ForCausalLMLoss, metrics, expected_vocabulary=shape[-1], devices=("cpu",))
    rng = torch.get_rng_state().clone()
    extra = extra or {}
    expected = ForCausalLMLoss(baseline, labels, shape[-1], denominator, **extra)
    actual = wrapper(candidate, labels, shape[-1], denominator, **extra)
    upstream = torch.tensor(scale, dtype=torch.float32)
    eg, = torch.autograd.grad(expected, baseline, grad_outputs=upstream)
    ag, = torch.autograd.grad(actual, candidate, grad_outputs=upstream)
    exact(actual, expected)
    exact(ag, eg)
    assert torch.equal(rng, torch.get_rng_state())
    if expected_replay:
        assert metrics.counts == {"native_sum_forward_enter": 1, "native_sum_forward_return": 1,
                                  "native_sum_backward_enter": 1, "native_sum_backward_return": 1}
    else:
        assert metrics.counts == {"native_fallback": 1}
    return {"shape": shape, "scale": str(scale), "denominator": str(denominator),
            "finite_exact_nonfinite_classes_and_signed_zero": True,
            "rng_unchanged": True, "telemetry": metrics.snapshot()}


def fallback_identity():
    captured = []
    def native(**kwargs):
        captured.append(kwargs)
        return "unchanged-native-output"
    z = torch.randn((2, 3, 7), generator=GENERATOR).half().requires_grad_()
    y = torch.zeros((2, 3), dtype=torch.long)
    wrapper = NativeHeadLoss(native, Telemetry(False), expected_vocabulary=7, devices=("cpu",))
    shifted = y.clone()
    assert wrapper(z, y, 7, 13, shift_labels=shifted, arbitrary=True) == "unchanged-native-output"
    assert captured[0]["logits"] is z and captured[0]["labels"] is y
    assert captured[0]["shift_labels"] is shifted and captured[0]["arbitrary"] is True
    return {"same_already_produced_logits_labels_extra_kwargs": True}


def invalid_labels():
    z = torch.randn((2, 3, 7), generator=GENERATOR).half().requires_grad_()
    y = torch.zeros((2, 3), dtype=torch.long)
    y[0, 1] = 7
    errors = []
    for fn in (ForCausalLMLoss, NativeHeadLoss(ForCausalLMLoss, Telemetry(False),
                                            expected_vocabulary=7, devices=("cpu",))):
        try:
            fn(z, y, 7, 13)
        except Exception as exc:
            errors.append(type(exc).__name__)
    assert errors == ["IndexError", "IndexError"]
    return {"native_error_types": errors}


def saved_storage():
    z = torch.randn((2, 3, 23), generator=GENERATOR).half().requires_grad_()
    y = torch.zeros((2, 3), dtype=torch.long)
    results = {}
    for name, fn in [("native", ForCausalLMLoss),
                     ("candidate", NativeHeadLoss(ForCausalLMLoss, Telemetry(False),
                                                 expected_vocabulary=23, devices=("cpu",)))]:
        saved = {}
        def pack(tensor):
            storage = tensor.untyped_storage()
            saved[storage.data_ptr()] = {"bytes": storage.nbytes(), "dtype": str(tensor.dtype)}
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            loss = fn(z, y, 23, 11)
        results[name] = {"unique_saved_bytes": sum(x["bytes"] for x in saved.values()),
                         "saved_metadata": list(saved.values())}
        del loss
    assert results["candidate"]["unique_saved_bytes"] < results["native"]["unique_saved_bytes"]
    return {"inventory": results, "scope": "Retained graph state only, NOT allocator/process peak"}


def version_checks():
    errors = []
    for fn in (ForCausalLMLoss, NativeHeadLoss(ForCausalLMLoss, Telemetry(False),
                                            expected_vocabulary=7, devices=("cpu",))):
        z = torch.randn((2, 3, 7), generator=GENERATOR).half().requires_grad_()
        y, denominator = torch.zeros((2, 3), dtype=torch.long), torch.tensor(9)
        loss = fn(z, y, 7, denominator)
        denominator.add_(1)
        try:
            loss.backward()
        except RuntimeError as exc:
            errors.append("version" in str(exc) or "inplace" in str(exc))
    assert errors == [True, True]
    return {"native_and_candidate_denominator_mutation_rejected": True}


def qwen_cpu_fixture():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    # Fixed task vocabulary but tiny synthetic decoder; no PEFT/NF4/Soup claim.
    config = Qwen2Config(vocab_size=151936, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, attention_dropout=0.1,
                         tie_word_embeddings=True, use_cache=False)
    config._attn_implementation = "eager"
    baseline = Qwen2ForCausalLM(config).half().train()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    original_init = Qwen2ForCausalLM.__init__
    telemetry = Telemetry(False)
    handle = install_pinned(telemetry)
    try:
        candidate = Qwen2ForCausalLM(copy.deepcopy(config)).half().train()
        candidate.load_state_dict(baseline.state_dict())
        for parameter in candidate.parameters():
            parameter.requires_grad_(False)
        assert isinstance(candidate.loss_function, NativeHeadLoss)
        # Explicit CPU fixture injection; production apply does not select CPU.
        candidate.loss_function.devices = ("cpu",)
        labels = torch.tensor([[-100, 3, 4], [-100, 8, 9]])
        inputs = torch.randn((2, 3, 8), generator=GENERATOR).half()
        names = list(candidate.state_dict())
        head_calls = [0, 0]
        def hook(index):
            def count(*_):
                head_calls[index] += 1
            return count
        handles = [baseline.lm_head.register_forward_hook(hook(0)),
                   candidate.lm_head.register_forward_hook(hook(1))]
        outputs, gradients, rng_after = [], [], []
        for model in (baseline, candidate):
            h = inputs.detach().clone().requires_grad_()
            torch.manual_seed(14431)
            output = model(inputs_embeds=h, labels=labels, num_items_in_batch=11,
                           output_hidden_states=False, output_attentions=False,
                           use_cache=False)
            gradient, = torch.autograd.grad(output.loss, h, grad_outputs=torch.tensor(3.0))
            outputs.append(output)
            gradients.append(gradient)
            rng_after.append(torch.get_rng_state().clone())
        exact(outputs[0].loss, outputs[1].loss)
        exact(outputs[0].logits, outputs[1].logits)
        exact(gradients[0], gradients[1])
        assert list(outputs[0]) == list(outputs[1]) and outputs[1]["logits"] is outputs[1].logits
        assert torch.equal(rng_after[0], rng_after[1])
        assert head_calls == [1, 1] and list(candidate.state_dict()) == names
        assert telemetry.counts["native_sum_backward_return"] == 1
        candidate.eval()
        with torch.no_grad():
            candidate(inputs_embeds=inputs, labels=labels, num_items_in_batch=11)
        assert telemetry.reasons["owner_or_eval"] == 1
        for item in handles:
            item.remove()
        restored = handle.restore()
        assert restored["constructor"] == "restored" and restored["restored_models"] == 1
        assert candidate.loss_function is ForCausalLMLoss
        assert Qwen2ForCausalLM.__init__ is original_init
        return {"tiny_real_qwen_forward_loss_gradient_output_mapping_rng_exact": True,
                "one_head_forward_each": head_calls, "eval_fallback_and_restore": True,
                "scope": "CPU FP16 tiny Qwen only; no Accelerate/PEFT/NF4/TRL/Rewind or T4"}
    finally:
        handle.restore()


def run(name, fn):
    try:
        RECORDS.append({"name": name, "status": "PASS", **fn()})
    except Exception as exc:
        RECORDS.append({"name": name, "status": "FAIL", "error": repr(exc),
                        "traceback": traceback.format_exc()})


if __name__ == "__main__":
    for name, args in [
        ("finite_integer_D", {}), ("scalar_tensor_D", {"denominator": torch.tensor(11)}),
        ("large_incoming_scale", {"scale": 65536.0}),
        ("full_vocab_short_T", {"shape": (4, 3, 151936)}),
        ("T512_small_vocab", {"shape": (4, 512, 19)}),
        ("all_ignored", {"all_ignored": True}),
        ("T1_all_final_ignored", {"shape": (4, 1, 23)}),
        ("ignored_final_nan", {"anomaly": float("nan")}),
        ("ignored_final_posinf", {"anomaly": float("inf")}),
        ("ignored_final_neginf", {"anomaly": -float("inf")}),
        ("connected_zero_nan", {"scale": 0.0, "anomaly": float("nan")}),
        ("incoming_inf", {"scale": float("inf")}),
        ("incoming_nan", {"scale": float("nan")}),
        ("zero_D", {"denominator": torch.tensor(0)}),
        ("negative_D", {"denominator": -11}),
        ("native_mean_fallback", {"denominator": None, "expected_replay": False}),
        ("float_D_fallback", {"denominator": 11.0, "expected_replay": False}),
        ("fp32_fallback", {"fp32": True, "expected_replay": False}),
        ("standard_output_controls", {"extra": {"output_attentions": False,
                                                  "output_hidden_states": False}}),
    ]:
        run(name, lambda args=args: compare(**args))
    for name, fn in [("fallback_identity", fallback_identity), ("invalid_labels", invalid_labels),
                     ("saved_storage_inventory", saved_storage),
                     ("denominator_version_checks", version_checks),
                     ("tiny_actual_qwen_cpu_integration", qwen_cpu_fixture)]:
        run(name, fn)
    result = {"scope": "CPU fixtures only; T4 UNTESTED", "torch": torch.__version__,
              "python": sys.version, "records": RECORDS,
              "pass": sum(x["status"] == "PASS" for x in RECORDS),
              "fail": sum(x["status"] == "FAIL" for x in RECORDS),
              "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "solution").glob("*.py"))}}
    (ROOT / "validation" / "cpu_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("scope", "pass", "fail")}))
    for item in RECORDS:
        if item["status"] == "FAIL":
            print(item["name"], item["traceback"])
    raise SystemExit(bool(result["fail"]))
