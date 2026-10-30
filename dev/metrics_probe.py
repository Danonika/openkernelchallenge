"""GPU correctness probe for solution.metrics, separate from measured training.

Run from the repository root with ``python dev/metrics_probe.py``.  Add
``--check-accuracy`` to validate and install the optional accuracy rewrite
within this development process.  The probe
uses the installed TRL helper and actual SFTTrainer.compute_loss as references.
It never loads the task model, changes a training setting, or fetches data.
"""

from __future__ import annotations

import collections
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton

from solution import metrics


def _compare(actual, expected, *, name, atol=3e-5, rtol=2e-5):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: shape/dtype differs: {actual.shape}/{actual.dtype}, "
            f"{expected.shape}/{expected.dtype}"
        )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    error = (actual[finite] - expected[finite]).abs()
    return {
        "name": name,
        "shape": list(actual.shape),
        "max_abs_error": error.max().item() if error.numel() else 0.0,
        "nan_rows": torch.isnan(actual).sum().item(),
    }


def _probe_actual_trainer(original_entropy, entropy_wrapper):
    """Compare actual pinned method with its one-expression runtime rewrite."""
    import trl.trainer.sft_trainer as sft

    original = sft.SFTTrainer.compute_loss
    original = getattr(original, "_soup_accuracy_original", original)
    rewritten = metrics._accuracy_without_copy(original)
    if rewritten is None:
        raise AssertionError("Pinned compute_loss did not match guarded accuracy rewrite")
    if rewritten.__closure__ is not original.__closure__:
        raise AssertionError("The rewrite did not retain the actual __class__ closure")
    if rewritten.__globals__ is not original.__globals__:
        raise AssertionError("The rewrite changed the original function globals")

    base = sft.SFTTrainer.__mro__[1]
    base_owned_compute_loss = "compute_loss" in base.__dict__
    base_compute_loss = base.compute_loss
    saved_entropy = sft.entropy_from_logits
    active_output = None
    loss = torch.tensor(0.75, device="cuda", requires_grad=True)

    def fake_parent(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        assert return_outputs and inputs["use_cache"] is False
        return loss, active_output

    class GatherIdentity:
        @staticmethod
        def gather_for_metrics(value):
            return value

    def trainer(model, virtual):
        instance = object.__new__(sft.SFTTrainer)
        instance.model = model
        instance.args = types.SimpleNamespace(use_liger_kernel=False)
        instance.num_virtual_tokens = virtual
        instance.accelerator = GatherIdentity()
        instance._metrics = collections.defaultdict(lambda: collections.defaultdict(list))
        instance._total_train_tokens = 17
        instance.aux_loss_enabled = True
        return instance

    records = []
    try:
        base.compute_loss = fake_parent
        # Cover ordinary batches, virtual tokens, pre-shifted labels, prefix
        # tuning, empty accuracy masks, eval, and the position_ids metric path.
        cases = [
            ("ordinary", 0, False, False, False, True),
            ("prompt_virtual", 2, False, False, False, True),
            ("prefix_virtual", 2, True, False, False, True),
            ("pre_shifted", 0, False, True, False, True),
            ("all_labels_ignored", 0, False, False, True, True),
            ("eval_position_ids", 0, False, False, False, False),
        ]
        for name, virtual, prefix, preshifted, ignore_all, training in cases:
            batch, sequence, vocab = 3, 9, 1031
            extra = virtual if virtual and not prefix else 0
            logits = torch.randn(batch, sequence + extra, vocab, device="cuda")
            # Exact ties must keep native argmax's first-index behavior.
            logits[:, 1, :] = 0
            logits[:, 2, 17] = 20
            logits[:, 2, 19] = 20
            logits.requires_grad_(True)
            active_output = types.SimpleNamespace(
                logits=logits, aux_loss=torch.tensor(0.125, device="cuda")
            )
            model = types.SimpleNamespace(
                training=training,
                active_adapter="default",
                peft_config={"default": types.SimpleNamespace(
                    peft_type=sft.PeftType.PREFIX_TUNING if prefix else "PROMPT_TUNING"
                )},
            )
            labels = torch.randint(0, vocab, (batch, sequence), device="cuda")
            with torch.no_grad():
                if preshifted:
                    labels.copy_(logits.argmax(-1))
                else:
                    labels[:, 1:] = logits[:, extra:-1].argmax(-1)
            labels[0, :3] = -100
            labels[1, -2:] = -100
            if ignore_all:
                labels.fill_(-100)
            attention = torch.ones((batch, sequence), dtype=torch.int64, device="cuda")
            attention[1, -2:] = 0
            inputs = {"shift_labels" if preshifted else "labels": labels}
            if training:
                inputs["attention_mask"] = attention
            else:
                inputs["position_ids"] = torch.arange(sequence, device="cuda")[None, :]
                inputs["_prediction_loss_only"] = False

            reference_trainer = trainer(model, virtual)
            candidate_trainer = trainer(model, virtual)
            sft.entropy_from_logits = original_entropy
            baseline_return = original(reference_trainer, model, dict(inputs), return_outputs=True)
            sft.entropy_from_logits = entropy_wrapper
            candidate_return = rewritten(candidate_trainer, model, dict(inputs), return_outputs=True)
            if baseline_return[0] is not candidate_return[0] or baseline_return[1] is not candidate_return[1]:
                raise AssertionError(f"{name}: loss/output identity changed")
            mode = "train" if training else "eval"
            expected = dict(reference_trainer._metrics[mode])
            actual = dict(candidate_trainer._metrics[mode])
            if expected.keys() != actual.keys():
                raise AssertionError(f"{name}: metric keys changed")
            for key in expected:
                if key == "entropy":
                    if not math.isclose(expected[key][0], actual[key][0], rel_tol=2e-5, abs_tol=3e-5):
                        raise AssertionError(f"{name}: aggregated entropy changed")
                elif expected[key] != actual[key]:
                    raise AssertionError(f"{name}: {key} changed")
            records.append({"name": name, "metrics": actual})
    finally:
        if base_owned_compute_loss:
            base.compute_loss = base_compute_loss
        else:
            delattr(base, "compute_loss")
        sft.entropy_from_logits = saved_entropy
    return records


def run(*, full_shape=True, check_accuracy=False):
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires a CUDA GPU")
    import trl
    import trl.trainer.utils as utils
    import trl.trainer.sft_trainer as sft

    original = getattr(utils.entropy_from_logits, "_soup_entropy_original", utils.entropy_from_logits)
    wrapper = metrics._make_entropy_wrapper(original)
    torch.manual_seed(72941)
    records = []
    with torch.no_grad():
        for shape, scale, offset in [
            ((151936,), 1.0, 0.0),
            ((2, 7, 151936), 0.1, 0.0),
            ((2, 7, 151936), 1.0, 0.0),
            ((2, 7, 151936), 8.0, 0.0),
            ((2, 7, 151936), 1.0, 10000.0),
            ((2, 3, 4097), 1.0, 0.0),
            ((2, 3, 1025), 1.0, 0.0),
            ((2, 3, 1), 1.0, 0.0),
        ]:
            x = (torch.randn(shape, device="cuda") * scale).half().float() + offset
            x.requires_grad_(True)  # Normal SFT input still has this flag under no_grad.
            if not metrics._can_use_entropy(x, 128):
                raise AssertionError("Normal no-grad SFT metric path unexpectedly fell back")
            records.append(_compare(
                wrapper(x), original(x), name=f"finite_{shape}_{scale}_{offset}"
            ))
            del x

        x = torch.zeros((6, 151936), device="cuda")
        x[1, 123] = 80.0
        x[2, 11] = float("nan")
        x[3, 4096] = float("inf")
        x[4, -1] = -float("inf")
        x[5, :] = -float("inf")
        records.append(_compare(wrapper(x), original(x), name="ties_peak_nonfinite"))
        del x

        x = torch.randn((2, 5, 1025), device="cuda")
        strided = x[:, :-1, :]
        assert not metrics._can_use_entropy(strided, 128)
        records.append(_compare(wrapper(strided), original(strided), name="strided_fallback"))
        half = x.half()
        assert not metrics._can_use_entropy(half, 128)
        records.append(_compare(wrapper(half), original(half), name="half_fallback", atol=0, rtol=0))
        cpu = torch.randn(2, 31)
        records.append(_compare(wrapper(cpu), original(cpu), name="cpu_fallback", atol=0, rtol=0))
        del x, strided, half, cpu

        if full_shape:
            # Approximately 1.16 GiB of FP32 logits, matching the maximum
            # prepared B4,T512,V151936 shape; no task data is involved.
            x = torch.randn((4, 512, 151936), device="cuda", dtype=torch.float16).float()
            records.append(_compare(wrapper(x), original(x), name="full_training_shape"))
            del x

    # The differentiable fallback must preserve both its value and gradient.
    x1 = torch.randn(2, 1031, device="cuda", requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    assert not metrics._can_use_entropy(x1, 128)
    y1, y2 = wrapper(x1), original(x2)
    records.append(_compare(y1, y2, name="gradient_fallback_value", atol=0, rtol=0))
    g1 = torch.autograd.grad(y1.sum(), x1)[0]
    g2 = torch.autograd.grad(y2.sum(), x2)[0]
    records.append(_compare(g1, g2, name="gradient_fallback_gradient", atol=0, rtol=0))
    del x1, x2, y1, y2, g1, g2

    accuracy_cases = _probe_actual_trainer(original, wrapper) if check_accuracy else []
    installed = metrics.install(accuracy_no_copy=check_accuracy)
    if utils.entropy_from_logits is not sft.entropy_from_logits:
        raise AssertionError("SFT's imported entropy binding was not patched")
    if check_accuracy and not installed["accuracy_no_copy"]:
        raise AssertionError("Validated accuracy rewrite did not install")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "ok": True,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "trl": trl.__version__,
        "gpu": torch.cuda.get_device_name(),
        "entropy_cases": records,
        "actual_trainer_cases": accuracy_cases,
        "installed": installed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-accuracy", action="store_true")
    parser.add_argument("--skip-full-shape", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(full_shape=not args.skip_full_shape,
                         check_accuracy=args.check_accuracy), separators=(",", ":")))
