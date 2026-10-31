"""Physical T4 joint L1e + scale/add validation at the task's q/v widths.

These tests use real PEFT 0.20.0 and native nested-NF4 bitsandbytes modules,
real FP16 CUDA autocast and GradScaler, and the installed production guards.
No GEMM, dropout mask, device predicate, or projection result is substituted.
The spies delegate native Dropout and original Triton kernels exactly once.

Run in a fresh process before solution.apply. The full selected-profile gate
separately establishes composition with NF4 forward caching, P1, compact MLP,
rotary, the head, optimizer, and Soup/Rewind. This targeted test isolates the
two LoRA epilogues and is neither a training benchmark nor a speed claim.
"""

from collections import Counter
import json
import os

import pytest
import torch


_PHYSICAL_CUDA = torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"
_INPUT_WIDTH, _BATCH, _LENGTH = 1536, 4, 128


def _run(layer, values, coefficient, *, scale, input_grad):
    x = values.detach().clone().requires_grad_(input_grad)
    parameters = {name: parameter for name, parameter in layer.named_parameters()
                  if parameter.requires_grad}
    assert set(parameters) == {"lora_A.default.weight", "lora_B.default.weight"}
    assert all(parameter.dtype == torch.float32 for parameter in parameters.values())
    variables = ({"input": x} if input_grad else {}) | parameters
    torch.manual_seed(923)
    torch.cuda.manual_seed_all(923)
    scaler = torch.amp.GradScaler("cuda", init_scale=scale)
    assert scaler.is_enabled()
    with torch.autocast("cuda", dtype=torch.float16):
        output = layer(x)
        loss = (output * coefficient).mean()
    gradients = torch.autograd.grad(scaler.scale(loss), tuple(variables.values()))
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    for name, gradient in zip(variables, gradients):
        assert gradient.dtype == torch.float32, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, f"vacuous zero gradient: {name}"
    torch.cuda.synchronize()
    return {"output": output.detach().cpu(), "loss": loss.detach().cpu(),
            "gradients": {name: gradient.detach().cpu() for name, gradient in zip(variables, gradients)},
            "cpu_rng": torch.random.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state().clone(),
            "scaler": scaler.state_dict()}


@pytest.mark.skipif(not _PHYSICAL_CUDA, reason="requires physical CUDA, not CPU or Triton interpreter")
@pytest.mark.parametrize("output_width,scale,input_grad", [
    (1536, 1.0, True), (1536, 65536.0, True),
    (256, 1.0, True), (256, 65536.0, True),
    (1536, 65536.0, False), (256, 65536.0, False),
], ids=["q_unscaled", "q_scaled", "v_unscaled", "v_scaled", "q_first_layer", "v_first_layer"])
def test_physical_t4_joint_lora_epilogues_native_nf4(
        monkeypatch, output_width, scale, input_grad, record_property):
    import bitsandbytes as bnb
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear
    from peft.tuners.lora.bnb import Linear4bit
    from . import lora_dropout as l1e, lora_epilogue as epilogue

    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("this production-composition gate requires an SM75 T4")
    assert torch.cuda.device_count() == 1
    assert not getattr(Linear4bit.forward, "_soup_candidate_lora_epilogue", False)
    assert not getattr(Linear4bit.forward, "_soup_candidate_lora_dropout", False)
    assert not getattr(torch.nn.Dropout.forward, "_soup_candidate_lora_dropout", False)
    # Snapshot class owners before the real installers assign them directly.
    monkeypatch.setattr(Linear, "forward", Linear.forward)
    monkeypatch.setattr(Linear4bit, "forward", Linear4bit.forward)
    monkeypatch.setattr(l1e, "_PEFT_CLASSES", l1e._PEFT_CLASSES)

    torch.manual_seed(421)
    base = bnb.nn.Linear4bit(_INPUT_WIDTH, output_width, bias=True,
        compute_dtype=torch.float16, compress_statistics=True, quant_type="nf4").half().to("cuda")
    base.bias.requires_grad_(False)
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none")
    layer = Linear4bit(base, "default", config, r=16, lora_alpha=32).train()
    layer.lora_A["default"].float()
    layer.lora_B["default"].float()
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(0, 0.02)
    assert base.weight.dtype == torch.uint8 and not base.weight.requires_grad
    assert base.weight.quant_state.nested and base.weight.quant_state.dtype == torch.float16
    assert base.weight.quant_state.quant_type == "nf4"
    assert layer.scaling["default"] == 2.0
    initial_parameters = {name: parameter.detach().cpu().clone()
                          for name, parameter in layer.named_parameters() if parameter.requires_grad}

    values = torch.randn((_BATCH, _LENGTH, _INPUT_WIDTH), device="cuda") * 0.25
    coefficient = torch.randn((_BATCH, _LENGTH, output_width), device="cuda")
    native_dropout = torch.nn.Dropout.forward
    dropout_calls = []

    def observed_native_dropout(module, x):
        before = torch.cuda.get_rng_state().clone()
        result = native_dropout(module, x)
        mask = None
        if input_grad:
            assert type(result.grad_fn).__name__ == "NativeDropoutBackward0"
            actual_mask = result.grad_fn._saved_result1
            assert actual_mask.dtype == torch.bool and actual_mask.shape == x.shape
            mask = actual_mask.detach().cpu().clone()
        dropout_calls.append({"shape": tuple(x.shape), "input_dtype": x.dtype,
            "output_dtype": result.dtype, "requires_grad": x.requires_grad,
            "before": before, "after": torch.cuda.get_rng_state().clone(), "mask": mask})
        return result

    # A class-level delegate observes the real RNG-consuming implementation;
    # no module hook or instance override changes production eligibility.
    monkeypatch.setattr(torch.nn.Dropout, "forward", observed_native_dropout)
    expected = _run(layer, values, coefficient, scale=scale, input_grad=input_grad)
    assert len(dropout_calls) == 1
    expected_dropout = dropout_calls.pop()

    assert l1e.patch_lora_dropout()
    assert epilogue.patch_lora_epilogue()
    installed_forward, installed_dropout = Linear4bit.forward, torch.nn.Dropout.forward
    assert l1e.patch_lora_dropout() and epilogue.patch_lora_epilogue()
    assert Linear4bit.forward is installed_forward
    assert torch.nn.Dropout.forward is installed_dropout
    counts = Counter()

    class DelegatingLaunch:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name
        def __getitem__(self, grid):
            launch = self.kernel[grid]
            def run(*args, **kwargs):
                result = launch(*args, **kwargs)
                counts[self.name] += 1
                return result
            return run

    for owner, name, label in (
        (l1e, "soup_dropout_half_gradient_epilogue", "dropout_backward"),
        (epilogue, "soup_lora_epilogue_forward", "scale_add_forward"),
        (epilogue, "soup_lora_epilogue_backward", "scale_add_backward"),
    ):
        monkeypatch.setattr(owner, name, DelegatingLaunch(getattr(owner, name), label))
    actual = _run(layer, values, coefficient, scale=scale, input_grad=input_grad)
    assert len(dropout_calls) == 1, "joint fast path must call native Dropout exactly once"
    actual_dropout = dropout_calls.pop()

    assert counts["scale_add_forward"] == counts["scale_add_backward"] == 1
    assert counts["dropout_backward"] == int(input_grad)
    assert l1e._ACTIVE_DROPOUT.get() is None
    for key in ("shape", "input_dtype", "output_dtype", "requires_grad"):
        assert actual_dropout[key] == expected_dropout[key]
    assert actual_dropout["shape"] == (_BATCH, _LENGTH, _INPUT_WIDTH)
    assert actual_dropout["input_dtype"] == actual_dropout["output_dtype"] == torch.float32
    for key in ("before", "after"):
        assert torch.equal(actual_dropout[key], expected_dropout[key])
    if input_grad:
        assert torch.equal(actual_dropout["mask"], expected_dropout["mask"])
    else:
        assert actual_dropout["mask"] is expected_dropout["mask"] is None
    for key in ("output", "loss"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert actual["gradients"].keys() == expected["gradients"].keys()
    for name in actual["gradients"]:
        torch.testing.assert_close(actual["gradients"][name], expected["gradients"][name],
            atol=0, rtol=0, msg=lambda message, n=name: f"joint q/v epilogues {n}: {message}")
    for key in ("cpu_rng", "cuda_rng"):
        assert torch.equal(actual[key], expected[key])
    assert actual["scaler"] == expected["scaler"] and actual["scaler"]["scale"] == scale
    for name, parameter in layer.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter.detach().cpu(), initial_parameters[name])
            assert parameter.grad is None  # autograd.grad never performs an optimizer update.
        else:
            assert parameter.grad is None
    record_property("joint_lora_physical_evidence", json.dumps({
        "batch": _BATCH, "length": _LENGTH, "input_width": _INPUT_WIDTH,
        "output_width": output_width, "scale": scale, "input_requires_grad": input_grad,
        "kernel_launches": dict(counts), "native_dropout_calls_per_forward": 1,
        "all_trainable_and_input_gradients_exact": True, "rng_exact": True,
        "native_nf4": True, "full_training_benchmark": False,
    }))
