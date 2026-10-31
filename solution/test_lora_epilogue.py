"""Exact rounding, wrapper, RNG and native-CUDA boundaries for LoRA fusion."""

import os

import pytest
import torch
import torch.nn.functional as F

from . import lora_epilogue as epilogue


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CPU_KERNEL = pytest.mark.skipif(not INTERPRETING, reason="CPU kernel tests require TRITON_INTERPRET=1")
CUDA_KERNEL = pytest.mark.skipif(
    not torch.cuda.is_available() or INTERPRETING, reason="requires actual CUDA execution, not the interpreter",
)


@CPU_KERNEL
@pytest.mark.parametrize("grad_dtype,scale", [
    (torch.float32, 1), (torch.float32, 65536), (torch.float16, 65536),
])
def test_epilogue_matches_rounding_and_gradients_without_saved_activations(grad_dtype, scale):
    torch.manual_seed(907)
    base = torch.randn(2, 5, 129, dtype=torch.float16, requires_grad=True)
    adapter = torch.randn_like(base, requires_grad=True)
    other_base = base.detach().clone().requires_grad_()
    other_adapter = adapter.detach().clone().requires_grad_()
    expected = base.float().clone() + adapter * 2.0
    actual = epilogue.lora_epilogue(other_base, other_adapter)
    assert actual.dtype == torch.float32
    assert actual.data_ptr() not in (other_base.data_ptr(), other_adapter.data_ptr())
    assert actual.grad_fn.saved_tensors == ()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Non-contiguous upstream gradients must retain their original precision.
    grad = (torch.randn(2, 129, 5) * 0.03141593 * scale).transpose(1, 2).to(grad_dtype)
    if grad_dtype == torch.float32 and scale == 65536:
        # Metadata-only negative views must be resolved before raw-pointer loads.
        grad = torch._neg_view(grad.contiguous())
    expected_grad = torch.autograd.grad(expected, (base, adapter), grad)
    actual_grad = torch.autograd.grad(actual, (other_base, other_adapter), grad)
    for got, want in zip(actual_grad, expected_grad):
        assert got.dtype == want.dtype == torch.float16
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(other_base, base, rtol=0, atol=0)
    torch.testing.assert_close(other_adapter, adapter, rtol=0, atol=0)


def _check_rounding_edges(device):
    base = torch.tensor([1, -1, 0, -0.0, 1, 1, -1, -1], dtype=torch.float16, device=device,
                        requires_grad=True)
    adapter = torch.tensor([32768, -32768, 0, -0.0, 1, -1, 2, -2], dtype=torch.float16,
                           device=device, requires_grad=True)
    other_base = base.detach().clone().requires_grad_()
    other_adapter = adapter.detach().clone().requires_grad_()
    expected = base.float().clone() + adapter * 2.0
    actual = epilogue.lora_epilogue(other_base, other_adapter)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    # Scaling in FP32 through the addition would incorrectly avoid overflow.
    assert torch.isinf(actual[:2]).all()
    assert torch.isfinite(base.float()[:2] + adapter.float()[:2] * 2).all()
    grad = torch.tensor([2**-25, 3 * 2**-25, -2**-25, -3 * 2**-25,
                         32768, -32768, float("inf"), float("nan")], device=device)
    expected_grad = torch.autograd.grad(expected, (base, adapter), grad)
    actual_grad = torch.autograd.grad(actual, (other_base, other_adapter), grad)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(torch.signbit(got[:-1]), torch.signbit(want[:-1]))
    assert torch.any(actual_grad[1][:4] != (grad[:4] * 2).half())


@CPU_KERNEL
def test_epilogue_preserves_intermediate_overflow_subnormals_and_signed_zeros():
    _check_rounding_edges("cpu")


def _make_cpu_layer():
    import bitsandbytes as bnb
    from bitsandbytes.functional import QuantState
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit

    torch.manual_seed(908)
    base = bnb.nn.Linear4bit(64, 96, bias=True, compute_dtype=torch.float16, quant_type="nf4")
    state = QuantState(absmax=torch.ones(96), shape=torch.Size((96, 64)),
                       dtype=torch.float16, blocksize=64, quant_type="nf4")
    base.weight = bnb.nn.Params4bit(
        torch.zeros((96 * 64 // 2, 1), dtype=torch.uint8), requires_grad=False,
        quant_state=state, bnb_quantized=True, quant_type="nf4",
    )
    base.bias.requires_grad_(False)
    layer = Linear4bit(base, "default", LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05),
                       r=16, lora_alpha=32).train()
    # Match Soup's FP16-training preparation, which restores trainable
    # adapter parameters to FP32 before GradScaler-managed training.
    layer.lora_A["default"].float()
    layer.lora_B["default"].float()
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(std=0.05)
    dense = torch.randn(96, 64, dtype=torch.float16) * 0.025
    return layer, dense


def _cpu_boundary_overrides(monkeypatch):
    # These overrides model the unavailable CUDA device / AMP admission
    # boundary only. Actual pinned PEFT and bnb forward methods still execute.
    monkeypatch.setattr(epilogue, "_is_cuda_tensor", lambda tensor: True)
    real_query = torch.is_autocast_enabled

    def query(device_type="cuda"):
        return real_query("cpu" if device_type == "cuda" else device_type)

    monkeypatch.setattr(torch, "is_autocast_enabled", query)
    monkeypatch.setattr(epilogue, "_cuda_fp16_autocast_active", lambda: real_query("cpu"))


def _probe(layer, x):
    import bitsandbytes as bnb
    from peft.tuners.lora.bnb import Linear4bit

    return epilogue._eligible_lora(layer, x, Linear4bit, bnb.nn.Linear4bit,
                                   bnb.nn.Linear4bit.forward, torch.nn.Linear.forward,
                                   torch.nn.Dropout.forward)


def test_guard_refuses_unproven_precision_variants_and_observable_hooks(monkeypatch):
    layer, _ = _make_cpu_layer()
    x = torch.randn(2, 7, 64, requires_grad=True)
    assert not _probe(layer, x)
    _cpu_boundary_overrides(monkeypatch)
    with torch.autocast("cpu", dtype=torch.float16):
        assert _probe(layer, x)
        layer.base_layer.compute_type_is_set = False
        assert not _probe(layer, x)
        layer.base_layer.compute_type_is_set = True
        layer.base_layer.compute_dtype = torch.float32
        assert not _probe(layer, x)
        layer.base_layer.compute_dtype = torch.float16
        layer.scaling["default"] = 3.0
        assert not _probe(layer, x)
        layer.scaling["default"] = 2.0
        layer.lora_dropout["default"].p = 0.1
        assert not _probe(layer, x)
        layer.lora_dropout["default"].p = 0.05
        layer.base_layer.bias.requires_grad_(True)
        assert not _probe(layer, x)
        layer.base_layer.bias.requires_grad_(False)
        handle = layer.base_layer.register_forward_pre_hook(lambda *args: None)
        assert not _probe(layer, x)
        handle.remove()
        handle = torch.nn.modules.module.register_module_full_backward_hook(lambda *args: None)
        assert not _probe(layer, x)
        handle.remove()
        layer.lora_variant["default"] = object()
        assert not _probe(layer, x)
        del layer.lora_variant["default"]
        layer.enable_adapters(False)
        assert not _probe(layer, x)
        layer.enable_adapters(True)
        assert _probe(layer, x)
    assert not _probe(layer, x)


@CPU_KERNEL
@pytest.mark.parametrize("scale,input_requires_grad", [(1, True), (1000, True), (65536, True), (65536, False)])
def test_pinned_peft_bnb_wrapper_keeps_dropout_rng_calls_and_all_gradients(monkeypatch, scale, input_requires_grad):
    import bitsandbytes as bnb
    from peft.tuners.lora.bnb import Linear4bit
    from torch.utils._python_dispatch import TorchDispatchMode

    layer, dense = _make_cpu_layer()
    _cpu_boundary_overrides(monkeypatch)
    # Register restoration before patch_lora_epilogue assigns the class method.
    monkeypatch.setattr(Linear4bit, "forward", Linear4bit.forward)
    dropout_inputs, gemm_inputs, events = [], [], []
    original_dropout = F.dropout

    def dropout(x, *args, **kwargs):
        dropout_inputs.append(x)
        return original_dropout(x, *args, **kwargs)

    def cpu_gemm(x, weight, bias=None, quant_state=None, **kwargs):
        gemm_inputs.append(x)
        return F.linear(x, dense, bias)

    class Casts(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func in (torch.ops.aten._to_copy.default, torch.ops.aten.clone.default):
                events.append((str(func), tuple(args[0].shape), args[0].dtype, result.dtype))
            return result

    monkeypatch.setattr(F, "dropout", dropout)
    monkeypatch.setattr(bnb, "matmul_4bit", cpu_gemm)
    x = torch.randn(2, 7, 64, requires_grad=input_requires_grad)
    other = x.detach().clone().requires_grad_(input_requires_grad)
    weights = (layer.lora_A["default"].weight, layer.lora_B["default"].weight)
    identities = {name: id(parameter) for name, parameter in layer.named_parameters()}
    torch.manual_seed(909)
    with torch.autocast("cpu", dtype=torch.float16):
        expected = layer(x)
    expected_rng = torch.random.get_rng_state().clone()
    grad = torch.randn_like(expected) * scale / 100
    expected_grad = torch.autograd.grad(expected, ((x,) if input_requires_grad else ()) + weights, grad)
    assert epilogue.patch_lora_epilogue()
    installed = Linear4bit.forward
    assert epilogue.patch_lora_epilogue() and Linear4bit.forward is installed
    torch.manual_seed(909)
    with Casts(), torch.autocast("cpu", dtype=torch.float16):
        assert _probe(layer, other)
        actual = layer(other)
    assert actual.grad_fn.saved_tensors == ()
    assert actual.grad_fn.needs_input_grad == (input_requires_grad, True)
    actual_grad = torch.autograd.grad(actual, ((other,) if input_requires_grad else ()) + weights, grad)
    assert identities == {name: id(parameter) for name, parameter in layer.named_parameters()}
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert len(dropout_inputs) == len(gemm_inputs) == 2
    assert dropout_inputs[0] is x and dropout_inputs[1] is other
    assert all(t.dtype == torch.float32 for t in dropout_inputs)
    assert all(t.dtype == torch.float16 for t in gemm_inputs)
    assert (gemm_inputs[1].grad_fn is not None) == input_requires_grad
    # The base result never undergoes a global FP16 -> FP32 output cast/clone.
    assert not [event for event in events if event[1] == tuple(actual.shape)]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(actual_grad, expected_grad):
        assert got.dtype == want.dtype == torch.float32
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_fallback_runs_original_once_and_rejects_invalid_arguments_before_rng(monkeypatch):
    import bitsandbytes as bnb
    from peft.tuners.lora.bnb import Linear4bit

    layer, dense = _make_cpu_layer()
    _cpu_boundary_overrides(monkeypatch)
    monkeypatch.setattr(Linear4bit, "forward", Linear4bit.forward)
    monkeypatch.setattr(bnb, "matmul_4bit", lambda x, weight, bias=None, **kwargs: F.linear(x, dense, bias))
    monkeypatch.setattr(epilogue, "lora_epilogue", lambda *args: pytest.fail("unsupported call reached kernel"))
    layer.scaling["default"] = 3.0
    x = torch.randn(2, 7, 64, requires_grad=True)
    torch.manual_seed(910)
    with torch.autocast("cpu", dtype=torch.float16):
        expected = layer(x)
    expected_rng = torch.random.get_rng_state().clone()
    assert epilogue.patch_lora_epilogue()
    torch.manual_seed(910)
    with torch.autocast("cpu", dtype=torch.float16):
        actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="adapter_names"):
        layer(x, adapter_names=["default"])
    assert torch.equal(torch.random.get_rng_state(), before)


def test_patch_declines_an_unreviewed_library_version(monkeypatch):
    original = epilogue.metadata.version
    monkeypatch.setattr(epilogue.metadata, "version", lambda name: "0.21.0" if name == "peft" else original(name))
    assert not epilogue.patch_lora_epilogue()


@CUDA_KERNEL
def test_lora_epilogue_rounding_edges_cuda_only():
    _check_rounding_edges("cuda")


@CUDA_KERNEL
@pytest.mark.parametrize("length,scale,input_requires_grad", [
    (32, 1, True), (32, 65536, True), (800, 1, True), (800, 65536, True),
    (32, 65536, False),
])
def test_native_cuda_peft_nf4_epilogue_and_gradscaler(monkeypatch, length, scale, input_requires_grad):
    import bitsandbytes as bnb
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit

    torch.manual_seed(911)
    base = bnb.nn.Linear4bit(64, 96, bias=True, compute_dtype=torch.float16,
                             compress_statistics=True, quant_type="nf4").half().to("cuda")
    base.bias.requires_grad_(False)
    layer = Linear4bit(base, "default", LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05),
                       r=16, lora_alpha=32).train()
    for name in ("lora_A", "lora_B"):
        getattr(layer, name)["default"].to(torch.float32)
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(std=0.05)
    monkeypatch.setattr(Linear4bit, "forward", Linear4bit.forward)
    x = torch.randn(2, length, 64, device="cuda", requires_grad=input_requires_grad)
    other = x.detach().clone().requires_grad_(input_requires_grad)
    weights = (layer.lora_A["default"].weight, layer.lora_B["default"].weight)
    upstream = torch.randn(2, length, 96, device="cuda") * 0.1
    scaler = torch.amp.GradScaler("cuda", init_scale=scale)
    rng = torch.cuda.get_rng_state().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        assert _probe(layer, x)
        expected = layer(x)
    expected_grad = torch.autograd.grad(scaler.scale((expected * upstream).mean()),
                                         ((x,) if input_requires_grad else ()) + weights)
    expected_rng = torch.cuda.get_rng_state().clone()
    seen = []
    original_epilogue = epilogue.lora_epilogue

    def counted(*args):
        seen.append(True)
        return original_epilogue(*args)

    monkeypatch.setattr(epilogue, "lora_epilogue", counted)
    assert epilogue.patch_lora_epilogue()
    torch.cuda.set_rng_state(rng)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = layer(other)
    actual_grad = torch.autograd.grad(scaler.scale((actual * upstream).mean()),
                                       ((other,) if input_requires_grad else ()) + weights)
    assert seen == [True]
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
