"""Lossless saved-activation checks; credited SwiGLU kernels stay excluded."""

import os

import pytest
import torch
import torch.nn.functional as F

from . import activation_storage as storage
from .swiglu import swiglu


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.mark.skipif(not INTERPRETING, reason="CPU kernel test requires TRITON_INTERPRET=1")
@pytest.mark.parametrize("grad_dtype,scale", [
    (torch.float32, 1), (torch.float32, 65536), (torch.float16, 65536),
])
def test_compact_saved_activations_preserve_fp32_forward_and_scaled_gradients(grad_dtype, scale):
    torch.manual_seed(683)
    # Non-contiguous tensors test the same contiguous-copy semantics as the
    # starter. Each value really originated as FP16 before FP32 activation use.
    gate = (torch.randn(3, 17, 9) * 0.4).half().float().transpose(1, 2).requires_grad_()
    up = (torch.randn(3, 17, 9) * 0.4).half().float().transpose(1, 2).requires_grad_()
    other_gate = gate.detach().clone().requires_grad_()
    other_up = up.detach().clone().requires_grad_()
    expected = swiglu(gate, up)
    actual = storage.compact_swiglu(other_gate, other_up)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Include non-half-representable upstream values in the FP32 case. The
    # scale is applied before backward, as GradScaler does for its loss output.
    grad = (torch.randn_like(actual) * 0.03141593 * scale).to(grad_dtype)
    expected_grad = torch.autograd.grad(expected, (gate, up), grad_outputs=grad)
    actual_grad = torch.autograd.grad(actual, (other_gate, other_up), grad_outputs=grad)
    for got, want in zip(actual_grad, expected_grad):
        assert got.dtype == want.dtype == torch.float32
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not INTERPRETING, reason="CPU kernel test requires TRITON_INTERPRET=1")
def test_saved_tensors_are_two_half_allocations_with_exactly_half_the_bytes():
    gate = torch.linspace(-2, 2, 5 * 8960).half().float().reshape(5, 8960).requires_grad_()
    up = torch.linspace(2, -2, 5 * 8960).half().float().reshape(5, 8960).requires_grad_()
    baseline = swiglu(gate, up)
    packed = storage.compact_swiglu(gate, up)
    before, after = baseline.grad_fn.saved_tensors, packed.grad_fn.saved_tensors
    assert len(before) == len(after) == 2
    assert all(t.dtype == torch.float32 for t in before)
    assert all(t.dtype == torch.float16 and not t.requires_grad for t in after)
    for original, compact in zip(before, after):
        assert original.data_ptr() != compact.data_ptr()
        torch.testing.assert_close(compact.float(), original, rtol=0, atol=0)
    assert sum(t.untyped_storage().nbytes() for t in after) * 2 == sum(
        t.untyped_storage().nbytes() for t in before
    )


@pytest.mark.skipif(not INTERPRETING, reason="CPU kernel test requires TRITON_INTERPRET=1")
@pytest.mark.parametrize("scale", [1, 65536])
def test_direct_half_inputs_fuse_only_the_original_casts(scale):
    torch.manual_seed(684)
    gate = (torch.randn(3, 17, 9) * 0.4).half().transpose(1, 2).requires_grad_()
    up = (torch.randn(3, 17, 9) * 0.4).half().transpose(1, 2).requires_grad_()
    other_gate = gate.detach().clone().requires_grad_()
    other_up = up.detach().clone().requires_grad_()
    # This explicit upcast is what the original Linear4bit return supplies.
    expected = swiglu(gate.float(), up.float())
    actual = storage.compact_swiglu(other_gate, other_up)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual) * 0.03141593 * scale
    expected_grad = torch.autograd.grad(expected, (gate, up), grad)
    actual_grad = torch.autograd.grad(actual, (other_gate, other_up), grad)
    for got, want in zip(actual_grad, expected_grad):
        assert got.dtype == want.dtype == torch.float16
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not INTERPRETING, reason="CPU kernel test requires TRITON_INTERPRET=1")
def test_contiguous_half_inputs_are_saved_without_extra_allocations():
    gate = torch.linspace(-2, 2, 5 * 8960).half().reshape(5, 8960).requires_grad_()
    up = torch.linspace(2, -2, 5 * 8960).half().reshape(5, 8960).requires_grad_()
    result = storage.compact_swiglu(gate, up)
    assert result.dtype == torch.float32
    saved = result.grad_fn.saved_tensors
    assert len(saved) == 2
    assert saved[0].data_ptr() == gate.data_ptr()
    assert saved[1].data_ptr() == up.data_ptr()
    assert all(t.dtype == torch.float16 for t in saved)


def _bnb_mlp():
    import bitsandbytes as bnb
    from bitsandbytes.functional import QuantState
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    torch.manual_seed(685)
    mlp = Qwen2MLP(Qwen2Config(hidden_size=16, intermediate_size=32, hidden_act="silu"))
    dense = {}
    for name, inputs, outputs in (("gate_proj", 16, 32), ("up_proj", 16, 32), ("down_proj", 32, 16)):
        layer = bnb.nn.Linear4bit(inputs, outputs, bias=False, compute_dtype=torch.float16, quant_type="nf4")
        state = QuantState(
            absmax=torch.ones((inputs * outputs + 63) // 64), shape=torch.Size((outputs, inputs)),
            dtype=torch.float16, blocksize=64, quant_type="nf4",
        )
        layer.weight = bnb.nn.Params4bit(
            torch.zeros(((inputs * outputs + 1) // 2, 1), dtype=torch.uint8),
            requires_grad=False, quant_state=state, bnb_quantized=True, quant_type="nf4",
        )
        dense[id(layer.weight)] = (torch.randn(outputs, inputs) * 0.1).half()
        setattr(mlp, name, layer)
    return mlp.train(), dense


def test_storage_guard_refuses_paths_without_the_fp16_representability_proof(monkeypatch):
    import bitsandbytes as bnb

    mlp, _ = _bnb_mlp()
    x = torch.randn(2, 3, 16, requires_grad=True)
    probe = lambda: storage._eligible_mlp(mlp, x, bnb.nn.Linear4bit, bnb.nn.Linear4bit.forward)
    assert not probe()  # CPU production fallback.
    monkeypatch.setattr(storage, "_is_cuda_tensor", lambda tensor: True)
    assert probe()
    mlp.gate_proj.compute_dtype = torch.float32
    assert not probe()
    mlp.gate_proj.compute_dtype = torch.float16
    mlp.up_proj.compute_type_is_set = False
    assert not probe()  # Would switch to FP32 on first FP32 input.
    mlp.up_proj.compute_type_is_set = True
    handle = mlp.gate_proj.register_forward_hook(lambda module, inputs, outputs: outputs + 1e-5)
    assert not probe()
    handle.remove()
    handle = mlp.gate_proj.register_full_backward_hook(lambda module, inputs, outputs: None)
    assert not probe()  # Internal gradient dtypes are observable to hooks.
    handle.remove()
    mlp.eval()
    assert not probe()
    mlp.train()
    assert probe()
    mlp.down_proj = torch.nn.Linear(32, 16, bias=False)
    assert not probe()


@pytest.mark.skipif(not INTERPRETING, reason="CPU kernel test requires TRITON_INTERPRET=1")
@pytest.mark.parametrize("scale", [1, 65536])
def test_real_linear4bit_forward_cast_boundary_and_mlp_input_gradient(monkeypatch, scale):
    import bitsandbytes as bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    mlp, dense = _bnb_mlp()
    calls = []
    cast_events = []

    from torch.utils._python_dispatch import TorchDispatchMode

    class RecordCasts(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            result = func(*args, **kwargs)
            if func == torch.ops.aten._to_copy.default:
                cast_events.append((tuple(args[0].shape), args[0].dtype, result.dtype))
            return result

    def cpu_gemm(input_tensor, weight, bias=None, quant_state=None, **kwargs):
        # Only the unavailable native CPU GEMM is replaced. Actual pinned
        # Linear4bit.forward performs its input/output casting around this.
        calls.append(input_tensor)
        return F.linear(input_tensor, dense[id(weight)], bias)

    def starter_forward(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    monkeypatch.setattr(bnb, "matmul_4bit", cpu_gemm)
    monkeypatch.setattr(Qwen2MLP, "forward", starter_forward)
    monkeypatch.setattr(storage, "_is_cuda_tensor", lambda tensor: True)
    x = torch.randn(2, 5, 16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    expected = mlp(x)
    assert storage.patch_compact_mlp_activations()
    with RecordCasts():
        actual = mlp(other)
    assert [tensor.dtype for tensor in calls] == [torch.float16] * 6
    # Distinct cast nodes preserve FP32 summation of the two input gradients.
    assert calls[3] is not calls[4]
    assert calls[3].grad_fn is not calls[4].grad_fn
    # Only two input casts, down-projection input cast, and final output cast.
    # In particular no full gate/up tensor is ever upcast to global FP32.
    assert cast_events == [
        ((2, 5, 16), torch.float32, torch.float16),
        ((2, 5, 16), torch.float32, torch.float16),
        ((2, 5, 32), torch.float32, torch.float16),
        ((2, 5, 16), torch.float16, torch.float32),
    ]
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual) * 29.3871 * scale / actual.numel()
    expected_grad, = torch.autograd.grad(expected, (x,), grad)
    actual_grad, = torch.autograd.grad(actual, (other,), grad)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    # A tempting shared-cast rewrite is NOT equivalent: the branches add in
    # FP16 and only then upcast. Keep a counterexample as an implementation
    # regression check, using the same real pinned Linear4bit boundaries.
    shared_input = x.detach().clone().requires_grad_()
    shared_half = shared_input.half()
    shared = mlp.down_proj(storage.compact_swiglu(
        mlp.gate_proj(shared_half), mlp.up_proj(shared_half),
    ))
    shared_grad, = torch.autograd.grad(shared, (shared_input,), grad)
    assert torch.any(shared_grad != expected_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical CUDA device")
def test_cuda_mixed_pointer_starter_backward_matches_fp32_saved_baseline():
    torch.manual_seed(690)
    gate = torch.randn(4, 512, 8960, device="cuda").half().float().requires_grad_()
    up = torch.randn_like(gate).half().float().requires_grad_()
    other_gate, other_up = gate.detach().clone().requires_grad_(), up.detach().clone().requires_grad_()
    expected = swiglu(gate, up)
    actual = storage.compact_swiglu(other_gate, other_up)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual) * 65536 / actual.numel()
    expected_grad = torch.autograd.grad(expected, (gate, up), grad)
    actual_grad = torch.autograd.grad(actual, (other_gate, other_up), grad)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical CUDA device")
def test_cuda_direct_half_swiglu_preserves_fp32_arithmetic_and_cast_boundaries():
    torch.manual_seed(691)
    gate = torch.randn(4, 512, 8960, device="cuda").half().requires_grad_()
    up = torch.randn_like(gate).requires_grad_()
    other_gate, other_up = gate.detach().clone().requires_grad_(), up.detach().clone().requires_grad_()
    expected = swiglu(gate.float(), up.float())
    actual = storage.compact_swiglu(other_gate, other_up)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual) * 65536 / actual.numel()
    expected_grad = torch.autograd.grad(expected, (gate, up), grad)
    actual_grad = torch.autograd.grad(actual, (other_gate, other_up), grad)
    for got, want in zip(actual_grad, expected_grad):
        assert got.dtype == torch.float16
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical CUDA device and bitsandbytes")
def test_cuda_native_nf4_mlp_activates_compact_path_without_guard_overrides(monkeypatch):
    import bitsandbytes as bnb
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    torch.manual_seed(692)
    mlp = Qwen2MLP(Qwen2Config(hidden_size=64, intermediate_size=128, hidden_act="silu"))
    for name, inputs, outputs in (("gate_proj", 64, 128), ("up_proj", 64, 128), ("down_proj", 128, 64)):
        layer = bnb.nn.Linear4bit(
            inputs, outputs, bias=False, compute_dtype=torch.float16,
            compress_statistics=True, quant_type="nf4",
        ).half().cuda()
        assert layer.compute_type_is_set and layer.compute_dtype == torch.float16
        assert layer.weight.quant_state.dtype == torch.float16
        setattr(mlp, name, layer)
    mlp.train()

    def starter_forward(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    monkeypatch.setattr(Qwen2MLP, "forward", starter_forward)
    x = torch.randn(4, 32, 64, device="cuda", requires_grad=True)
    other = x.detach().clone().requires_grad_()
    assert storage._eligible_mlp(mlp, other, bnb.nn.Linear4bit, bnb.nn.Linear4bit.forward)
    with torch.autocast("cuda", dtype=torch.float16):
        expected = mlp(x)
    assert storage.patch_compact_mlp_activations()
    with torch.autocast("cuda", dtype=torch.float16):
        actual = mlp(other)
    # Inspect the actual autograd graph rather than replacing a guard or GEMM.
    pending, seen, compact = [actual.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if type(node).__name__ == "_CompactSwiGLUBackward":
            compact.append(node)
        pending.extend(parent for parent, _ in node.next_functions)
    assert len(compact) == 1
    assert len(compact[0].saved_tensors) == 2
    assert all(t.dtype == torch.float16 for t in compact[0].saved_tensors)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual) * 65536 / actual.numel()
    expected_grad, = torch.autograd.grad(expected, (x,), grad)
    actual_grad, = torch.autograd.grad(actual, (other,), grad)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
