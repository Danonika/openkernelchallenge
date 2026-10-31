"""Native PEFT composition and exact dropout-boundary tests.

Interpreter tests explicitly substitute the unavailable CUDA admission check
and inspect the CPU native Dropout mask. They do not execute NF4 or a GPU.
"""
import copy
import os

import pytest
import torch
from peft import LoraConfig
from peft.tuners.lora.layer import Linear

from . import lora_dropout as l1e


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CPU_KERNEL = pytest.mark.skipif(not INTERPRETING, reason="requires TRITON_INTERPRET=1")
CUDA_KERNEL = pytest.mark.skipif(
    not torch.cuda.is_available() or INTERPRETING,
    reason="requires a physical CUDA device, not the interpreter")


def _layer(width=48, output=32, device="cpu"):
    torch.manual_seed(433)
    base = torch.nn.Linear(width, output, bias=True, device=device)
    base.requires_grad_(False)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "v_proj"], bias="none")
    layer = Linear(base, "default", config=cfg, r=16, lora_alpha=32).to(device).train()
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(0, 0.02)
    return layer


def _restore_globals(monkeypatch):
    monkeypatch.setattr(l1e, "_PEFT_CLASSES", l1e._PEFT_CLASSES)
    monkeypatch.setattr(Linear, "forward", Linear.forward)
    monkeypatch.setattr(torch.nn.Dropout, "forward", torch.nn.Dropout.forward)
    try:
        from peft.tuners.lora.bnb import Linear4bit
        monkeypatch.setattr(Linear4bit, "forward", Linear4bit.forward)
    except ImportError:
        pass


def _cpu_boundary(monkeypatch):
    monkeypatch.setattr(l1e, "_supported_device", lambda x:
        x.device.type == "cpu" and torch.is_autocast_enabled("cpu")
        and torch.get_autocast_dtype("cpu") == torch.float16)
    observations = []
    native_saved = l1e._saved_native_mask
    def saved(dropped, x):
        # CPU F.dropout saves a scaled float mask. Only this diagnostic
        # fixture recognizes it; the production path accepts native bool.
        fn = dropped.grad_fn
        if type(fn).__name__ == "MulBackward0":
            scaled = fn._saved_other
            keep = scaled != 0
            assert torch.equal(scaled, keep.float() * l1e._KEEP_SCALE_FP32)
            observations.append(keep)
            return keep
        return native_saved(dropped, x)
    monkeypatch.setattr(l1e, "_saved_native_mask", saved)
    return observations


def _run(layer, values, upstream, *, input_grad=True, device="cpu"):
    x = values.clone().requires_grad_(input_grad)
    torch.manual_seed(983)
    if device == "cuda":
        torch.cuda.manual_seed_all(983)
    with torch.autocast(device, dtype=torch.float16):
        out = layer(x)
    params = tuple(p for p in layer.parameters() if p.requires_grad)
    variables = (x, *params) if input_grad else params
    grads = torch.autograd.grad(out, variables, upstream)
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.random.get_rng_state()
    return out.detach(), grads, rng


def _equal(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    assert torch.equal(torch.signbit(actual[finite]), torch.signbit(expected[finite]))


@CPU_KERNEL
@pytest.mark.parametrize("shape,scale,input_grad", [
    ((17, 48), 1, True), ((2, 9, 48), 65536, True),
    ((65, 48), 1000, True), ((17, 48), 65536, False),
])
def test_real_peft_forward_rng_and_all_gradients(monkeypatch, shape, scale, input_grad):
    base = _layer()
    candidate = copy.deepcopy(base)
    g = torch.Generator().manual_seed(719)
    values = torch.randn(shape, generator=g) * 0.1
    upstream = torch.randn(shape[:-1] + (32,), generator=g) * (0.01 * scale)
    expected = _run(base, values, upstream, input_grad=input_grad)
    _restore_globals(monkeypatch)
    seen = _cpu_boundary(monkeypatch)
    assert l1e.patch_lora_dropout()
    assert l1e.patch_lora_dropout()
    actual = _run(candidate, values, upstream, input_grad=input_grad)
    _equal(actual[0], expected[0])
    for got, want in zip(actual[1], expected[1]):
        _equal(got, want)
    assert torch.equal(actual[2], expected[2])
    assert len(seen) == int(input_grad)
    assert l1e._ACTIVE_DROPOUT.get() is None
    assert "forward" not in candidate.__dict__
    assert "forward" not in candidate.lora_dropout["default"].__dict__


@CPU_KERNEL
def test_unknown_mask_keeps_native_graph_and_dropout_is_not_repeated(monkeypatch):
    base = _layer(8, 5)
    candidate = copy.deepcopy(base)
    values, upstream = torch.randn(2, 8) * 0.1, torch.randn(2, 5)
    expected = _run(base, values, upstream)
    _restore_globals(monkeypatch)
    _cpu_boundary(monkeypatch)
    monkeypatch.setattr(l1e, "_saved_native_mask", lambda *args: None)
    assert l1e.patch_lora_dropout()
    actual = _run(candidate, values, upstream)
    _equal(actual[0], expected[0])
    for got, want in zip(actual[1], expected[1]):
        _equal(got, want)
    assert torch.equal(actual[2], expected[2])


@CPU_KERNEL
def test_native_bool_mask_bridge_nonfinite_rounding_and_higher_order():
    x = torch.randn(2, 8, requires_grad=True)
    native, mask = torch.ops.aten.native_dropout(x, 0.05, True)
    assert l1e._saved_native_mask(native, x) is not None
    bridged = l1e.DropoutHalfBridge.apply(x, native.detach(), mask)
    reference = native.half()
    upstream = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"),
                             2**-24, -2**-24, 65504], dtype=torch.float16).repeat(2, 1)
    expected = torch.autograd.grad(reference, x, upstream, retain_graph=True)[0]
    actual = torch.autograd.grad(bridged, x, upstream, retain_graph=True)[0]
    _equal(actual, expected)
    # A non-contiguous gradient takes the native layout fallback.
    noncontig = torch.randn(8, 2, dtype=torch.float16).T
    _equal(torch.autograd.grad(bridged, x, noncontig, retain_graph=True)[0],
           torch.autograd.grad(reference, x, noncontig, retain_graph=True)[0])
    first_ref = torch.autograd.grad(reference.float().square().sum(), x, create_graph=True)[0]
    first_got = torch.autograd.grad(bridged.float().square().sum(), x, create_graph=True)[0]
    _equal(first_got, first_ref)
    _equal(torch.autograd.grad(first_got.sum(), x)[0],
           torch.autograd.grad(first_ref.sum(), x)[0])


def test_production_admission_rejects_cpu_and_preserves_eval(monkeypatch):
    base = _layer(8, 5)
    values = torch.randn(2, 8, requires_grad=True)
    _restore_globals(monkeypatch)
    assert l1e.patch_lora_dropout()
    with torch.autocast("cpu", dtype=torch.float16):
        assert l1e._eligible_dropout(base, values) is None
    base.eval()
    assert l1e._eligible_dropout(base, values) is None
    with pytest.raises(TypeError):
        base(values, unexpected_argument=True)
    assert l1e._ACTIVE_DROPOUT.get() is None


@CPU_KERNEL
@pytest.mark.parametrize("scale,input_grad", [(1, True), (65536, True), (65536, False)])
def test_joint_scale_add_and_dropout_epilogues_keep_both_active(monkeypatch, scale, input_grad):
    """Real pinned PEFT/BNB wrappers with a dense CPU base-GEMM stand-in.

    The stand-in is explicit: this verifies both patch owners and cast/RNG
    composition, not native NF4 execution.
    """
    import bitsandbytes as bnb
    import torch.nn.functional as F
    from . import lora_epilogue as epilogue
    from .test_lora_epilogue import _make_cpu_layer, _cpu_boundary_overrides
    base, dense = _make_cpu_layer()
    base.base_layer.compute_type_is_set = True
    candidate = copy.deepcopy(base)
    monkeypatch.setattr(bnb, "matmul_4bit", lambda x, weight, bias=None, **kwargs:
        F.linear(x, dense, bias))
    _cpu_boundary_overrides(monkeypatch)
    values = torch.randn(2, 7, 64) * 0.1
    upstream = torch.randn(2, 7, 96) * (0.01 * scale)
    expected = _run(base, values, upstream, input_grad=input_grad)
    _restore_globals(monkeypatch)
    seen = _cpu_boundary(monkeypatch)
    assert l1e.patch_lora_dropout()
    assert epilogue.patch_lora_epilogue()
    # Reapplication must not add another forward/dropout owner.
    forward = type(candidate).forward
    assert l1e.patch_lora_dropout()
    assert epilogue.patch_lora_epilogue()
    assert type(candidate).forward is forward
    original_epilogue = epilogue.lora_epilogue
    epilogue_calls = []
    def called(a, b):
        epilogue_calls.append((a.dtype, b.dtype))
        return original_epilogue(a, b)
    monkeypatch.setattr(epilogue, "lora_epilogue", called)
    actual = _run(candidate, values, upstream, input_grad=input_grad)
    _equal(actual[0], expected[0])
    for got, want in zip(actual[1], expected[1]):
        _equal(got, want)
    assert torch.equal(actual[2], expected[2])
    assert epilogue_calls == [(torch.float16, torch.float16)]
    assert len(seen) == int(input_grad)
    assert l1e._ACTIVE_DROPOUT.get() is None


@CUDA_KERNEL
def test_physical_t4_native_peft_dropout_gradient_and_rng(monkeypatch):
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("this production candidate is guarded to T4 SM75")
    base = _layer(device="cuda")
    candidate = copy.deepcopy(base)
    values = torch.randn(4, 17, 48, device="cuda") * 0.1
    upstream = torch.randn(4, 17, 32, device="cuda") * 1000
    expected = _run(base, values, upstream, device="cuda")
    _restore_globals(monkeypatch)
    assert l1e.patch_lora_dropout()
    seen = []
    native_saved = l1e._saved_native_mask
    def saved(out, x):
        result = native_saved(out, x)
        seen.append(result is not None)
        return result
    monkeypatch.setattr(l1e, "_saved_native_mask", saved)
    actual = _run(candidate, values, upstream, device="cuda")
    _equal(actual[0], expected[0])
    for got, want in zip(actual[1], expected[1]):
        _equal(got, want)
    assert torch.equal(actual[2], expected[2])
    assert seen == [True]


@CUDA_KERNEL
@pytest.mark.parametrize("length,input_grad,scale", [
    (17, True, 65536), (512, True, 1), (17, False, 65536),
])
def test_physical_t4_nf4_lora_dropout_gradient_and_rng(monkeypatch, length, input_grad, scale):
    """Both NF4 native forward paths plus scaled native adapter/input gradients."""
    import bitsandbytes as bnb
    from peft.tuners.lora.bnb import Linear4bit
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("production candidate is guarded to T4 SM75")
    torch.manual_seed(417)
    frozen = bnb.nn.Linear4bit(1536, 256, bias=True, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4").half().to("cuda")
    frozen.bias.requires_grad_(False)
    layer = Linear4bit(frozen, "default", LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05),
                       r=16, lora_alpha=32).train()
    layer.lora_A["default"].float()
    layer.lora_B["default"].float()
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(0, 0.02)
    values = torch.randn(4, length, 1536, device="cuda") * 0.1
    upstream = torch.randn(4, length, 256, device="cuda") * (0.01 * scale)
    expected = _run(layer, values, upstream, input_grad=input_grad, device="cuda")
    _restore_globals(monkeypatch)
    assert l1e.patch_lora_dropout()
    original_kernel = l1e.soup_dropout_half_gradient_epilogue
    launches = []
    class CountLaunch:
        def __getitem__(self, grid):
            native_launch = original_kernel[grid]
            def run(*args, **kwargs):
                launches.append(grid)
                return native_launch(*args, **kwargs)
            return run
    monkeypatch.setattr(l1e, "soup_dropout_half_gradient_epilogue", CountLaunch())
    actual = _run(layer, values, upstream, input_grad=input_grad, device="cuda")
    _equal(actual[0], expected[0])
    for got, want in zip(actual[1], expected[1]):
        _equal(got, want)
    assert torch.equal(actual[2], expected[2])
    assert len(launches) == int(input_grad)
