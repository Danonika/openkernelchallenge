"""NF4 representation tests: CPU interpreter math and native-library parity.

No CPU test establishes CUDA timing, dispatch, or training quality. The actual
CUDA integration test is skipped without a physical GPU.
"""

import os

import pytest
import torch


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CAN_EXECUTE = torch.cuda.is_available() or INTERPRETING
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_state(shape, dtype, blocksize, nested, device=DEVICE):
    from bitsandbytes.functional import QuantState, get_4bit_type

    count = shape[0] * shape[1]
    torch.manual_seed(783)
    # Every byte value and therefore every nibble combination is exercised.
    packed = (torch.arange((count + 1) // 2, device=device) % 256).to(torch.uint8).reshape(-1, 1)
    scale_count = (count + blocksize - 1) // blocksize
    if nested:
        scales = torch.randint(0, 256, (scale_count,), dtype=torch.uint8, device=device)
        state2 = QuantState(
            absmax=torch.rand((scale_count + 255) // 256, device=device) * 0.25,
            code=torch.linspace(-1, 1, 256, device=device),
            blocksize=256, dtype=torch.float32,
        )
        offset = torch.tensor(0.7314567565917969, device=device)
    else:
        scales = torch.rand(scale_count, device=device) + 0.125
        state2, offset = None, None
    state = QuantState(scales, shape=torch.Size(shape), code=get_4bit_type("nf4", device=device),
                       dtype=dtype, blocksize=blocksize, quant_type="nf4", state2=state2, offset=offset)
    return packed, state


def reference(packed, state):
    count = state.shape.numel()
    byte = packed.flatten().long()
    codes = torch.stack((byte >> 4, byte & 15), dim=-1).flatten()[:count]
    scale = state.absmax
    if state.nested:
        idx = torch.arange(scale.numel(), device=scale.device) // state.state2.blocksize
        # Separate operations mirror the real public API's FP32 rounding.
        scale = state.state2.code[scale.long()] * state.state2.absmax[idx]
        scale = scale + state.offset
    which_scale = torch.arange(count, device=packed.device) // state.blocksize
    decoded = (state.code[codes] * scale[which_scale]).to(state.dtype).reshape(state.shape)
    return decoded.t() if packed.shape[0] == 1 else decoded


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape,blocksize", [((16, 32), 32), ((37, 515), 64), ((7, 589), 128)])
def test_nf4_all_codes_tails_and_nested_boundaries(dtype, nested, shape, blocksize):
    from solution.nf4 import dequantize_nf4

    packed, state = make_state(shape, dtype, blocksize, nested)
    actual = dequantize_nf4(packed, state)
    torch.testing.assert_close(actual, reference(packed, state), rtol=0, atol=0)


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
def test_nf4_transposed_boundary_reuses_out_and_preserves_metadata():
    from solution.nf4 import dequantize_nf4

    packed, state = make_state((5, 129), torch.float16, 64, True)
    packed = packed.t()
    before = {key: value for key, value in state.__dict__.items()}
    out = torch.full(state.shape, float("nan"), dtype=state.dtype, device=packed.device)
    actual = dequantize_nf4(packed, state, out=out)
    assert actual.data_ptr() == out.data_ptr()
    assert actual.shape == (129, 5)
    torch.testing.assert_close(actual, reference(packed, state), rtol=0, atol=0)
    assert state.__dict__.keys() == before.keys()
    assert all(state.__dict__[key] is value for key, value in before.items())


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
def test_nf4_unaligned_output_and_odd_tail_do_not_overwrite_sentinels():
    from solution.nf4 import dequantize_nf4

    packed, state = make_state((3, 131), torch.float16, 64, True)
    count = state.shape.numel()
    for start in (1, 2):
        allocation = torch.full((count + 4,), 123.0, dtype=torch.float16, device=DEVICE)
        out = allocation[start:start + count].view(state.shape)
        actual = dequantize_nf4(packed, state, out=out)
        torch.testing.assert_close(actual, reference(packed, state), rtol=0, atol=0)
        assert torch.equal(allocation[:start], torch.full_like(allocation[:start], 123.0))
        assert torch.equal(allocation[start + count:], torch.full_like(allocation[start + count:], 123.0))


@pytest.mark.skipif(not CAN_EXECUTE, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_nf4_matches_native_bitsandbytes_representation(dtype, nested):
    import bitsandbytes.functional as F
    from solution.nf4 import dequantize_nf4

    torch.manual_seed(623)
    original = torch.randn((7, 128), device=DEVICE, dtype=dtype) * 0.1
    packed, state = F.quantize_4bit(original, blocksize=64,
                                  compress_statistics=nested, quant_type="nf4")
    expected = F.dequantize_4bit(packed, state)
    actual = dequantize_nf4(packed, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nf4_guards_delegate_unsupported_metadata():
    from solution.nf4 import _state_components

    packed, state = make_state((4, 128), torch.float16, 64, True, device="cpu")
    assert _state_components(packed, state, None, cuda_only=True) is None
    state.quant_type = "fp4"
    assert _state_components(packed, state, None) is None
    state.quant_type = "nf4"
    state.state2.dtype = torch.float16
    assert _state_components(packed, state, None) is None
    state.state2.dtype = torch.float32
    state.absmax = state.absmax.to(torch.int8)
    assert _state_components(packed, state, None) is None


@pytest.mark.skipif(not INTERPRETING, reason="test-only CPU interpreter boundary injection")
def test_nf4_patch_covers_both_boundaries_and_preserves_dispatch(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import bitsandbytes.functional as F
    from solution import nf4

    public_original = F.dequantize_4bit
    delegated = []

    def fallback(*args, **kwargs):
        delegated.append((args, kwargs))
        return "original-backend"

    fake_ops = SimpleNamespace(_dequant_linear_fallback=fallback)
    fake_cuda = ModuleType("bitsandbytes.backends.cuda")
    fake_cuda.ops = fake_ops
    monkeypatch.setitem(sys.modules, "bitsandbytes.backends.cuda", fake_cuda)
    monkeypatch.setattr(torch.version, "cuda", "test-only-CPU-boundary")
    monkeypatch.setattr(F, "dequantize_4bit", public_original)
    supported = nf4._components_supported
    monkeypatch.setattr(nf4, "_components_supported",
                        lambda *args, cuda_only=False, **kwargs: supported(*args, **kwargs))
    assert nf4.patch_bitsandbytes_nf4()
    installed = F.dequantize_4bit
    assert nf4.patch_bitsandbytes_nf4() and F.dequantize_4bit is installed

    packed, state = make_state((96, 64), torch.float16, 64, True, device="cpu")
    expected = public_original(packed, state)
    out = torch.empty_like(expected)
    actual = F.dequantize_4bit(packed.t(), state, out=out)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected.t(), rtol=0, atol=0)

    torch.manual_seed(19)
    x = torch.randn((2, 7, 64), dtype=torch.float32)
    bias = torch.randn((96,), dtype=torch.float32)
    # The forward fallback chooses A.dtype, independent of QuantState.dtype.
    state.dtype = torch.float32
    weight_fp32 = public_original(packed, state)
    state.dtype = torch.float16
    result = fake_ops._dequant_linear_fallback(
        x, packed, state.shape, state.state2.absmax, 64, "nf4", bias,
        state.absmax, state.state2.code, state.offset,
    )
    torch.testing.assert_close(result, torch.nn.functional.linear(x, weight_fp32, bias), rtol=0, atol=0)
    assert delegated == []
    assert fake_ops._dequant_linear_fallback(
        x, packed, state.shape, state.state2.absmax, 64, "fp4", bias,
        state.absmax, state.state2.code, state.offset,
    ) == "original-backend"
    assert len(delegated) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs actual CUDA bitsandbytes dispatch")
@pytest.mark.parametrize("length", [32, 800])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
def test_nf4_patched_linear4bit_forward_and_scaled_backward(monkeypatch, length, weight_dtype):
    import bitsandbytes as bnb
    import bitsandbytes.functional as F
    from bitsandbytes.backends.cuda import ops
    from solution import nf4

    torch.manual_seed(955)
    module = bnb.nn.Linear4bit(64, 96, bias=True, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4").to(weight_dtype).to("cuda")
    assert module.weight.quant_state.dtype == weight_dtype
    # T4 uses its native fused GEMM at M=64 for this matrix, whereas M=1600
    # exercises the dequant+linear fallback. Neither dispatch is overridden.
    x = torch.randn((2, length, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    dy = torch.randn((2, length, 96), device="cuda", dtype=torch.float16) * 128
    native_calls, decoder_calls = [], []
    native_impl, decoder_impl = ops._gemm_4bit_kernel_impl, nf4._execute

    def native(*args, **kwargs):
        native_calls.append(True)
        return native_impl(*args, **kwargs)

    def decode(*args, **kwargs):
        decoder_calls.append(True)
        return decoder_impl(*args, **kwargs)

    monkeypatch.setattr(ops, "_gemm_4bit_kernel_impl", native)
    with torch.autocast("cuda", dtype=torch.float16):
        expected = module(x)
    expected_dx, = torch.autograd.grad(expected, x, dy)
    expected_native_calls = len(native_calls)
    if torch.cuda.get_device_capability() == (7, 5):
        assert expected_native_calls == (1 if length == 32 else 0)
    native_calls.clear()
    old_public, old_fallback = F.dequantize_4bit, ops._dequant_linear_fallback
    monkeypatch.setattr(F, "dequantize_4bit", old_public)
    monkeypatch.setattr(ops, "_dequant_linear_fallback", old_fallback)
    monkeypatch.setattr(nf4, "_execute", decode)
    assert nf4.patch_bitsandbytes_nf4()
    assert nf4.patch_bitsandbytes_nf4()
    with torch.autocast("cuda", dtype=torch.float16):
        actual = module(x)
    assert len(native_calls) == expected_native_calls
    assert len(decoder_calls) == (0 if expected_native_calls else 1)
    actual_dx, = torch.autograd.grad(actual, x, dy)
    assert len(decoder_calls) == (1 if expected_native_calls else 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)
