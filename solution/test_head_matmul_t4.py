"""Documented fragment-model checks plus GPU-only tests of the real PTX.

The NumPy model checks intended register packing, tile coverage and edge
arithmetic.  It is NOT an interpreter for the emitted assembly.  GPU tests are
the validation of actual kernel execution; they skip when CUDA is unavailable.
"""

import os

import numpy as np
import pytest
import torch


def _coordinates(lane):
    group, pair = lane // 4, 2 * (lane % 4)
    a = [(group, pair), (group, pair+1), (group+8, pair), (group+8, pair+1)]
    b = [(pair, group), (pair+1, group)]
    c = [(group, pair), (group, pair+1), (group+8, pair), (group+8, pair+1)]
    return a, b, c


def _pack_pair(values):
    bits = np.asarray(values, dtype=np.float16).view(np.uint16).astype(np.uint32)
    return bits[0] | (bits[1] << np.uint32(16))


def _unpack_pair(register):
    return np.asarray([register & 65535, register >> np.uint32(16)], dtype=np.uint16).view(np.float16)


def test_nvidia_m16n8k8_fragment_coverage_and_packing():
    a = np.arange(128, dtype=np.float16).reshape(16, 8) - 64
    b = np.arange(64, dtype=np.float16).reshape(8, 8) - 32
    rebuilt_a, rebuilt_b = np.zeros_like(a), np.zeros_like(b)
    seen_a = np.zeros(a.shape, dtype=np.int32)
    seen_b = np.zeros(b.shape, dtype=np.int32)
    seen_c = np.zeros((16, 8), dtype=np.int32)
    for lane in range(32):
        ac, bc, cc = _coordinates(lane)
        a_registers = [_pack_pair([a[ac[j]], a[ac[j+1]]]) for j in (0, 2)]
        b_register = _pack_pair([b[bc[0]], b[bc[1]]])
        av = np.concatenate([_unpack_pair(reg) for reg in a_registers])
        bv = _unpack_pair(b_register)
        for index, value in zip(ac, av):
            rebuilt_a[index] = value
            seen_a[index] += 1
        for index, value in zip(bc, bv):
            rebuilt_b[index] = value
            seen_b[index] += 1
        for index in cc:
            seen_c[index] += 1
    np.testing.assert_array_equal(rebuilt_a, a)
    np.testing.assert_array_equal(rebuilt_b, b)
    assert np.all(seen_a == 1) and np.all(seen_b == 1) and np.all(seen_c == 1)


def _fragment_model(x, weight):
    """Model four physical warps, half2 registers, zero padding and C stores."""
    tokens, depth = x.shape
    vocab = weight.shape[0]
    out = np.empty((tokens, vocab), dtype=np.float16)
    writers = np.zeros((tokens, vocab), dtype=np.int32)
    for start_m in range(0, tokens, 32):
        for start_v in range(0, vocab, 64):
            for warp in range(4):
                row_base = start_m + (warp & 1) * 16
                col_base = start_v + (warp >> 1) * 32
                sums = [np.zeros((16, 8), dtype=np.float32) for _ in range(4)]
                for k in range(0, depth, 8):
                    atile = np.zeros((16, 8), dtype=np.float16)
                    btiles = [np.zeros((8, 8), dtype=np.float16) for _ in range(4)]
                    for lane in range(32):
                        ac, bc, _ = _coordinates(lane)
                        avalues = [
                            x[row_base+r, k+c] if row_base+r < tokens and k+c < depth else 0
                            for r, c in ac
                        ]
                        avalues = np.concatenate([
                            _unpack_pair(_pack_pair(avalues[:2])),
                            _unpack_pair(_pack_pair(avalues[2:])),
                        ])
                        for coord, value in zip(ac, avalues):
                            atile[coord] = value
                        for j in range(4):
                            bvalues = [
                                weight[col_base+8*j+c, k+r]
                                if col_base+8*j+c < vocab and k+r < depth else 0
                                for r, c in bc
                            ]
                            bvalues = _unpack_pair(_pack_pair(bvalues))
                            for coord, value in zip(bc, bvalues):
                                btiles[j][coord] = value
                    for j in range(4):
                        sums[j] += atile.astype(np.float32) @ btiles[j].astype(np.float32)
                for lane in range(32):
                    _, _, cc = _coordinates(lane)
                    for j in range(4):
                        for r, c in cc:
                            gr, gc = row_base+r, col_base+8*j+c
                            if gr < tokens and gc < vocab:
                                out[gr, gc] = sums[j][r, c]
                                writers[gr, gc] += 1
    return out, writers


@pytest.mark.parametrize("shape", [(1, 137, 35), (33, 65, 17), (32, 64, 16), (3, 11, 1536), (1, 3, 0)])
def test_fragment_model_masked_projection(shape):
    tokens, vocab, depth = shape
    rng = np.random.default_rng(917)
    x = rng.standard_normal((tokens, depth)).astype(np.float16)
    weight = (rng.standard_normal((vocab, depth)) / max(depth, 1)**0.5).astype(np.float16)
    actual, writers = _fragment_model(x, weight)
    expected = (x.astype(np.float32) @ weight.astype(np.float32).T).astype(np.float16)
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), rtol=3e-3, atol=2e-3)
    assert np.all(writers == 1), "every non-padding output must have exactly one writer"


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
    reason="actual inline PTX requires CUDA execution",
)
@pytest.mark.parametrize("shape", [(1, 137, 35), (33, 65, 17), (32, 64, 16), (5, 129, 1536), (256, 151936, 1536)])
def test_t4_projection_on_cuda(shape):
    from solution.head_matmul_t4 import project_fp16

    tokens, vocab, depth = shape
    torch.manual_seed(917)
    x = torch.randn((tokens, depth), device="cuda", dtype=torch.float16)
    weight = torch.randn((vocab, depth), device="cuda", dtype=torch.float16) / depth**0.5
    actual = project_fp16(x, weight)
    expected = torch.mm(x, weight.T)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


def test_t4_projection_refuses_nonempty_cpu():
    from solution.head_matmul_t4 import project_fp16

    with pytest.raises(ValueError, match="requires CUDA"):
        project_fp16(torch.zeros((1, 3), dtype=torch.float16), torch.zeros((2, 3), dtype=torch.float16))


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
    reason="actual inline PTX requires CUDA execution",
)
def test_t4_projection_unaligned_contiguous_views_on_cuda():
    from solution.head_matmul_t4 import project_fp16

    # Odd FP16 storage offsets deliberately make the pointers only 2-byte
    # aligned while preserving contiguity and an even hidden dimension.
    x = torch.randn(3*16+1, device="cuda", dtype=torch.float16)[1:].view(3, 16)
    weight = torch.randn(9*16+1, device="cuda", dtype=torch.float16)[1:].view(9, 16)
    assert x.is_contiguous() and weight.is_contiguous()
    assert x.data_ptr() % 4 == 2 and weight.data_ptr() % 4 == 2
    actual = project_fp16(x, weight)
    expected = torch.mm(x, weight.T)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
