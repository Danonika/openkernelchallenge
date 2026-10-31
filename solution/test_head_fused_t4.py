"""CPU layout/algebra checks and separately gated real-device fused-head tests.

The CPU model is not an interpreter for inline PTX. Real CUDA execution and
sanitizer tests remain required before enabling this research path.
"""

import os

import numpy as np
import pytest
import torch
import triton

from solution.head_fused_t4 import (
    fused_head_statistics_fp16, fused_head_logit_gradient_fp16,
    soup_head_t4_merge_statistics,
)


def test_cooperative_stage_copy_and_output_coverage():
    for height, copies in [(64, 8), (128, 16)]:
        writers = np.zeros((height, 20), dtype=np.int32)
        for tid in range(128):
            for i in range(copies):
                index = tid + i * 128
                row, pair = divmod(index, 16)
                writers[row, pair] += 1
        assert np.all(writers[:, :16] == 1)
        assert np.all(writers[:, 16:] == 0), "padding is never staged or consumed"
    output_writers = np.zeros((64, 128), dtype=np.int32)
    partial_writers = np.zeros((64, 2), dtype=np.int32)
    final_writers = np.zeros(64, dtype=np.int32)
    for tid in range(128):
        warp, lane = divmod(tid, 32)
        wm, wn = warp & 1, warp >> 1
        group, pair = lane >> 2, (lane & 3) * 2
        for q in range(4):
            row = wm * 32 + group + q * 8
            for j in range(8):
                for h in range(2):
                    output_writers[row, wn * 64 + pair + j * 8 + h] += 1
            if pair == 0:
                partial_writers[row, wn] += 1
                if wn == 0:
                    final_writers[row] += 1
    assert np.all(output_writers == 1)
    assert np.all(partial_writers == 1)
    assert np.all(final_writers == 1)


def test_shared_fragment_reads_visit_all_32_banks_without_conflict():
    # This checks the standard four-byte bank model, not measured throughput.
    for warp in range(4):
        for inner in range(4):
            for rows_per_warp, blocks in [(32, 4), (64, 8)]:
                for block in range(blocks):
                    banks = []
                    for lane in range(32):
                        row = ((warp & 1) if rows_per_warp == 32 else (warp >> 1)) * rows_per_warp
                        row += lane // 4 + block * 8
                        channel = inner * 8 + 2 * (lane % 4)
                        banks.append(((row * 40 + channel) * 2 // 4) % 32)
                    assert sorted(banks) == list(range(32))


def _staged_projection_model(x, weight):
    """Reconstruct documented MMA fragments from our padded shared addresses."""
    rows, depth = x.shape
    vocab = weight.shape[0]
    out = np.empty((rows, vocab), dtype=np.float16)
    for m0 in range(0, rows, 64):
        for v0 in range(0, vocab, 128):
            acc = np.zeros((4, 2, 8, 16, 8), dtype=np.float32)
            for k0 in range(0, depth, 32):
                shared_a = np.zeros((64, 40), dtype=np.float16)
                shared_b = np.zeros((128, 40), dtype=np.float16)
                for source, shared, start, copies in [(x, shared_a, m0, 8), (weight, shared_b, v0, 16)]:
                    for tid in range(128):
                        for i in range(copies):
                            row, pair = divmod(tid + 128 * i, 16)
                            if start + row < source.shape[0]:
                                shared[row, 2 * pair:2 * pair + 2] = source[start + row, k0 + 2 * pair:k0 + 2 * pair + 2]
                for warp in range(4):
                    wm, wn = warp & 1, warp >> 1
                    for inner in range(4):
                        afragment = np.zeros((2, 16, 8), dtype=np.float16)
                        bfragment = np.zeros((8, 8, 8), dtype=np.float16)
                        for lane in range(32):
                            group, pair = lane // 4, 2 * (lane % 4)
                            for mb in range(2):
                                for row_half in range(2):
                                    row = group + row_half * 8
                                    afragment[mb, row, pair:pair+2] = shared_a[wm*32 + mb*16 + row, inner*8 + pair:inner*8 + pair+2]
                            for j in range(8):
                                bfragment[j, pair:pair+2, group] = shared_b[wn*64 + j*8 + group, inner*8 + pair:inner*8 + pair+2]
                        for mb in range(2):
                            for j in range(8):
                                acc[warp, mb, j] += afragment[mb].astype(np.float32) @ bfragment[j].astype(np.float32)
            for warp in range(4):
                for lane in range(32):
                    wm, wn = warp & 1, warp >> 1
                    group, pair = lane // 4, 2 * (lane % 4)
                    for q in range(4):
                        mb, half = divmod(q, 2)
                        row = m0 + wm*32 + group + q*8
                        for j in range(8):
                            for h in range(2):
                                col = v0 + wn*64 + pair + j*8 + h
                                if row < rows and col < vocab:
                                    out[row, col] = acc[warp, mb, j, group+half*8, pair+h]
    return out


@pytest.mark.parametrize("shape", [(1, 1, 32), (3, 65, 64), (65, 129, 32), (3, 131, 1536)])
def test_staged_fragment_model_against_dense_projection(shape):
    rows, vocab, depth = shape
    rng = np.random.default_rng(584)
    x = rng.normal(size=(rows, depth)).astype(np.float16)
    w = (rng.normal(size=(vocab, depth)) / depth**0.5).astype(np.float16)
    actual = _staged_projection_model(x, w)
    expected = (x.astype(np.float32) @ w.astype(np.float32).T).astype(np.float16)
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), rtol=3e-3, atol=2e-3)


def test_staged_projection_model_is_independent_of_chunk_row_position():
    rng = np.random.default_rng(915)
    x = rng.normal(size=(65, 64)).astype(np.float16)
    w = (rng.normal(size=(129, 64)) / 8).astype(np.float16)
    whole = _staged_projection_model(x, w)
    selected = np.asarray([64, 0, 31, 32, 1, 63])
    compacted = _staged_projection_model(x[selected], w)
    chunked = np.concatenate([_staged_projection_model(x[part], w) for part in np.array_split(selected, 3)])
    np.testing.assert_array_equal(compacted, whole[selected])
    np.testing.assert_array_equal(chunked, whole[selected])


def _gradient_epilogue_model(scores, targets, row_max, log_normalizer, scale):
    """FP32 math plus the physical writer mapping; not a PTX interpreter."""
    rows, vocab = scores.shape
    centered = scores.astype(np.float16).astype(np.float32) - row_max[:, None]
    centered = centered - log_normalizer[:, None]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        # Hardware ex2.approx has its own approximation; this models the
        # explicit FP32 boundaries and writer coordinates, not its ULP error.
        probability = np.exp2(centered * np.float32(1.4426950408889634)).astype(np.float32)
        indicator = np.arange(vocab)[None, :] == targets[:, None]
        target_term = np.where(indicator, -np.float32(scale), np.float32(0.0))
        # Float64 holds the exact product of two FP32 operands. Its addition
        # models a single final FP32 rounding for these cases, rather than an
        # intermediate rounded (probability - indicator). This does not model
        # arbitrary hardware FMA edge cases or the device exponent's ULP error.
        derivative = (probability.astype(np.float64) * np.float64(np.float32(scale))
                      + target_term.astype(np.float64)).astype(np.float32)
        derivative = np.where(targets[:, None] != -100, derivative, np.float32(0.0)).astype(np.float16)
    output = np.empty((rows, vocab), dtype=np.float16)
    writers = np.zeros((rows, vocab), dtype=np.uint8)
    for m0 in range(0, rows, 64):
        for v0 in range(0, vocab, 128):
            for tid in range(128):
                warp, lane = divmod(tid, 32)
                wm, wn = warp & 1, warp >> 1
                group, pair = lane // 4, 2 * (lane % 4)
                for q in range(4):
                    row = m0 + wm * 32 + group + q * 8
                    if row >= rows:
                        continue
                    for j in range(8):
                        for h in range(2):
                            column = v0 + wn * 64 + pair + j * 8 + h
                            if column < vocab:
                                output[row, column] = derivative[row, column]
                                writers[row, column] += 1
    return output, writers


@pytest.mark.parametrize("scale", [1.0, 65536.0 / 17, -3.0, 0.0])
def test_gradient_epilogue_model_matches_scaled_cross_entropy(scale):
    torch.manual_seed(762)
    logits = torch.randn((65, 129)).half().float().requires_grad_()
    targets = torch.randint(0, 129, (65,))
    targets[::3] = -100
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum") * scale
    loss.backward()
    row_max = logits.detach().amax(-1)
    log_normalizer = (logits.detach() - row_max[:, None]).logsumexp(-1)
    actual, writers = _gradient_epilogue_model(
        logits.detach().numpy(), targets.numpy(), row_max.numpy(), log_normalizer.numpy(), scale,
    )
    np.testing.assert_allclose(actual.astype(np.float32), logits.grad.half().float().numpy(), rtol=2e-3, atol=2e-3)
    assert np.all(writers == 1)
    assert np.all(actual[targets.numpy() == -100].view(np.uint16) == 0)


@pytest.mark.parametrize("scale", [float("inf"), -float("inf"), float("nan")])
def test_gradient_epilogue_model_ignored_rows_are_positive_zero_with_nonfinite_scale(scale):
    actual, writers = _gradient_epilogue_model(
        np.zeros((3, 17), dtype=np.float16), np.full(3, -100),
        np.zeros(3, dtype=np.float32), np.full(3, np.log(17), dtype=np.float32), scale,
    )
    assert np.all(actual.view(np.uint16) == 0)
    assert np.all(writers == 1)


@pytest.mark.parametrize("scale", [float("inf"), -float("inf"), float("nan")])
def test_gradient_epilogue_model_matches_native_nonfinite_scale(scale):
    logits = torch.tensor([[-2.0, 0.0, 2.0]] * 3, requires_grad=True)
    targets = torch.tensor([0, 1, -100])
    (torch.nn.functional.cross_entropy(logits, targets, reduction="sum") * scale).backward()
    row_max = logits.detach().amax(-1)
    log_normalizer = (logits.detach() - row_max[:, None]).logsumexp(-1)
    actual, writers = _gradient_epilogue_model(
        logits.detach().numpy(), targets.numpy(), row_max.numpy(), log_normalizer.numpy(), scale,
    )
    np.testing.assert_equal(actual, logits.grad.half().numpy())
    assert np.all(actual[2].view(np.uint16) == 0)
    assert np.all(writers == 1)


def test_gradient_epilogue_model_preserves_large_common_logit_offset():
    vocab, offset, scale = 129, 65504.0, 65536.0 / 17
    targets = np.asarray([0, vocab - 1, -100])
    zero_scores = np.zeros((3, vocab), dtype=np.float16)
    log_normalizer = np.full(3, np.log(vocab), dtype=np.float32)
    base, _ = _gradient_epilogue_model(
        zero_scores, targets, np.zeros(3, dtype=np.float32), log_normalizer, scale,
    )
    shifted, writers = _gradient_epilogue_model(
        zero_scores + np.float16(offset), targets,
        np.full(3, offset, dtype=np.float32), log_normalizer, scale,
    )
    logits = torch.full((3, vocab), offset, requires_grad=True)
    (torch.nn.functional.cross_entropy(logits, torch.from_numpy(targets), reduction="sum") * scale).backward()
    np.testing.assert_array_equal(shifted, base)
    np.testing.assert_allclose(shifted.astype(np.float32), logits.grad.numpy(), rtol=5e-4, atol=0)
    assert np.all(writers == 1)


def test_gradient_epilogue_model_scales_before_fp16_underflow():
    vocab, denominator, upstream = 151936, 2048, 65536
    scores = np.zeros((1, vocab), dtype=np.float16)
    targets = np.asarray([0])
    row_max = np.zeros(1, dtype=np.float32)
    log_normalizer = np.full(1, np.log(vocab), dtype=np.float32)
    actual, writers = _gradient_epilogue_model(scores, targets, row_max, log_normalizer, upstream / denominator)
    wrongly_precast = np.float16((1.0 / vocab) / denominator) * np.float32(upstream)
    assert wrongly_precast == 0.0
    assert np.all(actual[0, 1:] > 0), "the real scale must precede the FP16 gradient cast"
    assert actual[0, 0] < 0 and np.all(writers == 1)


def _make_partials(scores):
    rows, vocab = scores.shape
    tiles = triton.cdiv(vocab, 128)
    maxima = torch.empty((rows, tiles), dtype=torch.float32, device=scores.device)
    sums = torch.empty_like(maxima)
    centered_sums = torch.empty_like(maxima)
    winners = torch.empty((rows, tiles), dtype=torch.int32, device=scores.device)
    for tile in range(tiles):
        values = scores[:, tile*128:(tile+1)*128].float()
        maximum, winner = values.max(-1)
        centered = values - maximum[:, None]
        exponential = centered.exp()
        maxima[:, tile] = maximum
        sums[:, tile] = exponential.sum(-1)
        centered_sums[:, tile] = (exponential * centered).sum(-1)
        winners[:, tile] = winner + tile * 128
    return maxima, sums, centered_sums, winners


@pytest.mark.parametrize("vocab,offset", [(1, 0), (65, 0), (129, 2048), (129, 65504), (151936, 0)])
def test_merge_kernel_preserves_loss_entropy_masks_and_ties(vocab, offset):
    if os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available():
        pytest.skip("Triton merge needs CUDA or TRITON_INTERPRET=1")
    device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
    torch.manual_seed(584)
    scores = (torch.randn((4, vocab), device=device) * 2 + offset).half()
    scores[1] = offset  # Equal logits must choose the first vocabulary index.
    targets = torch.tensor([0, -100, vocab-1, vocab//2], device=device)
    target_score = scores.float().gather(1, targets.clamp_min(0)[:, None]).squeeze(1).contiguous()
    partials = _make_partials(scores)
    outputs = [torch.empty(4, device=device, dtype=t) for t in
               (torch.float32, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)]
    tiles = partials[0].shape[1]
    soup_head_t4_merge_statistics[(4,)](
        targets, target_score, *partials, *outputs, TILES=tiles,
        BLOCK=triton.next_power_of_2(tiles), num_warps=4,
    )
    expected_loss = torch.nn.functional.cross_entropy(scores.float(), targets, reduction="none")
    logp = scores.float().log_softmax(-1)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(outputs[0], expected_loss, rtol=3e-5, atol=2e-4)
    torch.testing.assert_close(outputs[1], expected_entropy, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(outputs[2], scores.argmax(-1), rtol=0, atol=0)
    torch.testing.assert_close(outputs[3], scores.float().logsumexp(-1), rtol=3e-6, atol=2e-4)
    torch.testing.assert_close(outputs[4], scores.float().amax(-1), rtol=0, atol=0)
    torch.testing.assert_close(outputs[5], (scores.float() - outputs[4][:, None]).logsumexp(-1), rtol=3e-6, atol=3e-6)


def test_fused_entrypoint_rejects_cpu_and_unsupported_alignment_or_depth():
    with pytest.raises(ValueError, match="actual CUDA"):
        fused_head_statistics_fp16(torch.zeros((1, 32), dtype=torch.float16), torch.zeros((1, 32), dtype=torch.float16), torch.zeros(1, dtype=torch.int64))
    with pytest.raises(ValueError, match="multiple of 32"):
        fused_head_statistics_fp16(torch.zeros((1, 33), dtype=torch.float16), torch.zeros((1, 33), dtype=torch.float16), torch.zeros(1, dtype=torch.int64))
    unaligned = torch.zeros(33, dtype=torch.float16)[1:].reshape(1, 32)
    with pytest.raises(ValueError, match="four-byte aligned"):
        fused_head_statistics_fp16(unaligned, torch.zeros((1, 32), dtype=torch.float16), torch.zeros(1, dtype=torch.int64))


def test_fused_gradient_entrypoint_guards_saved_statistics_scale_and_output_aliasing():
    x = torch.zeros((2, 32), dtype=torch.float16)
    w = torch.zeros((32, 32), dtype=torch.float16)
    targets = torch.zeros(2, dtype=torch.int64)
    row_max, log_normalizer, scale = torch.zeros(2), torch.ones(2), torch.ones(())
    with pytest.raises(ValueError, match="actual CUDA"):
        fused_head_logit_gradient_fp16(x, w, targets, row_max, log_normalizer, scale)
    for maximum, normalizer in [(row_max.half(), log_normalizer), (row_max, log_normalizer.half())]:
        with pytest.raises(TypeError, match="FP32"):
            fused_head_logit_gradient_fp16(x, w, targets, maximum, normalizer, scale)
    with pytest.raises(ValueError, match="correctly sized"):
        fused_head_logit_gradient_fp16(x, w, targets, row_max, log_normalizer, torch.ones(2))
    for maximum, normalizer in [(row_max[:1], log_normalizer), (row_max, log_normalizer[:1])]:
        with pytest.raises(ValueError, match="correctly sized"):
            fused_head_logit_gradient_fp16(x, w, targets, maximum, normalizer, scale)
    for output in (x, w[:2]):
        with pytest.raises(ValueError, match="overlap"):
            fused_head_logit_gradient_fp16(x, w, targets, row_max, log_normalizer, scale, out=output)
    output = torch.zeros((2, 32), dtype=torch.float16)
    aliased_state = output.view(torch.float32).flatten()[:2]
    for maximum, normalizer in [(aliased_state, log_normalizer), (row_max, aliased_state)]:
        with pytest.raises(ValueError, match="overlap"):
            fused_head_logit_gradient_fp16(x, w, targets, maximum, normalizer, scale, out=output)


CUDA_ASSEMBLY_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
    reason="fused inline PTX requires actual CUDA execution",
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
    reason="fused inline PTX requires actual CUDA execution",
)
@pytest.mark.parametrize("shape", [(1, 1, 32), (17, 65, 32), (65, 129, 64), (5, 151936, 1536)])
def test_fused_head_statistics_on_cuda(shape):
    rows, vocab, depth = shape
    torch.manual_seed(584)
    x = torch.randn((rows, depth), device="cuda", dtype=torch.float16)
    w = torch.randn((vocab, depth), device="cuda", dtype=torch.float16) / depth**0.5
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[0] = -100
    outputs = fused_head_statistics_fp16(x, w, targets)
    scores = torch.mm(x, w.T).float()
    logp = scores.log_softmax(-1)
    maximum = scores.amax(-1)
    expected = (torch.nn.functional.cross_entropy(scores, targets, reduction="none"),
                -(logp.exp() * logp).sum(-1), scores.argmax(-1), scores.logsumexp(-1),
                maximum, (scores - maximum[:, None]).logsumexp(-1))
    for index in (0, 1, 3, 4, 5):
        torch.testing.assert_close(outputs[index], expected[index], rtol=3e-4, atol=4e-3)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)
    # All-equal logits check global first-index tie handling independently.
    outputs = fused_head_statistics_fp16(x, torch.zeros_like(w), targets)
    assert torch.equal(outputs[2], torch.zeros_like(targets))
    torch.testing.assert_close(outputs[1], torch.full_like(outputs[1], np.log(vocab)), rtol=3e-5, atol=3e-5)


@CUDA_ASSEMBLY_ONLY
@pytest.mark.parametrize("shape", [(1, 1, 32), (17, 65, 32), (65, 129, 64), (5, 151936, 1536)])
@pytest.mark.parametrize("upstream", [1.0, 65536.0])
def test_fused_head_logit_gradient_on_cuda(shape, upstream):
    rows, vocab, depth = shape
    torch.manual_seed(762)
    x = torch.randn((rows, depth), device="cuda", dtype=torch.float16)
    w = torch.randn((vocab, depth), device="cuda", dtype=torch.float16) / depth**0.5
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[::3] = -100
    row_max, log_normalizer = fused_head_statistics_fp16(x, w, targets)[4:6]
    scale = torch.tensor(upstream / max(1, rows), dtype=torch.float32, device="cuda")
    # A contiguous halfword-offset output tests two-byte alignment plus guards
    # on both ends of the output allocation; x and w remain four-byte aligned.
    storage = torch.full((rows * vocab + 2,), 77.0, device="cuda", dtype=torch.float16)
    out = storage[1:-1].reshape(rows, vocab)
    actual = fused_head_logit_gradient_fp16(x, w, targets, row_max, log_normalizer, scale, out=out)
    assert actual is out
    logits = torch.mm(x, w.T).float().requires_grad_()
    (torch.nn.functional.cross_entropy(logits, targets, reduction="sum") * scale).backward()
    torch.testing.assert_close(actual, logits.grad.half(), rtol=4e-3, atol=3e-3)
    assert torch.all(actual[targets == -100].view(torch.int16) == 0)
    assert storage[0].item() == 77.0 and storage[-1].item() == 77.0


@CUDA_ASSEMBLY_ONLY
@pytest.mark.parametrize("scale_value", [float("inf"), -float("inf"), float("nan")])
def test_fused_head_logit_gradient_ignored_nonfinite_scale_on_cuda(scale_value):
    x = torch.zeros((3, 32), device="cuda", dtype=torch.float16)
    w = torch.zeros((17, 32), device="cuda", dtype=torch.float16)
    targets = torch.full((3,), -100, dtype=torch.int64, device="cuda")
    row_max = torch.zeros(3, device="cuda")
    log_normalizer = torch.full((3,), np.log(17), device="cuda")
    scale = torch.tensor(scale_value, dtype=torch.float32, device="cuda")
    actual = fused_head_logit_gradient_fp16(x, w, targets, row_max, log_normalizer, scale)
    assert torch.all(actual.view(torch.int16) == 0)


@CUDA_ASSEMBLY_ONLY
def test_fused_head_logit_gradient_scales_before_fp16_underflow_on_cuda():
    vocab, depth = 151936, 32
    x = torch.zeros((1, depth), device="cuda", dtype=torch.float16)
    w = torch.zeros((vocab, depth), device="cuda", dtype=torch.float16)
    targets = torch.zeros(1, dtype=torch.int64, device="cuda")
    row_max, log_normalizer = fused_head_statistics_fp16(x, w, targets)[4:6]
    scale = torch.tensor(65536.0 / 2048, dtype=torch.float32, device="cuda")
    actual = fused_head_logit_gradient_fp16(x, w, targets, row_max, log_normalizer, scale)
    expected = torch.full_like(actual, (1.0 / vocab) * scale.item())
    expected[0, 0] = (1.0 / vocab - 1.0) * scale.item()
    assert torch.all(actual[0, 1:] > 0)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=0)


@CUDA_ASSEMBLY_ONLY
def test_fused_head_logit_gradient_full_state_and_compacted_chunks_on_cuda():
    torch.manual_seed(915)
    x = torch.randn((65, 64), device="cuda", dtype=torch.float16)
    w = torch.randn((129, 64), device="cuda", dtype=torch.float16) / 8
    targets = torch.randint(0, 129, (65,), device="cuda")
    targets[::3] = -100
    row_max, log_normalizer = fused_head_statistics_fp16(x, w, targets)[4:6]
    scale = torch.tensor(65536.0 / 65, dtype=torch.float32, device="cuda")
    full = fused_head_logit_gradient_fp16(x, w, targets, row_max, log_normalizer, scale)
    selected = torch.tensor([64, 0, 31, 32, 1, 63], device="cuda")
    compact = fused_head_logit_gradient_fp16(
        x[selected], w, targets[selected], row_max[selected], log_normalizer[selected], scale,
    )
    chunked = torch.cat([
        fused_head_logit_gradient_fp16(x[index], w, targets[index], row_max[index], log_normalizer[index], scale)
        for index in selected.split(2)
    ])
    torch.testing.assert_close(compact, full[selected], rtol=0, atol=0)
    torch.testing.assert_close(chunked, full[selected], rtol=0, atol=0)


@CUDA_ASSEMBLY_ONLY
def test_fused_head_centered_state_preserves_large_common_offset_on_cuda():
    # The single nonzero channel makes both tensor-core and native GEMM
    # projections exactly representable, isolating normalization cancellation.
    rows, vocab, depth = 3, 129, 32
    x = torch.zeros((rows, depth), device="cuda", dtype=torch.float16)
    x[:, 0] = 1
    weight = torch.zeros((vocab, depth), device="cuda", dtype=torch.float16)
    targets = torch.tensor([0, vocab - 1, -100], device="cuda")
    scale = torch.tensor(65536.0 / 17, dtype=torch.float32, device="cuda")
    base = fused_head_statistics_fp16(x, weight, targets)
    base_gradient = fused_head_logit_gradient_fp16(x, weight, targets, base[4], base[5], scale)
    weight[:, 0] = 65504
    shifted = fused_head_statistics_fp16(x, weight, targets)
    shifted_gradient = fused_head_logit_gradient_fp16(x, weight, targets, shifted[4], shifted[5], scale)
    for index in (0, 1, 2, 5):
        torch.testing.assert_close(shifted[index], base[index], rtol=0, atol=0)
    torch.testing.assert_close(shifted[4], torch.full_like(shifted[4], 65504), rtol=0, atol=0)
    torch.testing.assert_close(shifted_gradient, base_gradient, rtol=0, atol=0)
    logits = torch.mm(x, weight.T).float().requires_grad_()
    reference_ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    (reference_ce.sum() * scale).backward()
    torch.testing.assert_close(shifted[0], reference_ce, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(shifted_gradient, logits.grad.half(), rtol=1e-3, atol=0)
