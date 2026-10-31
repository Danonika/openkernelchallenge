"""Numerical tests; use a CUDA GPU or TRITON_INTERPRET=1 on CPU.

Interpreter results test arithmetic and masks, not CUDA execution/performance.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from .streamed_head import _summarize_logits, streamed_linear_ce


HAS_BACKEND = torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="requires CUDA or the Triton CPU interpreter")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _reference(hidden, weight, labels, denominator=None):
    scores = F.linear(hidden.to(torch.float16), weight.to(torch.float16)).float()
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
    token_loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), shifted.reshape(-1), reduction="none")
    normalizer = (shifted != -100).sum() if denominator is None else denominator
    loss = token_loss.sum() / normalizer
    logp = F.log_softmax(scores, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    return loss, token_loss.reshape_as(labels), entropy, scores.argmax(dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
@pytest.mark.parametrize("denominator", [None, 17])
def test_loss_metrics_and_scaled_gradient(dtype, scale, denominator):
    torch.manual_seed(719)
    hidden = (torch.randn(2, 7, 33, device=DEVICE) * 0.2).to(dtype).requires_grad_()
    weight = torch.randn(129, 33, device=DEVICE, dtype=torch.float32) * 0.1
    labels = torch.randint(0, 129, (2, 7), device=DEVICE)
    labels[0, :3] = -100
    labels[1, 4:] = -100
    reference_hidden = hidden.detach().clone().requires_grad_()
    result = streamed_linear_ce(hidden, weight, labels, denominator, chunk_size=3)
    expected = _reference(reference_hidden, weight, labels, denominator)
    torch.testing.assert_close(result.loss, expected[0], rtol=3e-6, atol=2e-6)
    torch.testing.assert_close(result.token_loss, expected[1], rtol=3e-6, atol=2e-6)
    torch.testing.assert_close(result.entropy, expected[2], rtol=3e-6, atol=2e-6)
    assert torch.equal(result.predictions, expected[3])
    assert torch.equal(result.token_loss[:, -1], torch.zeros_like(result.token_loss[:, -1]))
    (result.loss * scale).backward()
    (expected[0] * scale).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.005, atol=max(4e-6, scale * 2e-7))
    assert result.loss.requires_grad
    assert not result.token_loss.requires_grad
    assert not result.entropy.requires_grad


@pytest.mark.parametrize("vocab", [4113, 151936])
def test_vocabulary_tiles_entropy_and_tie_break(vocab):
    torch.manual_seed(811)
    scores = (torch.randn(3, vocab, device=DEVICE) * 2.0).to(torch.float16)
    scores[0, 7] = 50
    scores[0, vocab - 1] = 50  # Equal maxima across tiles: first index must win.
    scores[1] += 100  # Entropy must remain stable under a large common offset.
    targets = torch.tensor([vocab - 1, 3, -100], device=DEVICE)
    ce, entropy, lse = [torch.empty(3, device=DEVICE, dtype=torch.float32) for _ in range(3)]
    predictions = torch.empty(3, device=DEVICE, dtype=torch.long)
    _summarize_logits(scores, targets, ce, entropy, predictions, lse)
    reference_scores = scores.float()
    logp = F.log_softmax(reference_scores, dim=-1)
    expected_ce = F.cross_entropy(reference_scores, targets, reduction="none")
    expected_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(ce, expected_ce, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(entropy, expected_entropy, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(lse, torch.logsumexp(reference_scores, -1), rtol=2e-6, atol=2e-5)
    assert torch.equal(predictions, reference_scores.argmax(-1))
    assert predictions[0].item() == 7


def test_all_ignored_preserves_mean_nan_and_zero_gradient():
    hidden = torch.randn(1, 3, 9, device=DEVICE, requires_grad=True)
    weight = torch.randn(17, 9, device=DEVICE)
    labels = torch.full((1, 3), -100, device=DEVICE, dtype=torch.long)
    result = streamed_linear_ce(hidden, weight, labels, chunk_size=2)
    assert torch.isnan(result.loss)
    assert torch.equal(result.token_loss, torch.zeros_like(result.token_loss))
    result.loss.backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))


def test_external_tensor_denominator_and_no_weight_gradient():
    hidden = torch.randn(1, 2, 9, device=DEVICE, requires_grad=True)
    weight = torch.randn(19, 9, device=DEVICE, dtype=torch.float16)
    labels = torch.tensor([[5, 8]], device=DEVICE)
    result = streamed_linear_ce(hidden, weight, labels, torch.tensor(3, device=DEVICE), chunk_size=2)
    expected = _reference(hidden, weight, labels, 3)
    torch.testing.assert_close(result.loss, expected[0], rtol=3e-6, atol=3e-6)
    with pytest.raises(ValueError, match="frozen"):
        streamed_linear_ce(hidden, weight.requires_grad_(), labels)


def test_original_projection_forward_and_backward_path():
    torch.manual_seed(983)
    hidden = (torch.randn(1, 3, 35, device=DEVICE) * 0.2).requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.randn(137, 35, device=DEVICE, dtype=torch.float16) * 0.1
    labels = torch.tensor([[-100, 7, 32]], device=DEVICE)
    result = streamed_linear_ce(hidden, weight, labels, chunk_size=2, projection="triton", numerics="guarded")
    expected = _reference(reference_hidden, weight, labels)
    torch.testing.assert_close(result.loss, expected[0], rtol=1e-5, atol=5e-6)
    torch.testing.assert_close(result.entropy, expected[2], rtol=1e-5, atol=5e-6)
    assert torch.equal(result.predictions, expected[3])
    (result.loss * 65536).backward()
    (expected[0] * 65536).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.005, atol=0.03)


@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_sparse_backward_matches_dense_backward(scale):
    torch.manual_seed(1057)
    hidden = (torch.randn(2, 9, 35, device=DEVICE) * 0.2).requires_grad_()
    dense_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.randn(137, 35, device=DEVICE, dtype=torch.float16) * 0.1
    labels = torch.randint(0, 137, (2, 9), device=DEVICE)
    labels[0, :5] = -100
    labels[1, :2] = -100
    labels[1, 5:] = -100
    sparse = streamed_linear_ce(hidden, weight, labels, chunk_size=4, sparse_backward=True)
    dense = streamed_linear_ce(dense_hidden, weight, labels, chunk_size=4, sparse_backward=False)
    torch.testing.assert_close(sparse.loss, dense.loss, rtol=0, atol=0)
    torch.testing.assert_close(sparse.entropy, dense.entropy, rtol=0, atol=0)
    (sparse.loss * scale).backward()
    (dense.loss * scale).backward()
    torch.testing.assert_close(hidden.grad, dense_hidden.grad, rtol=0.005, atol=max(4e-6, scale * 2e-7))
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
    assert torch.equal(hidden.grad[shifted == -100], torch.zeros_like(hidden.grad[shifted == -100]))


NATIVE_PRECISION_CASES = (
    "guard_boundary_above", "guard_boundary_below", "cross_tile_ties",
    "uniform_ties", "ignored_negative_inf",
)


def _check_native_precision_case(kind, device, width, sparse):
    if kind.startswith("guard_boundary"):
        vocab = 151936
        labels = torch.tensor([[-100, 0, -100]], device=device)
        high = 32.0
        gap = 16.53125 if kind.endswith("above") else 16.5625
        column = torch.full((vocab,), high - gap, device=device)
        column[0] = high
    elif kind == "cross_tile_ties":
        vocab = 4103
        labels = torch.tensor([[-100, 3, 2048, 4098]], device=device)
        column = torch.zeros(vocab, device=device)
        column[[3, 2048, 4098]] = 3
    elif kind == "uniform_ties":
        vocab = 37
        labels = torch.tensor([[-100, 0, 1]], device=device)
        column = torch.zeros(vocab, device=device)
    else:
        assert kind == "ignored_negative_inf"
        vocab = 3
        labels = torch.tensor([[-100, 1, -100]], device=device)
        column = torch.tensor([float("inf"), 0.0, 1.0], device=device)
    hidden = torch.zeros((*labels.shape, width), device=device)
    hidden[..., 0] = -1 if kind == "ignored_negative_inf" else 1
    hidden.requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.zeros((vocab, width), device=device, dtype=torch.float16)
    weight[:, 0] = column
    actual = streamed_linear_ce(hidden, weight, labels, 2048, chunk_size=2, sparse_backward=sparse)
    expected = _reference(reference_hidden, weight, labels, 2048)
    for observed, wanted in zip((actual.loss, actual.token_loss, actual.entropy), expected[:3]):
        torch.testing.assert_close(observed, wanted, rtol=1e-6, atol=2e-6, equal_nan=True)
    assert torch.equal(actual.predictions, expected[3])
    (actual.loss * 65536).backward()
    (expected[0] * 65536).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.005, atol=0.02, equal_nan=True)
    if kind == "ignored_negative_inf":
        assert torch.isnan(hidden.grad[0, -1, 0]), "ignored zero dlogits times Inf must retain native NaN dH"


@pytest.mark.parametrize("kind", NATIVE_PRECISION_CASES)
@pytest.mark.parametrize("sparse", [False, True])
def test_native_head_precision_regressions(kind, sparse):
    # A one-channel exact projection isolates reductions from GEMM drift.
    # Above the old .01 guard, the CPU native CE is .01050030347 while the
    # custom summary is .01000213437. Merely centering the custom state is
    # insufficient; the default must use the backend's actual CE arithmetic.
    _check_native_precision_case(kind, "cpu", 1, sparse)


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
                    reason="native head needs physical CUDA validation without the interpreter")
@pytest.mark.parametrize("kind", NATIVE_PRECISION_CASES)
@pytest.mark.parametrize("sparse", [False, True])
def test_native_head_precision_cuda_only(kind, sparse):
    _check_native_precision_case(kind, "cuda", 1536, sparse)


def test_native_head_uses_bounded_native_chunks_and_padded_causal_indices(monkeypatch):
    from . import streamed_head

    def custom_statistics_are_forbidden(*args, **kwargs):
        raise AssertionError("default native mode must not launch discarded original head work")

    monkeypatch.setattr(streamed_head, "_summarize_logits", custom_statistics_are_forbidden)
    monkeypatch.setattr(streamed_head, "_project", custom_statistics_are_forbidden)
    original_statistics = streamed_head._native_row_statistics
    original_gradient = streamed_head._native_logit_gradient
    forward_rows, backward_targets = [], []

    def statistics(scores, targets):
        forward_rows.append(scores.shape[0])
        return original_statistics(scores, targets)

    def gradient(scores, targets, scale):
        backward_targets.append(targets.clone())
        assert scores.shape[0] <= 3
        return original_gradient(scores, targets, scale)

    monkeypatch.setattr(streamed_head, "_native_row_statistics", statistics)
    monkeypatch.setattr(streamed_head, "_native_logit_gradient", gradient)
    hidden = torch.zeros((2, 4, 32))
    hidden[..., 0] = torch.arange(1, 9).reshape(2, 4)
    hidden.requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.zeros((17, 32), dtype=torch.float16)
    weight[:, 0] = torch.arange(1, 18) / 1000
    labels = torch.full((2, 4), -100, dtype=torch.long)
    labels[0, -1], labels[1, -1] = 1, 2
    actual = streamed_linear_ce(hidden, weight, labels, 17, chunk_size=3)
    expected = _reference(reference_hidden, weight, labels, 17)
    (actual.loss * 65536).backward()
    (expected[0] * 65536).backward()
    assert forward_rows == [3, 3, 2]
    assert torch.equal(torch.cat(backward_targets), torch.tensor([1, 2]))
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.005, atol=0.02)
    active_rows = torch.nonzero(hidden.grad.reshape(8, 32).abs().sum(-1), as_tuple=False).flatten()
    # Flattening an unpadded B*(T-1) mask would incorrectly select [2,5].
    assert torch.equal(active_rows, torch.tensor([2, 6]))


def test_native_numerics_rejects_incompatible_projection_and_unknown_mode():
    hidden = torch.zeros((1, 2, 32))
    weight = torch.zeros((17, 32))
    labels = torch.zeros((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="native numerics requires"):
        streamed_linear_ce(hidden, weight, labels, projection="t4_fused")
    with pytest.raises(ValueError, match="numerics must"):
        streamed_linear_ce(hidden, weight, labels, numerics="unknown")


@pytest.mark.parametrize("high,gap", [(100.0, 24.0), (1000.0, 20.0)])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
@pytest.mark.parametrize("sparse", [False, True])
def test_saturated_large_offset_rows_use_native_loss_and_scaled_backward(high, gap, scale, sparse):
    # The old saved-LSE path lost the small normalizer in FP32, then produced
    # dH=-21.46875 instead of -6.23046875 for the first large-scale CPU case.
    # Near-certainty also magnifies reduction-tree differences, so this must
    # exercise native guarded rows rather than merely loosen a tolerance.
    hidden = torch.ones((1, 2, 1), device=DEVICE, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.full((151936, 1), high - gap, device=DEVICE)
    weight[0] = high
    labels = torch.tensor([[-100, 0]], device=DEVICE)
    actual = streamed_linear_ce(hidden, weight, labels, chunk_size=2, sparse_backward=sparse, numerics="guarded")
    expected = _reference(reference_hidden, weight, labels)
    torch.testing.assert_close(actual.loss, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual.token_loss, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual.entropy, expected[2], rtol=0, atol=0)
    (actual.loss * scale).backward()
    (expected[0] * scale).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["hidden_nan", "hidden_inf", "weight_inf", "weight_nan"])
@pytest.mark.parametrize("sparse", [False, True])
def test_ignored_nonfinite_rows_preserve_native_gradient_and_entropy(kind, sparse):
    hidden = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]], device=DEVICE)
    weight = torch.tensor([[1., 0.], [2., 1.], [-1., -2.]], device=DEVICE)
    if kind.startswith("hidden"):
        hidden[0, 1, 0] = float("nan") if kind.endswith("nan") else float("inf")
    else:
        weight[0, 0] = float("nan") if kind.endswith("nan") else float("inf")
    hidden.requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    labels = torch.tensor([[-100, 1, -100]], device=DEVICE)
    actual = streamed_linear_ce(hidden, weight, labels, chunk_size=2, sparse_backward=sparse)
    expected = _reference(reference_hidden, weight, labels)
    torch.testing.assert_close(actual.loss, expected[0], rtol=3e-6, atol=2e-6, equal_nan=True)
    torch.testing.assert_close(actual.entropy, expected[2], rtol=3e-6, atol=2e-6, equal_nan=True)
    assert torch.equal(actual.predictions, expected[3])
    (actual.loss * 65536).backward()
    (expected[0] * 65536).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.005, atol=0.02, equal_nan=True)
    assert torch.isnan(hidden.grad).any(), "an ignored nonfinite row must not be silently zeroed"


def test_centered_statistics_are_invariant_to_exact_large_common_offset():
    scores = torch.zeros((1, 17), device=DEVICE, dtype=torch.float16)
    scores[0, 0] = 12
    target = torch.zeros(1, device=DEVICE, dtype=torch.long)
    observations = []
    for offset in (0., 1024.):
        ce, entropy, lse = [torch.empty(1, device=DEVICE) for _ in range(3)]
        prediction = torch.empty(1, device=DEVICE, dtype=torch.long)
        maximum, log_total = _summarize_logits(scores + offset, target, ce, entropy, prediction, lse)
        observations.append((ce, entropy, log_total))
        torch.testing.assert_close(maximum, torch.tensor([12. + offset], device=DEVICE), rtol=0, atol=0)
    for before, after in zip(*observations):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
                    reason="inline PTX requires real CUDA execution, not the CPU interpreter")
@pytest.mark.parametrize("length,width,vocab", [(3, 35, 137), (19, 128, 4113), (17, 1536, 151936)])
def test_t4_mma_head_cuda_only(length, width, vocab):
    if torch.cuda.get_device_capability()[0] < 7 or torch.cuda.get_device_capability() < (7, 5):
        pytest.skip("mma.m16n8k8 FP16 requires sm75 or later")
    torch.manual_seed(1193)
    hidden = (torch.randn(1, length, width, device="cuda") * 0.2).requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.randn(vocab, width, device="cuda", dtype=torch.float16) * 0.1
    labels = torch.randint(0, vocab, (1, length), device="cuda")
    labels[:, : max(1, length // 3)] = -100
    result = streamed_linear_ce(hidden, weight, labels, chunk_size=16, projection="t4_mma", numerics="guarded")
    expected = _reference(reference_hidden, weight, labels)
    torch.testing.assert_close(result.loss, expected[0], rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(result.token_loss, expected[1], rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(result.entropy, expected[2], rtol=3e-4, atol=3e-4)
    # FP16 output rounding can create near-ties for two different GEMMs. Require
    # the chosen scores to remain within one small FP16 rounding neighborhood.
    reference_scores = F.linear(reference_hidden.detach().half(), weight).float()
    chosen_scores = reference_scores.gather(-1, result.predictions.unsqueeze(-1)).squeeze(-1)
    reference_max = reference_scores.amax(-1)
    torch.testing.assert_close(chosen_scores, reference_max, rtol=0.002, atol=0.002)
    (result.loss * 65536).backward()
    (expected[0] * 65536).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.02, atol=0.08)


def _install_cpu_fused_substitute(monkeypatch):
    """Instrument ONLY the unavailable CUDA assembly API for orchestration.

    This does not simulate PTX execution or establish GPU numerical accuracy.
    The reference projection is explicitly CPU-only, while production keeps
    the real CUDA-only helper and never selects this substitute itself.
    """
    from . import head_fused_t4
    from . import streamed_head

    trace = {"statistics": [], "gradient": []}

    def statistics(x, weight, targets):
        assert x.device.type == weight.device.type == targets.device.type == "cpu"
        assert x.dtype == weight.dtype == torch.float16
        assert not torch.is_autocast_enabled("cpu")
        scores = torch.mm(x, weight.t()).float()
        lse = torch.logsumexp(scores, dim=-1)
        maximum = scores.amax(-1)
        log_total = torch.logsumexp(scores - maximum[:, None], dim=-1)
        valid = targets != -100
        correct = scores.gather(1, targets.clamp_min(0).unsqueeze(1)).squeeze(1)
        token_loss = torch.where(valid, -((correct - maximum) - log_total), 0.0)
        logp = (scores - maximum[:, None]) - log_total[:, None]
        entropy = -(logp.exp() * logp).sum(-1)
        trace["statistics"].append({
            "hidden": x.clone(), "targets": targets.clone(), "lse": lse.clone(),
            "maximum": maximum.clone(), "log_total": log_total.clone(),
        })
        return token_loss, entropy, scores.argmax(-1), lse, maximum, log_total

    def gradient(x, weight, targets, row_max, log_normalizer, loss_gradient_over_denominator, out=None):
        assert x.device.type == "cpu" and not torch.is_autocast_enabled("cpu")
        assert loss_gradient_over_denominator.dtype == torch.float32
        assert loss_gradient_over_denominator.shape == ()
        # Verify that backward receives the original centered normalization,
        # including the exact row selection used by sparse backward.
        source = trace["statistics"][-1]
        matching = (x[:, None, :] == source["hidden"][None, :, :]).all(-1)
        assert matching.any(-1).all()
        indices = matching.to(torch.int64).argmax(-1)
        torch.testing.assert_close(row_max, source["maximum"][indices], rtol=0, atol=0)
        torch.testing.assert_close(log_normalizer, source["log_total"][indices], rtol=0, atol=0)
        scores = torch.mm(x, weight.t()).float()
        probability = ((scores - row_max[:, None]) - log_normalizer[:, None]).exp()
        indicator = torch.arange(weight.shape[0])[None, :] == targets[:, None]
        value = (probability - indicator.float()) * loss_gradient_over_denominator
        value = torch.where((targets != -100)[:, None], value, 0.0).half()
        trace["gradient"].append({
            "hidden": x.clone(), "targets": targets.clone(),
            "scale": loss_gradient_over_denominator.clone(),
            "maximum": row_max.clone(), "log_total": log_normalizer.clone(),
        })
        if out is not None:
            out.copy_(value)
            return out
        return value

    def ordinary_projection_is_forbidden(*args, **kwargs):
        raise AssertionError("t4_fused must not recompute logits with another projection")

    monkeypatch.setattr(head_fused_t4, "fused_head_statistics_fp16", statistics)
    monkeypatch.setattr(head_fused_t4, "fused_head_logit_gradient_fp16", gradient)
    monkeypatch.setattr(streamed_head, "_project", ordinary_projection_is_forbidden)
    return trace


@pytest.mark.parametrize("sparse,denominator,scale", [
    (False, None, 1.0), (True, None, 65536.0),
    (False, 13, 65536.0), (True, 13, 1.0),
])
def test_t4_fused_orchestration_all_forward_rows_and_scaled_backward(monkeypatch, sparse, denominator, scale):
    trace = _install_cpu_fused_substitute(monkeypatch)
    torch.manual_seed(1293)
    hidden = (torch.randn(2, 5, 32) * 0.2).requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.randn(137, 32) * 0.1
    labels = torch.tensor([[13, -100, 4, 3, 12], [2, 7, -100, -100, -100]])
    result = streamed_linear_ce(
        hidden, weight, labels, denominator, chunk_size=3,
        projection="t4_fused", sparse_backward=sparse, numerics="guarded",
    )
    expected = _reference(reference_hidden, weight, labels, denominator)
    torch.testing.assert_close(result.loss, expected[0], rtol=3e-6, atol=2e-6)
    torch.testing.assert_close(result.token_loss, expected[1], rtol=3e-6, atol=2e-6)
    torch.testing.assert_close(result.entropy, expected[2], rtol=3e-6, atol=2e-6)
    assert torch.equal(result.predictions, expected[3])
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:].reshape(-1)
    assert len(trace["statistics"]) == 1
    assert trace["statistics"][0]["hidden"].shape == (10, 32)
    assert torch.equal(trace["statistics"][0]["targets"], shifted)
    assert not result.token_loss.requires_grad and not result.entropy.requires_grad
    (result.loss * scale).backward()
    (expected[0] * scale).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.005, atol=max(4e-6, scale * 2e-7))
    expected_rows = torch.nonzero(shifted != -100).flatten() if sparse else torch.arange(10)
    calls = trace["gradient"]
    assert len(calls) == (len(expected_rows) + 2) // 3
    assert torch.equal(torch.cat([c["targets"] for c in calls]), shifted[expected_rows])
    torch.testing.assert_close(
        torch.cat([c["hidden"] for c in calls]), hidden.detach().half().reshape(10, 32)[expected_rows], rtol=0, atol=0,
    )
    divisor = (shifted != -100).sum() if denominator is None else denominator
    for call in calls:
        torch.testing.assert_close(call["scale"], torch.as_tensor(scale / divisor, dtype=torch.float32), rtol=0, atol=0)
    assert torch.equal(hidden.grad.reshape(10, 32)[shifted == -100], torch.zeros_like(hidden.grad.reshape(10, 32)[shifted == -100]))


@pytest.mark.parametrize("sparse,denominator", [(True, None), (False, None), (True, 7), (False, 7)])
def test_t4_fused_orchestration_all_ignored_rows(monkeypatch, sparse, denominator):
    trace = _install_cpu_fused_substitute(monkeypatch)
    hidden = torch.randn(2, 3, 32, requires_grad=True)
    weight = torch.randn(17, 32) * 0.1
    labels = torch.full((2, 3), -100, dtype=torch.long)
    result = streamed_linear_ce(
        hidden, weight, labels, denominator, chunk_size=4,
        projection="t4_fused", sparse_backward=sparse, numerics="guarded",
    )
    assert torch.isnan(result.loss) if denominator is None else result.loss.item() == 0.0
    assert torch.equal(result.token_loss, torch.zeros_like(result.token_loss))
    assert torch.isfinite(result.entropy).all()
    assert len(trace["statistics"]) == 1 and len(trace["statistics"][0]["targets"]) == 6
    (result.loss * 65536).backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))
    assert len(trace["gradient"]) == (0 if sparse else 2)


def test_t4_fused_orchestration_saturated_rows_use_native_fallback(monkeypatch):
    trace = _install_cpu_fused_substitute(monkeypatch)
    hidden = torch.zeros((1, 2, 32))
    hidden[..., 0] = 1
    hidden.requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.zeros((151936, 32))
    weight[:, 0] = 76
    weight[0, 0] = 100
    labels = torch.tensor([[-100, 0]])
    actual = streamed_linear_ce(hidden, weight, labels, chunk_size=2, projection="t4_fused", numerics="guarded")
    expected = _reference(reference_hidden, weight, labels)
    torch.testing.assert_close(actual.loss, expected[0], rtol=0, atol=0)
    (actual.loss * 65536).backward()
    (expected[0] * 65536).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    assert len(trace["statistics"]) == 1 and not trace["gradient"]


def test_t4_fused_requires_supported_width_and_real_cuda_boundary():
    with pytest.raises(ValueError, match="width divisible by 32"):
        streamed_linear_ce(torch.zeros(1, 2, 33), torch.zeros(17, 33),
                           torch.zeros(1, 2, dtype=torch.long), projection="t4_fused", numerics="guarded")
    with pytest.raises(ValueError, match="CUDA"):
        streamed_linear_ce(torch.zeros(1, 2, 32), torch.zeros(17, 32),
                           torch.zeros(1, 2, dtype=torch.long), projection="t4_fused", numerics="guarded")


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
                    reason="fused inline PTX requires physical CUDA execution")
@pytest.mark.parametrize("batch,length,width,vocab", [(2, 7, 32, 137), (2, 37, 1536, 151936)])
@pytest.mark.parametrize("sparse", [False, True])
def test_t4_fused_training_cuda_only(batch, length, width, vocab, sparse):
    if torch.cuda.get_device_capability() < (7, 5):
        pytest.skip("fused MMA projection requires sm75 or later")
    torch.manual_seed(1315)
    hidden = (torch.randn(batch, length, width, device="cuda") * 0.2).requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.randn(vocab, width, device="cuda", dtype=torch.float16) * 0.1
    labels = torch.randint(0, vocab, (batch, length), device="cuda")
    labels[:, :max(1, length // 3)] = -100
    labels[-1, -2:] = -100
    result = streamed_linear_ce(
        hidden, weight, labels, chunk_size=16, projection="t4_fused", sparse_backward=sparse, numerics="guarded",
    )
    expected = _reference(reference_hidden, weight, labels)
    torch.testing.assert_close(result.loss, expected[0], rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(result.token_loss, expected[1], rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(result.entropy, expected[2], rtol=5e-4, atol=5e-4)
    reference_scores = F.linear(reference_hidden.detach().half(), weight).float()
    chosen = reference_scores.gather(-1, result.predictions.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(chosen, reference_scores.amax(-1), rtol=0.002, atol=0.002)
    (result.loss * 65536).backward()
    (expected[0] * 65536).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.02, atol=0.08)


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
                    reason="fused exceptional-row detection needs physical CUDA execution")
@pytest.mark.parametrize("kind", ["hidden_nan", "weight_inf"])
def test_t4_fused_nonfinite_detection_and_native_fallback_cuda_only(kind):
    if torch.cuda.get_device_capability() < (7, 5):
        pytest.skip("fused MMA projection requires sm75 or later")
    torch.manual_seed(1483)
    hidden = torch.randn((1, 3, 32), device="cuda") * 0.2
    weight = torch.randn((137, 32), device="cuda") * 0.1
    if kind == "hidden_nan":
        hidden[0, 1, 0] = float("nan")  # Its shifted label is ignored.
    else:
        weight[0, 0] = float("inf")
    hidden.requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    labels = torch.tensor([[-100, 7, -100]], device="cuda")
    actual = streamed_linear_ce(hidden, weight, labels, chunk_size=2, projection="t4_fused", numerics="guarded")
    expected = _reference(reference_hidden, weight, labels)
    torch.testing.assert_close(actual.loss, expected[0], rtol=5e-4, atol=5e-4, equal_nan=True)
    torch.testing.assert_close(actual.entropy, expected[2], rtol=5e-4, atol=5e-4, equal_nan=True)
    (actual.loss * 65536).backward()
    (expected[0] * 65536).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.02, atol=0.08, equal_nan=True)
    assert torch.isnan(hidden.grad).any()
