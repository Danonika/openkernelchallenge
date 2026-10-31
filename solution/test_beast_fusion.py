"""Target-device contract checks for the real integrated Soup solution.

Run with the challenge's pinned torch/triton/bitsandbytes environment:
  python -m pytest solution/test_beast_fusion.py -q
No measurements/hidden rows are read. All GPU tests are clearly skipped on
CPU-only hosts rather than falsely reported as passing.
"""
import pytest
import torch
import torch.nn.functional as F


def _has_cuda():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda(), reason="Requires CUDA T4 or compatible device")
@pytest.mark.parametrize("seed", [7, 29])
def test_hero3_vocabulary_stream_loss_entropy_and_hidden_gradient(seed):
    from solution.hero3_vocab import _VocabStream

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch, length, hidden, vocab = 2, 7, 128, 8192
    x = torch.randn(batch, length, hidden, device="cuda", dtype=torch.float16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    w = torch.randn(vocab, hidden, device="cuda", dtype=torch.float16).detach()
    labels = torch.randint(vocab, (batch, length), device="cuda", dtype=torch.long)
    labels[:, :3] = -100
    shifted = labels.roll(-1, dims=-1)
    shifted[:, -1] = -100

    logits = torch.matmul(x.reshape(-1, hidden), w.T).reshape(batch, length, vocab)
    reference_logp = logits.float().log_softmax(-1)
    native_loss = F.cross_entropy(
        logits.float().reshape(-1, vocab),
        shifted.reshape(-1),
        reduction="sum", ignore_index=-100
    ) / (shifted != -100).sum()
    native_entropy = -(reference_logp.exp() * reference_logp).sum(-1)
    native_ids = logits.argmax(dim=-1)

    result = _VocabStream.apply(y, w, labels, None, 8192)
    actual_loss, tok_loss, entropy, predictions = result

    torch.testing.assert_close(actual_loss, native_loss, atol=3e-3, rtol=2e-3)
    torch.testing.assert_close(entropy, native_entropy, atol=3e-3, rtol=2e-3)
    assert torch.equal(predictions, native_ids)
    assert torch.equal(tok_loss[shifted == -100], torch.zeros_like(tok_loss[shifted == -100]))

    scale = torch.tensor(65536.0, device="cuda")
    native_grad = torch.autograd.grad(native_loss, x, scale)[0]
    our_grad = torch.autograd.grad(actual_loss, y, scale)[0]
    torch.testing.assert_close(our_grad, native_grad, atol=0.05, rtol=0.03)


@pytest.mark.skipif(not _has_cuda(), reason="Requires CUDA, bitsandbytes and quantized weights")
def test_grouped_qkv_frozen_base_gradient():
    import bitsandbytes as bnb
    from solution.hero3_qkv import _FrozenQKV

    torch.manual_seed(20261031)
    dims = [64, 16, 16]
    modules = [
        bnb.nn.Linear4bit(
            64, out, bias=True, compute_dtype=torch.float16,
            quant_type="nf4", compress_statistics=True
        ).half().cuda()
        for out in dims
    ]
    inp = torch.randn(2, 13, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    packed_inp = inp.detach().clone().requires_grad_(True)
    native = torch.cat([module(inp) for module in modules], dim=-1)
    fused = _FrozenQKV.apply(
        packed_inp,
        *(module.weight for module in modules),
        *(module.bias for module in modules),
    )
    torch.testing.assert_close(fused, native, rtol=0.015, atol=0.015)
    random_grad = torch.randn_like(native)
    native_dx = torch.autograd.grad(native, inp, random_grad)[0]
    fused_dx = torch.autograd.grad(fused, packed_inp, random_grad)[0]
    torch.testing.assert_close(fused_dx, native_dx, atol=0.04, rtol=0.04)


@pytest.mark.skipif(not _has_cuda(), reason="Requires CUDA native Philox dropout")
def test_padded_geometry_dropout_mask_and_rng():
    from solution.hero3_pack import NativeShapedDropout

    torch.manual_seed(20261031)
    torch.cuda.manual_seed_all(20261031)
    full = torch.randn(4, 12, 64, device="cuda", dtype=torch.float32)
    active = torch.tensor(
        [i for i in range(48) if i % 12 < (3 + i // 12)],
        device="cuda", dtype=torch.long
    )
    state = torch.cuda.get_rng_state().clone()
    full_native = F.dropout(full, p=0.05, training=True)
    state_native = torch.cuda.get_rng_state().clone()
    torch.cuda.set_rng_state(state)

    seed = torch.ones_like(full)
    _, mask = torch.ops.aten.native_dropout(seed, 0.05, True)
    selected = mask.reshape(-1, 64).index_select(0, active)
    compact = full.reshape(-1, 64).index_select(0, active)
    our = NativeShapedDropout.apply(compact, selected, 1.0 / 0.95)
    expected = full_native.reshape(-1, 64).index_select(0, active)
    torch.testing.assert_close(our, expected, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), state_native)


def test_hero3_module_inventory():
    from solution import settings
    assert settings.HEAD_MODE == "streamed"
    assert settings.FROZEN_STORAGE
    assert settings.FUSED_NF4
    assert settings.FUSED_RMSNORM
    assert settings.FUSED_ROTARY
    assert not settings.FUSED_LORA_DROPOUT
