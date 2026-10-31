"""Small CPU-only queue smoke: no CUDA queries, kernels, model loading or downloads."""
import itertools
import torch


def reference(q, k, c, s):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    while getattr(apply_rotary_pos_emb, "_original_rotary_candidate", False):
        apply_rotary_pos_emb = apply_rotary_pos_emb.__wrapped__
    return apply_rotary_pos_emb(q, k, c, s)


def exact(a, b):
    torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
    mask = b == 0
    assert torch.equal(torch.signbit(a[mask]), torch.signbit(b[mask]))


def inputs(b=2, t=7, hq=12, hk=2, d=128, cb=1, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(47131)
    def rand(shape):
        return torch.randn(shape, generator=generator, device=device)
    q = rand((b, t, hq, d)).transpose(1, 2).requires_grad_()
    k = rand((b, t, hk, d)).transpose(1, 2).requires_grad_()
    return q, k, rand((cb,t,d)), rand((cb,t,d))


def run():
    from .rotary import _supported, native_adjoint
    before = torch.get_rng_state().clone()
    q,k,c,s = inputs()
    y = reference(q,k,c,s)
    gs = tuple(torch.ones_like(x) * 65536 for x in y)
    expected = torch.autograd.grad(y, (q,k), gs)
    for g,e in zip(gs,expected):
        exact(native_adjoint(g,c,s), e)
    assert _supported(q,k,c,s,cuda_only=False)
    assert not _supported(q,k,c,s)
    assert not _supported(q,q,c,s,cuda_only=False)
    assert not _supported(q.contiguous(),k,c,s,cuda_only=False)
    c.requires_grad_()
    assert not _supported(q,k,c,s,cuda_only=False)
    assert torch.equal(before, torch.get_rng_state())
    return dict(passed=True, scope="CPU_METADATA_AND_NATIVE_ADJOINT_ONLY", gpu_executed=False,
                triton_executed=False, score=None, admission="UNKNOWN", torch=torch.__version__)


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
