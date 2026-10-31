"""Real CUDA parity test: saved gate/up activations survive host roundtrip."""
from contextlib import nullcontext
import pytest
import torch
from .activation_offload import _pack, _unpack, counters

def test_cpu_guard():
    from .activation_offload import _candidate
    assert not _candidate(torch.empty(512, 8960, dtype=torch.float16))
    assert not _candidate(torch.empty(512, 1536, dtype=torch.float16))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_fp16_roundtrip_forward_and_gradient_exact():
    gen = torch.Generator(device="cuda").manual_seed(832)
    gate = torch.randn((1, 1024, 8960), generator=gen, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    def run(optimised):
        g = gate.detach().clone().requires_grad_(True)
        u = up.detach().clone().requires_grad_(True)
        ctx = torch.autograd.graph.saved_tensors_hooks(_pack, _unpack) if optimised else nullcontext()
        from .activation_offload import _active, _release
        token = None
        session = None
        if optimised:
            session = {'device': g.device, 'pending': [], 'depth': 0, 'staged': 0}
            token = _active.set(session)
        try:
            with ctx:
                loss = (torch.nn.functional.silu(g) * u).square().mean()
        finally:
            if optimised:
                _release(session, session['depth'])
                _active.reset(token)
        loss.backward()
        torch.cuda.synchronize()
        return loss.detach().clone(), g.grad.detach().clone(), u.grad.detach().clone()
    ref = run(False)
    before = counters()
    got = run(True)
    assert counters()["packed"] > before["packed"]
    assert counters()["restored"] > before["restored"]
    for actual, expected in zip(got, ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
