"""GPU checks for half-retained MLP operands; run before measured harness.

This uses only public synthetic inputs. It imports and compares the unchanged
official starter activation, which remains excluded from own-kernel credit.
It does not inspect challenge rows or change the measured configuration.
"""

import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _difference(a, b):
    absolute = (a.float() - b.float()).abs()
    return {
        "max_abs": float(absolute.max().item()),
        "equal": bool(torch.equal(a, b)),
    }


def _activation_cases(device):
    from solution.mlp_storage import HalfSavedSwiGLU
    from solution.swiglu import SwiGLU

    gen = torch.Generator(device=device).manual_seed(7317)
    records = []
    for shape in ((5, 257), (67, 8960)):
        gate = (torch.randn(shape, device=device, generator=gen) * 3).half().float()
        up = (torch.randn(shape, device=device, generator=gen) * 2).half().float()
        incoming = torch.randn(shape, device=device, generator=gen)
        a, b = gate.detach().requires_grad_(), up.detach().requires_grad_()
        reference = SwiGLU.apply(a, b)
        reference_saved = sum(t.numel() * t.element_size() for t in reference.grad_fn.saved_tensors)
        dg_ref, du_ref = torch.autograd.grad(reference, (a, b), incoming)
        c, d = gate.detach().requires_grad_(), up.detach().requires_grad_()
        candidate = HalfSavedSwiGLU.apply(c, d)
        saved = candidate.grad_fn.saved_tensors
        assert all(t.dtype == torch.float16 for t in saved)
        candidate_saved = sum(t.numel() * t.element_size() for t in saved)
        dg, du = torch.autograd.grad(candidate, (c, d), incoming)
        assert reference.dtype == candidate.dtype == torch.float32
        assert dg.dtype == du.dtype == torch.float32
        for actual, expected in ((candidate, reference), (dg, dg_ref), (du, du_ref)):
            # Identical expression with identical widened values. A failure
            # is a real generated-code finding; do not relax it silently.
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        assert candidate_saved * 2 == reference_saved
        records.append({
            "shape": list(shape),
            "forward": _difference(candidate, reference),
            "dgate": _difference(dg, dg_ref),
            "dup": _difference(du, du_ref),
            "reference_saved_bytes": reference_saved,
            "candidate_saved_bytes": candidate_saved,
        })

    # A noncontiguous operand and incoming-gradient case exercises the wrappers.
    shape = (37, 259)
    gate = torch.randn((shape[1], shape[0]), device=device, generator=gen).half().float().t()
    up = torch.randn((shape[1], shape[0]), device=device, generator=gen).half().float().t()
    incoming = torch.randn((shape[1], shape[0]), device=device, generator=gen).t()
    assert not gate.is_contiguous() and not incoming.is_contiguous()
    a, b = gate.detach().requires_grad_(), up.detach().requires_grad_()
    ref = SwiGLU.apply(a, b)
    ref_grad = torch.autograd.grad(ref, (a, b), incoming)
    c, d = gate.detach().requires_grad_(), up.detach().requires_grad_()
    out = HalfSavedSwiGLU.apply(c, d)
    out_grad = torch.autograd.grad(out, (c, d), incoming)
    for actual, expected in ((out, ref), *zip(out_grad, ref_grad)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    records.append({"shape": list(shape), "noncontiguous": True, "equal": True})
    return records


def _nf4_mlp_case(device):
    from bitsandbytes.nn import Linear4bit
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from solution.mlp_storage import _eligible_mlp, install
    from solution.swiglu import SwiGLU

    config = Qwen2Config(hidden_size=1536, intermediate_size=8960, hidden_act="silu")
    module = Qwen2MLP(config)
    for name, in_features, out_features in (
        ("gate_proj", 1536, 8960),
        ("up_proj", 1536, 8960),
        ("down_proj", 8960, 1536),
    ):
        projection = Linear4bit(
            in_features, out_features, bias=False,
            compute_dtype=torch.float16, compress_statistics=True, quant_type="nf4",
        ).to(device)
        projection.requires_grad_(False)
        setattr(module, name, projection)

    assert install(), "MLP installation failed"
    installed_forward = Qwen2MLP.forward
    assert install() and Qwen2MLP.forward is installed_forward, "Installation is not idempotent"

    gen = torch.Generator(device=device).manual_seed(7318)
    results = []
    # The second case crosses bitsandbytes' M>1536 dispatch boundary.
    for shape in ((2, 7, 1536), (4, 512, 1536)):
        x = torch.randn(shape, device=device, generator=gen, dtype=torch.float32)
        assert _eligible_mlp(module, x, Linear4bit), "The expected NF4 projection guard did not match"
        reference_input = x.detach().requires_grad_()
        candidate_input = x.detach().requires_grad_()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            reference = module.down_proj(SwiGLU.apply(
                module.gate_proj(reference_input), module.up_proj(reference_input)
            ))
            candidate = module(candidate_input)
        nodes, seen = [candidate.grad_fn], set()
        found_candidate = False
        while nodes:
            node = nodes.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            found_candidate |= type(node).__name__ == "HalfSavedSwiGLUBackward"
            nodes.extend(next_node for next_node, _ in node.next_functions)
        assert found_candidate, "The installed module silently fell back"
        del nodes, seen, node
        incoming = torch.randn(reference.shape, device=device, generator=gen, dtype=torch.float32)
        dref, = torch.autograd.grad(reference, (reference_input,), incoming)
        dgot, = torch.autograd.grad(candidate, (candidate_input,), incoming)
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(dgot, dref, rtol=0, atol=0, equal_nan=True)
        results.append({
            "shape": list(x.shape),
            "forward": _difference(candidate, reference),
            "input_gradient": _difference(dgot, dref),
            "output_dtype": str(candidate.dtype),
            "input_gradient_dtype": str(dgot.dtype),
            "eligible": True,
            "patched_module_executed": found_candidate,
        })
        del reference_input, candidate_input, reference, candidate, incoming, dref, dgot

    # Explicit fallback checks use module metadata only, without changing task data.
    assert not _eligible_mlp(module, x.half(), Linear4bit)
    assert not _eligible_mlp(module, x[0, 0], Linear4bit)
    old = module.gate_proj.compute_dtype
    try:
        module.gate_proj.compute_dtype = torch.float32
        assert not _eligible_mlp(module, x, Linear4bit)
    finally:
        module.gate_proj.compute_dtype = old
    module.gate_proj.forward = lambda value: value
    try:
        assert not _eligible_mlp(module, x, Linear4bit)
    finally:
        del module.gate_proj.forward
    for target, method in (
        (module.gate_proj, "register_forward_hook"),
        (module.gate_proj, "register_forward_pre_hook"),
        (module.act_fn, "register_forward_hook"),
    ):
        handle = getattr(target, method)(lambda *args: None)
        try:
            assert not _eligible_mlp(module, x, Linear4bit)
        finally:
            handle.remove()
    assert _eligible_mlp(module, x, Linear4bit)
    return {"cases": results, "install_idempotent": True, "fallback_guards": "passed"}


def run():
    assert torch.cuda.is_available(), "This validation requires the T4 service"
    device = torch.device("cuda", torch.cuda.current_device())
    # Restore caller RNG state; all measurement runs still choose their own seeds.
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(7316)
        records = _activation_cases(device)
        nf4 = _nf4_mlp_case(device)
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    return {"activation_cases": records, "nf4_mlp": nf4, "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
