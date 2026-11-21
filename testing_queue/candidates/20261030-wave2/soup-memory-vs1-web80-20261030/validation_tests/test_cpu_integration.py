"""New original actual-QwenMLP/Linear4bit CPU-equivalent tests, not T4 proof."""
import ast
import functools
import hashlib
import json
import traceback
import weakref
import gc
from pathlib import Path

import torch
import bitsandbytes as bnb

from solution import adapter
from solution.fixture_common import make_mlp, compare, run_one, state_snapshot, check_state

torch.set_num_threads(2)
ROOT = Path(__file__).parent
GEN = torch.Generator(device="cpu").manual_seed(305002)


def paired(mlp, value, cotangent, *, repeat=False):
    adapter.uninstall()
    before = state_snapshot(mlp)
    baseline = run_one(mlp, value, cotangent, repeat=repeat)
    installed = adapter.install(mode="cpu_equivalent")
    assert installed["installed"], installed
    adapter.reset_stats()
    candidate = run_one(mlp, value, cotangent, repeat=repeat)
    records = {"output": compare(candidate[0], baseline[0]),
               "gradient": compare(candidate[1], baseline[1]),
               "native_graph_unchanged": candidate[2] == baseline[2],
               "stats": adapter.snapshot()}
    assert records["native_graph_unchanged"]
    assert records["stats"].get("encoded_operands") == 2, records
    assert records["stats"].get("restores") == (4 if repeat else 2), records
    check_state(mlp, before)
    adapter.uninstall()
    return records


def actual_module_parity():
    mlp = make_mlp()
    rows = []
    for shape in ((3, 64), (2, 3, 64)):
        x = torch.randn(shape, generator=GEN) * 0.2
        for scale in (1., 32768.):
            dy = torch.full(shape, scale)
            rows.append(paired(mlp, x, dy, repeat=scale == 1.))
    return {"actual_qwen_mlp_bnb_cases": rows, "note": "Real CPU quantized modules/native CPU backend; no CUDA equivalence inference"}


def nonfinite_contract():
    mlp = make_mlp()
    rows = []
    for special in (float("inf"), float("nan")):
        x = torch.zeros((2, 64)); x[0, 0] = special
        rows.append(paired(mlp, x, torch.ones_like(x)))
    x = torch.zeros((2, 64))
    dy = torch.zeros_like(x); dy[0, 0] = float("inf")
    rows.append(paired(mlp, x, dy))
    return {"cases": rows, "nan_payload_identity": "not claimed"}


def scoped_fallbacks():
    mlp = make_mlp()
    x = torch.ones((2, 64), requires_grad=True)
    observations = []
    def case(name, operation):
        adapter.reset_stats()
        operation()
        stats = adapter.snapshot()
        assert stats.get("encoded_operands", 0) == 0, (name, stats)
        observations.append({"case": name, "stats": stats})
    assert adapter.install(mode="cpu_equivalent")["installed"]
    case("half_input", lambda: mlp(x.half()).sum().backward())
    case("no_input_gradient", lambda: mlp(x.detach()))
    mlp.eval(); case("eval", lambda: mlp(x)); mlp.train()
    with torch.no_grad():
        case("no_grad", lambda: mlp(x))
    mlp.gate_proj.compute_dtype = torch.float32
    case("other_compute_dtype", lambda: mlp(x))
    mlp.gate_proj.compute_dtype = torch.float16
    hook = mlp.gate_proj.register_forward_hook(lambda module, args, out: None)
    case("foreign_gate_hook", lambda: mlp(x)); hook.remove()
    old_forward = mlp.gate_proj.forward
    mlp.gate_proj.forward = lambda value: old_forward(value)
    case("instance_projection_forward", lambda: mlp(x))
    del mlp.gate_proj.forward
    adapter.uninstall()
    assert adapter.install(mode="cpu_equivalent", first_order_only=False)["installed"]
    case("higher_order_preselected_fallback", lambda: mlp(x))
    adapter.uninstall()
    return observations


def existing_hooks_and_failures():
    mlp = make_mlp()
    x = torch.ones((2, 64)) * 0.1
    counts = []
    values = []
    for enabled in (False, True):
        adapter.uninstall()
        if enabled:
            assert adapter.install(mode="cpu_equivalent")["installed"]
        adapter.reset_stats()
        count = {"pack": 0, "unpack": 0}
        def pack(t):
            count["pack"] += 1
            return t.detach()
        def unpack(t):
            count["unpack"] += 1
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            values.append(run_one(mlp, x, torch.ones_like(x)))
        counts.append(count)
    assert counts[0] == counts[1]
    assert adapter.snapshot().get("existing_saved_hooks_declined") == 2
    assert adapter.snapshot().get("encoded_operands", 0) == 0
    compare(values[0][0], values[1][0]); compare(values[0][1], values[1][1])
    adapter.uninstall()
    # An escaping mutable alias introduced by a foreign module hook must
    # decline the codec, preserving native version checks for that case.
    escaped = []
    h = mlp.gate_proj.register_forward_hook(lambda m, args, out: escaped.append(out.detach()))
    assert adapter.install(mode="cpu_equivalent")["installed"]
    adapter.reset_stats()
    y = mlp(x.clone().requires_grad_())
    escaped[0].add_(1.)
    try:
        y.sum().backward()
    except RuntimeError as exc:
        assert "modified by an inplace operation" in str(exc)
    else:
        raise AssertionError("native escaped-alias version error lost")
    assert adapter.snapshot().get("encoded_operands", 0) == 0
    h.remove(); adapter.uninstall()
    return {"outer_hook_counts": counts, "escaped_alias": "native fallback/version check", "partial_operand_outcomes": "allowed by contract; not inferred atomic"}


def ownership_release_and_registration():
    mlp = make_mlp()
    assert adapter.install(mode="cpu_equivalent")["installed"]
    assert adapter.install(mode="cpu_equivalent")["reason"] == "already_installed"
    original = adapter._register_one
    refs = []
    def observe(raw, wide, half):
        refs.append((weakref.ref(wide), weakref.ref(half)))
        return original(raw, wide, half)
    adapter._register_one = observe
    try:
        x = torch.ones((2, 64), requires_grad=True)
        y = mlp(x)
    finally:
        adapter._register_one = original
    gc.collect()
    assert len(refs) == 2
    assert all(wide() is None for wide, half in refs), "widened producer objects retained"
    y.sum().backward()
    assert x.grad is not None
    hooks_before = [len(m._forward_hooks) + len(m._forward_pre_hooks) for m in (mlp.gate_proj, mlp.up_proj, mlp.act_fn, mlp.down_proj)]
    adapter.uninstall()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in (mlp.gate_proj, mlp.up_proj, mlp.act_fn, mlp.down_proj))
    return {"wide_python_objects_released_before_backward": True,
            "installed_hook_counts": hooks_before,
            "note": "Python object release only; allocation peak and hidden aliases not universally proved"}


def static_and_environment_gates():
    tree = ast.parse((ROOT / "solution/adapter.py").read_text())
    banned = {"item", "cpu", "numpy", "isfinite", "synchronize"}
    found = sorted({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in banned})
    assert not found, found
    assert not any(isinstance(n, ast.Attribute) and n.attr == "saved_tensors_hooks" for n in ast.walk(tree))
    declined = adapter.install(mode="production")
    assert not declined["installed"] and declined["reason"] == "target_torch_triton_version"
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    original = Qwen2MLP.forward
    try:
        Qwen2MLP.forward = lambda self, x: original(self, x)
        declined_source = adapter.install(mode="cpu_equivalent")
        assert not declined_source["installed"] and declined_source["reason"] == "source_identity"
    finally:
        Qwen2MLP.forward = original
    @functools.wraps(original)
    def decorated(self, x):
        return original(self, x)
    try:
        Qwen2MLP.forward = decorated
        declined_decorated = adapter.install(mode="cpu_equivalent")
        assert not declined_decorated["installed"] and declined_decorated["reason"] == "source_unavailable"
    finally:
        Qwen2MLP.forward = original
    return {"no_value_scan_or_broad_saved_hooks_in_adapter": True,
            "production_rejected_on_cpu_build": declined,
            "foreign_source_rejected": declined_source,
            "decorated_foreign_source_rejected": declined_decorated}


def main():
    records = []
    groups = [actual_module_parity, nonfinite_contract, scoped_fallbacks,
              existing_hooks_and_failures, ownership_release_and_registration,
              static_and_environment_gates]
    for group in groups:
        try:
            details = group()
            record = {"group": group.__name__, "status": "PASS", "details": details}
        except Exception as exc:
            record = {"group": group.__name__, "status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()}
        finally:
            adapter.uninstall()
        records.append(record)
        print(group.__name__, record["status"], flush=True)
    result = {"candidate": "memory-vs1", "scope": "actual tiny CPU modules, explicitly CPU-equivalent mode",
              "torch": torch.__version__, "torch_git": torch.version.git_version,
              "threads": torch.get_num_threads(), "bnb": bnb.__version__,
              "source_sha256": hashlib.sha256((ROOT / "solution/adapter.py").read_bytes()).hexdigest(),
              "results": records, "passed": sum(r["status"] == "PASS" for r in records),
              "failed": sum(r["status"] == "FAIL" for r in records),
              "not_validated": ["T4 restore compilation/numerics", "actual full model", "AMP/GradScaler on CUDA", "full upstream trajectory", "allocator peak/time/own share", "79 regressions or speed screen"]}
    (ROOT / "cpu_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "failed": result["failed"]}))
    raise SystemExit(bool(result["failed"]))


if __name__ == "__main__":
    main()
