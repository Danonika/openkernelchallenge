"""Future explicitly authorized T4 preflight. Not run by solution.apply().

No download, install, benchmark, tolerance search or harness modification.
Exact finite restored values/output/input gradients; nonfinite classes and
signed-zero identity. NaN payload bits are outside this declared gate.
"""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch

from . import adapter
from .fixture_common import make_mlp, compare, run_one, state_snapshot, check_state
from .restore import restore_half


def environment_gate():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight needs one visible, explicitly authorized CUDA GPU")
    if torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError("Preflight is fixed to T4/SM75")
    installed = adapter.install(mode="production")
    if not installed["installed"]:
        raise RuntimeError("Production source/environment gate: " + repr(installed))
    adapter.uninstall()
    return {"torch": torch.__version__, "torch_git": torch.version.git_version,
            "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)), "source_gate": installed}


def restore_gate():
    # Every binary16 encoding, including zero/subnormals/Inf/NaNs. Finite bits
    # must match native widening; NaN payload identity is deliberately excluded.
    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16).cuda()
    rows = []
    for length in (1, 7, 257, 65536, 65539):
        source = codes[:length].contiguous() if length <= codes.numel() else torch.cat((codes, codes[:3]))
        actual, expected = restore_half(source), source.float()
        rows.append({"length": length, **compare(actual, expected)})
        assert actual.stride() == expected.stride()
    return {"binary16_restore_cases": rows, "nan_payload_bits": "not asserted"}


def compare_pair(mlp, x, dy, amp, repeat=False):
    adapter.uninstall()
    baseline = run_one(mlp, x, dy, amp=amp, repeat=repeat)
    assert adapter.install(mode="production")["installed"]
    adapter.reset_stats()
    candidate = run_one(mlp, x, dy, amp=amp, repeat=repeat)
    result = {"shape": list(x.shape), "amp_fp16": amp,
              "output": compare(candidate[0], baseline[0]),
              "input_gradient": compare(candidate[1], baseline[1]),
              "stats": adapter.snapshot()}
    assert candidate[0].stride() == baseline[0].stride()
    assert candidate[1].stride() == baseline[1].stride()
    assert candidate[2] == baseline[2], "native graph topology changed"
    assert result["stats"].get("encoded_operands") == 2
    assert result["stats"].get("restores") == (4 if repeat else 2)
    adapter.uninstall()
    return result


def actual_module_gate(mlp):
    before = state_snapshot(mlp)
    rows = []
    for shape, amp, repeat in (((1, 3, 1536), False, True),
                               ((1, 3, 1536), True, False),
                               ((1, 356, 1536), True, False),
                               ((1, 513, 1536), True, False)):
        count = shape[0] * shape[1] * shape[2]
        x = torch.linspace(-.2, .2, count, device="cuda").reshape(shape)
        dy = torch.linspace(-.5, .5, count, device="cuda").reshape(shape)
        rows.append(compare_pair(mlp, x, dy, amp, repeat))
    check_state(mlp, before)
    return {"fixed_shape_cases": rows, "weight_and_quantstate_unchanged": True}


def exceptional_gate(mlp):
    rows = []
    shape = (1, 3, 1536)
    for kind in ("scaled_gradient", "input_inf", "input_nan", "cotangent_inf"):
        x = torch.full(shape, .125, device="cuda")
        dy = torch.ones_like(x)
        if kind == "scaled_gradient":
            dy.mul_(65536.)
        elif kind == "input_inf":
            x[0, 0, 0] = float("inf")
        elif kind == "input_nan":
            x[0, 0, 0] = float("nan")
        else:
            dy[0, 0, 0] = float("inf")
        rows.append({"kind": kind, **compare_pair(mlp, x, dy, True)})
    return rows


def scaler_optimizer_gate(mlp):
    rows = []
    initial = torch.linspace(-.1, .1, 3 * 1536, device="cuda").reshape(1, 3, 1536)
    for enabled in (False, True):
        adapter.uninstall()
        if enabled:
            assert adapter.install(mode="production")["installed"]
        adapter.reset_stats()
        x = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([x], lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
        with torch.autocast("cuda", dtype=torch.float16):
            # Real Qwen hidden states are ordinary Tensor values, not a
            # Parameter subclass. Preserve the optimizer leaf and its native
            # differentiable edge identically on baseline/candidate sides.
            hidden = x.view_as(x)
            y = mlp(hidden)
            loss = y.square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient = x.grad.detach().clone()
        scaler.step(optimizer)
        scaler.update()
        rows.append((loss.detach(), gradient, x.detach().clone(), scaler.get_scale(), adapter.snapshot()))
    for i in range(3):
        compare(rows[0][i], rows[1][i])
    assert rows[0][3] == rows[1][3]
    assert rows[1][4].get("encoded_operands") == 2, {"candidate_stats": rows[1][4], "baseline_stats": rows[0][4]}
    adapter.uninstall()
    return {"native_AdamW_and_GradScaler_leaf_input_control": "exact", "scale": rows[1][3],
            "not_claimed": "full q/v adapter optimizer trajectory"}


def saved_hook_fallback_gate(mlp):
    assert adapter.install(mode="production")["installed"]
    adapter.reset_stats()
    counts = {"pack": 0, "unpack": 0}
    def pack(t):
        counts["pack"] += 1
        return t.detach()
    def unpack(t):
        counts["unpack"] += 1
        return t
    x = torch.zeros((1, 3, 1536), device="cuda")
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        run_one(mlp, x, torch.ones_like(x), amp=True)
    stats = adapter.snapshot()
    assert stats.get("encoded_operands", 0) == 0
    assert stats.get("existing_saved_hooks_declined") == 2
    adapter.uninstall()
    return {"counts": counts, "stats": stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="preflight_memory_vs1.json")
    args = parser.parse_args()
    output = Path(args.out)
    result = {"candidate": "memory-vs1", "status": "RUNNING", "groups": [],
              "gates": "exact finite bits/strides/gradients and nonfinite classifications; no relaxed tolerance",
              "source_sha256": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
              "not_a_benchmark_or_admission": True}
    def record(name, function):
        details = function()
        torch.cuda.synchronize()
        result["groups"].append({"name": name, "status": "PASS", "details": details})
        output.write_text(json.dumps(result, indent=2) + "\n")
    active_name = "environment"
    try:
        record(active_name, environment_gate)
        from .version_sentinel import runtime_contract
        active_name = "cuda_version_witness"
        record(active_name, lambda: runtime_contract("cuda"))
        active_name = "restore_binary16"
        record(active_name, restore_gate)
        mlp = make_mlp(1536, 8960, "cuda")
        for name, function in (("actual_fixed_MLP", actual_module_gate),
                               ("exceptional_classes", exceptional_gate),
                               ("native_scaler_optimizer", scaler_optimizer_gate),
                               ("existing_saved_hook_fallback", saved_hook_fallback_gate)):
            active_name = name
            record(name, lambda function=function: function(mlp))
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["groups"].append({"name": active_name, "status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        adapter.uninstall()
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "groups": len(result["groups"])}), flush=True)
    raise SystemExit(result["status"] != "PASS")


if __name__ == "__main__":
    main()
