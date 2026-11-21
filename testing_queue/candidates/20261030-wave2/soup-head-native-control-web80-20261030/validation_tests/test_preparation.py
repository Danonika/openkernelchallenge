"""Stdlib/source and synthetic-result fixtures only. No candidate import/GPU."""
import ast
import base64
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "web_adapter"
SHARED = Path("/workspace/shared/soup_web_ready_20261030")
sys.path.insert(0, str(ADAPTER))
import web_runner as w
import prepare_asset_manifest

checks = []


def passed(name, **extra):
    checks.append({"test": name, "pass": True, **extra})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((ROOT / "source_manifest.json").read_text())
with zipfile.ZipFile(ROOT / "candidate.zip") as z:
    assert len(z.namelist()) == len(set(z.namelist()))
    assert set(z.namelist()) == set(manifest)
    for name, expected in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == expected
        compile(z.read(name), name, "exec")
    with zipfile.ZipFile(ROOT / "payloads/head.zip") as frozen:
        for name in frozen.namelist():
            assert z.read(name) == frozen.read(name)
        assert set(z.namelist()) - set(frozen.namelist()) == {"solution/web_preflight.py"}
passed("exact_zip_manifest_and_frozen_algorithm_bytes")

nb_path = ROOT / "03_head_native_control.ipynb"
nb = json.loads(nb_path.read_text())
assert nb_path.stat().st_size < 8 * 1024**2
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    assert cell["outputs"] == [] and cell["execution_count"] is None
    source = "".join(cell["source"])
    compile(source, f"cell{i}", "exec")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            key = getattr(node.targets[0], "id", "")
            if key == "PAYLOAD":
                payload = ast.literal_eval(node.value)
            if key == "PAYLOAD_SHA256":
                payload_hashes = ast.literal_eval(node.value)
assert set(payload) == set(payload_hashes)
for name, data in payload.items():
    raw = base64.b64decode(data, validate=True)
    assert raw == (ADAPTER / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == payload_hashes[name]
assert (ADAPTER / "head.zip").read_bytes() == (ROOT / "candidate.zip").read_bytes()
passed("cleared_valid_notebook_exact_embedded_payload", bytes=nb_path.stat().st_size)

# Specialized runner changes only profile lookup, mode selection and route gate.
base_ast = ast.parse((SHARED / "support/web_runner.py").read_text())
local_ast = ast.parse((ADAPTER / "web_runner.py").read_text())
base_funcs = {n.name: n for n in base_ast.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
local_funcs = {n.name: n for n in local_ast.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
for name in base_funcs:
    if name not in {"normalize", "main"}:
        assert ast.dump(base_funcs[name]) == ast.dump(local_funcs[name]), name
base_main = ast.unparse(base_funcs["main"])
local_main = ast.unparse(local_funcs["main"])
assert local_main == base_main.replace("choices=['baseline', 'fusion_a', 'fusion_b']", "choices=['baseline', *FUSIONS]")
for name in ("runtime_probe.py", "process_guard.py", "owned_process_tree.py", "prepare_asset_manifest.py", "official_manifest.json"):
    assert (ADAPTER / name).read_bytes() == (SHARED / "support" / name).read_bytes()
assert w.NATIVE_TIMEOUT == 80
assert "'--timeout', '80'" in local_main
assert "SOUP_REPO" in local_main and "SOUP_RESULTS" in local_main
assert "PAIRED_COMPLETE_RULES_FAILED" in local_main and "complete_measurements=None" in local_main
for name in ("web_runner.py", "process_guard.py", "owned_process_tree.py", "prepare_asset_manifest.py"):
    for node in ast.walk(ast.parse((ADAPTER / name).read_text())):
        if isinstance(node, ast.Import):
            assert all(not x.name.startswith(("torch", "triton", "transformers")) for x in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("torch", "triton", "transformers"))
passed("accepted_orchestration_unchanged_cap80_no_parent_cuda_import")

official = json.loads((ADAPTER / "official_manifest.json").read_text())


def prepare_repo(repo):
    repo.mkdir()
    for name in official:
        dst = repo / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((Path("/workspace/shared/soup_cloud_env/source") / name).read_bytes())


with tempfile.TemporaryDirectory() as td:
    repo = Path(td) / "repo"
    prepare_repo(repo)
    assert not (repo / ".git").exists()
    w.verify_official(repo, official)
    (repo / "solution").mkdir()
    (repo / "solution/old.py").write_text("preserved")
    w.install_bundle(repo, ADAPTER / "head.zip", manifest, Path(td) / "backup")
    assert (Path(td) / "backup/old.py").read_text() == "preserved"
    w.verify_official(repo, official)
    for name in manifest:
        assert digest(repo / name) == manifest[name]
passed("gitless_official_hashes_solution_only_overlay_backup")


def fixture():
    # Values are entirely synthetic; never publish as benchmark observations.
    data = {"task": {"seed": 1, "repeats": 2, "only": None}, "packages_changed": [],
            "runs": {}, "evals": {}, "summary": {"synthetic_fixture_only": True}}
    logs = {}
    for side in ("base", "solution"):
        data["runs"][side] = []
        data["evals"][side] = {"exit_code": 0, "mean_loss": 1.0}
        for number in (1, 2):
            name = f"{side}_{number}"
            trainer = {"fp16": True, "bf16": False, "per_device_train_batch_size": 4,
                       "gradient_accumulation_steps": 1, "seed": 1, "max_length": 512,
                       "torch_compile": False, "gradient_checkpointing": False}
            metrics = {"exit_code": 0, "steps": 64, "device_count": 1, "cuda_device": "Tesla T4",
                       "trainer": trainer, "timed_steps": 62, "timed_loop_s": 30.0,
                       "own_kernel_ms": {}}
            names = ("loss", "num_tokens", "entropy", "mean_token_accuracy", "grad_norm")
            row = {"run": name, "side": side, "exit_code": 0, "global_step": 64,
                   "wall_s": 70.0, "gpu_peak_mib": 1000.0, "metrics": metrics,
                   "adapter_config": {"r": 16, "lora_alpha": 32,
                                      "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"},
                   "adapter_tensors": {"fixture.lora_A": ["F32", [16, 1536]]},
                   "logged_metrics": list(names), "log": {key: [1.0] * 64 for key in names},
                   "rewind": [[[0, 1, 2, 3], [1, 1, 1, 1], [1.0] * 4] for _ in range(64)]}
            data["runs"][side].append(row)
            logs[name] = "\n".join("SOUP_HEAD_NATIVE " + json.dumps({"event": event, "count": 64})
                                   for event in ("native_sum_forward_return", "native_sum_backward_return"))
    return data, logs


process = {"exit_code": 0, "timed_out": False, "interrupted": False, "cleanup_confirmed": True}
data, logs = fixture()
good = w.normalize(data, "head", process, logs)
assert good["complete_fixed_task"] and good["score"] is None and good["admission"] == "UNKNOWN"
assert w.FUSIONS["head"]["requires_original_kernels"] is False and w.FUSIONS["head"]["kernels"] == []
passed("synthetic_complete_command_without_fabricated_owned_timing")

negative_cases = ("route_absent", "route_install_only", "route_count1", "route_count63",
                  "route_wrong_json", "missing_forward", "missing_backward",
                  "missing_adapter", "missing_metric", "missing_rewind", "partial_steps",
                  "timeout", "interrupt", "uncertain_cleanup", "missing_eval", "wrong_only",
                  "extra_side", "missing_packages")
for case in negative_cases:
    value, text, proc = copy.deepcopy(data), dict(logs), dict(process)
    row = value["runs"]["solution"][0]
    if case == "route_absent": text["solution_1"] = ""
    if case == "route_install_only": text["solution_1"] = 'SOUP_HEAD_NATIVE {"event":"installed","count":64}'
    if case == "route_count1": text["solution_1"] = text["solution_1"].replace('"count": 64', '"count": 1')
    if case == "route_count63": text["solution_1"] = text["solution_1"].replace('"count": 64', '"count": 63')
    if case == "route_wrong_json": text["solution_1"] = "SOUP_HEAD_NATIVE native_sum_backward_return count64"
    if case == "missing_forward": text["solution_1"] = text["solution_1"].splitlines()[1]
    if case == "missing_backward": text["solution_1"] = text["solution_1"].splitlines()[0]
    if case == "missing_adapter": row.pop("adapter_tensors")
    if case == "missing_metric": row["log"].pop("num_tokens")
    if case == "missing_rewind": row.pop("rewind")
    if case == "partial_steps": row["metrics"]["steps"] = 63
    if case == "timeout": proc["timed_out"] = True
    if case == "interrupt": proc["interrupted"] = True
    if case == "uncertain_cleanup": proc["cleanup_confirmed"] = False
    if case == "missing_eval": value["evals"]["base"] = {}
    if case == "wrong_only": value["task"]["only"] = "solution"
    if case == "extra_side": value["runs"]["unexpected"] = []
    if case == "missing_packages": value.pop("packages_changed")
    rejected = w.normalize(value, "head", proc, text)
    assert not rejected["complete_fixed_task"] and rejected["complete_measurements"] is None
    assert rejected["score"] is None and rejected["admission"] == "UNKNOWN"
passed("synthetic_incomplete_output_and_route_negatives", cases=list(negative_cases))

# Exercise the real main orchestration with subprocess/GPU/ownership functions
# stubbed out. This checks failed official rules cannot retain a full comparison.
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    repo, results, working = td / "repo", td / "results", td / "working"
    prepare_repo(repo)
    working.mkdir()
    invoked, quarantines = [], []

    def fake_run(command, **kwargs):
        invoked.append(command)
        kwargs["log"].write_text("synthetic stub only\n")
        if any(str(x).endswith("runtime_probe.py") for x in command):
            Path(command[-1]).write_text(json.dumps({"passed": True, "resolved_public_hub_cache": str(td)}))
        elif "solution.web_preflight" in command:
            Path(command[command.index("--out") + 1]).write_text(json.dumps({"status": "PASS_NATIVE_HEAD_PREFLIGHT_ONLY"}))
        elif any(str(x).endswith("run_task.py") for x in command):
            assert command[command.index("--timeout") + 1] == "80"
            out = Path(command[command.index("--out") + 1]); out.mkdir()
            (out / "result.json").write_text(json.dumps(data))
            for name, content in logs.items():
                (out / name).mkdir()
                (out / name / "train.log").write_text(content)
        elif any(str(x).endswith("check_rules.py") for x in command):
            return {**process, "exit_code": 1}
        return dict(process)

    real_path = Path
    def routed_path(value):
        return working if str(value) == "/kaggle/working" else real_path(value)

    with patch.object(w, "Path", routed_path), patch.object(w, "LOCK", td / "lock"), \
         patch.object(w, "MARKER", td / "marker"), patch.object(w.process_guard, "run", fake_run), \
         patch.object(w.tree, "prepare"), patch.object(w.tree, "begin_quarantine"), \
         patch.object(w.tree, "quarantine", quarantines.append), patch.object(w.tree, "finish_quarantine"), \
         patch.object(prepare_asset_manifest, "build", return_value=({}, {})), \
         patch.dict(os.environ, {"SOUP_REPO": str(repo), "SOUP_RESULTS": str(results)}), \
         patch.object(sys, "argv", ["web_runner.py", "--mode", "head"]):
        try:
            w.main()
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("Rules failure did not fail the job")
    status = json.loads(next(results.glob("*/status.json")).read_text())
    assert status["status"] == "PAIRED_COMPLETE_RULES_FAILED"
    assert status["complete_command_evidence"] is True
    assert status["official_checks_passed"] is False
    assert status["complete_fixed_task"] is False and status["complete_measurements"] is None
    assert status["score"] is None and status["admission"] == "UNKNOWN"
    assert quarantines and next(results.glob("*/raw/official_output/result.json")).is_file()
passed("synthetic_official_rules_failure_captures_raw_evidence_without_comparison")

assert "torch" not in sys.modules and "triton" not in sys.modules
report = {"status": "SOURCE_STATIC_SYNTHETIC_CHECKS_PASS", "checks": checks,
          "notebook_sha256": digest(nb_path), "source_zip_sha256": digest(ROOT / "candidate.zip"),
          "gpu_executed": False, "candidate_imported": False, "notebook_executed": False,
          "preflight_executed": False, "authenticated_requests": False, "submitted": False,
          "t4_status": "UNTESTED", "standalone_admission": "BLOCKED_ZERO_OWNED_TRITON"}
(ROOT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
