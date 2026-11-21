"""CPU/stdlib-only packet checks, with no GPU imports or execution."""
import ast
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_candidate as runner

checks = {}
for path in [*ROOT.joinpath("solution").glob("*.py"), *ROOT.glob("*.py"),
             *ROOT.joinpath("validation").glob("*.py")]:
    compile(path.read_text(), str(path), "exec")
checks["all_python_syntax"] = True

notebook = json.loads((ROOT / "candidate.ipynb").read_text())
assert notebook["nbformat"] == 4
namespace = {}
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        assert cell["outputs"] == [] and cell["execution_count"] is None
        compile("".join(cell["source"]), "notebook_cell_" + str(index), "exec")
checks["notebook_json_cleared_and_code_syntax"] = True

# The payload cell consists solely of literals. Never execute notebook code.
payload = ast.parse("".join(notebook["cells"][3]["source"]))
for node in payload.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id == "EMBEDDED_FILES":
            embedded = ast.literal_eval(node.value)
            break
else:
    raise AssertionError("embedded payload not found")
for name, encoded in embedded.items():
    assert base64.b64decode(encoded) == (ROOT / name).read_bytes(), name
checks["notebook_embedded_bytes_match_packet"] = True

manifest = json.loads((ROOT / "candidate-files.json").read_text())
info = json.loads((ROOT / "bundle-info.json").read_text())
assert hashlib.sha256((ROOT / "candidate.zip").read_bytes()).hexdigest() == info["source_sha256"]
assert info["files"] == manifest
with zipfile.ZipFile(ROOT / "candidate.zip") as archive:
    assert set(archive.namelist()) == set(manifest)
    for name, digest in manifest.items():
        raw = archive.read(name)
        assert hashlib.sha256(raw).hexdigest() == digest
        assert raw == (ROOT / name).read_bytes()
checks["zip_source_hash_manifest_match"] = True

sys.path.insert(0, "/redacted/local/soup_cycle/test_queue")
from bundle_candidate import bundle
raw1, manifest1 = bundle(ROOT / "solution")
raw2, manifest2 = bundle(ROOT / "solution")
assert raw1 == raw2 == (ROOT / "candidate.zip").read_bytes()
assert manifest1 == manifest2 == manifest
checks["official_packager_deterministic"] = True

source = Path("/redacted/local/soup_cloud_env/source")
official = json.loads((ROOT / "official-source-manifest.json").read_text())
runner.source_preflight(source, official)
assert not any(name.startswith("solution/") for name in official)
checks["unchanged_official_commit_and_file_hashes"] = True

spec = json.loads((ROOT / "submission.template.json").read_text())
example = json.loads(Path("/redacted/local/soup_cycle/test_queue/examples/http_submission.json").read_text())
assert set(spec) == set(example)
assert spec["baseline_id"] == "REPLACE_WITH_MEASURED_GCP_BASELINE_ID"
assert spec["artifact_kind"] == "solution_bundle" and spec["entrypoint"] == "soup_screen_v1"
checks["queue_template_shape_and_unresolved_baseline"] = True

one = {"exit_code": 0, "global_step": 64, "adapter_config": {"r": 16},
       "adapter_tensors": {"weight": ["F32", [16, 8]]}, "rewind": [[1]],
       "logged_metrics": ["loss"], "metrics": {"exit_code": 0, "steps": 64}}
good = {"packages_changed": [], "runs": {side: [copy.deepcopy(one), copy.deepcopy(one)]
                                         for side in ("base", "solution")},
        "evals": {side: {"exit_code": 0, "mean_loss": 2.05} for side in ("base", "solution")}}
assert runner.complete_result(good) == []
for broken in ("exit", "partial", "missing_adapter", "missing_eval", "nonfinite_eval", "one_run", "packages"):
    bad = copy.deepcopy(good)
    if broken == "exit":
        bad["runs"]["solution"][0]["exit_code"] = -9
    elif broken == "partial":
        bad["runs"]["solution"][0]["metrics"]["steps"] = 63
    elif broken == "missing_adapter":
        bad["runs"]["solution"][0]["adapter_tensors"] = None
    elif broken == "missing_eval":
        del bad["evals"]["solution"]
    elif broken == "nonfinite_eval":
        bad["evals"]["solution"]["mean_loss"] = float("nan")
    elif broken == "one_run":
        bad["runs"]["solution"].pop()
    elif broken == "packages":
        bad["packages_changed"] = ["changed"]
    assert runner.complete_result(bad), broken
checks["completion_positive_and_seven_negative_fixtures"] = True

with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)
    runner.unpack_reviewed(ROOT / "candidate.zip", manifest, temp / "repo")
    assert all(runner.sha(temp / "repo" / name) == digest for name, digest in manifest.items())
    changed = dict(manifest)
    changed["solution/__init__.py"] = "0" * 64
    try:
        runner.unpack_reviewed(ROOT / "candidate.zip", changed, temp / "reject")
    except RuntimeError:
        pass
    else:
        raise AssertionError("corrupt bundle accepted")
    control = subprocess.run([sys.executable, str(ROOT / "run_candidate.py"),
                              "--repo", "missing", "--cache", "missing", "--out", str(temp / "new"),
                              "--backend", "GCP"], capture_output=True, text=True)
    assert control.returncode != 0 and "Explicit --execute" in control.stderr
    assert not (temp / "new").exists()
checks["bundle_rejection_and_no_execute_safe_default"] = True

with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)
    repo, cache = temp / "repo", temp / "cache"
    repo.mkdir()
    model = cache / runner.MODEL
    snapshot = model / "snapshots" / runner.REVISION
    for path in (model / "refs", model / "blobs", snapshot):
        path.mkdir(parents=True, exist_ok=True)
    (model / "refs" / "main").write_text(runner.REVISION)
    def asset(name, raw):
        blob = model / "blobs" / hashlib.sha256(raw).hexdigest()
        blob.write_bytes(raw)
        (snapshot / name).symlink_to(blob)
    asset("config.json", b"{}")
    asset("tokenizer.json", b"{}")
    asset("model.safetensors", b"fixture-only-not-real-weights")
    saved_data = runner.DATA
    runner.DATA = {}
    try:
        assert runner.asset_preflight(repo, cache)["revision"] == runner.REVISION
        (snapshot / "model.safetensors").unlink()
        asset("shard-1.safetensors", b"fixture-shard")
        asset("model.safetensors.index.json", json.dumps({"weight_map": {"w": "shard-1.safetensors"}}).encode())
        assert runner.asset_preflight(repo, cache)["revision"] == runner.REVISION
        (snapshot / "shard-1.safetensors").unlink()
        try:
            runner.asset_preflight(repo, cache)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Missing indexed shard accepted")
    finally:
        runner.DATA = saved_data
checks["single_file_and_indexed_cache_fixtures"] = True

assert not any("triton.jit" in p.read_text() for p in (ROOT / "solution").glob("*.py"))
checks["zero_triton_source_honestly_declared"] = True

result = {"scope": "Static/stdlib CPU checks only; no model, GPU or remote execution",
          "checks": checks, "passed": len(checks), "failed": 0,
          "source_sha256": info["source_sha256"],
          "notebook_sha256": runner.sha(ROOT / "candidate.ipynb")}
(ROOT / "validation" / "static_results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
