"""Build an offline service notebook while preserving every template cell.

This builder only packages source. It does not authenticate, submit a job,
install packages, change the challenge, or claim a measured result.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile


REPO = Path(__file__).resolve().parents[1]
SERVICE_SOURCE_COMMIT = "3a4953f76288969bb91956f114eda5b7e541f104"
TEMPLATE_SHA256 = "bea1521d4c9d68b27591a483b19463b86ef49a0d7ea1fd8ec5e73b61001b81f8"


def source_hash(directory):
    """Same ordered bytes as the pinned official solution_sha256 helper."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def code(identifier, source):
    compile(source, identifier, "exec")
    return {"cell_type": "code", "id": identifier, "metadata": {},
            "execution_count": None, "outputs": [], "source": source}


def build(repo, template, output):
    template_bytes = template.read_bytes()
    if hashlib.sha256(template_bytes).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("Template differs from the reviewed service-source snapshot; review it before rebuilding")
    original = json.loads(template_bytes)
    if len(original["cells"]) != 3 or [c["id"] for c in original["cells"]] != ["cell0", "cell1", "cell2"]:
        raise ValueError("Unexpected service template structure")

    solution = repo / "solution"
    paths = sorted(p for p in solution.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    if any(p.suffix != ".py" or p.is_symlink() for p in paths):
        raise ValueError("The embedded solution must contain ordinary Python source only")
    helper = repo / "tools/run_t4_readiness.py"
    paths.append(helper)
    members = {str(p.relative_to(repo)): p.read_bytes() for p in paths}
    for name, raw in members.items():
        compile(raw, name, "exec")
    manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(members.items()):
            item = zipfile.ZipInfo(name, date_time=(2026, 10, 30, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, raw)
    payload = buffer.getvalue()
    payload_sha = hashlib.sha256(payload).hexdigest()
    solution_sha = source_hash(solution)
    protected = None
    for node in ast.parse(helper.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROTECTED_SHA256" for t in node.targets):
            protected = ast.literal_eval(node.value)
    if not protected:
        raise ValueError("Missing protected-file hashes")

    install = '''import base64, hashlib, io, stat, zipfile
from pathlib import PurePosixPath

# No imports of Torch/CUDA here. The training subprocess imports solution.
FUSION_SOLUTION_SHA256 = __SOLUTION_SHA__
FUSION_ARCHIVE_SHA256 = __ARCHIVE_SHA__
FUSION_FILES = __FILES__
FUSION_PROTECTED = __PROTECTED__
FUSION_PAYLOAD = (
__PAYLOAD__
)
repo = Path(os.environ["SOUP_REPO"]).resolve(strict=True)
results = Path(os.environ["SOUP_RESULTS"])
results.mkdir(parents=True, exist_ok=True)
for name, expected in FUSION_PROTECTED.items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, name
payload = base64.b64decode(FUSION_PAYLOAD, validate=True)
assert hashlib.sha256(payload).hexdigest() == FUSION_ARCHIVE_SHA256
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert set(archive.namelist()) == set(FUSION_FILES)
    assert len(archive.namelist()) == len(FUSION_FILES)
    for item in archive.infolist():
        name = PurePosixPath(item.filename)
        assert not name.is_absolute() and ".." not in name.parts
        assert name.parts[0] == "solution" or str(name) == "tools/run_t4_readiness.py"
        assert not stat.S_ISLNK(item.external_attr >> 16)
        raw = archive.read(item)
        assert hashlib.sha256(raw).hexdigest() == FUSION_FILES[str(name)], str(name)
        target = repo.joinpath(*name.parts)
        assert target.resolve().is_relative_to(repo) and not target.is_symlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)

digest = hashlib.sha256()
for path in sorted((repo / "solution").rglob("*")):
    if path.is_file() and "__pycache__" not in path.parts:
        digest.update(str(path.relative_to(repo / "solution")).encode())
        digest.update(path.read_bytes())
assert digest.hexdigest() == FUSION_SOLUTION_SHA256, "Unexpected extra or changed solution source"
for name, expected in FUSION_PROTECTED.items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, name
(results / "fusion_source_manifest.json").write_text(json.dumps({
    "solution_sha256": FUSION_SOLUTION_SHA256,
    "archive_sha256": FUSION_ARCHIVE_SHA256,
    "files": FUSION_FILES,
    "protected_sources_unchanged": True,
    "service_template_source_commit": __TEMPLATE_COMMIT__,
    "target_seconds": 33,
    "performance_status": "not_measured_by_install_cell"
}, indent=2) + "\\n")
print("Installed fused candidate:", FUSION_SOLUTION_SHA256)
print("All optimization source is on disk under", repo / "solution")
'''
    encoded = base64.b64encode(payload).decode()
    install = (install.replace("__SOLUTION_SHA__", repr(solution_sha))
               .replace("__ARCHIVE_SHA__", repr(payload_sha))
               .replace("__FILES__", repr(manifest))
               .replace("__PROTECTED__", repr(protected))
               .replace("__TEMPLATE_COMMIT__", repr(SERVICE_SOURCE_COMMIT))
               .replace("__PAYLOAD__", "\n".join("    " + repr(encoded[i:i+120])
                                                   for i in range(0, len(encoded), 120))))
    readiness = '''# Selected-source tests run in child processes, then exit before measurement.
# The helper uses a disposable test compiler cache, removed before this returns.
# Bounded reports are copied to SOUP_RESULTS even when a readiness gate fails.
# It does not run training or modify the template's measurement command.
readiness_work = Path("/kaggle/working/fusion-readiness")
readiness_reports = results / "fusion_readiness"
readiness_reports.mkdir(parents=True, exist_ok=True)
copied, omitted, report_bytes = [], [], 0
try:
    readiness = subprocess.run([
        sys.executable, str(repo / "tools/run_t4_readiness.py"),
        "--local-snapshot", "--output-dir", str(readiness_work)
    ], cwd=repo)
finally:
    for path in sorted(readiness_work.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative, size = str(path.relative_to(readiness_work)), path.stat().st_size
        if (path.suffix not in (".json", ".jsonl", ".log", ".yaml")
                or size >= 4 * 1024**2 or report_bytes + size > 8 * 1024**2
                or len(copied) >= 20):
            omitted.append({"file": relative, "bytes": size})
            continue
        target = readiness_reports / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(relative)
        report_bytes += size
    (readiness_reports / "export_manifest.json").write_text(json.dumps({
        "copied_reports": copied, "omitted_reports": omitted,
        "copied_bytes": report_bytes, "readiness_workspace": str(readiness_work),
        "export_limit_bytes": 8 * 1024**2, "export_limit_files": 20
    }, indent=2) + "\\n")
if readiness.returncode:
    raise RuntimeError("Physical T4 readiness failed; inspect fusion_readiness reports before training")
'''
    failed_export = '''# Preserve the original export cell and also retain reports if training raises.
# This synchronous, failure-only hook does no work while training is running.
from IPython import get_ipython

def _fusion_export_failed_cell(execution):
    if execution.error_in_exec is None and execution.error_before_exec is None:
        return
    failed_run = globals().get("run")
    if not isinstance(failed_run, Path) or not failed_run.is_dir():
        return
    copied, errors = [], []
    try:
        for path in failed_run.rglob("*"):
            if (path.is_file() and not path.is_symlink()
                    and path.suffix in (".json", ".jsonl", ".log", ".yaml")
                    and path.stat().st_size < 4 * 1024**2):
                target = results / path.relative_to(failed_run)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                copied.append(str(path.relative_to(failed_run)))
    except Exception as error:
        errors.append({"error": str(error)})
    try:
        (results / "fusion_failed_cell_export.json").write_text(json.dumps({
            "trigger": "notebook_cell_error", "copied_reports": copied,
            "copy_errors": errors, "training_success_claimed": False
        }, indent=2) + "\\n")
        print("Retained failed-cell reports:", len(copied), "copy errors:", len(errors))
    except Exception as error:
        print("Could not write failure-export summary:", type(error).__name__)

shell = get_ipython()
assert shell is not None, "Run this artifact as a notebook in the prepared service"
shell.events.register("post_run_cell", _fusion_export_failed_cell)
'''
    intro = f'''# Soup Fusion: service candidate

**Source candidate; GPU results pending.** Fixed profile: cached head with 128-token CE backward chunks, native GEMMs/dropout, original NF4/RMSNorm/LoRA gradient kernels, and compact MLP saved activations.

This notebook embeds actual Python files. `solution.apply()` takes no arguments and installs patches before model/trainer creation. It does not start training. The parent notebook keeps no CUDA context.

The three original service-template cells are preserved exactly. The added source, readiness and failure-report cells sit between setup and the template's training command. A synchronous failure-only hook also copies small reports if training raises before the original export cell. No packages are installed, no network is used inside the notebook, and the protected harness/task/tests are hash-checked.

The template's `--timeout 900` is unchanged. The service still enforces its separate **80-second cutoff per measured training command**. A notebook timeout does not extend that cutoff.

The default template runs the solution against a compatible saved public baseline. This is development feedback; it does not establish paired Kaggle admission, hidden-row quality, a score, or a sub-33-second result. Inspect job state, `result.error`, metrics and exported logs together.

Template source: `BAITC-Hacks/hack-57faaa42-duo@{SERVICE_SOURCE_COMMIT}`. This pinned source is not an attestation of the deployed runtime. Solution SHA-256: `{solution_sha}`.
'''
    notebook = copy.deepcopy(original)
    notebook["cells"] = [
        {"cell_type": "markdown", "id": "fusion-intro", "metadata": {}, "source": intro},
        copy.deepcopy(original["cells"][0]),
        code("fusion-install", install),
        code("fusion-readiness", readiness),
        code("fusion-failure-export", failed_export),
        copy.deepcopy(original["cells"][1]),
        copy.deepcopy(original["cells"][2]),
    ]
    for old, position in zip(original["cells"], (1, 5, 6)):
        assert old == notebook["cells"][position]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            compile(source if isinstance(source, str) else "".join(source), cell["id"], "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    return {"notebook": str(output), "notebook_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "template_sha256": TEMPLATE_SHA256, "service_source_commit": SERVICE_SOURCE_COMMIT,
            "solution_sha256": solution_sha, "archive_sha256": payload_sha,
            "embedded_files": len(members), "template_cells_preserved": True,
            "code_cells_compile": True, "physical_gpu_execution": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--template", type=Path, default=REPO / "templates/notebook_service_template.ipynb")
    parser.add_argument("--output", type=Path, default=REPO / "notebooks/Soup_Fusion_Service.ipynb")
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(), args.template.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
