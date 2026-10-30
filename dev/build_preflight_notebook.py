"""Build a diagnostic-only notebook; never alter the measured service template."""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
from pathlib import Path

from build_service_notebook import make_installer


ROOT = Path(__file__).resolve().parents[1]
PROBES = ("rmsnorm_probe.py", "mlp_probe.py", "metrics_probe.py", "run_gpu_probes.py")


def _run_preflight(solution_manifest, encoded_dev_files):
    """Standalone notebook runtime: no Torch imports in the notebook parent."""
    import base64
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    import traceback

    repo = Path(os.environ["SOUP_REPO"])
    reports = Path(os.environ["SOUP_RESULTS"]) / "preflight"
    reports.mkdir(parents=True, exist_ok=True)

    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def retire(process, include_descendants):
        # The GPU runner creates a separate process group for each probe.
        # On an outer timeout, collect and stop those descendants as well.
        groups = {process.pid}
        discovery_error = None
        try:
            if include_descendants:
                listing = subprocess.run(
                    ["ps", "-eo", "pid=,ppid=,pgid="], capture_output=True,
                    text=True, timeout=5, check=True,
                )
                rows = [tuple(map(int, line.split())) for line in listing.stdout.splitlines()]
                descendants = {process.pid}
                while True:
                    found = {pid for pid, parent, _ in rows if parent in descendants}
                    if found <= descendants:
                        break
                    descendants.update(found)
                groups.update(group for pid, _, group in rows if pid in descendants)
        except Exception as error:
            # Still stop the known group if process discovery itself fails.
            discovery_error = f"{type(error).__name__}: {error}"
        groups.discard(os.getpgrp())
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        return groups, discovery_error

    def child(name, command, cwd, environment, timeout):
        started = time.monotonic()
        record = {"name": name, "command": command, "timeout_seconds": timeout}
        stdout = stderr = ""
        process = None
        groups = set()
        interrupted = None
        try:
            process = subprocess.Popen(
                command, cwd=cwd, env=environment, start_new_session=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                record["state"] = "passed" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                record["state"] = "timeout"
                groups, cleanup_error = retire(process, include_descendants=True)
                if cleanup_error:
                    record["cleanup_error"] = cleanup_error
                stdout, stderr = process.communicate(timeout=15)
            finally:
                # Normal completion should leave no descendants. Retire the
                # original group too, in case a child retained that group.
                if process.poll() is not None:
                    retired, _ = retire(process, include_descendants=False)
                    groups.update(retired)
            record["returncode"] = process.returncode
        except BaseException as error:
            record.update(state="error", error=f"{type(error).__name__}: {error}")
            if process is not None:
                try:
                    retired, cleanup_error = retire(process, include_descendants=True)
                    groups.update(retired)
                    if cleanup_error:
                        record["cleanup_error"] = cleanup_error
                    stdout, stderr = process.communicate(timeout=15)
                    record["returncode"] = process.returncode
                except BaseException as cleanup_error:
                    record["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            if not isinstance(error, Exception):
                record["interrupted"] = True
                interrupted = error
        record["elapsed_seconds"] = time.monotonic() - started
        record["process_reaped"] = process is None or process.poll() is not None
        record["retired_process_groups"] = sorted(groups)
        (reports / f"{name}.stdout.log").write_text(stdout, encoding="utf-8")
        (reports / f"{name}.stderr.log").write_text(stderr, encoding="utf-8")
        write_json(reports / f"{name}.json", record)
        if interrupted is not None:
            raise interrupted
        if not record["process_reaped"] or "cleanup_error" in record:
            raise RuntimeError(f"Could not confirm cleanup for {name}; inspect its saved status")
        return record

    summary = {"purpose": "development preflight only; no measured training or admission result",
               "notebook_parent_imports_torch": False, "state": "failed"}
    workspace = None
    try:
        # Needed only for exceptional cleanup of the runner's nested groups.
        # Check availability before starting any regression or GPU subprocess.
        subprocess.run(["ps", "-eo", "pid=,ppid=,pgid="], capture_output=True,
                       text=True, timeout=5, check=True)
        with tempfile.TemporaryDirectory(prefix="duo-preflight-", dir="/kaggle/working") as temporary:
            workspace = Path(temporary)
            for item in solution_manifest:
                data = (repo / item["path"]).read_bytes()
                if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise RuntimeError(f"Installed solution differs from preflight manifest: {item['path']}")
                destination = workspace / item["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            dev_files = json.loads(base64.b64decode(encoded_dev_files, validate=True))
            for item in dev_files:
                data = base64.b64decode(item["data_base64"], validate=True)
                if hashlib.sha256(data).hexdigest() != item["sha256"] or len(data) != item["size_bytes"]:
                    raise RuntimeError("Embedded development source hash mismatch")
                name = item["name"]
                if Path(name).name != name or not name.endswith(".py"):
                    raise RuntimeError("Unsafe development source name")
                destination = workspace / "dev" / name
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(data)

            regression_env = dict(os.environ)
            regression_env.update(
                SOUP_APPLY_SOLUTION="1",
                TRITON_CACHE_DIR=str(workspace / "regression-cache" / "triton"),
                TORCHINDUCTOR_CACHE_DIR=str(workspace / "regression-cache" / "inductor"),
                PYTHONUNBUFFERED="1",
            )
            summary["regression"] = child(
                "regression", [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                repo, regression_env, 600,
            )
            probe_env = dict(os.environ, PYTHONUNBUFFERED="1")
            # The existing runner assigns fresh caches to each probe child.
            summary["gpu_runner"] = child(
                "gpu_runner", [sys.executable, str(workspace / "dev" / "run_gpu_probes.py"),
                               "--out", str(reports / "gpu"), "--timeout", "300"],
                workspace, probe_env, 1050,
            )
            if all(summary[name]["state"] == "passed" and summary[name]["process_reaped"]
                   for name in ("regression", "gpu_runner")):
                summary["state"] = "passed"
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        (reports / "preflight_error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        summary["temporary_workspace_removed"] = workspace is None or not workspace.exists()
        if not summary["temporary_workspace_removed"]:
            summary["state"] = "failed"
        write_json(reports / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["state"] != "passed":
        raise RuntimeError("Development preflight failed; inspect SOUP_RESULTS/preflight reports")


def build(output, replace=False):
    installer, metadata = make_installer()
    if len(metadata["files"]) != 6:
        raise ValueError("This preflight expects the current six-file solution; review changes before rebuilding")
    files = []
    for name in PROBES:
        data = (ROOT / "dev" / name).read_bytes()
        compile(data, name, "exec")
        files.append({"name": name, "size_bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "data_base64": base64.b64encode(data).decode("ascii")})
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    runtime = inspect.getsource(_run_preflight)
    runtime += "\n_run_preflight(\n    " + repr(metadata["files"]) + ",\n    "
    runtime += repr(base64.b64encode(payload).decode("ascii")) + ",\n)\n"
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "soup_preflight": {"purpose": "development-only", "solution_manifest": metadata,
                                        "dev_files": [{k: x[k] for k in ("name", "size_bytes", "sha256")} for x in files],
                                        "dev_payload_sha256": hashlib.sha256(payload).hexdigest()}},
        "cells": [
            {"cell_type": "markdown", "id": "preflight-purpose", "metadata": {},
             "source": "# Soup development preflight\n\nSynthetic GPU comparisons and the official Soup regression suite. This notebook does **not** run measured training or establish admission. Its parent uses only the standard library; child compiler caches and temporary probe sources are removed. Reports are written under `SOUP_RESULTS/preflight` before a failed check raises. The measured four-cell notebook is a separate artifact.\n"},
            {"cell_type": "code", "id": "preflight-install", "metadata": {},
             "execution_count": None, "outputs": [], "source": installer},
            {"cell_type": "code", "id": "preflight-checks", "metadata": {},
             "execution_count": None, "outputs": [], "source": runtime},
        ],
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], cell["id"], "exec")
    encoded = (json.dumps(notebook, indent=1) + "\n").encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != encoded and not replace:
        raise FileExistsError(f"Refusing to overwrite a different notebook: {output}")
    if not output.exists() or replace:
        with output.open("wb" if replace else "xb") as stream:
            stream.write(encoded)
    return {"file": str(output.resolve()), "bytes": len(encoded),
            "notebook_sha256": hashlib.sha256(encoded).hexdigest(),
            "solution_payload_sha256": metadata["payload_sha256"],
            "dev_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "code_cells_compile": True, "gpu_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "notebooks" / "soup_preflight_v1.ipynb")
    parser.add_argument("--replace", action="store_true", help="Regenerate the selected preflight artifact")
    args = parser.parse_args()
    print(json.dumps(build(args.out, replace=args.replace), indent=2))
