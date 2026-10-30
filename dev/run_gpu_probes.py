"""Run synthetic GPU checks in disposable child processes.

Example, from the candidate repository root:
    python dev/run_gpu_probes.py --out /kaggle/working/probe_reports

The parent imports only the standard library and never creates a CUDA context.
Every child receives fresh Triton and Inductor cache directories; they are
deleted after that child exits and must never be reused by measured training.
This runner is a development check, not part of solution.apply() or the harness.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _run_probe(root, name, arguments, timeout):
    command = [sys.executable, str(root / "dev" / f"{name}_probe.py"), *arguments]
    record = {
        "probe": name,
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cache_policy": "fresh disposable Triton and Inductor caches per child",
    }
    started = time.monotonic()
    # The remote candidate root lives under /kaggle/working. Keep temporary
    # compiler files on that working disk, then remove them before returning.
    with tempfile.TemporaryDirectory(prefix=f".probe-{name}-", dir=root) as temporary:
        cache_root = Path(temporary)
        environment = os.environ.copy()
        environment["TRITON_CACHE_DIR"] = str(cache_root / "triton")
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor")
        environment["PYTHONUNBUFFERED"] = "1"
        timed_out = False
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=os.name == "posix",
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Also retire any compiler children before deleting caches or
                # allowing the next probe/training process to start.
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                stdout, stderr = process.communicate()
            record.update({
                "state": "timeout" if timed_out else (
                    "passed" if process.returncode == 0 else "failed"
                ),
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            })
        except OSError as error:
            record.update({
                "state": "launch_error", "returncode": None,
                "stdout": "", "stderr": str(error),
            })
    record["elapsed_seconds"] = time.monotonic() - started
    record["cache_removed"] = not cache_root.exists()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="Explicit directory for per-probe stdout/stderr/status JSON")
    parser.add_argument("--timeout", type=float, default=600,
                        help="Maximum seconds for each child process (default: 600)")
    parser.add_argument("--check-accuracy", action="store_true",
                        help="Also validate the optional metrics accuracy rewrite")
    arguments = parser.parse_args(argv)
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    output = arguments.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    specifications = (
        ("rmsnorm", []),
        ("mlp", []),
        ("metrics", ["--check-accuracy"] if arguments.check_accuracy else []),
    )
    summaries = []
    for name, flags in specifications:
        print(f"Starting {name} probe", flush=True)
        record = _run_probe(root, name, flags, arguments.timeout)
        report = output / f"{name}.json"
        _write_json(report, record)
        summaries.append({
            "probe": name,
            "state": record["state"],
            "returncode": record["returncode"],
            "elapsed_seconds": record["elapsed_seconds"],
            "cache_removed": record["cache_removed"],
            "report": report.name,
        })
        print(f"{name}: {record['state']} ({record['elapsed_seconds']:.1f}s)", flush=True)
    passed = all(item["state"] == "passed" and item["cache_removed"] for item in summaries)
    summary = {
        "state": "passed" if passed else "failed",
        "parent_uses_cuda": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "probes": summaries,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
