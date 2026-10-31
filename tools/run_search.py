"""Measure fixed candidate profiles in isolated source snapshots on a T4.

Each profile must pass native GPU readiness and all unchanged official checks.
Every measured profile uses fresh compiler caches and the full paired protocol.
Stops after an admitted median below the requested target, unless asked to
continue. A failed or slow candidate is retained as evidence, never a winner.

This is a benchmark driver, not a submission or a world-record certification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from select_profile import PROFILES, select_profile
from run_t4_readiness import _stop_child_tree


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def clean_environment():
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env.pop("SOUP_APPLY_SOLUTION", None)
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.update(CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return env


def copy_source(destination):
    def ignore(directory, names):
        return [name for name in names if name in
                {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "validation", "data", "MANIFEST.json"}
                or name == "runs" or name.startswith("runs_")
                or name.endswith((".pyc", ".zip", ".ipynb"))]
    shutil.copytree(ROOT, destination, ignore=ignore)


def commit_snapshot(repo, profile, env):
    commands = [
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.name=Kernel Fusion", "-c", "user.email=kernel-fusion@localhost",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", f"Fixed benchmark candidate: {profile}"],
    ]
    for command in commands:
        subprocess.run(command, cwd=repo, env=env, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                   env=env, text=True).strip()


def run_logged(command, repo, path, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(map(str, command)), flush=True)
    with path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            return process.wait()
        except KeyboardInterrupt:
            # run_t4_readiness owns cancellation of its complete child tree.
            import signal
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                _stop_child_tree(process)
                process.wait()
            raise
        except BaseException:
            _stop_child_tree(process)
            process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES),
                        default=["fused", "cached256", "epilogues", "streamed"])
    parser.add_argument("--out", type=Path, default=ROOT / "runs_search")
    parser.add_argument("--record-seconds", type=float, default=33.0)
    parser.add_argument("--continue-after-target", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.record_seconds) or args.record_seconds <= 0:
        parser.error("--record-seconds must be positive and finite")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("each profile may appear only once")
    if args.plan:
        from select_profile import profile_values
        print(json.dumps({"target_seconds": args.record_seconds,
                          "profiles": {p: profile_values(p) for p in args.profiles},
                          "protocol": "two repeats: base, solution, solution, base",
                          "selection": "all official checks pass, then lowest median wall time",
                          "stop_after_target": not args.continue_after_target}, indent=2))
        return 0
    out = args.out.resolve()
    if out == ROOT or ROOT.is_relative_to(out):
        parser.error("--out must not be the source directory or its parent")
    if out.is_relative_to(ROOT) and not out.relative_to(ROOT).parts[0].startswith("runs_"):
        parser.error("an output inside the source must start with runs_ to avoid recursive copying")
    if out.exists() and any(out.iterdir()):
        parser.error("output is not empty; choose a fresh --out to preserve all prior evidence")
    env = clean_environment()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "search_summary.json"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "target_seconds": args.record_seconds, "user_reported_baseline_seconds": 77,
              "status": "preflight", "profiles": [], "best_admitted_profile": None,
              "target_met_under_official_public_protocol": False,
              "world_record_verified": False, "hidden_jury_admission": "not_established"}
    write_json(summary_path, report)
    # A subprocess owns CUDA initialization; it exits before any measured run.
    probe = out / "preflight.json"
    command = [sys.executable, "tools/run_t4_readiness.py", "--probe-output", str(probe)]
    code = run_logged(command, ROOT, out / "preflight.log", env)
    data = json.loads(probe.read_text()) if probe.is_file() else {}
    hardware = data.get("hardware", {})
    if (code or not hardware.get("cuda_available") or hardware.get("capability") != [7, 5]
            or hardware.get("device_count") != 1 or "T4" not in hardware.get("name", "")):
        report.update(status="blocked_no_single_t4", preflight=data)
        write_json(summary_path, report)
        print(f"A physical single-visible T4 is required. Evidence: {summary_path}")
        return 2
    report["preflight"] = data
    for profile in args.profiles:
        repo = out / profile
        copy_source(repo)
        values = select_profile(profile, repo)
        write_json(repo / "PROFILE.json", {"profile": profile, "settings": values,
                    "origin": "delivered source candidate; this snapshot has separate fixed settings"})
        commit = commit_snapshot(repo, profile, env)
        record = {"profile": profile, "source_directory": str(repo),
                  "local_commit": commit, "settings": values, "status": "running"}
        report["profiles"].append(record)
        report["status"] = "running"
        write_json(summary_path, report)
        command = [sys.executable, "tools/run_t4_readiness.py", "--official", "--local-snapshot",
                   "--expected-commit", commit, "--output-dir", "runs/readiness",
                   "--official-out", "runs/final"]
        code = run_logged(command, repo, repo / "runs/driver.log", env)
        readiness_file = repo / "runs/readiness/readiness.json"
        readiness = json.loads(readiness_file.read_text()) if readiness_file.is_file() else {}
        record.update(returncode=code, status=readiness.get("status", "runner_failed"))
        result_file = repo / "runs/final/result.json"
        if code == 0 and result_file.is_file() and readiness.get("official_checks_passed"):
            verdict_file = repo / "runs/final/target_verdict.json"
            command = [sys.executable, "tools/verify_record.py", str(result_file),
                       "--record-seconds", str(args.record_seconds),
                       "--readiness-report", str(readiness_file), "--output", str(verdict_file)]
            verdict_code = run_logged(command, repo, repo / "runs/target_verdict.log", env)
            verdict = json.loads(verdict_file.read_text()) if verdict_file.is_file() else {}
            result = json.loads(result_file.read_text())
            record.update(official_summary=result.get("summary"), verdict=verdict,
                          verdict_returncode=verdict_code)
            if (verdict_code == 0 and
                    verdict.get("target_met_under_official_public_protocol") is True and
                    verdict.get("passes_recorded_public_automatic_gates") is True):
                report["target_met_under_official_public_protocol"] = True
        admitted = [r for r in report["profiles"]
                    if r.get("status") == "public_official_checks_passed"
                    and r.get("verdict", {}).get("passes_recorded_public_automatic_gates") is True
                    and r.get("official_summary", {}).get("solution", {}).get("ok")]
        if admitted:
            best = min(admitted, key=lambda r: r["verdict"]["measurements"]["solution"]["wall_s"])
            report["best_admitted_profile"] = best["profile"]
            report["best_source_directory"] = best["source_directory"]
            report["best_result"] = str(Path(best["source_directory"]) / "runs/final/result.json")
        write_json(summary_path, report)
        if report["target_met_under_official_public_protocol"] and not args.continue_after_target:
            break
    report["status"] = ("target_met" if report["target_met_under_official_public_protocol"]
                        else "measured_target_not_met" if report["best_admitted_profile"]
                        else "no_profile_admitted")
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(summary_path, report)
    print(json.dumps(report, indent=2))
    return 0 if report["best_admitted_profile"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
