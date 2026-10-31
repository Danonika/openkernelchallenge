#!/usr/bin/env python3
"""Offline paired official harness driver; --execute is explicit GPU authority.

The notebook embeds this file and the exact reviewed source ZIP. This driver
never installs packages, downloads assets, submits queue jobs, or edits the
provided official checkout. It stages a fresh copy with only solution/ replaced.
"""
import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import zipfile

COMMIT = "c7c2fcd15972ce8b898d8d3ff732ee192c817ceb"
MODEL = "models--Qwen--Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PINS = {"soup-cli": "0.77.0", "torch": "2.11.0+cu128", "triton": "3.6.0",
        "transformers": "5.16.1", "peft": "0.20.0", "trl": "0.29.1",
        "bitsandbytes": "0.50.2", "accelerate": "1.14.0", "datasets": "4.8.5"}
DATA = {"data/train.jsonl": "93d036d912bd81c337dc29893e13a76f36161d96f896ef4e02492e2f1a21a0dd",
        "data/eval_public.jsonl": "b7f85b78ba3186dce4deab7cfb8bf8dcf7998e1517c5749ac0f5005886c6f70e"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_preflight(repo, expected):
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise RuntimeError("Official checkout must be at the pinned commit")
    for name, wanted in expected.items():
        path = repo / name
        if path.is_symlink() or not path.is_file() or sha(path) != wanted:
            raise RuntimeError("Official source mismatch: " + name)
        committed = subprocess.check_output(["git", "-C", str(repo), "show", COMMIT + ":" + name])
        if hashlib.sha256(committed).hexdigest() != wanted:
            raise RuntimeError("Pinned Git blob mismatch: " + name)


def asset_preflight(repo, cache):
    for name, wanted in DATA.items():
        path = repo / name
        if path.is_symlink() or not path.is_file() or sha(path) != wanted:
            raise RuntimeError("Missing or different public data: " + name)
    root = cache / MODEL
    for folder in (root, root / "refs", root / "blobs", root / "snapshots", root / "snapshots" / REVISION):
        if folder.is_symlink() or not folder.is_dir():
            raise RuntimeError("Prepared pinned model cache missing: " + str(folder))
    ref = root / "refs" / "main"
    if ref.is_symlink() or ref.read_text().strip() != REVISION:
        raise RuntimeError("Model refs/main must resolve to the frozen public revision")
    files, memo = {}, {}
    snapshot = root / "snapshots" / REVISION
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            if path.is_symlink():
                raise RuntimeError("Linked snapshot directories rejected")
            continue
        actual = path.resolve(strict=True)
        if not actual.is_relative_to((root / "blobs").resolve()) or not actual.is_file():
            raise RuntimeError("Snapshot links must resolve inside the selected public blobs")
        if actual not in memo:
            memo[actual] = sha(actual)
        files[path.relative_to(cache).as_posix()] = memo[actual]
    required = ("config.json", "tokenizer.json")
    if any(not (snapshot / name).is_file() for name in required):
        raise RuntimeError("Prepared snapshot is incomplete")
    # The pinned retained snapshot has a single model.safetensors. Also accept
    # a genuine indexed snapshot only when all of its referenced shards exist.
    single = snapshot / "model.safetensors"
    index_path = snapshot / "model.safetensors.index.json"
    if not single.is_file():
        if not index_path.is_file():
            raise RuntimeError("Prepared model weights are missing")
        index = json.loads(index_path.read_text())
        if not index.get("weight_map") or any(not (snapshot / name).is_file()
                                              for name in set(index["weight_map"].values())):
            raise RuntimeError("Prepared weight shards missing")
    return {"revision": REVISION, "files": files, "data_files": DATA}


def hardware_preflight():
    if platform.python_version() != "3.13.15":
        raise RuntimeError("Python must be the recorded 3.13.15; no installation is attempted")
    actual = {name: version(name) for name in PINS}
    if actual != PINS:
        raise RuntimeError("Installed direct package pins differ: " + repr(actual))
    # The CUDA probe runs in a child so its context is gone before measurement.
    probe = "import json,torch; print(json.dumps({'count':torch.cuda.device_count(),'name':torch.cuda.get_device_name(0),'cap':list(torch.cuda.get_device_capability(0)),'cuda':torch.version.cuda}))"
    visible = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    if visible != {"count": 1, "name": "Tesla T4", "cap": [7, 5], "cuda": "12.8"}:
        raise RuntimeError("One visible Tesla T4 / CUDA 12.8 required: " + repr(visible))
    # Only check idle after the child has exited; this process never imports Torch.
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    rows = [[field.strip() for field in row.split(",")] for row in inventory.strip().splitlines()]
    if not rows or sum(int(row[3]) for row in rows) >= 1000:
        raise RuntimeError("GPU is not idle under the official summed-memory rule")
    return {"python": platform.python_version(), "packages": actual, "physical_inventory": rows,
            "visible_device": visible, "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "platform": platform.platform()}


def unpack_reviewed(bundle, manifest, stage):
    with zipfile.ZipFile(bundle) as archive:
        if set(archive.namelist()) != set(manifest) or len(archive.namelist()) != len(manifest):
            raise RuntimeError("ZIP source set does not match manifest")
        for name, wanted in manifest.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "solution":
                raise RuntimeError("Unsafe bundle member")
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != wanted:
                raise RuntimeError("ZIP member hash mismatch: " + name)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)


def run_logged(command, cwd, env, log, timeout, lock_fd):
    from process_guard import run
    return run(command, cwd=cwd, env=env, log=log, timeout=timeout, lock_fd=lock_fd)


def complete_result(result):
    reasons = []
    if result.get("packages_changed"):
        reasons.append("packages_changed")
    for side in ("base", "solution"):
        runs = result.get("runs", {}).get(side, [])
        if len(runs) != 2:
            reasons.append(side + "_run_count")
        for run in runs:
            metrics = run.get("metrics", {})
            if (run.get("exit_code") != 0 or metrics.get("exit_code") != 0
                    or metrics.get("steps") != 64 or run.get("global_step") != 64
                    or not run.get("adapter_config") or not run.get("adapter_tensors")
                    or not run.get("rewind") or not run.get("logged_metrics")):
                reasons.append(run.get("run", side) + "_incomplete")
        evaluation = result.get("evals", {}).get(side, {})
        loss = evaluation.get("mean_loss")
        if evaluation.get("exit_code") != 0 or type(loss) not in (float, int) or not math.isfinite(loss):
            reasons.append(side + "_evaluation_incomplete")
    return reasons


def activation_evidence(output):
    evidence = {}
    for log in sorted(Path(output).glob("solution_*/train.log")):
        counts = {}
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("SOUP_HEAD_NATIVE "):
                try:
                    entry = json.loads(line.split(" ", 1)[1])
                    counts[entry["event"]] = max(counts.get(entry["event"], 0), entry["count"])
                except (ValueError, KeyError, TypeError):
                    pass
        evidence[log.parent.name] = counts
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("GCP", "KAGGLE"), required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--outer-timeout", type=int, default=8100)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Prepared only. Explicit --execute is required for separately authorized GPU execution.")
    if args.timeout <= 0 or args.outer_timeout < 6 * args.timeout + 1800:
        raise SystemExit("Freeze adequate identical baseline/candidate timeouts before the run")
    if os.path.lexists("/tmp/soup-gpu0.lock.blocked"):
        raise SystemExit("GPU quarantine marker exists; operator reconciliation is required")
    # Canonical lock shared with all six notebooks; fail rather than queue or overlap.
    descriptor = os.open("/tmp/soup-gpu0.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as gpu_lock:
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("GPU0 is owned by another authorized notebook; no run started")
        if os.path.lexists("/tmp/soup-gpu0.lock.blocked"):
            raise SystemExit("GPU quarantine appeared before lock acquisition; no run started")
        root = Path(__file__).resolve().parent
        official = json.loads((root / "official-source-manifest.json").read_text())
        manifest = json.loads((root / "candidate-files.json").read_text())
        bundle_info = json.loads((root / "bundle-info.json").read_text())
        if sha(root / "candidate.zip") != bundle_info["source_sha256"]:
            raise RuntimeError("Bundle ZIP hash mismatch")
        if args.out.exists():
            raise RuntimeError("Output path exists; use a new path. Never reuse measured state.")
        # Set child environment before importing Torch/Transformers anywhere here.
        os.environ.update(CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                          PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                          HF_HUB_CACHE=str(args.cache.resolve()), HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        source_preflight(args.repo.resolve(), official)
        assets = asset_preflight(args.repo.resolve(), args.cache.resolve())
        environment = hardware_preflight()
        args.out.mkdir(parents=True, exist_ok=False)
        stage = args.out / "repo"
        stage.mkdir()
        # Copy exactly reviewed immutable non-solution bytes, not a mutable checkout.
        for name in official:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((args.repo / name).read_bytes())
        for name in DATA:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((args.repo / name).read_bytes())
        unpack_reviewed(root / "candidate.zip", manifest, stage)
        save(args.out / "environment.json", environment)
        save(args.out / "assets.json", assets)
        save(args.out / "cohort.json", {"backend": args.backend, "commit": COMMIT,
             "source_sha256": bundle_info["source_sha256"], "timeout_s": args.timeout,
             "outer_timeout_s": args.outer_timeout, "repeats": 2, "seed": 1,
             "baseline": "Fresh native base_1 and base_2 in this run; no historical timing imported"})
        result_dir = args.out.resolve() / "measured"
        env = dict(os.environ)
        command = [sys.executable, "harness/run_task.py", "--repeats", "2", "--seed", "1",
                   "--timeout", str(args.timeout), "--out", str(result_dir)]
        outer = run_logged(command, stage, env, args.out / "harness.log", args.outer_timeout, gpu_lock.fileno())
        report = {"status": "FAILED_OR_PARTIAL", "outer": outer, "backend": args.backend,
                  "admission": "BLOCKED_NO_ORIGINAL_TRITON", "score": None,
                  "quality": None, "ratios": None, "own_triton_share": 0.0,
                  "native_replay_activation": activation_evidence(result_dir)}
        raw_path = result_dir / "result.json"
        reasons = []
        if outer["exit_code"] != 0 or outer["timed_out"] or outer["interrupted"]:
            reasons.append("outer_failure_or_interruption")
        if raw_path.is_file():
            raw = json.loads(raw_path.read_text())
            reasons += complete_result(raw)
            report["raw_result_sha256"] = sha(raw_path)
        else:
            reasons.append("missing_native_result")
        report["completion_problems"] = reasons
        if not reasons:
            checked = run_logged([sys.executable, "harness/check_rules.py", "--result", str(raw_path)],
                                 stage, env, args.out / "official-checks.log", 900, gpu_lock.fileno())
            report["official_checks_process"] = checked
            checks = result_dir / "checks.json"
            if checks.is_file():
                report["checks_sha256"] = sha(checks)
            activated = (len(report["native_replay_activation"]) == 2 and
                         all(value.get("native_sum_backward_return", 0) == 64
                             for value in report["native_replay_activation"].values()))
            report["status"] = "COMPLETED_RESEARCH_CONTROL" if activated else "COMPLETED_BUT_CANDIDATE_INACTIVE"
            report["quality"] = {side: raw["evals"][side]["mean_loss"] for side in ("base", "solution")}
            report["ratios"] = raw.get("summary", {}).get("ratios")
            report["note"] = "Only complete-run raw research comparisons; zero own Triton means no admission or score."
        # Check immutable source bytes after execution as well; no silently edited harness.
        changed = [name for name, wanted in official.items() if sha(stage / name) != wanted]
        report["official_files_changed"] = changed
        if changed:
            report.update(status="INVALID_SOURCE_MUTATION", quality=None, ratios=None)
        save(args.out / "research-result.json", report)
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
