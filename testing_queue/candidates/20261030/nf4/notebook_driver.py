"""Prepared-assets-only paired runner. This is orchestration, not solution code.

No downloads, installs, queue submissions, edits of prepared checkout, or scoring.
Calling run requires current authorization and an idle single-visible-T4 session.
"""
import hashlib
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid

COMMIT = "c7c2fcd15972ce8b898d8d3ff732ee192c817ceb"
QUARANTINE_PATH = Path("/tmp/soup-gpu0.lock.blocked")
GPU_LOCK_PATH = Path("/tmp/soup-gpu0.lock")
DATA = {"data/train.jsonl": "93d036d912bd81c337dc29893e13a76f36161d96f896ef4e02492e2f1a21a0dd",
        "data/eval_public.jsonl": "b7f85b78ba3186dce4deab7cfb8bf8dcf7998e1517c5749ac0f5005886c6f70e"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def process_record(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return {"parent": int(fields[1]), "state": fields[0], "birth": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def remember_owned(known):
    snapshot = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        pid = int(stat.parent.name)
        record = process_record(pid)
        if record is not None:
            snapshot[pid] = record
    alive = {pid for pid, birth in known.items()
             if pid in snapshot and snapshot[pid]["birth"] == birth}
    while True:
        new = {pid for pid, record in snapshot.items() if record["parent"] in alive}
        if new <= alive:
            break
        alive |= new
    for pid in alive:
        known[pid] = snapshot[pid]["birth"]
    return known


def signal_owned(pid, birth, sig):
    # Open a stable process handle, then verify its /proc birth identity. A PID
    # reused after that check cannot redirect the signal to a different process.
    try:
        fd = os.pidfd_open(pid, 0)
    except ProcessLookupError:
        return
    try:
        record = process_record(pid)
        if record is None or record["birth"] != birth or record["state"] == "Z":
            return
        try:
            signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            pass
    finally:
        os.close(fd)


def terminate_owned_tree(proc, known):
    # Freeze each birth-identified process, rescan for children created before its
    # stop, and repeat until no new identities remain. Never kill a shared group.
    while True:
        remember_owned(known)
        before = dict(known)
        for pid, birth in known.items():
            signal_owned(pid, birth, signal.SIGSTOP)
        remember_owned(known)
        if known == before:
            break
    for pid, birth in sorted(known.items(), reverse=True):
        signal_owned(pid, birth, signal.SIGKILL)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    # SIGKILL delivery is asynchronous. Wait for observed identities to become
    # absent/zombie rather than mistaking a still-transitioning task for a leak.
    cleanup_deadline = time.monotonic() + 5
    while True:
        live = []
        for pid, birth in known.items():
            record = process_record(pid)
            if record is not None and record["birth"] == birth and record["state"] != "Z":
                live.append(pid)
        if not live or time.monotonic() >= cleanup_deadline:
            break
        time.sleep(0.01)
    if live:
        # Never turn uncertain cleanup into permission for a subsequent notebook.
        QUARANTINE_PATH.write_text(json.dumps({"reason": "owned_cleanup_uncertain", "pids": live}))
    return sorted(known), not live


def command(cmd, cwd, env, log, timeout):
    start = time.perf_counter()
    timed_out = False
    cleanup_pids = []
    cleanup_confirmed = None
    interrupted = None
    known = {}
    p = None
    with Path(log).open("x") as stream:
        try:
            p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            root = process_record(p.pid)
            if root is not None:
                known[p.pid] = root["birth"]
            deadline = time.monotonic() + timeout
            while True:
                remember_owned(known)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                try:
                    code = p.wait(timeout=min(remaining, 0.5))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
        except subprocess.TimeoutExpired:
            timed_out = True
            cleanup_pids, cleanup_confirmed = terminate_owned_tree(p, known)
            code = p.returncode
        except BaseException as exc:
            interrupted = type(exc).__name__
            if p is not None:
                if p.pid not in known:
                    root = process_record(p.pid)
                    if root is not None:
                        known[p.pid] = root["birth"]
                cleanup_pids, cleanup_confirmed = terminate_owned_tree(p, known)
            # A constructor interruption may leave no returned PID handle; its
            # evidence stays unknown and quarantine is mandatory, never retried.
            raise
        finally:
            record = {"command": cmd, "exit_code": p.returncode if p is not None else None,
                      "outer_timeout": timed_out, "interrupted": interrupted,
                      "outer_wall_s": time.perf_counter() - start, "log": str(log),
                      "timeout_owned_pids_signalled": cleanup_pids,
                      "cleanup_confirmed": cleanup_confirmed}
            if timed_out or interrupted is not None:
                # A process census cannot prove that no descendant escaped before
                # it was observed. Quarantine this lane until explicit operator
                # reconciliation; do not equate observed cleanup with safe retry.
                QUARANTINE_PATH.write_text(json.dumps({"reason": "outer_attempt_interrupted_or_timed_out",
                    "log": str(log), "observed_cleanup_confirmed": cleanup_confirmed,
                    "requires_operator_reconciliation": True}))
            dump(str(log) + ".receipt.json", record)
    if cleanup_confirmed is False:
        raise RuntimeError("Owned process cleanup uncertain; GPU lock blocked marker written")
    return record


def run(config, payload, frozen_manifest):
    if config.get("authorized") is not True:
        raise RuntimeError("Prepared only. Explicit current GPU execution authorization is required.")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd cleanup support is required before any GPU work")
    # Check kernel support now, not for the first time during an interruption.
    probe_fd = os.pidfd_open(os.getpid(), 0)
    os.close(probe_fd)
    # Canonical cross-notebook lock: no second GPU lane can overlap this pair.
    if QUARANTINE_PATH.exists():
        raise RuntimeError("GPU0 cleanup requires reconciliation; blocked marker exists")
    with GPU_LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another reviewed notebook owns GPU0; not started") from exc
        previous_term = signal.getsignal(signal.SIGTERM)
        owner = uuid.uuid4().hex
        try:
            if QUARANTINE_PATH.exists():
                raise RuntimeError("GPU0 cleanup requires reconciliation; blocked marker exists")
            # Crash-first marker: SIGKILL/kernel death cannot run Python cleanup.
            # It persists after flock is released by the OS, blocking every next
            # cooperating notebook until explicit operator reconciliation.
            with QUARANTINE_PATH.open("x") as marker:
                json.dump({"owner": owner, "pid": os.getpid(), "state": "active_attempt",
                           "requires_operator_reconciliation_if_abandoned": True}, marker)
                marker.flush()
                os.fsync(marker.fileno())
            def terminate(signum, frame):
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, terminate)
            result = _run_locked(config, payload, frozen_manifest)
            # Normal complete return only. Timeout/interrupt writes a marker with
            # no matching owner, and any exception leaves the active marker.
            marker = json.loads(QUARANTINE_PATH.read_text())
            if marker.get("owner") != owner:
                raise RuntimeError("Attempt cleanup quarantined; operator reconciliation required")
            QUARANTINE_PATH.unlink()
            return result
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run_locked(config, payload, frozen_manifest):
    if config.get("authorized") is not True:
        raise RuntimeError("Prepared only. Explicit current GPU execution authorization is required.")
    if config.get("backend") not in ("KAGGLE_T4", "GCP"):
        raise ValueError("Declare KAGGLE_T4 or GCP; each run measures its own same-environment baseline")
    prepared = Path(config["prepared_repo"]).expanduser().resolve(strict=True)
    cache = Path(config["hf_hub_cache"]).expanduser().resolve(strict=True)
    output_parent = Path(config["output_parent"]).expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    runroot = output_parent / ("nf4_blockpair_" + uuid.uuid4().hex)
    runroot.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
               HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", SOUP_NO_AUDIT_LOG="1")
    env.pop("TRITON_INTERPRET", None)
    # No implicit GPU use in the notebook kernel. All GPU work is in child processes.
    helper = runroot / "support"
    helper.mkdir()
    for name, contents in payload.items():
        if name.startswith("support/"):
            dest = runroot / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(contents)
    commit = subprocess.check_output(["git", "-C", str(prepared), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("Prepared checkout is not pinned official starter")
    for rel, wanted in frozen_manifest.items():
        if sha(prepared / rel) != wanted:
            raise RuntimeError("Frozen official source mismatch: " + rel)
    for rel, wanted in DATA.items():
        if sha(prepared / rel) != wanted:
            raise RuntimeError("Fixed public data mismatch: " + rel)
    # Check exact public cache revision/bytes, never fetch a model or prepare data.
    assets = command([sys.executable, str(helper / "prepare_asset_manifest.py"), "--cache", str(cache),
                      "--repo", str(prepared), "--out", str(runroot / "model_manifest.json")],
                     prepared, env, runroot / "assets.log", 600)
    if assets["exit_code"] != 0:
        raise RuntimeError("Prepared assets are incomplete; see " + assets["log"])
    # Build an isolated source copy from local pinned git objects, not a network clone.
    raw = subprocess.check_output(["git", "-C", str(prepared), "archive", COMMIT])
    stage = runroot / "repo"
    stage.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            dest = (stage / member.name).resolve()
            if not dest.is_relative_to(stage) or member.issym() or member.islnk():
                raise RuntimeError("Unexpected archive path/link")
        archive.extractall(stage, filter="data")
    shutil.rmtree(stage / "solution")
    (stage / "solution").mkdir()
    for name, contents in payload.items():
        if name.startswith("solution/"):
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(contents)
    (stage / "data").mkdir(exist_ok=True)
    for rel in DATA:
        shutil.copy2(prepared / rel, stage / rel)
    for rel, wanted in frozen_manifest.items():
        if sha(stage / rel) != wanted:
            raise RuntimeError("Staged official source mismatch: " + rel)
    env["PYTHONPATH"] = str(stage)
    # Inventory subprocess exits before preflight/measurement: no persistent CUDA context.
    inventory = command([sys.executable, str(helper / "check_kaggle_environment.py"), "--gpu",
                         "--require-four-cpus", "--out", str(runroot / "environment.json")],
                        stage, env, runroot / "inventory.log", 120)
    if inventory["exit_code"] != 0:
        raise RuntimeError("Pinned environment/T4 check failed; see " + inventory["log"])
    current = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if sum(int(line) for line in current.splitlines()) >= 1000:
        raise RuntimeError("GPU idle-start requirement not met")
    manifest = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in payload.items() if name.startswith("solution/")}
    dump(runroot / "executed_source_manifest.json", manifest)
    dump(runroot / "cohort.json", {"backend": config["backend"], "official_commit": COMMIT,
         "seed": 1, "repeats_each": 2, "order": ["base", "solution", "solution", "base"],
         "native_timeout_s": 900, "outer_timeout_s": 10800, "baseline_id": None,
         "baseline_note": "Own paired baseline, no inherited Kaggle79s limit; not a registered queue baseline",
         "score": None, "model_manifest_sha256": sha(runroot / "model_manifest.json")})
    # Fixed, bounded fixture; no tuning. Its caches are isolated and never reused by measurements.
    pre_env = dict(env, TRITON_CACHE_DIR=str(runroot / "preflight_triton"),
                   TORCHINDUCTOR_CACHE_DIR=str(runroot / "preflight_inductor"))
    preflight = command([sys.executable, str(helper / "gpu_preflight.py"), "--out", str(runroot / "preflight.json")],
                        stage, pre_env, runroot / "preflight.log", 600)
    if preflight["exit_code"] != 0 or not (runroot / "preflight.json").is_file():
        dump(runroot / "summary.json", {"status": "PREFLIGHT_FAILED_OR_INCOMPLETE", "score": None,
              "admission": "UNKNOWN", "preflight": preflight})
        raise RuntimeError("Exact preflight failed/incomplete; measured pair not started")
    # The preflight subprocess has exited; recheck idle now, not only before it.
    current = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if sum(int(line) for line in current.splitlines()) >= 1000:
        raise RuntimeError("Post-preflight GPU idle-start requirement not met")
    # Unmodified official runner provides fresh child process/model/adapter/optimizer/output
    # and empty Triton/Inductor directories for every measured side and repetition.
    output = runroot / "measured"
    receipt = command([sys.executable, "harness/run_task.py", "--repeats", "2", "--seed", "1",
                       "--timeout", "900", "--out", str(output)], stage, env,
                      runroot / "paired.log", 10800)
    result_path = output / "result.json"
    strict_complete = False
    checker = None
    details = []
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        for side in ("base", "solution"):
            runs = result.get("runs", {}).get(side, [])
            side_ok = len(runs) == 2
            for r in runs:
                m = r.get("metrics", {})
                ok = (r.get("exit_code") == 0 and m.get("exit_code") == 0
                      and r.get("global_step") == 64 and m.get("steps") == 64
                      and bool(r.get("adapter_tensors")) and bool(r.get("adapter_config")))
                side_ok = side_ok and ok
                details.append({"side": side, "run": r.get("run"), "complete": bool(ok),
                      "command_wall_s": r.get("wall_s"), "external_peak_mib": r.get("gpu_peak_mib"),
                      "own_kernel_ms": m.get("own_kernel_ms"), "timed_loop_s": m.get("timed_loop_s"),
                      "triton_launches": m.get("triton_launches")})
            ev = result.get("evals", {}).get(side, {})
            loss = ev.get("mean_loss")
            side_ok = side_ok and ev.get("exit_code") == 0 and isinstance(loss, (int, float)) and math.isfinite(loss)
            if side == "base":
                strict_complete = side_ok
            else:
                strict_complete = strict_complete and side_ok
        strict_complete = bool(strict_complete and receipt["exit_code"] == 0 and not receipt["outer_timeout"])
        # Even unsuccessful result gets truthful official checker output, not a score.
        checker = command([sys.executable, "harness/check_rules.py", "--result", str(result_path)],
                          stage, env, runroot / "checks.log", 900)
    activation = []
    for log in output.glob("solution_*/train.log"):
        activation.extend({"log": str(log), "receipt": line.strip()} for line in log.read_text(errors="replace").splitlines()
                          if line.startswith("SOUP_NF4_ROUTE "))
    dump(runroot / "activation_receipts.json", activation)
    checks = json.loads((output / "checks.json").read_text()) if (output / "checks.json").exists() else None
    summary = {"status": "COMPLETE_RESEARCH_PAIR" if strict_complete else "FAILED_PARTIAL_OR_TIMEOUT",
               "complete": strict_complete, "score": None, "admission": "UNKNOWN",
               "backend": config["backend"], "runs": details, "outer": receipt, "checker": checker,
               "result_sha256": sha(result_path) if result_path.exists() else None,
               "checks_sha256": sha(output / "checks.json") if checks is not None else None,
               "activation_receipt_count": len(activation),
               "all_10_preliminary_checks_passed": bool(strict_complete and checker and checker["exit_code"] == 0 and checks and len(checks.get("checks", [])) == 10 and all(c.get("ok") is True for c in checks["checks"])),
               "limits": "Official checks are preliminary; no hidden quality or final Kaggle admission. GCP is its own cohort."}
    dump(runroot / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not strict_complete:
        raise RuntimeError("Attempt incomplete; all evidence retained, score remains null: " + str(runroot))
    return runroot
