"""Isolated Linux subreaper guardian. Owns children and holds the GPU lock.

The guardian watches a parent-liveness pipe: driver SIGKILL or other abrupt
death closes the pipe and triggers quarantined cleanup. PIDFD signaling in the
shared task helper prevents PID reuse races; descendants are only this dedicated
guardian's children, including newly adopted orphans from nested sessions.
This is task orchestration, never solution/ work or kernel participation.
"""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import owned_process_tree as tree


def quarantine(path, reason):
    """First failure wins; never overwrite or automatically clear a marker."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"status": "OPERATOR_RECONCILIATION_REQUIRED", "reason": reason,
                   "guardian_pid": os.getpid(), "created_unix": time.time()}, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_receipt(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def guardian_main():
    config = json.loads(sys.stdin.readline())
    read_fd, lock_fd = config["parent_pipe_fd"], config["lock_fd"]
    started, requested_signal, process = time.monotonic(), [None], None
    result = {"exit_code": 125, "root_exit_code": None, "timed_out": False,
              "interrupted": False, "cleanup_confirmed": False,
              "reason": None, "guardian_pid": os.getpid()}
    def request(signum, frame):
        requested_signal[0] = signum
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request)
    try:
        tree.prepare()  # required before any child; never silently degrade
        with Path(config["log"]).open("x") as stream:
            process = subprocess.Popen(config["command"], cwd=config["cwd"],
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while True:
                ready, _, _ = select.select([read_fd], [], [], 0.05)
                if ready and os.read(read_fd, 1) == b"":
                    result.update(reason="parent_liveness_pipe_closed", interrupted=True)
                    break
                if requested_signal[0] is not None:
                    result.update(reason="guardian_signal_" + str(requested_signal[0]), interrupted=True)
                    break
                if time.monotonic() - started >= config["timeout"]:
                    result.update(reason="outer_timeout", timed_out=True)
                    break
                returncode = process.poll()
                if returncode is not None:
                    result["root_exit_code"] = returncode
                    if tree.descendants():
                        result["reason"] = "root_exited_with_owned_descendants"
                    else:
                        result.update(exit_code=returncode, cleanup_confirmed=True)
                    break
            if result["reason"] is not None:
                quarantine(config["marker"], result["reason"])
                # cleanup ignores repeated INT/TERM, stops/rescans/kills by
                # pidfd and waits/reaps without any timeout-unlock escape.
                tree.cleanup(process)
                result.update(root_exit_code=process.returncode, cleanup_confirmed=True)
                if result["timed_out"]:
                    result["exit_code"] = 124
                elif result["interrupted"]:
                    result["exit_code"] = 130
    except BaseException as exc:
        result.update(reason="guardian_exception:" + repr(exc), exit_code=125)
        try:
            quarantine(config["marker"], result["reason"])
        finally:
            if process is not None:
                tree.cleanup(process)
            result["cleanup_confirmed"] = not tree.descendants()
    finally:
        result["outer_wall_s"] = time.monotonic() - started
        _write_receipt(config["receipt"], result)
        os.close(read_fd)
        # The inherited open-file description retains the canonical flock even
        # if the driver has died. Close ONLY after owned-tree cleanup completes.
        os.close(lock_fd)
    return result["exit_code"]


def run(command, *, cwd, env, log, timeout, lock_fd,
        marker="/tmp/soup-gpu0.lock.blocked"):
    """Dedicated driver API. Any BaseException asks guardian cleanup and waits."""
    receipt = Path(str(log) + ".process.json").resolve()
    if receipt.exists():
        raise RuntimeError("Process receipt already exists; refuse to overwrite")
    read_fd, write_fd = os.pipe()
    config = {"command": [str(x) for x in command], "cwd": str(Path(cwd).resolve()),
              "log": str(Path(log).resolve()), "timeout": timeout,
              "receipt": str(receipt), "parent_pipe_fd": read_fd,
              "lock_fd": lock_fd, "marker": str(marker)}
    guardian = None
    interrupted = None
    def request(signum, frame):
        raise InterruptedError("Driver received signal " + str(signum))
    old_handlers = {sig: signal.signal(sig, request) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        guardian = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--guardian"],
                                    stdin=subprocess.PIPE, env=env,
                                    pass_fds=(read_fd, lock_fd), start_new_session=True,
                                    text=True)
        os.close(read_fd)
        read_fd = None
        guardian.stdin.write(json.dumps(config) + "\n")
        guardian.stdin.close()
        guardian.wait()
    except BaseException as exc:
        interrupted = repr(exc)
        # Ignore repeated stop signals until the guardian confirms cleanup.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if write_fd is not None:
            os.close(write_fd)
            write_fd = None
        if guardian is not None:
            if guardian.stdin is not None and not guardian.stdin.closed:
                guardian.stdin.close()
            guardian.wait()
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if not receipt.is_file():
        quarantine(marker, "guardian_missing_receipt")
        raise RuntimeError("Guardian receipt missing; GPU quarantine requires operator reconciliation")
    result = json.loads(receipt.read_text())
    if interrupted is not None:
        result.update(parent_exception=interrupted, interrupted=True)
        if result["exit_code"] == 0:
            result["exit_code"] = 130
        quarantine(marker, "driver_interruption:" + interrupted)
    if not result["cleanup_confirmed"]:
        quarantine(marker, "guardian_uncertain_cleanup")
        raise RuntimeError("Guardian cleanup uncertain; quarantine retained")
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--guardian"]:
        raise SystemExit("Internal guardian entrypoint only")
    raise SystemExit(guardian_main())
