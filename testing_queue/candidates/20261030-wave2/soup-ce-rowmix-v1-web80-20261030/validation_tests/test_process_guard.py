# Adapted path/output only from the frozen head lane guardian fixtures.
"""CPU sleeping-child tests only: nested sessions, signals, BaseException, parent death."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent / "web_adapter"
records = []


def wait_for(fn, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = fn()
        if result:
            return result
        time.sleep(0.03)
    raise AssertionError("CPU fixture wait expired")


def fixture(name, signal_to_parent=None, timeout=20, orphan=False):
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        leaf = "import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); time.sleep(100)" % str(tmp / "leaf.pid")
        root_code = ("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r],start_new_session=True); " % leaf)
        root_code += "time.sleep(.2)" if orphan else "time.sleep(100)"
        parent = '''import fcntl,json,os,signal,sys
from pathlib import Path
sys.path.insert(0, ROOT)
import process_guard
lock=open(LOCK,'w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def custom(sig,frame): raise RuntimeError('fixture BaseException path')
signal.signal(signal.SIGUSR1,custom)
result=process_guard.run([sys.executable,'-c',COMMAND],cwd=DIR,env=dict(os.environ),
                        log=LOG,timeout=TIMEOUT,lock_fd=lock.fileno(),marker=MARKER)
Path(RESULT).write_text(json.dumps(result))
'''
        definitions = {"ROOT": str(ROOT), "LOCK": str(tmp / "lock"), "COMMAND": root_code,
                       "DIR": str(tmp), "LOG": str(tmp / "child.log"), "TIMEOUT": timeout,
                       "MARKER": str(tmp / "blocked"), "RESULT": str(tmp / "result.json")}
        parent = "\n".join(f"{key}={value!r}" for key, value in definitions.items()) + "\n" + parent
        proc = subprocess.Popen([sys.executable, "-c", parent], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        leaf_pid = int(wait_for(lambda: (tmp / "leaf.pid").read_text() if (tmp / "leaf.pid").is_file() else None))
        if signal_to_parent is not None:
            proc.send_signal(signal_to_parent)
        stdout, stderr = proc.communicate(timeout=15)
        receipt = tmp / "child.log.process.json"
        wait_for(receipt.is_file)
        result = json.loads(receipt.read_text())
        assert result["cleanup_confirmed"], result
        assert result["exit_code"] != 0, result
        wait_for(lambda: not Path(f"/proc/{leaf_pid}").exists())
        assert (tmp / "blocked").is_file()
        # A separate process proves flock released only after observed cleanup.
        probe = subprocess.run([sys.executable, "-c", "import fcntl; f=open(%r,'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)" % str(tmp / "lock")], capture_output=True)
        assert probe.returncode == 0
        records.append({"name": name, "pass": True, "guardian_receipt": result,
                        "nested_session_child_gone": True, "quarantine_preserved": True,
                        "lock_available_after_cleanup": True, "parent_exit": proc.returncode})


fixture("nested_session_timeout", timeout=0.6)
fixture("parent_SIGINT", signal.SIGINT)
fixture("parent_SIGTERM", signal.SIGTERM)
fixture("parent_BaseException", signal.SIGUSR1)
fixture("abrupt_parent_SIGKILL", signal.SIGKILL)
fixture("root_exit_orphan_reparenting", orphan=True)
with tempfile.TemporaryDirectory() as directory:
    import fcntl
    sys.path.insert(0, str(ROOT))
    import process_guard
    tmp = Path(directory)
    with (tmp / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        normal = process_guard.run([sys.executable, "-c", "print('normal CPU completion')"],
                                   cwd=tmp, env=dict(os.environ), log=tmp / "normal.log",
                                   timeout=3, lock_fd=lock.fileno(), marker=tmp / "blocked")
    assert normal["exit_code"] == 0 and normal["cleanup_confirmed"]
    assert not (tmp / "blocked").exists()
    records.append({"name": "normal_completion_no_quarantine", "pass": True})
result = {"scope": "CPU sleeping children only; no GPU or canonical marker touched",
          "passed": len(records), "failed": 0, "records": records}
(ROOT.parent / "validation" / "process_guard_results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"passed": len(records), "failed": 0}))
