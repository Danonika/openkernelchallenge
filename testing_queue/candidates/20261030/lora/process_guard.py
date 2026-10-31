"""Linux-only dedicated-runner cleanup; no GPU imports or background activity.

Original task-orchestration helper. Not part of solution/ or own-kernel credit.
"""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def prepare():
    """Fail before GPU work if race-safe process ownership is unavailable."""
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd process ownership required')
    fd=os.pidfd_open(os.getpid());os.close(fd)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:
        raise RuntimeError('Cannot enable child subreaper; no GPU job started')
    enabled=ctypes.c_int(0)
    if libc.prctl(37,ctypes.byref(enabled),0,0,0)!=0 or enabled.value!=1:
        raise RuntimeError('Child subreaper verification failed')


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return int(fields[1]),int(fields[19]),fields[0]
    except (OSError,ValueError,IndexError):return None


def descendants():
    records={}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            rec=identity(int(p.name))
            if rec is not None:records[int(p.name)]=rec
    owned={os.getpid()}
    while True:
        add={pid for pid,(parent,_,_) in records.items() if parent in owned}-owned
        if not add:break
        owned.update(add)
    return {pid:records[pid] for pid in owned if pid!=os.getpid() and pid in records}


def signal_exact(pid,birth,sig):
    """Never signal a recycled PID. The FD pins identity through the send."""
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:return
    try:
        now=identity(pid)
        if now is not None and now[1]==birth:
            try:signal.pidfd_send_signal(fd,sig,None,0)
            except ProcessLookupError:pass
    finally:os.close(fd)


def _try_signal(pid,birth,sig):
    try:signal_exact(pid,birth,sig)
    except OSError as exc:
        # Do not unwind/release GPU ownership when cleanup is uncertain.
        quarantine('Cleanup signal failed: '+repr(exc))
        print('SOUP_CLEANUP_PENDING '+repr(exc),file=sys.stderr,flush=True)


def cleanup(proc):
    saved={s:signal.signal(s,signal.SIG_IGN) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        frozen=set()
        while True:
            observed=descendants()
            new={(pid,v[1]) for pid,v in observed.items()}-frozen
            if not new:break
            for pid,birth in new:_try_signal(pid,birth,signal.SIGSTOP)
            frozen.update(new)
        for pid,birth in frozen:_try_signal(pid,birth,signal.SIGKILL)
        proc.wait()
        # No timeout-based release: if a GPU child is stuck in kernel exit,
        # retain this supervisor and its lock until exit is actually observed.
        while True:
            while True:
                try:
                    pid,_=os.waitpid(-1,os.WNOHANG)
                    if not pid:break
                except ChildProcessError:break
            remaining=descendants()
            if not remaining:break
            for pid,(_,birth,_) in remaining.items():_try_signal(pid,birth,signal.SIGKILL)
            time.sleep(.05)
    finally:
        for sig,handler in saved.items():signal.signal(sig,handler)


def run(argv,*,cwd,env,stdout,timeout=None,stderr=subprocess.STDOUT):
    """Run in dedicated supervisor only; cleanup nested sessions on any failure."""
    proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
    try:
        code=proc.wait(timeout=timeout)
        if code < 0:quarantine('Child terminated by signal '+str(-code))
        if descendants():
            quarantine('Child exited leaving descendants')
            cleanup(proc)
            raise RuntimeError('Child exited leaving descendants; cleaned and rejected')
        return code
    except BaseException as exc:
        quarantine('Outer timeout/interruption/error: '+repr(exc))
        cleanup(proc)
        raise


def install_signal_handlers():
    def interrupted(signum,frame):
        quarantine('Runner interrupted by signal '+str(signum))
        raise InterruptedError('Runner interrupted by signal '+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)

# A crash-surviving marker is placed before any GPU subprocess. Only its exact
# owner may clear it on a clean, non-interrupted exit with no live descendants.
# Interrupted/timed-out work leaves it for explicit operator reconciliation.
_MARKER = None
_TOKEN = None
_KEEP_MARKER = False


def check_quarantine(path):
    if Path(path).exists():
        raise RuntimeError('GPU0 quarantined: reconcile '+str(path)+' before another run')


def begin_quarantine(path):
    global _MARKER,_TOKEN,_KEEP_MARKER
    import json,uuid
    _MARKER=Path(path);_TOKEN=uuid.uuid4().hex;_KEEP_MARKER=False
    fd=os.open(str(_MARKER),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as out:
        json.dump({'owner_token':_TOKEN,'owner_pid':os.getpid(),'status':'OWNED_GPU_WORK_IN_PROGRESS'},out)
        out.flush();os.fsync(out.fileno())


def quarantine(reason):
    global _KEEP_MARKER
    _KEEP_MARKER=True
    if _MARKER is None:return
    import json
    try:value=json.loads(_MARKER.read_text())
    except (OSError,ValueError):return
    if value.get('owner_token')!=_TOKEN:return
    value.update(status='BLOCKED_OPERATOR_RECONCILIATION_REQUIRED',reason=str(reason))
    _MARKER.write_text(json.dumps(value)+'\n')


def finish_quarantine():
    if _MARKER is None or _KEEP_MARKER:return
    if descendants():
        quarantine('Live owned descendants remain at supervisor exit')
        return
    import json
    try:value=json.loads(_MARKER.read_text())
    except (OSError,ValueError):return
    if value.get('owner_token')==_TOKEN:_MARKER.unlink()
