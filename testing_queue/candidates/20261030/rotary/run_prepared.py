"""Future opt-in runner for an already-prepared cloud T4. No downloads or installs."""
import argparse
import datetime
import ctypes
import fcntl
import hashlib
import importlib.util
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
import zipfile

COMMIT="c7c2fcd15972ce8b898d8d3ff732ee192c817ceb"
_UNCERTAIN=False


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def enable_subreaper():
    # The unchanged harness creates nested sessions. Adopt orphaned grandchildren
    # into this dedicated runner so they cannot disappear from our cleanup tree.
    if not hasattr(os,"pidfd_open") or not hasattr(signal,"pidfd_send_signal"):
        raise RuntimeError("Race-safe Linux pidfd signals unavailable; refusing GPU execution")
    fd=os.pidfd_open(os.getpid(),0)
    try:signal.pidfd_send_signal(fd,0,None,0)
    finally:os.close(fd)
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise RuntimeError("Linux child-subreaper setup failed; refusing GPU execution")


def _proc_record(pid):
    try:
        text=Path(f"/proc/{pid}/stat").read_text()
        fields=text[text.rfind(")")+2:].split()
        return (int(fields[1]),int(fields[19]),fields[0])  # parent, birth ticks, state
    except (OSError,ValueError,IndexError):return None


def _owned_descendants():
    records={}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            value=_proc_record(int(path.name))
            if value is not None:records[int(path.name)]=value
    owned={os.getpid()};changed=True
    while changed:
        added={pid for pid,(parent,_,_) in records.items() if parent in owned and pid not in owned}
        changed=bool(added);owned.update(added)
    return {pid:records[pid] for pid in owned if pid!=os.getpid() and pid in records}


def _signal_same_process(pid,birth,sig):
    try:fd=os.pidfd_open(pid,0)
    except ProcessLookupError:return
    try:
        # Verify identity AFTER pinning it. No check-then-os.kill PID reuse race.
        now=_proc_record(pid)
        if now is not None and now[1]==birth:
            try:signal.pidfd_send_signal(fd,sig,None,0)
            except ProcessLookupError:pass
    finally:os.close(fd)


def _cleanup_owned(proc):
    # All child processes of this dedicated runner belong to its active command.
    # Freeze recursively before killing so children cannot escape via new sessions.
    saved={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        frozen=set()
        while True:
            children=_owned_descendants()
            new={(pid,value[1]) for pid,value in children.items()}-frozen
            if not new:break
            for pid,birth in new:_signal_same_process(pid,birth,signal.SIGSTOP)
            frozen.update(new)
        for pid,birth in frozen:_signal_same_process(pid,birth,signal.SIGKILL)
        proc.wait()
        # Lock stays held if a kernel-blocked descendant has not actually exited.
        while True:
            while True:
                try:
                    pid,_=os.waitpid(-1,os.WNOHANG)
                    if not pid:break
                except ChildProcessError:break
            remaining=_owned_descendants()
            if not remaining:break
            for pid,(_,birth,_) in remaining.items():_signal_same_process(pid,birth,signal.SIGKILL)
            time.sleep(0.05)
    finally:
        for sig,handler in saved.items():signal.signal(sig,handler)


def command(argv,cwd,env,log,timeout):
    global _UNCERTAIN
    with Path(log).open("x") as handle:
        proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=proc.wait(timeout=timeout)
            if _owned_descendants():
                _UNCERTAIN=True
                _cleanup_owned(proc)
                raise RuntimeError("Command exited leaving descendants; cleaned up, run rejected")
            if code<0:_UNCERTAIN=True
            return code
        except BaseException as exc:
            if isinstance(exc,(subprocess.TimeoutExpired,KeyboardInterrupt,InterruptedError)):
                _UNCERTAIN=True
            try:_cleanup_owned(proc)
            except BaseException:
                _UNCERTAIN=True
                raise
            raise


def verify_frozen(repo,expected):
    for name,wanted in expected.items():
        if sha(repo/name)!=wanted:raise RuntimeError("Official source hash mismatch: "+name)
    actual={str(p.relative_to(repo)) for name in ("harness","task","tests") for p in (repo/name).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",".pyo")}
    if actual!=set(expected):raise RuntimeError("Unexpected files in frozen official source")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-repo",required=True);p.add_argument("--hub-cache",required=True)
    p.add_argument("--run-root",required=True);p.add_argument("--backend",choices=["GCP","KAGGLE_T4"],required=True)
    p.add_argument("--timeout",type=int,default=900)
    p.add_argument("--confirm-cloud-gpu-run",action="store_true")
    a=p.parse_args()
    if not a.confirm_cloud_gpu_run:raise SystemExit("Explicit future cloud GPU authorization required")
    if a.timeout<=0:raise ValueError("Positive predeclared native timeout required")
    blocked=Path("/tmp/soup-gpu0.lock.blocked")
    if blocked.exists():raise SystemExit("GPU0 quarantined by /tmp/soup-gpu0.lock.blocked; never auto-clear")
    enable_subreaper()
    def interrupted(signum,frame):raise InterruptedError("Runner interrupted by signal "+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    # All six candidate notebooks serialize through this canonical host GPU0 lock.
    gpu_lock=open("/tmp/soup-gpu0.lock","a")
    try:fcntl.flock(gpu_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        gpu_lock.close();raise SystemExit("GPU0 is locked by another candidate; no work launched")
    blocked=Path("/tmp/soup-gpu0.lock.blocked")
    if blocked.exists():
        fcntl.flock(gpu_lock.fileno(),fcntl.LOCK_UN);gpu_lock.close()
        raise SystemExit("GPU0 quarantined by /tmp/soup-gpu0.lock.blocked; inspect resource and marker, never auto-clear")
    marker_token=uuid.uuid4().hex
    here=Path(__file__).resolve().parent;repo=Path(a.prepared_repo).resolve();cache=Path(a.hub_cache).resolve()
    root=Path(a.run_root).resolve();root.mkdir(parents=True,exist_ok=True)
    run=root/("rotary_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8])
    run.mkdir(exist_ok=False)
    summary=dict(status="PREPARING",backend=a.backend,score=None,admission="UNKNOWN",
                 official_kaggle_admission="NOT_ESTABLISHED",run=str(run),gpu_tests_run=False)
    write(run/"review_summary.json",summary)
    try:
        # Crash quarantine: only this exact runner may remove its own token after
        # establishing that no owned descendants remain. SIGKILL leaves it intact.
        with blocked.open("x") as marker:
            json.dump(dict(owner="rotary",pid=os.getpid(),token=marker_token,run=str(run),
                           reason="Active runner; cleanup not yet established"),marker)
        if os.environ.get("TRITON_INTERPRET"):raise RuntimeError("Unset TRITON_INTERPRET before GPU work")
        expected=read(here/"official_frozen_manifest.json");verify_frozen(repo,expected)
        observed=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        if observed!=COMMIT:raise RuntimeError("Wrong prepared official commit")
        if subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=all","--","harness","task","tests"],text=True):
            raise RuntimeError("Official harness/task/tests are not clean")
        cpus=sorted(os.sched_getaffinity(0))
        if len(cpus)<4:raise RuntimeError("Four available logical CPUs required")
        os.sched_setaffinity(0,cpus[:4])
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",CUDA_DEVICE_ORDER="PCI_BUS_ID",HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1",HF_HUB_CACHE=str(cache),HF_HOME=str(run/"empty_hf_home"),
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1",
                 TOKENIZERS_PARALLELISM="false",SOUP_NO_AUDIT_LOG="1")
        for key in ("SOUP_ROTARY_CODEGEN_DIR","PYTHONPATH","TRITON_INTERPRET","HF_TOKEN","HUGGING_FACE_HUB_TOKEN"):
            env.pop(key,None)
        # Checker executes in a separate process so the runner retains no CUDA context.
        rc=command([sys.executable,str(here/"check_kaggle_environment.py"),"--gpu","--require-four-cpus","--repo",str(repo),"--out",str(run/"environment.json")],here,env,run/"environment.log",120)
        if rc:raise RuntimeError("Target environment mismatch; see environment.json")
        assets=load_module(here/"prepare_asset_manifest.py","rotary_assets")
        model,data=assets.build(cache,repo);write(run/"public_model_manifest.json",model);write(run/"public_data_manifest.json",data)
        # Unmodified official tree in a fresh private run directory; input checkout remains untouched.
        work=run/"official";work.mkdir()
        archive=subprocess.check_output(["git","-C",str(repo),"archive",COMMIT])
        with tarfile.open(fileobj=io.BytesIO(archive),mode="r:") as tar:
            tar.extractall(work,filter="data")
        verify_frozen(work,expected)
        (work/"data").mkdir(exist_ok=True)
        for name in data:shutil.copyfile(repo/name,work/name)
        # Replace only the fresh archive's solution folder, never official harness/task/tests.
        shutil.rmtree(work/"solution")
        manifest=read(here/"source_manifest.json")
        with zipfile.ZipFile(here/"candidate.zip") as bundle:
            if set(bundle.namelist())!=set(manifest):raise RuntimeError("ZIP member set mismatch")
            for name,wanted in manifest.items():
                raw=bundle.read(name)
                if hashlib.sha256(raw).hexdigest()!=wanted or not name.startswith("solution/") or ".." in Path(name).parts:
                    raise RuntimeError("Bundle source mismatch")
                path=work/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        write(run/"candidate_source_manifest.json",manifest)
        summary.update(source_sha256=sha(here/"candidate.zip"),cpu_affinity=cpus[:4],native_timeout_s=a.timeout,repeats=2)
        # Separate process/cache/output preflight; its warm compiler state never enters training.
        preenv=dict(env,TRITON_CACHE_DIR=str(run/"preflight_triton"),TORCHINDUCTOR_CACHE_DIR=str(run/"preflight_inductor"))
        rc=command([sys.executable,"-m","solution.gpu_preflight","--out",str(run/"gpu_preflight")],work,preenv,run/"gpu_preflight.log",900)
        if rc:raise RuntimeError("GPU arithmetic/routing/PTX preflight failed")
        summary["gpu_tests_run"]=True
        # GPU preflight exited, so its context is gone. Stop rather than measuring
        # around an unrelated workload or a leaked child context.
        processes=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
        write(run/"post_preflight_idle.json",dict(compute_processes=processes.stdout.strip(),stderr=processes.stderr))
        if processes.stdout.strip():raise RuntimeError("GPU compute process remains after preflight; idle state unproved")
        # Official fixed task only. run_task itself generates fresh processes, state,
        # outputs and caches per side/repeat and externally samples nvidia-smi.
        out=run/"paired"
        env.update(TRITON_CACHE_DIR=str(run/"harness_triton"),TORCHINDUCTOR_CACHE_DIR=str(run/"harness_inductor"))
        rc=command([sys.executable,"-B","harness/run_task.py","--repeats","2","--seed","1","--timeout",str(a.timeout),"--out",str(out)],work,env,run/"paired.log",1800+6*a.timeout+300)
        summary["harness_exit"]=rc
        if rc or not (out/"result.json").is_file():raise RuntimeError("Harness failed or produced no result")
        result=read(out/"result.json")
        complete=True
        for side in ("base","solution"):
            rows=result.get("runs",{}).get(side,[])
            complete &= len(rows)==2
            for row in rows:
                m=row.get("metrics",{})
                complete &= row.get("exit_code")==0 and m.get("exit_code")==0 and m.get("steps")==64 and row.get("global_step")==64
                complete &= isinstance(row.get("gpu_peak_mib"),(int,float)) and row["gpu_peak_mib"]>0
                if side=="solution":
                    own=m.get("own_kernel_ms",{})
                    complete &= all(own.get(n,0)>0 for n in ("rotary_apply_f32","rotary_adjoint_f32"))
            ev=result.get("evals",{}).get(side,{})
            complete &= ev.get("exit_code")==0 and isinstance(ev.get("mean_loss"),(int,float)) and math.isfinite(ev["mean_loss"])
        summary["complete_fixed_task_and_kernel_activation"]=bool(complete)
        # Always preserve raw failures; score.py is deliberately never called.
        rc=command([sys.executable,"-B","harness/check_rules.py","--result",str(out/"result.json")],work,env,run/"official_checks.log",2100)
        checks=read(out/"checks.json") if (out/"checks.json").exists() else {}
        gates=checks.get("checks",[])
        all_pass=complete and rc==0 and len(gates)==10 and all(x.get("ok") is True for x in gates)
        verify_frozen(work,expected)
        summary.update(status="COMPLETE_CHECKS_PASS" if all_pass else "FAILED_OR_INADMISSIBLE",
                       official_checks_passed=bool(all_pass),checks_exit=rc,
                       measurements=result.get("summary") if complete else None,
                       result_sha256=sha(out/"result.json"),
                       caution="Local paired research only; no score or official admission assigned")
    except BaseException as exc:
        summary.update(status="BLOCKED_OR_FAILED",error=repr(exc),measurements=None,
                       outer_cancellation=isinstance(exc,(subprocess.TimeoutExpired,KeyboardInterrupt,InterruptedError)))
    finally:
        cleanup_confirmed=not _owned_descendants()
        summary["owned_descendant_cleanup_confirmed"]=cleanup_confirmed
        quarantine_required=_UNCERTAIN or summary.get("outer_cancellation",False) or not cleanup_confirmed
        summary["operator_reconciliation_required"]=bool(quarantine_required)
        if cleanup_confirmed and not quarantine_required and blocked.exists():
            try:
                marker=read(blocked)
                if marker.get("token")==marker_token:blocked.unlink()
            except (OSError,ValueError):pass  # Never erase an unrecognized quarantine.
        summary["quarantine_marker_remaining"]=blocked.exists()
        write(run/"review_summary.json",summary)
        fcntl.flock(gpu_lock.fileno(),fcntl.LOCK_UN);gpu_lock.close()
    print(json.dumps(summary,indent=2))
    if summary["status"]!="COMPLETE_CHECKS_PASS":raise SystemExit(1)

if __name__=="__main__":main()
