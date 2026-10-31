"""Self-contained prepared-runtime runner. No download/install/deployment/submission.

This file is embedded byte-for-byte in candidate.ipynb. Executing run() is a
future GPU operation and is not part of package preparation or CPU validation.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
import zipfile

COMMIT = "c7c2fcd15972ce8b898d8d3ff732ee192c817ceb"
CANDIDATE_ID = "soup-ce-v3-routing-20261030"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PINS = {"torch":"2.11.0+cu128","triton":"3.6.0","soup-cli":"0.77.0",
        "transformers":"5.16.1","peft":"0.20.0","trl":"0.29.1",
        "bitsandbytes":"0.50.2","accelerate":"1.14.0","datasets":"4.8.5"}
DATA_SHA256 = {"train.jsonl":"93d036d912bd81c337dc29893e13a76f36161d96f896ef4e02492e2f1a21a0dd",
               "eval_public.jsonl":"b7f85b78ba3186dce4deab7cfb8bf8dcf7998e1517c5749ac0f5005886c6f70e"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def shell(args, *, cwd=None, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def _proc_identity(pid):
    try:
        # comm can contain spaces: parse from the final ')' rather than split it.
        fields=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        return int(fields[1]),fields[19]  # parent PID, process start-time ticks
    except (OSError,ValueError,IndexError): return None


def _process_table():
    table={}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit(): continue
        try:
            fields=(path/"stat").read_text().rsplit(")",1)[1].split()
            table[int(path.name)]={"ppid":int(fields[1]),"start":fields[19],"state":fields[0]}
        except (OSError,ValueError,IndexError): pass
    return table


def terminate_owned_tree(proc, root_start, tracked=None):
    """Freeze first, then discover/freeze the tree to prevent the census/spawn race.

    Start-time identity guards PID reuse. This runs inside the notebook, not a
    global subreaper: it never adopts or kills unrelated notebook children.
    Quarantine always remains after interruption, even if this cleanup verifies.
    """
    import signal
    owned=dict(tracked or {});owned[proc.pid]=root_start;stopped=set();errors=[]
    def signal_owned(pid,start,sig):
        identity=_proc_identity(pid)
        if identity is None or identity[1]!=start: return
        try: os.kill(pid,sig)
        except ProcessLookupError: pass
        except OSError as error: errors.append(f"{pid}:{type(error).__name__}")
    # The root cannot create a child after this stop is delivered. Iteratively
    # stop existing descendants before taking the final complete census.
    signal_owned(proc.pid,root_start,signal.SIGSTOP)
    deadline=time.monotonic()+5
    stable=0
    while stable<2 and time.monotonic()<deadline:
        table=_process_table();added=False
        live_owned={pid for pid,start in owned.items() if pid in table and table[pid]["start"]==start}
        for pid,record in table.items():
            if record["ppid"] in live_owned and pid not in owned:
                owned[pid]=record["start"];added=True
        for pid,start in list(owned.items()):
            if pid not in stopped:
                signal_owned(pid,start,signal.SIGSTOP);stopped.add(pid)
        # Stopped roots preserve ancestry until every live branch is stopped.
        table=_process_table()
        unsettled=[pid for pid,start in owned.items() if pid in table
                   and table[pid]["start"]==start and table[pid]["state"] not in ("T","t","Z","X")]
        stable=stable+1 if not added and not unsettled else 0
        time.sleep(0.01)
    for pid,start in list(owned.items()):
        if pid!=proc.pid: signal_owned(pid,start,signal.SIGKILL)
    signal_owned(proc.pid,root_start,signal.SIGKILL)
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: errors.append("root_wait_timeout")
    deadline=time.monotonic()+5
    while True:
        table=_process_table()
        remaining=[pid for pid,start in owned.items() if pid in table
                   and table[pid]["start"]==start and table[pid]["state"] not in ("Z","X")]
        if not remaining or time.monotonic()>=deadline: break
        time.sleep(0.02)
    report={"known_owned_count":len(owned),"known_owned_pids":sorted(owned),
            "remaining_live_known_pids":remaining,"errors":errors,
            "frozen_census_stable":stable>=2,
            "verified_known_tree_terminated":not remaining and not errors and stable>=2,
            "quarantine_required":True,
            "limit":"No SIGKILL/power-loss or already-unobserved orphan guarantee; operator reconciliation remains required."}
    return report


def mark_quarantine(path, reason):
    """Immutable first incident; never auto-clear or overwrite another incident."""
    path=Path(path)
    payload={"schema":1,"status":"OPERATOR_RECONCILIATION_REQUIRED","reason":reason,
             "runner_pid":os.getpid(),"utc_epoch_s":time.time(),"automatic_clear":False}
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|getattr(os,"O_NOFOLLOW",0),0o600)
    except FileExistsError: return
    with os.fdopen(fd,"w") as file:
        file.write(json.dumps(payload,sort_keys=True)+"\n");file.flush();os.fsync(file.fileno())


def create_active_marker(path, owner_token):
    """Durable owner-tagged fail-closed guard, installed before the first GPU call."""
    payload={"schema":1,"status":"ACTIVE_OWNER_GUARD","owner_token":owner_token,
             "runner_pid":os.getpid(),"utc_epoch_s":time.time(),
             "recovery":"If not removed by normal verified owner completion, operator reconciliation is required."}
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|getattr(os,"O_NOFOLLOW",0),0o600)
    with os.fdopen(fd,"w") as file:
        file.write(json.dumps(payload,sort_keys=True)+"\n");file.flush();os.fsync(file.fileno())
    directory=os.open(str(Path(path).parent),os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
    try: os.fsync(directory)
    finally: os.close(directory)


def clear_owned_marker(path, owner_token):
    """Only the same process/token may remove its marker on verified success."""
    path=Path(path)
    try:
        fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
        with os.fdopen(fd,"r") as file:
            identity=os.fstat(file.fileno())
            value=json.loads(file.read(4096))
        current=path.lstat()
    except (OSError,ValueError,TypeError): return False
    if value.get("owner_token")!=owner_token or value.get("runner_pid")!=os.getpid(): return False
    if (identity.st_dev,identity.st_ino)!=(current.st_dev,current.st_ino): return False
    path.unlink()
    directory=os.open(str(path.parent),os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
    try: os.fsync(directory)
    finally: os.close(directory)
    return True


def observe_owned(owned):
    table=_process_table();changed=True
    while changed:
        changed=False
        live={pid for pid,start in owned.items() if pid in table and table[pid]["start"]==start}
        for pid,value in table.items():
            if value["ppid"] in live and pid not in owned:
                owned[pid]=value["start"];changed=True
    return [pid for pid,start in owned.items() if pid in table and table[pid]["start"]==start
            and table[pid]["state"] not in ("Z","X")]


@contextmanager
def interruption_handlers():
    import signal
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"runner interrupted by signal {signum}")
    old={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for sig in old: signal.signal(sig,interrupted)
        yield
    finally:
        for sig,handler in old.items(): signal.signal(sig,handler)


def execute(command, *, cwd, env, logfile, timeout,
            quarantine_path="/tmp/soup-gpu0.lock.blocked"):
    started=time.monotonic()
    with interruption_handlers(), Path(logfile).open("w") as log:
        import signal
        # Defer parent interrupts until proc+identity are registered. A tiny exec
        # launcher unblocks them in the child; no unsafe threaded preexec_fn.
        original_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
        proc=None;root_start=None;tracked={}
        launcher="import os,signal,sys; signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGINT,signal.SIGTERM}); os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
        try:
            proc=subprocess.Popen([sys.executable,"-c",launcher,*command],cwd=cwd,env=env,stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
            identity=_proc_identity(proc.pid)
            root_start=identity[1] if identity else None
            tracked[proc.pid]=root_start
            signal.pthread_sigmask(signal.SIG_SETMASK,original_mask)
            deadline=time.monotonic()+timeout
            while True:
                live=observe_owned(tracked)
                code=proc.poll()
                if code is not None:
                    survivors=[pid for pid in observe_owned(tracked) if pid!=proc.pid]
                    if survivors: raise RuntimeError("Owned descendants survived normal command exit")
                    save(str(logfile)+".processes.json",{"normal_exit":True,"exit_code":code,
                         "known_owned_count":len(tracked),"remaining_live_known_pids":[]})
                    break
                if time.monotonic()>=deadline: raise subprocess.TimeoutExpired(command,timeout)
                time.sleep(min(0.1,max(0.001,deadline-time.monotonic())))
        except BaseException as error:
            import signal
            reason="outer_timeout" if isinstance(error,subprocess.TimeoutExpired) else type(error).__name__
            if proc is None:
                signal.pthread_sigmask(signal.SIG_SETMASK,original_mask)
                raise
            # A durable shared marker precedes cleanup and remains even when
            # cleanup succeeds. No other notebook may begin pending reconciliation.
            mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
            try:
                marker_error=None
                try: mark_quarantine(quarantine_path,reason)
                except OSError as failure: marker_error=f"{type(failure).__name__}: {failure}"
                cleanup=terminate_owned_tree(proc,root_start,tracked)
                cleanup["quarantine_write_error"]=marker_error
                save(str(logfile)+".cleanup.json",cleanup)
                log.write("\nOUTER_INTERRUPTION "+json.dumps({"reason":reason,"cleanup":cleanup})+"\n");log.flush()
                if not cleanup["verified_known_tree_terminated"] or marker_error:
                    raise RuntimeError("Cleanup or quarantine persistence could not be confirmed; operator reconciliation required") from error
                if isinstance(error,subprocess.TimeoutExpired): code=-9
                else: raise
            finally: signal.pthread_sigmask(signal.SIG_SETMASK,original_mask)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,original_mask)
    return code,time.monotonic()-started


def validate_zip(raw, manifest):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest):
            raise ValueError("ZIP member set differs from reviewed source manifest")
        for name in names:
            p = Path(name)
            if not name.startswith("solution/") or p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe ZIP member")
            if hashlib.sha256(archive.read(name)).hexdigest() != manifest[name]:
                raise ValueError("ZIP member hash mismatch: " + name)
    return True


def parse_routes(log):
    records=[]; partial=0
    if not Path(log).exists(): return {"status":"UNKNOWN", "records":[], "partial_lines":0}
    for line in Path(log).read_text(errors="replace").splitlines():
        prefix="SOUP_CE_V3_EVENT "
        if prefix not in line: continue
        try: record=json.JSONDecoder().raw_decode(line.split(prefix,1)[1])[0]
        except (ValueError, TypeError): partial+=1; continue
        if isinstance(record,dict): records.append(record)
    return {"status":"OBSERVED_ROUTE_RECORDS" if records else "UNKNOWN",
            "records":records, "partial_lines":partial,
            "last_record":records[-1] if records else None,
            "interpretation":"Route and enqueue metadata only. Not GPU timing/share or completion."}


def public_model_manifest(cache):
    """Same pinned cache identity/boundary as the prepared queue asset contract."""
    root=Path(cache)/"models--Qwen--Qwen2.5-1.5B-Instruct"
    for directory in (root,root/"refs",root/"snapshots",root/"blobs"):
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError("Selected public cache directories must exist and be real")
    ref=root/"refs/main"
    if ref.is_symlink() or not ref.is_file() or ref.read_text().strip()!=MODEL_REVISION:
        raise RuntimeError("Prepared refs/main must identify the exact pinned model revision")
    snapshot=root/"snapshots"/MODEL_REVISION
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError("Pinned snapshot missing; no download or link reconstruction attempted")
    files={str(ref.relative_to(cache)):sha(ref)};memo={}
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            if path.is_symlink(): raise RuntimeError("Linked snapshot directory rejected")
            continue
        resolved=path.resolve(strict=True)
        if not resolved.is_relative_to((root/"blobs").resolve()) or not resolved.is_file():
            raise RuntimeError("Snapshot file must resolve only inside selected public model blobs")
        if resolved not in memo: memo[resolved]=sha(resolved)
        files[str(path.relative_to(cache))]=memo[resolved]
    if len(files)<2 or not (snapshot/"config.json").is_file(): raise RuntimeError("Prepared model snapshot incomplete")
    return {"model_id":"Qwen/Qwen2.5-1.5B-Instruct","revision":MODEL_REVISION,"files":files}


def completed_result(result):
    """Never infer completion from wrapper exit, last progress, or nested step 64 alone."""
    reasons=[]
    runs=result.get("runs",{})
    task=result.get("task",{})
    if task.get("repeats")!=2 or task.get("seed")!=1 or task.get("only") is not None:
        reasons.append("Expected fixed repeat2/seed1 paired task")
    for side in ("base","solution"):
        entries=runs.get(side,[])
        if len(entries)!=2: reasons.append(side+": expected exactly two native runs")
        for entry in entries:
            metrics=entry.get("metrics",{})
            if entry.get("exit_code") != 0 or metrics.get("exit_code") != 0:
                reasons.append(side+": native process incomplete/failed")
            if entry.get("global_step") != 64 or metrics.get("steps") != 64:
                reasons.append(side+": not all fixed 64 steps verified")
            if not entry.get("adapter_config") or not entry.get("adapter_tensors"):
                reasons.append(side+": adapter artifact missing")
            for key in ("wall_s","gpu_peak_mib"):
                val=entry.get(key)
                if isinstance(val,bool) or not isinstance(val,(int,float)) or not math.isfinite(val) or val <= 0:
                    reasons.append(side+": missing complete "+key)
        evaluation=result.get("evals",{}).get(side,{})
        loss=evaluation.get("mean_loss")
        if evaluation.get("exit_code") != 0 or isinstance(loss,bool) or not isinstance(loss,(int,float)) or not math.isfinite(loss):
            reasons.append(side+": public quality missing/failed")
    if result.get("packages_changed") != []: reasons.append("packages changed or not inventoried")
    return not reasons,reasons


def _run_locked(config, bundle_bytes, source_manifest, official_manifest):
    if config.get("execute_gpu") is not True:
        raise RuntimeError("Preparation only. Set execute_gpu=True only for an authorized cloud T4 run.")
    if config.get("backend") not in ("GCP","KAGGLE"):
        raise ValueError("backend must explicitly be GCP or KAGGLE; GCP is never official Kaggle admission")
    repo=Path(config["official_repo"]).resolve(); data=Path(config["data_dir"]).resolve()
    cache=Path(config["hf_hub_cache"]).resolve(); work=Path(config["work_root"]).resolve()
    for path in (repo,data,cache):
        if not path.is_dir(): raise FileNotFoundError(f"Prepared asset missing: {path}; no download/bootstrap attempted")
    # Existing packages only. Never mutate them or silently select a different wheel.
    versions={name:importlib.metadata.version(name) for name in PINS}
    if sys.version_info[:3] != (3,13,15) or versions != PINS:
        raise RuntimeError("Exact prepared Python3.13.15/direct package pins required: "+repr(versions))
    importlib.metadata.version("pytest")  # required by unchanged official regression runner
    if shell(["git","-C",str(repo),"rev-parse","HEAD"]) != COMMIT:
        raise RuntimeError("Official starter commit mismatch")
    status=shell(["git","-C",str(repo),"status","--porcelain","--untracked-files=all","--","harness","task","tests"])
    if status: raise RuntimeError("Official harness/task/tests must be unchanged: "+status)
    for relative, digest in official_manifest.items():
        if sha(repo/relative) != digest: raise RuntimeError("Official source hash mismatch: "+relative)
    for name,digest in DATA_SHA256.items():
        if sha(data/name) != digest: raise RuntimeError("Prepared public data hash mismatch: "+name)
    validate_zip(bundle_bytes,source_manifest)
    # No torch imported into this notebook process; GPU visibility is fixed before
    # all actual-class, preflight and measured subprocess imports.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",CUDA_DEVICE_ORDER="PCI_BUS_ID",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",PYTHONUNBUFFERED="1",
             TOKENIZERS_PARALLELISM="false",SOUP_NO_AUDIT_LOG="1",HF_HUB_OFFLINE="1",
             HF_DATASETS_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_CACHE=str(cache),
             HF_HOME=str(cache.parent))
    runroot=work/(CANDIDATE_ID+"-"+uuid.uuid4().hex)
    runroot.mkdir(parents=True,exist_ok=False)
    report={"candidate_id":CANDIDATE_ID,"status":"PREPARING","backend":config["backend"],
            "official_commit":COMMIT,"measured_baseline_id":None,"score":None,
            "automatic_admission":"UNKNOWN","runroot":str(runroot),
            "source_sha256":hashlib.sha256(bundle_bytes).hexdigest(),
            "environment":{"python":sys.version,"packages":versions},
            "limit":"Research pair only; official Kaggle admission and jury review remain separate."}
    save(runroot/"receipt.json",report)
    try:
        gpu_code="""import json,torch
assert torch.version.cuda=='12.8'
assert torch.cuda.device_count()==1
assert torch.cuda.get_device_name(0)=='Tesla T4'
assert torch.cuda.get_device_capability(0)==(7,5)
print(json.dumps({'visible_count':torch.cuda.device_count(),'device':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}))
"""
        report["gpu"]=json.loads(shell([sys.executable,"-c",gpu_code],env=env))
        inventory=shell(["nvidia-smi","--query-gpu=index,name,driver_version,memory.used,memory.total","--format=csv,noheader,nounits"],env=env)
        report["nvidia_smi_inventory"]=inventory
        active=shell(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],env=env)
        if any(line.strip().isdigit() for line in active.splitlines()):
            raise RuntimeError("A compute process is still using a physical GPU, possibly this notebook; no measurement starts")
        total_used=sum(int(line.split(",")[-2].strip()) for line in inventory.splitlines())
        if total_used >= 1000: raise RuntimeError("GPU(s) are not idle; never stop unrelated workloads")
        report["cpu_affinity"]=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None
        # The queue-prepared public model revision and refs/main are fixed;
        # actual byte hashing is additional binding, not permission to use drift.
        cache_manifest=public_model_manifest(cache)
        cache_code="from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_files_only=True, token=False))"
        resolved_snapshot=Path(shell([sys.executable,"-c",cache_code],env=env)).resolve()
        expected_snapshot=(cache/"models--Qwen--Qwen2.5-1.5B-Instruct"/"snapshots"/MODEL_REVISION).resolve()
        if resolved_snapshot!=expected_snapshot: raise RuntimeError("Offline main resolved a different model snapshot")
        save(runroot/"model_cache_manifest.json",cache_manifest)
        report["model_cache_manifest_sha256"]=sha(runroot/"model_cache_manifest.json")
        report["data_sha256"]=DATA_SHA256
        # Fresh clean source tree; original checkout, prior solutions and outputs
        # are preserved. Only solution/ is replaced in this private staging copy.
        stage=runroot/"source"; stage.mkdir()
        raw=subprocess.check_output(["git","-C",str(repo),"archive",COMMIT])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for member in archive.getmembers():
                p=Path(member.name)
                if p.is_absolute() or ".." in p.parts or not (member.isfile() or member.isdir()):
                    raise ValueError("Unsafe official archive entry")
            archive.extractall(stage,filter="data")
        shutil.rmtree(stage/"solution")
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive: archive.extractall(stage)
        for name,digest in source_manifest.items():
            if sha(stage/name) != digest: raise RuntimeError("Staged candidate differs: "+name)
        for name,digest in official_manifest.items():
            if sha(stage/name) != digest: raise RuntimeError("Staged official file differs: "+name)
        (stage/"data").mkdir(exist_ok=True)
        for name in DATA_SHA256: shutil.copyfile(data/name,stage/"data"/name)
        save(runroot/"source_manifest.json",source_manifest)
        save(runroot/"official_manifest.json",official_manifest)
        (runroot/"candidate.zip").write_bytes(bundle_bytes)
        # Fresh process/cache for a new v3 actual-class gate; previous v2 results
        # are historical and cannot satisfy this gate. No training modifications.
        preflight_env=dict(env,TRITON_CACHE_DIR=str(runroot/"preflight_triton_cache"),
                           TORCHINDUCTOR_CACHE_DIR=str(runroot/"preflight_inductor_cache"))
        preflight_cmd=[sys.executable,"-m","solution.verify_integration","--out",str(runroot/"integration_preflight.json")]
        preflight_code,preflight_wall=execute(preflight_cmd,cwd=stage,env=preflight_env,
            logfile=runroot/"integration_preflight.log",timeout=300)
        report["preflight_wall_s"]=preflight_wall
        report["preflight_exit_code"]=preflight_code
        if preflight_code != 0: raise RuntimeError("v3 actual-class gate failed or timed out; no training follows")
        preflight=json.loads((runroot/"integration_preflight.json").read_text())
        if preflight.get("status") != "passed_gpu_integration_preflight": raise RuntimeError("v3 gate status missing")
        # Unchanged official harness: own same-session baseline, candidate, full
        # subprocess wall, external nvidia-smi, fresh state/cache, public eval,
        # and CUDA-event own kernel timing after exactly two warmup steps.
        # 900s is the official default, not the censored Kaggle79s screen budget.
        report["native_timeout_s"]=900
        report["measured_order"]=["base","solution","solution","base"]
        cmd=[sys.executable,"harness/run_task.py","--repeats","2","--seed","1",
             "--timeout","900","--out",str(runroot/"runs")]
        harness_code,wrapper_wall=execute(cmd,cwd=stage,env=env,
            logfile=runroot/"harness.log",timeout=9000)
        report["harness_exit_code"]=harness_code
        report["complete_wrapper_wall_s"]=wrapper_wall
        report["outer_watchdog_s"]=9000
        result_path=runroot/"runs"/"result.json"
        if not result_path.exists(): raise RuntimeError("Official result absent; completed metrics unknown")
        result=json.loads(result_path.read_text())
        report["routes"]={p.parent.name:parse_routes(p) for p in (runroot/"runs").glob("solution_*/train.log")}
        complete,reasons=completed_result(result)
        report["native_pair_complete"]=complete
        report["completion_blockers"]=reasons
        report["raw_result_sha256"]=sha(result_path)
        report["comparison"]=result.get("summary") if complete and harness_code == 0 else None
        if not complete or harness_code != 0:
            report["status"]="FAILED_OR_CENSORED_NO_SCORE"
            return report
        checks=[sys.executable,"harness/check_rules.py","--result",str(result_path)]
        checks_code,checks_wall=execute(checks,cwd=stage,env=env,
            logfile=runroot/"checks.log",timeout=900)
        report["official_checks_exit_code"]=checks_code
        report["official_checks_wall_s"]=checks_wall
        report["official_checks_path"]=str(runroot/"runs"/"checks.json")
        report["status"]="RESEARCH_PAIR_COMPLETE_REVIEW_REQUIRED"
        # All ten raw checks may pass, but no automatic official admission or
        # leaderboard score is manufactured from this public research run.
        return report
    except BaseException as error:
        report.update(status="BLOCKED_OR_FAILED_NO_SCORE",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # Original frozen files must remain unchanged even after failed runs.
        if (runroot/"source").exists():
            report["official_files_preserved"]=all((runroot/"source"/p).exists() and sha(runroot/"source"/p)==h for p,h in official_manifest.items())
            report["candidate_source_preserved"]=all((runroot/"source"/p).exists() and sha(runroot/"source"/p)==h for p,h in source_manifest.items())
            report["routes"]={p.parent.name:parse_routes(p) for p in (runroot/"runs").glob("solution_*/train.log")}
        if report.get("official_files_preserved") is False or report.get("candidate_source_preserved") is False:
            report.update(status="SOURCE_MUTATED_NO_SCORE",comparison=None)
        save(runroot/"receipt.json",report)
        print(json.dumps({"status":report["status"],"receipt":str(runroot/"receipt.json"),"score":None,"admission":"UNKNOWN"},indent=2))


def run(config, bundle_bytes, source_manifest, official_manifest):
    """One canonical nonblocking GPU0 lock covers preparation, gates and pair."""
    if config.get("execute_gpu") is not True:
        raise RuntimeError("Preparation only. Set execute_gpu=True only for an authorized cloud T4 run.")
    import fcntl
    flags=os.O_CREAT | os.O_RDWR | getattr(os,"O_NOFOLLOW",0)
    quarantine=Path("/tmp/soup-gpu0.lock.blocked")
    if quarantine.exists() or quarantine.is_symlink():
        raise RuntimeError("GPU0 is quarantined; operator reconciliation is required before any GPU work")
    fd=os.open("/tmp/soup-gpu0.lock",flags,0o600)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("GPU0 lane is owned by another process; no GPU work started") from error
        if quarantine.exists() or quarantine.is_symlink():
            raise RuntimeError("GPU0 quarantine appeared while acquiring lock; no GPU work started")
        owner_token=uuid.uuid4().hex
        create_active_marker(quarantine,owner_token)
        with interruption_handlers():
            try:
                report=_run_locked(config,bundle_bytes,source_manifest,official_manifest)
                # Failures retain the active guard. Only a clean complete pair,
                # successful official checks, unchanged source and idle GPU can
                # remove THIS process/token's marker. SIGKILL cannot remove it.
                normal=(report.get("status")=="RESEARCH_PAIR_COMPLETE_REVIEW_REQUIRED"
                        and report.get("official_checks_exit_code")==0
                        and report.get("official_files_preserved") is True
                        and report.get("candidate_source_preserved") is True)
                if normal:
                    active=shell(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"])
                    if any(line.strip().isdigit() for line in active.splitlines()):
                        raise RuntimeError("GPU compute process survived normal finish; active owner guard retained")
                    if not clear_owned_marker(quarantine,owner_token):
                        raise RuntimeError("Owner marker changed or could not be cleared; no foreign marker removed")
                    report["lane_guard"]="OWN_MARKER_CLEARED_AFTER_NORMAL_VERIFIED_SUCCESS"
                else: report["lane_guard"]="RETAINED_OPERATOR_RECONCILIATION_REQUIRED"
                if report.get("runroot"): save(Path(report["runroot"])/"receipt.json",report)
                return report
            except KeyboardInterrupt:
                mark_quarantine(quarantine,"outer_runner_interruption")
                raise
    finally:
        # The persistent blocked marker prevents reuse even after a lock release.
        # Never unlink the lock inode; a failed/interrupted owner guard is not cleared here.
        fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    run(json.loads(Path(args.config).read_text()),(root/"candidate.zip").read_bytes(),
        json.loads((root/"source_manifest.json").read_text()),
        json.loads((root/"official_manifest.json").read_text()))
