#!/usr/bin/env python3
"""Trusted reviewed-bundle worker. Fixed commands, no submission-controlled shell.

GPU execution requires a target-validated Docker boundary; CPU smoke is trusted local code only.
No automatic retry. All automatic admission/score fields stay UNKNOWN/null.
"""
import argparse, contextlib, fcntl, hashlib, io, json, math, os, re, shutil, signal, socket, stat, subprocess, sys, time, uuid, zipfile
from pathlib import Path
from soup_queue import Queue, QueueError, canonical, digest, require, sha, safe_relative, validate_artifact
from prepare_asset_manifest import DATA as PUBLIC_DATA, REVISION as MODEL_REVISION

ROOT=Path(__file__).resolve().parent
COMMIT='c7c2fcd15972ce8b898d8d3ff732ee192c817ceb'
MODEL='Qwen/Qwen2.5-1.5B-Instruct'
MODEL_DIR='models--Qwen--Qwen2.5-1.5B-Instruct'
GPU_LOCK_PATH=Path('/tmp/soup-queue-gpu0.lock')
IDENTITY=('candidate_id','source_sha256','backend','baseline_id','environment_fingerprint','task_fingerprint','harness_commit')

def read_json(p): return json.loads(Path(p).read_text())
def write_json(p,value):
    p=Path(p); data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with p.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    return digest(data)
def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def checked_file(p,expected):
    p=Path(p); require(sha(expected) and p.is_file() and not p.is_symlink(),'Missing/linked evidence file')
    require(file_hash(p)==expected,'Evidence hash mismatch: '+p.name); return p

@contextlib.contextmanager
def output_file(root, relative, *, metadata_only=False):
    """Open regular output safely; metadata probes do not require read permission."""
    parts=safe_relative(relative).parts;fds=[]
    try:
        fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);fds.append(fd)
        for part in parts[:-1]:
            fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);fds.append(fd)
        flags=os.O_PATH|os.O_NOFOLLOW if metadata_only else os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK
        fd=os.open(parts[-1],flags,dir_fd=fd);fds.append(fd)
        require(stat.S_ISREG(os.fstat(fd).st_mode),'Container output is not a regular file: '+relative)
        yield fd
    finally:
        for fd in reversed(fds):os.close(fd)

def output_bytes(root,relative,limit=16*1024*1024,optional=False):
    try:
        with output_file(root,relative) as fd:
            require(os.fstat(fd).st_size<=limit,'Container output exceeds read bound: '+relative)
            chunks=[];remaining=limit+1
            while remaining:
                chunk=os.read(fd,min(1024*1024,remaining))
                if not chunk:break
                chunks.append(chunk);remaining-=len(chunk)
            value=b''.join(chunks);require(len(value)<=limit,'Container output grew past read bound: '+relative)
            return value
    except FileNotFoundError:
        if optional:return None
        raise

def output_present(root,relative):
    try:
        with output_file(root,relative,metadata_only=True) as fd:return os.fstat(fd).st_size>0
    except FileNotFoundError:return False

def boundary_binding(c):
    return {'docker_binary':str(Path(c.get('docker_binary','/usr/bin/docker')).resolve()),'container_uid':c.get('container_uid',10001),'container_gid':c.get('container_gid',10001),'cpu_affinity':c['cpu_affinity'],'policy_version':2}

def boundary_helpers():
    return {n:file_hash(ROOT/n) for n in ('runner_adapter.py','runner_child.py','check_kaggle_environment.py','container_selftest.py','prepare_asset_manifest.py')}

def checked_config(value):
    c=read_json(value) if isinstance(value,(str,Path)) else dict(value)
    required={'queue_url','workspace_root','python','backend','baseline_id','resource','environment_fingerprint','task_fingerprint','harness_commit','baseline_fingerprint','lease_seconds','profiles','cpu_affinity'}
    require(required<=set(c),'Missing trusted worker configuration')
    require(c['backend'] in ('CPU','GCP','KAGGLE_T4'),'Unsupported backend')
    require(c['backend']!='CPU' or c.get('trusted_cpu_smoke_ack') is True,'CPU smoke is local trusted code only; explicit acknowledgement required')
    require(c['harness_commit']==COMMIT,'Only pinned official starter supported')
    require(all(sha(c[k]) for k in ('environment_fingerprint','task_fingerprint','baseline_fingerprint')),'Cohort fingerprints must be exact hashes')
    require(Path(c['python']).is_absolute() and Path(c['python']).is_file(),'Explicit installed interpreter required')
    require(10<=c['lease_seconds']<=3600,'Lease outside bound')
    require(type(c['cpu_affinity']) is list and c['cpu_affinity'] and all(type(x) is int and x>=0 for x in c['cpu_affinity']),'Explicit CPU affinity required')
    require(len(set(c['cpu_affinity']))==len(c['cpu_affinity']),'Duplicate affinity')
    require(set(c['cpu_affinity'])<=os.sched_getaffinity(0),'Requested CPUs unavailable')
    require(c['backend']=='CPU' or len(c['cpu_affinity'])==4,'GPU cohort requires four logical CPUs')
    require(type(c['profiles']) is dict,'Runner-local profile allowlist required')
    for h,p in c['profiles'].items():
        require(sha(h) and type(p) is dict and p.get('kind') in ('cpu_smoke_v1','soup_screen_v1'),'Only reviewed CPU smoke or single native Soup screen profiles')
        require(type(p.get('files')) is dict and p['files'],'Exact reviewed files required')
        for name,hsh in p['files'].items():safe_relative(name); require(sha(hsh),'Invalid reviewed file hash')
        require(type(p.get('timeout_s')) is int and p['timeout_s']>0,'Positive explicit timeout required')
        if p['kind']=='cpu_smoke_v1':
            require(c['backend']=='CPU' and re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*',p.get('module','')),'Invalid reviewed CPU module')
            module_path=p['module'].replace('.','/')
            require(module_path+'.py' in p['files'] or module_path+'/__init__.py' in p['files'],'Smoke module must be in reviewed source manifest')
        else:
            require(c['backend']!='CPU' and p.get('expected_steps')==64,'Full official 64-step task required')
            require(type(p.get('outer_timeout_s')) is int and p['outer_timeout_s']>p['timeout_s'],'Explicit separate outer watchdog required')
            require(p['timeout_s']==c.get('baseline_timeout_s') and p['outer_timeout_s']==c.get('baseline_outer_timeout_s'),'Candidate and baseline timeout budgets must match')
            require(all(n.startswith('solution/') for n in p['files']),'GPU candidate may replace only solution/')
            require('solution/__init__.py' in p['files'],'Candidate entrypoint required')
            for n,hsh in p.get('support_files',{}).items():safe_relative(n); require(n.startswith('solution/') and sha(hsh) and n not in p['files'],'Invalid pristine support allowlist')
    return c

def child_env(run,c):
    # Do not inherit worker bearer token, cloud credentials, PYTHONPATH, HOME, HF_TOKEN or proxy settings.
    home=run/'home'; tmp=run/'tmp'; home.mkdir(); tmp.mkdir()
    env={'PATH':str(Path(c['python']).parent)+':/usr/bin:/bin','HOME':str(home),'TMPDIR':str(tmp),'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':str(len(c['cpu_affinity'])),'MKL_NUM_THREADS':str(len(c['cpu_affinity'])),'OPENBLAS_NUM_THREADS':str(len(c['cpu_affinity'])),'HF_HOME':str(home/'hf'),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
    if c['backend']!='CPU':
        env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',SOUP_NO_AUDIT_LOG='1',HF_HUB_CACHE=str(Path(c['hf_hub_cache']).resolve()))
    return env

def process_table():
    """Linux owned-process identities; PID start ticks prevent PID-reuse kills."""
    table={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            if path.stat().st_uid!=os.getuid():continue
            text=path.read_text(); fields=text[text.rfind(')')+2:].split()
            table[int(path.parent.name)]=(int(fields[1]),int(fields[19]),fields[0])
        except (OSError,ValueError,IndexError):pass
    return table

def supervised_process(argv,cwd,env,timeout_s,log_path,heartbeat=None,poll_s=0.2):
    """Fresh process, durable file stdout, bounded owned-descendant cleanup. No shell."""
    require(type(argv) is list and all(type(x) is str for x in argv),'argv list only')
    start=time.monotonic(); tracked={}; timed_out=False; heartbeat_error=None; last_heartbeat=0
    with Path(log_path).open('xb') as log:
        proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        while True:
            table=process_table(); pending={proc.pid}|{pid for pid,ticks in tracked.items() if pid in table and table[pid][1]==ticks}
            changed=True
            while changed:
                old=len(pending); pending|={pid for pid,v in table.items() if v[0] in pending}; changed=len(pending)!=old
            for pid in pending:
                if pid in table: tracked.setdefault(pid,table[pid][1])
            rc=proc.poll(); elapsed=time.monotonic()-start
            if rc is not None:break
            if elapsed>=timeout_s:timed_out=True; break
            if heartbeat is not None and elapsed-last_heartbeat>=10:
                try:heartbeat(); last_heartbeat=elapsed
                except Exception as exc:
                    heartbeat_error=type(exc).__name__; break
            time.sleep(poll_s)
        # Include remembered descendants even if native harness used new sessions.
        table=process_table(); killed=[]
        for pid,ticks in reversed(list(tracked.items())):
            if pid in table and table[pid][1]==ticks and table[pid][2]!='Z':
                try:os.kill(pid,signal.SIGKILL); killed.append(pid)
                except ProcessLookupError:pass
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:pass
        log.flush(); os.fsync(log.fileno())
    remaining=process_table(); survivors=[pid for pid,ticks in tracked.items() if pid in remaining and remaining[pid][1]==ticks and remaining[pid][2]!='Z']
    return {'argv':argv,'exit_code':proc.returncode,'elapsed_s':time.monotonic()-start,'timed_out':timed_out,'heartbeat_error':heartbeat_error,'owned_descendants_killed':killed,'surviving_owned_pids':survivors,'stdout_sha256':file_hash(log_path)}

def stage_bundle(raw,destination,files):
    validate_artifact(raw,'solution_bundle')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        actual={e.filename:digest(z.read(e)) for e in z.infolist() if not e.is_dir()}
        require(actual==files,'Exact bundle file allowlist/hash mismatch')
        for name in sorted(actual):
            path=destination/str(safe_relative(name)); path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(z.read(name))
    return actual

def git_output(repo,*args):
    p=subprocess.run(['git','-C',str(repo),*args],capture_output=True,check=False,timeout=30)
    require(p.returncode==0,'Pinned repository inspection failed'); return p.stdout

def stage_official(c,destination):
    require(c.get('data_files')==PUBLIC_DATA,'Exact pinned public train/evaluation hashes required')
    repo=Path(c['repo']).resolve(); require(git_output(repo,'rev-parse','HEAD').decode().strip()==COMMIT,'Official commit mismatch')
    tree=git_output(repo,'ls-tree','-rz','--full-tree',COMMIT); manifest={}
    for entry in tree.split(b'\0'):
        if not entry:continue
        metadata,name=entry.split(b'\t',1); mode,kind,oid=metadata.split(); name=name.decode(); safe_relative(name)
        require(kind==b'blob' and mode in (b'100644',b'100755'),'Unsupported tracked object')
        data=git_output(repo,'show',COMMIT+':'+name); actual=repo/name
        require(actual.is_file() and not actual.is_symlink() and actual.read_bytes()==data,'Tracked official source differs: '+name)
        manifest[name]=digest(data)
        path=destination/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    require(set(c['data_files'])=={'data/train.jsonl','data/eval_public.jsonl'},'Exact public task data paths required')
    for name,h in c['data_files'].items():
        path=checked_file(repo/name,h); target=destination/name; target.parent.mkdir(exist_ok=True); shutil.copyfile(path,target); manifest[name]=h
    return manifest

def verify_cache(c):
    manifest=read_json(checked_file(c['model_cache_manifest'],c['model_cache_manifest_sha256']))
    require(manifest.get('model_id')==MODEL and manifest.get('revision')==MODEL_REVISION,'Exact fixed model/revision manifest required')
    root=Path(c['hf_hub_cache']).resolve(); revision=manifest['revision']; snapshot=root/MODEL_DIR/'snapshots'/revision
    ref=MODEL_DIR+'/refs/main'; require(ref in manifest['files'],'Warm cache main ref must be bound')
    require((root/ref).read_text().strip()==revision,'Cache main points to a different revision')
    actual={str(p.relative_to(root)) for p in snapshot.rglob('*') if p.is_file()}
    expected={n for n in manifest['files'] if n.startswith(MODEL_DIR+'/snapshots/'+revision+'/')}
    require(actual==expected and actual,'Snapshot file set differs from cohort')
    require(set(manifest['files'])==expected|{ref},'Only this public snapshot and refs/main may enter the mounted cache view')
    for name,h in manifest['files'].items():
        safe_relative(name); path=root/name; require(path.resolve().is_relative_to(root),'Cache link escapes cache root')
        require(path.is_file() and file_hash(path)==h,'Cached model bytes mismatch: '+name)
    return manifest

def host_fingerprint():
    ident=Path('/etc/machine-id').read_text().strip()
    return digest(canonical({'machine_id_sha256':digest(ident.encode()),'hostname':socket.gethostname()}).encode())

def finite_eval(e):
    return e.get('exit_code')==0 and type(e.get('mean_loss')) in (int,float) and math.isfinite(e['mean_loss']) and e['mean_loss']>=0

@contextlib.contextmanager
def host_gpu_lock():
    """One fixed host lock for baseline AND candidates, independent of queue resource strings."""
    blocked=Path(str(GPU_LOCK_PATH)+'.blocked')
    fd=os.open(GPU_LOCK_PATH,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_uid==os.getuid(),'Host GPU lock is not owned by worker')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise QueueError('Another owned baseline/candidate holds the host GPU lock')
        require(not blocked.exists(),'Prior container cleanup is uncertain; manual reconciliation required')
        yield
    finally:os.close(fd)

def docker_spec(c):
    require(c.get('execution_boundary')=='docker_v1','GPU profile requires the Docker boundary, not host execution')
    binary=Path(c.get('docker_binary','/usr/bin/docker'))
    require(binary.is_absolute() and binary.is_file() and os.access(binary,os.X_OK),'Docker unavailable; GPU execution disabled, no host fallback')
    image=c.get('container_image','');require(re.fullmatch(r'(?:[A-Za-z0-9./:_-]+@)?sha256:[0-9a-f]{64}',image),'Exact local image sha256 ID or repository@sha256 digest required')
    require(c.get('container_python','/usr/local/bin/python')=='/usr/local/bin/python','Fixed image interpreter path is /usr/local/bin/python')
    require(type(c.get('container_uid',10001)) is int and c.get('container_uid',10001)>0 and type(c.get('container_gid',10001)) is int and c.get('container_gid',10001)>0,'Nonroot container identity required')
    require(c.get('container_uid',10001)!=os.getuid(),'Container candidate UID must differ from supervisor UID')
    return str(binary),image

def prepare_container_dirs(run):
    helpers=run/'launcher';helpers.mkdir()
    for name in ('runner_child.py','check_kaggle_environment.py','container_selftest.py'):shutil.copyfile(ROOT/name,helpers/name)
    output=run/'container_output';output.mkdir(mode=0o777);output.chmod(0o777)
    for name in ('home','tmp'):(output/name).mkdir(mode=0o777);(output/name).chmod(0o777)
    (run/'public_cache').mkdir()
    return helpers,output

def container_argv(c,run,name,mode,*,side='solution',timeout_s=900):
    binary,image=docker_spec(c);uid=c.get('container_uid',10001);gid=c.get('container_gid',10001)
    command=[binary,'run','--pull=never','--name',name,'--label','org.soup.queue.owner='+name,'--network','none','--user',f'{uid}:{gid}','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges:true','--ipc','private','--pids-limit','512','--cpuset-cpus',','.join(map(str,c['cpu_affinity'])),'--gpus','device=0','--tmpfs','/tmp:rw,nosuid,nodev,size=536870912','--workdir','/task','--entrypoint','/usr/local/bin/python']
    mounts=[(run/'repo','/task',True),(run/'public_cache','/hf_cache',True),(run/'launcher','/runner',True),(run/'container_output','/output',False)]
    for source,target,ro in mounts:
        require(',' not in str(source),'Comma in mount path is unsupported')
        command+=['--mount','type=bind,src='+str(source)+',dst='+target+(',readonly' if ro else '')]
    env={'PATH':'/usr/local/bin:/usr/bin:/bin','USER':'soup-candidate','LOGNAME':'soup-candidate','HOME':'/output/home','TMPDIR':'/output/tmp','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','HF_HOME':'/output/home/redacted','HF_HUB_CACHE':'/hf_cache','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'0','CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','SOUP_NO_AUDIT_LOG':'1','NVIDIA_DRIVER_CAPABILITIES':'compute,utility'}
    if mode=='selftest':env['SOUP_HOST_SENTINEL']=str(run/'supervisor_private_sentinel')
    for key,value in env.items():command+=['--env',key+'='+value]
    command+=[image,'-B']
    if mode=='selftest':return command+['/runner/container_selftest.py']
    command+=['/runner/runner_child.py','--cpus',','.join(map(str,c['cpu_affinity']))]
    if mode=='environment':return command+['environment','--checker','/runner/check_kaggle_environment.py','--out','/output/environment.json']
    require(mode=='harness','No arbitrary container command accepted')
    return command+['harness','--repo','/task','--side',side,'--timeout',str(timeout_s),'--out','/output/runs']

def container_process(c,run,mode,timeout,heartbeat=None,side='solution',native_timeout=900):
    binary,image=docker_spec(c);name='soupq-'+uuid.uuid4().hex
    # Inspect only; never pull/build/install through the worker.
    inspected=subprocess.run([binary,'image','inspect',image],capture_output=True,text=True,timeout=20)
    require(inspected.returncode==0,'Pinned local image missing; no automatic pull')
    argv=container_argv(c,run,name,mode,side=side,timeout_s=native_timeout)
    process=None;inspection=None;cleanup=False
    try:
        process=supervised_process(argv,str(run),{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},timeout,run/(mode+'.container.log'),heartbeat)
    finally:
        try:
            info=subprocess.run([binary,'inspect',name],capture_output=True,text=True,timeout=20)
            if info.returncode==0:
                inspection=json.loads(info.stdout)[0]
                require(inspection.get('Config',{}).get('Labels',{}).get('org.soup.queue.owner')==name,'Refuse cleanup of foreign container')
                subprocess.run([binary,'rm','--force',name],capture_output=True,timeout=30)
            remaining=subprocess.run([binary,'ps','-aq','--filter','name=^/'+name+'$'],capture_output=True,text=True,timeout=20)
            cleanup=remaining.returncode==0 and not remaining.stdout.strip()
        except Exception:
            cleanup=False
        if not cleanup:
            marker=Path(str(GPU_LOCK_PATH)+'.blocked')
            if not marker.exists():write_json(marker,{'container':name,'reason':'owned_container_cleanup_unknown'})
    require(cleanup,'Container cleanup unknown; host GPU lane quarantined')
    require(process is not None,'Container did not produce process evidence')
    process['container_name']=name;process['container_inspect']=inspection;process['container_cleanup_confirmed']=cleanup;process['local_image_inspect']=json.loads(inspected.stdout)
    return process

def check_container_policy(inspection,c,run=None):
    if not inspection:return False
    h=inspection.get('HostConfig',{});cfg=inspection.get('Config',{});mounts=inspection.get('Mounts',[])
    binds=[m for m in mounts if m.get('Type','bind')=='bind']
    extras=[m for m in mounts if m.get('Type','bind')!='bind']
    if any(m.get('Type')!='tmpfs' or m.get('Destination')!='/tmp' for m in extras):return False
    actual={(m.get('Destination'),m.get('RW')) for m in binds}
    if run is not None:
        sources={'/task':str(run/'repo'),'/hf_cache':str(run/'public_cache'),'/runner':str(run/'launcher'),'/output':str(run/'container_output')}
        if any(m.get('Source')!=sources.get(m.get('Destination')) for m in binds):return False
    return (h.get('NetworkMode')=='none' and h.get('ReadonlyRootfs') is True and h.get('Privileged') is False and h.get('PidMode','')=='' and h.get('IpcMode')=='private' and 'ALL' in h.get('CapDrop',[]) and any(x.startswith('no-new-privileges') for x in h.get('SecurityOpt',[])) and cfg.get('User')==f"{c.get('container_uid',10001)}:{c.get('container_gid',10001)}" and actual=={('/task',False),('/hf_cache',False),('/runner',False),('/output',True)})

def validate_boundary_record(c):
    require(c.get('gpu_execution_enabled') is True,'GPU execution disabled until target Docker selftest and explicit activation')
    p=checked_file(c['container_validation_record'],c['container_validation_record_sha256']);r=read_json(p)
    require(r.get('status')=='passed_target_container_selftest' and r.get('host_fingerprint')==host_fingerprint() and r.get('container_image')==c['container_image'],'Missing same-host/image boundary acceptance')
    require(r.get('helpers')==boundary_helpers() and r.get('boundary_binding')==boundary_binding(c),'Boundary code/configuration changed; rerun target selftest')
    return r

def guard_gpu(c,run,env=None,heartbeat=None):
    validate_boundary_record(c);manifest=verify_cache(c)
    # Hardlink only reviewed public bytes into a dedicated cache view. No unrelated cache files are mounted.
    cache=Path(c['hf_hub_cache']).resolve();view=run/'public_cache'
    for name in manifest['files']:
        target=view/name;target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link((cache/name).resolve(),target)
        except OSError as exc:raise QueueError('Public cache/workspace must support same-filesystem hardlinks; no implicit copying or downloads') from exc
    process=container_process(c,run,'environment',120,heartbeat)
    raw=output_bytes(run/'container_output','environment.json',optional=True);environment=json.loads(raw) if raw is not None else {}
    require(process['exit_code']==0 and not process['timed_out'] and check_container_policy(process['container_inspect'],c,run) and environment.get('passed') is True,'Strict same-Kaggle container stack/device check failed')
    smi=subprocess.run(['/usr/bin/nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    require(smi.returncode==0,'Cannot inspect all host GPUs for idleness')
    rows=[[int(x.strip()) for x in line.split(',')] for line in smi.stdout.splitlines()]
    require(rows and all(mem<=1000 and util==0 for mem,util in rows),'Host GPU not idle; unrelated work is not stopped')
    return {'environment_sha256':digest(raw),'environment':environment,'host_fingerprint':host_fingerprint(),'physical_gpu_idle_rows':rows,'container_process':process}

def container_selftest(c):
    c=checked_config(c);require(c['backend']!='CPU','Target container gate is separate from local CPU smoke')
    docker_spec(c)
    root=Path(c['workspace_root']).resolve();root.mkdir(parents=True,exist_ok=True);run=root/('boundary_'+uuid.uuid4().hex);run.mkdir(mode=0o700)
    (run/'repo').mkdir();(run/'repo/readonly_marker').write_text('public test input');(run/'supervisor_private_sentinel').write_text('not mounted; no real secret')
    prepare_container_dirs(run)
    with host_gpu_lock():process=container_process(c,run,'selftest',60)
    raw=output_bytes(run/'container_output','container_selftest.json',limit=65536,optional=True);report=json.loads(raw) if raw is not None else None
    passed=process['exit_code']==0 and not process['timed_out'] and check_container_policy(process['container_inspect'],c,run) and isinstance(report,dict) and report.get('status')=='passed'
    record={'status':'passed_target_container_selftest' if passed else 'failed','host_fingerprint':host_fingerprint(),'container_image':c['container_image'],'helpers':boundary_helpers(),'boundary_binding':boundary_binding(c),'process':process,'child_report':report,'not_hostile_gpu_security_proof':True,'not_training_or_performance':True}
    h=write_json(run/'container_validation_record.json',record);return {'status':record['status'],'container_validation_record':str(run/'container_validation_record.json'),'container_validation_record_sha256':h}

def base_result(spec,run_reference,raw_sha,manifest_sha=None,allowlist_sha=None):
    return {'schema_version':1,**{k:spec[k] for k in IDENTITY},'outcome':'failed',
      'execution':{'run_reference':run_reference,'command_exit_code':None,'wrapper_exit_code':None,'nested_metrics_exit_code':None,'verified_global_step':None,'visible_step':None,'expected_steps':None,'adapter_saved':None,'wrapper_elapsed_s':None,'termination_reason':'command_error','task_completion_status':'UNKNOWN'},
      'metrics':{'command_runtime_s':None,'observed_runtime_s':None,'runtime_kind':'unavailable','peak_memory_mib':None,'peak_kind':'unavailable','memory_scope':'unknown','eval_loss':None,'own_kernel_share':None,'admission':'UNKNOWN','score':None,'runtime_scope':{'cpu_contract_v1':'cpu_contract_process','soup_screen_v1':'native_train_command'}[spec['entrypoint']]},
      'preflight':{'status':'NOT_RUN','scope':'NOT_RUN','raw_sha256':None,'source_sha256':None},
      'provenance':{'raw_result_sha256':raw_sha,'checks_sha256':None,'collector':'reviewed-runner-v1','admission_reviewer':None,'score_evidence_sha256':None,'runtime_json_pointer':None,'peak_json_pointer':None,'actual_route_counts':None,'own_kernel_names':None,'executed_source_manifest_sha256':manifest_sha,'source_set_matches_reviewed':True if manifest_sha and allowlist_sha else None,'route_evidence_state':'unknown','source_allowlist_sha256':allowlist_sha,'route_evidence_scope':'unknown','activation_status':'UNKNOWN','own_share_scope':'unknown','own_share_evidence_sha256':None}}

def finish_cpu(spec,reference,record,manifest_sha,allowlist_sha):
    r=base_result(spec,reference,digest(canonical(record).encode()),manifest_sha,allowlist_sha); e=r['execution'];m=r['metrics']; p=record['process']; report=record.get('child_result')
    ok=p['exit_code']==0 and not p['timed_out'] and not p['heartbeat_error'] and not p['surviving_owned_pids'] and isinstance(report,dict) and report.get('status')=='passed'
    reason='completed' if ok else 'timeout' if p['timed_out'] else 'unknown' if p['heartbeat_error'] else 'signal' if p['exit_code'] is not None and p['exit_code']<0 else 'command_error'
    r['outcome']='completed' if ok else 'aborted' if reason in ('timeout','signal','unknown') else 'failed'
    e.update(command_exit_code=p['exit_code'],wrapper_exit_code=p['exit_code'],wrapper_elapsed_s=p['elapsed_s'],termination_reason=reason,task_completion_status='NOT_APPLICABLE')
    m.update(command_runtime_s=p['elapsed_s'] if ok else None,observed_runtime_s=p['elapsed_s'],runtime_kind='complete' if ok else 'right_censored' if reason in ('timeout','signal') else 'partial_observation' if reason=='unknown' else 'error_duration')
    r['preflight']={'status':'PASS' if ok else 'FAIL' if isinstance(report,dict) and report.get('status')=='failed' and not p['timed_out'] and not p['heartbeat_error'] else 'UNKNOWN','scope':'CPU_CONTRACT','raw_sha256':r['provenance']['raw_result_sha256'],'source_sha256':spec['source_sha256']}
    r['provenance']['runtime_json_pointer']='/process/elapsed_s'; return r

def normalize_screen(spec,reference,record,manifest_sha,allowlist_sha,expected_steps=64):
    """Normalize only official outer evidence, never candidate claims about score/admission."""
    r=base_result(spec,reference,digest(canonical(record).encode()),manifest_sha,allowlist_sha);e=r['execution'];m=r['metrics'];p=r['provenance']; proc=record['process'];raw=record.get('harness_result') or {}
    runs=raw.get('runs',{}).get('solution',[]); native=runs[0] if len(runs)==1 else {}; nested=native.get('metrics') or {}
    native_exit=native.get('exit_code'); wall=native.get('wall_s'); step=native.get('global_step'); adapter=bool(native.get('adapter_tensors')) and bool(native.get('adapter_config')) and record.get('adapter_saved_on_disk') is True
    e.update(command_exit_code=native_exit,wrapper_exit_code=proc['exit_code'],nested_metrics_exit_code=nested.get('exit_code'),verified_global_step=step,expected_steps=expected_steps,adapter_saved=adapter if native else None,wrapper_elapsed_s=proc['elapsed_s'])
    e['visible_step']=record.get('visible_step')
    complete=native_exit==0 and proc['exit_code']==0 and not proc['timed_out'] and not proc['heartbeat_error'] and not proc['surviving_owned_pids'] and nested.get('exit_code') in (None,0) and step==expected_steps and adapter and not raw.get('packages_changed')
    # The unmodified official logger records TIMEOUT explicitly; -9 without that marker is only a signal.
    reason='completed' if complete else 'timeout' if proc['timed_out'] or record.get('native_timeout') is True else 'unknown' if proc['heartbeat_error'] else 'signal' if (native_exit is not None and native_exit<0) or (proc['exit_code'] is not None and proc['exit_code']<0) else 'command_error' if native_exit not in (None,0) or proc['exit_code']!=0 else 'task_incomplete'
    r['outcome']='completed' if complete else 'aborted' if reason in ('timeout','signal','unknown') else 'failed'; e.update(termination_reason=reason,task_completion_status='VERIFIED_COMPLETE' if complete else 'INCOMPLETE' if native else 'UNKNOWN')
    if type(wall) in (int,float) and math.isfinite(wall) and wall>=0:
        m.update(command_runtime_s=wall if complete else None,observed_runtime_s=wall,runtime_kind='complete' if complete else 'right_censored' if reason in ('timeout','signal') else 'partial_observation' if reason=='unknown' else 'error_duration');p['runtime_json_pointer']='/harness_result/runs/solution/0/wall_s'
    peak=native.get('gpu_peak_mib')
    if type(peak) in (int,float) and math.isfinite(peak) and peak>=0:
        m.update(peak_memory_mib=peak,peak_kind='complete' if complete else 'partial',memory_scope='gpu_physical_sum');p['peak_json_pointer']='/harness_result/runs/solution/0/gpu_peak_mib'
    evaluation=raw.get('evals',{}).get('solution',{})
    if complete and evaluation.get('exit_code')==0 and type(evaluation.get('mean_loss')) in (int,float) and math.isfinite(evaluation['mean_loss']) and evaluation['mean_loss']>=0:m['eval_loss']=evaluation['mean_loss']
    # Preserve raw per-kernel maps in record. Automatic share/admission require separate review and remain unknown.
    return r

def validate_result(r,spec):
    # The validator is pure; do not open/create a second queue database.
    Queue._validate_result(None,r,spec)

def same_host_baseline(c,guard):
    path=checked_file(c['baseline_record'],c['baseline_record_sha256']); b=read_json(path)
    baseline_spec={k:c[k] for k in ('baseline_id','backend','environment_fingerprint','task_fingerprint','harness_commit')}
    baseline_spec['evidence_sha256']=c['baseline_record_sha256']
    require(digest(canonical(baseline_spec).encode())==c['baseline_fingerprint'],'Registered baseline fingerprint must bind this exact measured baseline record')
    require(b.get('status')=='measured_base_complete' and b.get('host_fingerprint')==guard['host_fingerprint'],'A completed baseline on this host is required')
    require(b.get('backend')==c['backend'] and b.get('baseline_id')==c['baseline_id'] and b.get('container_image')==c['container_image'],'Baseline cohort/image mismatch')
    for k in ('task_fingerprint','environment_fingerprint','harness_commit','data_files','model_cache_manifest_sha256','cpu_affinity','baseline_timeout_s','baseline_outer_timeout_s'):require(b.get(k)==c[k],'Baseline binding differs: '+k)
    # GPU inventory/driver and direct package pins must match the measured baseline, not merely a caller fingerprint.
    for k in ('python','packages','nvidia_smi_inventory','visible_device_names','cpu_affinity'):require(b['environment'].get(k)==guard['environment'].get(k),'Baseline runtime differs: '+k)
    expected=path.parent/'container_output/runs/result.json';require(b.get('raw_result_path')==str(expected),'Baseline raw path must be its own output')
    rawbytes=output_bytes(path.parent/'container_output','runs/result.json');require(digest(rawbytes)==b['raw_result_sha256'],'Baseline raw evidence hash mismatch');raw=json.loads(rawbytes);base=raw.get('runs',{}).get('base',[])
    require(len(base)==1 and base[0].get('exit_code')==0 and base[0].get('global_step')==64 and base[0].get('adapter_tensors') and finite_eval(raw.get('evals',{}).get('base',{})),'Baseline raw evidence is incomplete')
    return b

def execute_claim(claim,c,artifact,run,heartbeat=None):
    spec=claim['spec'];h=spec['source_sha256'];profile=c['profiles'].get(h)
    require(profile is not None,'Bundle not in runner-owned approved hash allowlist')
    require(claim['review'].get('approved') is True and claim['review'].get('source_sha256')==h,'Exact reviewed source approval required')
    for k in ('backend','baseline_id','baseline_fingerprint','environment_fingerprint','task_fingerprint','harness_commit'):require(spec[k]==c[k],'Claim does not match configured cohort: '+k)
    require(spec['artifact_kind']=='solution_bundle','Notebook execution is unsupported; submit a reviewed source bundle')
    expected_entry='cpu_contract_v1' if profile['kind']=='cpu_smoke_v1' else 'soup_screen_v1';require(spec['entrypoint']==expected_entry,'Entrypoint/profile mismatch')
    raw=Path(artifact).read_bytes();require(digest(raw)==h,'Downloaded artifact hash mismatch')
    env=child_env(run,c);stage=run/'repo';stage.mkdir();reference=run.name; official=None;guard=None
    if profile['kind']=='soup_screen_v1':
        require(profile['timeout_s']==c.get('baseline_timeout_s') and profile['outer_timeout_s']==c.get('baseline_outer_timeout_s'),'Candidate and baseline must use identical declared timeout budgets')
        official=stage_official(c,stage);prepare_container_dirs(run)
        guard=guard_gpu(c,run,env,heartbeat);same_host_baseline(c,guard)
        shutil.rmtree(stage/'solution')
    files=stage_bundle(raw,stage,profile['files']);allowlist=dict(profile['files'])
    if official is not None:
        for n,hsh in profile.get('support_files',{}).items():
            require(official.get(n)==hsh,'Support file is not exact pinned starter source')
            target=stage/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(c['repo'])/n,target);allowlist[n]=hsh
    actual={str(p.relative_to(stage)):file_hash(p) for p in (stage/'solution').rglob('*') if p.is_file()} if official is not None else {str(p.relative_to(stage)):file_hash(p) for p in stage.rglob('*') if p.is_file()}
    require(actual==allowlist,'Staged source set differs')
    manifest_sha=write_json(run/'executed_source_manifest.json',actual);allowlist_sha=write_json(run/'reviewed_source_allowlist.json',allowlist)
    cpus=','.join(map(str,c['cpu_affinity'])); launcher=[c['python'],'-B',str(ROOT/'runner_child.py'),'--cpus',cpus]
    if official is None:
        command=launcher+['smoke','--root',str(stage),'--module',profile['module'],'--out',str(run/'child_result.json')]
        process=supervised_process(command,str(stage),env,profile['timeout_s'],run/'child.log',heartbeat)
        resultpath=run/'child_result.json';child=read_json(resultpath) if resultpath.exists() else None
        record={'process':process,'child_result':child,'source_sha256':h,'not_gpu_or_training':True};r=finish_cpu(spec,reference,record,manifest_sha,allowlist_sha)
    else:
        process=container_process(c,run,'harness',profile['outer_timeout_s'],heartbeat,native_timeout=profile['timeout_s'])
        require(check_container_policy(process['container_inspect'],c,run),'Container boundary changed')
        out=run/'container_output';raw=output_bytes(out,'runs/result.json',optional=True);native=json.loads(raw) if raw is not None else None;logbytes=output_bytes(out,'runs/solution_1/train.log',optional=True);log=logbytes.decode(errors='replace') if logbytes is not None else ''
        progress=[int(x) for x in re.findall(r'(\d+)/64\b',log)]
        record={'process':process,'harness_result':native,'guard':guard,'official_source_manifest':official,'adapter_saved_on_disk':output_present(out,'runs/solution_1/adapter/adapter_model.safetensors'),'native_timeout':bool(re.search(r'^TIMEOUT after \d+ s$',log,re.M)),'visible_step':max(progress) if progress else None,'train_log_sha256':digest(logbytes) if logbytes is not None else None}
        r=normalize_screen(spec,reference,record,manifest_sha,allowlist_sha,profile['expected_steps'])
    # Hash exact durable canonical raw evidence bytes, not candidate-generated JSON admission claims.
    rawbytes=canonical(record).encode();(run/'runner_raw.json').write_bytes(rawbytes);r['provenance']['raw_result_sha256']=digest(rawbytes)
    validate_result(r,spec);write_json(run/'normalized_result.json',r);return r

def consume_once(client,trusted_config):
    c=checked_config(trusted_config)
    if c['backend']!='CPU':
        docker_spec(c);validate_boundary_record(c)
    response=client.request('POST','/v1/claims',{'backend':c['backend'],'baseline_id':c['baseline_id'],'resource':c['resource'],'lease_seconds':c['lease_seconds']})
    claim=response.get('claim')
    if claim is None:return response
    root=Path(c['workspace_root']).resolve();root.mkdir(parents=True,exist_ok=True);run=root/('run_'+uuid.uuid4().hex);run.mkdir(mode=0o700)
    write_json(run/'claim.json',claim)
    endpoint=f"/v1/jobs/{claim['job_id']}/attempts/{claim['attempt_id']}"
    def heartbeat():client.request('POST',endpoint+'/heartbeat',{'lease_seconds':c['lease_seconds']})
    started=False
    try:
        artifact=run/'source.zip';client.download(claim['job_id'],artifact,claim['spec']['source_sha256'])
        # Fail closed before start when queue authority and local allowlist disagree.
        require(claim['spec']['source_sha256'] in c['profiles'],'No local reviewed profile for this exact ZIP hash')
        client.request('POST',endpoint+'/start',{'run_reference':run.name});started=True
        with host_gpu_lock() if c['backend']!='CPU' else contextlib.nullcontext():
            result=execute_claim(claim,c,artifact,run,heartbeat)
        client.request('POST',endpoint+'/complete',result)
        return {'run_dir':str(run),'result':result}
    except Exception as exc:
        evidence={'status':'worker_error','error_type':type(exc).__name__,'message':str(exc),'start_record_sent':started,'no_automatic_retry':True}
        h=write_json(run/'worker_error.json',evidence)
        # Never rerun after an uncertain network/launch outcome. Keep the resource locked for review.
        try:client.request('POST',endpoint+'/uncertain',{'reason':'runner_crash_report','evidence_sha256':h})
        except Exception:pass
        raise

def measure_baseline(c):
    c=checked_config(c);require(c['backend']!='CPU','Baseline is actual full Soup, never CPU smoke')
    require(type(c.get('baseline_timeout_s')) is int and c['baseline_timeout_s']>0 and type(c.get('baseline_outer_timeout_s')) is int and c['baseline_outer_timeout_s']>c['baseline_timeout_s'],'Explicit local-platform baseline budgets required')
    with host_gpu_lock():
        root=Path(c['workspace_root']).resolve();root.mkdir(parents=True,exist_ok=True);run=root/('baseline_'+uuid.uuid4().hex);run.mkdir(mode=0o700);stage=run/'repo';stage.mkdir();official=stage_official(c,stage);prepare_container_dirs(run)
        guard=guard_gpu(c,run)
        process=container_process(c,run,'harness',c['baseline_outer_timeout_s'],side='base',native_timeout=c['baseline_timeout_s']);write_json(run/'process.json',process)
        require(check_container_policy(process['container_inspect'],c,run),'Container boundary changed')
        out=run/'container_output';path=out/'runs/result.json';rawbytes=output_bytes(out,'runs/result.json',optional=True);raw=json.loads(rawbytes) if rawbytes is not None else {};base=raw.get('runs',{}).get('base',[])
        good=process['exit_code']==0 and not process['timed_out'] and not process['surviving_owned_pids'] and len(base)==1 and base[0].get('exit_code')==0 and base[0].get('global_step')==64 and bool(base[0].get('adapter_tensors')) and bool(base[0].get('adapter_config')) and output_present(out,'runs/base_1/adapter/adapter_model.safetensors') and base[0].get('metrics',{}).get('exit_code') in (None,0) and finite_eval(raw.get('evals',{}).get('base',{})) and not raw.get('packages_changed')
        record={'status':'measured_base_complete' if good else 'baseline_failed','host_fingerprint':guard['host_fingerprint'],'environment':guard['environment'],'container_image':c['container_image'],'raw_result_path':str(path),'raw_result_sha256':digest(rawbytes) if rawbytes is not None else None,'official_source_manifest':official,'not_official_kaggle_admission':True,**{k:c[k] for k in ('backend','baseline_id','task_fingerprint','environment_fingerprint','harness_commit','data_files','model_cache_manifest_sha256','cpu_affinity','baseline_timeout_s','baseline_outer_timeout_s')}}
        h=write_json(run/'baseline_record.json',record);return {'run_dir':str(run),'baseline_record':str(run/'baseline_record.json'),'baseline_record_sha256':h,'status':record['status']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--once',action='store_true');g.add_argument('--baseline',action='store_true');g.add_argument('--selftest',action='store_true');p.add_argument('--token-env',default='SOUP_QUEUE_WORKER_TOKEN');a=p.parse_args();c=checked_config(a.config)
    if a.selftest:result=container_selftest(c)
    elif a.baseline:result=measure_baseline(c)
    else:
        from agent_client import Client
        result=consume_once(Client(c['queue_url'],os.environ.get(a.token_env,'')),c)
    print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
