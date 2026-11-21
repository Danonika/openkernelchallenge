#!/usr/bin/env python3
"""Offline paired official harness runner. Preparing this file does not run a GPU.

Input directories are pre-provisioned. No install/download/queue submission.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import tempfile
import uuid

import process_guard

COMMIT = 'c7c2fcd15972ce8b898d8d3ff732ee192c817ceb'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
PINS = {'torch':'2.11.0+cu128','triton':'3.6.0','soup-cli':'0.77.0','transformers':'5.16.1',
        'peft':'0.20.0','trl':'0.29.1','bitsandbytes':'0.50.2','accelerate':'1.14.0','datasets':'4.8.5'}
DATA = {'train.jsonl':'93d036d912bd81c337dc29893e13a76f36161d96f896ef4e02492e2f1a21a0dd',
        'eval_public.jsonl':'b7f85b78ba3186dce4deab7cfb8bf8dcf7998e1517c5749ac0f5005886c6f70e'}
KERNEL = 'lora_half_scale_add'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def command(cmd, cwd=None, env=None):
    with tempfile.TemporaryFile(mode='w+t') as out, tempfile.TemporaryFile(mode='w+t') as err:
        code=process_guard.run(cmd,cwd=cwd,env=env,stdout=out,stderr=err)
        out.seek(0);err.seek(0);stdout,stderr=out.read(),err.read()
        if code:raise subprocess.CalledProcessError(code,cmd,stdout,stderr)
        return stdout.strip()


def frozen_hashes(repo):
    names = command(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', COMMIT,
                     '--', 'harness', 'task', 'tests']).splitlines()
    if not names: raise RuntimeError('Official frozen source tree missing')
    out = {}
    for name in names:
        p = repo / name
        expected = subprocess.run(['git','-C',str(repo),'show',f'{COMMIT}:{name}'], check=True, capture_output=True).stdout
        if p.is_symlink() or not p.is_file() or p.read_bytes() != expected:
            raise RuntimeError('Frozen official file mismatch: ' + name)
        out[name] = hashlib.sha256(expected).hexdigest()
    status = command(['git','-C',str(repo),'status','--porcelain','--untracked-files=all','--','harness','task','tests'])
    if status: raise RuntimeError('Dirty frozen official paths: ' + status)
    return out


def verify_assets(repo, cache):
    model = cache / 'models--Qwen--Qwen2.5-1.5B-Instruct'
    if (model/'refs/main').read_text().strip() != REVISION:
        raise RuntimeError('Cached model main must be the pinned public revision')
    snapshot = model/'snapshots'/REVISION
    if not snapshot.is_dir(): raise RuntimeError('Pinned model snapshot is absent; provision it outside measurement')
    files = {}
    for p in sorted(snapshot.rglob('*')):
        if p.is_file():
            target = p.resolve(strict=True)
            if not target.is_relative_to(model.resolve()): raise RuntimeError('Model link escapes selected cache')
            files[str(p.relative_to(cache))] = sha(target)
    if not any(n.endswith('.safetensors') for n in files): raise RuntimeError('Cached safetensors weights missing')
    for name, expected in DATA.items():
        if sha(repo/'data'/name) != expected: raise RuntimeError('Public data hash mismatch: ' + name)
    return {'model_id':'Qwen/Qwen2.5-1.5B-Instruct','revision':REVISION,'files':files,'data_files':DATA}


def assess(result, out):
    """Independent fail-closed completeness gate; never assigns an admission/score."""
    problems = []
    runs = result.get('runs', {})
    shares = []
    for side in ('base', 'solution'):
        group = runs.get(side, [])
        if len(group) != 2: problems.append(f'{side}: expected exactly two runs')
        for run in group:
            m = run.get('metrics') or {}
            if run.get('exit_code') != 0 or m.get('exit_code') != 0 or m.get('steps') != 64 or run.get('global_step') != 64:
                problems.append(f"{run.get('run')}: partial/failed native training")
            if not isinstance(run.get('wall_s'), (int,float)) or run['wall_s'] <= 0:
                problems.append('Missing full native command wall time')
            if not isinstance(run.get('gpu_peak_mib'), (int,float)) or run['gpu_peak_mib'] <= 0:
                problems.append('Missing external GPU peak')
            adapter = Path(run.get('adapter') or '/missing')
            if not all((adapter/f).is_file() and (adapter/f).stat().st_size > 0 for f in ['adapter_model.safetensors','adapter_config.json']):
                problems.append(f'{side}: saved adapter missing')
            if side == 'solution':
                ms = (m.get('own_kernel_ms') or {}).get(KERNEL)
                sec = m.get('timed_loop_s')
                count = (m.get('triton_launches') or {}).get(KERNEL, 0)
                path = (m.get('triton_files') or {}).get(KERNEL, '')
                if not (count > 0 and path.startswith('solution/') and isinstance(ms,(int,float)) and math.isfinite(ms) and ms > 0 and isinstance(sec,(int,float)) and math.isfinite(sec) and sec > 0 and m.get('timed_steps') == 62 and m.get('triton_watch_error') is None):
                    problems.append(f'{side}: actual original kernel activation/time missing')
                else: shares.append(ms/(1000*sec))
        ev = (result.get('evals') or {}).get(side, {})
        loss = ev.get('mean_loss')
        if ev.get('exit_code') != 0 or not isinstance(loss,(int,float)) or not math.isfinite(loss) or loss <= 0:
            problems.append(f'{side}: public evaluation incomplete')
    if result.get('packages_changed') != []: problems.append('Packages changed or not recorded')
    complete = not problems
    return {'status': 'COMPLETE_MEASUREMENT' if complete else 'FAILED_OR_INCOMPLETE',
            'problems':problems, 'score':None, 'admission':'UNKNOWN',
            'quality_scope':'public_only', 'same_environment_baseline':complete,
            'own_share_steps_3_onward':shares if complete else None,
            'own_share_min':min(shares) if complete and len(shares)==2 else None,
            'own_share_gate':(min(shares)>=.05) if complete and len(shares)==2 else None,
            'ratios':result.get('summary',{}).get('ratios') if complete else None,
            'raw_result_sha256':sha(out/'result.json')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--solution',type=Path,default=Path(__file__).parent/'solution')
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--backend',choices=['GCP','KAGGLE'],default='GCP')
    a=p.parse_args()
    process_guard.prepare()
    process_guard.install_signal_handlers()
    if os.environ.get('TRITON_INTERPRET') not in (None,'','0'):
        raise RuntimeError('TRITON_INTERPRET must be unset for target measurement')
    if platform.python_version() != '3.13.15': raise RuntimeError('Exact prepared Python 3.13.15 required; no install is performed')
    versions = {n:importlib.metadata.version(n) for n in PINS}
    if versions != PINS: raise RuntimeError('Exact package pins mismatch: '+json.dumps(versions))
    if command(['git','-C',str(a.repo),'rev-parse','HEAD']) != COMMIT: raise RuntimeError('Official commit mismatch')
    frozen = frozen_hashes(a.repo)
    assets = verify_assets(a.repo,a.cache)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_CACHE=str(a.cache.resolve()), PYTHONUNBUFFERED='1')
    # Shared across all six local notebooks. Keep fd live until process exit,
    # including every baseline/candidate/evaluation/check. No polling/auto retry.
    process_guard.check_quarantine('/tmp/soup-gpu0.lock.blocked')
    lock_fd = os.open('/tmp/soup-gpu0.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('GPU 0 is owned by another notebook; do not overlap jobs')
    process_guard.check_quarantine('/tmp/soup-gpu0.lock.blocked')
    process_guard.begin_quarantine('/tmp/soup-gpu0.lock.blocked')
    # A short-lived child may initialize CUDA; this parent must never hold a
    # CUDA context across external-memory measurement of the training children.
    probe_code = ("import json, torch; n=torch.cuda.device_count(); "
                  "print(json.dumps({'cuda':torch.version.cuda,'count':n,"
                  "'names':[torch.cuda.get_device_name(i) for i in range(n)]}))")
    gpu_probe = json.loads(command([sys.executable, '-c', probe_code], env=dict(os.environ)))
    if gpu_probe != {'cuda':'12.8','count':1,'names':['Tesla T4']}:
        raise RuntimeError('Require one visible Tesla T4 and Torch compiled CUDA 12.8')
    available = sorted(os.sched_getaffinity(0))
    if len(available)<4: raise RuntimeError('Four available logical CPUs required')
    os.sched_setaffinity(0, available[:4])
    active = command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
    if any(line.strip().isdigit() for line in active.splitlines()):
        raise RuntimeError('Another compute process uses a physical GPU; keep the measurement exclusive')
    inventory = command(['nvidia-smi','--query-gpu=name,driver_version,memory.used','--format=csv,noheader,nounits'])
    if sum(int(line.rsplit(',',1)[1].strip()) for line in inventory.splitlines()) > 1000:
        raise RuntimeError('Physical GPUs not idle; do not overlap work')
    a.out_root.mkdir(parents=True,exist_ok=True)
    job = a.out_root / ('lora-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8])
    job.mkdir(); staged=job/'repo'; staged.mkdir()
    for dirname in ('harness','task','tests'):
        shutil.copytree(a.repo/dirname,staged/dirname,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('README.md','LICENSE','NOTICE'):
        if (a.repo/name).is_file(): shutil.copy2(a.repo/name,staged/name)
    shutil.copytree(a.solution,staged/'solution',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (staged/'data').mkdir()
    for name in DATA: shutil.copy2(a.repo/'data'/name,staged/'data'/name)
    # Do not copy adapters, optimizer state, runs, or compiler caches.
    save(job/'environment.json',{'backend':a.backend,'packages':versions,'python':platform.python_version(),
        'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'nvidia_smi':inventory,'gpu_probe_short_lived':gpu_probe,'gpu_lock':'/tmp/soup-gpu0.lock','official_commit':COMMIT,'frozen_source_sha256':frozen,
        'hardware_equivalence_claim':False})
    save(job/'assets.json',assets)
    save(job/'source.json',{str(f.relative_to(staged)):sha(f) for f in sorted((staged/'solution').rglob('*.py'))})
    out=job/'runs'
    env=dict(os.environ)
    cmd=[sys.executable,str(staged/'harness/run_task.py'),'--repeats','2','--seed','1','--timeout','900','--out',str(out)]
    save(job/'invocation.json',{'argv':cmd,'baseline_id':None,'native_budget_s':900,
        'order':['base','solution','solution','base'],'notes':'Own same-session baseline; no transplanted Kaggle 79 s budget. Offline warmup reuses cache.'})
    print('Evidence directory: '+str(job),flush=True)
    started=time.perf_counter()
    with (job/'harness.log').open('x') as log:
        try:
            code=process_guard.run(cmd,cwd=staged,env=env,stdout=log)
        except BaseException as exc:
            save(job/'assessment.json',{'status':'FAILED_OR_INTERRUPTED','score':None,'admission':'UNKNOWN',
                 'cleanup_verified':not process_guard.descendants(),'error':repr(exc)})
            raise
    wall=time.perf_counter()-started
    save(job/'harness_wall.json',{'outer_returncode':code,'whole_harness_wall_s':wall})
    if code != 0 or not (out/'result.json').is_file():
        save(job/'assessment.json',{'status':'FAILED_OR_INCOMPLETE','score':None,'admission':'UNKNOWN','outer_returncode':code})
        raise RuntimeError('Harness failed; retained logs, no score/admission')
    result=json.loads((out/'result.json').read_text())
    assessment=assess(result,out)
    # Preserve result.json verbatim even if the native wrapper exits zero on a failure.
    save(job/'assessment.json',assessment)
    if assessment['status']!='COMPLETE_MEASUREMENT':
        print(json.dumps(assessment,indent=2)); raise RuntimeError('Nested result incomplete; no score/admission')
    with (job/'official-checks.log').open('x') as log:
        check_code=process_guard.run([sys.executable,str(staged/'harness/check_rules.py'),'--result',str(out/'result.json')],
                                     cwd=staged,env=env,stdout=log,timeout=900)
    assessment.update(official_checks_returncode=check_code, admission='UNKNOWN',score=None,
                      final_official_kaggle_admission='SEPARATE_JURY_GATE')
    for rel,expected in frozen.items():
        if sha(staged/rel)!=expected: raise RuntimeError('Frozen file changed during execution: '+rel)
    save(job/'assessment.json',assessment)
    print(json.dumps(assessment,indent=2))
    if check_code: raise RuntimeError('Unchanged official checks failed; research evidence only')


if __name__=='__main__':
    try:main()
    finally:process_guard.finish_quarantine()
