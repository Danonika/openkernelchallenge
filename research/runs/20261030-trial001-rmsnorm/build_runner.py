"""Build a credential-free, self-contained Kaggle script from original trial files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ['solution/__init__.py', 'solution/rmsnorm.py', 'solution/verify_trial001.py']
payload = {name: (ROOT/name).read_text() for name in FILES}
hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name,text in payload.items()}
template = r'''import os,sys,subprocess,time,json,pathlib,signal,hashlib,importlib.metadata,math
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['PYTHONUNBUFFERED']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
ROOT=pathlib.Path('/kaggle/working')
OUT=ROOT/'trial001_evidence'
OUT.mkdir(exist_ok=True)
START=time.monotonic()
PAYLOAD=__PAYLOAD__
HASHES=__HASHES__
COMMIT='c7c2fcd15972ce8b898d8d3ff732ee192c817ceb'
PHASE='starting'
def deadline(sig,frame): raise TimeoutError('Trial001 outer 3000-second budget reached')
signal.signal(signal.SIGALRM,deadline);signal.alarm(3000)
def terminate_owned(proc):
    # Only this wrapper's subprocess descendants; no credential/process-env reads.
    try:
        import psutil
        children=psutil.Process(proc.pid).children(recursive=True)
        for child in reversed(children):
            try: child.kill()
            except psutil.Error: pass
    except (ImportError,Exception):
        pass
    try: os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: pass
def run(cmd,name,cwd=None,timeout=600,extra_env=None,allow_failure=False):
    global PHASE
    PHASE=name
    remaining=max(1,int(2950-(time.monotonic()-START)))
    print('PHASE',name,'COMMAND',cmd,flush=True)
    env=dict(os.environ);env.update(extra_env or {})
    logfile=OUT/(name+'.log')
    with logfile.open('w') as log:
        proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=proc.wait(timeout=min(timeout,remaining))
        except BaseException:
            terminate_owned(proc)
            raise
    content=logfile.read_text(errors='replace')
    print(content[-16000:],flush=True)
    print('PHASE_END',name,'EXIT',code,flush=True)
    if code and not allow_failure:
        raise RuntimeError(f'{name} exited {code}; see {logfile.name}')
    return code,content
try:
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True)
    assert gpu and all('T4' in line for line in gpu.splitlines()), 'Actual T4 required'
    print('ACTUAL_GPUS',gpu,flush=True)
    run([sys.executable,'-m','pip','install','-q','soup-cli[train]==0.77.0','transformers==5.16.1','peft==0.20.0','trl==0.29.1','bitsandbytes==0.50.2','accelerate==1.14.0','pytest'],'install',timeout=900)
    packages=['torch','triton','soup-cli','transformers','peft','trl','bitsandbytes','accelerate','datasets']
    versions={p:importlib.metadata.version(p) for p in packages}
    assert versions['torch']=='2.11.0+cu128' and versions['triton']=='3.6.0', 'Baseline Torch/Triton build required; no compiler replacement'
    repo=ROOT/'soup-trial001'
    assert not repo.exists(), 'Fresh official clone required'
    run(['git','clone','-q','https://github.com/MakazhanAlpamys/soup-kernel-challenge.git',str(repo)],'clone')
    run(['git','checkout','-q',COMMIT],'checkout',cwd=repo)
    for name,source in PAYLOAD.items():
        path=repo/name
        assert name.startswith('solution/') and '..' not in pathlib.Path(name).parts
        assert hashlib.sha256(source.encode()).hexdigest()==HASHES[name]
        path.write_text(source)
    run(['git','diff','--exit-code','--','harness','task','tests'],'frozen_source_check',cwd=repo)
    manifest={'trial':'trial001','catalog_id':'liger_castaware_norm','starter_commit':COMMIT,
              'source_sha256':HASHES,'versions':versions,'physical_gpus':gpu,
              'purpose':'conditional original RMSNorm trial, not guaranteed admission',
              'command':['harness/run_task.py','--repeats','2','--timeout','900'],
              'server_cap_seconds':3600,'outer_cap_seconds':3000}
    (OUT/'environment.json').write_text(json.dumps(manifest,indent=2))
    print('ENVIRONMENT',json.dumps(manifest),flush=True)
    run([sys.executable,'-m','solution.verify_trial001','--out',str(OUT/'gpu_preflight.json')],'gpu_preflight',cwd=repo,timeout=300)
    preflight=json.loads((OUT/'gpu_preflight.json').read_text())
    assert preflight['status']=='passed', 'No training after ordinary-case preflight failure'
    _,regression_log=run([sys.executable,'-m','pytest','tests','-q'],'patched_regressions',cwd=repo,timeout=300,extra_env={'SOUP_APPLY_SOLUTION':'1'})
    assert '79 passed' in regression_log, 'All 79 unmodified Soup regressions required'
    run([sys.executable,'-m','pytest','solution','-q'],'starter_tests',cwd=repo,timeout=300)
    run([sys.executable,'task/prepare_data.py'],'public_data',cwd=repo,timeout=600)
    runs=ROOT/'runs_trial001'
    run([sys.executable,'harness/run_task.py','--repeats','2','--timeout','900','--out',str(runs)],'paired_harness',cwd=repo,timeout=2400)
    result=json.loads((runs/'result.json').read_text())
    codes={side:[r.get('exit_code') for r in result.get('runs',{}).get(side,[])] for side in ('base','solution')}
    evals={side:result.get('evals',{}).get(side,{}) for side in ('base','solution')}
    nested_success=all(codes[side]==[0,0] and evals[side].get('exit_code')==0
                       and isinstance(evals[side].get('mean_loss'),(int,float))
                       and math.isfinite(evals[side]['mean_loss']) for side in codes)
    checks_code,_=run([sys.executable,'harness/check_rules.py','--result',str(runs/'result.json')],'official_checks',cwd=repo,timeout=300,allow_failure=True)
    status={'status':'measured_all_checks_passed' if nested_success and checks_code==0 else 'measured_checks_failed',
            'nested_training_exit_codes':codes,'nested_training_and_eval_success':nested_success,
            'official_checks_exit_code':checks_code,'elapsed_seconds':time.monotonic()-START,
            'result_sha256':hashlib.sha256((runs/'result.json').read_bytes()).hexdigest(),
            'note':'Read exact checks and raw runs; no official jury or leaderboard claim.'}
    (OUT/'session_status.json').write_text(json.dumps(status,indent=2))
    print('TRIAL001_TERMINAL',json.dumps(status),flush=True)
except BaseException as exc:
    status={'status':'failed','phase':PHASE,'error_type':type(exc).__name__,'error':str(exc),
            'elapsed_seconds':time.monotonic()-START}
    (OUT/'session_status.json').write_text(json.dumps(status,indent=2))
    print('TRIAL001_TERMINAL',json.dumps(status),flush=True)
    raise
finally:
    signal.alarm(0)
'''
script=template.replace('__PAYLOAD__',repr(payload)).replace('__HASHES__',repr(hashes))
compile(script,'kaggle_runner.py','exec')
(ROOT/'kaggle_runner.py').write_text(script)
(ROOT/'source_manifest.json').write_text(json.dumps({'files':hashes,'runner_sha256':hashlib.sha256(script.encode()).hexdigest()},indent=2)+'\n')
print(json.dumps({'runner':str(ROOT/'kaggle_runner.py'),'source_hashes':hashes},indent=2))
