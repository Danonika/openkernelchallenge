import os, sys, subprocess, time, json, pathlib, signal, importlib.metadata
os.environ['CUDA_VISIBLE_DEVICES']='0'
ROOT=pathlib.Path('/kaggle/working')
START=time.monotonic()
def deadline(sig,frame): raise TimeoutError('Baseline session outer budget reached')
signal.signal(signal.SIGALRM,deadline); signal.alarm(3000)
def run(cmd,cwd=None,timeout=900):
    left=max(1,int(2950-(time.monotonic()-START)))
    print('RUN',cmd,flush=True)
    subprocess.run(cmd,cwd=cwd,check=True,timeout=min(timeout,left))
try:
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
    print(gpu,flush=True)
    assert 'T4' in gpu, 'T4 required; no run on substitute hardware'
    run([sys.executable,'-m','pip','install','-q','soup-cli[train]==0.77.0','transformers==5.16.1','peft==0.20.0','trl==0.29.1','bitsandbytes==0.50.2','accelerate==1.14.0','pytest'],timeout=900)
    repo=ROOT/'soup-official-baseline'
    assert not repo.exists(), 'Fresh baseline directory required'
    run(['git','clone','-q','https://github.com/MakazhanAlpamys/soup-kernel-challenge.git',str(repo)])
    run(['git','checkout','-q','c7c2fcd15972ce8b898d8d3ff732ee192c817ceb'],cwd=repo)
    versions={p:importlib.metadata.version(p) for p in ['torch','triton','soup-cli','transformers','peft','trl','bitsandbytes','accelerate']}
    (ROOT/'baseline_environment.json').write_text(json.dumps({'versions':versions,'gpu':gpu,'starter_commit':'c7c2fcd15972ce8b898d8d3ff732ee192c817ceb','purpose':'single baseline smoke, not final score'},indent=2))
    print(versions,flush=True)
    run([sys.executable,'task/prepare_data.py'],cwd=repo,timeout=600)
    run([sys.executable,'-m','pytest','solution','-q'],cwd=repo,timeout=300)
    run([sys.executable,'harness/run_task.py','--only','base','--repeats','1','--timeout','900','--out','/kaggle/working/runs_baseline'],cwd=repo,timeout=2400)
    (ROOT/'baseline_session_status.json').write_text(json.dumps({'status':'completed','elapsed_seconds':time.monotonic()-START}))
except BaseException as exc:
    (ROOT/'baseline_session_status.json').write_text(json.dumps({'status':'failed','error_type':type(exc).__name__,'error':str(exc),'elapsed_seconds':time.monotonic()-START}))
    raise
finally:
    signal.alarm(0)
