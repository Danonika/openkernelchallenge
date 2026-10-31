"""Actual-batch native-return MLP shadow only; no optimizer, train loop, baseline replay or score."""
import argparse,fcntl,hashlib,json,os,shutil,sys,threading,time,uuid,zipfile
from pathlib import Path
import process_guard
import owned_process_tree as tree
COMMIT='c7c2fcd15972ce8b898d8d3ff732ee192c817ceb'
MODE='actual_batch_native_return_mlp_shadow_v1'
STATUS='COMPLETE_NATIVE_RETURN_MLP_SHADOW_DIAGNOSTIC'
LOCK=Path('/tmp/soup-gpu0.lock');MARKER=Path('/tmp/soup-gpu0.lock.blocked')
EXPORT_BUDGET=56*1024**2
EXPORT_OMITTED={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
 p=Path(p);pending=p.with_name(p.name+'.pending');pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(p)


def verify_official(repo,manifest):
 if not repo.is_dir() or repo.is_symlink():raise RuntimeError('SOUP_REPO must be a real prepared job directory')
 for name,wanted in manifest.items():
  p=repo/name
  if not p.is_file() or p.is_symlink() or sha(p)!=wanted:raise RuntimeError('Protected official source mismatch: '+name)
 actual={str(p.relative_to(repo)) for folder in ('harness','task','tests') for p in (repo/folder).rglob('*')
         if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
 if actual!=set(manifest):raise RuntimeError('Unexpected protected official files')


def export_small(run,export):
 """Read-only incremental export; no adapters, weights or compiler caches."""
 used=sum(p.stat().st_size for p in export.rglob('*') if p.is_file())
 excluded={'adapter','triton_cache','inductor_cache','preflight_triton','preflight_inductor','original_solution'}
 for base,dirs,files in os.walk(run):
  dirs[:]=[d for d in dirs if d not in excluded and not d.startswith('checkpoint-') and not (Path(base)/d).is_symlink()]
  for name in files:
   path=Path(base)/name;rel=path.relative_to(run)
   try:
    if not path.is_symlink() and path.suffix in ('.json','.jsonl','.log','.yaml'):
     if path.stat().st_size>=4*1024**2:
      EXPORT_OMITTED[str(rel)]='per_file_limit_4MiB';continue
     dest=export/'raw'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
     if not dest.exists() and sum(1 for p in export.rglob('*') if p.is_file())>=24:
      EXPORT_OMITTED[str(rel)]='persistent_file_limit_24';continue
     prior=dest.stat().st_size if dest.exists() else 0
     size=path.stat().st_size
     if used-prior+size>EXPORT_BUDGET:
      EXPORT_OMITTED[str(rel)]='aggregate_budget_56MiB';continue
     temp=dest.with_name(dest.name+'.copying');shutil.copyfile(path,temp);temp.replace(dest);used=used-prior+size
   except (FileNotFoundError,OSError):pass


class Exporter:
 def __init__(self,run,export):self.run=run;self.export=export;self.stop=threading.Event();self.thread=None
 def __enter__(self):
  def loop():
   while not self.stop.wait(1):export_small(self.run,self.export)
  self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
 def __exit__(self,*exc):self.stop.set();self.thread.join();export_small(self.run,self.export)


def idle_check(env,repo,run,lockfd):
 code='''import json,subprocess,sys
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
assert not p.stdout.strip(),'GPU processes still active after probe/preflight'
r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
rows=[line.split(',') for line in r.stdout.splitlines() if line.strip()]
assert rows and sum(int(x[1]) for x in rows)<1000,'GPU physical inventory not idle'
print(json.dumps({'idle':True,'physical_memory_rows':rows}))'''
 return process_guard.run([sys.executable,'-c',code],cwd=repo,env=env,log=run/'idle.log',timeout=20,lock_fd=lockfd)


def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=[MODE],required=True);a=p.parse_args()
 here=Path(__file__).resolve().parent
 if not os.environ.get('SOUP_REPO') or not os.environ.get('SOUP_RESULTS'):raise SystemExit('Prepared environment missing; no manual/network fallback')
 repo=Path(os.environ['SOUP_REPO']).absolute();results=Path(os.environ['SOUP_RESULTS']).absolute()
 if repo.is_symlink() or results.is_symlink() or repo==results:raise SystemExit('Real distinct prepared directories required')
 if not Path('/kaggle/working').is_dir():raise SystemExit('Prepared working directory missing')
 results.mkdir(parents=True,exist_ok=True);identity=MODE+'-'+uuid.uuid4().hex[:12]
 export=results/identity;export.mkdir(exist_ok=False);run=Path('/kaggle/working')/identity;run.mkdir(exist_ok=False)
 summary={'status':'PREPARING_NATIVE_ONE_STEP_DIAGNOSTIC','scope':'ACTUAL_BATCH_NATIVE_RETURN_MLP_SHADOW_DIAGNOSTIC_ONLY',
  'mode':MODE,'source_commit_reference':COMMIT,'complete_fixed_task':False,'complete_diagnostic':False,
  'optimizer_steps':0,'training_step_calls':None,'score':None,'admission':'UNKNOWN',
  'worker_timeout_s':80,'ephemeral_diagnostic_owner_in_child':True,'solution_installed':False,'training_loop_invoked':False,'public_evaluation_invoked':False}
 write(export/'status.json',summary);started=time.monotonic();lockfd=None
 try:
  if MARKER.exists():raise RuntimeError('GPU quarantine requires operator reconciliation')
  lockfd=os.open(LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if MARKER.exists():raise RuntimeError('GPU quarantine detected after lock')
  tree.prepare();tree.begin_quarantine(MARKER)
  official=read(here/'official_manifest.json');verify_official(repo,official)
  bundle=read(here/'worker_manifest.json');sources=bundle['sources']
  if set(sources)!={'batch_binding.json','batch_hashes.json','native_source_pins.json','profiler_helpers.py','solution/__init__.py','solution/backward_phase.py','solution/compact_mlp_install.py','solution/forward_observer.py','solution/mlp_half/__init__.py','solution/mlp_half/native_owner.py','solution/mlp_half/owned_half.py','solution/mlp_half/starter_half.py','solution/mlp_half/starter_swiglu.py','solution/mlp_observation_session.py','solution/row_adjoint.py','solution/trainer_shadow.py','worker.py'}:raise RuntimeError('Exact diagnostic source inventory required')
  archive=here/'diagnostic_sources.zip'
  if not archive.is_file() or archive.is_symlink() or sha(archive)!=bundle['zip_sha256']:raise RuntimeError('Frozen source archive changed')
  source_root=here/'diagnostic_source';source_root.mkdir(exist_ok=False)
  with zipfile.ZipFile(archive) as z:
   infos=z.infolist()
   if len(infos)!=len(sources) or {i.filename for i in infos}!=set(sources):raise RuntimeError('Diagnostic source archive inventory mismatch')
   if sum(i.file_size for i in infos)>8*1024**2:raise RuntimeError('Diagnostic source archive exceeds bound')
   for info in infos:
    if info.file_size>1024**2 or ((info.external_attr>>16)&0o170000)==0o120000:raise RuntimeError('Unsupported source archive member')
    raw=z.read(info)
    if hashlib.sha256(raw).hexdigest()!=sources[info.filename]:raise RuntimeError('Frozen source bytes changed: '+info.filename)
    target=source_root/info.filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  worker=source_root/'worker.py'
  write(run/'diagnostic_source_manifest.json',sources);summary['worker_sha256']=sources['worker.py']
  env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
  env.pop('SOUP_ROTARY_CODEGEN_DIR',None)
  with Exporter(run,export):
   probe=process_guard.run([sys.executable,str(here/'runtime_probe.py'),str(run/'runtime.json')],cwd=repo,env=env,log=run/'runtime.log',timeout=40,lock_fd=lockfd)
   if probe['exit_code'] or not read(run/'runtime.json').get('passed'):raise RuntimeError('Runtime gate failed')
   runtime=read(run/'runtime.json');summary['runtime']=runtime
   from prepare_asset_manifest import build
   assets,data=build(runtime['resolved_public_hub_cache'],repo)
   write(run/'model_cache_manifest.json',assets);write(run/'public_data_manifest.json',data)
   before=run/'idle_before';before.mkdir()
   idle=idle_check(env,repo,before,lockfd)
   if idle['exit_code']:raise RuntimeError('Pre-worker GPU idle gate failed')
   worker_env=dict(env,TRITON_CACHE_DIR=str(run/'preflight_triton'),TORCHINDUCTOR_CACHE_DIR=str(run/'preflight_inductor'))
   process=process_guard.run([sys.executable,'-B',str(worker),'--out',str(run/'profile.json'),'--repo',str(repo)],cwd=repo,env=worker_env,log=run/'worker.log',timeout=80,lock_fd=lockfd)
   summary['diagnostic_process']=process
   if process['exit_code'] or process.get('timed_out') or process.get('interrupted') or process.get('cleanup_confirmed') is not True:
    raise RuntimeError('One-step diagnostic process incomplete or censored')
   report=read(run/'profile.json')
   if (report.get('status')!=STATUS or report.get('scope')!='ACTUAL_BATCH_NATIVE_RETURN_MLP_SHADOW_DIAGNOSTIC_ONLY'
       or report.get('complete_fixed_task') is not False or report.get('score') is not None or report.get('admission')!='UNKNOWN'
       or type(report.get('optimizer_steps')) is not int or report['optimizer_steps']!=0
       or type(report.get('training_step_calls')) is not int or report['training_step_calls']!=1
       or report.get('native_backward_result_returned') is not True
       or report.get('range_certificate') is not False or report.get('production_clearance') is not False):
    raise RuntimeError('One-step diagnostic scope/status contract failed')
   after=run/'idle_after';after.mkdir()
   idle=idle_check(env,repo,after,lockfd)
   if idle['exit_code']:raise RuntimeError('Post-worker GPU idle gate failed')
   verify_official(repo,official)
   summary.update(status=STATUS,complete_diagnostic=True,training_step_calls=1,diagnostic_report='raw/profile.json')
 except BaseException as error:
  summary.update(status='BLOCKED_FAILED_OR_CENSORED_NATIVE_DIAGNOSTIC',complete_diagnostic=False,error=repr(error))
  if lockfd is not None:tree.quarantine('Native diagnostic runner failure: '+repr(error))
 finally:
  export_small(run,export)
  if EXPORT_OMITTED:
   summary.update(status='BLOCKED_FAILED_OR_CENSORED_NATIVE_DIAGNOSTIC',complete_diagnostic=False,error='Required bounded export incomplete')
  summary.update(orchestrator_wall_s=time.monotonic()-started,export_omitted=dict(EXPORT_OMITTED),
      output_budget_bytes=EXPORT_BUDGET,persistent_export_file_limit=24,score=None,admission='UNKNOWN',complete_fixed_task=False)
  write(export/'status.json',summary)
  if lockfd is not None:tree.finish_quarantine();os.close(lockfd)
 print(json.dumps({'export':str(export),'status':summary['status'],'complete_diagnostic':summary['complete_diagnostic'],'complete_fixed_task':False,'score':None,'admission':'UNKNOWN'}),flush=True)
 raise SystemExit(0 if summary['complete_diagnostic'] else 1)
if __name__=='__main__':main()
