"""SOUP_REPO/SOUP_RESULTS adapter for reviewed trusted web-job notebooks.

No credentials, network calls, installs, job submission or task/source rewrite.
Uses the advertised 80-second per-native-run budget; never defeats external caps.
"""
import argparse,fcntl,hashlib,json,math,os,shutil,sys,threading,time,uuid,zipfile
from pathlib import Path
import process_guard
from evidence_reader import events
import owned_process_tree as tree

NATIVE_TIMEOUT=80
COMMIT='c7c2fcd15972ce8b898d8d3ff732ee192c817ceb'
LOCK=Path('/tmp/soup-gpu0.lock');MARKER=Path('/tmp/soup-gpu0.lock.blocked')
FUSIONS=json.loads((Path(__file__).resolve().parent/'candidate_profile.json').read_text())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 p=Path(p);pending=p.with_name(p.name+'.pending');pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(p)
def finite(x):return type(x) in (float,int) and math.isfinite(x)


def verify_official(repo,manifest):
 if not repo.is_dir() or repo.is_symlink():raise RuntimeError('SOUP_REPO must be a real prepared job directory')
 for name,wanted in manifest.items():
  p=repo/name
  if not p.is_file() or p.is_symlink() or sha(p)!=wanted:raise RuntimeError('Protected official source mismatch: '+name)
 actual={str(p.relative_to(repo)) for folder in ('harness','task','tests') for p in (repo/folder).rglob('*')
         if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
 if actual!=set(manifest):raise RuntimeError('Unexpected protected official files')
 # Identity is exact prepared bytes; .git is deliberately not required.


def install_bundle(repo,bundle,manifest,backup):
 if sha(bundle)!=FUSIONS[bundle.stem]['zip_sha256']:raise RuntimeError('Unexpected candidate ZIP identity')
 with zipfile.ZipFile(bundle) as z:
  if set(z.namelist())!=set(manifest):raise RuntimeError('Bundle member mismatch')
  for name,h in manifest.items():
   parts=Path(name).parts
   if not name.startswith('solution/') or '..' in parts or Path(name).is_absolute():raise RuntimeError('Unsafe source path')
   raw=z.read(name)
   if hashlib.sha256(raw).hexdigest()!=h:raise RuntimeError('Source hash mismatch: '+name)
   if name.endswith('.py'):compile(raw,name,'exec')
  # Only the prepared editable job's solution/ changes. Retain its prior bytes locally.
  if (repo/'solution').exists():shutil.move(str(repo/'solution'),str(backup))
  for name in manifest:
   target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))


EXPORT_BUDGET=56*1024**2  # room for atomic-copy temporaries/status below 64 MiB
EXPORT_OMITTED={}

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


def output_problems(row):
 """Required fixed-task outputs, not just exit/step counters, for baseline too."""
 problems=[]
 config=row.get('adapter_config');tensors=row.get('adapter_tensors')
 if not isinstance(config,dict) or not config or config.get('r')!=16 or config.get('lora_alpha')!=32 or config.get('base_model_name_or_path')!='Qwen/Qwen2.5-1.5B-Instruct':
  problems.append('adapter_config_missing_or_wrong')
 if not isinstance(tensors,dict) or not tensors or not all(isinstance(v,list) and len(v)==2 and isinstance(v[0],str) and isinstance(v[1],list) and v[1] and all(type(n) is int and n>0 for n in v[1]) for v in tensors.values()):
  problems.append('adapter_tensor_header_missing_or_malformed')
 metrics=('loss','num_tokens','entropy','mean_token_accuracy','grad_norm')
 logged=row.get('logged_metrics')
 if not isinstance(logged,list) or not set(metrics)<=set(logged):problems.append('logged_metrics_missing')
 log=row.get('log')
 for key in metrics:
  values=log.get(key) if isinstance(log,dict) else None
  if not isinstance(values,list) or len(values)!=64 or not all(finite(x) for x in values):problems.append('metric_series_missing_or_nonfinite:'+key)
 rewind=row.get('rewind')
 if not isinstance(rewind,list) or len(rewind)!=64 or not all(isinstance(batch,list) and len(batch)==3 and all(isinstance(v,list) and len(v)==4 for v in batch) and all(type(x) is int and x>=0 for x in batch[0]) and all(finite(x) and x>0 for x in batch[1]) and all(finite(x) for x in batch[2]) for batch in (rewind or [])):
  problems.append('rewind_missing_or_malformed')
 trainer=(row.get('metrics') or {}).get('trainer')
 expected={'fp16':True,'bf16':False,'per_device_train_batch_size':4,'gradient_accumulation_steps':1,'seed':1,'max_length':512,'torch_compile':False,'gradient_checkpointing':False}
 if not isinstance(trainer,dict) or not trainer or any(trainer.get(k)!=v for k,v in expected.items()):problems.append('trainer_contract_missing_or_wrong')
 return problems


def normalize(result,mode,process,logs):
 """Exploratory one-solution public screen; never a paired or official score."""
 required={'solution':1}
 errors=[];observations=[]
 if process.get('exit_code')!=0 or process.get('timed_out') or process.get('interrupted') or not process.get('cleanup_confirmed'):errors.append('outer_process_incomplete')
 if result.get('packages_changed')!=[]:errors.append('packages_changed_or_unrecorded')
 task=result.get('task',{})
 if type(task.get('seed')) is not int or task['seed']!=1 or type(task.get('repeats')) is not int or task['repeats']!=1 or task.get('only')!='solution':errors.append('task_identity')
 if set(result.get('evals',{}))!={'solution'} or set(result.get('summary',{}))!={'solution'} or result.get('profile_run') is not None:errors.append('unexpected_eval_summary_or_profile')
 if set(result.get('runs',{}))!=set(required):errors.append('unexpected_or_missing_run_sides')
 for side,count in required.items():
  rows=result.get('runs',{}).get(side,[])
  if len(rows)!=count:errors.append(side+':repeat_count')
  if [r.get('run') for r in rows]!=[side+'_'+str(i+1) for i in range(count)] or any(r.get('side')!=side for r in rows):errors.append(side+':run_identity')
  ev=result.get('evals',{}).get(side,{})
  if (ev.get('exit_code')!=0 or not finite(ev.get('mean_loss')) or ev['mean_loss']<=0
      or type(ev.get('rows')) is not int or ev['rows']!=256
      or type(ev.get('tokens')) is not int or ev['tokens']<=0
      or ev.get('eval_file')!=task.get('eval_file')):errors.append(side+':public_eval_incomplete')
  for row in rows:
   m=row.get('metrics') or {};name=row.get('run','unknown')
   if ev.get('adapter')!=row.get('adapter') or not isinstance(row.get('adapter'),str):errors.append(name+':eval_adapter_identity')
   errors.extend(name+':'+problem for problem in output_problems(row))
   observations.append({'run':name,'exit_code':row.get('exit_code'),'observed_wall_s':row.get('wall_s'),'observed_peak_mib':row.get('gpu_peak_mib'),'steps':m.get('steps'),'scope':'raw observation; complete status decided separately'})
   if row.get('exit_code')!=0 or m.get('exit_code')!=0 or m.get('steps')!=64 or row.get('global_step')!=64:errors.append(name+':native_incomplete')
   if not finite(row.get('wall_s')) or row['wall_s']<=0 or not finite(row.get('gpu_peak_mib')) or row['gpu_peak_mib']<=0:errors.append(name+':measurement_missing')
   if m.get('device_count')!=1 or m.get('cuda_device')!='Tesla T4':errors.append(name+':device_contract')
   if side=='solution':
    text=logs.get(name,'');own=m.get('own_kernel_ms') or {}
    if m.get('timed_steps')!=62 or not finite(m.get('timed_loop_s')) or m['timed_loop_s']<=0:errors.append(name+':timed_window')
    profile=FUSIONS[mode]
    if profile['requires_original_kernels']:
     if not own or not any(finite(v) and v>0 for v in own.values()):errors.append(name+':owned_kernel_timing_missing')
     for kernel in profile['kernels']:
      if not finite(own.get(kernel)) or own[kernel]<=0:errors.append(name+':kernel_missing:'+kernel)
    for rule in profile['routes']:
     seen=False
     for record in events(text,rule['prefix']):
      if 'event' in rule and record.get('event')!=rule['event']:continue
      if any(type(record.get(k)) is not type(v) or record.get(k)!=v for k,v in rule.get('match',{}).items()):continue
      if 'positive_field' in rule and (not finite(record.get(rule['positive_field'])) or record[rule['positive_field']]<=0):continue
      seen=True;break
     if not seen:errors.append(name+':route_missing:'+rule['prefix'])
 complete=not errors
 return {'status':'SINGLE_SOLUTION_COMPLETE_PUBLIC_SCREEN' if complete else 'FAILED_PARTIAL_OR_CENSORED',
         'complete_fixed_task':complete,'complete_single_screen':complete,'errors':errors,'observations':observations,
         'scope':'EXPLORATORY_SINGLE_SOLUTION_PUBLIC_EVAL_NOT_PAIRED',
         'paired_comparison_complete':False,'paired_ratio':None,'official_checks':'NOT_RUN_SINGLE_SCREEN',
         'complete_measurements':result.get('summary') if complete else None,
         'score':None,'admission':'UNKNOWN','official_kaggle_admission':'NOT_ESTABLISHED'}


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
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=list(FUSIONS),required=True);a=p.parse_args()
 here=Path(__file__).resolve().parent
 # Environment names come verbatim from the user's deployed template.
 if not os.environ.get('SOUP_REPO') or not os.environ.get('SOUP_RESULTS'):raise SystemExit('Missing deployed SOUP_REPO/SOUP_RESULTS; no manual-path or network fallback')
 repo=Path(os.environ['SOUP_REPO']).absolute();results=Path(os.environ['SOUP_RESULTS']).absolute()
 if repo.is_symlink() or results.is_symlink() or repo==results:raise SystemExit('Job checkout/export paths must be distinct real directories')
 if not Path('/kaggle/working').is_dir():raise SystemExit('Deployed /kaggle/working missing')
 results.mkdir(parents=True,exist_ok=True)
 identity=a.mode+'-'+uuid.uuid4().hex[:12];export=results/identity;export.mkdir(exist_ok=False)
 run=Path('/kaggle/working')/identity;run.mkdir(exist_ok=False)
 summary={'status':'PREPARING','mode':a.mode,'source_commit_reference':COMMIT,'native_timeout_requested_s':NATIVE_TIMEOUT,'external_cap_advertised_s':80,'external_cap_semantics_verified':False,'runtime_api_observed_by_parent':True,'runtime_api_is_device_execution_proof':False,'score':None,'admission':'UNKNOWN','complete_fixed_task':False}
 write(export/'status.json',summary)
 lockfd=None;started=time.monotonic()
 try:
  if MARKER.exists():raise RuntimeError('GPU quarantine requires operator reconciliation')
  lockfd=os.open(LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if MARKER.exists():raise RuntimeError('GPU quarantine detected after lock')
  tree.prepare();tree.begin_quarantine(MARKER)
  manifest=read(here/'official_manifest.json');verify_official(repo,manifest)
  env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
  # Preserve service PATH and enforcement environment; never replace its wrappers.
  env.pop('SOUP_ROTARY_CODEGEN_DIR',None)
  with Exporter(run,export):
   probe=process_guard.run([sys.executable,str(here/'runtime_probe.py'),str(run/'runtime.json')],cwd=repo,env=env,log=run/'runtime.log',timeout=40,lock_fd=lockfd)
   if probe['exit_code'] or not read(run/'runtime.json').get('passed'):raise RuntimeError('Deployed runtime gate failed')
   runtime=read(run/'runtime.json');summary['runtime']=runtime
   # Prepared public cache is located by the installed HF runtime, never guessed/downloaded.
   from prepare_asset_manifest import build
   assets,data=build(runtime['resolved_public_hub_cache'],repo);write(run/'model_cache_manifest.json',assets);write(run/'public_data_manifest.json',data)
   if a.mode!='baseline':
    bundle=here/(a.mode+'.zip');source_manifest=read(here/(a.mode+'_manifest.json'))
    install_bundle(repo,bundle,source_manifest,run/'original_solution')
    write(run/'candidate_source_manifest.json',source_manifest);summary['source_sha256']=sha(bundle)
    preenv=dict(env,TRITON_CACHE_DIR=str(run/'preflight_triton'),TORCHINDUCTOR_CACHE_DIR=str(run/'preflight_inductor'))
    pre=process_guard.run([sys.executable,'-B','-m',FUSIONS[a.mode]['preflight'],'--out',str(run/'preflight.json')],cwd=repo,env=preenv,log=run/'preflight.log',timeout=80,lock_fd=lockfd)
    if pre['exit_code'] or read(run/'preflight.json').get('status')!=FUSIONS[a.mode]['preflight_status']:raise RuntimeError('Candidate preflight incomplete; no training started')
   idle=idle_check(env,repo,run,lockfd)
   if idle['exit_code']:raise RuntimeError('GPU idle gate failed')
   verify_official(repo,manifest)
   out=run/'official_output'
   command=[sys.executable,'-B',str(repo/'harness/run_task.py'),'--repeats','1','--seed','1','--timeout','80','--out',str(out)]
   command+=['--only','solution']
   write(run/'requested_command.json',{'argv':command,'single_training_hard_cap_s':80,'cap_bypass':False})
   summary['status']='RUNNING';write(export/'status.json',summary)
   # Single-screen supervisor covers one separately capped train and public eval;
   # it does not extend any native training run or the service's external limit.
   process=process_guard.run(command,cwd=repo,env=env,log=run/'harness.log',timeout=300,lock_fd=lockfd)
   raw=read(out/'result.json') if (out/'result.json').is_file() else {}
   logs={r.get('run',''): (out/r.get('run','')/'train.log').read_text(errors='replace') for rows in raw.get('runs',{}).values() for r in rows if (out/r.get('run','')/'train.log').is_file()}
   summary.update(normalize(raw,a.mode,process,logs));summary['process']=process
   # Official paired checks require a base; iteration screen does not fabricate one.
   summary['official_checks']='NOT_RUN_SINGLE_SCREEN'
   summary['paired_comparison_complete']=False;summary['paired_ratio']=None
   verify_official(repo,manifest)
   if not summary['complete_fixed_task']:tree.quarantine('Incomplete/censored web screen; operator reconciliation before retry')
 except BaseException as exc:
  summary.update(status='BLOCKED_FAILED_OR_CENSORED',error=repr(exc),complete_fixed_task=False,complete_measurements=None)
  if lockfd is not None:tree.quarantine('Web runner failure: '+repr(exc))
 finally:
  summary['orchestrator_wall_s']=time.monotonic()-started
  summary['score']=None;summary['admission']='UNKNOWN'
  summary['complete_single_screen']=bool(summary['complete_fixed_task'])
  summary['scope']='EXPLORATORY_SINGLE_SOLUTION_PUBLIC_EVAL_NOT_PAIRED'
  summary['paired_comparison_complete']=False;summary['paired_ratio']=None
  summary['official_checks']='NOT_RUN_SINGLE_SCREEN'
  export_small(run,export);summary['export_omitted']=dict(EXPORT_OMITTED);summary['output_budget_bytes']=EXPORT_BUDGET;write(export/'status.json',summary)
  if lockfd is not None:
   tree.finish_quarantine();os.close(lockfd)
 print(json.dumps({'export':str(export),'status':summary['status'],'complete_fixed_task':summary['complete_fixed_task'],'score':None,'admission':'UNKNOWN'},indent=2))
 raise SystemExit(0 if summary['complete_fixed_task'] else 1)
if __name__=='__main__':main()
