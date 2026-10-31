import argparse,fcntl,hashlib,json,math,os,shutil,sys,threading,time,uuid,zipfile

from pathlib import Path

import process_guard

import owned_process_tree as tree

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

EXPORT_BUDGET=56*1024**2

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

def idle_check(env,repo,run,lockfd):
 code='''import json,subprocess,sys
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
assert not p.stdout.strip(),'GPU processes still active after probe/preflight'
r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
rows=[line.split(',') for line in r.stdout.splitlines() if line.strip()]
assert rows and sum(int(x[1]) for x in rows)<1000,'GPU physical inventory not idle'
print(json.dumps({'idle':True,'physical_memory_rows':rows}))'''
 return process_guard.run([sys.executable,'-c',code],cwd=repo,env=env,log=run/'idle.log',timeout=20,lock_fd=lockfd)
