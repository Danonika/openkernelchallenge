#!/usr/bin/env python3
"""Fail-closed package compatibility check; GPU inventory only with explicit --gpu."""
import argparse,hashlib,importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path
EXPECTED={'torch':'2.11.0+cu128','triton':'3.6.0','soup-cli':'0.77.0','transformers':'5.16.1','peft':'0.20.0','trl':'0.29.1','bitsandbytes':'0.50.2','accelerate':'1.14.0','datasets':'4.8.5'}
COMMIT='c7c2fcd15972ce8b898d8d3ff732ee192c817ceb'
def inspect(gpu=False,repo=None,require_four_cpus=False):
 errors=[];versions={}
 for name,wanted in EXPECTED.items():
  try:actual=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:actual=None
  versions[name]=actual
  if actual!=wanted:errors.append(f'{name}: expected {wanted}, observed {actual}')
 if sys.version_info[:3]!=(3,13,15):errors.append('Exact Python3.13.15 required')
 affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
 if require_four_cpus and (affinity is None or len(affinity)!=4):errors.append('Process affinity must be exactly four logical CPUs; this does not equate CPU hardware')
 out={'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),'packages':versions,'expected_packages':EXPECTED,'cpu_affinity':affinity,'gpu_checked':gpu,'official_commit_expected':COMMIT,'errors':errors,'equivalent_hardware_claim':False}
 if repo is not None:
  root=Path(repo).resolve();r=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
  out['repo_commit']=r.stdout.strip()
  if r.returncode or out['repo_commit']!=COMMIT:errors.append('Official repository commit mismatch')
  r=subprocess.run(['git','-C',str(root),'status','--porcelain','--untracked-files=all','--','harness','task','tests'],capture_output=True,text=True);out['frozen_paths_status']=r.stdout
  if r.returncode or r.stdout:errors.append('Official harness/task/tests must be unchanged, including untracked files')
 if gpu:
  try:
   import torch
   out['cuda_version']=torch.version.cuda;out['torch_git_version']=torch.version.git_version
   if torch.version.cuda!='12.8':errors.append('Torch compiled CUDA12.8 required')
   out['visible_device_count']=torch.cuda.device_count();out['visible_device_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
   if out['visible_device_count']!=1 or out['visible_device_names']!=['Tesla T4']:errors.append('Exactly one visible Tesla T4 required')
   r=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=10);out['nvidia_smi_inventory']=r.stdout.strip()
   if r.returncode:errors.append('nvidia-smi inventory failed; native memory sampling unavailable')
  except Exception as e:errors.append('GPU inventory failed: '+type(e).__name__)
 out['passed']=not errors;return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--repo');p.add_argument('--require-four-cpus',action='store_true');a=p.parse_args();result=inspect(a.gpu,a.repo,a.require_four_cpus);Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'errors':result['errors'],'report':a.out}));raise SystemExit(not result['passed'])
