"""Read-only, short-lived deployed-template gate. Never prints secret values."""
import importlib.metadata as metadata,json,os,platform,subprocess,sys
from pathlib import Path
EXPECTED={'torch':'2.11.0+cu128','triton':'3.6.0','numpy':'2.3.5','soup-cli':'0.77.0','transformers':'5.16.1','peft':'0.20.0','trl':'0.29.1','bitsandbytes':'0.50.2','accelerate':'1.14.0','datasets':'4.8.5'}

def main():
 out=Path(sys.argv[1]);r={'passed':False,'gpu_workload_executed':False,'score':None,'admission':'UNKNOWN'}
 try:
  versions={k:metadata.version(k) for k in EXPECTED}
  r.update(python=platform.python_version(),packages=versions,uid=os.getuid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
  assert platform.python_version()=='3.13.15','Python pin mismatch'
  assert versions==EXPECTED,'Exact package pins mismatch'
  assert os.getuid()==10001,'Deployed runtime uid mismatch'
  assert len(os.sched_getaffinity(0))==4,'Four assigned CPUs required'
  for p,wanted in [('memory.max',str(30*1024**3)),('memory.swap.max','0'),('cpuset.cpus.effective','0-3')]:
   actual=(Path('/sys/fs/cgroup')/p).read_text().strip();r[p]=actual
   assert actual==wanted,'Deployed cgroup mismatch: '+p
  assert not Path('/var/run/docker.sock').exists(),'Unexpected Docker socket'
  assert not Path('/etc/soup-queue/worker.env').exists(),'Unexpected worker environment file'
  assert not any('TOKEN' in k and k!='TOKENIZERS_PARALLELISM' for k in os.environ),'Token-bearing environment rejected without reading its value'
  assert not os.environ.get('TRITON_INTERPRET'),'Interpreter mode forbidden'
  import torch,triton,numpy
  assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8'
  assert triton.__version__=='3.6.0' and numpy.__version__=='2.3.5'
  assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4'
  assert torch.cuda.get_device_capability(0)==(7,5)
  r.update(gpu_name=torch.cuda.get_device_name(0),cuda_build=torch.version.cuda,torch_git=torch.version.git_version)
  from huggingface_hub.constants import HF_HUB_CACHE
  r['resolved_public_hub_cache']=str(Path(HF_HUB_CACHE).resolve())
  r['passed']=True
 except BaseException as e:r['error']=repr(e)
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
 raise SystemExit(0 if r['passed'] else 1)
if __name__=='__main__':main()
