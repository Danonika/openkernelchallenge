"""Bounded component timing, never full training or model-memory measurement."""
import argparse,hashlib,inspect,json,os,statistics,time,traceback
from pathlib import Path
TS=(133,235,317,485,512);GROUPS=5;REPETITIONS=8;WARMUP=3
VARIANTS={'native':'expanded_qwen_add_norm','standalone':'historical_original_trial001_norm_after_native_add','fused':'original_two_output_residual_norm'}
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=tuple(VARIANTS),required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
 report={'scope':'MICROBENCH_ONLY','status':'FAILED_MICROBENCH_ONLY','method':a.method,'method_variants':VARIANTS,'training_executed':False,'complete_fixed_task':False,'score':None,'admission':'UNKNOWN','rows':[],
 'measurement_contract':{'cold':'Synchronized first component call in private initially empty compiler cache; includes allocation/dispatch/compile/device work, not pure compile time.','warm':'CUDA-event stream intervals include Python/autograd dispatch gaps; not kernel sums.','order':'Fixed native/standalone/fused order, no counterbalanced performance claim.','cache':'Fresh child per method; cache reused across shapes only inside child.','both_outputs':'FP32 norm output and rounded FP32 residual; both cotangents participate.','groups':GROUPS,'repetitions':REPETITIONS}}
 write(out,report)
 try:
  assert not os.environ.get('TRITON_INTERPRET');cache=Path(os.environ['TRITON_CACHE_DIR']);assert cache.is_dir() and not any(cache.iterdir())
  import torch,triton,transformers
  from transformers.models.qwen2 import modeling_qwen2 as qwen
  from solution.residual_norm import fused_residual_norm
  from solution.contracts import fixture,native,engineering_close
  from solution.web_preflight import EXPECTED_QWEN_BLOB
  from old_solution.rmsnorm import frozen_rms32
  assert torch.__version__=='2.11.0+cu128' and triton.__version__=='3.6.0' and transformers.__version__=='5.16.1'
  assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4' and torch.cuda.get_device_capability(0)==(7,5)
  raw=Path(inspect.getfile(qwen)).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==EXPECTED_QWEN_BLOB
  def standalone(x,b,w):z=x+b;return frozen_rms32(z,w),z
  fn={'native':native,'standalone':standalone,'fused':fused_residual_norm}[a.method]
  def wall(call):
   torch.cuda.synchronize();start=time.perf_counter();value=call();torch.cuda.synchronize();return value,time.perf_counter()-start
  def warm(call):
   for _ in range(WARMUP):call()
   torch.cuda.synchronize();times=[]
   for _ in range(GROUPS):
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
    for _ in range(REPETITIONS):call()
    end.record();end.synchronize();times.append(begin.elapsed_time(end)/REPETITIONS)
   assert all(x>0 for x in times)
   return {'cuda_event_ms_per_call':times,'median_ms':statistics.median(times)}
  scratch=torch.ones(16,device='cuda');scratch.add_(1);torch.cuda.synchronize();del scratch
  for t in TS:
   x,b,w=fixture(4*t,device='cuda');dy=torch.full_like(x,128.);dz=torch.full_like(x,-64.);rng=torch.cuda.get_rng_state().clone()
   y,forward=wall(lambda:fn(x,b,w));backward=lambda:torch.autograd.grad(y,(x,b),(dy,dz),retain_graph=True)
   grads,back=wall(backward);expected=native(x,b,w);eg=torch.autograd.grad(expected,(x,b),(dy,dz))
   ferrors=[engineering_close(g,e,forward=True) for g,e in zip(y,expected)];gerrors=[engineering_close(g,e,scale=128.) for g,e in zip(grads,eg)]
   assert torch.equal(y[1],expected[1]);assert torch.equal(rng,torch.cuda.get_rng_state())
   row={'B':4,'T':t,'H':1536,'dtype':'float32','frozen_weight':True,'both_output_gradients':True,'residual_exact':True,'forward_errors':ferrors,'gradient_errors':gerrors,'cold_first_call_wall_s':{'forward':forward,'backward':back}}
   row['warm_forward']=warm(lambda:fn(x,b,w));row['warm_backward']=warm(backward);assert torch.equal(rng,torch.cuda.get_rng_state())
   report['rows'].append(row);write(out,report)
   del y,grads,expected,eg,backward,x,b,w,dy,dz
  report.update(status='PASSED_MICROBENCH_ONLY',gpu_executed=True,native_source_blob=EXPECTED_QWEN_BLOB)
 except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
 write(out,report);print(json.dumps({'method':a.method,'status':report['status'],'rows':len(report['rows'])}));raise SystemExit(report['status']!='PASSED_MICROBENCH_ONLY')
if __name__=='__main__':main()
