"""SM75 compilation only. Never launches or queries a GPU; artifacts stay local."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from solution.residual_norm import residual_norm_f32,residual_norm_adjoint_mixed
rows=[]
for b in ('fp16','fp32'):
 for keep in (False,True):
  sig={x:'*fp32' for x in ('A','W','Z','Y','R')};sig['B']='*'+b
  c=triton.compile(ASTSource(residual_norm_f32,signature=sig,constexprs={'H':1536,'EPS':1e-6,'KEEP':keep,'BLOCK':2048}),target=GPUTarget('cuda',75,32),options={'num_warps':4,'enable_fp_fusion':False})
  assert '.target sm_75' in c.asm['ptx'];rows.append({'kind':'forward','b':b,'keep':keep,'ptx_sha256':hashlib.sha256(c.asm['ptx'].encode()).hexdigest()})
 for needs in ((True,True),(True,False),(False,True)):
  sig={x:'*fp32' for x in ('DY','DZ','Z','W','R','DA')};sig['DB']='*'+b
  c=triton.compile(ASTSource(residual_norm_adjoint_mixed,signature=sig,constexprs={'H':1536,'HAS_BYPASS':True,'NEED_A':needs[0],'NEED_B':needs[1],'BLOCK':2048}),target=GPUTarget('cuda',75,32),options={'num_warps':4,'enable_fp_fusion':False})
  assert '.target sm_75' in c.asm['ptx'];rows.append({'kind':'backward','b':b,'needs':list(needs),'ptx_sha256':hashlib.sha256(c.asm['ptx'].encode()).hexdigest()})
out=Path(__file__).resolve().parents[1]/'validation/offline_compile.json';out.write_text(json.dumps({'scope':'OFFLINE_COMPILE_ONLY','gpu_executed':False,'triton':triton.__version__,'variants':rows},indent=2)+'\n');print('Offline SM75 variants:',len(rows))
