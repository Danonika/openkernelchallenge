"""Local3.9 offline compiler only, no driver/GPU and no shipped binaries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.pop('TRITON_INTERPRET',None)
import sys,json,hashlib,tempfile,re
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import torch,triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from prototype.kernel_original import soup_experimental_inline_nf4_gemm,soup_experimental_nf4_operand_probe
assert not torch.cuda.is_initialized()
rows=[]
with tempfile.TemporaryDirectory(prefix='nf4-fused-compile-') as cache:
 os.environ['TRITON_CACHE_DIR']=cache
 for n,k in [(1536,1536),(256,1536)]:
  sig={'A':'*fp16','PACKED':'*u8','QABS':'*u8','CODE2':'*fp32','ABS2':'*fp32','OFFSET':'*fp32','OUT':'*fp16','M':'i32'}
  src=ASTSource(soup_experimental_inline_nf4_gemm,signature=sig,constexprs={'N':n,'K':k,'BM':64,'BN':64,'BK':64})
  compiled=triton.compile(src,target=GPUTarget('cuda',75,32),options={'num_warps':4,'num_stages':2,'enable_fp_fusion':False})
  ptx=compiled.asm['ptx']
  assert '.target sm_75' in ptx
  assert 'mma.sync' not in ptx and 'wmma' not in ptx
  assert 'fma.rn.f32' in ptx
  rows.append({'shape':[n,k],'shared_bytes':compiled.metadata.shared,'ptx_bytes':len(ptx),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'tensorcore_sm75_mma':False,'fma_rn_f32_instruction_count':ptx.count('fma.rn.f32'),'metadata':str(compiled.metadata)})
assert not torch.cuda.is_initialized()
report={'status':'COMPILES_BUT_TENSORCORE_FEASIBILITY_BLOCKED_FMA_ONLY','source_sha256':hashlib.sha256((root/'prototype/kernel_original.py').read_bytes()).hexdigest(),'local_triton':triton.__version__,'local_torch':torch.__version__,'target_triton':'3.6.0','target_primary_source':'https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp','target_source_lines':[344,350],'target_source_observation':'BlockedToMMA rejects compute capability below80 and selects FMA path','target_compiler_executed':False,'gpu_executed':False,'binaries_shipped':False,'rows':rows}
(root/'validation/offline_original.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
