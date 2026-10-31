import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.pop('TRITON_INTERPRET',None)
import sys,json,hashlib,tempfile,re
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import torch,triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from solution.kernel import soup_nf4_mma_laneprobe
assert not torch.cuda.is_initialized()
rows=[]
with tempfile.TemporaryDirectory(prefix='nf4-mma-offline-') as cache:
 os.environ['TRITON_CACHE_DIR']=cache
 for bias in (False,True):
  src=ASTSource(soup_nf4_mma_laneprobe,signature={'A':'*fp16','PACKED':'*u8','QABS':'*u8','CODE2':'*fp32','ABS2':'*fp32','OFFSET':'*fp32','Bias':'*fp16','Out':'*fp16','M':'i32'},constexprs={'N':1536,'K':1536,'BIAS':bias})
  c=triton.compile(src,target=GPUTarget('cuda',75,32),options={'num_warps':1,'num_stages':1,'enable_fp_fusion':False})
  ptx=c.asm['ptx'];ttgir=c.asm['ttgir']
  print('LAYOUT',*[x for x in ttgir.splitlines() if x.startswith('#')],sep='\n')
  assert 'mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32' in ptx and '.target sm_75' in ptx
  assert 'fma.rn.f32' not in ptx
  assert 'sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]' in ttgir
  rows.append({'bias':bias,'shared_bytes':c.metadata.shared,'mma_static_instructions':ptx.count('mma.sync.aligned.m16n8k8'),'no_scalar_fma':True,'ptx_bytes':len(ptx),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'lane_layout':'one scalar per lane,32lanes,1warp','ttgir_sha256':hashlib.sha256(ttgir.encode()).hexdigest()})
assert not torch.cuda.is_initialized()
report={'status':'LOCAL_OFFLINE_MMA_LANEPROBE_PASS','source_sha256':hashlib.sha256((root/'solution/kernel.py').read_bytes()).hexdigest(),'local_triton':triton.__version__,'target_triton':'3.6.0','target_compiler_executed':False,'gpu_executed':False,'binaries_shipped':False,'rows':rows}
(root/'validation/offline_mma.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
