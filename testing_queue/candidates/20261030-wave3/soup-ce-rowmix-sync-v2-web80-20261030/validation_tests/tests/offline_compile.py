"""CPU-host compile of the new original mixed adjoint; no device launch or binary export."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.pop('TRITON_INTERPRET',None)
from pathlib import Path
import hashlib,json,tempfile
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from solution.mixed_rows import soup_ce_mixed_backward
root=Path(__file__).resolve().parents[1]
source=ASTSource(soup_ce_mixed_backward,signature={
    'Z':'*fp16','Target':'*i64','Maximum':'*fp32','LogSum':'*fp32',
    'Q':'*fp32','NativeRow':'*i32','Grad':'*fp16'},
    constexprs={'V':151936,'BLOCK':2048})
with tempfile.TemporaryDirectory(prefix='ce-mixed-offline-') as cache:
    os.environ['TRITON_CACHE_DIR']=cache
    compiled=triton.compile(source,target=GPUTarget('cuda',75,32),
                           options={'num_warps':4,'enable_fp_fusion':False})
    ptx=compiled.asm['ptx']
    assert '.target sm_75' in ptx
    assert 'ex2.approx' in ptx and 'st.global.b16' in ptx
    assert compiled.metadata.shared==0
    report={'status':'OFFLINE_SM75_COMPILE_PASS','local_compiler':triton.__version__,
            'target_compiler':'3.6.0','target_compiler_executed':False,
            'gpu_executed':False,'binaries_shipped':False,'shared_bytes':compiled.metadata.shared,
            'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
            'limits':'Triton3.9 CPU-host compiler only; not target3.6 or numerical/performance proof.'}
(root/'validation/offline_compile_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
