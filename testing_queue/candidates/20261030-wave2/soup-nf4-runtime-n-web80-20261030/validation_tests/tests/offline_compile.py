"""CPU-host SM75 offline compile: no driver initialization or launch, no shipped cubin."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.pop('TRITON_INTERPRET',None)
import importlib.util,importlib.metadata,json,tempfile,time,hashlib,re,ast
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
old=module('old_nf4_packed',Path('/workspace/shared/soup_web_wave2_20261030/nf4/solution/packed_nf4.py')).soup_nf4_blockpair_decode
new=module('new_nf4_packed',ROOT/'solution/packed_nf4.py').soup_nf4_blockpair_runtime_n
assert 'N' in new.do_not_specialize
assert [p.name for p in new.params if p.is_constexpr]==['SCALES','PACK_HALF']
rows=[];source_hashes={}
with tempfile.TemporaryDirectory(prefix='nf4-runtime-offline-') as cache:
 os.environ['TRITON_CACHE_DIR']=cache
 for variant,fn in [('old_constexpr_N',old),('new_runtime_N',new)]:
  source_hashes[variant]={}
  for dtype,packed in [('fp16',True),('fp32',False),('fp16',False)]:
   hashes=[]
   for n in [393216,2359296,13762560]:
    sig={'PACKED':'*u8','QABS':'*u8','CODE2':'*fp32','ABS2':'*fp32','OFFSET':'*fp32','OUT':'*'+dtype}
    const={'SCALES':16,'PACK_HALF':packed}
    if variant=='old_constexpr_N':const['N']=n
    else:sig['N']='i32'
    src=ASTSource(fn,signature=sig,constexprs=const);digest=src.hash();hashes.append(digest)
    if any(row['source_hash']==digest for row in rows):continue
    begin=time.perf_counter();compiled=triton.compile(src,target=GPUTarget('cuda',75,32),options={'num_warps':4,'enable_fp_fusion':False});elapsed=time.perf_counter()-begin
    ptx=compiled.asm['ptx'];mul=bool(re.search(r'\bmul(?:\.[a-z]+)*\.f32\b',ptx));add=bool(re.search(r'\badd(?:\.[a-z]+)*\.f32\b',ptx));fma=bool(re.search(r'\bfma(?:\.[a-z]+)*\.f32\b',ptx))
    assert compiled.metadata.shared==0 and mul and add and not fma
    rows.append({'variant':variant,'dtype':dtype,'packed_half':packed,'n':n if variant=='old_constexpr_N' else 'runtime_i32','source_hash':digest,'host_compile_s':elapsed,'shared_bytes':compiled.metadata.shared,'separate_f32_mul_add':mul and add,'fma':fma,'ptx_store_b32': 'st.global.b32' in ptx,'ptx_bytes':len(ptx),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest()})
   source_hashes[variant][dtype+('_packed' if packed else '_scalar')]=hashes
for hashes in source_hashes['new_runtime_N'].values():assert len(set(hashes))==1
for hashes in source_hashes['old_constexpr_N'].values():assert len(set(hashes))==3
report={'status':'OFFLINE_SM75_COMPILE_PASS','compiler':importlib.metadata.version('triton'),'target_compiler':'3.6.0','gpu_executed':False,'target_device_executed':False,'binaries_shipped':False,'production_aligned_half_plus_fp32_signature_count':{'old':6,'new':2},'counts_are_possible_keys_not_observed_server_compiles':True,'source_hashes':source_hashes,'rows':rows,'host_compile_sums':{v:sum(row['host_compile_s'] for row in rows if row['variant']==v and not(row['dtype']=='fp16' and not row['packed_half'])) for v in source_hashes},'limits':'Local3.9 host compile costs are not target3.6 costs, GPU timings, complete task time or admission. Pointer alignment/dtype changes can still create keys.'}
(ROOT/'validation/offline_compile_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
