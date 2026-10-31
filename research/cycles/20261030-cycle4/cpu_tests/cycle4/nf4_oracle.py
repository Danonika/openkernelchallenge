#!/usr/bin/env python3
"""Independent scalar/vector NF4 metadata decode plus bounded actual bnb CPU probes.
Format constants are data, not copied implementation. Actual CPU parity is not CUDA parity.
"""
import hashlib,importlib.metadata,json,math,os,struct,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F
torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(26103042)
ROOT=Path(__file__).resolve().parent;RESULTS=[]
NF4=[-1.,-.6961928009986877,-.5250730514526367,-.39491748809814453,-.28444138169288635,-.18477343022823334,-.09105003625154495,0.,.07958029955625534,.16093020141124725,.24611230194568634,.33791524171829224,.44070982933044434,.5626170039176941,.7229568362236023,1.]
FORMAT_SOURCE='https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/utils.py#L15-L34'
def f32(x):
    try:return struct.unpack('f',struct.pack('f',x))[0]
    except OverflowError:return math.copysign(float('inf'),x)
def f16(x):
    try:return struct.unpack('e',struct.pack('e',x))[0]
    except OverflowError:return math.copysign(float('inf'),x)
def compare(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    finite=torch.isfinite(a)&torch.isfinite(b)
    error=(a[finite].double()-b[finite].double()).abs()
    mismatch=(a!=b)&~(a.isnan()&b.isnan())
    return dict(different_elements=int(mismatch.sum()),total_elements=a.numel(),max_abs=error.max().item() if error.numel() else None,
                mismatch_indices=mismatch.nonzero().tolist()[:64])
def exact(a,b):
    c=compare(a,b);assert c['different_elements']==0,c;return c
def pack_codes(codes):
    c=codes.to(torch.uint8)
    if c.numel()%2:c=torch.cat((c,torch.tensor([15],dtype=torch.uint8)))
    return ((c[::2]<<4)|c[1::2]).view(-1,1)
def scalar_decode(packed,state,dtype=None,api_bc=True):
    # Independent scalar indexing and explicit binary32/16 rounding.
    raw=packed.view(torch.uint8).reshape(-1).tolist();n=math.prod(state.shape);outer=state.absmax.reshape(-1).tolist()
    values=[];outdtype=state.dtype if dtype is None else dtype
    for i in range(n):
        byte=raw[i//2];q=byte>>4 if i%2==0 else byte&15;j=i//state.blocksize
        if state.nested:
            code=float(state.state2.code[int(outer[j])]);scale2=float(state.state2.absmax[j//state.state2.blocksize])
            scale=f32(f32(code*scale2)+float(state.offset))
        else:scale=f32(float(outer[j]))
        value=f32(f32(NF4[q])*scale)
        values.append(f16(value) if outdtype==torch.float16 else value)
    out=torch.tensor(values,dtype=outdtype).reshape(state.shape)
    return out.t() if api_bc and packed.shape[0]==1 else out
def vector_decode(packed,state,dtype=None,api_bc=True):
    # Independent vector bit extraction and flattened block gather.
    raw=packed.view(torch.uint8).reshape(-1);n=math.prod(state.shape);i=torch.arange(n)
    q=torch.where(i%2==0,raw[i//2]>>4,raw[i//2]&15).long();j=i//state.blocksize
    if state.nested:
        blocks=torch.arange(state.absmax.numel())
        scales=state.state2.code[state.absmax.long()]*state.state2.absmax[blocks//state.state2.blocksize]
        scales=scales+state.offset
    else:scales=state.absmax.float()
    out=(torch.tensor(NF4,dtype=torch.float32)[q]*scales[j]).to(state.dtype if dtype is None else dtype).reshape(state.shape)
    return out.t() if api_bc and packed.shape[0]==1 else out
def constructed(shape,codebook,dtype=torch.float32,nested=True,outer_block=64,inner_block=256):
    n=math.prod(shape);blocks=math.ceil(n/outer_block)
    if nested:
        scales2=torch.linspace(.125,.25,math.ceil(blocks/inner_block))
        state2=F.QuantState(absmax=scales2,code=codebook,blocksize=inner_block,dtype=torch.float32)
        state=F.QuantState(absmax=(torch.arange(blocks)%256).to(torch.uint8),shape=torch.Size(shape),
                           code=torch.tensor(NF4),blocksize=outer_block,quant_type='nf4',dtype=dtype,offset=torch.tensor(.5),state2=state2)
    else:
        state=F.QuantState(absmax=torch.arange(1,blocks+1,dtype=torch.float32),shape=torch.Size(shape),
                           code=torch.tensor(NF4),blocksize=outer_block,quant_type='nf4',dtype=dtype)
    return pack_codes(torch.arange(n)%16),state
def run(name,fn):
    try:
        details=fn();r=dict(name=name,status='FAIL' if details.get('actual_cpu_parity_failures',0)>0 else 'PASS',details=details)
    except Exception as e:r=dict(name=name,status='FAIL',error=repr(e),traceback=traceback.format_exc())
    RESULTS.append(r);print(json.dumps(r,allow_nan=False),flush=True)

def exhaustive_metadata():
    actual=F.create_dynamic_map().cpu().float();assert actual.shape==(256,)
    assert torch.equal(F.get_4bit_type('nf4',device='cpu'),torch.tensor(NF4))
    bits=actual.view(torch.int32).tolist();raw=actual.numpy().tobytes()
    table=dict(origin='Public bitsandbytes.functional.create_dynamic_map on installed0.50.2',dtype='float32',
               uint32_bits=[x&0xffffffff for x in bits],sha256=hashlib.sha256(raw).hexdigest(),entries=256)
    (ROOT/'actual_dynamic_codebook_bits.json').write_text(json.dumps(table,indent=2)+'\n')
    synthetic=torch.linspace(-1,1,256).pow(3)
    cases=[];parity_failures=0
    for label,codebook in (('actual_default_dynamic',actual),('declared_synthetic_cubic',synthetic)):
        for dtype in (torch.float16,torch.float32):
            packed,state=constructed((3,5483),codebook,dtype)
            scalar=scalar_decode(packed,state);vector=vector_decode(packed,state);exact(scalar,vector)
            runtime=F.dequantize_4bit(packed,quant_state=state);parity=compare(runtime,vector)
            parity_failures+=parity['different_elements']>0
            cases.append(dict(codebook=label,dtype=str(dtype),shape=list(state.shape),scalar_vector_exact=True,actual_cpu_parity=parity))
    # Aligned even-last-dimension control complements the odd-tail generic path.
    packed_aligned,state_aligned=constructed((128,128),actual,torch.float16)
    aligned_expected=vector_decode(packed_aligned,state_aligned);exact(scalar_decode(packed_aligned,state_aligned),aligned_expected)
    aligned_parity=compare(F.dequantize_4bit(packed_aligned,quant_state=state_aligned),aligned_expected)
    parity_failures+=aligned_parity['different_elements']>0
    cases.append(dict(codebook='actual_default_dynamic',dtype='torch.float16',shape=[128,128],
                      aligned_native_decode_eligible=True,scalar_vector_exact=True,actual_cpu_parity=aligned_parity))
    # Arbitrary outer QuantState.code is ignored by NF4 dispatch; quant_type selects the canonical table.
    packed,state=constructed((3,65),actual);baseline=F.dequantize_4bit(packed,quant_state=state)
    state.code=torch.zeros(16);changed=F.dequantize_4bit(packed,quant_state=state);exact(changed,baseline)
    return dict(cases=cases,all_16_nf4_by_all_256_scale_indices=True,largest_elements=16449,
                flattened_boundaries=[63,64,16383,16384],odd_tail_spare_nibble_ignored=True,
                actual_default_table_sha256=table['sha256'],outer_code_field_not_nf4_selector=True,
                actual_cpu_parity_failures=parity_failures,
                decision='Independent decoders agree and exhaustive manually constructed QuantStates compare to installed CPU decode. Actual-table data and synthetic tables are distinguished; no CUDA or real-model state claim.')

def actual_quantstate_and_layouts():
    cases=[];parity_failures=0
    for shape in ((2,64),(3,65),(3,66)):
        for dtype in (torch.float16,torch.float32):
            x=torch.linspace(-1.3,1.7,math.prod(shape)).view(shape).to(dtype)
            packed,state=F.quantize_4bit(x,blocksize=64,compress_statistics=True,quant_type='nf4')
            native=F.dequantize_4bit(packed,quant_state=state);scalar=scalar_decode(packed,state);vector=vector_decode(packed,state)
            exact(scalar,vector);parity=compare(native,vector);parity_failures+=parity['different_elements']>0
            cases.append(dict(shape=list(shape),dtype=str(dtype),actual_quantizer_returned_state=True,nested=state.nested,
                              state2_blocksize=state.state2.blocksize,state2_dtype=str(state.state2.dtype),actual_cpu_parity=parity))
    codebook=F.create_dynamic_map();packed,state=constructed((3,5),codebook)
    layout_cases=[]
    for name,a in (('canonical_column',packed),('packed_row_bc',packed.T),('byte_reinterpreted_float_storage',packed.flatten().view(torch.float32).view(-1,1))):
        expected=vector_decode(a,state);exact(scalar_decode(a,state),expected)
        native=F.dequantize_4bit(a,quant_state=state);parity=compare(native,expected);parity_failures+=parity['different_elements']>0
        layout_cases.append(dict(layout=name,packed_shape=list(a.shape),packed_dtype=str(a.dtype),output_shape=list(native.shape),output_stride=list(native.stride()),actual_cpu_parity=parity))
    tiny,state=constructed((1,2),codebook);tiny_native=F.dequantize_4bit(tiny,quant_state=state)
    exact(tiny_native,vector_decode(tiny,state));assert tiny.shape==(1,1) and tiny_native.shape==(2,1)
    return dict(actual_state_cases=cases,layout_cases=layout_cases,
                single_byte_canonical_bc_ambiguity=dict(packed_shape=[1,1],state_shape=[1,2],actual_output_shape=[2,1]),
                actual_cpu_parity_failures=parity_failures,
                decision='Tiny actual nested CPU quantizer states and public decode layouts are checked. Byte reinterpretation is not numeric casting; single-byte canonical storage also triggers the API transpose quirk.')

def dtype_and_nested_metadata():
    actual=F.create_dynamic_map();packed,state=constructed((3,5),actual,torch.float16)
    direct32=vector_decode(packed,state,dtype=torch.float32)
    backward=F.dequantize_4bit(packed,quant_state=state).float()
    expected=scalar_decode(packed,state,dtype=torch.float16).float();exact(backward,expected)
    g=torch.tensor([[.2,.7,-.4],[1.,-.3,.9]])
    good=g@backward;illegal=g@direct32;difference=compare(illegal,good);assert difference['different_elements']>0
    a=torch.linspace(-.7,.8,10).view(2,5)
    forward_correct=a@direct32.T;forward_forced_state_dtype=a@backward.T
    forward_difference=compare(forward_forced_state_dtype,forward_correct);assert forward_difference['different_elements']>0
    # Generic decode metadata can carry an independent inner blocksize. This is NOT MatMul4Bit forward eligibility.
    p2,s2=constructed((3,65),actual,inner_block=64)
    vector=vector_decode(p2,s2);exact(scalar_decode(p2,s2),vector)
    runtime=F.dequantize_4bit(p2,quant_state=s2);inner_parity=compare(runtime,vector)
    return dict(backward_state_half_then_grad_float_vs_illegal_direct_float=difference,
                forward_requested_float_vs_illegal_state_half=forward_difference,
                generic_inner_block64_cpu_parity=inner_parity,matmul_forward_inner_block256_requirement_not_executed=True,
                actual_cpu_parity_failures=int(inner_parity['different_elements']>0),
                decision='Forward requested dtype and backward QuantState-dtype-then-grad-dtype are distinct contracts. Generic dequant inner-block metadata is not MatMul4Bit forward eligibility; no optional CPU GEMM package installed.')

def native_row_alignment_limit():
    cases=[];failures=0;outputs={}
    for shape in ((3,64),(3,65),(3,66),(3,80),(1,240)):
        n=math.prod(shape);packed=torch.full(((n+1)//2,1),255,dtype=torch.uint8)
        scales=torch.arange(1,math.ceil(n/64)+1,dtype=torch.float32)
        state=F.QuantState(absmax=scales,shape=torch.Size(shape),code=torch.tensor(NF4),blocksize=64,quant_type='nf4',dtype=torch.float32)
        scalar=scalar_decode(packed,state);vector=vector_decode(packed,state);exact(scalar,vector)
        native=F.dequantize_4bit(packed,quant_state=state);parity=compare(native,vector)
        failures+=parity['different_elements']>0;outputs[shape]=native
        cases.append(dict(shape=list(shape),all_nf4_codes_are_positive_one=True,blocksize=64,scales=scales.tolist(),
                          actual_cpu_parity=parity,case_status='FAIL' if parity['different_elements'] else 'PASS'))
    same_bytes_shape_contrast=compare(outputs[(3,80)],outputs[(1,240)].reshape(3,80))
    return dict(cases=cases,same_bytes_shape_only_contrast=same_bytes_shape_contrast,actual_cpu_parity_failures=failures,
                expected_cpu_alignment_hazard='Pinned CPU AVX512 row scale indexing lacks dim1%blocksize guard; actual dispatch observed via public CPU operator only, no instruction-level profiling.',
                decision='Retain any [3,80] parity failure as a CPU-backend oracle limitation. Do not redefine the flat-format oracle to match it, infer CUDA failure, or install/fix unrelated kernels.')

run('nf4_exhaustive_independent_metadata_and_cpu_decode',exhaustive_metadata)
run('actual_nested_quantstate_and_public_layouts',actual_quantstate_and_layouts)
run('forward_backward_dtype_and_inner_block_contracts',dtype_and_nested_metadata)
run('actual_cpu_row_alignment_parity_probe',native_row_alignment_limit)
payload=dict(cycle=4,lane='nf4',generated_utc=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,bitsandbytes=bnb.__version__,
             threads=[torch.get_num_threads(),torch.get_num_interop_threads()],eager_policy=os.environ.get('TORCHDYNAMO_DISABLE'),
             nf4_format_source=FORMAT_SOURCE,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             summary=dict(passed=sum(r['status']=='PASS' for r in RESULTS),failed=sum(r['status']=='FAIL' for r in RESULTS)),
             candidates=['SOUP-NEW-003','SOUP-NEW-007'],
             not_validated=['CUDA decode/quantizer/GEMM parity','real model QuantState','optional CPU GEMM path','frozen-scale cache invalidation','actual GPU FMA behavior','training/performance or graph points'],results=RESULTS)
(ROOT/'nf4_results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
print(json.dumps(payload['summary']));sys.exit(bool(payload['summary']['failed']))
