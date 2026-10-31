"""One sparse witness, values and stages only; no timing or validation claim."""
import argparse,hashlib,json,os,traceback
from pathlib import Path
OBSERVATIONS={}


def run():
    if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Target hardware required')
    import torch,triton,bitsandbytes as bnb
    from bitsandbytes import functional as F
    from . import guarded_forward as g
    from .fixtures import EXPECTED,call,exact,engineering_close
    from .operand_probe import operand_probe
    if (torch.__version__!='2.11.0+cu128' or triton.__version__!='3.6.0' or bnb.__version__!='0.50.2'
            or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='Tesla T4'
            or torch.cuda.get_device_capability(0)!=(7,5)):
        raise RuntimeError('Exact target required')
    def describe(x):
        cpu=x.detach().cpu().contiguous();raw=cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        values=cpu.reshape(-1).float();uniques,counts=torch.unique(values,return_counts=True)
        result={'shape':list(x.shape),'dtype':str(x.dtype),'sha256_raw_bytes':hashlib.sha256(raw).hexdigest(),
                'first_values':values[:8].tolist(),'unique_values':uniques[:32].tolist(),'unique_counts':counts[:32].tolist(),
                'total_unique_values':uniques.numel()}
        if cpu.dtype==torch.float16:result['first_fp16_u16_bits']=(cpu.reshape(-1)[:8].view(torch.int16).int()&65535).tolist()
        return result
    def check(fn,*args):
        try:
            result=fn(*args)
            return {'passed':True,'details':result}
        except BaseException as exc:return {'passed':False,'error':repr(exc)}
    torch.manual_seed(991)
    w=torch.ones(256,1536,device='cuda',dtype=torch.float16)
    p,s=F.quantize_4bit(w,blocksize=64,compress_statistics=True,quant_type='nf4')
    a=torch.zeros(65,1536,device='cuda',dtype=torch.float16);a[:,0]=1.;a[:,1]=.0004
    b=torch.full((256,),.0002,device='cuda',dtype=torch.float16)
    decoded=F.dequantize_4bit(p,s);observed=torch.empty_like(decoded)
    operand_probe[(triton.cdiv(decoded.numel(),256),)](p,s.absmax,s.state2.code,s.state2.absmax,s.offset,observed,decoded.numel(),256,num_warps=4,enable_fp_fusion=False)
    OBSERVATIONS.update(inputs={'activation_nonzero_prefix':describe(a[0,:2]),'bias':describe(b),
        'native_decoded_weight':describe(decoded),'observed_decoded_weight':describe(observed),
        'operand_exact':check(exact,observed,decoded)},
        environment={'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            'allow_fp16_accumulation':getattr(torch.backends.cuda.matmul,'allow_fp16_accumulation',None),
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32},
        fixture={'M':65,'N':256,'K':1536,'plain_tensor_bias':True,'actual_Linear4bit_module':False})
    exact(observed,decoded)
    native_before=call((a,p,s,None));native_after=call((a,p,s,b))
    native_explicit=torch.nn.functional.linear(a,decoded,b)
    with g.temporary_native_fallback(EXPECTED):
        candidate_before=call((a,p,s,None));candidate_after=call((a,p,s,b))
    OBSERVATIONS['route']={'counts':dict(g.COUNTS),'fallback_reasons':dict(g.FALLBACK_REASONS),
        'input_type':type(a).__name__,'packed_type':type(p).__name__,'bias_type':type(b).__name__}
    if g.COUNTS['custom']!=2:raise RuntimeError('Diagnostic candidate inactive')
    fp32_before=torch.nn.functional.linear(a.float(),decoded.float())
    fp32_after=fp32_before+b.float()
    fp32_native_bias=torch.nn.functional.linear(a.float(),decoded.float(),b.float())
    correct_stage=fp32_after.half()
    incorrect_stage=fp32_before.half()+b
    native_postround=native_before+b
    outputs={'native_materialized_half_without_bias':native_before,'native_with_bias':native_after,
        'native_explicit_Flinear_with_bias':native_explicit,'candidate_materialized_half_without_bias':candidate_before,
        'candidate_with_bias':candidate_after,'fp32_reference_without_bias':fp32_before,
        'fp32_reference_plus_fp32_bias':fp32_after,'fp32_native_Flinear_with_bias':fp32_native_bias,
        'fp32_acc_plus_bias_then_half':correct_stage,'half_acc_then_half_bias':incorrect_stage,
        'native_materialized_half_plus_half_bias':native_postround}
    OBSERVATIONS['outputs']={name:describe(value) for name,value in outputs.items()}
    OBSERVATIONS['comparisons']={
        'candidate_vs_native_fixed_engineering':check(engineering_close,candidate_after,native_after),
        'candidate_vs_native_exact':check(exact,candidate_after,native_after),
        'candidate_matches_fp32_bias_before_half':check(exact,candidate_after,correct_stage),
        'candidate_before_matches_fp32_reference_then_half':check(exact,candidate_before,fp32_before.half()),
        'native_matches_fp32_bias_before_half':check(exact,native_after,correct_stage),
        'native_matches_materialized_postround_bias':check(exact,native_after,native_postround),
        'native_dispatch_matches_explicit_Flinear':check(exact,native_after,native_explicit),
        'fp32_addition_matches_fp32_Flinear_bias':check(exact,fp32_after,fp32_native_bias),
        'staged_bias_negative_witness_nonvacuous':not torch.equal(correct_stage,incorrect_stage)}
    codegen=[]
    for key,c in g.COMPILED.items():
        ptx=c.asm['ptx'];ir=c.asm['ttgir'];layouts=[line for line in ir.splitlines() if '#ttg.blocked<' in line]
        valid=('.target sm_75' in ptx and 'mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32' in ptx
            and 'fma.rn.f32' not in ptx and '.ftz' not in ptx and len(layouts)==1
            and 'sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]' in layouts[0])
        codegen.append({'key':list(key),'same_original_MMA_layout_gate':valid,'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest()})
        if not valid:raise RuntimeError('Diagnostic original MMA codegen gate failed')
    OBSERVATIONS['codegen']=codegen
    torch.cuda.synchronize()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    report={'status':'FAILED_DIAGNOSTIC_ONLY','scope':'DIAGNOSTIC_ONLY','candidate_validated':False,
        'training_executed':False,'timing_executed':False,'score':None,'admission':'UNKNOWN'}
    try:run();report['status']='COMPLETED_DIAGNOSTIC_ONLY'
    except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
    report['observations']=OBSERVATIONS
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'candidate_validated':False}),flush=True)
    raise SystemExit(report['status']!='COMPLETED_DIAGNOSTIC_ONLY')
if __name__=='__main__':main()
