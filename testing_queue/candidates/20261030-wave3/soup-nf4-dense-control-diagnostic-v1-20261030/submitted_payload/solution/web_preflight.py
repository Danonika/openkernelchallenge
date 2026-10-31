"""Target-only original lane MMA and unchanged native-backward preflight."""
import argparse,hashlib,json,os,traceback
from pathlib import Path

PROGRESS = {}

def run():
    if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Interpreter is not a device gate')
    import torch,triton,bitsandbytes as bnb
    from bitsandbytes import functional as F
    from bitsandbytes.autograd._functions import MatMul4Bit
    from . import guarded_forward as g
    from .fixtures import EXPECTED,SHAPES,fixture,call,exact,engineering_close
    from .operand_probe import operand_probe
    from .reference_forward import original_reference_forward
    from .materialize import materialize_native_w16,verify_materialize_owners
    verify_materialize_owners()
    if (torch.__version__!='2.11.0+cu128' or triton.__version__!='3.6.0'
            or bnb.__version__!='0.50.2' or torch.cuda.device_count()!=1
            or torch.cuda.get_device_name(0)!='Tesla T4' or torch.cuda.get_device_capability(0)!=(7,5)):
        raise RuntimeError('Exact target runtime required')
    rows=[];PROGRESS["forward_rows"]=rows;backward=MatMul4Bit.backward
    for m,n in SHAPES:
        for bias in (False,True):
            PROGRESS.update(stage="forward_operand",case={"M":m,"N":n,"bias":bias},completed_forward_rows=len(rows))
            data=fixture(m,n,bias);a,p,s,b=data
            expected=call(data)
            w=F.dequantize_4bit(p,s);observed=torch.empty_like(w)
            operand_probe[(triton.cdiv(w.numel(),256),)](p,s.absmax,s.state2.code,s.state2.absmax,s.offset,observed,w.numel(),256,
                num_warps=4,enable_fp_fusion=False)
            exact(observed,w)
            materialized=materialize_native_w16(a,p,s.shape,s.state2.absmax,64,'nf4',s.absmax,s.state2.code,s.offset)
            exact(materialized,w)
            PROGRESS.update(stage="candidate_forward",operand_fp16_exact_for_current_case=True)
            before=g.COUNTS['custom']
            with g.temporary_native_fallback(EXPECTED):actual=call(data)
            PROGRESS.update(counts=dict(g.COUNTS),fallback_reasons=dict(g.FALLBACK_REASONS))
            if g.COUNTS['custom']!=before+1:raise RuntimeError('Candidate fallback inactive; reasons='+repr(g.FALLBACK_REASONS))
            original=original_reference_forward(data)
            exact(actual,original)
            error=engineering_close(actual,expected)
            rows.append({'M':m,'N':n,'K':1536,'bias':bias,'operand_fp16_exact':True,'frozen_original_MMA_output_exact':True,'fresh_native_materialization_exact':True,**error})
    PROGRESS.update(stage="codegen",completed_forward_rows=len(rows))
    codegen=[];PROGRESS["codegen"]=codegen
    for key,c in g.COMPILED.items():
        ptx=c.asm['ptx'];ir=c.asm['ttgir']
        layouts=[line for line in ir.splitlines() if '#ttg.blocked<' in line]
        expected_layout='sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]'
        if ('.target sm_75' not in ptx or 'mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32' not in ptx
                or ptx.count('mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32')!=8
                or 'fma.rn.f32' in ptx or '.ftz' in ptx
                or len(layouts)!=1 or expected_layout not in layouts[0]):
            raise RuntimeError('SM75 lane layout/MMA/separate-scale codegen gate failed')
        if c.n_spills!=0 or c.metadata.shared!=0:raise RuntimeError('Unexpected spills/shared-memory resource gate')
        codegen.append({'registers_per_thread':c.n_regs,'spills':c.n_spills,'shared_bytes':c.metadata.shared,
            'mma_static_instructions':8,'key':list(key),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
            'ttgir_sha256':hashlib.sha256(ir.encode()).hexdigest(),'mma':True,'no_scalar_fma_or_ftz':True,'lane_layout_verified':True})
    PROGRESS.update(stage="native_backward")
    gradients=[];PROGRESS["gradients"]=gradients
    for n in (256,1536):
        data=fixture(65,n,True)
        for scale in (1.,128.,65536.):
            def one(custom):
                a,p,s,b=data;a=a.detach().clone().requires_grad_();b=b.detach().clone().requires_grad_()
                current=(a,p,s,b)
                if custom:
                    with g.temporary_native_fallback(EXPECTED):out=call(current)
                else:out=call(current)
                # Fixed cotangent independent of either numerical forward output.
                up=torch.linspace(-.01,.01,out.numel(),device='cuda').reshape_as(out).to(out.dtype)*scale
                ga,gb=torch.autograd.grad(out,(a,b),up)
                return ga,gb
            reference=one(False);before=g.COUNTS['custom'];actual=one(True)
            if g.COUNTS['custom']!=before+1:raise RuntimeError('Native autograd forward bypassed candidate')
            for x,y in zip(actual,reference):exact(x,y)
            gradients.append({'N':n,'scale':scale,'native_input_and_bias_gradients_exact':True})
    # Nonfinite/zero-class checks cannot be replaced by tolerances.
    PROGRESS.update(stage="classes_and_bias_witness")
    classes=[];PROGRESS["class_fixtures"]=classes
    for mode in ('zero','nan','posinf','neginf'):
        data=list(fixture(65,256,False));data[0].zero_()
        if mode!='zero':data[0][:,0]=dict(nan=float('nan'),posinf=float('inf'),neginf=-float('inf'))[mode]
        expected=call(data)
        with g.temporary_native_fallback(EXPECTED):actual=call(data)
        if mode=='zero':exact(actual,expected)
        else:engineering_close(actual,expected)
        classes.append(mode)
    # Candidate arithmetic-stage witness. Native CUDA epilogue is not assumed.
    PROGRESS.update(stage="candidate_bias_stage")
    w=torch.ones(256,1536,device='cuda',dtype=torch.float16)
    p,s=F.quantize_4bit(w,blocksize=64,compress_statistics=True,quant_type='nf4')
    a=torch.zeros(65,1536,device='cuda',dtype=torch.float16);a[:,0]=1.;a[:,1]=.0004
    b=torch.full((256,),.0002,device='cuda',dtype=torch.float16);data=(a,p,s,b)
    expected=call(data)
    with g.temporary_native_fallback(EXPECTED):actual=call(data)
    from .bias_stage_gate import verify_bias_stage
    decoded=F.dequantize_4bit(p,s)
    fp32_reference=torch.nn.functional.linear(a.float(),decoded.float())
    native_without_bias=call((a,p,s,None))
    bias_stage=verify_bias_stage(actual,expected,fp32_reference,b,native_without_bias)
    bias_stage['source_bound_reason_for_revised_gate']=json.loads((Path(__file__).with_name('bias_gate_provenance.json')).read_text())
    PROGRESS['bias_stage']=bias_stage
    # Real native modules/Parameters must activate before timing is allowed.
    from .module_preflight import run as module_run, PROGRESS as MODULE_PROGRESS
    PROGRESS.update(stage="actual_native_Linear4bit",module_progress=MODULE_PROGRESS)
    module_checks=module_run()
    PROGRESS['module_checks']=module_checks
    if MatMul4Bit.backward is not backward:raise RuntimeError('Native backward identity changed')
    torch.cuda.synchronize()
    return {'rows':rows,'codegen':codegen,'gradients':gradients,'class_fixtures':classes,
        'bias_stage':bias_stage,'module_checks':module_checks,'counts':dict(g.COUNTS),
        'native_backward_identity_preserved':True,'task_quality':'UNKNOWN'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    report={'status':'FAILED_PREFLIGHT_ONLY','scope':'MICROBENCH_ONLY','training_executed':False,'score':None,'admission':'UNKNOWN'}
    try:report['checks']=run();report['status']='PASSED_PREFLIGHT_ONLY'
    except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc(),progress=dict(PROGRESS))
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'output':args.out}),flush=True)
    raise SystemExit(report['status']!='PASSED_PREFLIGHT_ONLY')
if __name__=='__main__':main()
