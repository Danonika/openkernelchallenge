"""Real native Linear4bit/Parameter activation; synthetic standalone inputs only.

This target-only gate follows the source-bound revised MMA arithmetic gate and
runs before any timing. No pretrained model, task activation or training claim.
"""
import argparse,hashlib,inspect,json,os,traceback
from pathlib import Path


PROGRESS={}

def run():
    if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Target hardware required')
    import torch,triton,bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit,Params4bit
    from bitsandbytes.autograd._functions import MatMul4Bit
    from . import guarded_forward as g
    from .fixtures import EXPECTED,exact,engineering_close
    if (torch.__version__!='2.11.0+cu128' or triton.__version__!='3.6.0' or bnb.__version__!='0.50.2'
            or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='Tesla T4'
            or torch.cuda.get_device_capability(0)!=(7,5)):
        raise RuntimeError('Exact target runtime required')
    from .guarded_forward import verify_native_owner
    module_forward=Linear4bit.forward;autograd_forward=MatMul4Bit.forward
    expected_owners={
        'linear4bit_forward':('bitsandbytes.nn.modules','Linear4bit.forward','d52cf63457717cc718ca0a449dad0967171e679be90d5d2adcafe2e8a141093d'),
        'matmul4bit_forward':('bitsandbytes.autograd._functions','MatMul4Bit.forward','29f4e21a4fe99db3f6e0f94dbf6cbe8426220f9163c0501c86d33f362a54ccb6'),
    }
    for key,fn in (('linear4bit_forward',module_forward),('matmul4bit_forward',autograd_forward)):
        owner,qualname,digest=expected_owners[key]
        if (not inspect.isfunction(fn) or hasattr(fn,'__wrapped__') or (fn.__module__,fn.__qualname__)!=(owner,qualname)
                or hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()!=digest):
            raise RuntimeError('Native module forward source/owner mismatch: '+key)
    backward=MatMul4Bit.backward;rows=[];PROGRESS['rows']=rows
    counts_before=dict(g.COUNTS)
    for n in (256,1536):
        torch.manual_seed(78+n)
        module=Linear4bit(1536,n,bias=True,compute_dtype=torch.float16,compress_statistics=True,quant_type='nf4')
        source=torch.randn(n,1536,dtype=torch.float16)*.1
        module.weight=Params4bit(source,requires_grad=False,compress_statistics=True,quant_type='nf4')
        module.bias=torch.nn.Parameter(torch.randn(n,dtype=torch.float16)*.03,requires_grad=False)
        module=module.to('cuda');module.train()
        if type(module.bias) is not torch.nn.Parameter or module.bias.dtype!=torch.float16:
            raise RuntimeError('Actual native Parameter fixture not established')
        weight_id=id(module.weight);bias_id=id(module.bias)
        base=torch.randn(4,17,1536,device='cuda',dtype=torch.float16)*.1
        packed_before=module.weight.detach().clone();bias_before=module.bias.detach().clone()
        for trainable_bias in (False,True):
            module.bias.requires_grad_(trainable_bias)
            for scale in (1.,128.,65536.):
                PROGRESS['current_case']={'N':n,'bias_trainable':trainable_bias,'scale':scale}
                def one(custom):
                    a=base.detach().clone().requires_grad_()
                    before=torch.cuda.get_rng_state().clone()
                    if custom:
                        with g.temporary_native_fallback(EXPECTED):output=module(a)
                    else:output=module(a)
                    cotangent=torch.linspace(-.01,.01,output.numel(),device='cuda').reshape_as(output).half()*scale
                    parameters=(a,module.bias) if trainable_bias else (a,)
                    gradients=torch.autograd.grad(output,parameters,cotangent)
                    if not torch.equal(before,torch.cuda.get_rng_state()):raise RuntimeError('RNG changed')
                    return output.detach(),gradients
                expected,reference_grad=one(False);count=g.COUNTS['custom'];actual,actual_grad=one(True)
                if g.COUNTS['custom']!=count+1:
                    raise RuntimeError('Actual Linear4bit module inactive: '+repr(g.FALLBACK_REASONS))
                error=engineering_close(actual,expected)
                for x,y in zip(actual_grad,reference_grad):exact(x,y)
                if (id(module.weight)!=weight_id or id(module.bias)!=bias_id
                        or MatMul4Bit.backward is not backward or type(module.bias) is not torch.nn.Parameter
                        or Linear4bit.forward is not module_forward or MatMul4Bit.forward is not autograd_forward):
                    raise RuntimeError('Native Parameter/backward identity changed')
                exact(module.weight.detach(),packed_before);exact(module.bias.detach(),bias_before)
                rows.append({'N':n,'K':1536,'shape':[4,17,1536],'bias_trainable':trainable_bias,'scale':scale,
                    'actual_native_Linear4bit_forward_active':True,'native_backward_gradients_exact':True,
                    'input_kind':'synthetic standalone module, not captured fixed-task q/v activations',
                    'weight_and_bias_unchanged':True,**error})
    torch.cuda.synchronize()
    if len(rows)!=12 or g.COUNTS['custom']-counts_before['custom']!=12:raise RuntimeError('Incomplete actual module activation gate')
    return {'rows':rows,'counts':dict(g.COUNTS),'counts_before':counts_before,'source_owners':expected_owners,
            'fallback_reasons':dict(g.FALLBACK_REASONS),
            'fixed_task_training':False,'task_quality':'UNKNOWN'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    report={'status':'FAILED_MODULE_PREFLIGHT','scope':'MICROBENCH_ONLY','score':None,'admission':'UNKNOWN'}
    try:report['checks']=run();report['status']='PASSED_MODULE_PREFLIGHT_ONLY'
    except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc(),progress=PROGRESS)
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    raise SystemExit(report['status']!='PASSED_MODULE_PREFLIGHT_ONLY')
if __name__=='__main__':main()
