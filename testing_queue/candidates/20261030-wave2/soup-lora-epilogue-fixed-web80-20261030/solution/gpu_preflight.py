"""Explicit target-only standalone LoRA gate; not run during preparation.

Actual pinned PEFT/bitsandbytes q/v shapes, native dropout/GEMM/clone and scaled
VJPs; two input dtypes. No optimizer, model training or timing/share claim.
"""
import argparse
import json
from pathlib import Path
import traceback
import torch
from . import patch, epilogue


def exact(actual,expected,label):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype,label+': metadata'
    assert torch.equal(actual.isnan(),expected.isnan()),label+': NaN'
    assert torch.equal(actual.isposinf(),expected.isposinf()),label+': +Inf'
    assert torch.equal(actual.isneginf(),expected.isneginf()),label+': -Inf'
    finite=actual.isfinite()&expected.isfinite()
    assert torch.equal(actual[finite],expected[finite]),label+': finite values'
    zeros=finite&(actual==0)&(expected==0)
    assert torch.equal(actual.signbit()[zeros],expected.signbit()[zeros]),label+': signed zero'


def run(report):
    import bitsandbytes as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4'
    assert torch.cuda.get_device_capability(0)==(7,5)
    native_forward,native_backward=Linear4bit.forward,MatMul4Bit.backward
    patch.install()
    installed=Linear4bit.forward;patch.install();assert Linear4bit.forward is installed
    torch.manual_seed(110731)
    config=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,init_lora_weights=False,
                      target_modules=['q_proj','v_proj'],bias='none')
    layers=[]
    for width in (1536,256):
        base=bnb.nn.Linear4bit(1536,width,bias=False,compute_dtype=torch.float16,
                              compress_statistics=True,quant_type='nf4')
        base.weight=bnb.nn.Params4bit(torch.randn(width,1536,dtype=torch.float32)*.02,
                                     requires_grad=False,compress_statistics=True,
                                     quant_type='nf4',blocksize=64)
        layer=Linear4bit(base.cuda(),'default',config=config,r=16,lora_alpha=32).cuda().train()
        assert layer.lora_A['default'].weight.dtype==torch.float32
        assert layer.lora_B['default'].weight.dtype==torch.float32
        assert layer.lora_dropout['default'].p==.05
        layers.append(layer)
    initial={f'{i}:{name}':value.detach().clone() for i,layer in enumerate(layers) for name,value in layer.named_parameters()}
    results=[]
    for dtype in (torch.float16,torch.float32):
        data=torch.randn(4,89,1536,device='cuda',dtype=dtype)
        rng_cpu,rng_cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
        before=epilogue._COUNTS['launch_submitted']
        def evaluate(candidate):
            for layer in layers:layer.zero_grad(set_to_none=True)
            x=data.clone().requires_grad_(True);masks=[]
            hooks=[layer.lora_dropout['default'].register_forward_hook(
                lambda module,args,out:masks.append((out!=0).clone())) for layer in layers]
            try:
                with torch.autocast('cuda',dtype=torch.float16):
                    outputs=[layer(x) if candidate else native_forward(layer,x) for layer in layers]
                    loss=sum(y.float().square().mean() for y in outputs)
                loss.backward(torch.tensor(65536.,device='cuda',dtype=torch.float32))
                return {'outputs':[y.detach().clone() for y in outputs],'loss':loss.detach().clone(),
                        'input_grad':x.grad.clone(),'grads':[p.grad.clone() for layer in layers for p in layer.parameters() if p.requires_grad],
                        'masks':masks,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
            finally:
                for hook in hooks:hook.remove()
        reference=evaluate(False)
        torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda)
        candidate=evaluate(True)
        for a,b in zip(candidate['outputs'],reference['outputs']):exact(a,b,'output')
        exact(candidate['loss'],reference['loss'],'loss')
        exact(candidate['input_grad'],reference['input_grad'],'scaled input gradient')
        for a,b in zip(candidate['grads'],reference['grads']):exact(a,b,'scaled adapter gradient')
        for a,b in zip(candidate['masks'],reference['masks']):assert torch.equal(a,b)
        assert not torch.equal(candidate['masks'][0],candidate['masks'][1])
        assert torch.equal(candidate['cpu_rng'],reference['cpu_rng'])
        assert torch.equal(candidate['cuda_rng'],reference['cuda_rng'])
        assert epilogue._COUNTS['launch_submitted']-before==2
        results.append({'input_dtype':str(dtype),'M':356,'qv_outputs_scaled_VJPs_rng_exact':True,
                        'own_epilogue_launches':2,'native_dropout_separate':True})
        del reference,candidate,data
    for i,layer in enumerate(layers):
        for name,value in layer.named_parameters():
            assert torch.equal(value.detach().view(torch.uint8),initial[f'{i}:{name}'].view(torch.uint8))
    assert MatMul4Bit.backward is native_backward
    patch.uninstall();assert Linear4bit.forward is native_forward
    report.update(status='GPU_PREFLIGHT_PASSED',cases=results,native_backward_identity=True,
                  native_parameters_unchanged=True,source_ownership_restored=True,
                  runtime_counts=dict(epilogue._COUNTS))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report={'status':'RUNNING','score':None,'admission':'UNKNOWN',
            'scope':'Bounded standalone PEFT/NF4 LoRA fixture, not full-model training or useful-share proof'}
    try:run(report)
    except Exception as exc:report.update(status='GPU_PREFLIGHT_FAILED',error=repr(exc),traceback=traceback.format_exc())
    finally:patch.uninstall()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'out':str(a.out)}),flush=True)
    return 0 if report['status']=='GPU_PREFLIGHT_PASSED' else 1
if __name__=='__main__':raise SystemExit(main())
