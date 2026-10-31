"""Bounded actual-vocabulary target gate for the isolated original CE tile API.

Not a projection recomputation proof: the streaming owner separately verifies
repeated full-H/V GEMM bits and complete hidden gradients/GradScaler behavior.
"""
import argparse,json,os,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from . import ce_tile as tile
from .original import materialized_ce as ce


def exact(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise RuntimeError('Shape or dtype changed')
    if not torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8)):
        raise RuntimeError('Frozen original CE gradient bits changed')


def metric_error(actual,expected,rtol):
    for f in (torch.isnan,torch.isposinf,torch.isneginf):
        if not torch.equal(f(actual),f(expected)):raise RuntimeError('Native metric nonfinite class changed')
    finite=actual.isfinite()&expected.isfinite()
    torch.testing.assert_close(actual[finite],expected[finite],rtol=rtol,atol=1e-6)
    return {'max_abs':float((actual[finite]-expected[finite]).abs().max()) if finite.any() else 0.,
            'rtol':rtol,'atol':1e-6,'native_nonfinite_classes_exact':True}


def run_cases(device,progress):
    v=tile.VOCABULARY;generator=torch.Generator(device=device).manual_seed(911)
    base=torch.randn(4,3,v,device=device,generator=generator).half()
    labels=torch.tensor([[0,1,2],[0,3,4],[0,-100,5],[0,6,7]],device=device)
    rows=[];progress['rows']=rows
    for mode in ('all_safe','mixed','all_near','all_ignored','ignored_nan'):
        z=base.clone();y=labels.clone();target=ce._targets(y).flatten()
        if mode=='all_safe':z.zero_()
        elif mode in ('mixed','all_near'):
            chosen=[0,4] if mode=='mixed' else [0,1,3,4,7,9,10]
            for index in chosen:
                z.view(-1,v)[index].zero_();z.view(-1,v)[index,target[index]]=30.
        elif mode=='all_ignored':y.fill_(-100);target=ce._targets(y).flatten()
        else:z[0,2,0]=float('nan')
        for scale in (1.,65536.):
            progress['current_case']={'mode':mode,'scale':scale}
            original=z.clone().requires_grad_()
            loss=ce.materialized_causal_loss(original,y,13)
            wanted,=torch.autograd.grad(loss,original,torch.tensor(scale,device=device))
            result=tile.forward_tile(z.reshape(-1,v),target)
            with torch.no_grad():
                upstream=tile.prepare_upstream(torch.tensor(scale,device=device)/13)
                got=tile.backward_tile(z.reshape(-1,v).clone(),target,result.state,upstream)
            exact(got.reshape_as(wanted),wanted)
            native_nll=F.cross_entropy(z.float().reshape(-1,v),target,ignore_index=-100,reduction='none')
            lp=F.log_softmax(z.float().reshape(-1,v),-1);entropy=-(lp.exp()*lp).sum(-1)
            nll_error=metric_error(result.token_nll,native_nll,1e-4)
            entropy_error=metric_error(result.token_entropy,entropy,.01)
            if not torch.equal(result.first_argmax,z.reshape(-1,v).argmax(-1)):
                raise RuntimeError('First-index argmax changed')
            expected_route={'all_safe':'all_safe','mixed':'mixed_rows'}.get(mode,'native')
            if result.state.metadata.route!=expected_route:raise RuntimeError('Required numerical route inactive')
            if sum(t.numel() for t in result.state.tensors)>4*target.numel():raise RuntimeError('Non-row-sized saved state')
            rows.append({'mode':mode,'scale':scale,'route':result.state.metadata.route,
                         'frozen_ce_gradient_exact':True,'nll':nll_error,'entropy':entropy_error,
                         'argmax_exact':True,'saved_state_elements':sum(t.numel() for t in result.state.tensors)})
    # Genuine scale-before-half discriminator at full vocabulary.
    progress['current_case']='full_vocabulary_scale_before_half_negative'
    z=torch.zeros(1,v,device=device,dtype=torch.float16);target=torch.zeros(1,device=device,dtype=torch.int64)
    result=tile.forward_tile(z,target)
    with torch.no_grad():
        ordinary=tile.backward_tile(z,target,result.state,tile.prepare_upstream(torch.tensor(1.,device=device)/2048))
        scaled=tile.backward_tile(z,target,result.state,tile.prepare_upstream(torch.tensor(65536.,device=device)/2048))
    if (torch.count_nonzero(ordinary[0,1:])!=0 or torch.count_nonzero(scaled[0,1:])!=v-1
            or torch.count_nonzero((ordinary.float()*65536)[0,1:])!=0):
        raise RuntimeError('Scale-before-half negative witness not established')
    progress['scale_before_half_negative']=True
    return {'rows':rows,'scale_before_half_negative':True,'saved_state_only_row_sized':True,
            'projection_recompute_verified_here':False,'task_quality':'UNKNOWN','admission':'UNKNOWN'}


def run(progress=None):
    import triton
    if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Interpreter is not target evidence')
    if torch.__version__!='2.11.0+cu128' or triton.__version__!='3.6.0':raise RuntimeError('Pinned runtime required')
    if (torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='Tesla T4'
            or torch.cuda.get_device_capability(0)!=(7,5)):raise RuntimeError('Pinned one T4 required')
    progress={} if progress is None else progress
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    result=run_cases(torch.device('cuda:0'),progress)
    torch.cuda.synchronize()
    if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):
        raise RuntimeError('Global RNG changed')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    report={'status':'FAILED_TILE_PREFLIGHT_ONLY','progress':{},'score':None,'admission':'UNKNOWN'}
    try:report['checks']=run(report['progress']);report['status']='PASSED_TILE_PREFLIGHT_ONLY'
    except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
    Path(a.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    raise SystemExit(report['status']!='PASSED_TILE_PREFLIGHT_ONLY')
if __name__=='__main__':main()
