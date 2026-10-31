"""Bounded actual-vocabulary target gate; no task speed/admission inference."""
import json
import os
import weakref
import torch
import torch.nn.functional as F
from . import materialized_ce as ce


class Capture:
    def __init__(self):self.offers=[];self.loss_ref=None;self.completions=0
    def __call__(self,**fields):self.offers.append(fields);return True
    def complete(self,loss):self.completions+=1;self.loss_ref=weakref.ref(loss);return True


def error(actual,expected,*,atol,rtol):
    if not torch.equal(torch.isfinite(actual),torch.isfinite(expected)):
        raise RuntimeError('Metric finite classes differ')
    if not torch.isfinite(actual).all():raise RuntimeError('Finite fixture produced nonfinite summary')
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    delta=(actual-expected).abs()
    return {'max_abs':float(delta.max().item()),'max_rel':float((delta/expected.abs().clamp_min(1e-6)).max().item()),'atol':atol,'rtol':rtol}


def run_cases(device):
    v=ce.VOCABULARY
    generator=torch.Generator(device=device).manual_seed(563)
    logits=torch.randn((4,3,v),device=device,dtype=torch.float32,generator=generator).half()
    labels=torch.tensor([[0,1,2],[0,3,4],[0,-100,5],[0,6,7]],device=device,dtype=torch.int64)
    logits[0,0].zero_();logits[0,0,1]=30.
    logits[1,1].zero_();logits[1,1,4]=30.
    # All ignored/final rows still need entropy/first-index. One tie crosses tiles.
    logits[:,2].fill_(-3.);logits[:,2,1]=4.;logits[:,2,-1]=4.
    if v>2048:logits[:,2,2048]=4.
    original_logits=logits.clone();original_labels=labels.clone()
    rows=[]
    for scale in (1.,65536.):
        a=logits.clone().requires_grad_();b=logits.clone().requires_grad_();capture=Capture();diag=ce.Diagnostics()
        actual=ce.materialized_causal_loss(a,labels,13,summary_sink=capture,diagnostics=diag)
        reference=ce.materialized_causal_loss(b,labels,13)
        if (len(capture.offers)!=1 or capture.completions!=1 or capture.loss_ref() is not actual
                or capture.offers[0]['route']!='mixed_rows'):
            raise RuntimeError('Mixed summary/normalized loss completion did not activate exactly once')
        if not torch.equal(actual.reshape(1).view(torch.uint8),reference.reshape(1).view(torch.uint8)):
            raise RuntimeError('Summary changed main CE loss bits')
        summary=capture.offers[0];target=ce._targets(labels)
        native_nll=F.cross_entropy(a.detach().float().view(-1,v),target.flatten(),reduction='none').view_as(labels)
        native_logp=F.log_softmax(a.detach().float(),-1)
        native_entropy=-(native_logp.exp()*native_logp).sum(-1)
        nll_error=error(summary['token_nll'],native_nll,atol=1e-6,rtol=1e-4)
        entropy_error=error(summary['token_entropy'],native_entropy,atol=1e-6,rtol=.01)
        if not torch.equal(summary['first_argmax'],a.detach().float().argmax(-1)):
            raise RuntimeError('First-index argmax mismatch')
        near=torch.tensor([0,4],device=device)
        if not torch.equal(summary['token_nll'].flatten()[near].view(torch.uint8),native_nll.flatten()[near].view(torch.uint8)):
            raise RuntimeError('Native near-row NLL bits changed')
        # Keep the exact native T-1 mask/multiply/sum/count/clamp geometry.
        rewind=[]
        for i in range(4):
            mask=(labels[i,1:]!=-100).float();denominator=mask.sum().clamp_min(1)
            row_actual=(summary['token_nll'][i,:-1]*mask).sum()/denominator
            row_native=(native_nll[i,:-1]*mask).sum()/denominator
            rewind.append(error(row_actual.reshape(1),row_native.reshape(1),atol=1e-6,rtol=1e-4))
        attention_mask=torch.tensor([[1,1,0],[1,1,1],[1,0,1],[1,1,0]],device=device)
        metric_actual=(summary['token_entropy']*attention_mask).sum()/attention_mask.sum()
        metric_native=(native_entropy*attention_mask).sum()/attention_mask.sum()
        aggregate=error(metric_actual.reshape(1),metric_native.reshape(1),atol=1e-6,rtol=.01)
        for key in ('token_nll','token_entropy','first_argmax'):
            value=summary[key]
            if value.requires_grad or value.grad_fn is not None or not value.is_contiguous():raise RuntimeError('Detached summary contract')
            value.zero_()  # Consumer storage must not alias a CE saved tensor.
        q=torch.tensor(scale,device=device)
        ga,=torch.autograd.grad(actual,a,grad_outputs=q);gb,=torch.autograd.grad(reference,b,grad_outputs=q)
        if not torch.equal(ga.view(torch.uint8),gb.view(torch.uint8)):
            raise RuntimeError('Summary changed CE gradient bits')
        rows.append({'scale':scale,'loss_and_gradients_unchanged_bits':True,'native_near_nll_bits':True,
                     'nll':nll_error,'entropy':entropy_error,'attention_mask_entropy':aggregate,'rewind_rows':rewind})
    # All-safe publication also required; all-near/ignored/nonfinite remain native.
    for mode in ('all_safe','all_near','all_ignored','ignored_nan'):
        z=logits.clone();y=labels.clone();capture=Capture()
        if mode=='all_safe':z.zero_()
        elif mode=='all_ignored':y.fill_(-100)
        elif mode=='ignored_nan':z[0,-1,0]=float('nan')
        else:
            target=ce._targets(y).flatten()
            for r,t in enumerate(target.tolist()):
                if t!=-100:z.view(-1,v)[r].zero_();z.view(-1,v)[r,t]=30.
        loss=ce.materialized_causal_loss(z.requires_grad_(),y,13,summary_sink=capture)
        if mode=='all_safe':
            if len(capture.offers)!=1 or capture.offers[0]['route']!='all_safe' or capture.loss_ref() is not loss:
                raise RuntimeError('All-safe publication did not activate')
        elif capture.offers or capture.completions:raise RuntimeError('Native fallback published a summary')
    if not torch.equal(logits,original_logits) or not torch.equal(labels,original_labels):raise RuntimeError('Input mutation')
    return {'vocabulary':v,'mixed_rows':rows,'all_safe_publish':True,'native_fallback_no_publish':True,
            'ignored_final_rows_included':True,'tie_across_tile_boundary':True,
            'engineering_gate_not_official_quality':True,'task_quality':'UNKNOWN','admission':'UNKNOWN'}


def run():
    import triton
    if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Interpreter is not a target gate')
    if str(torch.__version__)!='2.11.0+cu128' or triton.__version__!='3.6.0':raise RuntimeError('Target version mismatch')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('One target GPU required')
    prop=torch.cuda.get_device_properties(0)
    if (prop.major,prop.minor,prop.multi_processor_count)!=(7,5,40):raise RuntimeError('Target T4/SM75 mismatch')
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    result=run_cases(torch.device('cuda:0'))
    torch.cuda.synchronize()
    if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):raise RuntimeError('RNG state changed')
    return result


if __name__=='__main__':
    print(json.dumps({'status':'PASSED_PRODUCER_PREFLIGHT_ONLY','checks':run()},allow_nan=False),flush=True)
