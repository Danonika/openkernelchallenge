#!/usr/bin/env python3
"""Independent tiny dense-forward / guarded-compact-backward CPU integration oracle.
Materialized forward and diagnostic tensors are intentional, NOT memory savings.
"""
import hashlib,json,math,sys,traceback
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(26103041)
ROOT=Path(__file__).resolve().parent;RESULTS=[]

def clean(o):
    if isinstance(o,torch.Tensor):return clean(o.detach().tolist())
    if isinstance(o,float) and not math.isfinite(o):return "NaN" if math.isnan(o) else "+Inf" if o>0 else "-Inf"
    if isinstance(o,dict):return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)):return [clean(v) for v in o]
    return o
def kinds(x):
    y=torch.zeros_like(x,dtype=torch.int8);y[x.isnan()]=1;y[torch.isposinf(x)]=2;y[torch.isneginf(x)]=3;return y
def cmp(a,b):
    ka,kb=kinds(a),kinds(b);use=(ka==0)&(kb==0);err=(a[use].double()-b[use].double()).abs()
    return dict(nonfinite_class_mismatch=int((ka!=kb).sum()),mismatch_indices=(ka!=kb).nonzero().tolist(),
                finite_max_abs=err.max().item() if err.numel() else None,reference_classes=kb.tolist())
def exact(a,b):
    c=cmp(a,b);assert c['nonfinite_class_mismatch']==0 and c['finite_max_abs'] in (None,0.),c;return c
def run(name,fn):
    try:r=dict(name=name,status='PASS',details=fn())
    except Exception as e:r=dict(name=name,status='FAIL',error=repr(e),traceback=traceback.format_exc())
    RESULTS.append(r);print(json.dumps(clean(r),allow_nan=False),flush=True)
def shift(labels):return torch.cat((labels[:,1:],torch.full_like(labels[:,:1],-100)),1)
def ce(z,targets,denominator):
    if denominator is None:return F.cross_entropy(z.float().reshape(-1,z.shape[-1]),targets.reshape(-1),ignore_index=-100,reduction='mean')
    return F.cross_entropy(z.float().reshape(-1,z.shape[-1]),targets.reshape(-1),ignore_index=-100,reduction='sum')/denominator
def metrics(z16,labels,attention,row_ids):
    with torch.no_grad():
        z=z16.float();lp=z.log_softmax(-1);token_entropy=-(lp.exp()*lp).sum(-1)
        entropy=(token_entropy*attention).sum()/attention.sum()
        truth=labels[:,1:];valid=truth.ne(-100);pred=z[:,:-1].argmax(-1)
        count=valid.sum();hits=(pred.eq(truth)&valid).sum()
        accuracy=hits.float()/count if count.item()>0 else torch.zeros(())
        row_loss=[];counts=[]
        for i in range(z.shape[0]):
            per=F.cross_entropy(z[i,:-1],truth[i],reduction='none',ignore_index=-100).float()
            n=valid[i].sum();row_loss.append((per*valid[i]).sum()/n.clamp_min(1));counts.append(n)
        return dict(row_ids=tuple(row_ids),entropy=entropy,accuracy=accuracy,supervised_count=count,
                    attention_tokens=attention.sum(),
                    row_losses=torch.stack(row_loss),row_counts=torch.stack(counts),token_entropy=token_entropy)
def packet_exact(a,b):
    assert a['row_ids']==b['row_ids']
    for k in ('entropy','accuracy','supervised_count','attention_tokens','row_losses','row_counts','token_entropy'):exact(a[k],b[k])
    assert all(not value.requires_grad for value in a.values() if isinstance(value,torch.Tensor))

class GuardedHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,w,labels,attention,denominator,row_ids,trace):
        w16=w.half();z=F.linear(h.half(),w16);targets=shift(labels)
        loss=ce(z,targets,denominator)
        trace['packet']=metrics(z,labels,attention,row_ids)
        ctx.tensor_denominator=isinstance(denominator,torch.Tensor)
        ctx.save_for_backward(h,w,labels,*([denominator] if ctx.tensor_denominator else []))
        ctx.denominator=None if ctx.tensor_denominator else denominator;ctx.trace=trace
        ctx.finite_w=bool(w16.isfinite().all());ctx.finite_z=bool(z.isfinite().all())
        ctx.forward_selected=z.reshape(-1,z.shape[-1])[targets.flatten().ne(-100)].detach()
        return loss
    @staticmethod
    def backward(ctx,incoming):
        h,w,labels,*extra=ctx.saved_tensors;denominator=extra[0] if ctx.tensor_denominator else ctx.denominator
        targets=shift(labels);use=targets.flatten().ne(-100);u=int(use.sum())
        d=torch.as_tensor(float(u) if denominator is None else denominator,dtype=incoming.dtype)
        normalized=incoming/d
        checks=dict(effective_weight_finite=ctx.finite_w,all_forward_logits_finite=ctx.finite_z,
                    incoming_scalar_finite=bool(incoming.isfinite()),denominator_positive_finite=bool(d.isfinite() and d>0),
                    normalized_upstream_finite=bool(normalized.isfinite()),has_supervised_rows=u>0)
        fast=all(checks.values());trace=ctx.trace
        trace['dispatch']=dict(route='compact_native_ce' if fast else 'dense_reference_fallback',
                               checks=checks,failed_checks=[k for k,v in checks.items() if not v],selected_rows=u,total_rows=targets.numel())
        with torch.enable_grad():
            hh=h.detach().requires_grad_(True)
            if fast:
                selected=hh.reshape(-1,hh.shape[-1])[use];hh16=selected.half();hh16.retain_grad()
                z=F.linear(hh16,w.half());selected_loss=ce(z,targets.flatten()[use],denominator)
                dx=torch.autograd.grad(selected_loss,hh,grad_outputs=incoming)[0]
                trace['selected_logits_exact']=bool(torch.equal(z.detach(),ctx.forward_selected))
                dx16=torch.zeros_like(h).half().reshape(-1,h.shape[-1]);dx16[use]=hh16.grad
                trace['dx16']=dx16.view_as(h)
            else:
                hh16=hh.half();hh16.retain_grad();z=F.linear(hh16,w.half())
                dense_loss=ce(z,targets,denominator)
                dx=torch.autograd.grad(dense_loss,hh,grad_outputs=incoming)[0]
                trace['selected_logits_exact']=None;trace['dx16']=hh16.grad
        return dx,None,None,None,None,None,None

def paired(h0,w,labels,attention,row_ids,denominator,scale):
    h=h0.clone(memory_format=torch.preserve_format).requires_grad_();h16=h.half();h16.retain_grad()
    z=F.linear(h16,w.half());loss=ce(z,shift(labels),denominator);packet=metrics(z,labels,attention,row_ids)
    dx=torch.autograd.grad(loss,h,grad_outputs=torch.tensor(scale,dtype=loss.dtype))[0]
    other=h0.clone(memory_format=torch.preserve_format).requires_grad_();trace={}
    candidate=GuardedHead.apply(other,w,labels,attention,denominator,row_ids,trace)
    grad=torch.autograd.grad(candidate,other,grad_outputs=torch.tensor(scale,dtype=candidate.dtype))[0]
    packet_exact(trace['packet'],packet);loss_cmp=exact(candidate,loss)
    dx_cmp=exact(grad,dx);dx16_cmp=exact(trace['dx16'],h16.grad)
    if trace['dispatch']['route']=='compact_native_ce':assert trace['selected_logits_exact']
    return dict(loss=loss.detach(),loss_comparison=loss_cmp,dx_comparison=dx_cmp,dx16_comparison=dx16_cmp,
                actual_dense_leaf_stride=list(h.stride()),actual_dense_head_input_stride=list(h16.stride()),
                packet=packet,dispatch=trace['dispatch'],selected_logits_exact=trace['selected_logits_exact'])

def finite_integration():
    cases=[]
    for vocabulary in (19,31):
        for pattern in ('mixed','ignored_sequence','finite_ties'):
            h=torch.randn(3,7,26)[...,::2]*.2;w=torch.randn(vocabulary,13)*.2
            labels=torch.randint(0,vocabulary,(3,7));attention=torch.ones(3,7)
            if pattern=='mixed':labels[:,::3]=-100;attention[1,-1]=0;labels[1,-1]=-100
            elif pattern=='ignored_sequence':labels[0]=-100;labels[1,:4]=-100
            else:w.zero_();labels[:]=0
            u=int(shift(labels).ne(-100).sum())
            for denominator in (None,float(u+5)):
                for scale in (.25,65536.):
                    out=paired(h,w,labels,attention,(41,7,99),denominator,scale)
                    assert out['dispatch']['route']=='compact_native_ce'
                    assert out['packet']['row_counts'].sum()==u
                    if pattern=='finite_ties':assert out['packet']['accuracy']==1
                    cases.append(dict(vocabulary=vocabulary,pattern=pattern,denominator=denominator,scale=scale,
                                      loss=out['loss'],dispatch=out['dispatch'],loss_metrics_gradients_exact=True,
                                      selected_fp16_logits_exact=out['selected_logits_exact'],
                                      dense_leaf_stride=out['actual_dense_leaf_stride'],dense_head_input_stride=out['actual_dense_head_input_stride']))
    tensor_d=torch.tensor(u+5,dtype=torch.int64)
    tensor_case=paired(h,w,labels,attention,(41,7,99),tensor_d,.25)
    assert tensor_case['dispatch']['route']=='compact_native_ce'
    cases.append(dict(pattern='immutable_scalar_int64_tensor_denominator',denominator=int(tensor_d),scale=.25,
                      loss_metrics_gradients_exact=True,selected_fp16_logits_exact=tensor_case['selected_logits_exact']))
    return dict(cases=cases,trials=len(cases),dense_materialized_forward=True,actual_streaming_memory_savings=False,
                decision='Combined reference-ordered metrics and guarded compact native-autograd backward match these CPU fixtures exactly, including selected FP16 logits/dX16. Dense-versus-gathered GPU GEMM equivalence remains open.')

def exception_integration():
    hbase=torch.tensor([[[1.,0.],[.3,.2],[.1,.4]],[[.2,.1],[.5,.2],[.4,.3]]])
    wbase=torch.tensor([[2.,2.],[-2.,-2.],[1.,-1.]])
    labelsbase=torch.tensor([[0,0,1],[1,2,0]]);attbase=torch.ones(2,3);cases=[]
    names=('ignored_nan','padded_nan','last_position_nan','ignored_positive_infinity','finite_operands_head_overflow',
           'effective_weight_overflow','weight_nan','incoming_inf','incoming_nan','zero_denominator','nan_denominator',
           'infinite_denominator','normalized_upstream_overflow','all_ignored_mean','all_ignored_positive_denominator',
           't1_native_mean','t1_supplied_denominator','last_nan_connected_zero_upstream')
    for name in names:
        h=hbase.clone();w=wbase.clone();labels=labelsbase.clone();att=attbase.clone();denominator=7.;scale=1.
        if name in ('ignored_nan','padded_nan'):
            h[0,0,0]=float('nan');labels[0,1]=-100
            if name=='padded_nan':att[0,0]=0
        elif name=='last_position_nan':h[0,-1,0]=float('nan')
        elif name=='ignored_positive_infinity':h[0,0,0]=float('inf');labels[0,1]=-100
        elif name=='finite_operands_head_overflow':h[0,0]=40000.;labels[0,1]=-100
        elif name=='effective_weight_overflow':w[0,0]=100000.
        elif name=='weight_nan':w[0,0]=float('nan')
        elif name=='incoming_inf':scale=float('inf')
        elif name=='incoming_nan':scale=float('nan')
        elif name=='zero_denominator':denominator=0.
        elif name=='nan_denominator':denominator=float('nan')
        elif name=='infinite_denominator':denominator=float('inf')
        elif name=='normalized_upstream_overflow':denominator=1e-40;scale=65536.
        elif name=='all_ignored_mean':labels[:]=-100;denominator=None
        elif name=='all_ignored_positive_denominator':labels[:]=-100
        elif name in ('t1_native_mean','t1_supplied_denominator'):
            h=h[:,:1];labels=labels[:,:1];att=att[:,:1]
            if name=='t1_native_mean':denominator=None
        elif name=='last_nan_connected_zero_upstream':h[0,-1,0]=float('nan');scale=0.
        out=paired(h,w,labels,att,(12,5),denominator,scale)
        assert out['dispatch']['route']=='dense_reference_fallback'
        if name.startswith('t1_'):
            assert out['packet']['accuracy']==0 and out['packet']['row_counts'].sum()==0
            assert out['packet']['row_losses'].sum()==0 and out['packet']['entropy'].isfinite()
        cases.append(dict(case=name,loss=out['loss'],entropy=out['packet']['entropy'],dispatch=out['dispatch'],
                          loss_comparison=out['loss_comparison'],gradient_comparison=out['dx_comparison'],dx16_comparison=out['dx16_comparison']))
    return dict(cases=cases,all_reference_fallbacks_match_positional_nonfinite_classes=True,
                full_hidden_state_retained_for_backward_only_guard_failures=True,
                decision='Conservative guards plus recomputation of the original full geometry preserve these exceptional CPU paths. Finite loss is not a sufficient guard; upstream failures require the original full hidden state.')

def metric_discriminators():
    labels=torch.tensor([[0,0,1]]);attention=torch.tensor([[1.,0.,1.]])
    z=torch.tensor([[[1.,0.,-1.],[float('nan'),0.,1.],[0.,0.,0.]]],dtype=torch.float16)
    labels[0,2]=-100
    p=metrics(z,labels,attention,(88,))
    assert p['entropy'].isnan()
    finite_selected_entropy=(-(z.float().log_softmax(-1).exp()*z.float().log_softmax(-1)).sum(-1))[attention.bool()].mean()
    assert finite_selected_entropy.isfinite()
    mixed=torch.tensor([[[0.,-float('inf'),1.]]],dtype=torch.float16)
    lp=mixed.float().log_softmax(-1);explicit_entropy=-(lp.exp()*lp).sum(-1)
    assert explicit_entropy.isnan().all()
    # A final-time NaN is excluded from accuracy/rewind targets but not full main CE graph.
    h=torch.tensor([[[1.,0.],[.3,.2],[float('nan'),0.]]]);w=torch.tensor([[2.,2.],[-2.,-2.],[1.,-1.]])
    labels2=torch.tensor([[0,0,1]]);out=paired(h,w,labels2,torch.ones(1,3),(88,),7.,1.)
    assert torch.isfinite(out['loss']) and torch.isfinite(out['packet']['row_losses']).all()
    hleaf=h.clone().requires_grad_();z2=F.linear(hleaf.half(),w.half());full=ce(z2,shift(labels2),7.)
    full_dx=torch.autograd.grad(full,hleaf)[0]
    hleaf2=h.clone().requires_grad_();trim=F.linear(hleaf2[:,:-1].half(),w.half())
    trim_loss=ce(trim,labels2[:,1:],7.);trim_dx=torch.autograd.grad(trim_loss,hleaf2)[0]
    contrast=cmp(trim_dx,full_dx);assert contrast['nonfinite_class_mismatch']==2
    return dict(masked_padding_entropy=p['entropy'],incorrect_selected_only_entropy=finite_selected_entropy,
                mixed_finite_negative_inf_explicit_entropy=explicit_entropy,
                final_nan_main_loss=out['loss'],final_nan_rewind_losses=out['packet']['row_losses'],
                trimmed_final_position_gradient_comparison=contrast,
                decision='Do not select entropy mask rows, replace 0*−Inf with zero, or drop final ignored logits from the main CE graph: these alter observed exceptional semantics.')

class PacketQueue:
    def __init__(self):self.queue=deque();self.seen=set();self.train_attention_tokens=0
    def record(self,call_id,packet,training):
        if not training:return
        if call_id in self.seen:raise ValueError('duplicate training packet')
        self.seen.add(call_id);self.queue.append((call_id,packet));self.train_attention_tokens+=int(packet['attention_tokens'])
    def consume(self,expected):
        if not self.queue:raise ValueError('empty training FIFO')
        if self.queue[0][0]!=expected:raise ValueError('stale or out-of-order packet')
        return self.queue.popleft()[1]
def packet_lifecycle():
    q=PacketQueue();z=torch.zeros(3,5,7).half();labels=torch.zeros(3,5,dtype=torch.long);labels[0]=-100;labels[1,:3]=-100
    p1=metrics(z,labels,torch.ones(3,5),(41,7,99));q.record('trainA',p1,True)
    p2=metrics(z,labels.flip(0),torch.ones(3,5),(99,7,41));q.record('eval',p2,False)
    assert len(q.queue)==1 and q.train_attention_tokens==15
    rejected=[]
    for kind,fn in (('duplicate',lambda:q.record('trainA',p1,True)),('wrong_order',lambda:q.consume('trainB'))):
        try:fn()
        except ValueError as e:rejected.append(dict(case=kind,error=str(e)))
        else:raise AssertionError('packet lifecycle failure was hidden')
    q.record('trainB',p2,True);a=q.consume('trainA');b=q.consume('trainB')
    assert a['row_ids']==(41,7,99) and b['row_ids']==(99,7,41) and not q.queue
    try:q.consume('trainA')
    except ValueError as e:rejected.append(dict(case='stale_reconsume',error=str(e)))
    else:raise AssertionError('duplicate consumption was accepted')
    global_loss=ce(z,shift(labels),None);wrong=a['row_losses'].mean()
    assert abs(global_loss.item()-wrong.item())>.1
    assert a['row_counts'].sum()==shift(labels).ne(-100).sum()
    assert q.train_attention_tokens==30 and a['attention_tokens']!=a['supervised_count']
    return dict(train_packets_recorded=2,eval_packets_enqueued=0,rejections=rejected,
                train_attention_tokens=q.train_attention_tokens,per_packet_supervised_tokens=int(a['supervised_count']),
                per_packet_attention_tokens=int(a['attention_tokens']),eval_did_not_increment_training_tokens=True,
                original_row_order_preserved=True,global_mean_loss=global_loss,wrong_average_row_means=wrong,
                actual_trainer_integration=False,
                decision='Synthetic FIFO protocol rejects stale/duplicate packets and isolates eval from train. Row means cannot replace global CE; this is not verified Soup/TRL integration.')

run('combined_finite_head_metrics_and_guarded_backward',finite_integration)
run('full_geometry_exceptional_guard_dispatch',exception_integration)
run('metric_nonfinite_and_final_position_discriminators',metric_discriminators)
run('synthetic_metric_packet_lifecycle',packet_lifecycle)
payload=dict(cycle=4,lane='head',generated_utc=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,
             torch_git_version=torch.version.git_version,threads=[torch.get_num_threads(),torch.get_num_interop_threads()],
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             summary=dict(passed=sum(r['status']=='PASS' for r in RESULTS),failed=sum(r['status']=='FAIL' for r in RESULTS)),
             candidates=['SOUP-NEW-001','SOUP-NEW-002','SOUP-TRAIN-001'],
             not_validated=['GPU gathered/dense GEMM equivalence','actual trainer hooks/FIFO','streamed forward or memory savings','CUDA GradScaler/optimizer behavior','signed-zero/NaN payload bit identity','full quality or performance graph points'],results=RESULTS)
(ROOT/'head_results.json').write_text(json.dumps(clean(payload),indent=2,allow_nan=False)+'\n')
print(json.dumps(payload['summary']));sys.exit(bool(payload['summary']['failed']))
