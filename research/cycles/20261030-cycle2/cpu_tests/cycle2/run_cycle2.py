#!/usr/bin/env python3
"""Original bounded CPU fixtures: centered vocabulary merges and nonfinite parity.

No model, trainer, bitsandbytes, CUDA, source kernels or official tests imported.
Cycle-one artifacts are read-only and are not imported or overwritten.
"""
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.manual_seed(26103002)
ROOT=Path(__file__).resolve().parent
RESULTS=[]


def kind(x):
    out=torch.zeros_like(x,dtype=torch.int8)
    out[torch.isnan(x)]=1
    out[torch.isposinf(x)]=2
    out[torch.isneginf(x)]=3
    return out


def describe(x):
    k=kind(x)
    return dict(shape=list(x.shape),finite=int((k==0).sum()),nan=int((k==1).sum()),
                posinf=int((k==2).sum()),neginf=int((k==3).sum()),
                classes=k.tolist())


def compare(a,b):
    ka,kb=kind(a),kind(b)
    both=(ka==0)&(kb==0)
    delta=(a[both].double()-b[both].double()).abs()
    return dict(nonfinite_class_mismatch=int((ka!=kb).sum()),
                mismatch_indices=(ka!=kb).nonzero().tolist(),
                finite_max_abs=delta.max().item() if delta.numel() else None)


def finite_close(a,b,atol=3e-6,rtol=2e-5):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


def run(name,candidates,fn):
    try:
        r=dict(name=name,candidates=candidates,status="PASS",details=fn())
    except Exception as e:
        r=dict(name=name,candidates=candidates,status="FAIL",error=repr(e),traceback=traceback.format_exc())
    RESULTS.append(r)
    print(json.dumps(r,allow_nan=False),flush=True)


def partial(q,ids,valid=None):
    # q contains already FP16-rounded finite logits; padding is metadata only.
    if valid is None: valid=torch.ones(len(ids),dtype=torch.bool)
    batch=q.shape[0]
    if not valid.any():
        return (torch.full((batch,),-float("inf")),torch.zeros(batch),torch.zeros(batch),
                torch.full((batch,),2**31-1,dtype=torch.long))
    masked=torch.where(valid[None,:],q,torch.tensor(-float("inf")))
    m=masked.max(-1).values
    centered=masked-m[:,None]
    exp=centered.exp()
    s=exp.sum(-1)
    # Inactive lanes must contribute zero, not 0 * -inf.
    c=torch.where(valid[None,:],exp*centered,torch.zeros_like(exp)).sum(-1)
    ties=masked==m[:,None]
    index=torch.where(ties&valid[None,:],ids[None,:],2**31-1).min(-1).values
    return m,s,c,index


def merge(a,b):
    ma,sa,ca,ia=a;mb,sb,cb,ib=b
    m=torch.maximum(ma,mb)
    # Explicit identity handling prevents 0 * -inf when a tile is empty.
    da=torch.where(sa>0,ma-m,torch.zeros_like(m))
    db=torch.where(sb>0,mb-m,torch.zeros_like(m))
    wa=torch.where(sa>0,da.exp(),torch.zeros_like(m))
    wb=torch.where(sb>0,db.exp(),torch.zeros_like(m))
    s=wa*sa+wb*sb
    c=wa*(ca+sa*da)+wb*(cb+sb*db)
    index=torch.where(ma>mb,ia,torch.where(mb>ma,ib,torch.minimum(ia,ib)))
    return m,s,c,index


def finish(state,q,target):
    m,s,c,index=state
    logs=s.log()
    ce=logs-(q.gather(1,target[:,None]).squeeze(1)-m)
    entropy=logs-c/s
    prob=(q-m[:,None]).exp()/s[:,None]
    return ce,entropy,prob,index


def tiles(v,widths):
    out=[];start=0;i=0
    while start<v:
        end=min(v,start+widths[i%len(widths)])
        out.append((start,end));start=end;i+=1
    return out


def merged_finite():
    trials=0;maxerr={"ce":0.,"entropy":0.,"probability":0.}
    by_order={order:{"ce":0.,"entropy":0.,"probability":0.} for order in ("forward","reverse","balanced")}
    near_certain_witness=[]
    worst=None
    for v in (1,2,3,7,16,19,31,33,127,257):
        patterns={"ordinary":(torch.randn(3,v)*5).half().float(),
                  "high_offset":(32768+torch.randn(3,v)*64).half().float(),
                  "equal_max_half":torch.full((3,v),65504.).half().float(),
                  "near_certain":torch.cat((torch.full((3,1),32768.),torch.full((3,v-1),32752.)),1).half().float()}
        tied=(torch.randn(3,v)*.1).half().float()
        tied[:,0]=7
        if v>1:tied[:,-1]=7
        patterns["cross_tile_ties"]=tied
        for label,q0 in patterns.items():
            storage=torch.zeros(3,v*2);storage[:,::2]=q0
            q=storage[:,::2]
            target=torch.tensor([0,v//2,v-1])
            lp=q.log_softmax(-1)
            ref=(F.cross_entropy(q,target,reduction="none"),-(lp.exp()*lp).sum(-1),q.softmax(-1),q.argmax(-1))
            for widths in ((1,),(2,),(7,),(16,),(64,),(3,1,7,2,11)):
                ranges=tiles(v,widths)
                states=[partial(q[:,a:b],torch.arange(a,b)) for a,b in ranges]
                for order in ("forward","reverse","balanced"):
                    work=list(states if order!="reverse" else reversed(states))
                    if order=="balanced":
                        while len(work)>1:
                            work=[merge(work[i],work[i+1]) if i+1<len(work) else work[i] for i in range(0,len(work),2)]
                        state=work[0]
                    else:
                        state=work[0]
                        for p in work[1:]:state=merge(state,p)
                    got=finish(state,q,target)
                    for idx,key in enumerate(("ce","entropy","probability")):
                        finite_close(got[idx],ref[idx])
                        err=(got[idx]-ref[idx]).abs().max().item()
                        by_order[order][key]=max(by_order[order][key],err)
                        if err>maxerr[key]:
                            maxerr[key]=err
                            if key=="entropy":worst=dict(vocabulary=v,pattern=label,widths=list(widths),order=order)
                    assert torch.equal(got[3],ref[3])
                    if v==257 and label=="near_certain" and widths==(1,):
                        near_certain_witness.append(dict(order=order,target_zero_ce=got[0][0].item(),
                                                         baseline_target_zero_ce=ref[0][0].item(),
                                                         entropy=got[1][0].item(),baseline_entropy=ref[1][0].item(),
                                                         target_zero_probability=got[2][0,0].item(),baseline_target_zero_probability=ref[2][0,0].item()))
                    trials+=1
    return dict(trials=trials,max_abs_errors=maxerr,max_abs_errors_by_merge_order=by_order,
                near_certain_singleton_tile_witness=near_certain_witness,worst_entropy_case=worst,
                exploratory_tolerances=dict(atol=3e-6,rtol=2e-5,challenge_gate=False),
                argmax_first_global_index_exact=True,noncontiguous_rows=True,
                largest_logits_shape=[3,257],actual_gpu_kernels_tested=False,
                decision="Centered merge is CPU-supported for finite rounded logits, with explicit empty-tile handling and global-index ties; no bitwise/GPU/performance claim.")


def padding_identity():
    q=torch.tensor([[2.,2.,-1.],[65504.,65504.,65504.]])
    base=partial(q,torch.arange(3))
    padded=F.pad(q,(0,5),value=-float("inf"))
    valid=torch.arange(8)<3
    with_tail=partial(padded,torch.arange(8),valid)
    for a,b in zip(base,with_tail):assert torch.equal(a,b)
    empty=partial(torch.full((2,8),-float("inf")),torch.arange(8,16),torch.zeros(8,dtype=torch.bool))
    for order in ((base,empty),(empty,base)):
        got=merge(*order)
        for a,b in zip(got,base):assert torch.equal(a,b)
    ma,sa,ca,ia=base;mb,sb,cb,ib=empty;m=torch.maximum(ma,mb)
    # Tempting unguarded centered merge includes 0 * -inf.
    naive_c=(ma-m).exp()*(ca+sa*(ma-m))+(mb-m).exp()*(cb+sb*(mb-m))
    assert torch.isnan(naive_c).all()
    # Reverse processing plus "take left on tie" is not global-first argmax.
    left=partial(q[:,:1],torch.tensor([0]));right=partial(q[:,1:2],torch.tensor([1]))
    wrong=torch.where(right[0]>=left[0],right[3],left[3])
    correct=merge(right,left)[3]
    assert torch.equal(wrong,torch.ones(2,dtype=torch.long)) and torch.equal(correct,torch.zeros(2,dtype=torch.long))
    return dict(padded_tail_exact=True,empty_tile_identity_both_orders_exact=True,
                naive_empty_merge=describe(naive_c),wrong_reverse_tie_indices=wrong.tolist(),
                correct_global_tie_indices=correct.tolist(),
                decision="Reject unguarded empty-tile moment merges and traversal-order tie selection.")


def gradient_logits(z0,targets,scale=1.,denominator=1.,compact=False,explicit_nll=False):
    z=z0.clone().requires_grad_()
    use=targets.ne(-100)
    active=z[use] if compact else z
    t=targets[use] if compact else targets
    if explicit_nll:
        lp=active.float().log_softmax(-1);lp.retain_grad()
        loss=F.nll_loss(lp,t,ignore_index=-100,reduction="sum")/denominator
    else:
        loss=F.cross_entropy(active.float(),t,ignore_index=-100,reduction="sum")/denominator
    (loss*scale).backward()
    out=dict(loss=loss.detach(),gradient=z.grad)
    if explicit_nll:out["nll_to_logsoftmax_gradient"]=lp.grad
    return out


def ignored_nonfinite():
    rows={"nan":[float("nan"),0.,1.],"positive_inf":[float("inf"),0.,1.],
          "all_negative_inf":[-float("inf")]*3,"mixed_finite_negative_inf":[0.,-float("inf"),1.],
          "finite_control":[0.,-2.,1.]}
    evidence=[]
    for name,row in rows.items():
        for all_ignored in (False,True):
            z=torch.tensor([[1.,0.,-1.],row],dtype=torch.float16)
            target=torch.tensor([-100 if all_ignored else 0,-100])
            dense=gradient_logits(z,target)
            compact=gradient_logits(z,target,compact=True)
            separate=gradient_logits(z,target,explicit_nll=True)
            assert compare(dense["gradient"],separate["gradient"])["nonfinite_class_mismatch"]==0
            assert torch.equal(separate["nll_to_logsoftmax_gradient"][1],torch.zeros(3))
            cmp=compare(dense["gradient"],compact["gradient"])
            if name in ("nan","positive_inf","all_negative_inf"):
                assert cmp["nonfinite_class_mismatch"]==3
                assert torch.isnan(dense["gradient"][1]).all()
            else:assert cmp["nonfinite_class_mismatch"]==0 and cmp["finite_max_abs"]==0
            evidence.append(dict(ignored_row=name,all_ignored=all_ignored,
                                 dense_loss=describe(dense["loss"]),dense_gradient=describe(dense["gradient"]),
                                 compact_gradient=describe(compact["gradient"]),comparison=cmp,
                                 nll_ignored_gradient_zero=True))
    return dict(cases=evidence,
                decision="Unconditional scatter-zero changes baseline nonfinite propagation. Preserve finite-path opportunity but guard all actual rounded logits or use an exact reference fallback.")


def incoming_nonfinite():
    records=[]
    z=torch.tensor([[1.,0.,-1.],[1.,0.,-1.]],dtype=torch.float16)
    target=torch.tensor([0,-100])
    for name,scale,denom in (("finite",128.,7.),("positive_inf_scale",float("inf"),7.),
                             ("nan_scale",float("nan"),7.),("zero_denominator",1.,0.),
                             ("inf_denominator",1.,float("inf")),("nan_denominator",1.,float("nan"))):
        dense=gradient_logits(z,target,scale,denom)
        compact=gradient_logits(z,target,scale,denom,compact=True)
        cmp=compare(dense["gradient"],compact["gradient"])
        assert cmp["nonfinite_class_mismatch"]==0 and (cmp["finite_max_abs"] in (None,0.))
        p=z[:1].float().softmax(-1)-torch.tensor([[1.,0.,0.]])
        manual=((p/denom)*scale).half()
        manual_cmp=compare(dense["gradient"][:1],manual)
        assert torch.equal(dense["gradient"][1],torch.zeros(3,dtype=torch.float16))
        if name in ("positive_inf_scale","zero_denominator"):
            assert manual_cmp["nonfinite_class_mismatch"]==1
        records.append(dict(case=name,dense_gradient=describe(dense["gradient"]),
                            compact_autograd_comparison=cmp,naive_selected_gradient=describe(manual),
                            naive_selected_comparison=manual_cmp))
    return dict(cases=records,
                decision="Compaction with reference autograd preserves this finite-logit fixture; naive softmax-minus-onehot loses target-NaN parity for infinite effective upstream gradients. Guard or preserve reference operation order.")


def nonfinite_weights():
    records=[]
    zeros=torch.zeros(2,3,dtype=torch.float16)
    for label,value in (("nan",float("nan")),("positive_inf",float("inf")),("negative_inf",-float("inf")),("finite_control",1.)):
        w=torch.ones(3,2,dtype=torch.float16);w[0,0]=value
        dense=zeros@w
        scatter=torch.zeros_like(dense)
        cmp=compare(dense,scatter)
        if label!="finite_control":assert cmp["nonfinite_class_mismatch"]==2
        else:assert cmp["nonfinite_class_mismatch"]==0
        records.append(dict(weight_case=label,dense_zero_row_matmul=describe(dense),scatter_comparison=cmp))
    raw=torch.tensor([[100000.,1.],[1.,1.],[1.,1.]])
    assert torch.isfinite(raw).all() and not torch.isfinite(raw.half()).all()
    return dict(cases=records,finite_fp32_weight_can_cast_to_nonfinite_fp16=True,
                decision="Zero dlogits @ nonfinite W is not zero. Validate effective FP16 head weights, not only FP32 source weights, before zero-row elision.")


def overflow_logits():
    h0=torch.tensor([[1.,0.],[40000.,40000.]])
    w=torch.tensor([[2.,2.],[-2.,-2.],[1.,-1.]])
    target=torch.tensor([0,-100]);out=[]
    for compact in (False,True):
        h=h0.clone().requires_grad_()
        q=h[target.ne(-100)] if compact else h
        t=target[target.ne(-100)] if compact else target
        z=q.half()@w.half().T;z.retain_grad()
        loss=F.cross_entropy(z.float(),t,ignore_index=-100,reduction="sum")
        loss.backward()
        out.append(dict(logits=z.detach(),dz=z.grad,dh=h.grad,loss=loss.detach()))
    dense,compact=out
    assert torch.isfinite(h0.half()).all() and torch.isfinite(w.half()).all()
    assert not torch.isfinite(dense["logits"][1]).all()
    cmp=compare(dense["dh"],compact["dh"])
    assert cmp["nonfinite_class_mismatch"]==2
    return dict(rounded_inputs_and_weights_finite=True,
                dense_logits=describe(dense["logits"]),dense_loss=describe(dense["loss"]),
                dense_dlogits=describe(dense["dz"]),dense_hidden_gradient=describe(dense["dh"]),
                compact_hidden_gradient=describe(compact["dh"]),comparison=cmp,
                decision="Finite inputs/weights alone do not guard FP16 GEMM-output overflow. Check actual rounded logits already produced by metric forward, otherwise exact fallback.")


run("finite_centered_vocabulary_tile_merge",["SOUP-NEW-001","SOUP-NEW-002"],merged_finite)
run("padding_empty_tile_and_global_argmax",["SOUP-NEW-001","SOUP-NEW-002"],padding_identity)
run("ignored_nonfinite_logits_and_nll_boundary",["SOUP-TRAIN-001"],ignored_nonfinite)
run("nonfinite_incoming_gradient_and_denominator",["SOUP-TRAIN-001","SOUP-NEW-002"],incoming_nonfinite)
run("nonfinite_effective_head_weights",["SOUP-TRAIN-001"],nonfinite_weights)
run("finite_operands_fp16_logit_overflow",["SOUP-TRAIN-001"],overflow_logits)

payload=dict(schema_version=1,cycle=2,generated_utc=datetime.now(timezone.utc).isoformat(),
             scope="Cloud-only original bounded CPU numerical/semantic fixtures",
             environment=dict(python=sys.version.split()[0],torch=torch.__version__,torch_git_version=torch.version.git_version,
                              cuda_available=torch.cuda.is_available(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads()),
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             nonfinite_classes={"0":"finite","1":"NaN","2":"+Inf","3":"-Inf"},
             summary=dict(passed=sum(r["status"]=="PASS" for r in RESULTS),failed=sum(r["status"]=="FAIL" for r in RESULTS)),
             not_validated=["CUDA kernels or T4 timing","actual CUDA GradScaler/optimizer overflow skips","full model quality",
                            "actual integration or legal routing","GPU reduction/GEMM rounding","challenge gates or owned-kernel share"],results=RESULTS)
(ROOT/"results.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
summary={k:payload[k] for k in ("cycle","generated_utc","scope","environment","script_sha256","summary","not_validated")}
summary["outcomes"]=[dict(name=r["name"],candidates=r["candidates"],status=r["status"],
                          decision=r.get("details",{}).get("decision",r.get("error"))) for r in RESULTS]
(ROOT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
print(json.dumps(payload["summary"]))
sys.exit(bool(payload["summary"]["failed"]))
