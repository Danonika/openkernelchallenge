#!/usr/bin/env python3
"""Original toy CPU tests: LoRA B-tail ownership and per-mask saved hooks.

This consolidates an autograd boundary, NOT GPU launches. No source kernel is
copied; no model, data, official test, trainer or CUDA dependency is imported.
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

torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(26103023)
ROOT=Path(__file__).resolve().parent
RESULTS=[]
CONFLICT="Calling register_hooks on a saved tensor whose hooks have already been set. Hint: only one pair of hooks is allowed at a time."


def classes(x):
    out=torch.zeros_like(x,dtype=torch.int8)
    out[torch.isnan(x)]=1;out[torch.isposinf(x)]=2;out[torch.isneginf(x)]=3
    return out


def comparison(a,b):
    ca,cb=classes(a),classes(b);finite=(ca==0)&(cb==0)
    e=(a[finite].double()-b[finite].double()).abs()
    mismatch=int((ca!=cb).sum())
    err=e.max().item() if e.numel() else None
    return dict(nonfinite_class_mismatch=mismatch,finite_max_abs=err,
                reference_nonfinite=int((cb!=0).sum()),
                exact_finite_and_nonfinite_classes=(mismatch==0 and err in (None,0.)))


def assert_match(a,b,atol=1e-6,rtol=2e-5):
    c=comparison(a,b)
    assert c["nonfinite_class_mismatch"]==0
    finite=torch.isfinite(a)&torch.isfinite(b)
    torch.testing.assert_close(a[finite],b[finite],atol=atol,rtol=rtol)
    return c


def run(name,ids,fn):
    try:r=dict(name=name,candidates=ids,status="PASS",details=fn())
    except Exception as e:r=dict(name=name,candidates=ids,status="FAIL",error=repr(e),traceback=traceback.format_exc())
    RESULTS.append(r);print(json.dumps(r,allow_nan=False),flush=True)


class BScaleAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx,s16,b32,base32):
        b16=b32.half()
        branch16=s16@b16.T
        out=base32+branch16*2.
        ctx.save_for_backward(s16,b16)
        return out

    @staticmethod
    def backward(ctx,g32):
        s16,b16=ctx.saved_tensors
        d16=(g32.half()*2.).half()
        ds16=d16@b16
        db32=(d16.T@s16).float()
        return ds16,db32,g32


def graph_summary(y):
    node=y.grad_fn
    children=[None if n is None else n.name() for n,_ in node.next_functions]
    return dict(root=node.name(),direct_inputs=children)


def composite_lora():
    m,h,r=17,13,16
    values={"x":torch.randn(m,h)*.2}
    for branch,n in (("q",23),("v",11)):
        values["a"+branch]=torch.randn(r,h)*.2
        values["b"+branch]=torch.randn(n,r)*.2
        values["w"+branch]=torch.randn(n,h)*.2
    records=[];graphs={}
    for mode in ("normal","subnormal","overflow"):
        for training in (True,False):
            upstream={}
            for branch,n in (("q",23),("v",11)):
                g=torch.linspace(-.2,.3,m*n).view(m,n)
                if mode=="subnormal":g.fill_(3*2.**-27)
                if mode=="overflow":g.fill_(40000.)
                upstream[branch]=g
            paired=[]
            for custom in (False,True):
                torch.manual_seed(13711)
                x=values["x"].clone().requires_grad_()
                params={k:values[k].clone().requires_grad_() for k in ("aq","bq","av","bv")}
                ys={};bases={};ss={};drops={}
                for branch in ("q","v"):
                    base=x@values["w"+branch].T;base.retain_grad();bases[branch]=base
                    base_clone=base.clone()  # Deliberately preserve baseline defensive clone.
                    dropped=F.dropout(x,p=.05,training=training);drops[branch]=dropped.detach()
                    s16=dropped.half()@params["a"+branch].half().T
                    s16.retain_grad();ss[branch]=s16
                    if custom:ys[branch]=BScaleAdd.apply(s16,params["b"+branch],base_clone)
                    else:ys[branch]=base_clone+(s16@params["b"+branch].half().T)*2.
                rng_after_forward=torch.get_rng_state().clone()
                if mode=="normal" and training:
                    graphs["custom" if custom else "baseline"]=graph_summary(ys["q"])
                torch.autograd.backward((ys["q"],ys["v"]),(upstream["q"],upstream["v"]))
                grads={"x":x.grad,**{k:p.grad for k,p in params.items()},
                       **{"base_"+k:v.grad for k,v in bases.items()},
                       **{"s16_"+k:v.grad for k,v in ss.items()}}
                paired.append(dict(outputs={k:v.detach() for k,v in ys.items()},grads=grads,
                                   rng=rng_after_forward,post_backward_rng=torch.get_rng_state(),drops=drops))
            baseline,custom=paired
            assert torch.equal(baseline["rng"],custom["rng"])
            assert torch.equal(baseline["post_backward_rng"],custom["post_backward_rng"])
            assert torch.equal(custom["rng"],custom["post_backward_rng"])
            for branch in ("q","v"):
                assert torch.equal(baseline["outputs"][branch],custom["outputs"][branch])
                assert torch.equal(baseline["drops"][branch],custom["drops"][branch])
            if training:assert not torch.equal(custom["drops"]["q"],custom["drops"]["v"])
            comps={k:assert_match(custom["grads"][k],baseline["grads"][k]) for k in baseline["grads"]}
            for k in ("x","aq","bq","av","bv","base_q","base_v"):
                assert custom["grads"][k].dtype==torch.float32
            records.append(dict(mode=mode,training=training,outputs_exact=True,rng_endstate_exact=True,
                                q_v_separate_dropout_preserved=True,gradient_comparisons=comps))
    return dict(cases=records,shape=dict(M=m,H=h,rank=r,qN=23,vN=11),graphs=graphs,
                frozen_base="Independent FP32 toy linear, not a bnb emulation",defensive_base_clone_retained=True,
                decision="CPU supports B+scale+add ownership while retaining A/dropout and exact cast boundaries. This custom Function still runs ordinary Torch ops; no launch fusion or T4 feasibility measured.")


def bool_pack(mask):
    flat=mask.reshape(-1).to(torch.uint8)
    flat=F.pad(flat,(0,(-flat.numel())%8))
    shifts=torch.arange(8,dtype=torch.uint8)
    bits=(flat.view(-1,8)<<shifts).sum(-1).to(torch.uint8)
    return ("packed_bool",bits,tuple(mask.shape))


def bool_unpack(packet):
    if isinstance(packet,torch.Tensor):return packet
    _,bits,shape=packet
    shifts=torch.arange(8,dtype=torch.uint8)
    flat=((bits[:,None]>>shifts)&1).flatten()
    return flat[:math.prod(shape)].bool().view(shape)


def register_one_mask(y,expected_shape,training=True,fail_pack=None):
    stats=dict(pack_calls=0,unpack_calls=0,packed_payload_bytes=0,identity_fallbacks=0)
    if not training:return "skip_eval",stats
    node=y.grad_fn
    if node is None:return "skip_no_node",stats
    if node.name()!="NativeDropoutBackward0" or not hasattr(node,"_raw_saved_result1"):
        return "skip_unsupported_node",stats
    expected_device=y.device  # Never retain y/its graph inside the saved pack closure.
    entered=False
    def pack(t):
        nonlocal entered
        entered=True;stats["pack_calls"]+=1
        if fail_pack=="value":raise ValueError("intentional pack failure")
        if fail_pack=="conflict_text":raise RuntimeError(CONFLICT)
        if t.dtype!=torch.bool or tuple(t.shape)!=tuple(expected_shape) or t.device!=expected_device:
            stats["identity_fallbacks"]+=1
            return t.detach()
        packet=bool_pack(t)
        stats["packed_payload_bytes"]+=packet[1].numel()
        return packet
    def unpack(t):
        stats["unpack_calls"]+=1
        return bool_unpack(t)
    try:node._raw_saved_result1.register_hooks(pack,unpack)
    except RuntimeError as e:
        # Known pre-mutation conflict only; an entered callback must never be hidden.
        if not entered and str(e)==CONFLICT:return "skip_existing_hooks",stats
        raise
    return "registered",stats


def selective_hooks():
    cases=[]
    for n in (1,7,8,9,257):
        paired=[]
        for enabled in (False,True):
            torch.manual_seed(18800+n)
            x=torch.linspace(.1,.9,n).requires_grad_()
            sin=x.sin();square=x.square()
            y=torch.ops.aten.native_dropout.default(x,.05,True)[0]
            saved_sin_before=sin.grad_fn._saved_self.data_ptr()
            saved_square_before=square.grad_fn._saved_self.data_ptr()
            status,stats=register_one_mask(y,x.shape) if enabled else ("baseline",{})
            assert sin.grad_fn._saved_self.data_ptr()==saved_sin_before==x.data_ptr()
            assert square.grad_fn._saved_self.data_ptr()==saved_square_before==x.data_ptr()
            output=y+sin+square;loss=output.sum();grads=[]
            for i in range(2):
                x.grad=None;loss.backward(retain_graph=(i==0));grads.append(x.grad.clone())
            paired.append(dict(output=output.detach(),grads=grads,rng=torch.get_rng_state(),status=status,stats=stats))
        base,hook=paired
        assert torch.equal(base["output"],hook["output"])
        assert torch.equal(base["rng"],hook["rng"])
        for a,b in zip(base["grads"],hook["grads"]):assert torch.equal(a,b)
        assert hook["stats"]["pack_calls"]==1 and hook["stats"]["unpack_calls"]==2
        cases.append(dict(length=n,outputs_gradients_rng_exact=True,retained_graph_backward_passes=2,
                          surrounding_saved_tensors_untouched=True,**hook["stats"]))
    # Unsupported situations deliberately retain the existing graph and behavior.
    x=torch.ones(9,requires_grad=True)
    with torch.no_grad():ng=torch.ops.aten.native_dropout.default(x,.05,True)[0]
    no_grad=register_one_mask(ng,x.shape)[0]
    eval_y=F.dropout(x,p=.05,training=False)
    eval_status=register_one_mask(eval_y,x.shape,training=False)[0]
    other_status=register_one_mask(x.sin(),x.shape)[0]
    assert (no_grad,eval_status,other_status)==("skip_no_node","skip_eval","skip_unsupported_node")
    # A metadata mismatch installs an explicit identity representation, not a bad pack.
    torch.manual_seed(933)
    mismatch=torch.ops.aten.native_dropout.default(x,.05,True)[0]
    expected_gradient=torch.autograd.grad(mismatch.sum(),x,retain_graph=True)[0]
    status,stats=register_one_mask(mismatch,(10,))
    observed=torch.autograd.grad(mismatch.sum(),x)[0]
    assert torch.equal(expected_gradient,observed) and stats["identity_fallbacks"]==1
    # Existing outer hooks remain authoritative; our callback must not run.
    outer=dict(pack=0,unpack=0)
    def outer_pack(t):outer["pack"]+=1;return t.detach()
    def outer_unpack(t):outer["unpack"]+=1;return t
    with torch.autograd.graph.saved_tensors_hooks(outer_pack,outer_unpack):
        existing=torch.ops.aten.native_dropout.default(x,.05,True)[0]
    conflict_status,conflict_stats=register_one_mask(existing,x.shape)
    existing.sum().backward()
    assert conflict_status=="skip_existing_hooks" and conflict_stats["pack_calls"]==0
    assert outer==dict(pack=1,unpack=1)
    failures=[]
    for mode,exception in (("value",ValueError),("conflict_text",RuntimeError)):
        broken=torch.ops.aten.native_dropout.default(x,.05,True)[0]
        try:register_one_mask(broken,x.shape,fail_pack=mode)
        except exception as e:failures.append(dict(mode=mode,propagated=True,error=str(e)))
        else:raise AssertionError("pack failure was silently swallowed")
        # A registration failure can leave hook state set: discard this test graph.
    return dict(cases=cases,fallbacks=dict(no_grad=no_grad,eval=eval_status,unsupported=other_status,
                                         metadata_mismatch=status,metadata_identity_fallbacks=stats["identity_fallbacks"],
                                         existing_hooks=conflict_status),outer_hooks_preserved=outer,
                intentional_pack_failures=failures,
                exact_target="NativeDropoutBackward0._raw_saved_result1 on installed CPU build",
                private_version_specific_api=True,original_masks_debug_cloned=False,
                output_graph_not_captured_by_pack_closure=True,
                memory_release_or_peak_savings_measured=False,
                decision="Per-saved-mask hooks preserve tested outputs/RNG/retained backward and leave neighboring saved tensors untouched. Skip known pre-mutation conflicts; propagate all pack failures. Actual PEFT/CUDA routing and memory benefit remain unvalidated.")


def cpu_autocast_contract():
    s0=(torch.randn(17,16)*.2).half();b0=torch.randn(23,16)*.2;base0=torch.randn(17,23)*.2
    records=[]
    for mode in ("normal","subnormal","overflow"):
        g=torch.linspace(-.2,.3,17*23).view(17,23)
        if mode=="subnormal":g.fill_(3*2.**-27)
        if mode=="overflow":g.fill_(40000.)
        paired=[]
        for use_autocast in (False,True):
            s=s0.clone().requires_grad_();b=b0.clone().requires_grad_();base=base0.clone().requires_grad_()
            if use_autocast:
                with torch.autocast("cpu",dtype=torch.float16):branch=F.linear(s,b)
            else:branch=s@b.half().T
            y=base.clone()+branch*2.
            y.backward(g)
            paired.append(dict(output=y.detach(),grads=dict(s=s.grad,b=b.grad,base=base.grad),
                               branch_dtype=str(branch.dtype)))
        explicit,auto=paired
        assert torch.equal(explicit["output"],auto["output"])
        assert auto["branch_dtype"]=="torch.float16"
        comps={k:assert_match(auto["grads"][k],explicit["grads"][k],atol=0,rtol=0) for k in explicit["grads"]}
        records.append(dict(mode=mode,output_exact=True,branch_dtype=auto["branch_dtype"],gradient_comparisons=comps))
    return dict(cases=records,
                decision="Installed CPU autocast F.linear agrees with explicit half-cast B-tail oracle in these fixtures; this does not prove CUDA autocast or actual PEFT dtype routing.")


run("lora_b_scale_add_custom_autograd_ownership",["SOUP-LORA-001"],composite_lora)
run("per_saved_native_dropout_mask_hooks",["SOUP-LORA-002"],selective_hooks)
run("cpu_autocast_b_tail_contract",["SOUP-LORA-001"],cpu_autocast_contract)

payload=dict(schema_version=1,cycle="2b",generated_utc=datetime.now(timezone.utc).isoformat(),
             scope="Cloud-only original bounded CPU LoRA and saved-mask fixtures",
             environment=dict(python=sys.version.split()[0],torch=torch.__version__,torch_git_version=torch.version.git_version,
                              cuda_available=torch.cuda.is_available(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads()),
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             summary=dict(passed=sum(r["status"]=="PASS" for r in RESULTS),failed=sum(r["status"]=="FAIL" for r in RESULTS)),
             not_validated=["actual CUDA/PEFT dropout dispatch and routing","actual bnb base layer","fused GPU launches/T4 resources",
                            "peak memory or storage release","actual GradScaler/optimizer steps","full training/quality or challenge gates"],results=RESULTS)
(ROOT/"results.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
summary={k:payload[k] for k in ("cycle","generated_utc","scope","environment","script_sha256","summary","not_validated")}
summary["outcomes"]=[dict(name=r["name"],candidates=r["candidates"],status=r["status"],
                          decision=r.get("details",{}).get("decision",r.get("error"))) for r in RESULTS]
(ROOT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
print(json.dumps(payload["summary"]))
sys.exit(bool(payload["summary"]["failed"]))
