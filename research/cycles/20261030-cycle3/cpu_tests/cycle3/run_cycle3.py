#!/usr/bin/env python3
"""Original cloud-CPU witnesses for frozen FP32 RMSNorm/residual and Qwen-style RoPE.
No model, dataset, copied kernel, official tests, CUDA or performance points.
"""
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(26103031)
ROOT=Path(__file__).resolve().parent;RESULTS=[]


def kind(x):
    out=torch.zeros_like(x,dtype=torch.int8)
    out[x.isnan()]=1;out[torch.isposinf(x)]=2;out[torch.isneginf(x)]=3
    return out


def compare(a,b):
    ka,kb=kind(a),kind(b);finite=(ka==0)&(kb==0)
    err=(a[finite].double()-b[finite].double()).abs()
    mismatch=int((ka!=kb).sum())
    maxabs=err.max().item() if err.numel() else None
    return dict(nonfinite_class_mismatch=mismatch,finite_max_abs=maxabs,
                exact_finite_and_classes=mismatch==0 and maxabs in (None,0.),
                reference_nonfinite=int((kb!=0).sum()))


def exact(a,b):
    c=compare(a,b);assert c["exact_finite_and_classes"],c
    return c


def run(name,ids,fn):
    try:r=dict(name=name,candidates=ids,status="PASS",details=fn())
    except Exception as e:r=dict(name=name,candidates=ids,status="FAIL",error=repr(e),traceback=traceback.format_exc())
    RESULTS.append(r);print(json.dumps(r,allow_nan=False),flush=True)


def norm(x,w):
    r=(x.pow(2).mean(-1,keepdim=True)+1e-6).rsqrt()
    return w*(x*r),r


def staged_dx(x,r,w,dy):
    g=dy*w
    direct=g*r
    dr=(g*x).sum(-1,keepdim=True)
    dv=(-.5*dr)*r.pow(3)
    variance=(dv/x.shape[-1])*(2*x)
    return direct+variance


def compact_dx(x,r,w,dy):
    g=dy*w
    return r*g-x*r.pow(3)*(g*x).mean(-1,keepdim=True)


class NormSaved(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w):
        y,r=norm(x,w);ctx.save_for_backward(x,r,w);return y
    @staticmethod
    def backward(ctx,dy):
        x,r,w=ctx.saved_tensors
        return staged_dx(x,r,w,dy),None


class NormRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w):
        y,_=norm(x,w);ctx.save_for_backward(x,w);return y
    @staticmethod
    def backward(ctx,dy):
        x,w=ctx.saved_tensors;_,r=norm(x,w)
        return staged_dx(x,r,w,dy),None


def norm_staged_and_state():
    trials=0;compact_different=0;compact_nonfinite=0;max_compact_abs=0.
    for width in (1,7,1536):
        for magnitude in (0.,1e-8,1.,1e8,1e20,3e38):
            for weight in (0.,1.,1e-30,1e20):
                for scale in (.25,65536.):
                    if magnitude==3e38:storage=torch.full((2,width*2),magnitude)
                    else:storage=torch.randn(2,width*2)*magnitude
                    x=storage[:,::2].detach().requires_grad_();w=torch.full((width,),weight)
                    assert torch.isfinite(x).all() and torch.isfinite(w).all()
                    dy=torch.randn_like(x)*scale
                    y,r=norm(x,w);ref=torch.autograd.grad(y,x,dy)[0]
                    with torch.no_grad():
                        got=staged_dx(x,r,w,dy);bad=compact_dx(x,r,w,dy);_,rr=norm(x,w)
                    exact(got,ref);exact(rr,r)
                    c=compare(bad,ref)
                    compact_different+=not c["exact_finite_and_classes"]
                    compact_nonfinite+=c["nonfinite_class_mismatch"]>0
                    max_compact_abs=max(max_compact_abs,c["finite_max_abs"] or 0.)
                    trials+=1
    # Frozen reference saved state can be recomputed, but reconstructing x loses information.
    x=torch.randn(4,7)*.001;w=torch.randn(7);dy=torch.randn_like(x)
    y,r=norm(x,w);ref=staged_dx(x,r,w,dy)
    xhalf=x.half().float();_,rhalf=norm(xhalf,w)
    half_comparison=compare(staged_dx(xhalf,rhalf,w,dy),ref)
    assert not half_comparison["exact_finite_and_classes"]
    n=x*r;reconstructed=n/r
    assert torch.count_nonzero(reconstructed!=x)>0
    zero_w=torch.zeros(7);zero_y,zr=norm(x,zero_w)
    invalid=zero_y/(zero_w*zr)
    assert invalid.isnan().all()
    huge=torch.full((1,1),3e38);_,hr=norm(huge,torch.ones(1))
    bad_x=(huge*hr)/hr
    assert bad_x.isnan().all()
    huge_staged=staged_dx(huge,hr,torch.ones(1),torch.ones_like(huge))
    huge_compact=compact_dx(huge,hr,torch.ones(1),torch.ones_like(huge))
    assert huge_staged.isnan().all() and torch.equal(huge_compact,torch.zeros_like(huge_compact))
    return dict(trials=trials,staged_backward_exact_finite_and_nonfinite_classes=True,
                recomputed_inverse_rms_exact=True,compact_formula_different_cases=compact_different,
                compact_formula_nonfinite_mismatch_cases=compact_nonfinite,
                compact_formula_max_finite_abs=max_compact_abs,
                fp16_saved_x_gradient_comparison=half_comparison,
                normalized_then_divided_x_max_abs=(reconstructed-x).abs().max().item(),
                zero_weight_reconstruction_all_nan=True,zero_inverse_rms_reconstruction_all_nan=True,
                finite_x_3e38_witness=dict(staged_gradient="NaN",compact_gradient=0.,forward_output=0.),
                decision="Frozen FP32 standalone norm needs the actual staged derivative for these exact CPU results. Reject universal compact-formula, FP16-saved-input, or reconstructed-input equivalence.")


def inventory(fn,x,w):
    records=[]
    xptr=x.untyped_storage().data_ptr();wptr=w.untyped_storage().data_ptr()
    def pack(t):
        st=t.untyped_storage();ptr=st.data_ptr()
        role="x" if ptr==xptr else "w" if ptr==wptr else "derived"
        records.append(dict(shape=list(t.shape),dtype=str(t.dtype),storage_ptr=ptr,storage_bytes=st.nbytes(),role=role))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):y=fn(x,w)
    grad=torch.autograd.grad(y,x,torch.ones_like(y))[0]
    unique={r["storage_ptr"]:r["storage_bytes"] for r in records}
    return dict(saved_references=len(records),unique_storages=len(unique),
                referenced_storage_bytes=sum(unique.values()),
                records=[{k:v for k,v in r.items() if k!="storage_ptr"} for r in records]),grad


def norm_storage():
    x=torch.randn(2,14)[:,::2].detach().requires_grad_();w=torch.randn(7)
    ref,grad=inventory(lambda x,w:norm(x,w)[0],x,w)
    saved,gs=inventory(NormSaved.apply,x,w)
    recompute,gr=inventory(NormRecompute.apply,x,w)
    exact(gs,grad);exact(gr,grad)
    assert (ref["saved_references"],ref["unique_storages"])==(5,3)
    assert (saved["saved_references"],saved["unique_storages"])==(3,3)
    assert recompute["unique_storages"]==2
    assert ref["referenced_storage_bytes"]==saved["referenced_storage_bytes"]
    wt=w.clone().requires_grad_()
    # A trainable affine weight uses the native fallback, never this frozen-only custom backward.
    y=norm(x,wt)[0] if wt.requires_grad else NormSaved.apply(x,wt)
    gx,gw=torch.autograd.grad(y,(x,wt),torch.ones_like(y))
    assert gw is not None and torch.isfinite(gw).all()
    no_grad_controls=[]
    for label,fn in (("reference",lambda x,w:norm(x,w)[0]),("save_x_r_w",NormSaved.apply),("recompute_r",NormRecompute.apply)):
        observed=[]
        def pack_no_grad(t):observed.append(tuple(t.shape));return t
        with torch.autograd.graph.saved_tensors_hooks(pack_no_grad,lambda t:t):
            out=fn(x.detach(),w)
        assert not out.requires_grad and len(observed)==0
        no_grad_controls.append(dict(implementation=label,output_requires_grad=False,saved_references=0))
    return dict(reference=ref,save_x_r_w=saved,recompute_r_from_x_w=recompute,
                trainable_weight_fallback_preserves_weight_gradient=True,
                no_grad_input_and_frozen_weight_controls=no_grad_controls,
                not_peak_memory_measurement=True,
                decision="Native frozen norm already saves only x/r/w storage. Deduplicating five references to three is not a storage saving; recomputing r removes only its tiny referenced storage in this toy inventory and adds unmeasured work.")


class ResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,w):
        residual=a+b;y,r=norm(residual,w)
        ctx.save_for_backward(residual,r,w);ctx.set_materialize_grads(False)
        return y,residual
    @staticmethod
    def backward(ctx,dy,dbypass):
        x,r,w=ctx.saved_tensors
        dx=None if dy is None else staged_dx(x,r,w,dy)
        if dbypass is not None:dx=dbypass if dx is None else dx+dbypass
        return dx,dx,None


class ResidualDefaultMaterialization(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,w):
        residual=a+b;y,r=norm(residual,w);ctx.save_for_backward(residual,r,w)
        return y,residual
    @staticmethod
    def backward(ctx,dy,dbypass):return ResidualNorm.backward(ctx,dy,dbypass)


def residual_contracts():
    records=[]
    for layout in ("independent","identical_inputs","overlapping_views"):
        for use in ("norm_only","residual_only","both_two_bypass_consumers"):
            values=torch.randn(2,8);other=torch.randn(2,7);w=torch.randn(7)
            dy=torch.randn(2,7);by1=torch.randn(2,7);by2=torch.randn(2,7)
            paired=[]
            for fused in (False,True):
                root=values.clone().requires_grad_();second=other.clone().requires_grad_()
                if layout=="independent":a=root[:,:7];b=second
                elif layout=="identical_inputs":a=root[:,:7];b=a
                else:a=root[:,:7];b=root[:,1:]
                before=root.detach().clone();before_second=second.detach().clone()
                if fused:y,res=ResidualNorm.apply(a,b,w)
                else:res=a+b;y=norm(res,w)[0]
                assert res.untyped_storage().data_ptr() not in (a.untyped_storage().data_ptr(),b.untyped_storage().data_ptr())
                if use=="norm_only":loss=(y*dy).sum()
                elif use=="residual_only":loss=(res*by1).sum()
                else:loss=(y*dy).sum()+(res*by1).sum()+(res*by2).sum()
                loss.backward()
                assert torch.equal(root.detach(),before) and torch.equal(second.detach(),before_second)
                paired.append(dict(y=y.detach(),res=res.detach(),grad=root.grad,second=second.grad))
            ref,got=paired;exact(got["y"],ref["y"]);exact(got["res"],ref["res"])
            c=compare(got["grad"],ref["grad"])
            assert c["nonfinite_class_mismatch"]==0
            torch.testing.assert_close(got["grad"],ref["grad"],atol=1e-6,rtol=2e-5)
            if ref["second"] is not None:torch.testing.assert_close(got["second"],ref["second"],atol=1e-6,rtol=2e-5)
            records.append(dict(layout=layout,use=use,root_gradient_comparison=c,inputs_unchanged=True,outputs_exact=True))
    # Deliberate accumulation-order counterexample with exactly representable direct/variance terms.
    witnesses=[]
    for width in (1,1536):
        values=[]
        for fused in (False,True):
            a=torch.full((1,width),512.,requires_grad=True);b=torch.full((1,width),512.,requires_grad=True);w=torch.ones(width)
            if fused:y,res=ResidualNorm.apply(a,b,w)
            else:res=a+b;y=norm(res,w)[0]
            torch.autograd.backward((y,res),(torch.full_like(y,2.**34),torch.ones_like(res)))
            values.append(a.grad.detach())
        assert torch.equal(values[0],torch.zeros_like(values[0])) and torch.equal(values[1],torch.ones_like(values[1]))
        witnesses.append(dict(width=width,native_gradient_all=0.,completed_norm_then_bypass_all=1.))
    unused_controls=[]
    for use in ("residual_only_norm_unused","residual_and_connected_zero_norm","norm_only"):
        grads=[]
        for implementation in ("native","explicit_none","default_materialization"):
            a=torch.full((1,1),1.5e38,requires_grad=True);b=torch.full((1,1),1.5e38,requires_grad=True);w=torch.ones(1)
            if implementation=="native":res=a+b;y=norm(res,w)[0]
            elif implementation=="explicit_none":y,res=ResidualNorm.apply(a,b,w)
            else:y,res=ResidualDefaultMaterialization.apply(a,b,w)
            if use=="residual_only_norm_unused":res.sum().backward()
            elif use=="residual_and_connected_zero_norm":torch.autograd.backward((y,res),(torch.zeros_like(y),torch.ones_like(res)))
            else:y.sum().backward()
            grads.append(a.grad.detach())
        exact(grads[1],grads[0])
        if use=="residual_only_norm_unused":assert grads[0].item()==1. and grads[2].isnan().all()
        else:assert all(g.isnan().all() for g in grads)
        unused_controls.append(dict(use=use,native_gradient="NaN" if grads[0].isnan().all() else grads[0].item(),
                                    explicit_none_gradient="NaN" if grads[1].isnan().all() else grads[1].item(),
                                    default_materialization_gradient="NaN" if grads[2].isnan().all() else grads[2].item()))
    version_checks=[]
    for fused in (False,True):
        a=torch.randn(2,7,requires_grad=True);b=torch.randn(2,7,requires_grad=True);w=torch.ones(7)
        if fused:y,res=ResidualNorm.apply(a,b,w)
        else:res=a+b;y=norm(res,w)[0]
        with torch.no_grad():res.add_(1)
        try:y.sum().backward()
        except RuntimeError as e:
            assert "modified by an inplace operation" in str(e)
            version_checks.append(True)
        else:raise AssertionError("saved residual mutation was not detected")
    return dict(ordinary_cases=records,exploratory_tolerance=dict(atol=1e-6,rtol=2e-5,challenge_gate=False),
                cancellation_witness=dict(a=512.,b=512.,dy=2.**34,bypass=1.,direct=2.**24,variance=-2.**24,
                                          cases=witnesses),unused_norm_output_controls=unused_controls,
                saved_residual_mutation_rejected=version_checks,
                decision="Standalone norm exactness does not establish residual-fusion exactness. Native graph branch accumulation can change 0 into 1 under regrouping; retain both outputs, alias handling, version checks and branch-order validation.")


def rotate_half(x):
    a,b=x.chunk(2,-1)
    return torch.cat((-b,a),-1)


def rope_ref(q,k,c,s,axis):
    c=c.unsqueeze(axis);s=s.unsqueeze(axis)
    return q*c+rotate_half(q)*s,k*c+rotate_half(k)*s


def rope_dx(g,c,s,axis):
    if g is None:return None
    c1,c2=c.unsqueeze(axis).chunk(2,-1);s1,s2=s.unsqueeze(axis).chunk(2,-1)
    g1,g2=g.chunk(2,-1)
    return torch.cat((g1*c1+g2*s2,g2*c2-g1*s1),-1)


class PairRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,c,s,axis):
        yq,yk=rope_ref(q,k,c,s,axis);ctx.save_for_backward(c,s);ctx.axis=axis
        ctx.set_materialize_grads(False)
        nondiff=[]
        if not q.requires_grad:nondiff.append(yq)
        if not k.requires_grad:nondiff.append(yk)
        if nondiff:ctx.mark_non_differentiable(*nondiff)
        return yq,yk
    @staticmethod
    def backward(ctx,gq,gk):
        c,s=ctx.saved_tensors
        return rope_dx(gq,c,s,ctx.axis),rope_dx(gk,c,s,ctx.axis),None,None,None


def rope_layout_adjoint():
    records=[]
    for axis in (1,2):
        for bpos in (1,2):
            for coeff in ("duplicated_halves","unequal_halves"):
                for req,use in (((True,True),"both"),((True,True),"q_only"),((True,False),"q_only"),((False,True),"k_only")):
                    b,t,hq,hk,d=2,7,12,2,128
                    q0=torch.randn(b,t,hq,d);k0=torch.randn(b,t,hk,d)
                    if axis==1:q0=q0.transpose(1,2);k0=k0.transpose(1,2)
                    theta=torch.randn(bpos,t,d//2)
                    c=torch.cat((theta.cos(),theta.cos()),-1);s=torch.cat((theta.sin(),theta.sin()),-1)
                    if coeff=="unequal_halves":c[...,d//2:]*=.7;s[...,d//2:]*=-.4
                    gq=torch.randn_like(q0);gk=torch.randn_like(k0);paired=[]
                    for custom in (False,True):
                        q=q0.clone(memory_format=torch.preserve_format).requires_grad_(req[0]);k=k0.clone(memory_format=torch.preserve_format).requires_grad_(req[1])
                        yq,yk=PairRoPE.apply(q,k,c,s,axis) if custom else rope_ref(q,k,c,s,axis)
                        assert yq.requires_grad==req[0] and yk.requires_grad==req[1]
                        loss=(yq*gq).sum() if use=="q_only" else (yk*gk).sum() if use=="k_only" else (yq*gq).sum()+(yk*gk).sum()
                        loss.backward()
                        paired.append(dict(yq=yq.detach(),yk=yk.detach(),dq=q.grad,dk=k.grad))
                    ref,got=paired;exact(got["yq"],ref["yq"]);exact(got["yk"],ref["yk"])
                    for key in ("dq","dk"):
                        if ref[key] is None:assert got[key] is None
                        else:exact(got[key],ref[key])
                    assert got["yq"].stride()==ref["yq"].stride() and got["yk"].stride()==ref["yk"].stride()
                    records.append(dict(axis=axis,position_batches=bpos,coefficient_pattern=coeff,requires_grad=list(req),used_outputs=use,
                                        exact=True,input_q_stride=list(q0.stride()),reference_output_q_stride=list(ref["yq"].stride()),
                                        input_k_stride=list(k0.stride()),reference_output_k_stride=list(ref["yk"].stride()),
                                        custom_output_q_stride=list(got["yq"].stride()),custom_output_k_stride=list(got["yk"].stride()),
                                        output_is_contiguous=ref["yq"].is_contiguous()))
    return dict(cases=records,largest_q_shape=[2,12,7,128],all_gradients_and_unused_none_exact=True,
                decision="Split-half pair adjoints match the tested FP32 strided/broadcast/unequal-head contracts. Preserve actual output strides and unused-gradient None semantics; this is not runtime Qwen dtype or CUDA evidence.")


def rope_storage_and_fallback():
    q=torch.randn(2,7,12,128).transpose(1,2).requires_grad_();k=torch.randn(2,7,2,128).transpose(1,2).requires_grad_()
    theta=torch.randn(1,7,64);c=torch.cat((theta.cos(),theta.cos()),-1);s=torch.cat((theta.sin(),theta.sin()),-1)
    storages={t.untyped_storage().data_ptr():name for name,t in (("q",q),("k",k),("cos",c),("sin",s))}
    inventories=[]
    for custom in (False,True):
        refs=[]
        def pack(t):refs.append((t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()));return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            yq,yk=PairRoPE.apply(q,k,c,s,1) if custom else rope_ref(q,k,c,s,1)
        unique=dict(refs)
        inv=dict(custom=custom,saved_references=len(refs),unique_storages=len(unique),
                 unique_roles=sorted(storages[p] for p in unique),referenced_storage_bytes=sum(unique.values()))
        assert inv["unique_roles"]==["cos","sin"]
        inventories.append(inv)
    assert inventories[0]["saved_references"]==4 and inventories[1]["saved_references"]==2
    assert inventories[0]["referenced_storage_bytes"]==inventories[1]["referenced_storage_bytes"]
    # Narrow FP32/frozen-coefficient/D128 dispatch contract; unsupported cases use the reference.
    cases=[]
    for name,q_dtype,c_dtype,trainable in (("all_fp16",torch.float16,torch.float16,False),
                                          ("mixed_half_q_float_coeff",torch.float16,torch.float32,False),
                                          ("trainable_coeff",torch.float32,torch.float32,True)):
        qq=q.detach().to(q_dtype).requires_grad_();kk=k.detach().to(q_dtype).requires_grad_()
        cc=c.to(c_dtype).detach().requires_grad_(trainable);ss=s.to(c_dtype).detach().requires_grad_(trainable)
        supported=all(t.dtype==torch.float32 for t in (qq,kk,cc,ss)) and not(cc.requires_grad or ss.requires_grad) and qq.shape[-1]==128
        assert not supported
        yq,yk=rope_ref(qq,kk,cc,ss,1)
        (yq.float().sum()+yk.float().sum()).backward()
        if trainable:assert cc.grad is not None and ss.grad is not None
        cases.append(dict(case=name,routed_reference=True,output_dtype=str(yq.dtype),q_gradient_dtype=str(qq.grad.dtype),coefficient_gradients_present=cc.grad is not None))
    return dict(saved_storage_inventory=inventories,fallback_cases=cases,not_peak_memory_measurement=True,
                decision="Native frozen-coefficient RoPE already saves cos/sin, not q/k. Four-to-two saved references is no unique-storage saving. Keep FP16/mixed/trainable-coefficient cases on reference until separately validated.")


def rope_negative_witnesses():
    x=torch.tensor([[[[1.,2.,3.,4.]]]])
    half=rotate_half(x)
    interleaved=torch.stack((-x[...,1::2],x[...,::2]),-1).flatten(-2)
    assert not torch.equal(half,interleaved)
    g=torch.tensor([[[[1.,2.,3.,4.]]]])
    c=torch.tensor([[[.2,.3,.7,.8]]]);s=torch.tensor([[[.1,.4,.6,.9]]])
    correct=rope_dx(g,c,s,1)
    inverse_style=g*c.unsqueeze(1)-rotate_half(g)*s.unsqueeze(1)
    assert not torch.equal(correct,inverse_style)
    # Same realistic duplicated trig coefficients; emulate one FP32 multiply-add
    # contraction in FP64 on bounded binary32 operands (not an executed GPU FMA).
    theta=torch.tensor(.6);cv=theta.cos();sv=theta.sin()
    a=torch.linspace(-3.1,2.7,4096);b=torch.linspace(.7,-2.3,4096).flip(0)
    second=(-b)*sv
    separate=a*cv+second
    contracted=(a.double()*cv.double()+second.double()).float()
    indices=(separate!=contracted).nonzero().flatten()
    assert indices.numel()>0;i=int(indices[0])
    return dict(split_half_result=half.flatten().tolist(),wrong_interleaved_result=interleaved.flatten().tolist(),
                unequal_half_correct_adjoint=correct.flatten().tolist(),wrong_inverse_style_adjoint=inverse_style.flatten().tolist(),
                fma_witness=dict(q_first=a[i].item(),q_second=b[i].item(),cos=cv.item(),sin=sv.item(),
                                 separate_fp32=separate[i].item(),single_contracted_product_add=contracted[i].item(),
                                 differing_elements=int(indices.numel()),tested_elements=4096,actual_gpu_fma=False),
                decision="Reject interleaved rotation and generic inverse-RoPE adjoint assumptions. FP32 contraction can change valid split-half/trig outputs; preserve multiply/add rounding unless separately accepted by actual gates.")


run("staged_norm_backward_and_saved_state",["cast-aware-norm"],norm_staged_and_state)
run("norm_unique_saved_storage_and_trainable_fallback",["cast-aware-norm"],norm_storage)
run("residual_alias_outputs_and_branch_association",["cast-aware-norm","residual-norm"],residual_contracts)
run("rope_fp32_layout_broadcast_and_adjoint",["attn-sgl-rounding-fusion"],rope_layout_adjoint)
run("rope_saved_storage_and_dtype_fallback",["attn-sgl-rounding-fusion"],rope_storage_and_fallback)
run("rope_wrong_layout_inverse_and_contraction",["attn-sgl-rounding-fusion"],rope_negative_witnesses)

payload=dict(schema_version=1,cycle=3,generated_utc=datetime.now(timezone.utc).isoformat(),
             scope="Original bounded cloud CPU norm/residual and RoPE semantic fixtures",
             environment=dict(python=sys.version.split()[0],torch=torch.__version__,torch_git_version=torch.version.git_version,
                              cuda_available=torch.cuda.is_available(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads()),
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             summary=dict(passed=sum(r["status"]=="PASS" for r in RESULTS),failed=sum(r["status"]=="FAIL" for r in RESULTS)),
             not_validated=["actual model dtype/stride routing","CUDA kernels/T4 timing","allocator peak memory or storage release",
                            "GPU accumulation/reduction order","signed-zero or NaN-payload bit identity","RoPE nonfinite inputs or gradients",
                            "FP16/mixed RoPE custom-kernel equivalence (reference fallback only)",
                            "full training quality or challenge numerical gates","performance graph points or own-kernel share"],
             rope_variant="qwen2-fp32-half-split-apply-only",results=RESULTS)
(ROOT/"results.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
summary={k:payload[k] for k in ("cycle","generated_utc","scope","environment","script_sha256","summary","not_validated")}
summary["outcomes"]=[dict(name=r["name"],candidates=r["candidates"],status=r["status"],
                          decision=r.get("details",{}).get("decision",r.get("error"))) for r in RESULTS]
(ROOT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
print(json.dumps(payload["summary"]))
sys.exit(bool(payload["summary"]["failed"]))
