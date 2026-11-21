"""CPU+Triton interpreter only. Not SM75/Triton3.6 or whole-task validation."""
import ast
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from solution import mixed_rows as m
from solution import materialized_ce as ce

assert os.environ.get('TRITON_INTERPRET')=='1'
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]


def inputs(v=17):
    z=torch.zeros(8,v,dtype=torch.float16)
    z[1,0]=12.;z[2,2]=10.5;z[3,3]=5.;z[7,0]=30.
    z[5]=torch.linspace(-2,2,v).half()
    y=torch.tensor([1,0,2,3,-100,5,0,0],dtype=torch.long)
    return z,y


def prepass(z,y):
    with torch.no_grad():
        f=z.float();maximum=f.amax(1);logsum=(f-maximum[:,None]).exp().sum(1).log()
        idx=torch.where(y==-100,0,y)
        selected=f.gather(1,idx[:,None]).flatten()-maximum-logsum
        selected=torch.where(y!=-100,selected,0.)
        guard=((y!=-100)&((selected>=-ce.NEAR_CERTAIN_NLL)|~selected.isfinite())).to(torch.int32)
    return maximum,logsum,selected,guard


def equal_classes(a,b):
    return torch.equal(a.isnan(),b.isnan()) and torch.equal(a.isposinf(),b.isposinf()) and torch.equal(a.isneginf(),b.isneginf())


def test_partition_excludes_ignored_preserves_near_guard_and_global_invalid():
    z,y=inputs();maximum,logsum,selected,guard=prepass(z,y)
    rows,reason=m.partition_rows(selected,y,guard)
    assert reason is None and rows.tolist()==[1,2,7]
    assert not torch.any(y[rows]==-100)
    assert ce.NEAR_CERTAIN_NLL==1e-3
    invalid=selected.clone();invalid[0]=float('nan')
    assert m.partition_rows(invalid,y,guard)[1]=='invalid_selected_logp'
    invalid[0]=float('inf');assert m.partition_rows(invalid,y,guard)[1]=='invalid_selected_logp'
    allnear=torch.where(y!=-100,1,0).to(torch.int32)
    assert m.partition_rows(selected,y,allnear)[1]=='all_supervised_rows_near_certain'


def test_native_near_selected_rows_exact_and_safe_values_not_recomputed():
    z,y=inputs();maximum,logsum,selected,guard=prepass(z,y)
    rows,_=m.partition_rows(selected,y,guard)
    out=m.mixed_selected_values(z,y,selected,rows)
    native=F.cross_entropy(z.float(),y,reduction='none',ignore_index=-100)
    assert torch.equal(out[rows].view(torch.uint8),(-native[rows]).view(torch.uint8))
    safe=guard==0
    assert torch.equal(out[safe].view(torch.uint8),selected[safe].view(torch.uint8))
    assert out.shape==selected.shape and torch.equal(z,inputs()[0])


def test_same_row_geometry_sum_and_grouped_sum_counterexample():
    g=torch.Generator().manual_seed(913);n=127
    losses=(torch.rand(n,generator=g)*torch.pow(10.,torch.randint(-6,8,(n,),generator=g))).float()
    mask=torch.arange(n)%3==0;target=torch.zeros(n,dtype=torch.long)
    values=-losses;ordered=m.ordered_nll_sum(values,target)
    full=torch.zeros(n,17);full[:,5]=values
    native_full=F.nll_loss(full,torch.full((n,),5,dtype=torch.long),reduction='sum')
    grouped=m.ordered_nll_sum(values[mask],target[mask])+m.ordered_nll_sum(values[~mask],target[~mask])
    assert torch.equal(ordered,native_full)
    assert ordered.item()==45099144. and grouped.item()==45099148.
    (ROOT/'validation/sum_order_discriminator.json').write_text(json.dumps({'scope':'CPU float32 NLL only; not proof of target GPU reduction bitidentity','rows':n,'full_order':ordered.item(),'subset_totals_added':grouped.item(),'difference':(grouped-ordered).item()},indent=2)+'\n')


def test_normalized_q_once_both_paths_disjoint_half_gradient_and_rng():
    z,y=inputs();maximum,logsum,selected,guard=prepass(z,y);rows,_=m.partition_rows(selected,y,guard)
    for upstream in (1.,65536.,-65536.,0.,-0.,1e20):
        a=z.clone().requires_grad_();b=z.clone().requires_grad_();denominator=torch.tensor(13,dtype=torch.int64)
        reference=F.cross_entropy(b.float(),y,reduction='sum')/denominator
        reference.backward(torch.tensor(upstream,dtype=torch.float32))
        kernels=[];native_q=[];kernel=m.soup_ce_mixed_backward;original_grad=torch.autograd.grad
        class TrackedKernel:
            def __getitem__(self,grid):
                def launch(*args,**kw):kernels.append(args[4].clone());return kernel[grid](*args,**kw)
                return launch
        def grad(*args,**kw):
            native_q.append(kw['grad_outputs'].clone());return original_grad(*args,**kw)
        rng=torch.get_rng_state().clone();diag=ce.Diagnostics()
        with patch.object(m,'soup_ce_mixed_backward',TrackedKernel()),patch.object(torch.autograd,'grad',side_effect=grad):
            actual=m.MixedRowSum.apply(a,y,maximum,logsum,selected,guard,rows,diag)/denominator
            actual.backward(torch.tensor(upstream,dtype=torch.float32))
        q=torch.tensor(upstream,dtype=torch.float32)/denominator
        assert len(kernels)==len(native_q)==1 and torch.equal(kernels[0],q) and torch.equal(native_q[0],q)
        assert torch.equal(rng,torch.get_rng_state())
        assert a.grad.dtype==torch.float16 and equal_classes(a.grad,b.grad)
        assert torch.equal(a.grad[rows].view(torch.uint8),b.grad[rows].view(torch.uint8))
        finite=a.grad.isfinite()&b.grad.isfinite();diff=(a.grad.float()-b.grad.float()).abs()
        assert torch.all(diff[finite] <= 2**-24 + .002*b.grad.float().abs()[finite])
        assert torch.isclose(actual,reference,atol=1e-6,rtol=1e-4)
        assert diag.events[0]['native_supervised_rows']==3 and diag.events[0]['safe_supervised_rows']==4 and diag.events[0]['ignored_rows']==1
        assert diag.events[-1]['phase']=='enqueued_not_completion'
        # A second division would produce a clearly different nonzero safe-row adjoint.
        if upstream==65536.:assert not torch.equal(a.grad[0],b.grad[0]/13)


def test_native_rows_not_stored_by_custom_kernel_then_index_copy_only():
    z,y=inputs();maximum,logsum,selected,guard=prepass(z,y)
    out=torch.full_like(z,123.);q=torch.tensor(65536./13,dtype=torch.float32)
    m.soup_ce_mixed_backward[(8,1)](z,y,maximum,logsum,q,guard,out,V=17,BLOCK=2048,num_warps=4,enable_fp_fusion=False)
    assert torch.all(out[guard!=0]==123.) and not torch.all(out[guard==0]==123.)
    source=ast.parse((ROOT/'solution/mixed_rows.py').read_text())
    backward=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='backward')
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='index_copy_' for n in ast.walk(backward))==1
    assert not any(isinstance(n,ast.AugAssign) for n in ast.walk(backward))


def test_nonfinite_q_and_lazy_q_full_native_backward():
    z,y=inputs();maximum,logsum,selected,guard=prepass(z,y);rows,_=m.partition_rows(selected,y,guard)
    for q in (torch.tensor(float('inf')),torch.tensor(float('nan')),torch._neg_view(torch.tensor(3.))):
        a=z.clone().requires_grad_();b=z.clone().requires_grad_();diag=ce.Diagnostics()
        expected,=torch.autograd.grad(F.cross_entropy(b.float(),y,reduction='sum'),b,grad_outputs=q)
        ctx=type('Context',(),{'saved_tensors':(a,y,maximum,logsum,guard,rows),'diagnostics':diag})()
        with torch.no_grad(),patch.object(m,'soup_ce_mixed_backward',side_effect=AssertionError('custom backward forbidden')):
            actual=m.MixedRowSum.backward(ctx,q)[0]
        assert equal_classes(actual,expected)
        finite=actual.isfinite()&expected.isfinite();assert torch.equal(actual[finite],expected[finite])
        assert diag.events[-1]['event']=='native_backward_fallback'


def test_denominator_version_and_higher_order_remain_native():
    z,y=inputs();maximum,logsum,selected,guard=prepass(z,y);rows,_=m.partition_rows(selected,y,guard)
    a=z.clone().requires_grad_();d=torch.tensor(13,dtype=torch.int64)
    loss=m.MixedRowSum.apply(a,y,maximum,logsum,selected,guard,rows,None)/d;d.add_(1)
    try:loss.backward()
    except RuntimeError as exc:assert 'modified by an inplace operation' in str(exc)
    else:raise AssertionError('denominator version not enforced')
    a=z.clone().requires_grad_();b=z.clone().requires_grad_();diag=ce.Diagnostics()
    actual=m.MixedRowSum.apply(a,y,maximum,logsum,selected,guard,rows,diag)
    ga,=torch.autograd.grad(actual,a,create_graph=True)
    gb,=torch.autograd.grad(F.cross_entropy(b.float(),y,reduction='sum'),b,create_graph=True)
    assert torch.equal(ga,gb) and ga.requires_grad
    assert diag.events[-1]['reason']=='higher_order'


def test_global_nonfinite_prepass_and_original_five_kernel_ASTs_unchanged():
    z,y=inputs();z[4,0]=float('nan');z[6,3]=float('inf')
    partial=torch.empty(8,1);bad=torch.empty(8,1,dtype=torch.int32)
    ce.soup_ce_exp_tile_max[(8,1)](z,y,partial,bad,V=17,NT=1,BLOCK=2048,num_warps=4,enable_fp_fusion=False)
    assert bad[4,0]>0 and bad[6,0]>0  # Includes ignored row4: no fakezero masking.
    old=Path('/redacted/local/soup_web_wave2_20261030/ce_quiet_v4/solution/materialized_ce.py')
    def kernels(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and any(ast.unparse(d)=='triton.jit' for d in n.decorator_list)}
    assert kernels(old)==kernels(ROOT/'solution/materialized_ce.py')
