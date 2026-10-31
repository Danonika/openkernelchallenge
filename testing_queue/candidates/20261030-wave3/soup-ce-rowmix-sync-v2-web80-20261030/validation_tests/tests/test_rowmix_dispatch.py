"""CPU interpreter dispatch proof; never target SM75 performance/correctness proof."""
from contextlib import ExitStack
from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F
from solution import materialized_ce as ce
from solution import mixed_rows as m


def fixture(v=17):
    z=torch.zeros(4,3,v,dtype=torch.float16)
    y=torch.tensor([[0,1,2],[0,3,4],[0,-100,5],[0,6,7]])
    z[0,0,1]=12.;z[1,1,4]=30.
    return z,y


def call(z,y,d=13):
    stats=ce.Diagnostics()
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor,'is_cuda',new=property(lambda _:True)))
        stack.enter_context(patch.object(torch.cuda,'get_device_capability',return_value=(7,5)))
        stack.enter_context(patch.object(ce,'VOCABULARY',z.shape[-1]))
        result=ce.materialized_causal_loss(z,y,d,diagnostics=stats)
    return result,stats


def classes(a,b):
    for name in ('isnan','isposinf','isneginf'):
        assert torch.equal(getattr(a,name)(),getattr(b,name)())
    finite=a.isfinite()&b.isfinite()
    assert torch.equal(a[finite],b[finite])


def test_mixed_dispatch_full_geometry_near_exact_and_safe_gradient_gate():
    z,y=fixture();a=z.clone().requires_grad_();b=z.clone().requires_grad_()
    actual,diag=call(a,y);expected=ce.native_causal_loss(b,y,13)
    ga,=torch.autograd.grad(actual,a,grad_outputs=torch.tensor(65536.))
    gb,=torch.autograd.grad(expected,b,grad_outputs=torch.tensor(65536.))
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-4)
    near=torch.tensor([0,4]);target=ce._targets(y)
    assert torch.equal(ga.view(-1,17)[near].view(torch.uint8),gb.view(-1,17)[near].view(torch.uint8))
    assert torch.all((ga.float()-gb.float()).abs()<=2**-24+.002*gb.float().abs())
    assert torch.all(ga.view(-1,17)[target.view(-1)==-100]==0)
    events={x['event']:x for x in diag.events}
    assert events['triton_forward']['route']=='mixed_rows'
    assert events['triton_forward']['native_supervised_rows']==2
    assert events['triton_forward']['safe_supervised_rows']==5
    assert events['triton_forward']['ignored_rows']==5
    assert events['triton_backward']['route']=='mixed_rows'


@pytest.mark.parametrize('empty',[False,True])
def test_all_near_and_allignored_are_full_native(empty):
    z,y=fixture();target=ce._targets(y).view(-1)
    if empty:y.fill_(-100)
    else:
        for row,idx in enumerate(target):
            if idx!=-100:z.view(-1,17)[row,idx]=30.
    a=z.clone().requires_grad_();b=z.clone().requires_grad_()
    actual,diag=call(a,y);expected=ce.native_causal_loss(b,y,13)
    ga,=torch.autograd.grad(actual,a);gb,=torch.autograd.grad(expected,b)
    classes(actual,expected);classes(ga,gb)
    reasons=[e.get('reason') for e in diag.events if e['event']=='native_forward_fallback']
    assert reasons==['no_supervised_rows' if empty else 'all_supervised_rows_near_certain']
    assert not any(e['event'].startswith('triton_forward') for e in diag.events)


@pytest.mark.parametrize('position',[(0,0,0),(0,2,0)])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('upstream',[0.,1.,float('inf'),float('nan')])
def test_nonfinite_supervised_or_ignored_global_native_geometry(position,value,upstream):
    z,y=fixture();z[position]=value
    a=z.clone().requires_grad_();b=z.clone().requires_grad_()
    actual,diag=call(a,y);expected=ce.native_causal_loss(b,y,13)
    q=torch.tensor(upstream)
    ga,=torch.autograd.grad(actual,a,grad_outputs=q);gb,=torch.autograd.grad(expected,b,grad_outputs=q)
    classes(actual,expected);classes(ga,gb)
    assert diag.events[-1]=={'event':'native_forward_fallback','reason':'nonfinite_logits_or_invalid_target'}
    assert not any(e['event']=='triton_forward' for e in diag.events)


@pytest.mark.parametrize('invalid',[-101,-1,17,999])
def test_invalid_target_stops_before_selected_load_and_raises_native_error(invalid):
    z,y=fixture();y[0,1]=invalid;a=z.requires_grad_()
    with patch.object(ce,'soup_ce_exp_selected',side_effect=AssertionError('selected load forbidden')):
        with pytest.raises(IndexError) as actual:call(a,y)
    with pytest.raises(IndexError) as expected:ce.native_causal_loss(a,y,13)
    assert str(actual.value)==str(expected.value)


def test_safe_route_uses_original_backward_and_no_native_subset():
    z,y=fixture();z.zero_();a=z.requires_grad_()
    with patch.object(m,'MixedRowSum',side_effect=AssertionError('mixed route forbidden')), patch.object(ce,'VOCABULARY',17):
        actual,diag=call(a,y);actual.backward()
    assert [e['event'] for e in diag.events][-2:]==['triton_forward','triton_backward']
    assert 'route' not in diag.events[-1]


@pytest.mark.parametrize('v',[17,2049])
def test_partition_helpers_preserve_order_under_row_permutation(v):
    generator=torch.Generator().manual_seed(190)
    n=8;z=torch.randn(n,v,generator=generator,dtype=torch.float32).half()
    y=torch.arange(n)%v;y[2]=-100
    for row in (1,5):z[row].zero_();z[row,y[row]]=30.
    with torch.no_grad():
        logp=F.log_softmax(z.float(),dim=-1);idx=y.clamp_min(0)
        selected=logp.gather(1,idx[:,None]).flatten();selected[y==-100]=0.
        guard=((y!=-100)&(selected>=-ce.NEAR_CERTAIN_NLL)).int()
        permutation=torch.tensor([7,2,1,5,0,6,3,4])
        partition,why=m.partition_rows(selected[permutation],y[permutation],guard[permutation]);rows=partition.native_rows
        assert why is None and rows.tolist()==[2,3]
        values=m.mixed_selected_values(z[permutation],y[permutation],selected[permutation],rows)
        expected=F.cross_entropy(z[permutation].float(),y[permutation],reduction='none')
        assert torch.equal(values[rows].view(torch.uint8),(-expected[rows]).view(torch.uint8))
        assert values.shape==selected.shape
