import sys,weakref
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest,torch
from solution.token_chunk_head import token_chunk_boundary
from old_solution.token_chunk_head import native_boundary
from shared_consumer import bridge_state as state
from shared_consumer.consumer import streaming_sink,native_order_row_losses
import cpu_tile_oracle as oracle


def fixture():
    g=torch.Generator().manual_seed(810)
    h=(torch.randn(2,5,8,generator=g)*.1).requires_grad_()
    w=torch.nn.Parameter(torch.randn(17,8,generator=g)*.1,requires_grad=False)
    labels=torch.tensor([[-100,1,2,-100,4],[-100,5,-100,7,8]])
    return h,w,labels
@pytest.mark.parametrize('chunk',[1,3,7])
@pytest.mark.parametrize('denominator',[None,13,torch.tensor(13,dtype=torch.int64)])
@pytest.mark.parametrize('scale',[1.,.125,65536.])
def test_global_reduction_true_upstream_full_vocabulary_adjoint(chunk,denominator,scale):
    h,w,labels=fixture();oracle.q_seen.clear()
    a,sa,_=native_boundary(h,w,labels,denominator)
    b,sb=token_chunk_boundary(h,w,labels,chunk,denominator,tile_api=oracle)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    ag,=torch.autograd.grad(a,h,torch.tensor(scale));bg,=torch.autograd.grad(b,h,torch.tensor(scale))
    torch.testing.assert_close(ag,bg,rtol=0,atol=0,equal_nan=True)
    assert len(oracle.q_seen)==1
    divisor=6 if denominator is None else denominator
    torch.testing.assert_close(oracle.q_seen[0],torch.tensor(scale)/divisor,rtol=0,atol=0)
    for k in ['entropy','token_nll','predictions']:torch.testing.assert_close(getattr(sa,k),getattr(sb,k),rtol=0,atol=0)


def test_saved_normalizer_mutation_raises_before_tile_backward():
    h,w,labels=fixture();oracle.captures.clear()
    loss,_=token_chunk_boundary(h,w,labels,3,13,tile_api=oracle)
    oracle.captures[0].add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):torch.autograd.grad(loss,h)


def test_external_global_denominator_mutation_raises():
    h,w,labels=fixture();den=torch.tensor(13,dtype=torch.int64)
    loss,_=token_chunk_boundary(h,w,labels,3,den,tile_api=oracle);den.add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):torch.autograd.grad(loss,h)


def test_all_ignored_global_mean_and_actual_upstream():
    h,w,labels=fixture();labels.fill_(-100)
    a,sa,_=native_boundary(h,w,labels);b,sb=token_chunk_boundary(h,w,labels,3,tile_api=oracle)
    assert torch.isnan(a) and torch.isnan(b)
    ag,=torch.autograd.grad(a,h,torch.tensor(65536.));bg,=torch.autograd.grad(b,h,torch.tensor(65536.))
    torch.testing.assert_close(ag,bg,rtol=0,atol=0,equal_nan=True)
    assert torch.count_nonzero(bg)==0


def test_one_common_streaming_summary_and_complete_normalized_loss():
    h,w,labels=fixture();owner=torch.nn.Module()
    with state.scope(owner,labels,vocabulary=17) as tr:
        tr.producer_owner_ref=weakref.ref(owner);sink=streaming_sink(owner,labels,vocabulary=17)
        loss,s=token_chunk_boundary(h,w,labels,3,13,summary_sink=sink,tile_api=oracle)
        assert tr.bind_outputs(loss,SimpleNamespace(loss=loss,logits=None))
        for key,value in [('token_entropy',s.entropy),('first_argmax',s.predictions),('token_nll',s.token_nll)]:assert tr.take(key,None,labels) is value
        assert tr.consumed==set(state._FIELDS)
        _,counts=native_order_row_losses(s.token_nll,labels);assert counts==[3,3]
    torch.autograd.grad(loss,h)


def test_no_higher_order_acceptance():
    h,w,labels=fixture();loss,_=token_chunk_boundary(h,w,labels,tile_api=oracle)
    with pytest.raises(RuntimeError,match='Higher-order'):torch.autograd.grad(loss,h,create_graph=True)
