import sys,weakref
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest,torch
from solution.token_chunk_head import token_chunk_boundary,native_boundary
from shared_consumer import bridge_state as state
from shared_consumer.consumer import streaming_sink,native_order_row_losses


def inputs():
    torch.manual_seed(820)
    return (torch.randn(2,4,8,requires_grad=True),torch.nn.Parameter(torch.randn(17,8),requires_grad=False),
            torch.tensor([[1,2,-100,4],[5,6,7,8]]))

@pytest.mark.parametrize('denominator',[None,9,torch.tensor(9,dtype=torch.int64)])
def test_parameter_and_native_denominator_contract(denominator):
    h,w,labels=inputs();a,_,_=native_boundary(h,w,labels,denominator);b,_=token_chunk_boundary(h,w,labels,3,denominator)
    torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
    ag,=torch.autograd.grad(a,h,torch.tensor(65536.));bg,=torch.autograd.grad(b,h,torch.tensor(65536.))
    torch.testing.assert_close(ag,bg,rtol=0,atol=0,equal_nan=True)


def test_common_streaming_sink_all_fields_and_loss_identity():
    h,w,labels=inputs();owner=torch.nn.Module()
    with state.scope(owner,labels,vocabulary=17) as transaction:
        transaction.producer_owner_ref=weakref.ref(owner)
        sink=streaming_sink(owner,labels,vocabulary=17)
        loss,s=token_chunk_boundary(h,w,labels,3,9,summary_sink=sink)
        assert transaction.bind_outputs(loss,SimpleNamespace(loss=loss,logits=None))
        assert transaction.take('token_entropy',None,labels) is s.entropy
        assert transaction.take('first_argmax',None,labels) is s.predictions
        assert transaction.take('token_nll',None,labels) is s.token_nll
        assert transaction.consumed==set(state._FIELDS)
        rows,counts=native_order_row_losses(s.token_nll,labels)
        assert counts==[2,3] and len(rows)==2
    assert state.current() is None and transaction.closed and transaction.summary is None
    assert torch.autograd.grad(loss,h)[0].shape==h.shape


def test_common_sink_rejects_recast_loss_identity():
    h,w,labels=inputs();owner=torch.nn.Module()
    with state.scope(owner,labels,vocabulary=17) as transaction:
        transaction.producer_owner_ref=weakref.ref(owner)
        sink=streaming_sink(owner,labels,vocabulary=17)
        loss,_=token_chunk_boundary(h,w,labels,3,9,summary_sink=sink)
        other=loss.clone()
        assert not transaction.bind_outputs(other,SimpleNamespace(loss=other,logits=None))


def test_failed_summary_publication_does_not_silently_return():
    h,w,labels=inputs()
    with pytest.raises(RuntimeError,match='summary publication failed'):
        token_chunk_boundary(h,w,labels,3,9,summary_sink=lambda **fields:False)


def test_higher_order_request_is_explicitly_unsupported():
    h,w,labels=inputs();loss,_=token_chunk_boundary(h,w,labels)
    with pytest.raises(RuntimeError,match='Higher-order'):
        torch.autograd.grad(loss,h,create_graph=True)


def test_cast_and_shift_snapshots_match_native_after_input_mutation():
    h,w,labels=inputs();a,_,_=native_boundary(h,w,labels,9);b,_=token_chunk_boundary(h,w,labels,3,9)
    with torch.no_grad():h.add_(1);w.add_(1);labels.fill_(-100)
    ag,=torch.autograd.grad(a,h,torch.tensor(128.));bg,=torch.autograd.grad(b,h,torch.tensor(128.))
    torch.testing.assert_close(ag,bg,rtol=0,atol=0,equal_nan=True)


@pytest.mark.parametrize('method',['native','chunk'])
def test_external_denominator_mutation_retains_native_saved_version_failure(method):
    h,w,labels=inputs();denominator=torch.tensor(9,dtype=torch.int64)
    loss=(native_boundary(h,w,labels,denominator)[0] if method=='native'
          else token_chunk_boundary(h,w,labels,3,denominator)[0])
    denominator.add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        torch.autograd.grad(loss,h)
