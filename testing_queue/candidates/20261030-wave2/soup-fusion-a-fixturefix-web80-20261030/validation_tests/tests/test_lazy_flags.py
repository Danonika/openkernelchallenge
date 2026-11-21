"""CPU-only lazy-storage guard fixtures. CUDA metadata is mocked, never queried.

The mock forces dispatch past is_cuda while keeping all tensor operations on CPU,
so these tests distinguish the new lazy-bit guard from the ordinary CPU fallback.
"""
import os
os.environ['TRITON_INTERPRET']='1'
from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
from solution.ce_component import materialized_ce as ce
from solution.ce_component.adapter import MaterializedCELoss
from solution.ce_component.telemetry import RouteStats
from solution.lora_component.epilogue import scale_add,_ScaleAdd
from solution.nf4_component.packed_nf4 import raw_supported

torch.set_num_threads(2)


def fake_cuda():return patch.object(torch.Tensor,'is_cuda',new=property(lambda self:True))


def test_ce_lazy_negative_logit_core_and_adapter_route_before_cuda_query():
    base=torch.randn(4,2,ce.VOCABULARY,dtype=torch.float16,requires_grad=True)
    logical=torch._neg_view(base)
    labels=torch.tensor([[0,1],[0,151935],[0,-100],[0,151552]])
    assert logical.is_neg() and logical.is_contiguous()
    reference=ce.native_causal_loss(logical,labels,7)
    diag=ce.Diagnostics()
    with fake_cuda(),patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('GPU query prohibited')):
        actual=ce.materialized_causal_loss(logical,labels,7,diagnostics=diag)
    assert torch.equal(actual,reference)
    assert diag.events==[{'event':'native_forward_fallback','reason':'dtype_or_stride'}]
    ga,=torch.autograd.grad(actual,base,retain_graph=True);gr,=torch.autograd.grad(reference,base)
    assert torch.equal(ga,gr)
    class Owner:training=True
    owner=Owner();stats=RouteStats()
    wrapper=MaterializedCELoss(owner,lambda **kw:ce.native_causal_loss(kw['logits'],kw['labels'],kw['num_items_in_batch']),stats)
    with fake_cuda(),patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('GPU query prohibited')):
        loss=wrapper(logical,labels,ce.VOCABULARY,num_items_in_batch=7)
    assert torch.equal(loss,reference);assert stats.reasons['dtype_layout_or_device']==1


def test_ce_lazy_negative_upstream_uses_native_backward():
    logits=torch.randn(4,2,5,dtype=torch.float16,requires_grad=True)
    labels=torch.tensor([[0,1],[0,2],[0,-100],[0,4]])
    target=ce._targets(labels);grad=torch._neg_view(torch.tensor(3.,dtype=torch.float32))
    ctx=SimpleNamespace(saved_tensors=(logits,target,torch.zeros(8),torch.zeros(8)),diagnostics=ce.Diagnostics())
    with torch.no_grad():actual=ce._TritonSumCE.backward(ctx,grad)[0]
    native=torch.nn.functional.cross_entropy(logits.float().view(-1,5),target.view(-1),reduction='sum',ignore_index=-100)
    expected,=torch.autograd.grad(native,logits,grad_outputs=grad)
    assert torch.equal(actual,expected)
    assert ctx.diagnostics.events[-1]['event']=='native_backward_fallback'


@pytest.mark.parametrize('negative_operand',['base','delta'])
def test_lora_lazy_negative_is_native_before_device_query(negative_operand):
    base=torch.randn(2,256,dtype=torch.float32,requires_grad=True)
    delta=torch.randn(2,256,dtype=torch.float16,requires_grad=True)
    if negative_operand=='base':base=torch._neg_view(base)
    else:delta=torch._neg_view(delta)
    reference=base+delta*2.
    with fake_cuda(),patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('GPU query prohibited')),patch.object(_ScaleAdd,'apply',side_effect=AssertionError('Optimized path prohibited')):
        actual=scale_add(base,delta,2.)
    assert torch.equal(actual,reference)
    got=torch.autograd.grad(actual,(base,delta),torch.ones_like(actual),retain_graph=True)
    want=torch.autograd.grad(reference,(base,delta),torch.ones_like(reference))
    assert all(torch.equal(a,b) for a,b in zip(got,want))


@pytest.mark.parametrize('metadata_index',[1,2,3])
def test_nf4_lazy_negative_float_metadata_is_rejected(metadata_index):
    shape=(256,1536);n=shape[0]*shape[1]
    packed=torch.zeros(n//2,1,dtype=torch.uint8)
    metadata=[torch.zeros(n//64,dtype=torch.uint8),torch.linspace(-1,1,256),torch.ones((n//64+255)//256),torch.tensor([.1])]
    metadata[metadata_index]=torch._neg_view(metadata[metadata_index])
    assert metadata[metadata_index].is_neg() and metadata[metadata_index].is_contiguous()
    with fake_cuda(),patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('GPU query prohibited')):
        assert not raw_supported(packed,shape,*metadata,torch.float16)
