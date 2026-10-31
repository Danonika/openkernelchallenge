"""Standalone CE lazy-storage guards; CPU metadata shims, never CUDA queries."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from solution import materialized_ce as ce
from solution.adapter import MaterializedCELoss
from solution.telemetry import RouteStats

torch.set_num_threads(2)


def fake_cuda():return patch.object(torch.Tensor,'is_cuda',new=property(lambda self:True))


def test_ce_lazy_negative_logit_core_and_adapter_before_cuda_query():
    base=torch.randn(4,2,ce.VOCABULARY,dtype=torch.float16,requires_grad=True)
    logical=torch._neg_view(base);labels=torch.tensor([[0,1],[0,151935],[0,-100],[0,151552]])
    assert logical.is_neg() and logical.is_contiguous()
    reference=ce.native_causal_loss(logical,labels,7);diag=ce.Diagnostics()
    with fake_cuda(),patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('GPU query prohibited')):
        actual=ce.materialized_causal_loss(logical,labels,7,diagnostics=diag)
    assert torch.equal(actual,reference)
    assert diag.events==[{'event':'native_forward_fallback','reason':'dtype_or_stride'}]
    a,=torch.autograd.grad(actual,base,retain_graph=True);b,=torch.autograd.grad(reference,base)
    assert torch.equal(a,b)
    class Owner:training=True
    owner=Owner();stats=RouteStats()
    wrapper=MaterializedCELoss(owner,lambda **kw:ce.native_causal_loss(kw['logits'],kw['labels'],kw['num_items_in_batch']),stats)
    with fake_cuda(),patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('GPU query prohibited')):
        loss=wrapper(logical,labels,ce.VOCABULARY,num_items_in_batch=7)
    assert torch.equal(loss,reference) and stats.reasons['dtype_layout_or_device']==1


def test_ce_lazy_negative_upstream_native_backward():
    logits=torch.randn(4,2,5,dtype=torch.float16,requires_grad=True)
    labels=torch.tensor([[0,1],[0,2],[0,-100],[0,4]]);target=ce._targets(labels)
    grad=torch._neg_view(torch.tensor(3.,dtype=torch.float32))
    ctx=SimpleNamespace(saved_tensors=(logits,target,torch.zeros(8),torch.zeros(8)),diagnostics=ce.Diagnostics())
    with torch.no_grad():actual=ce._TritonSumCE.backward(ctx,grad)[0]
    native=torch.nn.functional.cross_entropy(logits.float().view(-1,5),target.view(-1),reduction='sum',ignore_index=-100)
    expected,=torch.autograd.grad(native,logits,grad_outputs=grad)
    assert torch.equal(actual,expected) and ctx.diagnostics.events[-1]['event']=='native_backward_fallback'
