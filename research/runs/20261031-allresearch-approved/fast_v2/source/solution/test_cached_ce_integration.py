"""Control-flow integration checks for the opt-in cached CE backward.

CPU tests replace only the new device consumer with actual native CPU CE
autograd. They verify dispatch, scalar scale/divisor handling, native higher
derivatives, a real Qwen/PEFT head call, and pre-decoder mode rejection.
They do not establish native_fused CUDA numerical parity or timing.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from . import cached_head, cached_ce_backward, integration


def _native(logits, labels, denominator):
    targets = F.pad(labels, (0,1), value=-100)[...,1:].contiguous().view(-1)
    return F.cross_entropy(logits.float().view(-1,logits.shape[-1]), targets,
                           reduction="sum") / denominator


def _cpu_consumer(calls):
    def consume(flat, targets, divisor, grad_loss, chunk_size=128):
        calls.append(dict(shape=tuple(flat.shape), dtype=flat.dtype,
                          grad_enabled=torch.is_grad_enabled(),
                          divisor=divisor.detach().clone(), upstream=grad_loss.detach().clone(),
                          chunk=chunk_size))
        with torch.enable_grad():
            leaf = flat.detach().requires_grad_()
            loss = F.cross_entropy(leaf.float(),targets,reduction="sum") / divisor
            gradient, = torch.autograd.grad(loss,leaf,grad_outputs=grad_loss)
        return gradient
    return consume


@pytest.mark.parametrize("mode", ["native", "native_fused"])
def test_cached_ce_integration_first_order_dispatch_scale_and_six_returns(monkeypatch, mode):
    calls = []
    monkeypatch.setattr(cached_ce_backward,"native_logsoftmax_ce_backward",_cpu_consumer(calls))
    torch.manual_seed(841)
    values = (torch.randn(2,4,17)*.8).half()
    labels = torch.tensor([[-100,2,8,-100],[-100,1,3,4]])
    reference = values.clone().requires_grad_()
    expected, = torch.autograd.grad(_native(reference,labels,2048),reference,
                                    grad_outputs=torch.tensor(65536.))
    actual = values.clone().requires_grad_()
    result = cached_head.cached_logits_ce(actual,labels,2048,2,backward_mode=mode)
    gradient, = torch.autograd.grad(result.loss,actual,grad_outputs=torch.tensor(65536.))
    torch.testing.assert_close(gradient,expected,rtol=0,atol=0)
    assert len(calls) == int(mode == "native_fused")
    if calls:
        call = calls[0]
        assert call["shape"] == (8,17) and call["dtype"] == torch.float16
        assert call["grad_enabled"] is False and call["chunk"] == 2
        assert call["upstream"].dtype == call["divisor"].dtype == torch.float32
        assert call["upstream"].item() == 65536.
        assert (call["upstream"]/call["divisor"]).item() == 32.
    # This branch must also match the six input slots, without accessing ctx.
    assert cached_head._CachedLogitsCrossEntropy.backward(
        SimpleNamespace(),None,None,None,None) == (None,)*6


def test_cached_ce_integration_higher_order_always_bypasses_fused_consumer(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("higher-order derivative must retain native graph")
    monkeypatch.setattr(cached_ce_backward,"native_logsoftmax_ce_backward",forbidden)
    values = torch.tensor([[[.2,-.3,.8],[.7,.6,-.2],[.1,.2,.3]]],dtype=torch.float16)
    labels = torch.tensor([[-100,2,1]])
    probe = torch.tensor([[[1.,2.,3.],[4.,-2.,.5],[.7,.2,1.]]],dtype=torch.float16)
    reference = values.clone().requires_grad_()
    first, = torch.autograd.grad(_native(reference,labels,3),reference,create_graph=True)
    second, = torch.autograd.grad((first*probe).sum(),reference)
    actual = values.clone().requires_grad_()
    result = cached_head.cached_logits_ce(actual,labels,3,1,backward_mode="native_fused")
    actual_first, = torch.autograd.grad(result.loss,actual,create_graph=True)
    actual_second, = torch.autograd.grad((actual_first*probe).sum(),actual)
    torch.testing.assert_close(actual_first,first,rtol=0,atol=0)
    torch.testing.assert_close(actual_second,second,rtol=0,atol=0)


def test_cached_ce_integration_invalid_mode_rejected_before_statistics(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid mode must be rejected before computing statistics")
    monkeypatch.setattr(cached_head,"_summaries",forbidden)
    with pytest.raises(ValueError,match="backward_mode"):
        cached_head.cached_logits_ce(torch.zeros(1,2,3,dtype=torch.float16),
                                     torch.tensor([[-100,1]]),backward_mode="unknown")


def test_cached_ce_integration_invalid_profile_rejected_before_model_unwrap(monkeypatch):
    class CudaMetadataOnlyTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True
    class UnreachableModel(torch.nn.Module):
        def get_base_model(self):
            raise AssertionError("unsupported mode must be rejected before model unwrap/forward")
    # Explicit CPU-only metadata admission fixture, never a real CUDA tensor.
    tensor = torch.ones(1,2,dtype=torch.long).as_subclass(CudaMetadataOnlyTensor)
    inputs = dict(input_ids=tensor,labels=tensor,attention_mask=tensor)
    trainer = SimpleNamespace(args=SimpleNamespace(fp16=True,bf16=False,loss_type="nll"))
    monkeypatch.setattr(integration,"HEAD_MODE","cached")
    monkeypatch.setattr(integration,"HEAD_NUMERICS","native")
    monkeypatch.setattr(integration,"CACHED_CE_BACKWARD","unknown")
    assert integration._supported_causal_lm(trainer,UnreachableModel(),inputs,torch.nn.Module) is None


def test_cached_ce_integration_real_qwen_peft_passes_mode_and_keeps_one_head(monkeypatch):
    from .test_integration import _tiny_peft_model, _batch

    calls = []
    monkeypatch.setattr(cached_ce_backward,"native_logsoftmax_ce_backward",_cpu_consumer(calls))
    monkeypatch.setattr(integration,"CACHED_CE_BACKWARD","native_fused")
    monkeypatch.setattr(integration,"HEAD_NUMERICS","native")
    monkeypatch.setattr(integration,"CACHED_CE_CHUNK_TOKENS",3)
    monkeypatch.setattr(integration,"MLP_PADDING_COMPACTION",False)
    model, base = _tiny_peft_model()
    batch = _batch()
    original = type(base).forward
    while getattr(original,"_soup_candidate_head",False):
        original = original.__wrapped__
    torch.manual_seed(847)
    with torch.autocast("cpu",dtype=torch.float16):
        native = original(base,**batch,use_cache=False,num_items_in_batch=2048)
    native.loss.backward(torch.tensor(65536.))
    expected = {name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()
                if parameter.requires_grad}
    model.zero_grad(set_to_none=True)
    head_calls, decoder_calls = [], []
    head_hook = base.lm_head.register_forward_hook(
        lambda module,inputs,outputs:head_calls.append(tuple(outputs.shape)))
    decoder_hook = base.model.register_forward_pre_hook(
        lambda module,inputs:decoder_calls.append(True))
    absent = object()
    before = base.__dict__.get("_loss_function",absent)
    try:
        torch.manual_seed(847)
        with torch.autocast("cpu",dtype=torch.float16):
            outputs,result = integration._forward_cached(
                base,original,**batch,use_cache=False,num_items_in_batch=2048)
        result.loss.backward(torch.tensor(65536.))
    finally:
        head_hook.remove()
        decoder_hook.remove()
    assert head_calls == [(2,5,67)] and decoder_calls == [True]
    assert len(calls) == 1 and calls[0]["shape"] == (10,67)
    assert (calls[0]["upstream"]/calls[0]["divisor"]).item() == 32.
    assert outputs.logits is None and "logits" not in outputs
    assert base.__dict__.get("_loss_function",absent) is before
    torch.testing.assert_close(result.loss,native.loss,rtol=3e-6,atol=3e-6)
    for name,parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad,expected[name],rtol=0,atol=0)
