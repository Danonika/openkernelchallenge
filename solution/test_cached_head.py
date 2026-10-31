"""Numerical and consumer checks for the retained-native-head candidate.

CPU launches require TRITON_INTERPRET=1 before process startup.  They establish
only interpreter/reference behavior.  These same core tests use CUDA when it
is actually available; no CPU result is a substitute for a T4 run.
"""

from functools import wraps
import os

import pytest
import torch
import torch.nn.functional as F

from . import integration
from .cached_head import cached_logits_ce
from .test_integration import _batch, _tiny_peft_model, _trainer, patched_classes


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
CAN_RUN = torch.cuda.is_available() or INTERPRETING
DEVICE = "cuda" if torch.cuda.is_available() and not INTERPRETING else "cpu"
needs_kernel = pytest.mark.skipif(not CAN_RUN, reason="requires CUDA or TRITON_INTERPRET=1")


def _native(logits, labels, denominator):
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    logp = F.log_softmax(logits.float(), dim=-1)
    token_loss = F.nll_loss(logp.view(-1, logits.shape[-1]), shifted.view(-1),
                            reduction="none").view_as(labels)
    divisor = (shifted != -100).sum().float() if denominator is None else denominator
    return (token_loss.sum() / divisor, token_loss,
            -(logp.exp() * logp).sum(-1), logits.argmax(-1))


def _assert_gradient(actual, expected):
    # CPU native row backward is exact here.  Different CUDA softmax reduction
    # schedules can differ slightly; physical tests have a bounded tolerance.
    if actual.device.type == "cpu":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=3e-6, equal_nan=True)


@needs_kernel
@pytest.mark.parametrize("vocab,chunk", [(17,1), (4099,3), (151936,2)])
@pytest.mark.parametrize("scale", [1., 1000., 65536.])
@pytest.mark.parametrize("denominator", [None, 13])
def test_cached_summary_and_scaled_native_gradient(vocab, chunk, scale, denominator):
    torch.manual_seed(763)
    base = (torch.randn(2,3,vocab,device=DEVICE) * .8).half()
    labels = torch.tensor([[-100, 3, 8], [-100, 9, -100]], device=DEVICE)
    reference = base.clone().requires_grad_()
    expected = _native(reference, labels, denominator)
    expected[0].backward(torch.tensor(scale, device=DEVICE))
    candidate = base.clone().requires_grad_()
    result = cached_logits_ce(candidate, labels, denominator, chunk)
    result.loss.backward(torch.tensor(scale, device=DEVICE))
    torch.testing.assert_close(result.loss, expected[0], rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(result.token_loss, expected[1], rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(result.entropy, expected[2], rtol=3e-6, atol=3e-6)
    assert torch.equal(result.predictions, expected[3])
    _assert_gradient(candidate.grad, reference.grad)
    assert not result.token_loss.requires_grad and not result.entropy.requires_grad
    assert torch.equal(result.token_loss[:, -1], torch.zeros(2,device=DEVICE))


@needs_kernel
@pytest.mark.parametrize("kind", ["saturated", "nan_ignored", "posinf", "neginf", "mixed_neginf"])
@pytest.mark.parametrize("denominator", [None, 7])
def test_saturated_and_nonfinite_rows_keep_native_behavior(kind, denominator):
    z = torch.zeros(1,3,151936 if kind == "saturated" else 23,
                     device=DEVICE,dtype=torch.float16)
    labels = torch.tensor([[-100, 0, 4]],device=DEVICE)
    if kind == "saturated":
        z[0,0,0] = 20
        z[0,1,4] = 19
    elif kind == "nan_ignored":
        labels[0,1] = -100
        z[0,0,7] = float("nan")
    elif kind == "posinf":
        z[0,0,0] = float("inf")
    elif kind == "neginf":
        z[0,0,:] = -float("inf")
    else:
        z[0,0,0] = -float("inf")
    ref = z.clone().requires_grad_()
    expected = _native(ref,labels,denominator)
    expected[0].backward(torch.tensor(65536.,device=DEVICE))
    candidate = z.clone().requires_grad_()
    result = cached_logits_ce(candidate,labels,denominator,1)
    result.loss.backward(torch.tensor(65536.,device=DEVICE))
    for actual,wanted in zip((result.loss,result.token_loss,result.entropy),expected[:3]):
        torch.testing.assert_close(actual,wanted,rtol=3e-6,atol=3e-6,equal_nan=True)
    assert torch.equal(result.predictions,expected[3])
    _assert_gradient(candidate.grad,ref.grad)


@needs_kernel
@pytest.mark.parametrize("denominator", [None,0,9])
def test_all_ignored_denominator_semantics(denominator):
    z = torch.zeros(1,4,31,device=DEVICE,dtype=torch.float16,requires_grad=True)
    labels = torch.full((1,4),-100,device=DEVICE,dtype=torch.long)
    result = cached_logits_ce(z,labels,denominator,2)
    if denominator in (None,0):
        assert torch.isnan(result.loss)
    else:
        assert result.loss.item() == 0
    result.loss.backward(torch.tensor(65536.,device=DEVICE))
    assert torch.equal(z.grad,torch.zeros_like(z))
    assert torch.equal(result.token_loss,torch.zeros(1,4,device=DEVICE))


@needs_kernel
def test_saves_original_half_payload_and_no_full_float_payload():
    z = torch.randn(2,4,71,device=DEVICE).half().requires_grad_()
    labels = torch.tensor([[-100,2,3,4],[-100,5,6,-100]],device=DEVICE)
    saved=[]
    def pack(tensor):
        saved.append((tensor.data_ptr(),tuple(tensor.shape),tensor.dtype))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        result = cached_logits_ce(z,labels,8,2)
    assert (z.data_ptr(),tuple(z.shape),torch.float16) in saved
    assert not any(shape == tuple(z.shape) and dtype == torch.float32 for _,shape,dtype in saved)
    assert len(saved) == 3  # original logits, shifted labels, scalar denominator
    result.loss.backward()


@needs_kernel
def test_native_head_is_called_once_and_keeps_its_full_backward_geometry():
    torch.manual_seed(188)
    head=torch.nn.Linear(11,31,bias=False,device=DEVICE)
    head.weight.requires_grad_(False)
    hidden=torch.randn(2,5,11,device=DEVICE,requires_grad=True)
    labels=torch.tensor([[-100,3,4,5,6],[-100,2,-100,7,8]],device=DEVICE)
    with torch.autocast(DEVICE,dtype=torch.float16):
        native_logits=head(hidden)
        reference=_native(native_logits,labels,11)[0]
    reference.backward(torch.tensor(1000.,device=DEVICE))
    expected=hidden.grad.clone()
    hidden.grad=None
    calls=[]
    hook=head.register_forward_hook(lambda module,inputs,outputs:calls.append(tuple(outputs.shape)))
    try:
        with torch.autocast(DEVICE,dtype=torch.float16):
            logits=head(hidden)
            result=cached_logits_ce(logits,labels,11,3)
        result.loss.backward(torch.tensor(1000.,device=DEVICE))
    finally:
        hook.remove()
    assert calls == [(2,5,31)]
    _assert_gradient(hidden.grad,expected)


@needs_kernel
def test_higher_order_uses_native_differentiable_backward():
    z=torch.tensor([[[.2,-.3,.8],[.7,.6,-.2],[.1,.2,.3]]],
                     device=DEVICE,dtype=torch.float16,requires_grad=True)
    labels=torch.tensor([[-100,2,1]],device=DEVICE)
    native=z.detach().clone().requires_grad_()
    loss=_native(native,labels,3)[0]
    first,=torch.autograd.grad(loss,native,create_graph=True)
    probe=torch.tensor([[[1.,2.,3.],[4.,-2.,.5],[.7,.2,1.]]],device=DEVICE,dtype=torch.float16)
    second,=torch.autograd.grad((first*probe).sum(),native)
    result=cached_logits_ce(z,labels,3,1)
    actual_first,=torch.autograd.grad(result.loss,z,create_graph=True)
    actual_second,=torch.autograd.grad((actual_first*probe).sum(),z)
    _assert_gradient(actual_first,first)
    _assert_gradient(actual_second,second)


def test_cached_head_rejects_incompatible_metadata_before_kernel():
    z=torch.zeros(1,2,3,dtype=torch.float16)
    labels=torch.zeros(1,2,dtype=torch.long)
    with pytest.raises(TypeError,match="FP16"):
        cached_logits_ce(z.float(),labels)
    with pytest.raises(ValueError,match="positive"):
        cached_logits_ce(z,labels,chunk_size=0)
    with pytest.raises(ValueError,match="scalar"):
        cached_logits_ce(z,labels,torch.ones(2))
    with pytest.raises(ValueError,match="contiguous"):
        cached_logits_ce(torch.zeros(1,3,2,dtype=torch.float16).transpose(1,2),labels)
    with pytest.raises(RuntimeError,match="out of bounds"):
        cached_logits_ce(z,torch.tensor([[-100,3]]))


@pytest.mark.skipif(not INTERPRETING,reason="CPU consumer test requires TRITON_INTERPRET=1")
@pytest.mark.parametrize("scale", [1.,65536.])
def test_cached_peft_accelerate_metrics_rewind_and_gradients(monkeypatch,patched_classes,scale):
    from accelerate.utils.operations import convert_outputs_to_fp32
    from soup_cli.trainer.rewind_hf import row_losses,row_losses_from_token_losses

    _,trainer_class,_,reference_compute=patched_classes
    model,base=_tiny_peft_model()
    batch=_batch()
    expected_trainer,candidate_trainer=_trainer(model),_trainer(model)
    original_forward=model.forward
    visits=[]
    head_calls=[]
    @wraps(original_forward)
    def cpu_amp_forward(*args,**kwargs):
        visits.append((torch.is_autocast_enabled("cpu"),torch.is_autocast_enabled("cuda")))
        with torch.autocast("cpu",dtype=torch.float16):
            return original_forward(*args,**kwargs)
    monkeypatch.setattr(model,"forward",convert_outputs_to_fp32(cpu_amp_forward))
    hook=base.lm_head.register_forward_hook(lambda module,inputs,outputs:head_calls.append(tuple(outputs.shape)))
    try:
        torch.manual_seed(567)
        expected_loss,expected=reference_compute(expected_trainer,model,dict(batch),
                                                  return_outputs=True,num_items_in_batch=9)
        expected_rng=torch.get_rng_state().clone()
        (expected_loss*scale).backward()
        expected_gradients={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        expected_rewind=row_losses(expected.logits,batch["labels"])
        model.zero_grad(set_to_none=True)
        monkeypatch.setattr(integration,"HEAD_MODE","cached")
        monkeypatch.setattr(integration,"CACHED_CE_CHUNK_TOKENS",3)
        monkeypatch.setattr(integration,"_supported_causal_lm",lambda *args:base)
        enabled=torch.is_autocast_enabled
        dtype=torch.get_autocast_dtype
        monkeypatch.setattr(torch,"is_autocast_enabled",lambda device="cuda":enabled("cpu") if device=="cuda" else enabled(device))
        monkeypatch.setattr(torch,"get_autocast_dtype",lambda device:torch.float16 if device=="cuda" else dtype(device))
        original_loss=base.loss_function
        existed="_loss_function" in base.__dict__
        torch.manual_seed(567)
        loss,outputs=trainer_class.compute_loss(candidate_trainer,model,dict(batch),
                                                return_outputs=True,num_items_in_batch=9)
        assert torch.equal(torch.get_rng_state(),expected_rng)
        (loss*scale).backward()
        assert base.loss_function is original_loss
        assert ("_loss_function" in base.__dict__)==existed
    finally:
        hook.remove()
    assert visits==[(False,False),(False,False)]  # outer Trainer AMP is off
    assert head_calls==[(2,5,67),(2,5,67)]
    assert outputs.logits is None and "logits" not in outputs
    assert integration._HEAD_CALL.get() is None
    torch.testing.assert_close(loss,expected_loss,rtol=3e-6,atol=3e-6)
    for name,param in model.named_parameters():
        if param.requires_grad:
            _assert_gradient(param.grad,expected_gradients[name])
    for metric in ("entropy","mean_token_accuracy","num_tokens"):
        assert candidate_trainer._metrics["train"][metric]==pytest.approx(expected_trainer._metrics["train"][metric],rel=3e-6,abs=3e-6)
    recorded=getattr(candidate_trainer,integration._REWIND_STEP)
    actual_rewind=row_losses_from_token_losses(recorded.token_loss,batch["labels"])
    assert actual_rewind[1]==expected_rewind[1]
    assert actual_rewind[0]==pytest.approx(expected_rewind[0],rel=3e-6,abs=3e-6)


def test_scoped_loss_restored_on_exception_without_forward_retry(patched_classes):
    _,_,reference_forward,_=patched_classes
    model,base=_tiny_peft_model()
    saved=base.loss_function
    existed="_loss_function" in base.__dict__
    calls=[]
    hook=base.lm_head.register_forward_hook(lambda *args:calls.append(1))
    try:
        # No autocast: deliberately unsupported FP32 output is rejected after
        # this one original forward.  No decoder retry can consume extra RNG.
        with pytest.raises(TypeError,match="FP16"):
            integration._forward_cached(base,reference_forward,**_batch(),use_cache=False)
    finally:
        hook.remove()
    assert calls==[1]
    assert base.loss_function is saved
    assert ("_loss_function" in base.__dict__)==existed
