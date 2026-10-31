"""Sparse cached-head control checks; no timing or admission claim.

Native CPU tests use actual Torch projection, CE, and backward. The fused
dispatch smoke replaces only the device CE helper with native Torch to test
its call contract. Physical T4 tests use the real chosen consumer and compare
the changed backward GEMM to the full native head using predeclared FP16 norm
bounds. Those component bounds do not replace official training quality.
"""

import json
import os

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from . import sparse_cached_head as sparse
from .cached_head import cached_logits_ce


INTERPRETING = os.environ.get("TRITON_INTERPRET")=="1"
PHYSICAL = torch.cuda.is_available() and not INTERPRETING
DEVICE = "cuda" if PHYSICAL else "cpu"
HAS_TRITON = PHYSICAL or INTERPRETING


def _reference(hidden,weight,labels,denominator=None):
    scores = F.linear(hidden.half(),weight.half()).reshape(-1,weight.shape[0])
    targets = F.pad(labels,(0,1),value=-100)[...,1:].contiguous().reshape(-1)
    lp = F.log_softmax(scores.float(),dim=-1)
    token = F.nll_loss(lp,targets,reduction="none")
    divisor = (targets!=-100).sum().float() if denominator is None else denominator
    return (token.sum()/divisor,token.reshape_as(labels),
            -(lp.exp()*lp).sum(-1).reshape_as(labels),scores.argmax(-1).reshape_as(labels))


def _norm_check(actual,expected):
    assert torch.equal(torch.isnan(actual),torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual),torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual),torch.isneginf(expected))
    finite = torch.isfinite(actual)&torch.isfinite(expected)
    a,r = actual[finite].float(),expected[finite].float()
    if not a.numel():
        return
    error = a-r
    norm = torch.linalg.vector_norm(r)
    maximum = r.abs().max()
    if norm.item()==0:
        assert torch.equal(a,r)
        return
    # Fixed FP16-level diagnostic bounds, independent of observed errors.
    # They are not the official 3% aggregate training-curve criterion.
    epsilon = torch.finfo(torch.float16).eps
    assert torch.linalg.vector_norm(error).item() <= 2*epsilon*norm.item()
    assert error.abs().max().item() <= 4*epsilon*maximum.item()


class _MatmulCalls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []
    def __torch_dispatch__(self,function,types,args=(),kwargs=None):
        result = function(*args,**(kwargs or {}))
        if function is torch.ops.aten.mm.default:
            self.calls.append((tuple(args[0].shape),tuple(args[1].shape),result.dtype))
        return result


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32])
@pytest.mark.parametrize("scale",[1.,65536.])
@pytest.mark.parametrize("denominator",[None,17])
def test_sparse_cached_native_metrics_and_scaled_hidden_gradient(
        dtype,scale,denominator,monkeypatch,record_property):
    torch.manual_seed(659)
    values = (torch.randn(2,7,33,device=DEVICE)*.2).to(dtype)
    weight = (torch.randn(137,33,device=DEVICE)*.1).half()
    labels = torch.randint(0,137,(2,7),device=DEVICE)
    labels[0,:4] = -100
    labels[1,4:] = -100
    reference = values.clone().requires_grad_()
    expected = _reference(reference,weight,labels,denominator)
    expected[0].backward(torch.tensor(scale,device=DEVICE))
    # Keep the full, independent F.linear/native-statistics reference above.
    # Native CUDA row reductions can choose different schedules for 14 rows
    # and chunks of 3. R8's former exact comparison against 14 rows failed for
    # two of 14 statistics by 4.76837158203125e-7. Test projection identity and
    # SAME-CHUNK native statistics exactly, while retaining the independently
    # defined full native comparison at the pre-existing cached-head bound.
    original_summaries = sparse.cached_head._summaries
    with torch.no_grad():
        reference_scores = F.linear(values.half(),weight).reshape(-1,weight.shape[0])
        targets = F.pad(labels,(0,1),value=-100)[...,1:].contiguous().reshape(-1)
        native_stats = original_summaries(reference_scores,targets,3,"native")
        divisor = (targets!=-100).sum().float() if denominator is None else denominator
        native_expected = (native_stats[0].sum()/divisor,
                           native_stats[0].reshape_as(labels),
                           native_stats[1].reshape_as(labels))
    projection_checks = []
    def observed_summaries(scores,actual_targets,chunk,numerics):
        assert chunk==3 and numerics=="native"
        projection_checks.append(tuple(scores.shape))
        record_property("soup_full_native_logits_exact",torch.equal(scores,reference_scores))
        torch.testing.assert_close(scores,reference_scores,rtol=0,atol=0)
        assert torch.equal(actual_targets,targets)
        return original_summaries(scores,actual_targets,chunk,numerics)
    monkeypatch.setattr(sparse.cached_head,"_summaries",observed_summaries)
    hidden = values.clone().requires_grad_()
    result = sparse.sparse_cached_linear_ce(hidden,weight,labels,denominator,chunk_size=3)
    result.loss.backward(torch.tensor(scale,device=DEVICE))
    actual_stats = (result.loss,result.token_loss,result.entropy)
    old_exact_comparison = {}
    for name,actual,wanted in zip(("loss","token_loss","entropy"),actual_stats,expected[:3]):
        difference = (actual.double()-wanted.double()).abs()
        old_exact_comparison[name] = {
            "exact":torch.equal(actual,wanted),
            "mismatched":int((actual!=wanted).sum().item()),
            "elements":actual.numel(),
            "max_abs":difference.max().item(),
            "max_rel":(difference/wanted.double().abs().clamp_min(torch.finfo(torch.float64).tiny)).max().item(),
        }
    # Persist the original exact-oracle discrepancies in JUnit even if a later
    # strict same-chunk or unchanged full-native bound assertion fails.
    record_property("soup_unchunked_native_statistics_exact_comparison",
                    json.dumps(old_exact_comparison,sort_keys=True))
    record_property("soup_native_statistics_chunk_rows",3)
    record_property("soup_native_statistics_unchunked_bound","rtol=3e-6,atol=3e-6; established test_cached_head bound")
    assert projection_checks==[(14,137)]
    for actual,wanted in zip(actual_stats,native_expected):
        torch.testing.assert_close(actual,wanted,rtol=0,atol=0,equal_nan=True)
    for actual,wanted in zip(actual_stats,expected[:3]):
        torch.testing.assert_close(actual,wanted,rtol=3e-6,atol=3e-6,equal_nan=True)
    assert torch.equal(result.predictions,native_stats[2].reshape_as(labels))
    assert torch.equal(result.predictions,expected[3])
    _norm_check(hidden.grad,reference.grad)
    assert hidden.grad.dtype==dtype and weight.grad is None
    assert all(not x.requires_grad for x in (result.token_loss,result.entropy,result.predictions))
    targets = F.pad(labels,(0,1),value=-100)[...,1:]
    assert torch.equal(hidden.grad[targets==-100],torch.zeros_like(hidden.grad[targets==-100]))


@pytest.mark.parametrize("gap",[16.53125,24.])
@pytest.mark.parametrize("scale",[1.,65536.])
def test_sparse_cached_full_vocab_cpu_native_boundary_control(gap,scale):
    # Actual CPU tensors at the previously problematic vocabulary size.
    # A one-channel projection isolates CE and sparse dH from GEMM tiling.
    values = torch.ones(1,3,1)
    weight = torch.full((151936,1),32-gap,dtype=torch.float16)
    weight[0,0] = 32
    labels = torch.tensor([[-100,0,-100]])
    reference = values.clone().requires_grad_()
    expected = _reference(reference,weight,labels,2048)
    expected[0].backward(torch.tensor(scale))
    hidden = values.clone().requires_grad_()
    result = sparse.sparse_cached_linear_ce(hidden,weight,labels,2048,2)
    result.loss.backward(torch.tensor(scale))
    for actual,wanted in zip((result.loss,result.token_loss,result.entropy),expected[:3]):
        torch.testing.assert_close(actual,wanted,rtol=0,atol=0,equal_nan=True)
    assert torch.equal(result.predictions,expected[3])
    _norm_check(hidden.grad,reference.grad)


@pytest.mark.parametrize("sparse_backward",[False,True])
def test_sparse_cached_one_full_forward_one_selected_backward_and_saved_payload(monkeypatch,sparse_backward):
    hidden = torch.zeros(2,4,9,requires_grad=True)
    with torch.no_grad():
        hidden[...,0] = torch.arange(1,9).reshape(2,4)
    weight = torch.zeros(17,9,dtype=torch.float16)
    weight[:,0] = torch.arange(1,18)/1000
    labels = torch.full((2,4),-100,dtype=torch.long)
    labels[0,-1],labels[1,-1] = 1,2
    saved,seen_targets,scales = [],[],[]
    original = sparse._native_logit_gradient
    def observed(scores,targets,scale):
        seen_targets.append(targets.clone())
        scales.append(scale.item())
        assert scores.shape[0]<=3
        return original(scores,targets,scale)
    monkeypatch.setattr(sparse,"_native_logit_gradient",observed)
    def pack(value):
        saved.append((value.data_ptr(),tuple(value.shape),value.dtype))
        return value
    mm = _MatmulCalls()
    with mm:
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value):
            result = sparse.sparse_cached_linear_ce(hidden,weight,labels,2048,3,
                                                     sparse_backward=sparse_backward)
        result.loss.backward(torch.tensor(65536.))
    work_rows = 2 if sparse_backward else 8
    assert mm.calls == [((8,9),(9,17),torch.float16),
                        ((work_rows,17),(17,9),torch.float16)]
    assert len(saved)==6
    assert (hidden.data_ptr(),(2,4,9),torch.float32) in saved
    assert (weight.data_ptr(),(17,9),torch.float16) in saved
    assert any(shape==(8,17) and dtype==torch.float16 for _,shape,dtype in saved)
    assert not any(shape==(8,17) and dtype==torch.float32 for _,shape,dtype in saved)
    assert any(shape==(8,) and dtype==torch.bool for _,shape,dtype in saved)
    expected_targets = torch.tensor([1,2]) if sparse_backward else F.pad(labels,(0,1),value=-100)[...,1:].reshape(-1)
    assert torch.equal(torch.cat(seen_targets),expected_targets)
    assert all(value==32 for value in scales)
    nonzero = torch.nonzero(hidden.grad.reshape(8,9).abs().sum(-1),as_tuple=False).flatten()
    assert torch.equal(nonzero,torch.tensor([2,6]))


@pytest.mark.parametrize("denominator",[None,0,7])
@pytest.mark.parametrize("scale",[1.,65536.])
def test_sparse_cached_all_ignored_finite_rows_skip_backward_gemm(denominator,scale):
    hidden = torch.randn(1,4,9,device=DEVICE,requires_grad=True)
    weight = torch.randn(17,9,device=DEVICE).half()
    labels = torch.full((1,4),-100,device=DEVICE,dtype=torch.long)
    mm = _MatmulCalls()
    with mm:
        result = sparse.sparse_cached_linear_ce(hidden,weight,labels,denominator,2)
        result.loss.backward(torch.tensor(scale,device=DEVICE))
    assert len(mm.calls)==1
    assert torch.equal(hidden.grad,torch.zeros_like(hidden.grad))
    assert torch.equal(result.token_loss,torch.zeros_like(result.token_loss))
    assert torch.isnan(result.loss) if denominator in (None,0) else result.loss.item()==0


@pytest.mark.parametrize("kind",["weight_positive_inf","weight_negative_inf","weight_nan","hidden_nan","half_overflow"])
@pytest.mark.parametrize("numerics",["native","guarded"])
def test_sparse_cached_unsafe_ignored_rows_preserve_native_nan_dh(kind,numerics):
    if numerics=="guarded" and not HAS_TRITON:
        pytest.skip("guarded forward requires CUDA or explicit interpreter")
    values = torch.ones(1,3,9,device=DEVICE)
    weight = torch.zeros(17,9,device=DEVICE,dtype=torch.float16)
    weight[:,0] = torch.linspace(-.1,.1,17,device=DEVICE)
    labels = torch.full((1,3),-100,device=DEVICE,dtype=torch.long)
    if kind.startswith("weight_"):
        weight[0,0] = {"weight_positive_inf":float("inf"),
                       "weight_negative_inf":-float("inf"),"weight_nan":float("nan")}[kind]
    elif kind=="hidden_nan":
        values[0,-1,0] = float("nan")
    else:
        values.fill_(200)
        weight.fill_(200)
    reference = values.clone().requires_grad_()
    expected = _reference(reference,weight,labels,7)
    expected[0].backward(torch.tensor(65536.,device=DEVICE))
    hidden = values.clone().requires_grad_()
    result = sparse.sparse_cached_linear_ce(hidden,weight,labels,7,2,numerics=numerics)
    result.loss.backward(torch.tensor(65536.,device=DEVICE))
    torch.testing.assert_close(result.token_loss,expected[1],rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(result.entropy,expected[2],rtol=0,atol=0,equal_nan=True)
    assert torch.equal(result.predictions,expected[3])
    _norm_check(hidden.grad,reference.grad)
    assert torch.isnan(hidden.grad[0,-1,0])


def test_sparse_cached_fused_consumer_dispatch_uses_upstream_before_half(monkeypatch):
    calls = []
    def native_test_substitute(scores,targets,divisor,upstream,chunk_size=128):
        calls.append((scores.shape[0],divisor.dtype,upstream.dtype,(upstream/divisor).item()))
        return sparse._native_logit_gradient(scores,targets,upstream/divisor)
    monkeypatch.setattr(sparse,"native_logsoftmax_ce_backward",native_test_substitute)
    hidden = torch.randn(2,4,9,requires_grad=True)
    weight = torch.randn(17,9).half()
    labels = torch.tensor([[-100,2,3,-100],[-100,4,-100,-100]])
    actual = sparse.sparse_cached_linear_ce(hidden,weight,labels,2048,2,backward_mode="native_fused")
    actual.loss.backward(torch.tensor(65536.))
    assert calls == [(2,torch.float32,torch.float32,32.),(1,torch.float32,torch.float32,32.)]
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("backward_mode",["native","native_fused"])
def test_sparse_cached_higher_order_replays_full_native_head_before_selection(monkeypatch,backward_mode):
    def forbidden(*args,**kwargs):
        raise AssertionError("higher-order branch must bypass compaction and custom CE")
    monkeypatch.setattr(sparse,"native_logsoftmax_ce_backward",forbidden)
    monkeypatch.setattr(torch,"nonzero",forbidden)
    values = torch.tensor([[[.2,-.3],[.7,.6],[.1,.2]]])
    weight = torch.tensor([[.1,.7],[-.2,.3],[.6,-.5]],dtype=torch.float16)
    labels = torch.tensor([[-100,2,1]])
    probe = torch.tensor([[[1.,2.],[4.,-2.],[.7,.2]]])
    reference = values.clone().requires_grad_()
    first, = torch.autograd.grad(_reference(reference,weight,labels,3)[0],reference,create_graph=True)
    second, = torch.autograd.grad((first*probe).sum(),reference)
    hidden = values.clone().requires_grad_()
    mm = _MatmulCalls()
    with mm:
        result = sparse.sparse_cached_linear_ce(hidden,weight,labels,3,1,backward_mode=backward_mode)
        actual_first, = torch.autograd.grad(result.loss,hidden,create_graph=True)
    assert sum(a==(3,2) and b==(2,3) for a,b,_ in mm.calls)==2
    actual_second, = torch.autograd.grad((actual_first*probe).sum(),hidden)
    torch.testing.assert_close(actual_first,first,rtol=0,atol=0)
    torch.testing.assert_close(actual_second,second,rtol=0,atol=0)


@pytest.mark.skipif(not HAS_TRITON,reason="guarded forward needs CUDA or the explicit interpreter")
def test_sparse_cached_guarded_stats_still_cover_all_rows(monkeypatch):
    original,calls = sparse.cached_head._summaries,[]
    def observed(scores,targets,chunk,numerics):
        calls.append((tuple(scores.shape),targets.numel(),numerics))
        return original(scores,targets,chunk,numerics)
    monkeypatch.setattr(sparse.cached_head,"_summaries",observed)
    torch.manual_seed(147)
    hidden = (torch.randn(1,5,9,device=DEVICE)*.1).requires_grad_()
    weight = (torch.randn(137,9,device=DEVICE)*.1).half()
    labels = torch.tensor([[-100,-100,3,-100,-100]],device=DEVICE)
    result = sparse.sparse_cached_linear_ce(hidden,weight,labels,7,2,numerics="guarded")
    expected = _reference(hidden,weight,labels,7)
    assert calls == [((5,137),5,"guarded")]
    for actual,wanted in zip((result.loss,result.token_loss,result.entropy),expected[:3]):
        torch.testing.assert_close(actual,wanted,rtol=3e-6,atol=3e-6)
    assert torch.equal(result.predictions,expected[3])
    result.loss.backward()


def test_sparse_cached_metadata_guards_before_projection(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("unsupported input must not project")
    monkeypatch.setattr(torch,"mm",forbidden)
    h,w,y = torch.zeros(1,2,3),torch.zeros(7,3,dtype=torch.float16),torch.tensor([[-100,1]])
    with pytest.raises(ValueError,match="frozen"):
        sparse.sparse_cached_linear_ce(h,w.requires_grad_(),y)
    w.requires_grad_(False)
    with pytest.raises(ValueError,match="native torch"):
        sparse.sparse_cached_linear_ce(h,w,y,projection="triton")
    with pytest.raises(ValueError,match="backward_mode"):
        sparse.sparse_cached_linear_ce(h,w,y,backward_mode="unknown")
    with pytest.raises(ValueError,match="positive"):
        sparse.sparse_cached_linear_ce(h,w,y,chunk_size=0)
    with pytest.raises(ValueError,match="scalar"):
        sparse.sparse_cached_linear_ce(h,w,y,num_items_in_batch=torch.ones(2))
    with pytest.raises(ValueError,match="contiguous"):
        sparse.sparse_cached_linear_ce(h,torch.zeros(3,7,dtype=torch.float16).t(),y)
    with pytest.raises(RuntimeError,match="out of bounds"):
        sparse.sparse_cached_linear_ce(h,w,torch.tensor([[-100,7]]))


@pytest.mark.skipif(not PHYSICAL,reason="requires physical T4 full vocabulary and changed GEMM geometry")
@pytest.mark.parametrize("backward_mode",["native","native_fused"])
@pytest.mark.parametrize("scale",[1.,65536.])
def test_sparse_cached_physical_full_vocabulary_scaled_gradient_norm(backward_mode,scale):
    assert torch.cuda.get_device_capability()==(7,5)
    torch.manual_seed(863)
    length = 256 if scale==1 else 1024
    values = (torch.randn(2,length,1536,device="cuda")*.2)
    weight = (torch.randn(151936,1536,device="cuda",dtype=torch.float16)*.03)
    labels = torch.randint(0,151936,(2,length),device="cuda")
    labels[:,:3*length//8] = -100
    labels[1,3*length//4:] = -100
    reference = values.clone().requires_grad_()
    with torch.autocast("cuda",enabled=False):
        logits = torch.mm(reference.reshape(-1,1536).half(),weight.t()).reshape(2,length,-1)
    expected = cached_logits_ce(logits,labels,2048,128,numerics="native",backward_mode="native")
    expected.loss.backward(torch.tensor(scale,device="cuda"))
    del logits
    hidden = values.clone().requires_grad_()
    result = sparse.sparse_cached_linear_ce(hidden,weight,labels,2048,128,
                                             backward_mode=backward_mode)
    result.loss.backward(torch.tensor(scale,device="cuda"))
    for actual,wanted in zip((result.loss,result.token_loss,result.entropy),
                              (expected.loss,expected.token_loss,expected.entropy)):
        torch.testing.assert_close(actual,wanted,rtol=0,atol=0,equal_nan=True)
    assert torch.equal(result.predictions,expected.predictions)
    _norm_check(hidden.grad,reference.grad)
