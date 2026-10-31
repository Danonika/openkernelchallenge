"""Tiny CPU numerical/negative witnesses, not training or target performance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pytest,torch
import torch.nn.functional as F
from solution.token_chunk_head import native_boundary,token_chunk_boundary
from token_chunk_reference import consume_summaries

def fixture(seed=42):
    g=torch.Generator().manual_seed(seed)
    h=(torch.randn(2,5,8,generator=g)*.1).requires_grad_()
    w=torch.randn(17,8,generator=g)*.1
    labels=torch.tensor([[-100,1,2,-100,4],[-100,5,-100,7,8]])
    mask=torch.tensor([[1,1,1,1,1],[1,1,1,1,0]])
    return h,w,labels,mask

@pytest.mark.parametrize('scale',[1.,65536.,.125])
@pytest.mark.parametrize('denominator',[None,13])
@pytest.mark.parametrize('chunk',[1,3,7])
def test_token_chunks_preserve_native_scaled_gradient_and_metrics(scale,denominator,chunk):
    h,w,labels,mask=fixture();rng=torch.get_rng_state().clone()
    native,native_summary,raw=native_boundary(h,w,labels,denominator)
    result,summary=token_chunk_boundary(h,w,labels,chunk,denominator)
    torch.testing.assert_close(result,native,rtol=2e-6,atol=2e-6)
    ng,=torch.autograd.grad(native,h,torch.tensor(scale));cg,=torch.autograd.grad(result,h,torch.tensor(scale))
    torch.testing.assert_close(cg,ng,rtol=0,atol=0,equal_nan=True)
    for a,b in zip(consume_summaries(summary,labels,mask),consume_summaries(native_summary,labels,mask)):
        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
    assert not summary.entropy.requires_grad and not summary.token_nll.requires_grad
    assert torch.equal(rng,torch.get_rng_state())

def test_gradscaler_after_precomputed_hidden_gradient_is_not_equivalent():
    h=torch.zeros((1,1,1),requires_grad=True);w=torch.tensor([[1.],[-1.]])
    z=F.linear(h.half(),w.half());loss=F.cross_entropy(z.float().reshape(1,2),torch.tensor([0]))
    unscaled,=torch.autograd.grad(loss,h,retain_graph=True)
    scaled,=torch.autograd.grad(loss,h,torch.tensor(65536.))
    wrong=unscaled*65536
    assert torch.isneginf(scaled).all() and torch.isfinite(wrong).all()
    assert wrong.item()==-65536

def test_vocabulary_chunk_partial_half_gradients_can_nan_while_full_gradient_zero():
    w=torch.tensor([[-1.],[1.],[-2.],[-2.]])
    h=torch.zeros((1,1),requires_grad=True)
    full=F.linear(h.half(),w.half())
    chunks=torch.cat([F.linear(h.half(),part.half()) for part in w.split(2)],dim=-1)
    assert torch.equal(full,chunks)
    f=F.cross_entropy(full.float(),torch.tensor([0]));c=F.cross_entropy(chunks.float(),torch.tensor([0]))
    fg,=torch.autograd.grad(f,h,torch.tensor(65536.),retain_graph=True)
    cg,=torch.autograd.grad(c,h,torch.tensor(65536.))
    assert torch.equal(fg,torch.zeros_like(fg)) and torch.isnan(cg).all()

def test_scaling_after_fp16_logits_gradient_loses_nonselected_mass():
    z=torch.zeros((1,151936),dtype=torch.float16,requires_grad=True)
    loss=F.cross_entropy(z.float(),torch.tensor([0]),reduction='sum')/2048
    unscaled,=torch.autograd.grad(loss,z,retain_graph=True)
    scaled,=torch.autograd.grad(loss,z,torch.tensor(65536.))
    assert torch.count_nonzero(unscaled[0,1:])==0
    assert torch.count_nonzero(scaled[0,1:])==151935
    assert torch.count_nonzero((unscaled.float()*65536)[0,1:])==0

def test_native_entropy_cannot_prune_unsupervised_or_zero_attention_nonfinite_rows():
    logits=torch.tensor([[float('inf'),0.],[0.,0.]])
    logp=F.log_softmax(logits,dim=-1);entropy=-(logp.exp()*logp).sum(-1)
    assert torch.isnan((entropy*torch.tensor([0.,1.])).sum())
    assert torch.isfinite(entropy[1])

def test_weighted_exp_entropy_formula_loses_native_near_certain_tail():
    z=torch.tensor([[30.,0.]])
    logp=F.log_softmax(z,dim=-1);native=-(logp.exp()*logp).sum(-1)
    wrong=torch.logsumexp(z,-1)-(z*F.softmax(z,-1)).sum(-1)
    assert native.item()>0 and wrong.item()==0

def test_all_ignored_native_mean_nan_zero_gradient_and_rewind_zero():
    h,w,labels,mask=fixture();labels.fill_(-100)
    a,sa,_=native_boundary(h,w,labels);b,sb=token_chunk_boundary(h,w,labels,3)
    assert torch.isnan(a) and torch.isnan(b)
    ag,=torch.autograd.grad(a,h);bg,=torch.autograd.grad(b,h)
    assert torch.equal(ag,bg) and torch.count_nonzero(bg)==0
    metrics=consume_summaries(sb,labels,mask)
    assert torch.equal(metrics[2],torch.zeros(2)) and torch.equal(metrics[3],torch.zeros(2,dtype=torch.int64))

def test_causal_shift_not_across_rows_and_final_entropy_is_preserved():
    h,w,labels,mask=fixture();w.zero_()
    loss,s=token_chunk_boundary(h,w,labels,3)
    assert torch.equal(s.predictions,torch.zeros_like(labels))
    assert torch.equal(s.token_nll[:,-1],torch.zeros(2))
    assert bool((s.entropy[:,-1]>0).all())
    _,_,rows,counts=consume_summaries(s,labels,mask)
    assert counts.tolist()==[3,3]

def test_frozen_head_and_lazy_view_guards():
    h,w,labels,mask=fixture()
    for weight in (w.requires_grad_(),):
        with pytest.raises(AssertionError):token_chunk_boundary(h,weight,labels)
    w.requires_grad_(False)
    with pytest.raises(AssertionError):token_chunk_boundary(torch._neg_view(h),w,labels)

def test_upstream_tiny_lora_gradients_scale_and_dropout_rng():
    torch.manual_seed(81)
    x=torch.randn(2,5,6);a=torch.randn(2,6,requires_grad=True)*.02;b=torch.randn(8,2,requires_grad=True)*.02
    base=torch.randn(6,8)*.1;w=torch.randn(17,8)*.1
    labels=torch.tensor([[-100,1,2,-100,4],[-100,5,-100,7,8]])
    state=torch.get_rng_state()
    h=x@base+(F.dropout(x,p=.05,training=True)@a.T)@b.T*2
    loss,_,_=native_boundary(h,w,labels,13)
    ng=torch.autograd.grad(loss,(a,b),torch.tensor(65536.));after=torch.get_rng_state()
    torch.set_rng_state(state)
    h=x@base+(F.dropout(x,p=.05,training=True)@a.T)@b.T*2
    loss,_=token_chunk_boundary(h,w,labels,3,13)
    cg=torch.autograd.grad(loss,(a,b),torch.tensor(65536.))
    for aa,bb in zip(cg,ng):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    assert torch.equal(after,torch.get_rng_state())
