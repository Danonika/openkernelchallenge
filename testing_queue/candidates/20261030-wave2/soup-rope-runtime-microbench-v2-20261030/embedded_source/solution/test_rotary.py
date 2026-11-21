"""Original fixtures; interpreter results are not CUDA code-generation evidence."""
import itertools
import os
import pytest
import torch
from .cpu_contract import inputs, reference, exact
from . import rotary

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
KERNEL = pytest.mark.skipif(not INTERPRET, reason="CPU interpreter explicitly disabled")


@pytest.mark.parametrize("cb", [1,2])
@pytest.mark.parametrize("scale", [1.,65536.])
@KERNEL
def test_finite_staged_arithmetic(cb,scale):
    q,k,c,s = inputs(cb=cb)
    versions = [x._version for x in (q,k,c,s)]
    snapshots = [x.detach().clone() for x in (q,k,c,s)]
    rng = torch.get_rng_state().clone()
    expected = reference(q,k,c,s)
    actual = rotary.fused_apply_rotary_pos_emb(q,k,c,s)
    gs = tuple(torch.linspace(-scale,scale,x.numel()).reshape(x.shape) for x in expected)
    eg = torch.autograd.grad(expected,(q,k),gs)
    ag = torch.autograd.grad(actual,(q,k),gs)
    for a,e,x in zip(actual,expected,(q,k)):
        exact(a,e)
        assert a.stride() == x.stride()
        assert a.data_ptr() != x.data_ptr()
    for a,e in zip(ag,eg): exact(a,e)
    for x,version,snap in zip((q,k,c,s),versions,snapshots):
        assert x._version == version
        exact(x,snap)
    assert torch.equal(rng,torch.get_rng_state())


@KERNEL
@pytest.mark.parametrize("layout", ["contiguous","expanded","sliced","negative_view"])
def test_upstream_layout(layout):
    q,k,c,s = inputs(t=3)
    def gradient(x):
        if layout == "expanded": return torch.ones((1,1,1,1)).expand_as(x)
        if layout == "contiguous": return torch.ones(x.shape)
        g = torch.ones((*x.shape[:-1], x.shape[-1]*2))[...,::2]
        return torch._neg_view(g) if layout == "negative_view" else g
    gs = tuple(gradient(x) for x in (q,k))
    e = torch.autograd.grad(reference(q,k,c,s),(q,k),gs)
    a = torch.autograd.grad(rotary.fused_apply_rotary_pos_emb(q,k,c,s),(q,k),gs)
    for aa,ee in zip(a,e): exact(aa,ee)


@KERNEL
@pytest.mark.parametrize("rq,rk,used", [(True,True,0),(True,True,1),(False,True,1),(True,False,0)])
def test_independent_unused_outputs(rq,rk,used):
    q,k,c,s = inputs(b=1,t=1)
    q.requires_grad_(rq);k.requires_grad_(rk)
    variables = tuple(x for x in (q,k) if x.requires_grad)
    expected,actual = reference(q,k,c,s),rotary.fused_apply_rotary_pos_emb(q,k,c,s)
    assert [x.requires_grad for x in actual] == [rq,rk]
    e = torch.autograd.grad(expected[used].sum(), variables, allow_unused=True)
    a = torch.autograd.grad(actual[used].sum(), variables, allow_unused=True)
    for aa,ee in zip(a,e):
        if ee is None: assert aa is None
        else: exact(aa,ee)


@KERNEL
def test_exhaustive_signed_zero_native_slice_adjoint():
    values = torch.tensor(list(itertools.product([0.,-0.,1.,-1.],repeat=6)))
    n = len(values)
    q = torch.zeros((1,n,1,2)).transpose(1,2).requires_grad_()
    k = q.detach().clone().requires_grad_()
    c,s = values[:,:2].reshape(1,n,2).contiguous(), values[:,2:4].reshape(1,n,2).contiguous()
    g = values[:,4:].reshape_as(q)
    e,a = reference(q,k,c,s),rotary.fused_apply_rotary_pos_emb(q,k,c,s)
    for aa,ee in zip(a,e): exact(aa,ee)
    eg = torch.autograd.grad(e,(q,k),(g,g))
    ag = torch.autograd.grad(a,(q,k),(g,g))
    for aa,ee in zip(ag,eg): exact(aa,ee)
    exact(rotary.native_adjoint(g,c,s),eg[0])


@KERNEL
def test_nonfinite_classes_and_subnormals():
    vals = torch.tensor([0.,-0.,float('inf'),float('-inf'),float('nan'),1e-40,-1e-40,1.])
    q = vals.repeat(8).reshape(1,8,1,8).transpose(1,2).requires_grad_()
    k = q.detach().clone().requires_grad_()
    c = vals.repeat(8).reshape(1,8,8).contiguous()
    s = torch.flip(c,[-1]).contiguous()
    e,a = reference(q,k,c,s),rotary.fused_apply_rotary_pos_emb(q,k,c,s)
    gs = (q.detach(),k.detach())
    eg,ag = torch.autograd.grad(e,(q,k),gs),torch.autograd.grad(a,(q,k),gs)
    for aa,ee in zip((*a,*ag),(*e,*eg)): exact(aa,ee)


@KERNEL
def test_coefficient_mutation_rejected_by_autograd():
    q,k,c,s = inputs(t=1)
    a = rotary.fused_apply_rotary_pos_emb(q,k,c,s)
    c.add_(1)
    with pytest.raises(RuntimeError,match="modified by an inplace operation"):
        a[0].sum().backward()


@KERNEL
def test_higher_order_native_fallback():
    q,k,c,s = inputs(t=1)
    g = torch.ones_like(q,requires_grad=True)
    a = rotary.fused_apply_rotary_pos_emb(q,k,c,s)[0]
    e = reference(q,k,c,s)[0]
    ag, = torch.autograd.grad(a,q,g,create_graph=True)
    eg, = torch.autograd.grad(e,q,g,create_graph=True)
    exact(ag,eg)
    aa, = torch.autograd.grad(ag,g,torch.ones_like(ag))
    ee, = torch.autograd.grad(eg,g,torch.ones_like(eg))
    exact(aa,ee)


def test_metadata_rejections_and_smoke():
    from .cpu_contract import run
    assert run()["passed"]
    q,k,c,s = inputs()
    reason = lambda *a,**kw: rotary._support_reason(*a,cuda_only=False,**kw)
    assert reason(q,q,c,s) is not None
    assert reason(torch._neg_view(q),k,c,s) is not None
    assert reason(q,k,c,s,unsqueeze_dim=2) is not None
    assert reason(q.half(),k,c,s) is not None
    assert reason(q,k,c.expand(2,-1,-1),s.expand(2,-1,-1)) is not None
    assert reason(q,k,c[:,:-1],s[:,:-1]) is not None


def test_patch_idempotent_cpu_native_and_rng(monkeypatch):
    from transformers.models.qwen2 import modeling_qwen2 as module
    original = module.apply_rotary_pos_emb
    monkeypatch.setattr(module,"apply_rotary_pos_emb",original)
    q,k,c,s = inputs()
    e = original(q,k,c,s)
    rng = torch.get_rng_state().clone()
    assert rotary.patch_qwen2_rotary()
    patched = module.apply_rotary_pos_emb
    assert rotary.patch_qwen2_rotary()
    assert module.apply_rotary_pos_emb is patched
    a = patched(q,k,c,s)
    for aa,ee in zip(a,e): exact(aa,ee)
    assert torch.equal(rng,torch.get_rng_state())


def test_fma_and_early_cast_negative_controls():
    a,b,c,s = [torch.tensor(x,dtype=torch.float32) for x in
               (-3.098583459854126,-2.299267292022705,0.8253356218338013,0.5646424889564514)]
    eager = a*c + (-b)*s
    contracted = (a.double()*c.double()+(-b*s).double()).float()
    assert eager.view(torch.int32) != contracted.view(torch.int32)
    c = s = torch.full((1,1,2),2**-0.5)
    g = torch.zeros((1,6,1,2),dtype=torch.float16)
    g[:,:2,:,0]=40000;g[:,:2,:,1]=-40000
    gradients=[]
    for early in (False,True):
        k = torch.zeros((1,1,1,2),requires_grad=True)
        y = reference(k,k,c,s)[1]
        y = y.half().repeat_interleave(6,1) if early else y.repeat_interleave(6,1).half()
        gradients.append(torch.autograd.grad(y,k,g)[0])
    assert torch.isfinite(gradients[0]).all()
    assert torch.isnan(gradients[1]).any()


@KERNEL
def test_real_qwen2_attention_interpreter_route(monkeypatch):
    from transformers.models.qwen2 import modeling_qwen2 as mod
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    torch.manual_seed(47132)
    cfg=Qwen2Config(hidden_size=1536,num_attention_heads=12,num_key_value_heads=2,
                   intermediate_size=16,num_hidden_layers=1,attention_dropout=0.,
                   rope_parameters={"rope_type":"default","rope_theta":1_000_000.})
    cfg._attn_implementation="sdpa"
    module=mod.Qwen2Attention(cfg,0).train()
    emb=mod.Qwen2RotaryEmbedding(cfg)
    hidden=torch.randn((1,3,1536),requires_grad=True)
    coeff=emb(hidden,torch.arange(3).unsqueeze(0))
    variables=(hidden,*module.parameters())
    grad=torch.ones_like(hidden)*64
    e=module(hidden,coeff,None)[0];eg=torch.autograd.grad(e,variables,grad)
    original=mod.apply_rotary_pos_emb
    monkeypatch.setattr(mod,"apply_rotary_pos_emb",original)
    # Test-only replace CUDA requirement; retain production geometry/alias guards.
    reason=rotary._support_reason
    def cpu_reason(*args,**kwargs):
        kwargs["cuda_only"]=False
        return reason(*args,**kwargs)
    monkeypatch.setattr(rotary,"_support_reason",cpu_reason)
    assert rotary.patch_qwen2_rotary()
    before=rotary._COUNTS["forward_launch_returned"]
    a=module(hidden,coeff,None)[0];ag=torch.autograd.grad(a,variables,grad)
    assert rotary._COUNTS["forward_launch_returned"]==before+2
    exact(a,e)
    for aa,ee in zip(ag,eg):torch.testing.assert_close(aa,ee,rtol=2e-5,atol=2e-5,equal_nan=True)
