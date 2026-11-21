"""Original shared numerical fixtures; no target import until called."""
import torch
H=1536

def native(a,b,w,eps=1e-6):
    z=a+b
    variance=z.pow(2).mean(-1,keepdim=True)
    return w*(z*torch.rsqrt(variance+eps)),z

def staged_dx(dy,z,w,inverse):
    g=dy*w;direct=g*inverse;dinverse=(g*z).sum(-1,keepdim=True)
    dvariance=(-0.5*dinverse)*((inverse*inverse)*inverse)
    return direct+(dvariance/z.shape[-1])*(2*z)

class CpuBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,w):
        z=a+b;inverse=torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)
        ctx.save_for_backward(z,w,inverse);ctx.set_materialize_grads(False)
        return w*(z*inverse),z
    @staticmethod
    def backward(ctx,dy,dz):
        if dy is None:return dz,dz,None
        z,w,inverse=ctx.saved_tensors;dx=staged_dx(dy,z,w,inverse)
        if dz is not None:dx=dx+dz
        return dx,dx,None

def fixture(rows,scale=1.,device='cpu',alias='independent',only_a=False):
    # Pure deterministic values; no global CPU/CUDA RNG change.
    values=torch.arange((rows+1)*H,device=device,dtype=torch.float32).reshape(rows+1,H)
    values=(values.remainder(997)/499.-1.)*scale
    if alias=='independent':a=values[:rows].clone();b=values[1:].clone()*0.37
    elif alias=='same':a=values[:rows].clone();b=a
    elif alias=='overlap':a=values[:rows];b=values[1:]
    else:raise ValueError(alias)
    a.requires_grad_(True)
    if not only_a:b.requires_grad_(True)
    w=1.+torch.arange(H,device=device,dtype=torch.float32).remainder(19)*0.007
    return a,b,w

def engineering_close(actual,expected,*,forward=False,scale=1.):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype==torch.float32
    for predicate in (torch.isnan,torch.isposinf,torch.isneginf):
        assert torch.equal(predicate(actual),predicate(expected)),'nonfinite class differs'
    finite=torch.isfinite(actual)&torch.isfinite(expected)
    a=actual[finite]/scale;b=expected[finite]/scale
    error=float((a-b).abs().max()) if a.numel() else 0.
    if forward:torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
    else:torch.testing.assert_close(a,b,rtol=1e-3,atol=2e-5)
    return {'max_abs_error_scaled':error,'scale_for_reporting':scale,'nonfinite_class_exact':True}

def compare(fn,rows,activation_scale=1.,upstream_scale=1.,device='cpu',alias='independent',use='both',only_a=False):
    a,b,w=fixture(rows,activation_scale,device,alias,only_a)
    versions=[x._version for x in (a,b,w)];snaps=[x.detach().clone() for x in (a,b,w)]
    # A real connected nonzero cotangent; no changes to production scaling.
    dy=torch.linspace(-.7,.9,rows*H,device=device).reshape(rows,H)*upstream_scale
    dz=torch.linspace(.2,-.3,rows*H,device=device).reshape(rows,H)*upstream_scale
    actual=fn(a,b,w);expected=native(a,b,w)
    out=[engineering_close(x,y,forward=True) for x,y in zip(actual,expected)]
    assert torch.equal(actual[1],expected[1]),'rounded residual changed'
    ins=(a,b) if b.requires_grad and b is not a else (a,)
    selected=(0,1) if use=='both' else (0,) if use=='norm_only' else (1,)
    up=(dy,dz)
    got=torch.autograd.grad(tuple(actual[i] for i in selected),ins,tuple(up[i] for i in selected))
    want=torch.autograd.grad(tuple(expected[i] for i in selected),ins,tuple(up[i] for i in selected))
    grad=[engineering_close(x,y,scale=upstream_scale) for x,y in zip(got,want)]
    for x,v,s in zip((a,b,w),versions,snaps):assert x._version==v and torch.equal(x,s),'input mutated'
    return {'rows':rows,'activation_scale':activation_scale,'upstream_scale':upstream_scale,'alias':alias,'use':use,'only_a_requires_grad':only_a,'forward':out,'gradients':grad}

def extreme_diagnostics(fn,device='cpu'):
    # Known expanded-graph cancellation: discrepancy is disclosed, not promoted.
    results=[]
    for label,fn1 in [('native_expanded',native),('candidate_atomic',fn)]:
        a=torch.full((1,H),512.,device=device,requires_grad=True);b=torch.full_like(a,512.,requires_grad=True);w=torch.ones(H,device=device)
        y,z=fn1(a,b,w);g=torch.autograd.grad((y,z),(a,), (torch.full_like(y,2.**34),torch.ones_like(z)))[0]
        results.append({'method':label,'case':'cancellation','min':float(g.min()),'max':float(g.max()),'nan':int(g.isnan().sum())})
        for use in ('residual_only','connected_zero_norm'):
            a=torch.full((1,H),1.5e38,device=device,requires_grad=True);b=torch.full_like(a,1.5e38,requires_grad=True)
            y,z=fn1(a,b,w)
            g=torch.autograd.grad((z,) if use=='residual_only' else (y,z),(a,), (torch.ones_like(z),) if use=='residual_only' else (torch.zeros_like(y),torch.ones_like(z)))[0]
            results.append({'method':label,'case':use,'nan':int(g.isnan().sum()),'all_one':bool(torch.all(g==1))})
    return results
