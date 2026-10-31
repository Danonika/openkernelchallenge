"""Original two-output FP32 residual-add/norm component; no model hook.

One natural Qwen decoder boundary: z = residual + attention_output, then
norm(z); both rounded z and norm output remain observable. No in-place update,
precision narrowing, dWeight suppression for trainable weights, or fake work.
Backward retains staged norm arithmetic, then adds a genuine bypass cotangent.
This atomic boundary is NOT bitwise equivalent to all expanded native graphs.
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

@triton.jit
def residual_norm_f32(A,B,W,Z,Y,R,H:tl.constexpr,EPS:tl.constexpr,KEEP:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK);valid=col<H
    a=tl.load(A+row*H+col,valid,other=0).to(tl.float32)
    b=tl.load(B+row*H+col,valid,other=0).to(tl.float32)
    # Native FP32 add is rounded before square/normalization and before bypass.
    z=a+b
    w=tl.load(W+col,valid,other=0).to(tl.float32)
    mean_square=tl.div_rn(tl.sum(z*z,0),H)
    inverse=tl.rsqrt(mean_square+EPS)
    normalized=z*inverse
    y=w*normalized
    tl.store(Z+row*H+col,z,valid);tl.store(Y+row*H+col,y,valid)
    if KEEP:tl.store(R+row,inverse)

@triton.jit
def residual_norm_adjoint_f32(DY,DZ,Z,W,R,DX,H:tl.constexpr,HAS_BYPASS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK);valid=col<H
    z=tl.load(Z+row*H+col,valid,other=0).to(tl.float32)
    w=tl.load(W+col,valid,other=0).to(tl.float32)
    dy=tl.load(DY+row*H+col,valid,other=0).to(tl.float32)
    inverse=tl.load(R+row).to(tl.float32)
    g=dy*w;direct=g*inverse
    dinverse=tl.sum(g*z,0)
    inverse3=(inverse*inverse)*inverse
    dvariance=(-0.5*dinverse)*inverse3
    variance_path=tl.div_rn(dvariance,H)*(2.0*z)
    dx=direct+variance_path
    if HAS_BYPASS:dx=dx+tl.load(DZ+row*H+col,valid,other=0).to(tl.float32)
    tl.store(DX+row*H+col,dx,valid)


def supported(a,b,w,eps):
    return (type(a) is torch.Tensor and type(b) is torch.Tensor
        and type(w) in (torch.Tensor,torch.nn.Parameter)
        and a.device.type=='cuda' and a.device==b.device==w.device
        and a.dtype==b.dtype==w.dtype==torch.float32
        and a.ndim in (2,3) and a.shape==b.shape and a.shape[-1]==1536
        and 1<=a.numel()//1536<=2048 and w.shape==(1536,) and not w.requires_grad
        and all(x.is_contiguous() and not x.is_conj() and not x.is_neg() for x in (a,b,w))
        and type(eps) is float and eps==1e-6)

class ResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,w,eps,need_backward):
        rows=a.numel()//1536;z=torch.empty_like(a);y=torch.empty_like(a)
        r=torch.empty((rows,) if need_backward else (0,),device=a.device,dtype=torch.float32)
        residual_norm_f32[(rows,)](a,b,w,z,y,r,1536,eps,need_backward,2048,num_warps=4,enable_fp_fusion=False)
        ctx.set_materialize_grads(False)
        if need_backward:ctx.save_for_backward(z,w,r)
        return y,z
    @staticmethod
    @once_differentiable
    def backward(ctx,dy,dz):
        # An unused norm output must not execute its derivative from fake zeros.
        if dy is None:
            return dz if ctx.needs_input_grad[0] else None,dz if ctx.needs_input_grad[1] else None,None,None,None
        z,w,r=ctx.saved_tensors;dy=dy.contiguous()
        if dz is not None:dz=dz.contiguous()
        dx=torch.empty_like(z)
        residual_norm_adjoint_f32[(z.numel()//1536,)](dy,dy if dz is None else dz,z,w,r,dx,1536,dz is not None,2048,num_warps=4,enable_fp_fusion=False)
        return dx if ctx.needs_input_grad[0] else None,dx if ctx.needs_input_grad[1] else None,None,None,None

def fused_residual_norm(a,b,w,eps=1e-6):
    if not supported(a,b,w,eps):raise ValueError('Unsupported residual norm component contract')
    need=torch.is_grad_enabled() and (a.requires_grad or b.requires_grad)
    return ResidualNorm.apply(a,b,w,eps,need)
