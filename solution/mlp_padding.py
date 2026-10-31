"""Original CUDA masked scatter for MLP-only padding fallback.

In the primary full-decoder-packed path this wrapper deliberately stays idle;
it engages only when the outer Qwen model still uses the original B,T mask.
It is a fused original Triton op, distinct from the uncounted starter SwiGLU.
"""
import contextvars
import inspect
import torch
import triton
import triton.language as tl

_PLAN=contextvars.ContextVar('soup_padding_plan',default=None)

@triton.jit
def soup_scatter_valid_mlp_rows(COMPACT, ROWS, PREFIX, FULL,
                                WIDTH:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    keep=tl.load(ROWS+row)
    compact_row=tl.maximum(tl.load(PREFIX+row),0)
    value=tl.load(COMPACT+compact_row*WIDTH+col,
                  keep & (col<WIDTH),other=0.)
    tl.store(FULL+row*WIDTH+col,value,col<WIDTH)

class _ScatterRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx,compact,active,rows,prefix,total):
        width=compact.shape[-1]
        out=torch.empty((total,width),device=compact.device,dtype=compact.dtype)
        soup_scatter_valid_mlp_rows[(total,)](compact,rows,prefix,out,
             WIDTH=width,BLOCK=triton.next_power_of_2(width),num_warps=4)
        ctx.save_for_backward(active)
        return out

    @staticmethod
    def backward(ctx,grad):
        (active,)=ctx.saved_tensors
        return grad.index_select(0,active),None,None,None,None


def patch_mlp_padding():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model,Qwen2MLP
    prev_model=Qwen2Model.forward
    prev_mlp=Qwen2MLP.forward
    if getattr(prev_model,'_soup_own_mlp_padding',False):
        return True
    signature=inspect.signature(prev_model)

    def forward(self,*args,**kwargs):
        plan=None
        if self.training and torch.is_grad_enabled():
            try:
                mask=signature.bind_partial(self,*args,**kwargs).arguments.get('attention_mask')
            except TypeError:
                mask=kwargs.get('attention_mask')
            if (isinstance(mask,torch.Tensor) and mask.is_cuda and mask.ndim==2
                    and mask.numel()>0):
                rows=(mask!=0).flatten().contiguous()
                active=torch.nonzero(rows,as_tuple=False).flatten()
                if 0<active.numel()<rows.numel():
                    prefix=torch.cumsum(rows.to(torch.int32),dim=0)-1
                    plan=(active,tuple(mask.shape),rows,prefix)
        tok=_PLAN.set(plan)
        try:
            return prev_model(self,*args,**kwargs)
        finally:
            _PLAN.reset(tok)

    def mlp_forward(self,x):
        plan=_PLAN.get()
        if (plan is None or not isinstance(x,torch.Tensor) or not x.is_cuda
                or x.ndim!=3 or not x.is_contiguous()):
            return prev_mlp(self,x)
        active,shape,rows,prefix=plan
        if tuple(x.shape[:2])!=shape or x.device!=active.device:
            return prev_mlp(self,x)
        flat=x.reshape(-1,x.shape[-1])
        compact=prev_mlp(self,flat.index_select(0,active))
        if compact.ndim!=2 or compact.shape[0]!=active.numel() or not compact.is_contiguous():
            full=torch.zeros((flat.shape[0],compact.shape[-1]),device=x.device,dtype=compact.dtype)
            full.index_copy_(0,active,compact.reshape(active.numel(),-1))
        else:
            full=_ScatterRows.apply(compact,active,rows,prefix,flat.shape[0])
        return full.reshape(*x.shape[:2],-1)

    forward._soup_own_mlp_padding=True
    Qwen2Model.forward=forward
    Qwen2MLP.forward=mlp_forward
    return True
