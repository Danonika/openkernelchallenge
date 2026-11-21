"""Original native-GEMM token-chunk component; no model patch or Triton credit.

Token-axis chunks keep each hidden-gradient row's entire vocabulary reduction.
Backward recomputes the head AFTER the real upstream/GradScaler scalar arrives.
No Soup vendor chunk implementation is imported or copied. Native PyTorch ops
are reference primitives and earn no original-Triton credit. Target sliced/full
GEMM equivalence is tested, not assumed. No model/optimizer/training is installed.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class Summaries:
    entropy: torch.Tensor
    token_nll: torch.Tensor
    predictions: torch.Tensor


def _check(h,w,labels,chunk,denominator):
    assert type(h) is torch.Tensor and type(w) in (torch.Tensor,torch.nn.Parameter) and type(labels) is torch.Tensor
    assert h.device==w.device==labels.device and h.device.type in ('cpu','cuda')
    if h.device.type=='cuda':
        assert h.shape[-1]==1536 and w.shape==(151936,1536), 'Task-specific CUDA geometry'
        assert torch.cuda.get_device_capability(h.device)==(7,5), 'T4 component only'
        assert torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.float16, 'Native AMP FP16 boundary only'
    assert h.dtype==w.dtype==torch.float32 and labels.dtype==torch.int64
    assert h.ndim==3 and w.ndim==2 and labels.shape==h.shape[:2] and h.shape[-1]==w.shape[1]
    assert h.is_contiguous() and w.is_contiguous() and labels.is_contiguous()
    assert not any(x.is_neg() or x.is_conj() for x in (h,w,labels))
    assert not w.requires_grad,'Frozen head only: no vocabulary/weight-gradient accumulation proof'
    assert h.untyped_storage().data_ptr()!=w.untyped_storage().data_ptr()
    assert type(chunk) is int and chunk>0 and h.numel()>0 and w.shape[0]>0
    assert denominator is None or (type(denominator) is int and denominator>=0) or (type(denominator) is torch.Tensor and denominator.device==h.device and denominator.dtype==torch.int64 and denominator.ndim==0 and not denominator.requires_grad)
    # Native primitives retain their exceptional-value behavior; no data-dependent
    # success route, padding removal, global AMP/precision-setting change, or cache.


def shift_targets(labels):
    return F.pad(labels,(0,1),value=-100)[...,1:].contiguous()


def native_boundary(h,w,labels,denominator=None):
    # Explicit CPU emulation of CUDA AMP's FP32 -> FP16 linear boundary.
    raw=F.linear(h.to(torch.float16),w.to(torch.float16))
    targets=shift_targets(labels)
    loss=F.cross_entropy(raw.float().view(-1,w.shape[0]),targets.view(-1),ignore_index=-100,
                         reduction='mean' if denominator is None else 'sum')
    if denominator is not None:loss=loss/denominator
    public=raw.float()
    logp=F.log_softmax(public,dim=-1)
    nll=F.cross_entropy(public.reshape(-1,w.shape[0]),targets.reshape(-1),ignore_index=-100,reduction='none').reshape_as(labels)
    return loss,Summaries(-(logp.exp()*logp).sum(-1),nll,public.argmax(-1)),raw


class _TokenChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,w,labels,chunk,denominator):
        _check(h,w,labels,chunk,denominator)
        b,t,k=h.shape;v=w.shape[0]
        # FP32 inputs imply these casts create owned snapshots, not aliases.
        h16=h.to(torch.float16).reshape(-1,k)
        w16=w.to(torch.float16)
        targets=shift_targets(labels).reshape(-1)
        entropy=torch.empty(b*t,dtype=torch.float32,device=h.device)
        nll=torch.empty(b*t,dtype=torch.float32,device=h.device)
        predictions=torch.empty(b*t,dtype=torch.int64,device=h.device)
        for start in range(0,b*t,chunk):
            stop=min(start+chunk,b*t)
            logits=F.linear(h16[start:stop],w16).float()
            logp=F.log_softmax(logits,dim=-1)
            entropy[start:stop]=-(logp.exp()*logp).sum(-1)
            nll[start:stop]=F.cross_entropy(logits,targets[start:stop],ignore_index=-100,reduction='none')
            predictions[start:stop]=logits.argmax(-1)
            del logits,logp
        divisor=(targets!=-100).sum() if denominator is None else denominator
        # This small global reduction can differ by rounding from native NLL-sum.
        # It is NOT claimed bit-exact; task tolerances and target gates still apply.
        loss=nll.sum()/divisor
        # A caller-owned denominator tensor has native saved-version semantics.
        # Saving it (rather than only assigning ctx.divisor) catches mutation.
        tensor_divisor=isinstance(divisor,torch.Tensor)
        ctx.save_for_backward(h16,w16,targets,*((divisor,) if tensor_divisor else ()))
        ctx.shape=h.shape;ctx.chunk=chunk;ctx.divisor=None if tensor_divisor else divisor
        entropy=entropy.reshape(b,t);nll=nll.reshape(b,t);predictions=predictions.reshape(b,t)
        ctx.mark_non_differentiable(entropy,nll,predictions)
        return loss,entropy,nll,predictions

    @staticmethod
    def backward(ctx,grad_loss,_entropy,_nll,_predictions):
        if torch.is_grad_enabled():raise RuntimeError('Higher-order requests not supported by standalone component')
        saved=ctx.saved_tensors
        h16,w16,targets=saved[:3]
        divisor=saved[3] if len(saved)==4 else ctx.divisor
        # Main loss division's upstream gradient happens BEFORE native CE -> half.
        normalized=grad_loss/divisor
        result=torch.empty_like(h16,dtype=torch.float32)
        for start in range(0,h16.shape[0],ctx.chunk):
            stop=min(start+ctx.chunk,h16.shape[0])
            with torch.enable_grad():
                raw=F.linear(h16[start:stop],w16).detach().requires_grad_()
                loss_sum=F.cross_entropy(raw.float(),targets[start:stop],ignore_index=-100,reduction='sum')
                (dz16,)=torch.autograd.grad(loss_sum,raw,grad_outputs=normalized)
            # One full-V dot per hidden row; output rounded to FP16 BEFORE fp32 cast.
            result[start:stop]=(dz16 @ w16).float()
            del raw,loss_sum,dz16
        return result.reshape(ctx.shape),None,None,None,None


def token_chunk_boundary(h,w,labels,chunk=128,denominator=None,summary_sink=None):
    loss,entropy,nll,predictions=_TokenChunk.apply(h,w,labels,chunk,denominator)
    if summary_sink is not None:
        if not summary_sink(token_nll=nll,token_entropy=entropy,first_argmax=predictions,route='streaming'):
            raise RuntimeError('Streaming summary publication failed')
        if not summary_sink.complete(loss):
            raise RuntimeError('Streaming normalized-loss identity publication failed')
    return loss,Summaries(entropy,nll,predictions)

