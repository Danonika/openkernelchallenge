"""Original true token-axis streamed head using the two-pass summary reducer.

No full [tokens,vocabulary] logits or adjoint is retained or allocated.
Recomputation uses the same native FP16 tile projection shape as forward.
Independent row tiles preserve row order; tile GEMM numerical equivalence
requires measured quality validation, not a bitwise-equivalence claim.
"""
import torch
import torch.nn.functional as F
from .contracts import _check,shift_targets,Summaries


class _StreamedMomentCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,w,labels,chunk,denominator,tile_api):
        _check(h,w,labels,chunk,denominator)
        b,t,k=h.shape;rows=b*t;v=w.shape[0]
        h16=h.to(torch.float16).reshape(rows,k);w16=w.to(torch.float16)
        targets=shift_targets(labels).reshape(-1)
        entropy=torch.empty(rows,dtype=torch.float32,device=h.device)
        nll=torch.empty_like(entropy);predictions=torch.empty(rows,dtype=torch.int64,device=h.device)
        divisor=(targets!=-100).sum() if denominator is None else denominator
        tensor_divisor=isinstance(divisor,torch.Tensor)
        saved=[h16,w16,targets]+([divisor] if tensor_divisor else [])
        specs=[]
        for start in range(0,rows,chunk):
            stop=min(start+chunk,rows)
            raw=F.linear(h16[start:stop],w16)
            tile=tile_api.forward_tile(raw,targets[start:stop])
            nll[start:stop]=tile.token_nll;entropy[start:stop]=tile.token_entropy
            predictions[start:stop]=tile.first_argmax
            offset=len(saved);saved.extend(tile.state.tensors)
            specs.append((start,stop,tile.state.metadata,offset,len(tile.state.tensors)))
            del raw,tile
        internal=torch.where(targets!=-100,torch.zeros_like(targets),targets)
        total=F.nll_loss(-nll[:,None],internal,reduction='sum',ignore_index=-100)
        loss=total/divisor
        ctx.save_for_backward(*saved);ctx.tensor_divisor=tensor_divisor
        ctx.divisor=None if tensor_divisor else divisor;ctx.specs=tuple(specs)
        ctx.shape=h.shape;ctx.tile_api=tile_api
        entropy=entropy.reshape(b,t);nll=nll.reshape(b,t);predictions=predictions.reshape(b,t)
        ctx.mark_non_differentiable(entropy,nll,predictions)
        return loss,entropy,nll,predictions

    @staticmethod
    def backward(ctx,grad_loss,_entropy,_nll,_predictions):
        if torch.is_grad_enabled():raise RuntimeError('Higher-order head requests unsupported')
        saved=ctx.saved_tensors;h16,w16,targets=saved[:3]
        divisor=saved[3] if ctx.tensor_divisor else ctx.divisor
        upstream=ctx.tile_api.prepare_upstream(grad_loss/divisor)
        result=torch.empty(h16.shape,device=h16.device,dtype=torch.float32)
        for start,stop,plan,offset,count in ctx.specs:
            raw=F.linear(h16[start:stop],w16)
            state=ctx.tile_api.TileState(plan,tuple(saved[offset:offset+count]))
            dz=torch.empty_like(raw)
            ctx.tile_api.backward_tile(raw,targets[start:stop],state,upstream,out=dz)
            # Native AMP dH is rounded to half once, then widened once, with
            # no FP16 partial-sum accumulation or prescaled gradient cache.
            dh=dz@w16
            result[start:stop].copy_(dh)
            del raw,dz,dh
        return result.reshape(ctx.shape),None,None,None,None,None


def token_chunk_boundary(h,w,labels,chunk=256,denominator=None,summary_sink=None,*,tile_api=None):
    if tile_api is None:from . import moment_tile as tile_api
    loss,entropy,nll,predictions=_StreamedMomentCE.apply(h,w,labels,chunk,denominator,tile_api)
    if summary_sink is not None:
        if not summary_sink(token_nll=nll,token_entropy=entropy,first_argmax=predictions,route='streaming'):
            raise RuntimeError('Streamed moment summary publication failed')
        if not summary_sink.complete(loss):raise RuntimeError('Streamed moment loss publication failed')
    return loss,Summaries(entropy,nll,predictions)
