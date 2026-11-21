"""Original token-axis head recomputation with shared original CE tile primitives.

Small forward normalizers/routes are retained, never full logits or a gradient.
The actual global upstream arrives before CE backward and its FP16 narrowing.
No model patch, dtype policy change, task switch or duplicate metric consumer.
"""
import torch
import torch.nn.functional as F
from .contracts import _check,shift_targets,Summaries


class _TokenChunkCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,w,labels,chunk,denominator,tile_api):
        _check(h,w,labels,chunk,denominator)
        b,t,k=h.shape
        h16=h.to(torch.float16).reshape(-1,k);w16=w.to(torch.float16)
        targets=shift_targets(labels).reshape(-1)
        entropy=torch.empty(b*t,dtype=torch.float32,device=h.device)
        nll=torch.empty_like(entropy);predictions=torch.empty(b*t,dtype=torch.int64,device=h.device)
        divisor=(targets!=-100).sum() if denominator is None else denominator
        tensor_divisor=isinstance(divisor,torch.Tensor)
        saved=[h16,w16,targets]+([divisor] if tensor_divisor else [])
        specifications=[]
        for start in range(0,b*t,chunk):
            stop=min(start+chunk,b*t)
            raw=F.linear(h16[start:stop],w16)
            tile=tile_api.forward_tile(raw,targets[start:stop])
            nll[start:stop]=tile.token_nll;entropy[start:stop]=tile.token_entropy
            predictions[start:stop]=tile.first_argmax
            offset=len(saved);saved.extend(tile.state.tensors)
            specifications.append((start,stop,tile.state.metadata,offset,len(tile.state.tensors)))
            del raw,tile
        # Keep the native full-row NLL SUM reduction, not sums of tile means or
        # tile totals. This is NLL of selected full-vocabulary log-probabilities,
        # never cross entropy over one class. Exactly one global division.
        internal_target=torch.where(targets!=-100,torch.zeros_like(targets),targets)
        total=F.nll_loss(-nll[:,None],internal_target,reduction='sum',ignore_index=-100)
        loss=total/divisor
        ctx.save_for_backward(*saved)
        ctx.tensor_divisor=tensor_divisor;ctx.divisor=None if tensor_divisor else divisor
        ctx.shape=h.shape;ctx.tile_api=tile_api;ctx.specifications=tuple(specifications)
        entropy=entropy.reshape(b,t);nll=nll.reshape(b,t);predictions=predictions.reshape(b,t)
        ctx.mark_non_differentiable(entropy,nll,predictions)
        return loss,entropy,nll,predictions

    @staticmethod
    def backward(ctx,grad_loss,_entropy,_nll,_predictions):
        if torch.is_grad_enabled():raise RuntimeError('Higher-order requests not supported by standalone component')
        saved=ctx.saved_tensors;h16,w16,targets=saved[:3]
        divisor=saved[3] if ctx.tensor_divisor else ctx.divisor
        normalized=grad_loss/divisor
        upstream=ctx.tile_api.prepare_upstream(normalized)
        result=torch.empty_like(h16,dtype=torch.float32)
        for start,stop,plan,offset,count in ctx.specifications:
            # Same owned half snapshots, shape/strides and linear primitive as
            # forward. Target preflight checks repeat-GEMM bit identity before
            # accepting saved-normalizer reuse. No RNG or dropout in this head.
            raw=F.linear(h16[start:stop],w16)
            state=ctx.tile_api.TileState(plan,tuple(saved[offset:offset+count]))
            dz16=ctx.tile_api.backward_tile(raw,targets[start:stop],state,upstream)
            if dz16.dtype!=torch.float16 or dz16.shape!=raw.shape:
                raise RuntimeError('CE tile violated native half-adjoint boundary')
            result[start:stop]=(dz16@w16).float()
            del raw,dz16
        return result.reshape(ctx.shape),None,None,None,None,None


def token_chunk_boundary(h,w,labels,chunk=128,denominator=None,summary_sink=None,*,tile_api=None):
    if tile_api is None:
        from . import ce_tile as tile_api
    loss,entropy,nll,predictions=_TokenChunkCE.apply(h,w,labels,chunk,denominator,tile_api)
    if summary_sink is not None:
        if not summary_sink(token_nll=nll,token_entropy=entropy,first_argmax=predictions,route='streaming'):
            raise RuntimeError('Streaming summary publication failed')
        if not summary_sink.complete(loss):
            raise RuntimeError('Streaming normalized-loss identity publication failed')
    return loss,Summaries(entropy,nll,predictions)
