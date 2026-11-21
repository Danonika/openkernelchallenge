"""Full-M native AMP projection/adjoint geometry with chunked original CE.

The measured token-chunk adjoint GEMM drift is removed structurally: retain one
full FP16 raw-logit buffer, assemble a full FP16 dZ only after real upstream,
and perform exactly one full-M native dH GEMM. No public full FP32 logits or
softmax buffer is retained. This deliberately trades memory for native GEMM
geometry; it is not the old streaming memory claim and is not target-proven.
"""
import torch
import torch.nn.functional as F
from .contracts import _check,shift_targets,Summaries


class _FullGeometryCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,w,labels,chunk,denominator,tile_api):
        _check(h,w,labels,chunk,denominator)
        b,t,k=h.shape
        # Preserve the actual native 3D linear input before flattening its
        # output, as in the bitwise-confirmed same-H full-native replay.
        h16=h.to(torch.float16);w16=w.to(torch.float16)
        raw=F.linear(h16,w16).reshape(b*t,w.shape[0])
        targets=shift_targets(labels).reshape(-1)
        entropy=torch.empty(b*t,dtype=torch.float32,device=h.device)
        nll=torch.empty_like(entropy);predictions=torch.empty(b*t,dtype=torch.int64,device=h.device)
        divisor=(targets!=-100).sum() if denominator is None else denominator
        tensor_divisor=isinstance(divisor,torch.Tensor)
        saved=[raw,w16,targets]+([divisor] if tensor_divisor else [])
        specifications=[]
        for start in range(0,b*t,chunk):
            stop=min(start+chunk,b*t)
            tile=tile_api.forward_tile(raw[start:stop],targets[start:stop])
            nll[start:stop]=tile.token_nll;entropy[start:stop]=tile.token_entropy
            predictions[start:stop]=tile.first_argmax
            offset=len(saved);saved.extend(tile.state.tensors)
            specifications.append((start,stop,tile.state.metadata,offset,len(tile.state.tensors)))
            del tile
        # The original global reduction and normalized-loss identity remain.
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
        saved=ctx.saved_tensors;raw,w16,targets=saved[:3]
        divisor=saved[3] if ctx.tensor_divisor else ctx.divisor
        normalized=grad_loss/divisor
        upstream=ctx.tile_api.prepare_upstream(normalized)
        # Never precompute a unit-scale half gradient. Every original CE tile
        # sees the actual normalized upstream before its FP16 narrowing.
        full_dz=torch.empty_like(raw)
        for start,stop,plan,offset,count in ctx.specifications:
            state=ctx.tile_api.TileState(plan,tuple(saved[offset:offset+count]))
            dz16=ctx.tile_api.backward_tile(raw[start:stop],targets[start:stop],state,upstream)
            if dz16.dtype!=torch.float16 or dz16.shape!=raw[start:stop].shape:
                raise RuntimeError('CE tile violated native half-adjoint boundary')
            full_dz[start:stop]=dz16
            del dz16
        # Single native all-row/all-vocabulary adjoint, not partial GEMMs or
        # FP16 partial-adjoint accumulation. Cast its native half result once.
        result=(full_dz@w16).float()
        return result.reshape(ctx.shape),None,None,None,None,None


def token_chunk_boundary(h,w,labels,chunk=128,denominator=None,summary_sink=None,*,tile_api=None):
    if tile_api is None:
        from . import ce_tile as tile_api
    loss,entropy,nll,predictions=_FullGeometryCE.apply(h,w,labels,chunk,denominator,tile_api)
    if summary_sink is not None:
        if not summary_sink(token_nll=nll,token_entropy=entropy,first_argmax=predictions,route='streaming'):
            raise RuntimeError('Streaming summary publication failed')
        if not summary_sink.complete(loss):
            raise RuntimeError('Streaming normalized-loss identity publication failed')
    return loss,Summaries(entropy,nll,predictions)
