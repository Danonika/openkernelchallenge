"""Inactive cached-forward head with sparse native hidden-gradient GEMM.

Derived from this research candidate's retained native head, causal targets,
bounded native CE, and unsafe-row handling. No external kernel is copied.
The forward performs ONE full-shape native FP16 torch.mm and retains its
half logits. Loss, entropy and first argmax reuse cached_head._summaries for
every row, including prompts and the final position.

First-order backward gathers supervised OR nonfinite-entropy rows in bounded
score chunks, computes their genuinely upstream-scaled half dZ, and executes
ONE native torch.mm(dZ_selected, W_half). It scatters that half dH into zeros
and then restores the original hidden dtype. No forward projection is
replayed. The selected-row backward GEMM geometry is deliberately different
from the full native baseline; physical FP16-gradient and official training
quality gates remain necessary. Compaction has a real size-discovery sync.

Higher-order differentiation branches before compaction and reconstructs a
full native head/CE autograd graph from the saved ORIGINAL hidden tensor.
That exceptional path intentionally replays the projection and retains native
higher derivatives. CUDA graph capture is unsupported. Nothing is installed
or selected on import, and no timing or training speedup is claimed.
"""

import os

import torch
import torch.nn.functional as F

from . import cached_head
from .cached_ce_backward import native_logsoftmax_ce_backward
from .streamed_head import HeadResult, _native_logit_gradient


class _SparseCachedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, num_items_in_batch, chunk_size,
                sparse_backward, numerics, backward_mode):
        batch,length,width = hidden.shape
        rows = batch*length
        targets = F.pad(labels,(0,1),value=-100)[...,1:].contiguous().reshape(-1)
        torch._assert_async(((targets==-100)|((targets>=0)&(targets<weight.shape[0]))).all(),
                            "causal target index is out of bounds")
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            half_hidden = hidden.reshape(rows,width).to(torch.float16)
            half_weight = weight.to(torch.float16)
            scores = torch.mm(half_hidden,half_weight.t())
            token_loss,entropy,predictions = cached_head._summaries(
                scores,targets,chunk_size,numerics)
        if num_items_in_batch is None:
            divisor = (targets!=-100).sum().to(torch.float32)
        else:
            divisor = torch.as_tensor(num_items_in_batch,device=hidden.device,
                                      dtype=torch.float32).detach().reshape(())
        loss = token_loss.sum()/divisor
        # Even a lone -Inf logit makes true entropy nonfinite through 0*(-Inf).
        # A nonfinite weight also makes each projected row unsafe: multiplying
        # it by zero gives NaN, and a nonzero input gives Inf/NaN. Thus ignored
        # rows whose native 0-dZ times nonfinite W could produce NaN are kept.
        unsafe = ~torch.isfinite(entropy)
        ctx.save_for_backward(hidden,half_weight,scores,targets,divisor,unsafe)
        ctx.chunk_size = chunk_size
        ctx.sparse_backward = sparse_backward
        ctx.backward_mode = backward_mode
        ctx.set_materialize_grads(False)
        token_loss,entropy,predictions = (value.reshape(batch,length)
                                         for value in (token_loss,entropy,predictions))
        ctx.mark_non_differentiable(token_loss,entropy,predictions)
        return loss,token_loss,entropy,predictions

    @staticmethod
    def backward(ctx,grad_loss,_grad_token_loss,_grad_entropy,_grad_predictions):
        if grad_loss is None or not ctx.needs_input_grad[0]:
            return (None,)*8
        hidden,weight,scores,targets,divisor,unsafe = ctx.saved_tensors
        rows,width = scores.shape[0],hidden.shape[-1]
        if torch.is_grad_enabled():
            # Keep original hidden identity; a saved detached half activation
            # would erase the second derivative through the head.
            with torch.enable_grad(),torch.autocast(device_type=hidden.device.type,enabled=False):
                recomputed = torch.mm(hidden.reshape(rows,width).to(torch.float16),weight.t())
                loss = F.cross_entropy(recomputed.float(),targets,reduction="sum")/divisor
                gradient, = torch.autograd.grad(loss,hidden,grad_outputs=grad_loss.float(),
                                                create_graph=True)
            return gradient,None,None,None,None,None,None,None

        selected = (torch.nonzero((targets!=-100)|unsafe,as_tuple=False).flatten()
                    if ctx.sparse_backward else None)
        work_rows = selected.numel() if selected is not None else rows
        gradient_half = torch.zeros((rows,width),device=hidden.device,dtype=torch.float16)
        if work_rows:
            # Only selected dZ spans the vocabulary; gathered logits and native
            # FP32 CE workspaces remain bounded by the fixed row chunk.
            selected_dz = torch.empty((work_rows,scores.shape[1]),device=hidden.device,
                                      dtype=torch.float16)
            upstream = grad_loss.float()
            scale = upstream/divisor if ctx.backward_mode=="native" else None
            with torch.autocast(device_type=hidden.device.type,enabled=False):
                for start in range(0,work_rows,ctx.chunk_size):
                    stop = min(start+ctx.chunk_size,work_rows)
                    if selected is None:
                        score_chunk,target_chunk = scores[start:stop],targets[start:stop]
                    else:
                        indices = selected[start:stop]
                        score_chunk = scores.index_select(0,indices)
                        target_chunk = targets.index_select(0,indices)
                    if ctx.backward_mode=="native_fused":
                        dz = native_logsoftmax_ce_backward(score_chunk,target_chunk,divisor,
                                                          upstream,ctx.chunk_size)
                    else:
                        dz = _native_logit_gradient(score_chunk,target_chunk,scale)
                    selected_dz[start:stop].copy_(dz)
                    del score_chunk,target_chunk,dz
                selected_dh = torch.mm(selected_dz,weight)
            if selected is None:
                gradient_half = selected_dh
            else:
                gradient_half.index_copy_(0,selected,selected_dh)
        gradient = gradient_half.reshape(hidden.shape).to(hidden.dtype)
        return gradient,None,None,None,None,None,None,None


def sparse_cached_linear_ce(hidden,weight,labels,num_items_in_batch=None,
                            chunk_size=128,projection="torch",sparse_backward=True,
                            numerics="native",backward_mode="native"):
    """Return the same private HeadResult interface as streamed_linear_ce.

    Hidden is contiguous FP16/FP32 (B,T,D); weight is a contiguous frozen
    FP16/FP32 (V,D) tensor. Supplying the caller's cached FP16 weight avoids a
    new full weight cast. The loss differentiates only hidden. Native and
    guarded forward statistics retain their existing explicit meanings.
    Native or native_fused chooses the first-order CE consumer independently.

    sparse_backward=False is a fixed dense control: identical cached forward
    and one full-row backward GEMM. Neither setting replays the projection in
    first order. Higher-order autograd always uses native recomputation.
    """
    if hidden.ndim!=3 or weight.ndim!=2 or labels.shape!=hidden.shape[:2]:
        raise ValueError("expected hidden (B,T,D), weight (V,D), and labels (B,T)")
    if hidden.shape[-1]!=weight.shape[-1] or min(*hidden.shape,weight.shape[0])<1:
        raise ValueError("nonempty compatible head dimensions are required")
    if hidden.dtype not in (torch.float16,torch.float32) or weight.dtype not in (torch.float16,torch.float32):
        raise TypeError("expected FP16/FP32 hidden and weight")
    if labels.dtype!=torch.long:
        raise TypeError("labels must be int64")
    if hidden.device!=weight.device or labels.device!=hidden.device:
        raise ValueError("hidden, weight and labels must share a device")
    if not hidden.is_contiguous() or not weight.is_contiguous():
        raise ValueError("hidden and weight must be contiguous")
    if weight.requires_grad:
        raise ValueError("sparse cached head requires a frozen weight")
    if type(chunk_size) is not int or chunk_size<1:
        raise ValueError("chunk_size must be a positive integer")
    if type(sparse_backward) is not bool:
        raise TypeError("sparse_backward must be a bool")
    if projection!="torch":
        raise ValueError("sparse cached head requires the full native torch projection")
    if numerics not in ("native","guarded") or backward_mode not in ("native","native_fused"):
        raise ValueError("unsupported numerics or backward_mode")
    if isinstance(num_items_in_batch,torch.Tensor) and num_items_in_batch.numel()!=1:
        raise ValueError("num_items_in_batch must be scalar")
    if numerics=="guarded" and not hidden.is_cuda and os.environ.get("TRITON_INTERPRET")!="1":
        raise ValueError("guarded CPU statistics require explicit TRITON_INTERPRET=1")
    if torch.compiler.is_compiling():
        raise ValueError("sparse cached head does not support graph compilation")
    if hidden.is_cuda:
        with torch.cuda.device(hidden.device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("sparse cached head does not support CUDA graph capture")
    return HeadResult(*_SparseCachedLinearCrossEntropy.apply(
        hidden,weight,labels,num_items_in_batch,chunk_size,sparse_backward,numerics,backward_mode))
