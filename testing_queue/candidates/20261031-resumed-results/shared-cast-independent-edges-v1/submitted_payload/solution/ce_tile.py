"""Original CE/summary adapter with batched full-M scheduling and bounded replay.

No causal shifting, scalar reduction, division, model installation or hidden
autograd graph. Original kernels are imported byte-identically. The owner must
prove identical recomputed FP16 logits and save every state tensor through
ctx.save_for_backward. No logits or gradients are retained in state.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .original import materialized_ce as ce
from .original import mixed_rows as mixed
from .original.metric_summary import detached_metrics
from . import ce_native_replay as native_replay

VOCABULARY = 151936


def stamp(value):
    return (tuple(value.shape),tuple(value.stride()),value.dtype,value.device,
            value.storage_offset(),value._version)


@dataclass(frozen=True)
class TilePlan:
    route: str
    reason: str | None
    rows: int
    vocabulary: int
    target_stamp: tuple
    saved_stamps: tuple
    partition_counts: tuple


@dataclass(frozen=True)
class TileState:
    metadata: TilePlan
    tensors: tuple


@dataclass(frozen=True)
class TileForward:
    token_nll: torch.Tensor
    token_entropy: torch.Tensor
    first_argmax: torch.Tensor
    state: TileState


@dataclass(frozen=True)
class Upstream:
    tensor: torch.Tensor
    custom_eligible: bool
    tensor_stamp: tuple


def _check(logits,target):
    if (type(logits) is not torch.Tensor or type(target) is not torch.Tensor
            or not logits.is_cuda or logits.dtype!=torch.float16
            or logits.ndim!=2 or logits.shape[1]!=VOCABULARY
            or not 1<=logits.shape[0]<=2048
            or target.shape!=(logits.shape[0],) or target.dtype!=torch.int64
            or target.device!=logits.device
            or any(not x.is_contiguous() or x.is_conj() or x.is_neg() for x in (logits,target))):
        raise ValueError('Tile requires contiguous FP16[M,V151936], shifted int64[M], one owned CUDA device')


def _state(route,reason,logits,target,tensors=(),counts=()):
    plan=TilePlan(route,reason,logits.shape[0],logits.shape[1],stamp(target),
                  tuple(stamp(x) for x in tensors),tuple(counts))
    return TileState(plan,tuple(tensors))


def _native_forward(logits,target,reason):
    # Streaming cannot omit metrics on a native fallback: preserve native NaN,
    # ignored-target and first-index argmax behavior on every row.
    nll,entropy,predictions=native_replay.summaries(logits,target)
    return TileForward(nll,entropy,predictions,_state('native',reason,logits,target))


def forward_tile(logits,target):
    """Already shifted flat targets; returns detached per-row values and state.

Two small original classifier host reads remain real costs. No denominator
read, hardware query, duplicate forward softmax, or scalar tile loss occurs.
The enclosing owner checks pinned runtime/device once before all tiles.
"""
    _check(logits,target)
    n,v=logits.shape;nt=ce.triton.cdiv(v,ce.TILE)
    with torch.no_grad():
        partial=torch.empty((n,nt),device=logits.device,dtype=torch.float32)
        bad=torch.empty((n,nt),device=logits.device,dtype=torch.int32)
        maximum=torch.empty(n,device=logits.device,dtype=torch.float32)
        ce.soup_ce_exp_tile_max[(n,nt)](logits,target,partial,bad,
            V=v,NT=nt,BLOCK=ce.TILE,**ce._launch_config())
        invalid,any_supervised=torch.stack((torch.any(bad),torch.any(target!=-100))).tolist()
        if invalid:return _native_forward(logits,target,'nonfinite_logits_or_invalid_target')
        if not any_supervised:return _native_forward(logits,target,'no_supervised_rows')
        ce.soup_ce_exp_row_max[(n,)](partial,maximum,NT=nt,
            MERGE=ce.triton.next_power_of_2(nt),**ce._launch_config())
        logsum=torch.empty_like(maximum);selected=torch.empty_like(maximum)
        guard=torch.empty(n,device=logits.device,dtype=torch.int32)
        ce.soup_ce_exp_tile_sum[(n,nt)](logits,maximum,partial,
            V=v,NT=nt,BLOCK=ce.TILE,**ce._launch_config())
        ce.soup_ce_exp_selected[(n,)](logits,target,maximum,partial,logsum,selected,guard,
            V=v,NT=nt,MERGE=ce.triton.next_power_of_2(nt),NEAR=ce.NEAR_CERTAIN_NLL,
            **ce._launch_config())
        partition,reason=mixed.partition_rows(selected,target,guard)
        if reason is not None:return _native_forward(logits,target,reason)
        if partition.native_rows is None:
            values=selected;tensors=(maximum,logsum);route='all_safe'
        else:
            values=native_replay.selected_values(logits,target,selected,partition.native_rows)
            tensors=(maximum,logsum,guard,partition.native_rows);route='mixed_rows'
        nll=torch.where(target!=-100,-values,0.0)
        entropy,predictions=detached_metrics(logits,maximum,logsum)
        counts=(partition.native_supervised_rows,partition.safe_supervised_rows,partition.ignored_rows)
        return TileForward(nll,entropy,predictions,_state(route,None,logits,target,tensors,counts))


def prepare_upstream(normalized_upstream):
    """Check true post-global-normalization upstream ONCE per whole backward.

No prescaling before CE, no half gradient cache, no extra division. Nonfinite
upstream delegates to native CE autograd; higher-order is unsupported by the
enclosing recomputing-head component and rejected here too.
"""
    if torch.is_grad_enabled():raise RuntimeError('Higher-order tile backward unsupported')
    q=normalized_upstream
    if type(q) is not torch.Tensor or q.ndim!=0:
        raise ValueError('Actual scalar normalized upstream required')
    finite=bool(torch.isfinite(q).all().item())
    eligible=finite and q.dtype==torch.float32 and not q.is_neg() and not q.is_conj()
    q=q.contiguous()
    return Upstream(q,eligible,stamp(q))


def _native_backward(logits,target,q):
    return native_replay.gradient(logits,target,q)


def backward_tile(logits,target,state,upstream):
    """Use saved normalizers only with an owner-proven identical raw16 tile.

Caller reconstructs TileState(plan, saved_tensors_slice) after retrieving its
autograd ctx.saved_tensors; that provides native saved-version protection.
Metadata stamps additionally reject direct mutation of targets/state/upstream.
"""
    _check(logits,target)
    if torch.is_grad_enabled():raise RuntimeError('Higher-order tile backward unsupported')
    if type(state) is not TileState or type(state.metadata) is not TilePlan:
        raise ValueError('Owned tile state required')
    p=state.metadata
    if (p.rows!=logits.shape[0] or p.vocabulary!=logits.shape[1]
            or stamp(target)!=p.target_stamp
            or tuple(stamp(x) for x in state.tensors)!=p.saved_stamps):
        raise RuntimeError('Tile target/state metadata or version changed')
    if (type(upstream) is not Upstream or stamp(upstream.tensor)!=upstream.tensor_stamp
            or upstream.tensor.device!=logits.device):
        raise RuntimeError('Actual normalized upstream changed or has wrong device')
    q=upstream.tensor
    if p.route=='native' or not upstream.custom_eligible:
        return _native_backward(logits,target,q)
    if p.route not in ('all_safe','mixed_rows') or len(state.tensors)!=(2 if p.route=='all_safe' else 4):
        raise RuntimeError('Invalid tile route/state shape')
    maximum,logsum=state.tensors[:2];n,v=logits.shape
    gradient=torch.empty_like(logits)
    if p.route=='all_safe':
        ce.soup_ce_exp_backward[(n,ce.triton.cdiv(v,ce.TILE))](
            logits,target,maximum,logsum,q,gradient,V=v,BLOCK=ce.TILE,**ce._launch_config())
    else:
        guard,native_rows=state.tensors[2:]
        mixed.soup_ce_mixed_backward[(n,ce.triton.cdiv(v,ce.TILE))](
            logits,target,maximum,logsum,q,guard,gradient,V=v,BLOCK=ce.TILE,**ce._launch_config())
        # Native near-row CE sees actual q, then .float() backward narrows once.
        native_replay.scatter_gradient(logits,target,q,native_rows,gradient)
    return gradient
