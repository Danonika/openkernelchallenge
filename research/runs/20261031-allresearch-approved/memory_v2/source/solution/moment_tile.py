"""Original two-pass finite FP16 CE/entropy/argmax tile reduction.

The vocabulary read produces stable local max, exponential sum, positive
entropy moment and first argmax. A row merge forms the global normalizers and
genuine detached summaries. Exceptional/near-certain rows keep native CE.
No model values, loss normalization, upstream scale or RNG are changed.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from . import ce_tile as old
from .original import materialized_ce as ce
from .original import mixed_rows as mixed

TileState=old.TileState
TilePlan=old.TilePlan
TileForward=old.TileForward
prepare_upstream=old.prepare_upstream


@triton.jit
def soup_ce_moment_tiles(Z, PartialMax, PartialSum, PartialMoment, PartialArgmax,
                         Bad, V:tl.constexpr, NT:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0); tile=tl.program_id(1)
    col=tile*BLOCK+tl.arange(0,BLOCK); real=col<V
    x=tl.load(Z+row*V+col,mask=real,other=-float('inf')).to(tl.float32)
    finite=(x==x)&(tl.abs(x)!=float('inf'))
    maximum=tl.max(x,0)
    # All quantities are finite on the admitted route. Padding becomes zero
    # before exp/multiplication to avoid 0*infinity in the entropy moment.
    delta=tl.where(real,x-maximum,0.)
    e=tl.where(real,tl.exp(delta),0.)
    total=tl.sum(e,0)
    moment=tl.sum(e*(-delta),0)
    first=tl.min(tl.where(real&(x==maximum),col,2147483647),0)
    offset=row*NT+tile
    tl.store(PartialMax+offset,maximum);tl.store(PartialSum+offset,total)
    tl.store(PartialMoment+offset,moment);tl.store(PartialArgmax+offset,first)
    tl.store(Bad+offset,tl.sum((real&~finite).to(tl.int32),0))


@triton.jit
def soup_ce_moment_merge(Z, Target, PartialMax, PartialSum, PartialMoment,
                         PartialArgmax, Bad, Maximum, LogSum, Selected, Guard,
                         Invalid, Entropy, Argmax, V:tl.constexpr, NT:tl.constexpr,
                         MERGE:tl.constexpr, NEAR:tl.constexpr):
    row=tl.program_id(0); index=tl.arange(0,MERGE); real=index<NT
    lm=tl.load(PartialMax+row*NT+index,mask=real,other=-float('inf'))
    ls=tl.load(PartialSum+row*NT+index,mask=real,other=0.)
    le=tl.load(PartialMoment+row*NT+index,mask=real,other=0.)
    la=tl.load(PartialArgmax+row*NT+index,mask=real,other=2147483647)
    lb=tl.load(Bad+row*NT+index,mask=real,other=0)
    maximum=tl.max(lm,0)
    shift=tl.where(real,maximum-lm,0.)
    weights=tl.where(real,tl.exp(-shift),0.)
    total=tl.sum(ls*weights,0)
    logsum=tl.log(total)
    # E[-log p] = log(sum(exp(x-max))) + E[max-x]. Every moment
    # term is nonnegative, avoiding subtraction of two large close values.
    entropy=logsum+tl.sum(weights*(le+ls*shift),0)/total
    first=tl.min(tl.where(real&(lm==maximum),la,2147483647),0)
    target=tl.load(Target+row); supervised=target!=-100
    valid_target=(target>=0)&(target<V)
    safe=tl.where(supervised&valid_target,target,0)
    chosen=tl.load(Z+row*V+safe).to(tl.float32)
    logp=(chosen-maximum)-logsum
    invalid=(tl.sum(lb,0)>0)|(supervised&~valid_target)
    near=supervised&((logp>=-NEAR)|(logp!=logp)|(tl.abs(logp)==float('inf')))
    tl.store(Maximum+row,maximum);tl.store(LogSum+row,logsum)
    tl.store(Selected+row,tl.where(supervised,logp,0.))
    tl.store(Guard+row,near.to(tl.int32));tl.store(Invalid+row,invalid.to(tl.int32))
    tl.store(Entropy+row,entropy);tl.store(Argmax+row,first.to(tl.int64))


def forward_tile(logits,target):
    old._check(logits,target)
    n,v=logits.shape;nt=triton.cdiv(v,2048)
    with torch.no_grad():
        arrays=[torch.empty((n,nt),device=logits.device,dtype=torch.float32) for _ in range(3)]
        partial_argmax=torch.empty((n,nt),device=logits.device,dtype=torch.int32)
        bad=torch.empty_like(partial_argmax)
        maximum,logsum,selected,entropy=[torch.empty(n,device=logits.device,dtype=torch.float32) for _ in range(4)]
        guard=torch.empty(n,device=logits.device,dtype=torch.int32);invalid=torch.empty_like(guard)
        predictions=torch.empty(n,device=logits.device,dtype=torch.int64)
        soup_ce_moment_tiles[(n,nt)](logits,*arrays,partial_argmax,bad,
            V=v,NT=nt,BLOCK=2048,num_warps=4,enable_fp_fusion=False)
        soup_ce_moment_merge[(n,)](logits,target,*arrays,partial_argmax,bad,
            maximum,logsum,selected,guard,invalid,entropy,predictions,V=v,NT=nt,
            MERGE=triton.next_power_of_2(nt),NEAR=ce.NEAR_CERTAIN_NLL,
            num_warps=4,enable_fp_fusion=False)
        supervised=target!=-100
        # One three-int synchronization for finite/target/partition decisions.
        invalid_count,native_count,supervised_count=torch.stack((
            invalid.sum(dtype=torch.int64),guard.sum(dtype=torch.int64),
            supervised.sum(dtype=torch.int64))).tolist()
        if invalid_count:return old._native_forward(logits,target,'nonfinite_logits_or_invalid_target')
        if not supervised_count:return old._native_forward(logits,target,'no_supervised_rows')
        if native_count==supervised_count:return old._native_forward(logits,target,'all_supervised_rows_near_certain')
        if native_count:
            native_rows=torch.nonzero(guard,as_tuple=False).flatten()
            selected=mixed.mixed_selected_values(logits,target,selected,native_rows)
            # Preserve cancellation-sensitive near-certain summary rows.
            native_logits=logits.index_select(0,native_rows).float()
            lp=F.log_softmax(native_logits,-1)
            entropy.index_copy_(0,native_rows,-(lp.exp()*lp).sum(-1))
            tensors=(maximum,logsum,guard,native_rows);route='mixed_rows'
        else:tensors=(maximum,logsum);route='all_safe'
        nll=torch.where(supervised,-selected,0.)
        counts=(native_count,supervised_count-native_count,n-supervised_count)
        return TileForward(nll,entropy,predictions,old._state(route,None,logits,target,tensors,counts))


def backward_tile(logits,target,state,upstream,*,out=None):
    if out is None:return old.backward_tile(logits,target,state,upstream)
    old._check(logits,target)
    if out.shape!=logits.shape or out.dtype!=logits.dtype or out.device!=logits.device or not out.is_contiguous():
        raise ValueError('Native half direct adjoint destination mismatch')
    p=state.metadata
    if (torch.is_grad_enabled() or type(state) is not TileState or type(p) is not TilePlan
            or p.rows!=logits.shape[0] or p.vocabulary!=logits.shape[1]
            or old.stamp(target)!=p.target_stamp
            or tuple(old.stamp(x) for x in state.tensors)!=p.saved_stamps
            or type(upstream) is not old.Upstream or old.stamp(upstream.tensor)!=upstream.tensor_stamp
            or upstream.tensor.device!=logits.device):
        raise RuntimeError('Direct adjoint state/upstream invalid')
    if p.route=='native' or not upstream.custom_eligible:
        out.copy_(old._native_backward(logits,target,upstream.tensor));return out
    maximum,logsum=state.tensors[:2];n,v=logits.shape
    if p.route=='all_safe' and len(state.tensors)==2:
        ce.soup_ce_exp_backward[(n,triton.cdiv(v,ce.TILE))](logits,target,maximum,logsum,
            upstream.tensor,out,V=v,BLOCK=ce.TILE,**ce._launch_config())
    elif p.route=='mixed_rows' and len(state.tensors)==4:
        guard,native_rows=state.tensors[2:]
        mixed.soup_ce_mixed_backward[(n,triton.cdiv(v,ce.TILE))](logits,target,maximum,logsum,
            upstream.tensor,guard,out,V=v,BLOCK=ce.TILE,**ce._launch_config())
        dz=old._native_backward(logits.index_select(0,native_rows),target.index_select(0,native_rows),upstream.tensor)
        out.index_copy_(0,native_rows,dz)
    else:raise RuntimeError('Invalid direct adjoint route/state shape')
    return out
