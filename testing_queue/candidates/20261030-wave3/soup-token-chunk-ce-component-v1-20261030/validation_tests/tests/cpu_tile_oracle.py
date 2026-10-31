"""CPU-only semantic stand-in for orchestration tests, NEVER target payload."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
@dataclass(frozen=True)
class TilePlan:
    rows:int
@dataclass(frozen=True)
class TileState:
    metadata:TilePlan
    tensors:tuple
@dataclass(frozen=True)
class TileForward:
    token_nll:torch.Tensor
    token_entropy:torch.Tensor
    first_argmax:torch.Tensor
    state:TileState

captures=[]
q_seen=[]
def forward_tile(raw,target):
    public=raw.float();logp=F.log_softmax(public,-1)
    nll=F.cross_entropy(public,target,ignore_index=-100,reduction='none')
    entropy=-(logp.exp()*logp).sum(-1);pred=public.argmax(-1)
    # Small saved tensor exercises native owner version checks. No gradient or
    # full-vocabulary tensor is retained by this oracle state.
    maximum=public.amax(-1)
    captures.append(maximum)
    return TileForward(nll,entropy,pred,TileState(TilePlan(raw.shape[0]),(maximum,)))
def prepare_upstream(q):
    assert not torch.is_grad_enabled();q_seen.append(q.detach().clone());return q

def backward_tile(raw,target,state,q):
    assert len(state.tensors)==1 and state.metadata.rows==raw.shape[0]
    with torch.enable_grad():
        raw=raw.detach().requires_grad_();loss=F.cross_entropy(raw.float(),target,ignore_index=-100,reduction='sum')
        return torch.autograd.grad(loss,raw,grad_outputs=q)[0]
