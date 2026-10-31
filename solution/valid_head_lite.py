"""Original input-only MLP + valid-token head fusion, not competitor code."""
from contextvars import ContextVar
import torch
_mask=ContextVar('own_head_mask',default=None)
_counts={'on':0,'off':0}
def patch():
    from . import integration as i,streamed_head as h,hero3_pack as p
    a=i._forward_streamed; b=h.streamed_linear_ce
    def f(lm,**kw):
        t=_mask.set(kw.get('attention_mask'));u=p._LABEL_CONTEXT.set(kw.get('labels'))
        try:return a(lm,**kw)
        finally:_mask.reset(t);p._LABEL_CONTEXT.reset(u)
    def head(x,w,y,n=None,c=128,pj='torch',s=True):
        m=_mask.get()
        def fallback():
            _counts['off']+=1
            return b(x,w,y,n,c,pj,s)
        if not isinstance(m,torch.Tensor) or m.shape!=y.shape or m.device!=y.device or x.shape[:2]!=y.shape or not x.requires_grad or pj!='torch':
            return fallback()
        v=m!=0;l=v.sum(1)
        if not bool(l.gt(0).all().item()) or not bool((v[:,:-1]>=v[:,1:]).all().item()) or not bool(((y==-100)|v).all().item()):
            return fallback()
        ix=v.flatten().nonzero().flatten()
        if ix.numel()==v.numel():return fallback()
        yy=y.flatten().index_select(0,ix).clone()
        yy.index_fill_(0,torch.cat((l.new_zeros(1),l.cumsum(0)[:-1])).long(),-100)
        z=b(x.flatten(0,1).index_select(0,ix)[None],w,yy[None],n,c,pj,s)
        def expand(t):
            return y.new_zeros(y.shape,dtype=t.dtype).flatten().index_copy(0,ix,t.flatten()).view_as(y)
        _counts['on']+=1
        return h.HeadResult(z.loss,expand(z.token_loss),expand(z.entropy),expand(z.predictions))
    i._forward_streamed=f;h.streamed_linear_ce=head
    return True
