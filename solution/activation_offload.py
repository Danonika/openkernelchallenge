"""Original pipelined CUDA activation spill; T4 admission remains unverified."""
from functools import wraps
from contextvars import ContextVar
from threading import local
import os
import torch
_tls = local()
_active = ContextVar('own_spill', default=None)
_stats = dict(packed=0, restored=0, bytes=0, fallback=0, peak_staged=0)
class _Spill:
    __slots__ = ('host','ready','stage','shape','dtype','device')
    def __init__(self, host, event, stage, t):
        self.host,self.ready,self.stage = host,event,stage
        self.shape,self.dtype,self.device = tuple(t.shape),t.dtype,t.device
def counters():
    return dict(_stats)
def _candidate(t):
    if not isinstance(t,torch.Tensor) or not t.is_cuda or t.dtype!=torch.float16 or not t.is_contiguous() or t.ndim not in (2,3) or not t.numel():
        return False
    rows,width=t.numel()//t.shape[-1],t.shape[-1]
    return 512<=rows<=2048 and (width==8960 or (width==1536 and t.ndim==3))
def _stream(device):
    d=getattr(_tls,'streams',None)
    if d is None:
        _tls.streams=d={}
    key=(device.type,device.index)
    if key not in d:
        d[key]=torch.cuda.Stream(device=device)
    return d[key]
def _release(session,limit):
    compute=torch.cuda.current_stream(session['device'])
    pending=session['pending']
    while pending and pending[0][0]<=limit:
        _,v=pending.pop(0)
        compute.wait_event(v.ready)
        session['staged']-=v.stage.numel()*v.stage.element_size()
        v.stage=None
def _pack(t):
    s=_active.get()
    if s is None or not _candidate(t):
        return t
    try:
        host=torch.empty(t.shape,dtype=t.dtype,device='cpu',pin_memory=True)
    except (RuntimeError,MemoryError):
        _stats['fallback']+=1
        return t
    stage=torch.empty_like(t)
    stage.copy_(t)
    copy=_stream(t.device)
    compute=torch.cuda.current_stream(t.device)
    with torch.cuda.stream(copy):
        copy.wait_stream(compute)
        host.copy_(stage,non_blocking=True)
        event=torch.cuda.Event(blocking=False)
        event.record(copy)
    v=_Spill(host,event,stage,t)
    s['pending'].append((s['depth'],v))
    size=t.numel()*t.element_size()
    s['staged']+=size
    _stats['packed']+=1
    _stats['bytes']+=size
    _stats['peak_staged']=max(_stats['peak_staged'],s['staged'])
    if s['staged']>192*1024**2:
        _release(s,s['depth']-1)
    return v
def _unpack(v):
    if not isinstance(v,_Spill):
        return v
    compute=torch.cuda.current_stream(v.device)
    compute.wait_event(v.ready)
    x=torch.empty(v.shape,device=v.device,dtype=v.dtype)
    x.copy_(v.host,non_blocking=True)
    _stats['restored']+=1
    return x
def patch_decoder_activation_offload():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model,Qwen2DecoderLayer
    previous=Qwen2Model.forward
    if getattr(previous,'_soup_activation_spill',False):
        return True
    previous_layer=Qwen2DecoderLayer.forward
    @wraps(previous_layer)
    def layer_forward(self,*args,**kwargs):
        s=_active.get()
        if s is None:
            return previous_layer(self,*args,**kwargs)
        s['depth']+=1
        try:
            return previous_layer(self,*args,**kwargs)
        finally:
            _release(s,s['depth']-2)
    @wraps(previous)
    def forward(self,*args,**kwargs):
        if not self.training or not torch.is_grad_enabled() or getattr(self,'gradient_checkpointing',False) or not torch.cuda.is_available() or os.getenv('SOUP_DISABLE_ACTIVATION_SPILL')=='1':
            return previous(self,*args,**kwargs)
        ids=kwargs.get('input_ids',args[0] if args else None)
        if not isinstance(ids,torch.Tensor) or not ids.is_cuda or ids.ndim!=2 or ids.numel()<1024 or _active.get() is not None:
            return previous(self,*args,**kwargs)
        s={'device':ids.device,'pending':[],'depth':0,'staged':0}
        token=_active.set(s)
        try:
            with torch.autograd.graph.saved_tensors_hooks(_pack,_unpack):
                return previous(self,*args,**kwargs)
        finally:
            _release(s,s['depth'])
            _active.reset(token)
    forward._soup_activation_spill=True
    Qwen2DecoderLayer.forward=layer_forward
    Qwen2Model.forward=forward
    return True
