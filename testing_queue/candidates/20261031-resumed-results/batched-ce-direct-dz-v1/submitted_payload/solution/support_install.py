"""Source-pinned reversible composition of two independently reviewed modules.

Do not import the user's package entrypoint: it also installs unrelated patches
and a starter kernel. Only exact isolated original NF4 and RMSNorm modules enter
this composition. Counters observe Python calls, not GPU completion or timing.
"""
from collections import Counter
from functools import wraps
import hashlib,inspect,types
from .support import nf4,rmsnorm

_CREATORS=(nf4.patch_bitsandbytes_nf4,rmsnorm.patch_qwen2_rmsnorm)

def _created_code(factory,name):
    found=[c for c in factory.__code__.co_consts if isinstance(c,types.CodeType) and c.co_name==name]
    if len(found)!=1:raise RuntimeError("Reviewed patch closure code unavailable: "+name)
    return found[0]

_CREATED={
 "dequantize_4bit":(_created_code(_CREATORS[0],"dequantize_4bit"),nf4.__dict__,"original"),
 "_dequant_linear_fallback":(_created_code(_CREATORS[0],"dequant_linear_fallback"),nf4.__dict__,"original_fallback"),
 "forward":(_created_code(_CREATORS[1],"forward"),rmsnorm.__dict__,"original"),
}

def owned_wrapper(name,current,original,marker):
    code,globals_,closure=_CREATED[name]
    return (inspect.isfunction(current) and current.__code__ is code
            and current.__globals__ is globals_
            and inspect.getclosurevars(current).nonlocals=={closure:original}
            and getattr(current,"__wrapped__",None) is original
            and getattr(current,marker,None) is True)
from importlib.metadata import version

EXPECTED={
 'rmsnorm':'790fa612e4fa57acfcbea31da45f92ba91f9016f7260ef539f61c5530ba06a5f',
 'dequantize':'c928daa1321ac80321093537a64d2339751690b5e53b96aa4b302898197d9a15',
 'fallback':'ad09ce81b25cb0b8913f61c470b1b7942ac454bfc8aa27fe3a5aa7fa0d15cc6f',
}

def verify_native(name,fn):
    if hasattr(fn,'__wrapped__') or hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()!=EXPECTED[name]:
        raise RuntimeError('Foreign native auxiliary boundary: '+name)


class SupportInstallation:
    def __init__(self):self.records=[];self.counts=Counter();self.active=True
    def capture(self,owner,name,original,marker):
        current=getattr(owner,name)
        if not owned_wrapper(name,current,original,marker):
            raise RuntimeError('Auxiliary patch ownership failed: '+name)
        self.records.append((owner,name,original,current))
    def track(self,owner,name,event):
        original=getattr(owner,name)
        @wraps(original)
        def tracked(*args,**kwargs):
            self.counts[event]+=1
            return original(*args,**kwargs)
        setattr(owner,name,tracked);self.records.append((owner,name,original,tracked))
    def guard_metadata(self,norm_class,native_rms):
        # Raw-pointer kernels must not reinterpret logical lazy-negative views
        # or arbitrary Tensor subclasses. Keep the reviewed numerical modules
        # byte-identical and make unsupported calls take their native route.
        import torch
        from bitsandbytes.nn import Params4bit
        original=nf4._components_supported
        def physical(tensor,allowed):
            return type(tensor) in allowed and not tensor.is_neg() and not tensor.is_conj()
        @wraps(original)
        def supported(packed,shape,dtype,scales,blocksize,nested_scales=None,
                      nested_code=None,offset=None,nested_blocksize=256,out=None,*,cuda_only=False):
            if not physical(packed,(torch.Tensor,torch.nn.Parameter,Params4bit)):
                return False
            if any(tensor is not None and not physical(tensor,(torch.Tensor,torch.nn.Parameter))
                   for tensor in (scales,nested_scales,nested_code,offset,out)):
                return False
            return original(packed,shape,dtype,scales,blocksize,nested_scales,
                            nested_code,offset,nested_blocksize,out,cuda_only=cuda_only)
        nf4._components_supported=supported
        self.records.append((nf4,'_components_supported',original,supported))
        fused=norm_class.forward
        @wraps(fused)
        def forward(module,hidden_states):
            if (not physical(hidden_states,(torch.Tensor,))
                    or not physical(module.weight,(torch.Tensor,torch.nn.Parameter))):
                return native_rms(module,hidden_states)
            return fused(module,hidden_states)
        # One record per binding: the final owned outer wrapper restores the
        # captured native method directly, not a stale inner wrapper.
        index=next(i for i,(owner,name,old,ours) in enumerate(self.records)
                   if owner is norm_class and name=='forward' and ours is fused)
        norm_class.forward=forward
        self.records[index]=(norm_class,'forward',native_rms,forward)
    def intact(self):return self.active and all(getattr(owner,name) is ours for owner,name,old,ours in self.records)
    def status(self):
        return {'nf4':self.active,'rmsnorm':self.active,'owners_intact':self.intact(),
                'calls':dict(self.counts),'counters_are_python_entries_not_gpu_completion':True}
    def restore(self):
        restored=True
        for owner,name,old,ours in reversed(self.records):
            if getattr(owner,name) is ours:setattr(owner,name,old)
            else:restored=False
        self.active=False
        return restored


def install_support():
    if version('bitsandbytes')!='0.50.2' or version('transformers')!='5.16.1':
        raise RuntimeError('Pinned original NF4/RMS dependencies differ')
    import bitsandbytes.functional as functional
    from bitsandbytes.backends.cuda import ops
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from .support import nf4,rmsnorm
    native_dequant=functional.dequantize_4bit;native_fallback=ops._dequant_linear_fallback
    native_rms=Qwen2RMSNorm.forward
    for name,fn in [('dequantize',native_dequant),('fallback',native_fallback),('rmsnorm',native_rms)]:verify_native(name,fn)
    handle=SupportInstallation()
    boundaries=[(functional,'dequantize_4bit',native_dequant,'_soup_candidate_nf4'),
                (ops,'_dequant_linear_fallback',native_fallback,'_soup_candidate_nf4'),
                (Qwen2RMSNorm,'forward',native_rms,'_soup_kernel_rmsnorm')]
    try:
        if nf4.patch_bitsandbytes_nf4() is not True:raise RuntimeError('Original NF4 installation declined')
        handle.capture(functional,'dequantize_4bit',native_dequant,'_soup_candidate_nf4')
        handle.capture(ops,'_dequant_linear_fallback',native_fallback,'_soup_candidate_nf4')
        if rmsnorm.patch_qwen2_rmsnorm() is not True:raise RuntimeError('Original RMSNorm installation declined')
        handle.capture(Qwen2RMSNorm,'forward',native_rms,'_soup_kernel_rmsnorm')
        handle.guard_metadata(Qwen2RMSNorm,native_rms)
        handle.track(nf4,'_execute','nf4_decode')
        handle.track(rmsnorm,'rmsnorm','rmsnorm')
    except BaseException:
        # A patch can assign an owned wrapper and then raise before capture.
        # Recover only exact reviewed closure/code/globals owners, never a
        # foreign replacement merely spoofing marker and __wrapped__ fields.
        recorded={(id(owner),name) for owner,name,old,ours in handle.records}
        for owner,name,old,marker in boundaries:
            current=getattr(owner,name)
            if (id(owner),name) not in recorded and owned_wrapper(name,current,old,marker):
                handle.records.append((owner,name,old,current))
        handle.restore()
        raise
    return handle
