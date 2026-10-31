"""Source-pinned reversible installer for the target-proven half producer route.

The same mechanism already exists in the user's activation_storage.py. This is
its narrow composition boundary, not new kernel mathematics or kernel credit.
Only Qwen2MLP.forward is owned. The exact tested primitive/JIT/eligibility bytes
remain unchanged; the prepared guard explicitly recognizes this ONE replacement.
"""
from collections import Counter
import hashlib,inspect,weakref
from importlib.metadata import version
from pathlib import Path
from .mlp_half import native_owner as native,starter_half as half,starter_swiglu as starter

EXPECTED_FILES={
 'native_owner.py':'7b69f2a354def25cb53d4ca78f3c19082497838c5f6ee133d94694e6113d540c',
 'owned_half.py':'8a714205c4f1de8ff7997f9430e47fb417b54e66879b64fa80144896c52a807b',
 'starter_half.py':'3759f3761b3b3d2a8e018b80a8571c9cc1456b8adc1ed40e47e1c919132e7dda',
 'starter_swiglu.py':'e665910faeca6091c958ed7a191604484a44216f51d6fa069f5f48220e7472e6',
}
EXCLUDED_SYMBOLS=('example_swiglu_fwd_kernel','example_swiglu_bwd_kernel')
_ACTIVE=None


def _owners():
    return (native.verify_sources,native.current_sources,native.canonical_module_dispatch,
            native.canonical_instance_dispatch,native.NativeMLPHalfOwner,native.NativeMLPHalfOwner.reason,
            half.NativeMLPHalfOwner,half.OwnedStarterHalf,half.OwnedStarterHalf.forward,
            half.OwnedStarterHalf.backward,half.owned_starter_mlp,half.starter,
            starter.example_swiglu_fwd_kernel,starter.example_swiglu_bwd_kernel)


_OWNERS=_owners()
_CODES=tuple((fn,fn.__code__) for fn in _OWNERS if inspect.isfunction(fn))
_KERNEL_DETAILS=tuple((k,k.fn,k.fn.__code__,k.src) for k in
                      (starter.example_swiglu_fwd_kernel,starter.example_swiglu_bwd_kernel))


def _module_intact():
    try:
        return (all(a is b for a,b in zip(_owners(),_OWNERS))
                and all(fn.__code__ is code for fn,code in _CODES)
                and all(k.fn is fn and fn.__code__ is code and k.src==source for k,fn,code,source in _KERNEL_DETAILS)
                and half.starter is starter and type(starter.BLOCK) is int and starter.BLOCK==1024
                and _InstalledGuard is _GUARD_CLASS and _InstalledGuard.reason is native.NativeMLPHalfOwner.reason
                and _InstalledGuard.__init__ is _GUARD_INIT and _GUARD_INIT.__code__ is _GUARD_INIT_CODE
                and isinstance(_InstalledGuard.module,property) and _InstalledGuard.module.fget is _GUARD_GETTER
                and _GUARD_GETTER.__code__ is _GUARD_GETTER_CODE)
    except (AttributeError,TypeError):return False


class _InstalledGuard(native.NativeMLPHalfOwner):
    """Reuse exact reason() under a verified installation, without a strong model ref.

    Native constructor cannot be called after our intentional class replacement:
    it correctly requires original Qwen2MLP.forward. Instead the installation
    has already verified all original sources and records exact replacement
    ownership. ONLY its mlp entry is changed in this guard's expected identities.
    All other eligibility logic, canonical dispatch and producer checks stay
    the original target-tested NativeMLPHalfOwner.reason, byte-for-byte.
    """
    def __init__(self,module,installation):
        self._module_ref=weakref.ref(module)
        self.first_order_only=True
        self.sources=dict(installation.native_sources,mlp=installation.installed)
        self.dispatch=installation.dispatch
        self.identities={name:id(getattr(module,name,None)) for name in ('gate_proj','up_proj','down_proj','act_fn')}

    @property
    def module(self):return self._module_ref()


_GUARD_CLASS=_InstalledGuard
_GUARD_INIT=_InstalledGuard.__init__;_GUARD_INIT_CODE=_GUARD_INIT.__code__
_GUARD_GETTER=_InstalledGuard.module.fget;_GUARD_GETTER_CODE=_GUARD_GETTER.__code__

def _make_forward(installation):
    # Bind exact dispatchers rather than dynamically invoking installation.call.
    # Reject class/instance monkeypatches before any potentially changed method.
    call=_HANDLE_CALL;call_code=_HANDLE_CALL_CODE
    dispatch_check=_installer_dispatch_intact;dispatch_code=_DISPATCH_CHECK_CODE
    def forward(module,x):
        if (dispatch_check.__code__ is not dispatch_code or call.__code__ is not call_code
                or not dispatch_check(installation)):
            raise RuntimeError('Compact MLP installation dispatcher ownership changed')
        return call(installation,module,x)
    return forward


_FORWARD_CODE=next(c for c in _make_forward.__code__.co_consts if inspect.iscode(c) and c.co_name=='forward')
_FACTORY_CODE=_make_forward.__code__


class CompactMLPInstallation:
    def __init__(self,sources,dispatch):
        self.owner=native.Qwen2MLP;self.original=sources['mlp'];self.native_sources=sources
        self.native_codes={name:fn.__code__ for name,fn in sources.items()}
        self.dispatch=dispatch;self.dispatch_codes={name:fn.__code__ for name,fn in dispatch.items()}
        self.installed=_make_forward(self);self.active=True
        self.guards=weakref.WeakKeyDictionary();self.counts=Counter(owned=0,native=0);self.reasons=Counter()

    def _owns_forward(self):
        f=self.installed
        expected={'installation':self,'call':_HANDLE_CALL,'call_code':_HANDLE_CALL_CODE,
                  'dispatch_check':_installer_dispatch_intact,'dispatch_code':_DISPATCH_CHECK_CODE}
        return (inspect.isfunction(f) and f.__code__ is _FORWARD_CODE and f.__globals__ is globals()
                and not hasattr(f,'__wrapped__') and f.__closure__ is not None
                and len(f.__closure__)==5
                and all(value is expected[name] for name,value in zip(f.__code__.co_freevars,
                           (cell.cell_contents for cell in f.__closure__))))

    def intact(self):
        if not _installer_dispatch_intact(self):return False
        if not self.active or self.owner.forward is not self.installed or not self._owns_forward() or not _module_intact():return False
        if _make_forward.__code__ is not _FACTORY_CODE:return False
        current=native.current_sources()
        if any(current[name] is not (self.installed if name=='mlp' else fn) for name,fn in self.native_sources.items()):return False
        if any(fn.__code__ is not self.native_codes[name] for name,fn in self.native_sources.items()):return False
        import torch
        if any(getattr(torch.nn.Module,name) is not fn or fn.__code__ is not self.dispatch_codes[name] for name,fn in self.dispatch.items()):return False
        return True

    def call(self,module,x):
        if not self.intact():raise RuntimeError('Compact MLP source/callable ownership changed after installation')
        if type(module) is not self.owner:
            why='foreign_mlp'
        else:
            guard=self.guards.get(module)
            if guard is None:
                guard=_InstalledGuard(module,self);self.guards[module]=guard
            why=guard.reason(x)
        if why is not None:
            self.counts['native']+=1;self.reasons[why]+=1
            # Never module(x): that would re-enter this installed forward.
            return self.original(module,x)
        result=half.owned_starter_mlp(module,x)
        self.counts['owned']+=1
        return result

    def status(self):
        return {'active':self.active,'owners_intact':self.intact(),'counts':dict(self.counts),
                'reasons':dict(self.reasons),'tracked_live_modules':len(self.guards),
                'original_kernel_credit':0,'excluded_symbols':list(EXCLUDED_SYMBOLS),
                'source_sha256':dict(EXPECTED_FILES),'first_order_only':True,
                'exact_reference':'unchanged official starter, not native SiLU',
                'counters_are_python_route_returns_not_gpu_completion':True,
                'scope':'actual_native_FP16_gate_up_producers_FP32_output_and_dy_direct_half_adjoints'}

    def restore(self):
        if not self.active:return False
        restored=(self.owner.forward is self.installed and _HANDLE_OWNS.__code__ is _HANDLE_OWNS_CODE
                  and _HANDLE_OWNS(self))
        if restored:self.owner.forward=self.original
        self.active=False;self.guards.clear()
        return restored


_HANDLE_CLASS=CompactMLPInstallation
_HANDLE_METHODS=tuple((name,getattr(CompactMLPInstallation,name),getattr(CompactMLPInstallation,name).__code__)
                      for name in ('__init__','_owns_forward','intact','call','status','restore'))
_HANDLE_CALL=CompactMLPInstallation.call;_HANDLE_CALL_CODE=_HANDLE_CALL.__code__
_HANDLE_OWNS=CompactMLPInstallation._owns_forward;_HANDLE_OWNS_CODE=_HANDLE_OWNS.__code__


def _installer_dispatch_intact(installation):
    if CompactMLPInstallation is not _HANDLE_CLASS:return False
    if installation is not None and type(installation) is not _HANDLE_CLASS:return False
    return all(getattr(CompactMLPInstallation,name) is fn and fn.__code__ is code
               and (installation is None or name not in installation.__dict__)
               for name,fn,code in _HANDLE_METHODS)


_DISPATCH_CHECK_CODE=_installer_dispatch_intact.__code__


def install_compact_mlp():
    """Global-before-model-load; no numerical function/GPU/model call on install."""
    global _ACTIVE
    if _ACTIVE is not None and _ACTIVE.active:
        if not _ACTIVE.intact():raise RuntimeError('Compact MLP ownership changed after installation')
        return _ACTIVE
    if version('bitsandbytes')!='0.50.2' or version('transformers')!='5.16.1':
        raise RuntimeError('Pinned compact MLP dependencies differ')
    directory=Path(native.__file__).parent
    if (not _module_intact() or not _installer_dispatch_intact(None)
            or _installer_dispatch_intact.__code__ is not _DISPATCH_CHECK_CODE or _make_forward.__code__ is not _FACTORY_CODE
            or any(hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest for name,digest in EXPECTED_FILES.items())):
        raise RuntimeError('Exact target-proven compact MLP source/callable changed')
    sources=native.verify_sources();dispatch=native.canonical_module_dispatch()
    handle=CompactMLPInstallation(sources,dispatch)
    try:
        handle.owner.forward=handle.installed
        if not handle.intact():raise RuntimeError('Compact MLP installation failed exact ownership')
        _ACTIVE=handle
    except BaseException:
        # Restore only this exact closure/code/globals owner. Foreign mutation
        # after assignment is never overwritten, including during failure.
        handle.restore()
        raise
    return handle
