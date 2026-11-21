"""Fusion A: namespace-preserved NF4 + LoRA epilogue + materialized CE.

Installation is NF4 -> LoRA -> CE, before model construction. No component is
silently skipped: a refused component rolls back owned patches and aborts.
"""
from importlib.metadata import version
from .lifecycle import Ownership, note

_EXPECTED={"torch":"2.11.0+cu128","triton":"3.6.0","transformers":"5.16.1",
           "peft":"0.20.0","bitsandbytes":"0.50.2"}
_ownership=None
_active=False


def _owned_wrapper(current, original, marker=None):
    return (current is not original and getattr(current,"__wrapped__",None) is original
            and (marker is None or getattr(current,marker,False)))


def diagnostics():
    from . import ce_component
    value=ce_component.diagnostics()
    value["composition"]={"active":_active,"components":["nf4","lora","ce"],
                          "own_share":"UNMEASURED; component shares must not be added"}
    return value


def apply():
    global _ownership,_active
    if _active:
        _ownership.verify()
        from . import ce_component
        ce_component.apply()  # retains its own exact constructor ownership guard
        return True
    actual={name:version(name) for name in _EXPECTED}
    if actual!=_EXPECTED:
        raise RuntimeError("Fusion A requires exact prepared target pins; no partial install: "+repr(actual))
    from .nf4_component import routes as nf4
    from .lora_component import patch as lora
    from . import ce_component
    import bitsandbytes.functional as functional
    import bitsandbytes.backends.cuda.ops as cuda_ops
    from peft.tuners.lora import bnb as peft_bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    # Explicitly reject a pre-existing same-component hook: do not infer who
    # owns it, claim it as ours, or restore it after a failure.
    fwd,fun=cuda_ops._dequant_linear_fallback,functional.dequantize_4bit
    lora_forward=peft_bnb.Linear4bit.forward
    qwen_init=Qwen2ForCausalLM.__init__
    if (getattr(fwd,nf4._MARKER,False) or getattr(fun,nf4._MARKER,False)
            or lora._OWNER is not None or getattr(ce_component._handle,"active",False)):
        raise RuntimeError("Pre-existing component hook; no composition installed")
    owned=Ownership()
    def capture():
        existing={(id(b.owner),b.attribute) for b in owned.bindings}
        proposed=((cuda_ops,"_dequant_linear_fallback",fwd,"nf4_forward",nf4._MARKER),
                  (functional,"dequantize_4bit",fun,"nf4_functional",nf4._MARKER),
                  (peft_bnb.Linear4bit,"forward",lora_forward,"lora",None),
                  (Qwen2ForCausalLM,"__init__",qwen_init,"ce_constructor",None))
        for owner,attribute,prior,component,marker in proposed:
            current=getattr(owner,attribute)
            if (id(owner),attribute) not in existing and _owned_wrapper(current,prior,marker):
                owned.add(owner,attribute,prior,current,component)
    try:
        outcome=nf4.install();capture()
        if outcome.get("patched") is not True or outcome.get("already_patched"):
            raise RuntimeError("NF4 refused installation: "+repr(outcome))
        lora.install();capture()
        if lora._OWNER is None: raise RuntimeError("LoRA installation missing")
        if ce_component.apply() is not True: raise RuntimeError("CE installation refused")
        capture()
        if len(owned.bindings)!=4: raise RuntimeError("Expected four disjoint class/function bindings")
        owned.verify()
    except BaseException:
        capture()
        # CE may have installed per-model loss wrappers; its own restore handles
        # those without replacing foreign losses. No model is constructed here.
        ce_component.restore()
        restored=owned.restore()
        if lora._OWNER is not None and peft_bnb.Linear4bit.forward is lora_forward:
            lora._OWNER=None
        note("installation_rolled_back",restoration=restored)
        raise
    _ownership=owned;_active=True
    note("installed",order=["nf4","lora","ce"],bindings=4,
         task_changed=False,kernel_share="UNMEASURED_NOT_ADDITIVE")
    return True


def restore():
    global _active,_ownership
    from . import ce_component
    ce_result=ce_component.restore()
    if _ownership is None or not _active:
        return ce_result
    # CE constructor may already be native after CE's own restore; that is a
    # completed restoration, not an excuse to overwrite an unrelated constructor.
    restorations={}
    for binding in reversed(_ownership.bindings):
        if binding.component=="ce_constructor" and getattr(binding.owner,binding.attribute) is binding.previous:
            restorations[binding.component+":"+binding.attribute]="already_restored_by_ce"
        else: restorations[binding.component+":"+binding.attribute]=binding.restore()
    from .lora_component import patch as lora
    if lora._OWNER is not None:
        cls,original,wrapped=lora._OWNER
        if cls.forward is original: lora._OWNER=None
    _active=False
    note("restored",bindings=restorations)
    # Compatibility with the unchanged CE actual-class gate; extra composition
    # evidence is additive metadata, never a change to native model outputs.
    return dict(ce_result,composition_restoration=restorations)
