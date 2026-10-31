"""Owned lossless FP16 storage of a native pre-PEFT tied embedding/head.

No arbitrary FP32 compression, custom gather, new kernel, or kernel credit.
Native embedding lookup returns its exact values widened to FP32, including
native index errors. Outside native FP16 autocast the head reconstructs its
exact original FP32 operand transiently. No full FP32 backup is retained.
"""
from collections import Counter
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import version
import hashlib,inspect,weakref
from types import MethodType
import torch
import torch.nn.functional as F

EXPECTED={
 'prepare':('peft.utils.other','prepare_model_for_kbit_training','8a080fa0f22a185031e6ad97f15ed5bb19e79371209066302a34b21696df6df4'),
 'embedding':('torch.nn.modules.sparse','Embedding.forward','1dd1797f482e42eba706513121ea2339b2cf3592c806a4eac2bca0093abac66a'),
 'linear':('torch.nn.modules.linear','Linear.forward','c0ab8f42cc3bf9489809b19c321cc85f0b1d4eafebacb96bd8e132cb9e95ef76')}
_ACTIVE=None
VERIFY_CHUNK=4*1024*1024


def verify_source(name,fn):
    module,qualname,digest=EXPECTED[name]
    if (not inspect.isfunction(fn) or hasattr(fn,'__wrapped__') or (fn.__module__,fn.__qualname__)!=(module,qualname)
            or hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()!=digest):
        raise RuntimeError('Native frozen-storage owner changed: '+name)


def canonical_module_dispatch():
    source=inspect.getsourcefile(torch.nn.Module);result={}
    for name,qualname in (('__call__','Module._wrapped_call_impl'),('_call_impl','Module._call_impl')):
        fn=getattr(torch.nn.Module,name)
        if (not inspect.isfunction(fn) or hasattr(fn,'__wrapped__') or fn.__module__!='torch.nn.modules.module'
                or fn.__qualname__!=qualname or fn.__code__.co_filename!=source or inspect.getsourcefile(fn)!=source):
            raise RuntimeError('Foreign canonical Module dispatch owner')
        result[name]=fn
    return result


def _plain_dispatch(module):
    return (type(module) in (torch.nn.Embedding,torch.nn.Linear)
            and all(getattr(type(module),name) is getattr(torch.nn.Module,name) and name not in module.__dict__
                    for name in ('__call__','_call_impl'))
            and getattr(module,'_compiled_call_impl',None) is None)


def _method_is(module,fn):
    actual=module.__dict__.get('forward')
    return isinstance(actual,MethodType) and actual.__self__ is module and actual.__func__ is fn


def _embedding_metadata(module):
    return (module.num_embeddings,module.embedding_dim,module.padding_idx,module.max_norm,
            module.norm_type,module.scale_grad_by_freq,module.sparse)


@dataclass
class StorageRecord:
    model_ref:object
    embedding_ref:object
    head_ref:object
    parameter_ref:object
    half_storage:torch.Tensor
    embedding_metadata:tuple
    parameter_version:int
    storage_version:int
    shape:tuple
    device:object
    pointer:int


def _record(module,kind):
    owner=_ACTIVE
    if (_owner_dispatch_intact is not _DISPATCH_FN or _DISPATCH_FN.__code__ is not _DISPATCH_CODE
            or not _DISPATCH_FN(owner) or not owner.intact()):raise RuntimeError('Frozen-storage installation is not intact')
    record=(owner.by_head if kind=='head' else owner.by_embedding).get(id(module))
    if record is None or not owner.valid(record):raise RuntimeError('Unattested or changed frozen storage')
    if (record.head_ref() if kind=='head' else record.embedding_ref()) is not module:
        raise RuntimeError('Frozen-storage module identity changed')
    return owner,record


def _embedding_forward(module,token_ids):
    owner,record=_record(module,'embedding')
    # Captured native method uses F.embedding with all original module options.
    # Outer Module.__call__ still executes every original input-grad/user hook.
    result=owner.native_embedding(module,token_ids).float()
    owner.counts['embedding_calls']+=1
    return result


def _head_forward(module,hidden_states):
    owner,record=_record(module,'head')
    device_type=hidden_states.device.type
    if torch.is_autocast_enabled(device_type) and torch.get_autocast_dtype(device_type)==torch.float16:
        result=owner.native_linear(module,hidden_states)
        owner.counts['head_fp16_calls']+=1
        return result
    # Preserve native FP32 operands and dtype errors outside FP16 AMP. BF16
    # autocast, when requested by a public caller, also sees the original FP32
    # operand before its own native autocast conversion.
    result=F.linear(hidden_states,module.weight.float(),module.bias)
    owner.counts['head_native_operand_calls']+=1
    return result


def attest_weight(weight):
    owner=_ACTIVE
    if (_owner_dispatch_intact is not _DISPATCH_FN or _DISPATCH_FN.__code__ is not _DISPATCH_CODE
            or not _DISPATCH_FN(owner) or not owner.intact()):return False
    record=owner.by_weight.get(id(weight))
    return record is not None and record.parameter_ref() is weight and owner.valid(record)


def attest_head(model,head,weight,native_linear):
    owner=_ACTIVE
    if (_owner_dispatch_intact is not _DISPATCH_FN or _DISPATCH_FN.__code__ is not _DISPATCH_CODE
            or not _DISPATCH_FN(owner) or not owner.intact() or native_linear is not owner.native_linear):return False
    record=owner.by_head.get(id(head))
    return (model is not None and record is not None and record.model_ref() is model and record.head_ref() is head
            and record.parameter_ref() is weight and owner.valid(record))


class StorageInstallation:
    """Private injectable construction supports CPU source-boundary proofs.

    Public install_storage() always pins real PEFT/Qwen2 and CUDA. Tests may
    instantiate this owner with the inspected native PEFT function, a tiny CPU
    module class and device_type='cpu'; that is not target-runtime validation.
    """
    def __init__(self,peft_module,prepare,model_class,*,device_type='cuda'):
        self.peft=peft_module;self.original_prepare=prepare;self.model_class=model_class;self.device_type=device_type
        self.native_embedding=torch.nn.Embedding.forward;self.native_linear=torch.nn.Linear.forward
        self.native_codes={fn:fn.__code__ for fn in (prepare,self.native_embedding,self.native_linear)}
        self.dispatch=canonical_module_dispatch();self.dispatch_codes={name:fn.__code__ for name,fn in self.dispatch.items()}
        self.records=weakref.WeakKeyDictionary();self.live_records={};self.by_head=weakref.WeakValueDictionary()
        self.by_embedding=weakref.WeakValueDictionary();self.by_weight=weakref.WeakValueDictionary()
        self.counts=Counter(captured=0,installed=0,declined=0,embedding_calls=0,head_fp16_calls=0,head_native_operand_calls=0);self.reasons=Counter();self.active=True
        signature=inspect.signature(prepare)
        installation=self
        @wraps(prepare)
        def prepared(model,*args,**kwargs):
            if (_owner_dispatch_intact is not _DISPATCH_FN or _DISPATCH_FN.__code__ is not _DISPATCH_CODE
                    or not _DISPATCH_FN(installation) or not installation.intact()):raise RuntimeError('Frozen-storage preparation ownership changed')
            bound=signature.bind(model,*args,**kwargs);bound.apply_defaults()
            candidate=installation.capture(model,bound.arguments['use_gradient_checkpointing'])
            result=prepare(model,*args,**kwargs)
            if candidate is not None:installation.complete(candidate,result)
            return result
        self.prepared=prepared;self.prepared_code=prepared.__code__
        self.prepared_closure=tuple(cell.cell_contents for cell in prepared.__closure__)

    def intact(self):
        if (_owner_dispatch_intact is not _DISPATCH_FN or _DISPATCH_FN.__code__ is not _DISPATCH_CODE or not _DISPATCH_FN(self)):return False
        return (self.active and self.peft.prepare_model_for_kbit_training is self.prepared
                and self.prepared.__code__ is self.prepared_code
                and tuple(cell.cell_contents for cell in self.prepared.__closure__)==self.prepared_closure
                and torch.nn.Embedding.forward is self.native_embedding and torch.nn.Linear.forward is self.native_linear
                and all(fn.__code__ is code for fn,code in self.native_codes.items())
                and all(getattr(torch.nn.Module,name) is fn and fn.__code__ is self.dispatch_codes[name] for name,fn in self.dispatch.items())
                and _embedding_forward.__code__ is _EMBED_CODE and _head_forward.__code__ is _HEAD_CODE)

    def decline(self,reason):self.counts['declined']+=1;self.reasons[reason]+=1;return None

    def capture(self,model,checkpointing):
        if (type(model) is not self.model_class or not getattr(model,'is_loaded_in_4bit',False)
                or getattr(model,'is_loaded_in_8bit',False) or checkpointing or model in self.records):return self.decline('model_or_checkpointing')
        embedding=model.get_input_embeddings();head=model.get_output_embeddings()
        if (type(embedding) is not torch.nn.Embedding or type(head) is not torch.nn.Linear
                or not _plain_dispatch(embedding) or not _plain_dispatch(head)
                or 'forward' in embedding.__dict__ or 'forward' in head.__dict__
                or hasattr(embedding,'_hf_hook') or hasattr(head,'_hf_hook')
                or head.bias is not None or embedding.max_norm is not None or embedding.sparse or embedding.scale_grad_by_freq):
            return self.decline('module_owner_or_options')
        weight=embedding.weight
        if (type(weight) is not torch.nn.Parameter or weight is not head.weight or weight.dtype!=torch.float16
                or weight.device.type!=self.device_type or weight.ndim!=2 or not weight.is_contiguous()
                or weight.is_conj() or weight.is_neg() or not weight.numel()
                or tuple(weight.shape)!=(embedding.num_embeddings,embedding.embedding_dim)
                or tuple(weight.shape)!=(head.out_features,head.in_features)
                or not getattr(model.config,'tie_word_embeddings',False)):
            return self.decline('weight_provenance')
        self.counts['captured']+=1
        return (model,embedding,head,weight,weight.detach(),_embedding_metadata(embedding))

    def complete(self,candidate,model):
        original,embedding,head,weight,half,metadata=candidate
        if (model is not original or model.get_input_embeddings() is not embedding or model.get_output_embeddings() is not head
                or embedding.weight is not weight or head.weight is not weight or weight.requires_grad
                or weight.dtype!=torch.float32 or weight.shape!=half.shape or weight.device!=half.device
                or not weight.is_contiguous() or _embedding_metadata(embedding)!=metadata
                or 'forward' in embedding.__dict__ or 'forward' in head.__dict__):return self.decline('preparation_changed_owners')
        current=weight.detach().reshape(-1);origin=half.reshape(-1)
        # Exact FP32 bits, including negative zero. No full FP32 backup survives.
        for start in range(0,current.numel(),VERIFY_CHUNK):
            stop=min(start+VERIFY_CHUNK,current.numel());expected=origin[start:stop].float();actual=current[start:stop]
            if not bool(expected.isfinite().all()) or not torch.equal(actual.view(torch.int32),expected.view(torch.int32)):
                return self.decline('preparation_changed_exact_values')
        owner_ref=weakref.ref(self);parameter_id=id(weight)
        def parameter_gone(reference):
            owner=owner_ref()
            if owner is not None:
                current_record=owner.live_records.get(parameter_id)
                if current_record is not None and current_record.parameter_ref is reference:owner.live_records.pop(parameter_id,None)
        parameter_ref=weakref.ref(weight,parameter_gone)
        record=StorageRecord(weakref.ref(model),weakref.ref(embedding),weakref.ref(head),parameter_ref,half,
            metadata,weight._version,half._version,tuple(weight.shape),weight.device,half.data_ptr())
        try:
            weight.data=half
            embedding.forward=MethodType(_embedding_forward,embedding);head.forward=MethodType(_head_forward,head)
            self.records[model]=record;self.live_records[id(weight)]=record
            self.by_head[id(head)]=record;self.by_embedding[id(embedding)]=record;self.by_weight[id(weight)]=record
            if not self.valid(record):raise RuntimeError('Frozen-storage installation did not establish exact attestation')
        except BaseException as original_error:
            clean=True
            # Remove only exact owned instance bindings, never a foreign
            # replacement installed during a failing assignment/verification.
            for module,proxy in ((embedding,_embedding_forward),(head,_head_forward)):
                bound=module.__dict__.get('forward')
                if isinstance(bound,MethodType) and bound.__self__ is module and bound.__func__ is proxy:del module.forward
                elif 'forward' in module.__dict__:clean=False
            if (weight.data_ptr()==half.data_ptr() and weight.dtype==torch.float16
                    and weight._version==record.parameter_version and half._version==record.storage_version):
                weight.data=half.float()
            else:clean=False
            if self.records.get(model) is record:self.records.pop(model,None)
            if self.live_records.get(id(weight)) is record:self.live_records.pop(id(weight),None)
            if not clean:raise RuntimeError('Frozen-storage partial rollback preserved foreign state') from original_error
            raise
        self.counts['installed']+=1

    def valid(self,record):
        if not self.intact() or type(record) is not StorageRecord:return False
        model=record.model_ref();embedding=record.embedding_ref();head=record.head_ref();weight=record.parameter_ref()
        if weight is None or self.live_records.get(id(weight)) is not record:return False
        if model is not None and (type(model) is not self.model_class or self.records.get(model) is not record
                or embedding is None or head is None or model.get_input_embeddings() is not embedding
                or model.get_output_embeddings() is not head):return False
        # Provenance remains alive with an orphaned head/embedding/Parameter,
        # without retaining the former model. Existing proxies stay valid and
        # restore() must still restore every live owner after model collection.
        if embedding is not None and (type(embedding) is not torch.nn.Embedding or not _plain_dispatch(embedding)
                or embedding.weight is not weight or not _method_is(embedding,_embedding_forward)
                or _embedding_metadata(embedding)!=record.embedding_metadata):return False
        if head is not None and (type(head) is not torch.nn.Linear or not _plain_dispatch(head)
                or head.weight is not weight or head.bias is not None or not _method_is(head,_head_forward)):return False
        return (type(weight) is torch.nn.Parameter and weight.dtype==torch.float16 and not weight.requires_grad
                and tuple(weight.shape)==record.shape and weight.device==record.device and weight.is_contiguous()
                and not weight.is_neg() and not weight.is_conj() and weight.data_ptr()==record.pointer
                and weight._version==record.parameter_version and record.half_storage._version==record.storage_version)

    def status(self):
        return {'active':self.active,'owners_intact':self.intact(),'counts':dict(self.counts),'reasons':dict(self.reasons),
                'live_models':len(self.records),'live_storage_owners':len(self.live_records),'original_kernel_credit':0,'native_embedding_lookup':True,
                'provenance':'retained_actual_pre_PEFT_FP16_exact_post_PEFT_FP32_bits'}

    def restore(self):
        if not self.active:return False
        restored=True
        for record in list(self.live_records.values()):
            if not self.valid(record):restored=False;continue
            model=record.model_ref();embedding=record.embedding_ref();head=record.head_ref();weight=record.parameter_ref()
            weight.data=record.half_storage.float()
            if embedding is not None:del embedding.forward
            if head is not None:del head.forward
            if model is not None:self.records.pop(model,None)
            self.live_records.pop(id(weight),None)
        if self.peft.prepare_model_for_kbit_training is self.prepared:self.peft.prepare_model_for_kbit_training=self.original_prepare
        else:restored=False
        self.active=False
        return restored


_EMBED_CODE=_embedding_forward.__code__;_HEAD_CODE=_head_forward.__code__


def install_storage():
    global _ACTIVE
    if _ACTIVE is not None and _ACTIVE.active:
        if not _ACTIVE.intact():raise RuntimeError('Frozen-storage ownership changed after installation')
        return _ACTIVE
    if version('peft')!='0.20.0' or version('transformers')!='5.16.1':raise RuntimeError('Pinned storage dependencies differ')
    import peft
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    for name,fn in [('prepare',peft.prepare_model_for_kbit_training),('embedding',torch.nn.Embedding.forward),('linear',torch.nn.Linear.forward)]:verify_source(name,fn)
    handle=StorageInstallation(peft,peft.prepare_model_for_kbit_training,Qwen2ForCausalLM)
    try:
        peft.prepare_model_for_kbit_training=handle.prepared;_ACTIVE=handle
        if not handle.intact():raise RuntimeError('Storage preparation hook ownership not established')
    except BaseException:
        handle.restore();raise
    return handle


_OWNER_CLASS=StorageInstallation
_OWNER_METHODS=tuple((name,getattr(StorageInstallation,name),getattr(StorageInstallation,name).__code__)
                    for name in ('__init__','intact','decline','capture','complete','valid','status','restore'))

def _owner_dispatch_intact(owner):
    if StorageInstallation is not _OWNER_CLASS or type(owner) is not _OWNER_CLASS:return False
    if not all(getattr(StorageInstallation,name) is fn and fn.__code__ is code and name not in owner.__dict__
               for name,fn,code in _OWNER_METHODS):return False
    return all(globals().get(name) is fn and fn.__code__ is code for name,fn,code in _HELPER_PINS)

_DISPATCH_FN=_owner_dispatch_intact;_DISPATCH_CODE=_owner_dispatch_intact.__code__
_HELPER_PINS=tuple((name,globals()[name],globals()[name].__code__) for name in
    ('verify_source','canonical_module_dispatch','_plain_dispatch','_method_is','_embedding_metadata',
     '_record','_embedding_forward','_head_forward','attest_weight','attest_head','install_storage','_owner_dispatch_intact'))
