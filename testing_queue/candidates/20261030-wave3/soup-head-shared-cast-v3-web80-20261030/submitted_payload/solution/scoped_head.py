"""N33: one native float cast in an owned, loss-only Rewind transaction.

No custom loss/autograd/metrics. Private Qwen logits change to FP32 ONLY inside
this explicit native loss-only consumer chain; prepared public logits were FP32.
Never enable this for generic output-returning calls: cast-edge sharing can
change FP16 gradient accumulation with two differentiable consumers.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import inspect
import weakref

import torch
from torch.nn.modules import module as module_globals

_ACTIVE = ContextVar("soup_n33_owned_loss", default=None)
_HOOK_NAMES = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_GLOBAL_HOOK_NAMES = tuple("_global" + name for name in _HOOK_NAMES)


def no_hooks(module, *, own_head_hook=None):
    for name in _HOOK_NAMES:
        hooks = getattr(module, name, {})
        if name == "_forward_hooks" and own_head_hook is not None:
            if set(hooks) != {own_head_hook}:
                return False
        elif hooks:
            return False
    return True


def no_global_hooks():
    return not any(getattr(module_globals, name, {}) for name in _GLOBAL_HOOK_NAMES)


def raw_callable(value):
    return getattr(value, "__func__", value)


@dataclass(frozen=True)
class Owners:
    """Explicit dependency injection for CPU fixtures; production is pinned."""
    rewind_class: type
    sft_class: type
    hf_class: type
    peft_class: type
    qwen_class: type
    tuner_class: type
    rewind_loss: object
    sft_loss: object
    hf_loss: object
    peft_forward: object
    qwen_forward: object
    tuner_forward: object
    native_loss: object
    entropy_function: object
    rewind_rows: object
    get_base_model: object
    linear_forward: object
    converter_class: type
    converter_call: object
    converter_function: object
    converter_recursive: object
    converter_code: object
    autocast_code: object
    device_type: str = "cuda"
    vocabulary: int = 151936
    batch_size: int = 4


def prepared_chain_reason(model, owners):
    """Prove the normal Accelerate->autocast->original PEFT callable chain."""
    original = getattr(model, "_original_forward", None)
    if raw_callable(original) is not owners.peft_forward:
        return "original_peft_forward"
    outer = raw_callable(model.forward)
    if getattr(outer, "__code__", None) is not owners.converter_code:
        return "outer_conversion_code"
    converter = getattr(outer, "__wrapped__", None)
    if (type(converter) is not owners.converter_class
            or type(converter).__call__ is not owners.converter_call):
        return "outer_conversion_owner"
    try:
        closed = inspect.getclosurevars(outer).nonlocals
    except (TypeError, ValueError):
        return "outer_conversion_closure"
    if closed != {"model_forward": converter}:
        return "outer_conversion_closure"
    amp = converter.model_forward
    if (getattr(converter, "__wrapped__", None) is not amp
            or getattr(amp, "__code__", None) is not owners.autocast_code
            or getattr(amp, "__wrapped__", None) is not owners.peft_forward):
        return "autocast_callable_chain"
    try:
        amp_closure = inspect.getclosurevars(amp).nonlocals
    except (TypeError, ValueError):
        return "autocast_closure"
    context = amp_closure.get("autocast_instance")
    if (set(amp_closure) != {"autocast_instance", "func"}
            or amp_closure["func"] is not owners.peft_forward
            or type(context) is not torch.amp.autocast_mode.autocast
            or context.device != owners.device_type
            or context.fast_dtype != torch.float16 or not context._enabled):
        return "autocast_context"
    # Converter's global must still be the captured native recursive function.
    if (owners.converter_call.__globals__.get("convert_to_fp32") is not owners.converter_function
            or owners.converter_function.__globals__.get("recursively_apply") is not owners.converter_recursive):
        return "converter_global_changed"
    return None


class Installation:
    def __init__(self, owners, telemetry):
        self.o, self.telemetry = owners, telemetry
        cls = owners.rewind_class
        if cls.compute_loss is not owners.rewind_loss:
            raise RuntimeError("Foreign Rewind compute_loss is not overwritten")
        self.active = True

        @wraps(owners.rewind_loss)
        def loss_only(trainer, model, inputs, return_outputs=False, num_items_in_batch=None):
            reason, owner = self.guard(trainer, model, inputs, return_outputs)
            if reason is not None:
                telemetry.emit("native_fallback", reason=reason)
                # Nested calls must not inherit an outer transaction's head cast.
                token = _ACTIVE.set(None)
                try:
                    return owners.rewind_loss(trainer, model, inputs,
                        return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
                finally:
                    _ACTIVE.reset(token)
            transaction = {"owner": weakref.ref(owner), "calls": 0}
            head = owner.lm_head
            expected_weight = head.weight
            hook_handle = None

            def cast_once(module, args, output):
                if _ACTIVE.get() is not transaction:
                    return output
                if (not no_global_hooks() or not no_hooks(module, own_head_hook=hook_handle.id)
                        or owner.lm_head is not module or module.weight is not expected_weight
                        or module.weight.requires_grad or module.weight.dtype != torch.float32
                        or module.bias is not None or raw_callable(module.forward) is not owners.linear_forward
                        or owner.get_input_embeddings().weight is not expected_weight
                        or not owner.training or not torch.is_grad_enabled()
                        or type(output) is not torch.Tensor or not output.requires_grad
                        or output.dtype != torch.float16 or output.ndim != 3
                        or output.shape[:2] != inputs["labels"].shape
                        or output.shape[2] != owners.vocabulary or not output.is_contiguous()
                        or output.is_neg() or output.is_conj()
                        or output.device.type != owners.device_type
                        or not torch.is_autocast_enabled(owners.device_type)
                        or torch.get_autocast_dtype(owners.device_type) != torch.float16):
                    telemetry.emit("head_native_fallback", reason="inner_head_contract")
                    return output
                transaction["calls"] += 1
                if transaction["calls"] != 1:
                    raise RuntimeError("Multiple head invocations violate owned loss transaction")
                telemetry.emit("shared_cast", rows=output.shape[0]*output.shape[1],
                               vocabulary=output.shape[2])
                return output.float()

            hook_handle = head.register_forward_hook(cast_once)
            token = _ACTIVE.set(transaction)
            try:
                result = owners.rewind_loss(trainer, model, inputs,
                    return_outputs=False, num_items_in_batch=num_items_in_batch)
                if type(result) is not torch.Tensor or result.ndim != 0:
                    raise RuntimeError("Native loss-only transaction returned unexpected output")
                if transaction["calls"] == 1:
                    telemetry.emit("owned_loss_return")
                return result
            finally:
                _ACTIVE.reset(token)
                hook_handle.remove()

        self.replacement = loss_only
        cls.compute_loss = loss_only

    def guard(self, trainer, model, inputs, return_outputs):
        o = self.o
        def reject(reason):return reason, None
        if (not self.active or type(trainer) is not o.rewind_class
                or return_outputs is not False or _ACTIVE.get() is not None):
            return reject("not_owned_loss_only_call")
        if (o.rewind_class.compute_loss is not self.replacement
                or o.sft_class.compute_loss is not o.sft_loss
                or o.hf_class.compute_loss is not o.hf_loss
                or o.peft_class.forward is not o.peft_forward
                or o.qwen_class.forward is not o.qwen_forward
                or o.tuner_class.forward is not o.tuner_forward
                or torch.nn.Linear.forward is not o.linear_forward
                or o.sft_loss.__globals__.get("entropy_from_logits") is not o.entropy_function
                or o.rewind_loss.__globals__.get("row_losses") is not o.rewind_rows):
            return reject("native_callable_changed")
        args, acc = getattr(trainer,"args",None), getattr(trainer,"accelerator",None)
        if (type(model) is not o.peft_class or getattr(trainer,"model",None) is not model
                or not model.training or not torch.is_grad_enabled()
                or getattr(args,"use_liger_kernel",True) or getattr(args,"loss_type",None)!="nll"
                or getattr(args,"padding_free",True) or getattr(trainer,"num_virtual_tokens",None)!=0
                or getattr(trainer,"aux_loss_enabled",True)
                or getattr(trainer,"compute_loss_func",None) is not None
                or getattr(trainer,"label_smoother",None) is not None
                or getattr(args,"past_index",0) != -1):
            return reject("trainer_or_model_contract")
        if (getattr(acc,"num_processes",None)!=1 or getattr(acc,"parallelism_config",None) is not None
                or not getattr(acc,"native_amp",False) or getattr(acc,"mixed_precision",None)!="fp16"
                or getattr(getattr(acc,"device",None),"type",None)!=o.device_type):
            return reject("accelerator_contract")
        if (not isinstance(inputs,dict) or set(inputs)-{"input_ids","attention_mask","labels"}
                or any(type(inputs.get(k)) is not torch.Tensor for k in ("input_ids","attention_mask","labels"))):
            return reject("input_keys_or_types")
        labels, ids, mask = inputs["labels"],inputs["input_ids"],inputs["attention_mask"]
        if (labels.ndim!=2 or labels.shape[0]!=o.batch_size or not 1<=labels.shape[1]<=512
                or labels.shape!=ids.shape or labels.shape!=mask.shape
                or labels.dtype!=torch.int64 or ids.dtype!=torch.int64
                or labels.device!=ids.device or labels.device!=mask.device
                or labels.device.type!=o.device_type):
            return reject("input_geometry")
        if raw_callable(model.get_base_model) is not o.get_base_model:
            return reject("foreign_get_base_model")
        owner=model.get_base_model(); tuner=getattr(model,"base_model",None)
        if (type(owner) is not o.qwen_class or type(tuner) is not o.tuner_class
                or getattr(tuner,"model",None) is not owner
                or raw_callable(owner.forward) is not o.qwen_forward
                or raw_callable(tuner.forward) is not o.tuner_forward
                or owner.loss_function is not o.native_loss
                or getattr(owner.config,"model_type",None)!="qwen2"
                or getattr(owner.config,"vocab_size",None)!=o.vocabulary
                or not owner.training):
            return reject("inner_model_or_loss_owner")
        head=getattr(owner,"lm_head",None)
        if (type(head) is not torch.nn.Linear or head.bias is not None or head.weight.requires_grad
                or head.weight.dtype!=torch.float32 or head.weight.device!=labels.device
                or head.weight is not owner.get_input_embeddings().weight
                or raw_callable(head.forward) is not o.linear_forward):
            return reject("head_contract")
        if (not no_global_hooks() or not all(no_hooks(m) for m in (model,tuner,owner,head))
                or any(getattr(m,"_compiled_call_impl",None) is not None for m in (model,tuner,owner,head))):
            return reject("foreign_module_hook")
        config=getattr(model,"active_peft_config",None)
        if (getattr(config,"is_prompt_learning",True) or str(getattr(config,"peft_type",None)).split('.')[-1]!="LORA"
                or getattr(config,"alora_invocation_tokens",None) is not None):
            return reject("peft_contract")
        reason=prepared_chain_reason(model,o)
        return (reason,None) if reason else (None,owner)

    def restore(self):
        if not self.active:return False
        if self.o.rewind_class.compute_loss is self.replacement:
            self.o.rewind_class.compute_loss=self.o.rewind_loss
            restored=True
        else:restored=False
        self.active=False
        return restored
