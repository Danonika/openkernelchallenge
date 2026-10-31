"""Source-pinned CE producer + shared materialized/streaming metric consumer.

Only the materialized producer is installed here. No Qwen forward hook, output
rewrite, head cast, Trainer switch, task change, or full-logit removal is used.
"""
import inspect
import torch
from .chain import raw_callable, no_hooks, no_global_hooks, prepared_chain_reason
from .common import bridge_state as state
from .common.consumer import ConsumerInstallation, materialized_sink
from .producer.adapter import MaterializedCELoss, install_pinned_qwen2


class AuditedSink:
    def __init__(self,sink,telemetry):self.sink=sink;self.telemetry=telemetry
    def __call__(self,**fields):return self.sink(**fields)
    def complete(self,loss):
        completed=self.sink.complete(loss)
        if completed:self.telemetry.emit('summary_loss_completed')
        return completed


class Installation:
    def __init__(self,owners,rewind_token_rows,telemetry,producer_stats,rewind_state_class,chunked_attr,extra_owners=None):
        self.o=owners;self.telemetry=telemetry;self.active=True;self.pending=None
        self.extra_owners=extra_owners or {}
        self.rewind_state_class=rewind_state_class;self.producer=None;self.consumer=None
        def provider(owner,logits,labels):
            sink=materialized_sink(owner,logits,labels)
            if sink is None:return None
            self.pending=sink.transaction
            return AuditedSink(sink,telemetry)
        self.provider=provider
        try:
            self.producer=install_pinned_qwen2(producer_stats,summary_sink_provider=provider)
            self.consumer=ConsumerInstallation(rewind_class=owners.rewind_class,sft_class=owners.sft_class,
                hf_class=owners.hf_class,entropy_function=owners.entropy_function,
                rewind_rows=owners.rewind_rows,rewind_token_rows=rewind_token_rows,
                guard=self.guard,vocabulary=owners.vocabulary,chunked_attr=chunked_attr,
                producer_owner=lambda model:model.get_base_model())
        except BaseException:
            if self.producer is not None:self.producer.restore()
            self.active=False
            raise

    def flush(self):
        """One closed transaction, containing metadata/weak refs only, is audited.

        Active scope retains tensors only as the frozen common bridge normally
        does; close() clears them. This audit never computes a metric or kernel.
        """
        transaction=self.pending
        if transaction is None:return
        if not transaction.closed:raise RuntimeError('Previous summary transaction still active')
        self.pending=None
        if (transaction.invalid_reason is None and transaction.bound
                and transaction.consumed==set(state._FIELDS)):
            self.telemetry.emit('consumer_complete',source_kind=transaction.source_kind)
        else:
            self.telemetry.emit('consumer_native_summary_fallback',reason=transaction.invalid_reason or 'summary_not_published_or_consumed')

    def reason(self,trainer,model,inputs,return_outputs):
        o=self.o
        tests=[(not self.active,'installation_inactive'),
               (type(trainer) is not o.rewind_class,'trainer_class'),
               (return_outputs is not False,'output_request'),
               (state.current() is not None,'nested_transaction'),
               (type(model) is not o.peft_class,'peft_class'),
               (getattr(trainer,'model',None) is not model,'trainer_model_identity'),
               (not model.training,'model_not_training'),
               (not torch.is_grad_enabled(),'no_grad')]
        for failed,reason in tests:
            if failed:return reason
        e=self.extra_owners
        if e and (raw_callable(model._enable_peft_forward_hooks) is not e["peft_hooks"]
                or o.peft_forward.__globals__.get("get_alora_offsets_for_forward") is not e["alora_offsets"]
                or o.qwen_forward.__wrapped__.__globals__.get("CausalLMOutputWithPast") is not e["output_class"]):return "output_chain_helper_changed"
        c=self.consumer
        if (o.rewind_class.compute_loss is not c.outer or o.sft_class.compute_loss is not c.sft
                or o.hf_class.compute_loss is not o.hf_loss or o.peft_class.forward is not o.peft_forward
                or o.qwen_class.forward is not o.qwen_forward or o.tuner_class.forward is not o.tuner_forward
                or torch.nn.Linear.forward is not o.linear_forward
                or o.sft_loss.__globals__.get('entropy_from_logits') is not o.entropy_function):return 'native_callable_changed'
        args,acc=getattr(trainer,'args',None),getattr(trainer,'accelerator',None)
        # past_index was removed from pinned Transformers 5.16.1. Never invent
        # this field in fixtures or reject its native absence in production.
        checks=[(getattr(args,'use_liger_kernel',True),'liger_enabled'),
                (getattr(args,'loss_type',None)!='nll','loss_type'),
                (getattr(args,'padding_free',True),'padding_free'),
                (getattr(trainer,'num_virtual_tokens',None)!=0,'virtual_tokens'),
                (getattr(trainer,'aux_loss_enabled',True),'aux_loss'),
                (getattr(trainer,'compute_loss_func',None) is not None,'custom_loss'),
                (getattr(trainer,'label_smoother',None) is not None,'label_smoothing')]
        for failed,reason in checks:
            if failed:return reason
        rewind=getattr(trainer,'_soup_rewind_state',None)
        if type(rewind) is not self.rewind_state_class or rewind.failed:return 'rewind_state_inactive'
        if (getattr(acc,'num_processes',None)!=1 or getattr(acc,'parallelism_config',None) is not None
                or not getattr(acc,'native_amp',False) or getattr(acc,'mixed_precision',None)!='fp16'
                or getattr(getattr(acc,'device',None),'type',None)!=o.device_type):return 'accelerator_contract'
        if (not isinstance(inputs,dict) or set(inputs)-{'input_ids','attention_mask','labels'}
                or any(type(inputs.get(k)) is not torch.Tensor for k in ('input_ids','attention_mask','labels'))):return 'input_keys_or_types'
        labels,ids,mask=inputs['labels'],inputs['input_ids'],inputs['attention_mask']
        if (labels.ndim!=2 or labels.shape[0]!=o.batch_size or not 1<=labels.shape[1]<=512
                or labels.shape!=ids.shape or labels.shape!=mask.shape
                or labels.dtype!=torch.int64 or ids.dtype!=torch.int64
                or labels.device!=ids.device or labels.device!=mask.device
                or labels.device.type!=o.device_type or not labels.is_contiguous()):return 'input_geometry'
        if raw_callable(model.get_base_model) is not o.get_base_model:return 'foreign_get_base_model'
        owner=model.get_base_model();tuner=getattr(model,'base_model',None)
        if e and raw_callable(getattr(tuner,'_enable_peft_forward_hooks',None)) is not e['tuner_hooks']:return 'tuner_hook_helper_changed'
        if (type(owner) is not o.qwen_class or type(tuner) is not o.tuner_class
                or getattr(tuner,'model',None) is not owner
                or raw_callable(owner.forward) is not o.qwen_forward
                or raw_callable(tuner.forward) is not o.tuner_forward
                or getattr(owner.config,'model_type',None)!='qwen2'
                or getattr(owner.config,'vocab_size',None)!=o.vocabulary
                or getattr(owner.config,'return_dict',False) is not True or not owner.training):return 'inner_model_owner'
        installed=self.producer.models.get(owner);wrapper=owner.loss_function
        if (installed is None or type(wrapper) is not MaterializedCELoss or installed.wrapper is not wrapper
                or wrapper.owner() is not owner or wrapper.native_loss is not o.native_loss
                or wrapper.summary_sink_provider is not self.provider
                or self.producer.model_class.__init__ is not self.producer.hook):return 'ce_producer_owner'
        head=getattr(owner,'lm_head',None)
        if (type(head) is not torch.nn.Linear or head.bias is not None or head.weight.requires_grad
                or head.weight.dtype!=torch.float32 or head.weight.device!=labels.device
                or head.weight is not owner.get_input_embeddings().weight
                or raw_callable(head.forward) is not o.linear_forward):return 'head_contract'
        if (not no_global_hooks() or not all(no_hooks(m) for m in (model,tuner,owner,head))
                or any(getattr(m,'_compiled_call_impl',None) is not None for m in (model,tuner,owner,head))):return 'foreign_module_hook'
        config=getattr(model,'active_peft_config',None)
        if (getattr(config,'is_prompt_learning',True) or str(getattr(config,'peft_type',None)).split('.')[-1]!='LORA'
                or getattr(config,'alora_invocation_tokens',None) is not None):return 'peft_contract'
        return prepared_chain_reason(model,o)

    def guard(self,trainer,model,inputs,return_outputs):
        if state.current() is None:self.flush()
        reason=self.reason(trainer,model,inputs,return_outputs)
        if reason is not None:self.telemetry.emit('native_fallback',reason=reason)
        else:self.telemetry.emit('owned_consumer_guard')
        return reason

    def restore(self):
        self.flush()
        consumer=True if self.consumer is None else self.consumer.restore()
        producer={} if self.producer is None else self.producer.restore()
        self.active=False
        return {'consumer':consumer,'producer':producer}


def install_pinned(telemetry,producer_stats):
    from .source_owners import load_pinned_owners
    from soup_cli.trainer.rewind_hf import RewindState,_CHUNKED_STEP_ATTR
    owners,token_rows,extras=load_pinned_owners()
    return Installation(owners,token_rows,telemetry,producer_stats,RewindState,_CHUNKED_STEP_ATTR,extras)
