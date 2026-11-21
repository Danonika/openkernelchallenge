"""Owned production Qwen boundary for the separately target-tested producer.

Install only with verified native owners and a separately reviewed producer.
The decoder runs once. Unsupported hidden metadata uses the native head/loss
from those SAME decoder outputs. Once a streaming sink is requested there is no
materialized retry. Full-output/unowned/unsupported-argument calls are native.
"""
from functools import wraps
import torch
from .frozen_storage import attest_weight,attest_head
from .common import bridge_state as state
from .common.consumer import ConsumerInstallation,streaming_sink
from .ownership import OwnershipGuard
from .chain import raw_callable,no_hooks,no_global_hooks


def metadata_reason(hidden,weight,labels,denominator,*,vocabulary,hidden_size,device_type):
    if (type(hidden) is not torch.Tensor or type(weight) not in (torch.Tensor,torch.nn.Parameter)
            or type(labels) is not torch.Tensor):return 'hidden_types'
    if (hidden.ndim!=3 or hidden.shape[:2]!=labels.shape or hidden.shape[-1]!=hidden_size
            or weight.shape!=(vocabulary,hidden_size)):return 'hidden_geometry'
    if (hidden.dtype!=torch.float32 or not (weight.dtype==torch.float32 or attest_weight(weight)) or labels.dtype!=torch.int64
            or not hidden.requires_grad or weight.requires_grad):return 'hidden_dtype_gradient'
    if (hidden.device!=weight.device or hidden.device!=labels.device
            or hidden.device.type!=device_type):return 'hidden_device'
    if (not all(x.is_contiguous() for x in (hidden,weight,labels))
            or any(x.is_neg() or x.is_conj() for x in (hidden,weight,labels))):return 'hidden_layout'
    if (denominator is not None and not (type(denominator) is int and denominator>=0)
            and not (type(denominator) is torch.Tensor and denominator.device==hidden.device
                     and denominator.dtype==torch.int64 and denominator.ndim==0 and not denominator.requires_grad)):
        return 'denominator_contract'
    if not torch.is_autocast_enabled(device_type) or torch.get_autocast_dtype(device_type)!=torch.float16:
        return 'native_amp_contract'
    return None


class StreamingInstallation:
    """Source-owned production boundary, also dependency-injectable for CPU proof."""
    def __init__(self,*,owners,rewind_token_rows,rewind_state_class,chunked_attr,
                 extra_owners,producer,output_class,hidden_size=1536,chunk=128,observer=lambda *a,**kw:None):
        self.o=owners;self.producer=producer;self.output_class=output_class
        self.hidden_size=hidden_size;self.chunk=chunk;self.observe=observer;self.active=True;self.pending=None
        self.native_forward=owners.qwen_forward;self.consumer=None;self.owner_guard=None
        if owners.qwen_class.forward is not self.native_forward:raise RuntimeError('Foreign Qwen forward before installation')
        native_forward=self.native_forward

        @wraps(native_forward)
        def forward(model,input_ids=None,attention_mask=None,position_ids=None,
                    past_key_values=None,inputs_embeds=None,labels=None,use_cache=None,
                    logits_to_keep=0,**kwargs):
            transaction=state.current()
            owned=(self.active and transaction is not None
                and getattr(transaction,'producer_owner_ref',lambda:None)() is model
                and transaction.labels_ref() is labels)
            allowed={'num_items_in_batch','output_attentions','output_hidden_states','return_dict'}
            controls_ok=all(value is None or type(value) is bool
                for key,value in kwargs.items() if key in allowed-{'num_items_in_batch'})
            supported=(owned and model.training and torch.is_grad_enabled()
                and inputs_embeds is None and position_ids is None and past_key_values is None
                and use_cache is False and type(logits_to_keep) is int and logits_to_keep==0
                and not (set(kwargs)-allowed) and controls_ok
                and kwargs.get('return_dict',True) is not False
                and model.config.return_dict is True)
            if not supported:
                self.observe('native_forward',reason='unowned_or_arguments')
                return native_forward(model,input_ids=input_ids,attention_mask=attention_mask,
                    position_ids=position_ids,past_key_values=past_key_values,inputs_embeds=inputs_embeds,
                    labels=labels,use_cache=use_cache,logits_to_keep=logits_to_keep,**kwargs)
            # can_return_tuple consumes return_dict before native Qwen body.
            # All other kwargs go to the unchanged native decoder AND loss.
            native_kwargs=dict(kwargs);native_kwargs.pop('return_dict',None)
            expected_head=model.lm_head;expected_weight=expected_head.weight
            outputs=model.model(input_ids=input_ids,attention_mask=attention_mask,
                position_ids=position_ids,past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,use_cache=use_cache,**native_kwargs)
            self.observe('backbone_once')
            hidden=outputs.last_hidden_state;head=model.lm_head
            denominator=native_kwargs.get('num_items_in_batch')
            if (head is not expected_head or type(head) is not torch.nn.Linear
                    or head.weight is not expected_weight or head.bias is not None
                    or not (raw_callable(head.forward) is owners.linear_forward
                            or attest_head(model,head,head.weight,owners.linear_forward))
                    or not no_hooks(head) or not no_global_hooks()):
                reason='head_changed_during_backbone'
            else:
                reason=metadata_reason(hidden,head.weight,labels,denominator,
                    vocabulary=owners.vocabulary,hidden_size=hidden_size,device_type=owners.device_type)
            if reason is not None:
                self.observe('same_backbone_native_tail',reason=reason)
                logits=head(hidden[:,slice(0,None),:])
                loss=model.loss_function(logits=logits,labels=labels,vocab_size=model.config.vocab_size,**native_kwargs)
            else:
                # Request only after all supported metadata is known. Any later
                # producer failure is fatal; never repeat dropout or the head.
                sink=streaming_sink(model,labels,vocabulary=owners.vocabulary)
                if sink is None:raise RuntimeError('Owned streaming sink unavailable after decoder')
                self.pending=transaction
                loss,_summaries=producer(hidden,head.weight,labels,chunk=chunk,
                    denominator=denominator,summary_sink=sink)
                if (type(loss) is not torch.Tensor or loss.dtype!=torch.float32 or loss.ndim!=0
                        or not loss.requires_grad or transaction.loss_ref is None
                        or transaction.loss_ref() is not loss or transaction.invalid_reason is not None
                        or not transaction.completed):
                    raise RuntimeError('Streaming producer did not complete its returned FP32 loss')
                logits=None;self.observe('summary_loss_completed');self.observe('streaming_loss_return')
            return output_class(loss=loss,logits=logits,past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,attentions=outputs.attentions)
        self.forward=forward
        owners.qwen_class.forward=forward
        try:
            self.owner_guard=OwnershipGuard(owners,rewind_state_class,extra_owners,forward,
                lambda owner:None if owner.loss_function is owners.native_loss else 'foreign_native_loss')
            self.consumer=ConsumerInstallation(rewind_class=owners.rewind_class,sft_class=owners.sft_class,
                hf_class=owners.hf_class,entropy_function=owners.entropy_function,
                rewind_rows=owners.rewind_rows,rewind_token_rows=rewind_token_rows,
                guard=self.guard,vocabulary=owners.vocabulary,chunked_attr=chunked_attr,
                producer_owner=lambda model:model.get_base_model())
            self.owner_guard.consumer=self.consumer
        except BaseException:
            if owners.qwen_class.forward is forward:owners.qwen_class.forward=native_forward
            self.active=False
            raise

    def flush(self):
        transaction=self.pending
        if transaction is None:return
        if not transaction.closed:raise RuntimeError("Previous streaming transaction still active")
        self.pending=None
        if transaction.invalid_reason is None and transaction.bound and transaction.consumed==set(state._FIELDS):
            self.observe("consumer_complete",source_kind="streaming")
        else:self.observe("consumer_incomplete",reason=transaction.invalid_reason or "not_completely_consumed")

    def guard(self,trainer,model,inputs,return_outputs):
        if state.current() is None:self.flush()
        reason=self.owner_guard.reason(trainer,model,inputs,return_outputs)
        self.observe('native_consumer_fallback' if reason is not None else 'owned_consumer_guard',reason=reason)
        return reason

    def restore(self):
        if not self.active:return False
        self.flush()
        consumer=self.consumer.restore()
        if self.o.qwen_class.forward is self.forward:
            self.o.qwen_class.forward=self.native_forward;forward=True
        else:forward=False
        self.active=False;self.owner_guard.active=False
        return consumer and forward


def install_pinned(producer,*,chunk=128,observer=lambda *a,**kw:None):
    """Verify the pinned native chain, then install only the owned loss-only boundary."""
    from .source_owners import load_pinned_owners
    from soup_cli.trainer.rewind_hf import RewindState,_CHUNKED_STEP_ATTR
    owners,token_rows,extras=load_pinned_owners()
    return StreamingInstallation(owners=owners,rewind_token_rows=token_rows,rewind_state_class=RewindState,
        chunked_attr=_CHUNKED_STEP_ATTR,extra_owners=extras,producer=producer,
        output_class=extras['output_class'],hidden_size=1536,chunk=chunk,observer=observer)
