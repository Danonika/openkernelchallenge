"""Bounded consumer component; no production installer or automatic patching.

The caller must establish native source ownership and supply a metadata guard.
Only an owned outer Rewind loss-only transaction may use detached summaries.
Native HF still executes the model exactly once. Unsupported pre-forward calls
remain fully native; post-forward materialized misses use the SAME logits.
"""
from functools import wraps
from types import SimpleNamespace
import weakref
import torch
from . import bridge_state as state


def materialized_sink(inner_owner,raw_logits,labels):
    transaction=state.current()
    if transaction is None or getattr(transaction,'producer_owner_ref',lambda:None)() is not inner_owner:
        return None
    outer=transaction.owner_ref()
    return state.request_sink(outer,raw_logits,labels) if outer is not None else None


def streaming_sink(inner_owner,labels,*,vocabulary):
    transaction=state.current()
    if transaction is None or getattr(transaction,'producer_owner_ref',lambda:None)() is not inner_owner:
        return None
    outer=transaction.owner_ref()
    return state.request_streaming_sink(outer,labels,vocabulary=vocabulary) if outer is not None else None


def native_order_row_losses(token_nll,labels):
    """Match native row_losses' per-example T-1 FP32 mask/sum/clamp order."""
    losses=[];counts=[]
    with torch.no_grad():
        targets=labels[:,1:]
        for i in range(targets.size(0)):
            token_ce=token_nll[i,:-1].float()
            mask=(targets[i]!=-100).to(token_ce.dtype)
            tokens=mask.sum();total=(token_ce*mask).sum()
            losses.append(float(total/tokens.clamp(min=1.0)))
            counts.append(int(tokens.item()))
    return losses,counts


def publish_metrics(trainer,outputs,labels,attention_mask,entropy_function,transaction):
    logits=outputs.logits
    entropy=transaction.take('token_entropy',logits,labels)
    predictions=transaction.take('first_argmax',logits,labels)
    if entropy is None or predictions is None:
        if logits is None:raise RuntimeError('Streaming summary absent or invalid before metrics')
        with torch.no_grad():
            entropy=entropy_function(logits)
            predictions=logits[..., :-1,:].contiguous().argmax(dim=-1)
    else:predictions=predictions[...,:-1]
    with torch.no_grad():
        value=(entropy*attention_mask).sum()/attention_mask.sum()
        value=trainer.accelerator.gather_for_metrics(value).mean().item()
    trainer._metrics['train']['entropy'].append(value)
    tokens=trainer.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
    trainer._total_train_tokens+=tokens
    trainer._metrics['train']['num_tokens']=[trainer._total_train_tokens]
    with torch.no_grad():
        targets=labels[...,1:].contiguous();mask=targets!=-100
        correct=((predictions==targets)&mask).sum();total=mask.sum()
        correct=trainer.accelerator.gather_for_metrics(correct)
        total=trainer.accelerator.gather_for_metrics(total)
        total_sum=total.sum()
        accuracy=(correct.sum()/total_sum).item() if total_sum>0 else 0.0
    trainer._metrics['train']['mean_token_accuracy'].append(accuracy)


class ConsumerInstallation:
    """Explicit fixture/integration API. NOT a source-verifying installer.

    guard(trainer,model,inputs,return_outputs) returns None only when every
    production ownership/configuration requirement has been established.
    """
    def __init__(self,*,rewind_class,sft_class,hf_class,entropy_function,
                 rewind_rows,rewind_token_rows,guard,vocabulary,chunked_attr,producer_owner=lambda model:model):
        self.rewind_class=rewind_class;self.sft_class=sft_class;self.hf_class=hf_class
        self.original_rewind=rewind_class.compute_loss
        self.original_sft=sft_class.compute_loss;self.original_hf=hf_class.compute_loss
        self.original_rows=rewind_rows;self.original_token_rows=rewind_token_rows
        self.globals=self.original_rewind.__globals__;self.chunked_attr=chunked_attr
        if (self.globals.get('row_losses') is not rewind_rows
                or self.globals.get('row_losses_from_token_losses') is not rewind_token_rows):
            raise ValueError('Native Rewind helper owners required')
        original_rewind=self.original_rewind;original_sft=self.original_sft
        original_hf=self.original_hf;sentinel=object()

        @wraps(original_rewind)
        def outer(trainer,model,inputs,return_outputs=False,num_items_in_batch=None):
            intact=(rewind_class.compute_loss is outer and sft_class.compute_loss is sft
                and hf_class.compute_loss is original_hf and self.globals.get('row_losses') is rows
                and self.globals.get('row_losses_from_token_losses') is token_rows)
            reason=guard(trainer,model,inputs,return_outputs) if intact else 'consumer_owner_changed'
            if reason is not None or return_outputs or state.current() is not None:
                with state.suspended():
                    return original_rewind(trainer,model,inputs,return_outputs=return_outputs,
                                           num_items_in_batch=num_items_in_batch)
            previous=getattr(trainer,chunked_attr,sentinel);owned_packet=None
            with state.scope(model,inputs['labels'],vocabulary=vocabulary) as transaction:
                transaction.producer_owner_ref=weakref.ref(producer_owner(model))
                try:
                    result=original_rewind(trainer,model,inputs,return_outputs=False,
                                           num_items_in_batch=num_items_in_batch)
                    if transaction.source_kind=='streaming' and (
                            transaction.invalid_reason is not None
                            or transaction.consumed!=set(state._FIELDS)):
                        raise RuntimeError('Streaming summaries were not completely consumed')
                    return result
                finally:
                    owned_packet=getattr(transaction,'chunked_packet',None)
                    if owned_packet is not None and getattr(trainer,chunked_attr,sentinel) is owned_packet:
                        if previous is sentinel:delattr(trainer,chunked_attr)
                        else:setattr(trainer,chunked_attr,previous)

        @wraps(original_sft)
        def sft(trainer,model,inputs,return_outputs=False,num_items_in_batch=None):
            transaction=state.current()
            if (transaction is None or transaction.owner_ref() is not model
                    or transaction.labels_ref() is not inputs.get('labels')):
                return original_sft(trainer,model,inputs,return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
            parent=getattr(super(sft_class,trainer),'compute_loss',None)
            if getattr(parent,'__func__',None) is not original_hf:
                raise RuntimeError('Native HF parent changed inside owned transaction')
            labels=inputs['labels'];mask=inputs['attention_mask']
            inputs.pop('_prediction_loss_only',None);inputs['use_cache']=False
            loss,outputs=parent(model,inputs,return_outputs=True,num_items_in_batch=num_items_in_batch)
            bound=transaction.bind_outputs(loss,outputs)
            if outputs.logits is None:
                if not bound or transaction.source_kind!='streaming':
                    raise RuntimeError('Missing full logits without validated streaming summaries')
                transaction.chunked_packet=SimpleNamespace(token_loss=transaction.summary.token_nll)
                setattr(trainer,chunked_attr,transaction.chunked_packet)
            publish_metrics(trainer,outputs,labels,mask,entropy_function,transaction)
            return (loss,outputs) if return_outputs else loss

        def rows(logits,labels):
            transaction=state.current()
            token_nll=transaction.take('token_nll',logits,labels) if transaction else None
            return native_order_row_losses(token_nll,labels) if token_nll is not None else rewind_rows(logits,labels)

        def token_rows(token_loss,labels):
            transaction=state.current()
            if transaction is None:return rewind_token_rows(token_loss,labels)
            if (transaction.source_kind!='streaming' or transaction.summary is None
                    or token_loss is not transaction.summary.token_nll):
                raise RuntimeError('Foreign streaming token-loss packet')
            token_nll=transaction.take('token_nll',None,labels)
            if token_nll is None:raise RuntimeError('Invalid streaming Rewind summary')
            return native_order_row_losses(token_nll,labels)

        self.outer=outer;self.sft=sft;self.rows=rows;self.token_rows=token_rows
        rewind_class.compute_loss=outer;sft_class.compute_loss=sft
        self.globals['row_losses']=rows;self.globals['row_losses_from_token_losses']=token_rows

    def restore(self):
        result=True
        for owner,key,ours,old in [(self.rewind_class,'compute_loss',self.outer,self.original_rewind),
                                   (self.sft_class,'compute_loss',self.sft,self.original_sft)]:
            if getattr(owner,key) is ours:setattr(owner,key,old)
            else:result=False
        for key,ours,old in [('row_losses',self.rows,self.original_rows),
                             ('row_losses_from_token_losses',self.token_rows,self.original_token_rows)]:
            if self.globals.get(key) is ours:self.globals[key]=old
            else:result=False
        return result
