"""New production guard metadata tests. No CUDA is executed or emulated."""
import importlib.util,sys,weakref
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
sys.path.insert(0,'/workspace/shared/soup_head_shared_cast_v4/tests')
from test_scoped_head import setup
root=Path(__file__).resolve().parents[1]/'solution'
spec=importlib.util.spec_from_file_location('combined',root/'__init__.py',submodule_search_locations=[str(root)])
combined=importlib.util.module_from_spec(spec);sys.modules['combined']=combined;spec.loader.exec_module(combined)
from combined.install import Installation
from combined.telemetry import Telemetry
from combined.producer.adapter import MaterializedCELoss
from combined.common.consumer import ConsumerInstallation
from combined.common import bridge_state
from soup_cli.trainer.rewind_hf import RewindState,row_losses_from_token_losses,_CHUNKED_STEP_ATTR

class Stats:
 def record(self,*a,**kw):pass
 def observe(self,*a,**kw):pass

@pytest.fixture
def rig():
 t,m,x,o,sink=setup()
 delattr(t.args,'past_index')
 h=Installation.__new__(Installation);h.o=o;h.active=True;h.pending=None;h.extra_owners={};h.telemetry=Telemetry(False);h.rewind_state_class=RewindState
 h.provider=lambda *a:None
 owner=m.get_base_model();old=owner.loss_function
 wrapper=MaterializedCELoss(owner,old,Stats(),h.provider);owner.loss_function=wrapper
 h.producer=SimpleNamespace(models=weakref.WeakKeyDictionary({owner:SimpleNamespace(wrapper=wrapper)}),model_class=o.qwen_class,hook=o.qwen_class.__init__)
 h.consumer=ConsumerInstallation(rewind_class=o.rewind_class,sft_class=o.sft_class,hf_class=o.hf_class,
  entropy_function=o.entropy_function,rewind_rows=o.rewind_rows,rewind_token_rows=row_losses_from_token_losses,
  guard=h.guard,vocabulary=o.vocabulary,chunked_attr=_CHUNKED_STEP_ATTR,producer_owner=lambda model:model.get_base_model())
 try:yield h,t,m,x,owner,wrapper
 finally:
  h.consumer.restore();owner.loss_function=old


def test_native_absence_of_removed_past_index_is_supported(rig):
 h,t,m,x,*_=rig
 assert not hasattr(t.args,'past_index')
 assert h.reason(t,m,x,False) is None

@pytest.mark.parametrize('field,value,reason',[('use_liger_kernel',True,'liger_enabled'),('loss_type','dft','loss_type'),('padding_free',True,'padding_free')])
def test_individual_args_reasons(rig,field,value,reason):
 h,t,m,x,*_=rig;setattr(t.args,field,value);assert h.reason(t,m,x,False)==reason

@pytest.mark.parametrize('field,value,reason',[('num_virtual_tokens',1,'virtual_tokens'),('aux_loss_enabled',True,'aux_loss'),('compute_loss_func',lambda:None,'custom_loss'),('label_smoother',object(),'label_smoothing')])
def test_individual_trainer_reasons(rig,field,value,reason):
 h,t,m,x,*_=rig;setattr(t,field,value);assert h.reason(t,m,x,False)==reason

@pytest.mark.parametrize('which',['outer','tuner','qwen','head'])
def test_output_changing_hook_rejected_before_forward(rig,which):
 h,t,m,x,owner,_=rig;module={'outer':m,'tuner':m.base_model,'qwen':owner,'head':owner.lm_head}[which]
 hook=module.register_forward_hook(lambda _m,_a,out:out+1)
 try:assert h.reason(t,m,x,False)=='foreign_module_hook'
 finally:hook.remove()

@pytest.mark.parametrize('field,value',[('owner',lambda:None),('native_loss',lambda:None),('summary_sink_provider',lambda *a:None)])
def test_producer_identity_rejected(rig,field,value):
 h,t,m,x,owner,wrapper=rig;setattr(wrapper,field,value);assert h.reason(t,m,x,False)=='ce_producer_owner'


def test_no_rewind_or_failed_state_rejected(rig):
 h,t,m,x,*_=rig;t._soup_rewind_state.failed=True;assert h.reason(t,m,x,False)=='rewind_state_inactive'
 t._soup_rewind_state=None;assert h.reason(t,m,x,False)=='rewind_state_inactive'


def test_foreign_prepared_wrapper_rejected(rig):
 h,t,m,x,*_=rig;old=m.forward;m.forward=lambda **kw:old(**kw)
 assert h.reason(t,m,x,False)=='outer_conversion_code'


def test_output_request_rejected(rig):
 h,t,m,x,*_=rig;assert h.reason(t,m,x,True)=='output_request'


def test_nested_active_transaction_does_not_flush_or_raise(rig):
 h,t,m,x,*_=rig
 with bridge_state.scope(m,x['labels'],vocabulary=h.o.vocabulary) as tr:
  h.pending=tr
  assert h.guard(t,m,x,False)=='nested_transaction'
 h.flush();assert h.pending is None
 assert h.telemetry.counts['consumer_native_summary_fallback']==1


def test_closed_transaction_audit_consumption_and_no_tensor_retention(rig):
 h,t,m,x,*_=rig;labels=x['labels'];raw=torch.randn(*labels.shape,h.o.vocabulary,dtype=torch.float16,requires_grad=True)
 with bridge_state.scope(m,labels,vocabulary=h.o.vocabulary) as tr:
  h.pending=tr
  sink=bridge_state.request_sink(m,raw,labels)
  sink(token_nll=torch.ones_like(labels,dtype=torch.float32),token_entropy=torch.ones_like(labels,dtype=torch.float32),first_argmax=torch.zeros_like(labels),route='all_safe')
  loss=raw.float().sum();assert sink.complete(loss)
  public=raw.float();assert tr.bind_outputs(loss,SimpleNamespace(loss=loss,logits=public))
  for key in bridge_state._FIELDS:assert tr.take(key,public,labels) is not None
 assert tr.summary is None and tr.raw_ref is None and tr.loss_ref is None and tr.public_ref is None
 assert all(not isinstance(value,torch.Tensor) for value in vars(tr).values())
 h.flush();assert h.pending is None and h.telemetry.counts['consumer_complete']==1
 h.flush();assert h.telemetry.counts['consumer_complete']==1


def test_pinned_training_arguments_source_really_has_no_past_index():
 from transformers import TrainingArguments
 import dataclasses
 assert 'past_index' not in {field.name for field in dataclasses.fields(TrainingArguments)}
