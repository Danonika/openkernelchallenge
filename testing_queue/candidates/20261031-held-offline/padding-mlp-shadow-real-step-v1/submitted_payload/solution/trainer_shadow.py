"""One-shot native compute_loss seam: observe forward, then native-return shadow.

The captured native Rewind compute_loss runs once with its original arguments.
Its complete forward, entropy/accuracy and row-loss work finishes BEFORE any
MatMul4Bit backward ownership change. No optimizer or train-loop is invoked.
"""
import hashlib
import inspect
import types
import torch
from .row_adjoint import OriginalBatchMask
from .backward_phase import FrozenMLPBackwardPhase

REWIND_COMPUTE_LOSS_SHA='e05038c89369901e08f5fd46c9fa3dd312e2a0afc8c4eb3d54a3c79eb1fa716a'


class OneStepNativeShadow:
    def __init__(self,trainer,model,installation):
        original=trainer.compute_loss
        if ('compute_loss' in trainer.__dict__ or not inspect.ismethod(original)
                or original.__self__ is not trainer or original.__func__ is not type(trainer).compute_loss
                or original.__func__.__module__!='soup_cli.trainer.rewind_hf'
                or hashlib.sha256(inspect.getsource(original.__func__).encode()).hexdigest()!=REWIND_COMPUTE_LOSS_SHA):
            raise RuntimeError('Exact unwrapped native Rewind compute_loss required')
        self.trainer=trainer;self.model=model;self.installation=installation
        self.original=original;self.function=original.__func__;self.function_code=self.function.__code__
        self.calls=0;self.session=None;self.phase=None;self.plan=None;self.active=False;self.used=False
        self.cpu_rng=None;self.cuda_rng=None
        self.report={'scope':'NATIVE_RETURN_SHADOW_DIAGNOSTIC_ONLY','compute_loss_calls':0,
            'forward_reference':'unchanged accepted dense starter-half, not original native SiLU',
            'native_backward_result_returned':True,'optimizer_steps':0,'score':None,
            'range_certificate':False,'full_model_production_clearance':False}
        owner=self
        def compute_loss(actual,model,inputs,return_outputs=False,num_items_in_batch=None):
            if (actual is not owner.trainer or model is not owner.model or not owner.active
                    or owner.trainer.__dict__.get('compute_loss') is not owner.installed
                    or compute_loss.__code__ is not owner.installed_code or not owner._native_intact()):
                raise RuntimeError('One-step compute_loss seam ownership changed')
            if owner.calls:raise RuntimeError('Exactly one compute_loss call permitted')
            if type(inputs) is not dict or 'attention_mask' not in inputs:
                raise RuntimeError('Actual prepared batch attention mask missing')
            owner.calls+=1;owner.report['compute_loss_calls']=owner.calls
            mask=inputs['attention_mask'];owner.plan=OriginalBatchMask.prepare(mask)
            owner.report['original_batch']={'shape':list(mask.shape),'live_rows':owner.plan.live.numel(),
                'padded_rows':owner.plan.padded.numel(),'device':str(mask.device),'identity_bound':True}
            session=owner.installation.begin_session(owner.plan);owner.session=session
            try:
                result=owner.original(model,inputs,return_outputs=return_outputs,num_items_in_batch=num_items_in_batch)
                if inputs.get('attention_mask') is not mask or not owner.plan.unchanged():
                    raise RuntimeError('Original prepared attention mask changed during native forward')
                observations=session.observations
            finally:
                closed=owner.installation.end_session(session)
                owner.report['forward_session']=session.status()
                owner.report['forward_session_closed']=closed
                if not closed:raise RuntimeError('Forward observation session failed to close')
            owner.report['observed_mlp_calls']=len(observations)
            owner.report['bound_contexts']=sum(len(o.bindings) for o in observations)
            if owner.report['bound_contexts']:
                # No model forward or loss/metric operation occurs after this
                # point in the source-pinned native Trainer.training_step.
                phase=FrozenMLPBackwardPhase.from_observations(observations,shadow_only=True)
                phase.__enter__()
                owner.phase=phase
            owner.cpu_rng=torch.get_rng_state().clone()
            owner.cuda_rng=torch.cuda.get_rng_state().clone() if mask.device.type=='cuda' else None
            owner.report['phase_entered_after_complete_native_compute_loss']=owner.phase is not None
            return result
        self.installed=types.MethodType(compute_loss,trainer);self.installed_code=compute_loss.__code__

    def _native_intact(self):
        return (type(self.trainer).compute_loss is self.function and self.function.__code__ is self.function_code)

    def __enter__(self):
        if self.used or self.active or 'compute_loss' in self.trainer.__dict__ or not self._native_intact():
            raise RuntimeError('Native compute_loss seam must enter once')
        self.used=True;self.active=True;self.trainer.compute_loss=self.installed
        return self

    def __exit__(self,*error):
        failure=None
        try:
            if self.phase is not None:
                self.phase.__exit__(*error)
                self.report['events']=self.phase.events
                self.report['route_reason_counts']=[{'route':route,'reason':reason,'count':count}
                    for (route,reason),count in sorted(self.phase.counts.items())]
                self.report['backward_phase_restored']=not self.phase.active and self.phase.native.backward is self.phase.original
            else:
                self.report.update(events=[],route_reason_counts=[],backward_phase_restored=True)
            self.report['backward_cpu_rng_unchanged']=None if self.cpu_rng is None else torch.equal(self.cpu_rng,torch.get_rng_state())
            self.report['backward_cuda_rng_unchanged']=None if self.cuda_rng is None else torch.equal(self.cuda_rng,torch.cuda.get_rng_state())
            self.report['original_batch_unchanged']=self.plan is None or self.plan.unchanged()
        except BaseException as exc:failure=exc
        finally:
            owned=self.trainer.__dict__.get('compute_loss') is self.installed
            if owned:del self.trainer.__dict__['compute_loss']
            self.active=False
            self.report['compute_loss_restored']=owned and self._native_intact()
        if not self.report['compute_loss_restored']:raise RuntimeError('Foreign compute_loss mutation not overwritten')
        if failure is not None:raise failure
        return False
