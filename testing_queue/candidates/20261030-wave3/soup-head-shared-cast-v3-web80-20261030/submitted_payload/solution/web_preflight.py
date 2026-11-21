"""Target-only tiny actual PEFT/Accelerate/Rewind chain; not fixed-task timing."""
import argparse,importlib,json,os,traceback
from pathlib import Path


def run():
    if os.environ.get('TRITON_INTERPRET'):
        raise RuntimeError('Interpreter metadata is not target validation')
    import torch
    from types import SimpleNamespace
    from collections import defaultdict
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import LoraConfig,TaskType,get_peft_model
    from accelerate import Accelerator
    from soup_cli.trainer.rewind_hf import RewindState
    from .scoped_head import prepared_chain_reason,_ACTIVE
    package=importlib.import_module(__package__)
    if (torch.__version__!='2.11.0+cu128' or torch.cuda.device_count()!=1
            or torch.cuda.get_device_name(0)!='Tesla T4'
            or torch.cuda.get_device_capability(0)!=(7,5)):
        raise RuntimeError('Exact target Torch/one Tesla T4 required')
    if not package.apply():raise RuntimeError('Pinned production installer unavailable')
    handle=package._handle
    class Sink:
        def __init__(self):self.rows=[]
        def record_batch(self,**kw):self.rows.append(kw)
    def same(a,b):
        if a.shape!=b.shape or a.dtype!=b.dtype:raise RuntimeError('Shape/dtype mismatch')
        if a.is_floating_point():
            if not (torch.equal(a.isnan(),b.isnan()) and torch.equal(a.isposinf(),b.isposinf())
                    and torch.equal(a.isneginf(),b.isneginf())):raise RuntimeError('Nonfinite classes differ')
            finite=torch.isfinite(a)&torch.isfinite(b)
            if not torch.equal(a[finite],b[finite]):raise RuntimeError('Finite bits differ')
            if not torch.equal(a.signbit()[a==0],b.signbit()[b==0]):raise RuntimeError('Signed zero differs')
        elif not torch.equal(a,b):raise RuntimeError('Integer/RNG values differ')
    def one(shared,scale):
        torch.manual_seed(144)
        config=Qwen2Config(vocab_size=151936,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=512,tie_word_embeddings=True,attention_dropout=0.,use_cache=False)
        base=Qwen2ForCausalLM(config).to('cuda')
        model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,
            lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'))
        model.train();acc=Accelerator(mixed_precision='fp16')
        model=acc.prepare_model(model)
        acc.scaler=torch.amp.GradScaler('cuda',init_scale=scale,growth_interval=100)
        cls=handle.o.rewind_class;trainer=cls.__new__(cls)
        trainer.model=model;trainer.model_wrapped=model;trainer.accelerator=acc
        trainer.args=SimpleNamespace(use_liger_kernel=False,loss_type='nll',padding_free=False,
            past_index=-1,average_tokens_across_devices=False,n_gpu=1)
        trainer.num_virtual_tokens=0;trainer.aux_loss_enabled=False
        trainer.compute_loss_func=None;trainer.label_smoother=None;trainer.model_accepts_loss_kwargs=True
        trainer._metrics={'train':defaultdict(list),'eval':defaultdict(list)}
        trainer._total_train_tokens=0;trainer.state=SimpleNamespace(global_step=0)
        sink=Sink();state=RewindState(sink)
        for row in [4,8,15,16]:state.on_index(row)
        trainer._soup_rewind_state=state
        ids=torch.randint(0,151936,(4,8),device='cuda');labels=ids.clone();labels[:,0]=-100;labels[1,1:3]=-100
        inputs={'input_ids':ids,'labels':labels,'attention_mask':torch.ones_like(ids)}
        denominator=torch.tensor(26,device='cuda',dtype=torch.int64)
        reason=prepared_chain_reason(model,handle.o)
        if reason is not None:raise RuntimeError('Prepared chain guard: '+reason)
        if shared:
            reason,_=handle.guard(trainer,model,inputs,False)
            if reason is not None:raise RuntimeError('Production transaction guard: '+reason)
        names={n:id(p) for n,p in model.named_parameters()}
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        if not params or not all('lora_' in n for n in params):raise RuntimeError('Unexpected trainable set')
        opt=torch.optim.AdamW(params.values(),lr=.001)
        method=trainer.compute_loss if shared else lambda *a,**kw:handle.o.rewind_loss(trainer,*a,**kw)
        loss=method(model,inputs,num_items_in_batch=denominator)
        acc.backward(loss)
        scaled={n:p.grad.detach().clone() for n,p in params.items()}
        acc.unscale_gradients(opt)
        unscaled={n:p.grad.detach().clone() for n,p in params.items()}
        acc.scaler.step(opt);acc.scaler.update()
        if names!={n:id(p) for n,p in model.named_parameters()}:raise RuntimeError('Parameter identity changed')
        if base.lm_head.weight is not base.model.embed_tokens.weight:raise RuntimeError('Tie identity changed')
        if base.lm_head._forward_hooks or _ACTIVE.get() is not None:raise RuntimeError('Transaction leaked')
        if state.failed or state.dropped or len(sink.rows)!=1:raise RuntimeError('Native Rewind failed')
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite fixture loss')
        torch.cuda.synchronize()
        result={'loss':loss.detach().clone(),'scaled':scaled,'unscaled':unscaled,
            'parameters':{n:p.detach().clone() for n,p in params.items()},
            'metrics':dict(trainer._metrics['train']),'rewind':sink.rows,
            'scaler':acc.scaler.state_dict(),'rng':torch.cuda.get_rng_state().clone()}
        # Exit no-grad/output-returning calls must remain native. They use fresh
        # actual tensors and are checked separately only on the shared fixture.
        if shared:
            before=package.diagnostics()['counts'].get('shared_cast',0)
            out_loss,out=trainer.compute_loss(model,{'input_ids':ids,'labels':labels,'attention_mask':torch.ones_like(ids)},
                return_outputs=True,num_items_in_batch=denominator)
            if out.logits.dtype!=torch.float32 or package.diagnostics()['counts'].get('shared_cast',0)!=before:
                raise RuntimeError('Output-returning fallback changed')
        return result
    groups=[]
    try:
        for scale in [1.,128.,65536.]:
            a,b=one(False,scale),one(True,scale)
            same(a['loss'],b['loss']);same(a['rng'],b['rng'])
            for key in ['scaled','unscaled','parameters']:
                if a[key].keys()!=b[key].keys():raise RuntimeError('Trainable names differ')
                for name in a[key]:same(a[key][name],b[key][name])
            if a['metrics']!=b['metrics'] or a['rewind']!=b['rewind'] or a['scaler']!=b['scaler']:
                raise RuntimeError('Native metrics/Rewind/GradScaler differ')
            groups.append({'scale':scale,'native_loss_gradient_parameter_metrics_rewind_rng_exact':True})
        diag=package.diagnostics()
        if diag['counts'].get('shared_cast')!=3 or diag['counts'].get('owned_loss_return')!=3:
            raise RuntimeError('Production route inactive')
        return {'groups':groups,'actual_peft_accelerate_rewind_method_chain':True,
                'full_trainer_constructor':False,'nf4_decoder':False,'tiny_random_model_only':True,
                'fixed_task_training':False,'diagnostics':diag,'original_triton_kernels':[]}
    finally:package.restore()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    report={'status':'FAIL','gpu_execution_attempted':True,'gpu_executed':None,
            'score':None,'admission':'NOT_ELIGIBLE_ZERO_OWNED_KERNELS'}
    try:
        report['checks']=run();report.update(status='PASS_NATIVE_SHARED_CAST_PREFLIGHT_ONLY',gpu_executed=True)
    except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'output':args.out}),flush=True)
    raise SystemExit(report['status']!='PASS_NATIVE_SHARED_CAST_PREFLIGHT_ONLY')
if __name__=='__main__':main()
