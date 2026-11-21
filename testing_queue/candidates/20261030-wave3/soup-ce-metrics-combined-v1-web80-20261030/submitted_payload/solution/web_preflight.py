"""Bounded target proof using REAL SFTConfig and full native Trainer constructor.

Synthetic tiny decoder, actual V151936 and native PEFT/AMP/GradScaler. Compares
producer-only/native metric consumers with the SAME CE producer plus summaries.
Not a task benchmark, NF4 decoder proof, or eligibility declaration.
"""
import argparse,copy,gc,importlib,json,os,tempfile,traceback
from pathlib import Path


def same_tensor(a,b,path):
    import torch
    if a.shape!=b.shape or a.dtype!=b.dtype:raise RuntimeError(path+': shape/dtype')
    if not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
        raise RuntimeError(path+': bitwise value or nonfinite/sign mismatch')


def close_metric(a,b,path,rtol):
    import torch
    a=torch.as_tensor(a,dtype=torch.float64);b=torch.as_tensor(b,dtype=torch.float64)
    if not (torch.equal(a.isnan(),b.isnan()) and torch.equal(a.isposinf(),b.isposinf())
            and torch.equal(a.isneginf(),b.isneginf())):raise RuntimeError(path+': nonfinite classes')
    f=a.isfinite()&b.isfinite()
    torch.testing.assert_close(a[f],b[f],atol=1e-6 if rtol else 0,rtol=rtol,msg=path)


def run(progress):
    if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Interpreter is not target evidence')
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
    from trl import SFTConfig
    from peft import LoraConfig,TaskType,get_peft_model
    from soup_cli.trainer.rewind_hf import RewindState
    from .common import bridge_state
    from .producer.verify_metric_producer import run as producer_preflight
    package=importlib.import_module(__package__)
    if (torch.__version__!='2.11.0+cu128' or torch.cuda.device_count()!=1
            or torch.cuda.get_device_name(0)!='Tesla T4'
            or torch.cuda.get_device_capability(0)!=(7,5)):
        raise RuntimeError('Exact target Torch/one Tesla T4 required')
    progress.update(stage="producer_preflight",completed_groups=[])
    producer_checks=producer_preflight()
    progress["producer"]=producer_checks
    progress["stage"]="production_installer"
    package.apply();handle=package._handle
    class Sink:
        def __init__(self):self.rows=[]
        def record_batch(self,**kw):self.rows.append(kw)

    def one(combined,scale):
        progress.update(stage="full_trainer_constructor",current_scale=scale,current_mode="combined" if combined else "producer_only")
        torch.manual_seed(144)
        config=Qwen2Config(vocab_size=151936,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=512,tie_word_embeddings=True,attention_dropout=0.,
            use_cache=False,eos_token_id=1,pad_token_id=0)
        base=Qwen2ForCausalLM(config).to('cuda')
        model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,
            lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'))
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {'<pad>':0,'<eos>':1,'<unk>':2},unk_token='<unk>')),
            pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
        dataset=Dataset.from_dict({'input_ids':[[2,2,2,1]]*4,'attention_mask':[[1,1,1,1]]*4})
        with tempfile.TemporaryDirectory(prefix='ce_metric_preflight_') as directory:
            args=SFTConfig(output_dir=directory,fp16=True,bf16=False,
                per_device_train_batch_size=4,gradient_accumulation_steps=1,
                learning_rate=.0002,seed=1,report_to=[],gradient_checkpointing=False,
                max_length=512,packing=False,padding_free=False,use_liger_kernel=False,
                loss_type='nll',activation_offloading=False,dataloader_num_workers=0,
                dataset_kwargs={'skip_prepare_dataset':True})
            if hasattr(args,'past_index'):raise RuntimeError('Unexpected removed past_index field')
            trainer=handle.o.rewind_class(model=model,args=args,train_dataset=dataset,processing_class=tokenizer)
            acc=trainer.accelerator;model=acc.prepare_model(trainer.model)
            trainer.model=model;trainer.model_wrapped=model;model.train()
            acc.scaler=torch.amp.GradScaler('cuda',init_scale=scale,growth_interval=100)
            sink=Sink();rewind=RewindState(sink)
            for row in [4,8,15,16]:rewind.on_index(row)
            trainer._soup_rewind_state=rewind
            ids=torch.randint(0,151936,(4,8),device='cuda');labels=ids.clone();labels[:,0]=-100;labels[1,1:3]=-100
            mask=torch.ones_like(ids);mask[0,-1]=0
            inputs={'input_ids':ids,'labels':labels,'attention_mask':mask}
            denominator=torch.tensor(26,device='cuda',dtype=torch.int64)
            reason=handle.reason(trainer,model,inputs,False)
            progress.update(stage='actual_guard',actual_guard_reason=reason,actual_args_type=type(args).__module__+'.'+type(args).__qualname__,past_index_absent=not hasattr(args,'past_index'))
            if reason is not None:raise RuntimeError('Actual full Trainer guard: '+reason)
            # Negative output-mutating hooks are rejected before any forward;
            # the actual PEFT tuner bypasses inner Qwen module hooks.
            negative=[]
            for name,module in [('peft',model),('tuner',model.base_model),('qwen',base),('head',base.lm_head)]:
                hook=module.register_forward_hook(lambda m,a,out:out+1)
                try:
                    reason=handle.reason(trainer,model,inputs,False)
                    if reason!='foreign_module_hook':raise RuntimeError('Hook negative failed: '+name+': '+str(reason))
                    negative.append(name)
                finally:hook.remove()
            original_gather=acc.gather_for_metrics;gathers=[]
            def gather(value,*a,**kw):
                gathers.append(value.detach().cpu().clone())
                return original_gather(value,*a,**kw)
            acc.gather_for_metrics=gather
            names={n:id(p) for n,p in model.named_parameters()}
            params={n:p for n,p in model.named_parameters() if p.requires_grad}
            if not params or not all('lora_' in n for n in params):raise RuntimeError('Unexpected trainable set')
            optimizer=torch.optim.AdamW(params.values(),lr=.001)
            before=package.diagnostics()['consumer']['counts'].copy()
            method=trainer.compute_loss if combined else lambda *a,**kw:handle.o.rewind_loss(trainer,*a,**kw)
            progress["stage"]="forward_backward_optimizer"
            loss=method(model,inputs,num_items_in_batch=denominator)
            acc.backward(loss)
            scaled={n:p.grad.detach().cpu().clone() for n,p in params.items()}
            acc.unscale_gradients(optimizer)
            unscaled={n:p.grad.detach().cpu().clone() for n,p in params.items()}
            acc.scaler.step(optimizer);acc.scaler.update();handle.flush()
            after=package.diagnostics()['consumer']['counts']
            for event in ('summary_loss_completed','consumer_complete'):
                if after.get(event,0)-before.get(event,0)!=(1 if combined else 0):
                    raise RuntimeError('Production route activation mismatch: '+event)
            if names!={n:id(p) for n,p in model.named_parameters()}:raise RuntimeError('Parameter identity changed')
            if base.lm_head.weight is not base.model.embed_tokens.weight:raise RuntimeError('Tied head changed')
            if bridge_state.current() is not None or handle.pending is not None:raise RuntimeError('Summary scope leak')
            if rewind.failed or rewind.dropped or len(sink.rows)!=1:raise RuntimeError('Rewind failed')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite fixture loss')
            if len(gathers)!=4:raise RuntimeError('Native metric gather count changed')
            result={'loss':loss.detach().cpu().clone(),'scaled':scaled,'unscaled':unscaled,
                'parameters':{n:p.detach().cpu().clone() for n,p in params.items()},
                'rng':torch.cuda.get_rng_state().clone(),'metrics':copy.deepcopy(dict(trainer._metrics['train'])),
                'rewind':copy.deepcopy(sink.rows),'scaler':copy.deepcopy(acc.scaler.state_dict()),
                'gathers':copy.deepcopy(gathers),'hook_negatives':negative,
                'actual_args_type':type(args).__module__+'.'+type(args).__qualname__,
                'past_index_absent':not hasattr(args,'past_index')}
            # Public-output calls remain the native full-logit metric path.
            if combined:
                count=after.get('consumer_complete',0)
                for row in [4,8,15,16]:rewind.on_index(row)
                returned,outputs=trainer.compute_loss(model,{'input_ids':ids,'labels':labels,'attention_mask':mask},
                    return_outputs=True,num_items_in_batch=denominator)
                if (outputs.logits.dtype!=torch.float32 or outputs.logits.shape!=(4,8,151936)
                        or package.diagnostics()['consumer']['counts'].get('consumer_complete',0)!=count):
                    raise RuntimeError('Output-request fallback changed')
            acc.gather_for_metrics=original_gather
            acc.free_memory();torch.cuda.synchronize()
        return result

    groups=[]
    try:
        for scale in [1.,128.,65536.]:
            a=one(False,scale);gc.collect();torch.cuda.empty_cache()
            b=one(True,scale);gc.collect();torch.cuda.empty_cache()
            same_tensor(a['loss'].reshape(1),b['loss'].reshape(1),f'{scale}/loss')
            same_tensor(a['rng'],b['rng'],f'{scale}/rng')
            for key in ('scaled','unscaled','parameters'):
                if a[key].keys()!=b[key].keys():raise RuntimeError('Trainable names differ')
                for name in a[key]:same_tensor(a[key][name],b[key][name],f'{scale}/{key}/{name}')
            for key in ('entropy','mean_token_accuracy','num_tokens'):
                close_metric(a['metrics'][key],b['metrics'][key],f'{scale}/metrics/{key}',.01 if key=='entropy' else 0)
            for i,(x,y) in enumerate(zip(a['gathers'],b['gathers'])):
                close_metric(x,y,f'{scale}/gather/{i}',.01 if i==0 else 0)
            for x,y in zip(a['rewind'],b['rewind']):
                for field in ('step','micro','rows','row_tokens'):
                    if x[field]!=y[field]:raise RuntimeError('Rewind '+field+' changed')
                close_metric(x['row_loss'],y['row_loss'],f'{scale}/rewind',1e-4)
            if a['scaler']!=b['scaler']:raise RuntimeError('GradScaler state changed')
            groups.append({'scale':scale,'producer_loss_gradients_updates_rng_exact':True,
                'native_metric_gather_order_and_rewind_fifo_preserved':True,
                'entropy_rtol':.01,'row_nll_rtol':1e-4,'actual_args_type':b['actual_args_type'],
                'past_index_absent':b['past_index_absent'],'output_changing_hook_negatives':b['hook_negatives']})
            progress['completed_groups']=copy.deepcopy(groups)
        progress['stage']='complete'
        return {'producer':producer_checks,'groups':groups,'full_trainer_constructor':True,
                'actual_peft_accelerate_rewind_chain':True,'nf4_decoder':False,
                'tiny_random_model_only':True,'fixed_task_training':False,'diagnostics':package.diagnostics()}
    finally:package.restore()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    report={'status':'FAIL','gpu_execution_attempted':True,'gpu_executed':None,'score':None,'admission':'UNKNOWN'}
    report['partial_progress']={}
    try:report['checks']=run(report['partial_progress']);report.update(status='PASS_COMBINED_CE_METRIC_PREFLIGHT_ONLY',gpu_executed=True)
    except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'output':args.out}),flush=True)
    raise SystemExit(report['status']!='PASS_COMBINED_CE_METRIC_PREFLIGHT_ONLY')
if __name__=='__main__':main()
