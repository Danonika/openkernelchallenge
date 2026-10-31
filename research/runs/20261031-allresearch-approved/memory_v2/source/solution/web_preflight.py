"""Bounded real HF→NF4→PEFT→SFTConfig/Trainer→AMP streaming composition gate.

Synthetic local one-layer checkpoint with real H1536/V151936 and actual q/v
geometry. This loads through the real HF bitsandbytes NF4 quantizer, not fake
quantization attributes. No downloads, real pretrained weights, full-task
training or performance claim. Native-all versus combined-all, including the
reviewed NF4/RMS support. Main runtime uses the unchanged task separately.
"""
import argparse,copy,gc,hashlib,importlib,inspect,json,math,os,tempfile,traceback
from pathlib import Path


def classes(actual,expected,name):
    import torch
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise RuntimeError(name+': shape/dtype')
    for fn in (torch.isnan,torch.isposinf,torch.isneginf):
        if not torch.equal(fn(actual),fn(expected)):raise RuntimeError(name+': NaN/Inf class')


def metric(actual,expected,name,rtol=1e-4,atol=1e-6):
    import torch
    actual=torch.as_tensor(actual,dtype=torch.float64);expected=torch.as_tensor(expected,dtype=torch.float64)
    classes(actual,expected,name);finite=actual.isfinite()&expected.isfinite()
    torch.testing.assert_close(actual[finite],expected[finite],rtol=rtol,atol=atol,msg=name)
    delta=(actual[finite]-expected[finite]).abs()
    return {'max_abs':float(delta.max()) if delta.numel() else 0.,'rtol':rtol,'atol':atol}


def gradient(actual,expected,name):
    import torch
    classes(actual,expected,name);finite=actual.isfinite()&expected.isfinite()
    # CPU snapshots use FP64 reductions: finite FP32 gradients near1e20
    # can overflow an FP32 norm and falsely report zero relative error.
    a=actual[finite].double();b=expected[finite].double();delta=(a-b).abs()
    maxabs=float(delta.max()) if delta.numel() else 0.;refmax=float(b.abs().max()) if b.numel() else 0.
    error_norm=float(delta.norm());reference_norm=float(b.norm())
    relative=error_norm/max(reference_norm,1e-30) if b.numel() else 0.
    if not all(math.isfinite(v) for v in (maxabs,refmax,error_norm,reference_norm,relative)):
        raise RuntimeError(name+': nonfinite derived gradient statistic')
    if relative>.001 or maxabs>.005*refmax+1e-7:raise RuntimeError(f'{name}: gradient gate relL2={relative} maxabs={maxabs} refmax={refmax}')
    return {'relative_l2':relative,'max_abs':maxabs,'reference_max_abs':refmax}


def create_checkpoint(path):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(445)
    config=Qwen2Config(vocab_size=151936,hidden_size=1536,intermediate_size=8960,
        num_hidden_layers=1,num_attention_heads=12,num_key_value_heads=2,
        max_position_embeddings=512,tie_word_embeddings=True,attention_dropout=0.,
        use_cache=False,eos_token_id=1,pad_token_id=0)
    model=Qwen2ForCausalLM(config).to(dtype=torch.float16)
    model.save_pretrained(path,safe_serialization=True,max_shard_size='1GB')
    del model;gc.collect()


def load_nf4(checkpoint,task,*,owned_storage=False):
    import torch,bitsandbytes as bnb
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training,get_peft_model,LoraConfig,TaskType
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=task.training.double_quant_on)
    base=AutoModelForCausalLM.from_pretrained(str(checkpoint),local_files_only=True,
        trust_remote_code=False,device_map={'':0},torch_dtype=torch.float16,quantization_config=quant)
    if not base.is_loaded_in_4bit or getattr(base,'hf_quantizer',None) is None:
        raise RuntimeError('Actual HF NF4 loader/quantizer absent')
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=task.training.lora.r,
        lora_alpha=task.training.lora.alpha,lora_dropout=task.training.lora.dropout,
        target_modules=['q_proj','v_proj'],bias='none'))
    packed=[]
    for name,module in base.named_modules():
        if isinstance(module,bnb.nn.Linear4bit):
            qs=getattr(module.weight,'quant_state',None)
            if (type(module.weight) is not bnb.nn.Params4bit or qs is None
                    or qs.quant_type!='nf4' or module.weight.dtype!=torch.uint8):
                raise RuntimeError('Actual packed NF4 state absent: '+name)
            packed.append(name)
    if not packed or not any('q_proj' in n for n in packed) or not any('v_proj' in n for n in packed):
        raise RuntimeError('Actual q/v NF4 modules absent')
    if owned_storage:
        from .frozen_storage import attest_head
        if not attest_head(base,base.lm_head,base.lm_head.weight,torch.nn.Linear.forward):
            raise RuntimeError('Actual pre-PEFT storage provenance/head attestation absent')
    elif base.lm_head.weight.dtype!=torch.float32 or base.lm_head.weight.requires_grad:
        raise RuntimeError('Native prepare_model_for_kbit_training frozen FP32 head absent')
    return model,base,packed


def make_trainer(model,directory,owners,task):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast,TrainingArguments
    from trl import SFTConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16
    from soup_cli.utils.warmup import resolve_trainer_warmup_steps
    t=task.training
    initial=TrainingArguments(output_dir=str(directory),fp16=True,bf16=False,
        per_device_train_batch_size=t.batch_size,gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_train_epochs=t.epochs,learning_rate=t.lr,weight_decay=t.weight_decay,
        warmup_steps=resolve_trainer_warmup_steps(t.warmup_ratio),
        max_grad_norm=t.max_grad_norm,optim=t.optimizer,lr_scheduler_type=t.scheduler,
        seed=t.seed,logging_steps=t.logging_steps,save_steps=t.save_steps,
        remove_unused_columns=False,report_to=[],gradient_checkpointing=t.gradient_checkpointing,
        dataloader_num_workers=0)
    args=SFTTrainerWrapper._as_sft_config(initial,task.data.max_length,packing=t.packing)
    if type(args) is not SFTConfig or hasattr(args,'past_index'):raise RuntimeError('Actual SFTConfig contract changed')
    if (args.loss_type!='nll' or args.padding_free or args.activation_offloading or args.gradient_checkpointing
            or args.use_liger_kernel or args.average_tokens_across_devices is not True):
        raise RuntimeError('Actual task/SFTConfig flags do not match owned route')
    # Synthetic already-tokenized rows only. No real task data preprocessing or
    # task switch is changed; this constructor is a separate component gate.
    args.dataset_kwargs={'skip_prepare_dataset':True}
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {'<pad>':0,'<eos>':1,'<unk>':2},unk_token='<unk>')),
        pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    dataset=Dataset.from_dict({'input_ids':[[2,2,2,1]]*4,'attention_mask':[[1,1,1,1]]*4})
    trainer=owners.rewind_class(model=model,args=args,train_dataset=dataset,processing_class=tokenizer)
    # Reproduce Soup's actual train() boundary (sft.py2521): TRL may
    # downcast QLoRA toBF16; native Soup restores LoRA FP32 for T4GradScaler.
    dtype_source=Path(inspect.getfile(align_trainable_dtype_for_fp16))
    if hashlib.sha256(dtype_source.read_bytes()).hexdigest()!='80934a7f10387c114174801b41d45de3cbc53f85cd74f321bab9335cd3714d6f':
        raise RuntimeError('Native Soup dtype alignment source changed')
    before={n:str(p.dtype) for n,p in trainer.model.named_parameters() if p.requires_grad}
    casted=align_trainable_dtype_for_fp16(trainer.model,fp16=args.fp16,bf16=args.bf16)
    after={n:str(p.dtype) for n,p in trainer.model.named_parameters() if p.requires_grad}
    if not after or any(v!='torch.float32' for v in after.values()):
        raise RuntimeError('Native Soup trainable FP32 alignment failed')
    dtype_audit={'before_native_soup_alignment':before,'after_native_soup_alignment':after,
        'native_helper_cast_count':casted,'native_helper_source_sha256':'80934a7f10387c114174801b41d45de3cbc53f85cd74f321bab9335cd3714d6f'}
    prepared=trainer.accelerator.prepare_model(trainer.model)
    trainer.model=prepared;trainer.model_wrapped=prepared;prepared.train()
    return trainer,prepared,args,dtype_audit


def run(progress,save):
    import torch,triton
    from soup_cli.config.loader import load_config
    from soup_cli.trainer.rewind_hf import RewindState
    from .source_owners import load_pinned_owners
    from .common import bridge_state
    package=importlib.import_module(__package__)
    if (os.environ.get('TRITON_INTERPRET') or torch.__version__!='2.11.0+cu128'
            or triton.__version__!='3.6.0' or torch.cuda.device_count()!=1
            or torch.cuda.get_device_name(0)!='Tesla T4' or torch.cuda.get_device_capability(0)!=(7,5)):
        raise RuntimeError('Exact native target runtime required')
    task=load_config(Path(os.environ['SOUP_REPO'])/'task/task.yaml')
    if (task.training.stream_loss_chunk is not None or task.training.stream_layers
            or task.training.gradient_checkpointing or task.training.use_liger or task.training.use_cut_ce):
        raise RuntimeError('Immutable task optimization switches changed')
    owners,_,_=load_pinned_owners()
    class Sink:
        def __init__(self):self.rows=[]
        def record_batch(self,**kw):self.rows.append(kw)
    def phase(name,**kw):
        progress['current_phase']={'phase':name,**kw};save()

    with tempfile.TemporaryDirectory(prefix='streaming-nf4-preflight-',dir='/kaggle/working') as directory:
        root=Path(directory);checkpoint=root/'checkpoint';checkpoint.mkdir()
        phase('create_synthetic_local_checkpoint');create_checkpoint(checkpoint)
        phase('checkpoint_ready',bytes=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()))
        def one(combined,scale):
            phase('load_actual_hf_nf4',combined=combined,scale=scale)
            if combined:package.apply()
            torch.manual_seed(144)
            handle=package._handle if combined else None
            try:
                model,base,packed=load_nf4(checkpoint,task,owned_storage=combined)
                phase('construct_real_trainer',combined=combined,scale=scale,nf4_modules=packed)
                trainer,model,args,dtype_audit=make_trainer(model,root/f'trainer-{combined}-{int(scale)}',handle.o if combined else owners,task)
                acc=trainer.accelerator;acc.scaler=torch.amp.GradScaler('cuda',init_scale=scale,growth_interval=100)
                sink=Sink();rewind=RewindState(sink)
                for index in [4,8,15,16]:rewind.on_index(index)
                trainer._soup_rewind_state=rewind
                ids=torch.randint(0,151936,(4,37),device='cuda');labels=ids.clone();labels[:,:24]=-100
                labels[:,24:]=ids[:,24:];mask=torch.ones_like(ids);mask[0,-1]=0
                inputs={'input_ids':ids,'labels':labels,'attention_mask':mask}
                denominator=trainer._get_num_items_in_batch([inputs],acc.device)
                if not isinstance(denominator,torch.Tensor) or denominator.dtype!=torch.int64 or denominator.ndim!=0:
                    raise RuntimeError('Native Trainer global denominator absent')
                if int(denominator)!=(labels[:,1:]!=-100).sum().item():raise RuntimeError('Native causal denominator changed')
                hook_negatives=[]
                if combined:
                    reason=handle.owner_guard.reason(trainer,model,inputs,False)
                    if reason is not None:raise RuntimeError('Full Trainer streaming guard: '+str(reason))
                    for name,module in [('peft',model),('tuner',model.base_model),('qwen',base),('head',base.lm_head)]:
                        hook=module.register_forward_hook(lambda m,a,out:out)
                        try:
                            if handle.owner_guard.reason(trainer,model,inputs,False)!='foreign_module_hook':
                                raise RuntimeError('Foreign-hook guard inactive: '+name)
                            hook_negatives.append(name)
                        finally:hook.remove()
                actual_backbones=[]
                counter=base.model.register_forward_hook(lambda _m,_a,_out:actual_backbones.append(1))
                original_gather=acc.gather_for_metrics;gathers=[]
                def gather(value,*a,**kw):
                    gathers.append(value.detach().cpu().clone());return original_gather(value,*a,**kw)
                acc.gather_for_metrics=gather
                names={n:id(p) for n,p in model.named_parameters()}
                params={n:p for n,p in model.named_parameters() if p.requires_grad}
                if not params or not all('lora_' in n and ('q_proj' in n or 'v_proj' in n) for n in params):
                    raise RuntimeError('Unexpected trainable set')
                optimizer=torch.optim.AdamW(params.values(),lr=task.training.lr,weight_decay=task.training.weight_decay)
                before=copy.deepcopy(package.diagnostics()) if combined else None
                phase('actual_native_amp_forward_backward',combined=combined,scale=scale)
                loss=trainer.compute_loss(model,inputs,num_items_in_batch=denominator)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite synthetic fixture loss')
                after_forward=copy.deepcopy(package.diagnostics()) if combined else None
                acc.backward(loss)
                scaled={n:p.grad.detach().cpu().clone() for n,p in params.items()}
                acc.unscale_gradients(optimizer)
                unscaled={n:p.grad.detach().cpu().clone() for n,p in params.items()}
                # Same native Torch clip primitive used by single-device HF
                # after GradScaler unscale; do not unscale a second time.
                clip_norm=torch.nn.utils.clip_grad_norm_(tuple(params.values()),args.max_grad_norm)
                acc.scaler.step(optimizer);acc.scaler.update()
                if combined:handle.flush()
                after=copy.deepcopy(package.diagnostics()) if combined else None
                if combined:
                    for event in ('owned_consumer_guard','backbone_once','summary_loss_completed','streaming_loss_return','consumer_complete'):
                        if after['counts'].get(event,0)-before['counts'].get(event,0)!=1:
                            raise RuntimeError('Production streaming route not activated: '+event)
                    offload=after['activation_offload'];previous_offload=before['activation_offload']
                    if not offload.get('owners_intact') or not offload.get('active'):
                        raise RuntimeError('Actual backbone offload ownership absent')
                    for event in ('offloaded_forward','packed_tensors','unpacked_tensors'):
                        if offload['counts'].get(event,0)-previous_offload['counts'].get(event,0)<=0:
                            raise RuntimeError('Actual backbone offload route absent: '+event)
                    for event in ('nf4_decode','rmsnorm'):
                        if after['support']['calls'].get(event,0)-before['support']['calls'].get(event,0)<=0:
                            raise RuntimeError('Actual support route not activated: '+event)
                    compact=after['compact_mlp'];prior=before['compact_mlp']
                    if (compact.get('owners_intact') is not True or compact.get('active') is not True
                            or type(compact.get('original_kernel_credit')) is not int or compact['original_kernel_credit']!=0
                            or any(type(state['counts'].get(key)) is not int for state in (compact,prior) for key in ('owned','native'))
                            or compact['counts']['owned']-prior['counts']['owned']!=1
                            or compact['counts']['native']-prior['counts']['native']!=0):
                        raise RuntimeError('Actual I8960 compact MLP route absent or ownership/fallback changed')
                    storage=after['frozen_storage'];prior_storage=before['frozen_storage']
                    if (storage.get('owners_intact') is not True or storage.get('active') is not True
                            or type(storage.get('original_kernel_credit')) is not int or storage['original_kernel_credit']!=0
                            or any(type(storage['counts'].get(key)) is not int for key in
                                   ('installed','captured','declined','embedding_calls','head_fp16_calls','head_native_operand_calls'))
                            or type(storage.get('live_storage_owners')) is not int
                            or type(prior_storage['counts'].get('embedding_calls')) is not int
                            or storage['counts']['installed']!=1 or storage['counts']['captured']!=1
                            or storage['counts']['declined']!=0 or storage['live_storage_owners']!=1
                            or storage['counts']['embedding_calls']-prior_storage['counts']['embedding_calls']!=1
                            or storage['counts']['head_fp16_calls']!=0 or storage['counts']['head_native_operand_calls']!=0):
                        raise RuntimeError('Actual owned frozen embedding route absent or provenance changed')

                if len(actual_backbones)!=1:raise RuntimeError('Decoder executed more than once')
                if names!={n:id(p) for n,p in model.named_parameters()}:raise RuntimeError('Parameter identity changed')
                if base.lm_head.weight is not base.model.embed_tokens.weight:raise RuntimeError('Tied head identity changed')
                if bridge_state.current() is not None:raise RuntimeError('Streaming scope leaked')
                if rewind.failed or rewind.dropped or len(sink.rows)!=1 or len(gathers)!=4:
                    raise RuntimeError('Original Rewind/gather contract failed')
                result={'loss':loss.detach().cpu().clone(),'scaled':scaled,'unscaled':unscaled,
                    'parameters':{n:p.detach().cpu().clone() for n,p in params.items()},
                    'cpu_rng':torch.get_rng_state().clone(),'cuda_rng':torch.cuda.get_rng_state().clone(),
                    'metrics':copy.deepcopy(dict(trainer._metrics['train'])),'rewind':copy.deepcopy(sink.rows),
                    'scaler':copy.deepcopy(acc.scaler.state_dict()),'clip_norm':clip_norm.detach().cpu().clone(),'gathers':copy.deepcopy(gathers),
                    'diagnostics':after,'support_after_forward':after_forward['support'] if combined else None,'nf4_modules':packed,'actual_args_type':type(args).__module__+'.'+type(args).__qualname__,
                    'average_tokens_across_devices':args.average_tokens_across_devices,'native_soup_dtype_alignment':dtype_audit,'hook_negatives':hook_negatives}
                # Snapshot before this extra public call (do not alias metric/FIFO lists).
                if combined:
                    old_count=after['counts'].get('streaming_loss_return',0)
                    for index in [4,8,15,16]:rewind.on_index(index)
                    returned,outputs=trainer.compute_loss(model,{'input_ids':ids,'labels':labels,'attention_mask':mask},
                        return_outputs=True,num_items_in_batch=denominator)
                    if outputs.logits.dtype!=torch.float32 or outputs.logits.shape!=(4,37,151936):
                        raise RuntimeError('Public output request did not retain native logits')
                    if package.diagnostics()['counts'].get('streaming_loss_return',0)!=old_count:
                        raise RuntimeError('Public output request incorrectly streamed')
                    if len(actual_backbones)!=2:raise RuntimeError('Public call repeated decoder internally')
                    public_storage=package.diagnostics()['frozen_storage']['counts']
                    if (public_storage['embedding_calls']!=2 or public_storage['head_fp16_calls']!=1
                            or public_storage['head_native_operand_calls']!=0):
                        raise RuntimeError('Public native output did not use attested FP16-AMP head proxy exactly once')

                    del outputs,returned
                counter.remove();acc.gather_for_metrics=original_gather
                del loss,optimizer,params,trainer,model,base
                acc.free_memory();del acc;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                return result
            finally:
                if combined:
                    restored=package.restore()
                    if restored!={'support':True,'head':True,'compact_mlp':True,'frozen_storage':True,'activation_offload':True}:raise RuntimeError('Owned combined restoration failed: '+str(restored))
                gc.collect();torch.cuda.empty_cache()

        groups=[];progress['completed_groups']=groups
        for scale in (1.,65536.):
            native=one(False,scale);candidate=one(True,scale)
            phase('compare_native_all_vs_combined_all',scale=scale)
            checks={'loss':metric(candidate['loss'],native['loss'],'loss'),
                    'native_clip_norm':metric(candidate['clip_norm'],native['clip_norm'],'clip_norm',rtol=.001,atol=1e-7)}
            for key in ('cpu_rng','cuda_rng'):
                if not torch.equal(native[key],candidate[key]):raise RuntimeError(key+' changed')
            for key in ('scaled','unscaled'):
                if native[key].keys()!=candidate[key].keys():raise RuntimeError('LoRA parameter names changed')
                checks[key]={n:gradient(candidate[key][n],native[key][n],key+'/'+n) for n in native[key]}
            checks['updates']={n:metric(candidate['parameters'][n],native['parameters'][n],'update/'+n,rtol=.005,atol=1e-6) for n in native['parameters']}
            for key in ('entropy','mean_token_accuracy','num_tokens'):
                checks[key]=metric(candidate['metrics'][key],native['metrics'][key],key,rtol=.01 if key=='entropy' else 0,atol=1e-6 if key=='entropy' else 0)
            for i,(a,b) in enumerate(zip(candidate['gathers'],native['gathers'])):
                metric(a,b,'gather/'+str(i),rtol=.01 if i==0 else 0,atol=1e-6 if i==0 else 0)
            for a,b in zip(candidate['rewind'],native['rewind']):
                for key in ('step','micro','rows','row_tokens'):
                    if a[key]!=b[key]:raise RuntimeError('Rewind '+key+' changed')
                checks['rewind_rows']=metric(a['row_loss'],b['row_loss'],'rewind_rows')
            if candidate['scaler']!=native['scaler']:raise RuntimeError('GradScaler state/update changed')
            groups.append({'scale':scale,'checks':checks,'native_all_vs_combined_all':True,
                'actual_hf_nf4_loaded':True,'nf4_modules':candidate['nf4_modules'],
                'actual_args_type':candidate['actual_args_type'],'average_tokens_across_devices':True,
                'native_soup_dtype_alignment':candidate['native_soup_dtype_alignment'],
                'one_decoder_call_per_transaction':True,'rng_scaler_fifo_and_gather_order_preserved':True,
                'output_request_native':True,'foreign_hook_negatives':candidate['hook_negatives'],
                'support_after_forward':candidate['support_after_forward'],'diagnostics':candidate['diagnostics']});save()
            del native,candidate
    return {'groups':groups,'full_trainer_constructor':True,'actual_nf4_quantizer_and_peft_path':True,
        'synthetic_random_one_layer_checkpoint':True,'actual_pretrained_weights':False,'fixed_task_training':False,
        'H':1536,'V':151936,'decoder_layers':1,'intermediate_size':8960,'B':4,'T':37,'token_chunks':2,
        'predeclared_gates':{'loss_nll_rows':'rtol1e-4/atol1e-6','entropy':'rtol.01/atol1e-6',
            'lora_gradients':'FP64snapshot norms,relativeL2<=.001,maxabs<=.005*reference_max+1e-7,exactnonfiniteclasses',
            'lora_updates':'rtol.005/atol1e-6','scaler_rng_counts_accuracy':'exact'}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    report={'status':'FAILED_STREAMING_NF4_INTEGRATION_PREFLIGHT_ONLY','progress':{},
        'actual_target_execution':True,'fixed_task_training':False,'score':None,'admission':'UNKNOWN'}
    def save():
        tmp=out.with_name(out.name+'.pending');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    try:report['checks']=run(report['progress'],save);report['status']='PASS_PRODUCTION_STREAMED_CE_PREFLIGHT_ONLY'
    except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:save()
    print(json.dumps({'status':report['status'],'out':str(out)}),flush=True)
    raise SystemExit(0 if report['status']=='PASS_PRODUCTION_STREAMED_CE_PREFLIGHT_ONLY' else 1)
if __name__=='__main__':main()
