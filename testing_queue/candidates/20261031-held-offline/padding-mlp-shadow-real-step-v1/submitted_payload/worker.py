"""One actual-batch dense-starter forward/native-return backward shadow.
No optimizer, train loop, full baseline replay, profiler or performance claim.
"""
import argparse,hashlib,importlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
from profiler_helpers import tensor_digest

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    text=json.dumps(v,indent=2,allow_nan=False)+'\n'
    if len(text.encode())>2*1024*1024:raise RuntimeError('Diagnostic JSON exceeds2MiB bound')
    Path(p).write_text(text)
def source_binding(here):
    pins=json.loads((here/'native_source_pins.json').read_text());actual={}
    for relative,wanted in pins.items():
        package=relative.split('/',1)[0]
        spec=importlib.util.find_spec(package)
        root=Path(next(iter(spec.submodule_search_locations)))
        path=root/relative.split('/',1)[1]
        observed=sha(path)
        if observed!=wanted:raise RuntimeError('Native source pin mismatch: '+relative)
        actual[relative]={'sha256':observed,'path':str(path)}
    return actual

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--repo',required=True);a=ap.parse_args()
    out=Path(a.out);repo=Path(a.repo).resolve();here=Path(__file__).resolve().parent
    if out.exists():raise FileExistsError(out)
    start=time.perf_counter()
    r={'status':'FAILED_NATIVE_RETURN_MLP_SHADOW_DIAGNOSTIC','scope':'ACTUAL_BATCH_NATIVE_RETURN_MLP_SHADOW_DIAGNOSTIC_ONLY',
       'complete_fixed_task':False,'training_step_calls':0,'optimizer_steps':0,'score':None,'admission':'UNKNOWN',
       'warmup_forward_backward_calls':0,'fresh_weights_not_actual_step29_state':True,
       'task_quality_or_throughput_claim':False,'native_backward_result_returned':True,
       'range_certificate':False,'production_clearance':False,'profiler_used':False,
       'initialization_before_diagnostic':'Native config/data/tokenizer/full NF4 model/PEFT/real SFTTrainer, native trainable dtype alignment, Accelerator model preparation, actual batch collation, metadata/hashes. No forward warmup or optimizer creation.',
       'limits':['One fresh actual-batch dense starter-half forward and native-return backward; not original native-SiLU equivalence.',
                 'Compact live-row dX is only a discarded witness. No task quality, peak VRAM, throughput, score or production claim.',
                 'No optimizer, full baseline replay, backend flag change or exceptional-range certificate.']}
    def progress(phase):
        dump(out.parent/'progress.json',{'phase':phase,'elapsed_s':time.perf_counter()-start,
            'scope':r['scope'],'training_step_calls':r['training_step_calls']})
        print('NATIVE_RETURN_MLP_SHADOW '+phase,flush=True)
    try:
        progress('SOURCE_AND_RUNTIME_BINDING')
        sys.modules['unsloth']=None;sys.modules['unsloth_zoo']=None
        if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Interpreter forbidden')
        for envkey in ('TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
            cache=Path(os.environ[envkey]);cache.mkdir(parents=True,exist_ok=True)
            if any(cache.iterdir()):raise RuntimeError('Private empty initial cache required: '+envkey)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one CUDA device required')
        r['native_sources']=source_binding(here)
        r['read_only_precision_flags']={'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'allow_fp16_accumulation':getattr(torch.backends.cuda.matmul,'allow_fp16_accumulation',None),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'math_sdp_fp16_bf16_reduction_allowed':torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed() if hasattr(torch.backends.cuda,'fp16_bf16_reduction_math_sdp_allowed') else None}
        binding=json.loads((here/'batch_binding.json').read_text());expected=json.loads((here/'batch_hashes.json').read_text())
        r['batch_selection_binding']=binding
        if sha(repo/'data/train.jsonl')!=binding['asset_sha256']['train.jsonl']:raise RuntimeError('Public training bytes changed')
        from huggingface_hub.constants import HF_HUB_CACHE
        root=Path(HF_HUB_CACHE)/'models--Qwen--Qwen2.5-1.5B-Instruct'
        if (root/'refs/main').read_text().strip()!=binding['model_revision']:raise RuntimeError('Model cached revision changed')
        snapshot=root/'snapshots'/binding['model_revision']
        r['model_asset_binding']={n:sha(snapshot/n) for n in ('config.json','tokenizer.json','tokenizer_config.json')}
        if any(v!=binding['asset_sha256'][n] for n,v in r['model_asset_binding'].items()):raise RuntimeError('Config/tokenizer bytes changed')
        import yaml
        fixed=yaml.safe_load((repo/'task/task.yaml').read_text());configured=json.loads(json.dumps(fixed))
        setupdir=out.parent/'native_setup';setupdir.mkdir(exist_ok=False)
        configured['data']['train']=str(repo/'data/train.jsonl');configured['training']['seed']=1
        configured['output']=str(setupdir/'adapter')
        configpath=setupdir/'diagnostic_config.yaml';configpath.write_text(yaml.safe_dump(configured))
        r['configuration']={'fixed_task_sha256':sha(repo/'task/task.yaml'),'resolved_config':configured,
            'only_resolutions':['data.train absolute public path','training.seed1 (unchanged)','output diagnostic directory']}
        from soup_cli.config.loader import load_config
        from soup_cli.data.loader import load_dataset,data_config_for_task,task_preserves_source_columns
        from soup_cli.utils.eval_schedule import loader_data_config
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16
        cfg=load_config(configpath)
        for key in ('use_liger','use_cut_ce','use_flash_attn','gradient_checkpointing','packing','multipack','stream_layers','activation_offloading'):
            if getattr(cfg.training,key,False):raise RuntimeError('Non-native diagnostic option: '+key)
        progress('NATIVE_DATA_AND_FULL_MODEL_SETUP')
        dataset=load_dataset(data_config_for_task(loader_data_config(cfg),cfg.task),preserve_source_columns=task_preserves_source_columns(cfg.task))
        if len(dataset['train'])!=256:raise RuntimeError('Expected all256 public rows')
        wrapper=SFTTrainerWrapper(cfg,device='cuda',report_to='none',trust_remote_code=False)
        wrapper.setup(dataset);trainer=wrapper.trainer
        if not trainer.args.fp16 or trainer.args.bf16 or trainer.args.gradient_accumulation_steps!=1 or trainer.args.per_device_train_batch_size!=4:
            raise RuntimeError('Native training argument mismatch')
        if trainer.args.gradient_checkpointing or trainer.args.torch_compile or getattr(trainer,'padding_free',False):raise RuntimeError('Unsupported altered execution geometry')
        if trainer.optimizer is not None or trainer.lr_scheduler is not None:raise RuntimeError('No optimizer/scheduler may be created')
        r['native_alignment_cast_count']=align_trainable_dtype_for_fp16(trainer.model,fp16=trainer.args.fp16,bf16=trainer.args.bf16)
        trainer.model=trainer.accelerator.prepare_model(trainer.model);trainer.model_wrapped=trainer.model
        model=trainer.model;model.train();trainer.current_gradient_accumulation_steps=trainer.args.gradient_accumulation_steps
        base=model.get_base_model();config=base.config
        if (config.num_hidden_layers,config.hidden_size,config.intermediate_size,config.vocab_size)!=(28,1536,8960,151936):raise RuntimeError('Fixed model geometry mismatch')
        if getattr(config,'_commit_hash',None)!=binding['model_revision']:raise RuntimeError('Loaded model revision mismatch')
        r['model']={'class':type(base).__module__+'.'+type(base).__name__,'revision':config._commit_hash,
            'layers':config.num_hidden_layers,'H':config.hidden_size,'I':config.intermediate_size,'V':config.vocab_size,
            'attn_implementation':getattr(config,'_attn_implementation',None),'attention_dropout':getattr(config,'attention_dropout',None),
            'config_torch_dtype':str(getattr(config,'dtype',None)),
            'nf4_module_count':sum(type(m).__module__.startswith('bitsandbytes.') and type(m).__name__=='Linear4bit' for m in model.modules()),
            'trainer_class':type(trainer).__module__+'.'+type(trainer).__name__,'args_class':type(trainer.args).__module__+'.'+type(trainer.args).__name__,
            'fp16':trainer.args.fp16,'bf16':trainer.args.bf16,'gradient_accumulation_steps':trainer.args.gradient_accumulation_steps,
            'model_accepts_loss_kwargs':trainer.model_accepts_loss_kwargs,'loss_shifts_labels':trainer._loss_shifts_labels}
        if r['model']['nf4_module_count']!=196:raise RuntimeError('Actual full NF4 module count mismatch')
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        if len(trainable)!=112 or sum(p.numel() for p in trainable.values())!=2179072 or any(p.dtype!=torch.float32 or 'lora_' not in n for n,p in trainable.items()):raise RuntimeError('Native LoRA trainable identity mismatch')
        before={n:{'id':id(p),'version':p._version,**tensor_digest(p)} for n,p in trainable.items()}
        r['trainable_parameters']={'count':len(trainable),'elements':sum(p.numel() for p in trainable.values()),'dtype':'torch.float32',
            'before_combined_sha256':hashlib.sha256(json.dumps({n:x['sha256'] for n,x in before.items()},sort_keys=True).encode()).hexdigest()}
        batch=trainer.data_collator([trainer.train_dataset[i] for i in binding['rows']])
        if set(batch)!=set(expected['batch']):raise RuntimeError('Unexpected collated fields')
        r['batch']={n:tensor_digest(t) for n,t in batch.items()}
        for n,b in r['batch'].items():
            wanted=expected['batch'][n]
            if (b['shape'],b['dtype'],b['sha256'],b['bytes'])!=(wanted['shape'],wanted['dtype'],wanted['raw_sha256'],wanted['raw_bytes']):raise RuntimeError('Exact native collator bytes mismatch: '+n)
        if batch['attention_mask'].sum(1).tolist()!=binding['lengths'] or batch['labels'][:,1:].ne(-100).sum(1).tolist()!=binding['supervised_tokens']:raise RuntimeError('Actual batch token count mismatch')
        denominator=trainer._get_num_items_in_batch([batch],trainer.accelerator.device)
        if denominator is None or int(denominator)!=sum(binding['supervised_tokens']):raise RuntimeError('Native loss denominator mismatch')
        r['batch']['native_num_items_in_batch']=int(denominator)
        state=trainer._soup_rewind_state
        if state is None or state.failed or state.dropped or len(state._pending):raise RuntimeError('Native Rewind state unavailable')
        for index in binding['rows']:state.on_index(index)
        r['rewind_initialization']='Exact selected real row IDs queued explicitly because this diagnostic bypasses sampler; native fresh global_step0, recordstep1, not learned step29.'
        scaler=trainer.accelerator.scaler
        if scaler is None or not scaler.is_enabled() or scaler.get_scale()!=65536.:raise RuntimeError('Native initial GradScaler mismatch')
        r['initial_grad_scaler_scale']=scaler.get_scale()
        r['initial_grad_scaler_state']=scaler.state_dict()
        if any(n=='solution' or n.startswith('solution.') for n in sys.modules):raise RuntimeError('Diagnostic solution imported before native setup')
        from solution.compact_mlp_install import install_compact_mlp
        from solution.trainer_shadow import OneStepNativeShadow
        torch.cuda.synchronize();r['initialization_wall_s']=time.perf_counter()-start
        progress('ONE_ACTUAL_BATCH_NATIVE_RETURN_MLP_SHADOW')
        installation=install_compact_mlp();seam=None
        try:
            seam=OneStepNativeShadow(trainer,model,installation)
            with seam:
                r['training_step_calls']+=1
                loss=trainer.training_step(model,batch,num_items_in_batch=denominator)
                torch.cuda.synchronize()
        finally:
            if seam is not None:r['shadow']=seam.report
            try:r['dense_mlp_owner_before_restore']=installation.status()
            finally:r['dense_mlp_owner_restored']=installation.restore()
        if not r['dense_mlp_owner_restored']:raise RuntimeError('Dense MLP owner restoration failed')
        observed=r['shadow']
        if (observed.get('compute_loss_calls')!=1 or not observed.get('compute_loss_restored')
                or not observed.get('backward_phase_restored') or not observed.get('original_batch_unchanged')
                or observed.get('backward_cpu_rng_unchanged') is not True
                or observed.get('backward_cuda_rng_unchanged') is not True):
            raise RuntimeError('Native-return lifecycle/mask/backward-RNG contract failed')
        # Native fallbacks and every finite numeric difference are recorded;
        # completion is not numerical equivalence, eligibility or production PASS.
        r['observed_eligibility']={'bound_contexts':observed.get('bound_contexts',0),
            'expected_mlp_contexts_if_all_owned':84,'events':len(observed.get('events',[])),
            'shadow_eligible':sum(e['route']=='shadow_compact' for e in observed.get('events',[])),
            'native_fallbacks':sum(e['route']=='native' and e.get('label') is not None for e in observed.get('events',[])),
            'unowned_native_contexts':sum(e.get('label') is None for e in observed.get('events',[]))}
        progress('BOUNDED_SHADOW_AND_NO_UPDATE_CHECK')
        if trainer.optimizer is not None or trainer.lr_scheduler is not None or trainer.state.global_step!=0:raise RuntimeError('Unexpected optimizer or completed update')
        after={n:{'id':id(p),'version':p._version,**tensor_digest(p)} for n,p in trainable.items()}
        if before!=after:raise RuntimeError('Trainable parameter identity/version/bytes changed without authorized update')
        r['trainable_parameters']['unchanged_identity_version_and_bytes']=True
        r['trainable_parameters']['gradient_parameter_count']=sum(p.grad is not None for p in trainable.values())
        r['trainable_parameters']['nonfinite_gradient_elements']=sum(int((~torch.isfinite(p.grad)).sum().item()) for p in trainable.values() if p.grad is not None)
        # First-scale overflow may occur natively; record, never silently unscale/update it.
        r['post_grad_scaler_scale']=scaler.get_scale()
        r['post_grad_scaler_state']=scaler.state_dict()
        if r['initial_grad_scaler_state']!=r['post_grad_scaler_state']:raise RuntimeError('Scaler state changed without optimizer/update')
        r['scaler_scope']='Actual native initial-scale backward only; no unscale_, found_inf, optimizer step or scaler.update proof'
        r['rewind']={'failed':state.failed,'dropped':state.dropped,'pending':len(state._pending),'global_step':trainer.state.global_step}
        if state.failed or state.dropped or state._pending:raise RuntimeError('Native Rewind execution incomplete')
        r['descriptive_fresh_batch_loss_not_quality_score']=float(loss.item())
        if not torch.isfinite(loss).item():raise RuntimeError('Native diagnostic nonfinite loss')
        r['native_sources_after']=source_binding(here)
        if r['native_sources_after']!=r['native_sources']:raise RuntimeError('Native source changed')
        r['status']='COMPLETE_NATIVE_RETURN_MLP_SHADOW_DIAGNOSTIC'
    except BaseException as exc:r.update(error=repr(exc),traceback=traceback.format_exc())
    r['worker_wall_s']=time.perf_counter()-start
    dump(out,r);progress(r['status'])
    print(json.dumps({'status':r['status'],'training_step_calls':r['training_step_calls'],'optimizer_steps':r['optimizer_steps']}),flush=True)
    raise SystemExit(r['status']!='COMPLETE_NATIVE_RETURN_MLP_SHADOW_DIAGNOSTIC')
if __name__=='__main__':main()
