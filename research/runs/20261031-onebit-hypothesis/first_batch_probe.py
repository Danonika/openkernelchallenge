"""Native first-batch loss sensitivity, stopped before backward/optimizer.

Standalone diagnostic only: not solution.apply(), not a completed task, no
benchmark/admission claim. Uses the unchanged task fields and native Soup
loader, tokenizer, sampler, collator, LoRA setup and loss implementation.
Expanded binary tensors are numerical references, not a storage/speed kernel.
"""
import time
START=time.perf_counter()
import copy,hashlib,json,os,random,sys,tempfile,traceback
from pathlib import Path
import numpy as np
import yaml
import torch
import bitsandbytes as bnb
from trl import SFTTrainer
from soup_cli.trainer.rewind_hf import make_rewind_trainer_class

REPO=Path(os.environ['SOUP_REPO']);OUT=Path(os.environ['SOUP_RESULTS'])/'onebit_first_batch.json'
report={'status':'PREPARING','scope':'FIRST_NATIVE_TRAIN_BATCH_FORWARD_DIAGNOSTIC_ONLY',
 'optimizer_steps':0,'backward_calls':0,'completed_training_steps':0,'public_eval':'NOT_RUN',
 'official_admission':'UNKNOWN','benchmark_eligible':False,'variants':[],
 'limitations':['No full 64-step run, gradient check or held-out evaluation',
 'Weight binarization is lossy; admission is not established',
 'Expanded binary tensors are slow numerical references, not a packed kernel',
 'No speed or memory comparison; preparation occurs inside the diagnostic',
 'Loss values are genuine; repeated forwards are diagnostics, not logged training steps']}

def save():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+'\n')

class ProbeDone(BaseException):pass

def binarize(weight):
    blocks=weight.reshape(-1,64).float()
    alpha=blocks.abs().mean(dim=1,keepdim=True).half()
    return (torch.where(blocks>=0,1.,-1.)*alpha).half().reshape_as(weight)

cls=make_rewind_trainer_class(SFTTrainer)
native_loss=cls.compute_loss
native_linear=bnb.nn.Linear4bit.forward
active=set();binary={};names={};invocations=0

def linear(module,x,*args,**kwargs):
    if module not in active:return native_linear(module,x,*args,**kwargs)
    if args or kwargs:raise RuntimeError('Unexpected native Linear4bit call arguments')
    if module.bias is not None:raise RuntimeError('Unexpected bias in selected Qwen projection')
    return torch.nn.functional.linear(x.to(torch.float16),binary[module]).to(x.dtype)

def measured_loss(trainer,model,inputs,*args,**kwargs):
    global invocations
    invocations+=1
    if invocations!=1:raise RuntimeError('Expected exactly one native training-batch entry')
    assert model.training and int(trainer.state.global_step)==0
    assert trainer.args.fp16 and not trainer.args.bf16
    assert trainer.args.per_device_train_batch_size==4
    assert trainer.args.num_train_epochs==1 and trainer.args.gradient_accumulation_steps==1
    assert not trainer.args.gradient_checkpointing
    assert inputs['input_ids'].shape[0]==4
    report['batch']={'shape':list(inputs['input_ids'].shape),
      'input_ids_sha256':hashlib.sha256(inputs['input_ids'].detach().cpu().numpy().tobytes()).hexdigest(),
      'labels_sha256':hashlib.sha256(inputs['labels'].detach().cpu().numpy().tobytes()).hexdigest(),
      'supervised_shifted_tokens':int((inputs['labels'][:,1:]!=-100).sum().item()),
      'trainer_class':type(trainer).__name__,'model_training':model.training,
      'native_compute_loss_sha256':hashlib.sha256(__import__('inspect').getsource(native_loss).encode()).hexdigest()}
    report['trainable_parameters']={name:list(p.shape) for name,p in model.named_parameters() if p.requires_grad}
    assert report['trainable_parameters'] and all('lora_' in n for n in report['trainable_parameters'])
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    py_rng=random.getstate();np_rng=np.random.get_state()
    def evaluate(label,selected):
        active.clear();active.update(selected)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(py_rng);np.random.set_state(np_rng)
        cloned={k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in inputs.items()}
        with torch.no_grad():value=native_loss(trainer,model,cloned,*args,**kwargs)
        loss=value[0] if isinstance(value,tuple) else value
        actual=float(loss.detach().float().cpu());assert np.isfinite(actual)
        base=report['variants'][0]['loss'] if report['variants'] else actual
        gap=abs(actual-base)/max(abs(base),1e-12)
        report['variants'].append({'name':label,'loss':actual,'relative_loss_gap':gap,
          'within_1pct_diagnostic_threshold':gap<=.01,
          'binary_modules':[names[m] for m in selected],
          'quantized_weight_elements':sum(binary[m].numel() for m in selected),
          'not_an_official_check_result':True})
        torch.cuda.synchronize();save()
    report['status']='FIRST_BATCH_FORWARD_COMPARISONS';save()
    evaluate('native_nf4',[])
    prep=time.perf_counter()
    for name,module in model.named_modules():
        if isinstance(module,bnb.nn.Linear4bit):
            state=module.weight.quant_state
            assert state.quant_type=='nf4' and not module.weight.requires_grad
            dense=bnb.functional.dequantize_4bit(module.weight,quant_state=state)
            binary[module]=binarize(dense);names[module]=name
            del dense
    torch.cuda.synchronize();report['binary_precomputation_s']=time.perf_counter()-prep
    assert len(binary)==28*7
    bnb.nn.Linear4bit.forward=linear
    q0=[m for m in binary if '.layers.0.self_attn.q_proj.' in names[m]]
    assert len(q0)==1
    evaluate('one_q_projection_layer_0',q0)
    tail=[m for m in binary if any(f'.layers.{i}.' in names[m] for i in range(24,28))]
    assert len(tail)==4*7
    evaluate('last_four_decoder_layers',tail)
    evaluate('all_decoder_projections',list(binary))
    evaluate('native_nf4_repeat',[])
    assert report['variants'][0]['loss']==report['variants'][-1]['loss'], 'Native replay/RNG loss changed'
    report['status']='COMPLETE_FIRST_BATCH_DIAGNOSTIC';save()
    raise ProbeDone()

old_cwd=Path.cwd();old_argv=sys.argv[:]
try:
    assert torch.__version__=='2.11.0+cu128'
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4'
    task_path=REPO/'task/task.yaml';raw=task_path.read_bytes()
    report['fixed_task_file_sha256']=hashlib.sha256(raw).hexdigest()
    task=yaml.safe_load(raw);assert task['training']['quantization']=='4bit'
    assert task['training']['batch_size']==4 and task['training']['epochs']==1
    work=Path(tempfile.mkdtemp(prefix='onebit-first-batch-',dir='/kaggle/working'))
    task['data']['train']=str(REPO/'data/train.jsonl');task['output']=str(work/'output')
    config=work/'diagnostic.yaml';config.write_text(yaml.safe_dump(task,sort_keys=False))
    report['config_changes']=['Resolve same public training-file path','Separate diagnostic output directory']
    cls.compute_loss=measured_loss
    sys.modules['unsloth']=None;sys.modules['unsloth_zoo']=None
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['SOUP_NO_AUDIT_LOG']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    os.chdir(REPO);sys.argv=['soup','train','--config',str(config),'--yes']
    from soup_cli.cli import run
    try:run()
    except ProbeDone:pass
    assert report['status']=='COMPLETE_FIRST_BATCH_DIAGNOSTIC','Native first-batch probe did not complete'
    assert task_path.read_bytes()==raw,'Fixed task file changed'
except BaseException as exc:
    report['status']='FAILED_FIRST_BATCH_DIAGNOSTIC';report['error']=repr(exc);report['traceback']=traceback.format_exc()
finally:
    cls.compute_loss=native_loss;bnb.nn.Linear4bit.forward=native_linear
    active.clear();binary.clear();os.chdir(old_cwd);sys.argv=old_argv
    report['diagnostic_wall_s']=time.perf_counter()-START;save()
print(json.dumps({'status':report['status'],'variants':[(v['name'],v['loss']) for v in report['variants']],
                  'completed_training_steps':0,'benchmark_eligible':False}))
if report['status']!='COMPLETE_FIRST_BATCH_DIAGNOSTIC':raise RuntimeError(report.get('error','Incomplete diagnostic'))
