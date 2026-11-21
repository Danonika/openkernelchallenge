"""Read-only actual-Qwen weight study; not training or a benchmark submission."""
import time
START=time.perf_counter()
import gc,json,os,traceback
from pathlib import Path
import numpy as np
import torch
import bitsandbytes as bnb
from huggingface_hub import snapshot_download
from safetensors import safe_open
from codec import analyze,self_check,NF4,error_summary

MODEL='Qwen/Qwen2.5-1.5B-Instruct'
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
OUT=Path(os.environ['SOUP_RESULTS'])/'onebit_weight_probe.json'
report={'status':'RUNNING','scope':'ACTUAL_WEIGHT_COMPONENT_PROBE_NOT_TRAINING_OR_BENCHMARK',
 'model':MODEL,'revision':REVISION,'model_outputs_evaluated':False,'official_admission':'UNKNOWN',
 'original_kernel_credit':0,'samples':[],'matrix_multiply_probes':[],
 'limitations':['Weight error budgets are not training/eval loss tolerances',
 'No full-model quality, completed 64-step task, speed or target claim',
 'Only one sign/scale/exception codec; not an impossibility result',
 'NF4 scale metadata is shared and excluded on both sides of storage comparison',
 'All preparation belongs to this diagnostic; no precomputed model tensors are exported']}

def save():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+'\n')

def selected_groups(count):
    return np.unique(np.linspace(0,count-1,min(count,2048),dtype=np.int64))

def source_file(snapshot,name):
    index=snapshot/'model.safetensors.index.json'
    if index.exists():
        file=json.loads(index.read_text())['weight_map'][name]
        if Path(file).name!=file:raise ValueError('Unexpected shard filename')
        return snapshot/file
    file=snapshot/'model.safetensors'
    if not file.is_file():raise RuntimeError('Prepared public safetensors missing')
    return file

try:
    assert torch.__version__=='2.11.0+cu128','Pinned Torch required'
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4'
    torch.set_num_threads(2)
    report['self_check']=self_check();report['device']=torch.cuda.get_device_name(0)
    snapshot=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    config=json.loads((snapshot/'config.json').read_text())
    assert (config['hidden_size'],config['num_hidden_layers'],config['vocab_size'])==(1536,28,151936)
    names=[f'model.layers.{i}.{suffix}.weight' for i in (0,13,27)
           for suffix in ('self_attn.q_proj','mlp.down_proj')]
    names+=['model.layers.13.self_attn.v_proj.weight','model.layers.0.mlp.up_proj.weight']
    for name in names:
        if time.perf_counter()-START>180:raise RuntimeError('Component probe time budget exhausted')
        with safe_open(source_file(snapshot,name),framework='pt',device='cpu') as f:
            original=f.get_tensor(name).to(dtype=torch.float16).contiguous()
        weight=original.to('cuda');del original
        packed,state=bnb.functional.quantize_4bit(weight,blocksize=64,compress_statistics=True,quant_type='nf4')
        native=bnb.functional.dequantize_4bit(packed,quant_state=state)
        indices=selected_groups(native.numel()//64);index=torch.tensor(indices,device='cuda')
        raw=packed.reshape(-1,32).index_select(0,index).cpu().numpy()
        codes=np.empty((len(indices),64),dtype=np.uint8);codes[:,::2]=raw>>4;codes[:,1::2]=raw&15
        scales=bnb.functional.dequantize_blockwise(state.absmax,quant_state=state.state2)+state.offset
        chosen_scales=scales.index_select(0,index).cpu().numpy()
        selected_native=native.reshape(-1,64).index_select(0,index).cpu().numpy()
        codebook=state.code.cpu().numpy().astype(np.float32)
        reconstructed=(codebook[codes]*chosen_scales[:,None]).astype(np.float16)
        assert np.array_equal(reconstructed.view(np.uint16),selected_native.view(np.uint16)), 'NF4 reference reconstruction mismatch'
        started=time.perf_counter();analysis=analyze(codes,4,codebook)
        report['samples'].append({'tensor':name,'shape':list(weight.shape),'sampled_blocks':len(indices),
          'sample_selection':'Evenly spaced aligned blocks across the complete tensor',
          'reference':'Native bitsandbytes whole-tensor NF4 with double quantization, decoded FP16',
          'reference_reconstruction_bit_exact':True,'codec_preparation_and_analysis_s':time.perf_counter()-started,
          'analysis':analysis})
        if name in (names[0],names[3]):
            torch.manual_seed(17)
            x=torch.randn((32,native.shape[1]),device='cuda',dtype=torch.float16)
            blocks=native.reshape(-1,64).float()
            alpha=blocks.abs().mean(dim=1,keepdim=True).half()
            binary=(torch.where(blocks>=0,1.,-1.)*alpha).half().reshape_as(native)
            with torch.no_grad():
                base=torch.nn.functional.linear(x,native)
                changed=torch.nn.functional.linear(x,binary)
            torch.cuda.synchronize()
            report['matrix_multiply_probes'].append({'tensor':name,'inputs':'Synthetic normal FP16, not task activations',
              'error':error_summary(changed.float().cpu().numpy(),base.float().cpu().numpy()),
              'scope':'Projection sensitivity only; no speed or task-quality claim'})
            del x,blocks,alpha,binary,base,changed
        del weight,packed,state,native,index,scales
        gc.collect();save()
    name='model.embed_tokens.weight'
    with safe_open(source_file(snapshot,name),framework='pt',device='cpu') as f:
        view=f.get_slice(name);shape=view.get_shape()
        rows=[0,shape[0]//2,shape[0]-32]
        embedding=torch.cat([view[r:r+32,:].to(dtype=torch.float16) for r in rows]).numpy().reshape(-1,64)
    indices=selected_groups(len(embedding));embedding=embedding[indices]
    started=time.perf_counter()
    report['samples'].append({'tensor':name,'shape':shape,'sampled_blocks':len(indices),
       'sample_selection':'Evenly spaced blocks from three 32-row windows: first, middle, last',
       'reference':'Original public checkpoint values retained in FP16; no extra NF4 quantization',
       'analysis':analyze(embedding,16),'codec_preparation_and_analysis_s':time.perf_counter()-started})
    torch.cuda.synchronize();report['status']='COMPLETE_COMPONENT_STUDY'
except BaseException as exc:
    report['status']='FAILED_COMPONENT_STUDY';report['error']=repr(exc)
    report['traceback']=traceback.format_exc()
finally:
    report['diagnostic_wall_s']=time.perf_counter()-START;save()
print(json.dumps({'status':report['status'],'samples':len(report['samples']),
                  'diagnostic_wall_s':report['diagnostic_wall_s'],'not_a_benchmark':True}))
if report['status']!='COMPLETE_COMPONENT_STUDY':raise RuntimeError(report.get('error','Incomplete component probe'))
