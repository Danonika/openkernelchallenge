"""Actual tiny native Qwen+PEFT+AMP branch capture, not fixed-task weights/data."""
import gc,hashlib,inspect
from pathlib import Path

def run(fused):
    import torch
    from transformers import AutoConfig,Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from peft import LoraConfig,TaskType,get_peft_model
    from accelerate import Accelerator
    from .branch_capture import BranchCapture,replay
    from .web_preflight import EXPECTED_QWEN_BLOB
    raw=Path(inspect.getfile(qwen)).read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==EXPECTED_QWEN_BLOB
    # Config only, exact revision and offline. No public model weights downloaded.
    config=AutoConfig.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
        revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',local_files_only=True)
    observed={k:getattr(config,k) for k in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','rms_norm_eps','vocab_size')}
    assert (config.hidden_size,config.intermediate_size,config.num_attention_heads,config.num_key_value_heads,config.rms_norm_eps)==(1536,8960,12,2,1e-6)
    # Explicit standalone reduction: native decoder dimension/ops retained, one
    # random layer and small vocabulary; this is not the fixed training task.
    config.num_hidden_layers=1;config.layer_types=['full_attention'];config.vocab_size=257
    config.tie_word_embeddings=True;config.use_cache=False;config.attention_dropout=0.
    config.eos_token_id=1;config.pad_token_id=0
    torch.manual_seed(144)
    base=Qwen2ForCausalLM(config).to('cuda')
    model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'))
    acc=Accelerator(mixed_precision='fp16');model=acc.prepare_model(model);model.train()
    layer=base.model.layers[0]
    assert type(layer) is qwen.Qwen2DecoderLayer and type(layer.post_attention_layernorm) is qwen.Qwen2RMSNorm
    weight=layer.post_attention_layernorm.weight
    assert weight.dtype==torch.float32 and not weight.requires_grad
    params={n:(id(p),p._version) for n,p in model.named_parameters()}
    rows=[]
    try:
        for scale in (1.,65536.):
            model.zero_grad(set_to_none=True);torch.manual_seed(145)
            acc.scaler=torch.amp.GradScaler('cuda',init_scale=scale,growth_interval=100)
            ids=torch.randint(2,257,(4,17),device='cuda');labels=ids.clone();labels[:,0]=-100
            with BranchCapture(layer) as capture:
                output=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False)
                assert torch.isfinite(output.loss)
                acc.backward(output.loss)
            capture.require()
            dtype_observed={k:str(capture.tensors[k].dtype) for k in ('a','b','z','y','output')}
            assert dtype_observed=={'a':'torch.float32','b':'torch.float16','z':'torch.float32','y':'torch.float32','output':'torch.float32'},dtype_observed
            rng=torch.cuda.get_rng_state().clone()
            result=replay(capture,weight,fused,scale)
            assert torch.equal(rng,torch.cuda.get_rng_state())
            assert params=={n:(id(p),p._version) for n,p in model.named_parameters()}
            result.update(scale=scale,actual_native_qwen_decoder=True,actual_peft_qv_lora=True,
                actual_accelerate_fp16_and_gradscaler=True,native_branch_dtype_observed=dtype_observed,
                fixed_task_training=False,nf4_decoder=False,random_tiny_model=True,
                model_parameter_identity_and_versions_unchanged=True,replay_rng_unchanged=True)
            rows.append(result)
            del output,capture,ids,labels
        for module in layer.modules():
            assert not module._forward_hooks and not module._forward_pre_hooks
        return {'pinned_base_config_dimensions':observed,'fixture_overrides':{'num_hidden_layers':1,'vocab_size':257,'B':4,'T':17},'rows':rows,
                'proof_scope':'Native Qwen module/PEFT/AMP incoming branch gradients from tiny random model; not pretrained NF4 fixed-task activation distribution or parameter update equivalence.'}
    finally:
        model.zero_grad(set_to_none=True);acc.free_memory();del model,base;gc.collect();torch.cuda.empty_cache()
