"""Real Soup factory/optimizer/Rewind composition on a small CPU fixture.

The cached head uses native statistics and the L1e kernel is interpreted;
native CPU dropout and FP16 autocast explicitly substitute for the CUDA path.
This is NOT full Soup setup/CLI, NF4 execution, GPU validation, or a benchmark.
"""

import copy
from functools import wraps
import hashlib
import json
import os

import pytest
import torch

from . import cached_head, integration, lora_dropout
from .test_integration import patched_classes
from .test_lora_dropout import _restore_globals


INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"


def _model():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(173)
    config=Qwen2Config(vocab_size=37,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        head_dim=8,max_position_embeddings=64,tie_word_embeddings=True,
        attention_dropout=0.,pad_token_id=0,eos_token_id=1)
    config._attn_implementation="sdpa"
    model=get_peft_model(Qwen2ForCausalLM(config),LoraConfig(
        r=16,lora_alpha=32,lora_dropout=.05,target_modules=["q_proj","v_proj"],
        bias="none",task_type="CAUSAL_LM"))
    gen=torch.Generator().manual_seed(271)
    with torch.no_grad():
        for name,param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape,generator=gen)*.02)
    return model


def _data_and_tokenizer():
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    vocab={"<pad>":0,"<eos>":1,"<unk>":2}
    vocab.update({f"tok{i}":i for i in range(3,37)})
    tok=Tokenizer(WordLevel(vocab=vocab,unk_token="<unk>"))
    tok.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token="<pad>",
        eos_token="<eos>",unk_token="<unk>",model_max_length=64,padding_side="right")
    gen=torch.Generator().manual_seed(731)
    rows=[]
    for i,length in enumerate((9,6,4,11,8,7,10,5,12,9,6,13)):
        ids=torch.randint(3,37,(length,),generator=gen).tolist()
        ids[0]=i+3
        prompt=length if i==2 else min(2+i%3,length-1)
        rows.append({"input_ids":ids,"assistant_masks":[0]*prompt+[1]*(length-prompt)})
    return Dataset.from_list(rows),tokenizer,rows


def run_factory_pair(root,monkeypatch, *, packed_metrics=None):
    """Run four genuine HF optimizer steps through Soup's actual factory."""
    from accelerate.utils.operations import convert_outputs_to_fp32
    from transformers import TrainerCallback
    from trl import SFTConfig,SFTTrainer
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.monitoring.rewind_log import read_rewind_log
    from . import metric_transfer

    root.mkdir(parents=True, exist_ok=True)
    if packed_metrics is None:
        packed_metrics = integration.PACKED_METRICS
    monkeypatch.setattr(integration, "PACKED_METRICS", packed_metrics)
    monkeypatch.setattr(integration, "HEAD_NUMERICS", "native")
    assert integration.patch_training_head()
    packet_bytes = []
    copy_packet = metric_transfer._copy_packet_to_cpu
    def packet(segments):
        packet_bytes.append(sum(value.numel() * value.element_size() for value in segments))
        return copy_packet(segments)
    monkeypatch.setattr(metric_transfer, "_copy_packet_to_cpu", packet)

    class TracedSFT(SFTTrainer):
        def __init__(self,*args,**kwargs):
            self.fixture_trace=[]
            super().__init__(*args,**kwargs)
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            loss,outputs=super().compute_loss(model,inputs,return_outputs=True,
                                               num_items_in_batch=num_items_in_batch)
            self.fixture_trace.append({
                "rows":(inputs["input_ids"][:,0]-3).tolist(),
                "loss":float(loss.detach()),
                "has_logits":getattr(outputs,"logits",None) is not None,
            })
            return (loss,outputs) if return_outputs else loss

    class Snapshots(TrainerCallback):
        def __init__(self):
            self.gradients=[]
            self.parameters=[]
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            self.gradients.append({n:p.grad.detach().clone() for n,p in model.named_parameters()
                                   if p.requires_grad and p.grad is not None})
        def on_step_end(self,args,state,control,model=None,**kwargs):
            self.parameters.append({n:p.detach().clone() for n,p in model.named_parameters()})

    # Use the genuine native bool-mask operator for BOTH sides.  Standard
    # CPU F.dropout has a different saved-node structure than CUDA dropout.
    _restore_globals(monkeypatch)
    native_calls=[]
    def native_dropout_forward(self,x):
        native_calls.append((tuple(x.shape),self.p,self.training))
        return torch.ops.aten.native_dropout(x,self.p,self.training)[0]
    monkeypatch.setattr(torch.nn.Dropout,"forward",native_dropout_forward)
    ds,tokenizer,rows=_data_and_tokenizer()
    initial=_model()
    models=(initial,copy.deepcopy(initial))
    pairs=[]
    handles=[]
    amp_entries=[[],[]]

    try:
        for index,(tag,model) in enumerate(zip(("baseline","candidate"),models)):
            output=root/tag
            output.mkdir()
            config=SoupConfig(base="local-fusion-fixture",data={"train":"local.jsonl","max_length":64},
                training={"quantization":"none","gradient_checkpointing":False},output=str(output))
            wrapper=SFTTrainerWrapper(config,device="cpu")
            wrapper.model=model
            wrapper.tokenizer=tokenizer
            unwrapped_forward=model.forward
            @wraps(unwrapped_forward)
            def amp_forward(*args,_forward=unwrapped_forward,_trace=amp_entries[index],**kwargs):
                _trace.append(torch.is_autocast_enabled("cpu"))
                with torch.autocast("cpu",dtype=torch.float16):
                    return _forward(*args,**kwargs)
            model.forward=convert_outputs_to_fp32(amp_forward)
            args=SFTConfig(output_dir=str(output),use_cpu=True,fp16=False,bf16=False,
                report_to=[],disable_tqdm=True,per_device_train_batch_size=3,
                gradient_accumulation_steps=1,max_steps=4,learning_rate=2e-4,
                optim="adamw_torch",lr_scheduler_type="constant",max_grad_norm=1.,
                logging_steps=1,save_strategy="no",seed=419,data_seed=419,
                gradient_checkpointing=False,packing=False,padding_free=False,
                activation_offloading=False,use_liger_kernel=False,
                dataset_kwargs={"skip_prepare_dataset":True},
                dataloader_num_workers=0,dataloader_pin_memory=False)
            snapshot=Snapshots()
            kwargs={"model":model,"args":args,"train_dataset":ds,
                    "processing_class":tokenizer,"callbacks":[snapshot]}
            before_config=config.model_dump()
            trainer=wrapper._build_rewind_trainer(TracedSFT,kwargs,rows=rows,
                output_dir=output,batch_size=3,grad_accum=1)
            assert config.model_dump()==before_config
            head={"forward":[],"backward":[]}
            def fwd(module,inputs,out,target=head):
                target["forward"].append((tuple(inputs[0].shape),tuple(out.shape),str(out.dtype)))
            def bwd(module,grad_input,grad_output,target=head):
                target["backward"].append((tuple(grad_input[0].shape),tuple(grad_output[0].shape),str(grad_output[0].dtype)))
            handles.extend((model.get_base_model().lm_head.register_forward_hook(fwd),
                            model.get_base_model().lm_head.register_full_backward_hook(bwd)))
            pairs.append((wrapper,trainer,snapshot,head))

        runs=[]
        reducer_calls=[]
        epilogue_calls=[]
        original_summaries=cached_head._summaries
        original_kernel=lora_dropout.soup_dropout_half_gradient_epilogue
        original_guard=integration._supported_causal_lm
        for index,(wrapper,trainer,snapshot,head) in enumerate(pairs):
            if index:
                before_rng=torch.get_rng_state().clone()
                monkeypatch.setattr(integration,"HEAD_MODE","cached")
                monkeypatch.setattr(integration,"CACHED_CE_CHUNK_TOKENS",7)
                monkeypatch.setattr(integration,"_supported_causal_lm",lambda owner,model,inputs,qwen:
                    models[1].get_base_model() if model is models[1] and model.training
                    else original_guard(owner,model,inputs,qwen))
                enabled=torch.is_autocast_enabled
                dtype=torch.get_autocast_dtype
                monkeypatch.setattr(torch,"is_autocast_enabled",lambda device="cuda":
                    enabled("cpu") if device=="cuda" else enabled(device))
                monkeypatch.setattr(torch,"get_autocast_dtype",lambda device:
                    torch.float16 if device=="cuda" else dtype(device))
                monkeypatch.setattr(lora_dropout,"_supported_device",lambda x:
                    x.device.type=="cpu" and enabled("cpu") and dtype("cpu")==torch.float16)
                assert lora_dropout.patch_lora_dropout()
                def summaries(*args,**kwargs):
                    reducer_calls.append(tuple(args[0].shape))
                    return original_summaries(*args,**kwargs)
                monkeypatch.setattr(cached_head,"_summaries",summaries)
                class CountKernel:
                    def __getitem__(self,grid):
                        launch=original_kernel[grid]
                        def call(*args,**kwargs):
                            epilogue_calls.append(grid)
                            return launch(*args,**kwargs)
                        return call
                monkeypatch.setattr(lora_dropout,"soup_dropout_half_gradient_epilogue",CountKernel())
                assert torch.equal(before_rng,torch.get_rng_state())
            native_calls.clear()
            torch.manual_seed(991)
            trainer.train()
            state,writer=wrapper._rewind_state,wrapper._rewind_log
            assert not state.failed and not state.dropped and not writer.dropped and not writer.disabled
            wanted={"loss","grad_norm","learning_rate","entropy","num_tokens",
                    "mean_token_accuracy","step","epoch"}
            logs=[{k:v for k,v in row.items() if k in wanted}
                  for row in trainer.state.log_history if "loss" in row]
            runs.append({"gradients":snapshot.gradients,"params":snapshot.parameters,
                "trace":trainer.fixture_trace,"head":head,"logs":logs,
                "rewind":read_rewind_log(writer.path),"rng":torch.get_rng_state().clone(),
                "dropouts":list(native_calls)})

        baseline,candidate=runs
        assert len(candidate["gradients"])==len(baseline["gradients"])==4
        assert len(candidate["params"])==len(baseline["params"])==4
        for kind in ("gradients","params"):
            for actual,reference in zip(candidate[kind],baseline[kind]):
                assert actual.keys()==reference.keys()
                for name in actual:
                    torch.testing.assert_close(actual[name],reference[name],rtol=0,atol=0)
        assert torch.equal(candidate["rng"],baseline["rng"])
        assert candidate["head"]==baseline["head"]
        assert len(candidate["head"]["forward"])==len(candidate["head"]["backward"])==4
        assert candidate["dropouts"]==baseline["dropouts"]
        assert len(candidate["dropouts"])==16
        assert len(reducer_calls)==4 and len(epilogue_calls)==8
        assert packet_bytes == ([16,36]*4 if packed_metrics else [])
        assert amp_entries==[[False]*4,[False]*4]
        assert all(not row["has_logits"] for row in candidate["trace"])
        assert all(row["has_logits"] for row in baseline["trace"])
        assert len(candidate["logs"])==len(baseline["logs"])==4
        metric_error={}
        for actual,reference in zip(candidate["logs"],baseline["logs"]):
            assert actual.keys()==reference.keys()
            for key in actual:
                delta=abs(actual[key]-reference[key])
                metric_error[key]=max(metric_error.get(key,0.),delta)
                assert delta<=1e-6
        assert len(candidate["rewind"].batches)==len(baseline["rewind"].batches)==4
        actual_rows=[]
        row_loss_error=0.
        for actual,reference in zip(candidate["rewind"].batches,baseline["rewind"].batches):
            assert (actual.step,actual.micro,actual.rows,actual.row_tokens)==(reference.step,reference.micro,reference.rows,reference.row_tokens)
            assert len(actual.row_loss)==len(reference.row_loss)
            actual_rows.extend(actual.rows)
            for x,y in zip(actual.row_loss,reference.row_loss):
                row_loss_error=max(row_loss_error,abs(x-y))
                assert abs(x-y)<=1e-6
        assert sorted(actual_rows)==list(range(12))
        assert actual_rows==[row for item in candidate["trace"] for row in item["rows"]]
        assert integration._HEAD_CALL.get() is None
        assert lora_dropout._ACTIVE_DROPOUT.get() is None
        def snapshot_hash(snapshots):
            digest=hashlib.sha256()
            for step in snapshots:
                for name,value in sorted(step.items()):
                    digest.update(name.encode())
                    digest.update(str((value.dtype,tuple(value.shape))).encode())
                    digest.update(value.detach().contiguous().view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()
        return {"passed":True,"physical_gpu_execution":False,"nf4_execution":False,
            "full_soup_cli":False,"actual_soup_factory":True,"optimizer_steps":4,
            "microbatches":4,"exact_gradient_snapshots":4,"exact_parameter_snapshots":4,
            "rng_exact":True,"native_dropout_calls":16,"l1e_kernel_backwards":8,
            "cached_head_summary_calls":4,"native_head_forwards":4,"native_head_backwards":4,
            "metrics_max_abs":metric_error,"rewind_row_loss_max_abs":row_loss_error,
            "packed_metrics":packed_metrics,"metric_packet_bytes":packet_bytes,
            "candidate_metric_logs":candidate["logs"],
            "candidate_rewind_records":[{"step":b.step,"micro":b.micro,"rows":b.rows,
                "row_tokens":b.row_tokens,"row_loss":b.row_loss} for b in candidate["rewind"].batches],
            "candidate_gradients_sha256":snapshot_hash(candidate["gradients"]),
            "candidate_parameters_sha256":snapshot_hash(candidate["params"]),
            "candidate_rng_sha256":hashlib.sha256(candidate["rng"].numpy().tobytes()).hexdigest(),
            "row_order":actual_rows,"tokens_exact":True,"outer_cpu_autocast_off":True,
            "test_substitutions":["CPU FP16 model autocast", "CPU native bool-mask dropout on both sides",
                                  "explicit head/LoRA device admission overrides", "Triton interpreter"]}
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.skipif(not INTERPRETING,reason="requires explicit TRITON_INTERPRET=1 CPU fixture")
def test_soup_factory_cached_head_and_l1e_four_updates(tmp_path,monkeypatch,patched_classes):
    result=run_factory_pair(tmp_path,monkeypatch)
    (tmp_path/"fusion_factory_result.json").write_text(json.dumps(result,indent=2)+"\n")
    assert result["passed"]


@pytest.mark.skipif(not INTERPRETING,reason="requires explicit TRITON_INTERPRET=1 CPU fixture")
def test_soup_factory_packed_metrics_exact_four_updates(tmp_path,monkeypatch,patched_classes):
    reports=[]
    for enabled in (False,True):
        with monkeypatch.context() as context:
            reports.append(run_factory_pair(tmp_path/str(enabled),context,packed_metrics=enabled))
    unpacked,packed=reports
    for field in ("candidate_metric_logs","candidate_rewind_records",
                  "candidate_gradients_sha256","candidate_parameters_sha256",
                  "candidate_rng_sha256","row_order"):
        assert packed[field] == unpacked[field], field
    assert unpacked["metric_packet_bytes"] == []
    assert packed["metric_packet_bytes"] == [16,36]*4
    (tmp_path/"packed_factory_parity.json").write_text(json.dumps({
        "passed":True,"physical_gpu_execution":False,"factory_steps_per_variant":4,
        "comparison":"same native cached head and L1e, packed metrics disabled versus enabled",
        "metrics_rewind_gradients_parameters_rng_exact":True,
        "unpacked":unpacked,"packed":packed},indent=2)+"\n")
