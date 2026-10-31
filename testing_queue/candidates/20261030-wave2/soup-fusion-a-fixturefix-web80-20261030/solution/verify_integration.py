"""Prepared future T4 composition gate. Not executed during notebook preparation.

Covers real PEFT NF4+LoRA interaction and the unchanged real-Qwen CE boundary
fixture in one process with all three hooks. This is not full-task validation.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import traceback
import torch
import solution


def exact(actual, expected, label):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype,label+": metadata"
    assert torch.equal(actual.isnan(),expected.isnan()),label+": NaN class"
    assert torch.equal(actual.isposinf(),expected.isposinf()),label+": +Inf class"
    assert torch.equal(actual.isneginf(),expected.isneginf()),label+": -Inf class"
    finite=actual.isfinite()&expected.isfinite()
    assert torch.equal(actual[finite],expected[finite]),label+": finite values"
    zeros=finite&(actual==0)&(expected==0)
    assert torch.equal(actual.signbit()[zeros],expected.signbit()[zeros]),label+": signed zero"


@contextmanager
def native_decode_boundaries(ops,functional,forward,dequant):
    installed=ops._dequant_linear_fallback,functional.dequantize_4bit
    ops._dequant_linear_fallback,functional.dequantize_4bit=forward,dequant
    try: yield
    finally: ops._dequant_linear_fallback,functional.dequantize_4bit=installed


def run(report):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import bitsandbytes.backends.cuda.ops as ops
    from bitsandbytes.autograd._functions import MatMul4Bit
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit
    from peft.tuners.tuners_utils import cast_adapter_dtype
    from .nf4_component import routes
    from .lora_component import epilogue
    from .ce_component.verify_integration import run as verify_ce
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=="Tesla T4"
    assert torch.cuda.get_device_capability(0)==(7,5)
    native_fwd,native_fun=ops._dequant_linear_fallback,functional.dequantize_4bit
    native_lora,native_backward=Linear4bit.forward,MatMul4Bit.backward
    assert solution.apply() is True
    torch.manual_seed(91307)
    config=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,init_lora_weights=False,
                      target_modules=["q_proj","v_proj"],bias="none")
    layers=[]
    for width in (1536,256):
        base=bnb.nn.Linear4bit(1536,width,bias=False,compute_dtype=torch.float16,
                              compress_statistics=True,quant_type="nf4")
        base.weight=bnb.nn.Params4bit(torch.randn(width,1536,dtype=torch.float32)*.02,
                                     requires_grad=False,compress_statistics=True,
                                     quant_type="nf4",blocksize=64)
        base=base.cuda()
        # Pinned PEFT0.20.0 requires the config object, not historical constructor kwargs.
        layer=Linear4bit(base,"default",config=config,r=16,lora_alpha=32).cuda().train()
        # Direct construction bypasses PeftModel.__init__'s native adapter upcast.
        # Match Soup's get_peft_model(..., autocast_adapter_dtype=True) default;
        # never relax the actual task's FP32 adapter precondition.
        cast_adapter_dtype(layer, adapter_name="default", autocast_adapter_dtype=True)
        assert layer.lora_A["default"].weight.dtype==torch.float32
        assert layer.lora_B["default"].weight.dtype==torch.float32
        assert layer.lora_dropout["default"].p==.05
        layers.append(layer)
    data=torch.randn(4,89,1536,device="cuda",dtype=torch.float16)
    rng_cpu,rng_cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
    native_state={f"{index}:{name}":value.detach().clone() for index,layer in enumerate(layers) for name,value in layer.named_parameters()}
    def evaluate(candidate):
        for layer in layers:layer.zero_grad(set_to_none=True)
        x=data.clone().requires_grad_(True);masks=[]
        handles=[layer.lora_dropout["default"].register_forward_hook(
            lambda module,args,out:masks.append((out!=0).clone())) for layer in layers]
        try:
            with torch.autocast("cuda",dtype=torch.float16):
                outputs=[layer(x) if candidate else native_lora(layer,x) for layer in layers]
                loss=sum(value.float().square().mean() for value in outputs)
            loss.backward(torch.tensor(65536.,device="cuda",dtype=torch.float32))
            return {"outputs":[v.detach().clone() for v in outputs],"loss":loss.detach().clone(),
                    "input_grad":x.grad.clone(),"grads":[p.grad.clone() for layer in layers for p in layer.parameters() if p.requires_grad],
                    "masks":masks,"cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state()}
        finally:
            for hook in handles:hook.remove()
    with native_decode_boundaries(ops,functional,native_fwd,native_fun): reference=evaluate(False)
    torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda)
    candidate=evaluate(True)
    for a,b in zip(candidate["outputs"],reference["outputs"]):exact(a,b,"output")
    exact(candidate["loss"],reference["loss"],"loss")
    exact(candidate["input_grad"],reference["input_grad"],"input gradient")
    for a,b in zip(candidate["grads"],reference["grads"]):exact(a,b,"adapter gradient")
    for a,b in zip(candidate["masks"],reference["masks"]):assert torch.equal(a,b)
    assert not torch.equal(candidate["masks"][0],candidate["masks"][1])
    assert torch.equal(candidate["cpu_rng"],reference["cpu_rng"])
    assert torch.equal(candidate["cuda_rng"],reference["cuda_rng"])
    assert routes._COUNTS["forward"]>0 and routes._COUNTS["functional"]>0
    assert epilogue._COUNTS["launch_submitted"]==2
    assert MatMul4Bit.backward is native_backward
    for index,layer in enumerate(layers):
        for name,value in layer.named_parameters():
            assert torch.equal(value.detach().view(torch.uint8),native_state[f"{index}:{name}"].view(torch.uint8))
    report["actual_nf4_lora_pair"]={"passed":True,"nf4_routes":dict(routes._COUNTS),
        "lora_routes":dict(epilogue._COUNTS),"outputs_scaled_gradients_exact":True,
        "native_qv_dropout_and_rng_preserved":True,"native_bnb_backward_identity":True,
        "scope":"Real quantized q/v-shaped layers at M356; no optimizer or full Qwen training"}
    solution.restore()
    assert ops._dequant_linear_fallback is native_fwd and functional.dequantize_4bit is native_fun
    assert Linear4bit.forward is native_lora
    del candidate,reference,layers,native_state,data
    # Reuses the original, fixed-gate real-Qwen fixture. Its solution.apply()
    # installs all three hooks again; the actual NF4+LoRA activation is established
    # by the preceding nonvacuous pair, not inferred from this dense tiny model.
    ce_report={};verify_ce(ce_report)
    report["actual_ce_pair_under_composite_install"]=ce_report
    assert ce_report["status"]=="passed_gpu_integration_preflight"
    report["status"]="passed_gpu_integration_preflight"


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parent
    report={"status":"running","score":None,"admission":"UNKNOWN",
            "source_sha256":{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob("*.py"))},
            "scope":"Prepared composite integration gate, not full-task timing/share/quality"}
    try:run(report)
    except Exception as error:report.update(status="failed",error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
    finally:solution.restore()
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"out":str(args.out)}),flush=True)
    return 0 if report["status"]=="passed_gpu_integration_preflight" else 1


if __name__=="__main__":raise SystemExit(main())
