"""Explicit future T4 gate, not run during preparation. Small fixture models only.

python -m solution.gpu_preflight --out /new/preflight/path
All source/versions, numerical boundary and PTX checks fail closed. There is no
score/admission here. Kernel execution is a preflight separate from cold training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(status="FAILED",gpu_executed=False,score=None,admission="UNKNOWN",checks=[])
    try:
        if os.environ.get("TRITON_INTERPRET"):
            raise RuntimeError("GPU preflight refuses interpreter mode")
        import torch,triton,transformers
        from . import rotary
        from .cpu_contract import inputs,reference,exact
        if triton.__version__ != "3.6.0" or torch.__version__ != "2.11.0+cu128" or transformers.__version__ != "5.16.1":
            raise RuntimeError("Exact target Torch/Triton/Transformers versions required")
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!="Tesla T4":
            raise RuntimeError("One visible Tesla T4 required")
        os.environ["SOUP_ROTARY_CODEGEN_DIR"]=str(out/"ptx")
        # Original callable stays intact until all arithmetic tests have run.
        for b,t,cb in ((1,1,1),(2,17,2),(4,511,1),(4,512,4)):
            q,k,c,s=inputs(b=b,t=t,cb=cb,device="cuda")
            with torch.autocast("cuda",dtype=torch.float16):
                expected=reference(q,k,c,s)
                actual=rotary.fused_apply_rotary_pos_emb(q,k,c,s)
            gs=tuple(torch.linspace(-65536,65536,x.numel(),device="cuda").reshape(x.shape) for x in actual)
            eg=torch.autograd.grad(expected,(q,k),gs)
            ag=torch.autograd.grad(actual,(q,k),gs)
            for aa,ee in zip((*actual,*ag),(*expected,*eg)):exact(aa,ee)
            for y,x in zip(actual,(q,k)):assert y.stride()==x.stride()
            result["checks"].append(dict(name="scaled_boundary",b=b,t=t,coefficient_batch=cb))
        # Exhaustive signed-zero arithmetic; call one-input kernel boundary only,
        # without broadening the production dispatcher beyond D128/H12:2.
        import itertools
        v=torch.tensor(list(itertools.product([0.,-0.,1.,-1.],repeat=6)),device="cuda")
        q=torch.zeros((1,len(v),1,2),device="cuda").transpose(1,2).requires_grad_()
        k=q.detach().clone().requires_grad_()
        c,s=v[:,:2].reshape(1,len(v),2).contiguous(),v[:,2:4].reshape(1,len(v),2).contiguous()
        g=v[:,4:].reshape_as(q)
        e=reference(q,k,c,s)
        actual=tuple(rotary._FrozenRoPE.apply(x,c,s) for x in (q,k))
        eg=torch.autograd.grad(e,(q,k),(g,g));ag=torch.autograd.grad(actual,(q,k),(g,g))
        for aa,ee in zip((*actual,*ag),(*e,*eg)):exact(aa,ee)
        result["checks"].append(dict(name="exhaustive_signed_zero",combinations=len(v)))
        # First-order nonfinite classes/subnormal arithmetic: no NaN payload claim.
        v=torch.tensor([0.,-0.,float("inf"),float("-inf"),float("nan"),1e-40,-1e-40,1.],device="cuda")
        q=v.repeat(8).reshape(1,8,1,8).transpose(1,2).requires_grad_();k=q.detach().clone().requires_grad_()
        c=v.repeat(8).reshape(1,8,8).contiguous();s=c.flip(-1).contiguous()
        e=reference(q,k,c,s);actual=tuple(rotary._FrozenRoPE.apply(x,c,s) for x in (q,k))
        eg=torch.autograd.grad(e,(q,k),(q.detach(),k.detach()));ag=torch.autograd.grad(actual,(q,k),(q.detach(),k.detach()))
        for aa,ee in zip((*actual,*ag),(*e,*eg)):exact(aa,ee)
        result["checks"].append(dict(name="nonfinite_classes_subnormal_and_signed_zero"))
        # Exercise unchanged SDPA autocast/GQA with FP32 q/k inputs, both mask routes.
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from types import SimpleNamespace
        for masked in (False, True):
            q,k,c,s=inputs(b=2,t=17,device="cuda")
            v=torch.randn((2,17,2,128),device="cuda").transpose(1,2).requires_grad_()
            owner=SimpleNamespace(num_key_value_groups=6,is_causal=True)
            mask=torch.ones((2,1,17,17),device="cuda",dtype=torch.bool).tril() if masked else None
            def attention(fn):
                with torch.autocast("cuda",dtype=torch.float16):
                    qr,kr=fn(q,k,c,s)
                    return sdpa_attention_forward(owner,qr,kr,v,mask,dropout=0.,scaling=128**-0.5)[0]
            rng=torch.cuda.get_rng_state().clone()
            e,actual=attention(reference),attention(rotary.fused_apply_rotary_pos_emb)
            exact(actual,e);assert actual.dtype==torch.float16
            assert torch.equal(rng,torch.cuda.get_rng_state())
            g=torch.ones_like(e)*64
            eg=torch.autograd.grad(e,(q,k,v),g);ag=torch.autograd.grad(actual,(q,k,v),g)
            for aa,ee in zip(ag,eg):torch.testing.assert_close(aa,ee,rtol=2e-5,atol=2e-5,equal_nan=True)
            result["checks"].append(dict(name="unchanged_sdpa_autocast_gqa",masked=masked))
        # Unchanged real Qwen2Attention SDPA boundary, no pretrained model load.
        from transformers.models.qwen2 import modeling_qwen2 as mod
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        torch.manual_seed(47132)
        cfg=Qwen2Config(hidden_size=1536,num_attention_heads=12,num_key_value_heads=2,
                       intermediate_size=32,num_hidden_layers=1,attention_dropout=0.,
                       rope_parameters={"rope_type":"default","rope_theta":1_000_000.})
        cfg._attn_implementation="sdpa"
        module=mod.Qwen2Attention(cfg,0).cuda().train()
        emb=mod.Qwen2RotaryEmbedding(cfg).cuda()
        for masked in (False,True):
            hidden=torch.randn((2,17,1536),device="cuda",requires_grad=True)
            coeff=emb(hidden,torch.arange(17,device="cuda").unsqueeze(0))
            mask=torch.ones((2,1,17,17),device="cuda",dtype=torch.bool).tril() if masked else None
            variables=(hidden,*module.parameters());upstream=torch.randn_like(hidden)*64
            original=mod.apply_rotary_pos_emb
            rng=torch.cuda.get_rng_state().clone()
            e=module(hidden,coeff,mask)[0];eg=torch.autograd.grad(e,variables,upstream)
            try:
                assert rotary.patch_qwen2_rotary()
                count=rotary._COUNTS["forward_launch_returned"]
                actual=module(hidden,coeff,mask)[0];ag=torch.autograd.grad(actual,variables,upstream)
                assert rotary._COUNTS["forward_launch_returned"]==count+2
            finally:
                mod.apply_rotary_pos_emb=original
            assert torch.equal(rng,torch.cuda.get_rng_state())
            exact(actual,e)
            # Upstream shared-ancestry scheduling and SDPA reductions can differ;
            # raw scaled gradients are compared, not divided to hide a mismatch.
            for aa,ee in zip(ag,eg):torch.testing.assert_close(aa,ee,rtol=2e-5,atol=2e-5,equal_nan=True)
            result["checks"].append(dict(name="real_qwen2_sdpa_route",masked=masked))
        torch.cuda.synchronize();result["gpu_executed"]=True
        ptxfiles=sorted((out/"ptx").glob("*.ptx"));assert ptxfiles
        ptxchecks=[]
        for path in ptxfiles:
            text=path.read_text()
            assert not re.search(r"\bfma(?:\.|\s)",text),path.name
            assert not re.search(r"\b(?:mul|add)\.[^\n;]*ftz",text),path.name
            assert re.search(r"\bmul(?:\.rn)?\.f32",text),path.name
            assert re.search(r"\badd(?:\.rn)?\.f32",text),path.name
            ptxchecks.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        result.update(status="PASSED_PREFLIGHT_ONLY",ptx_checks=ptxchecks,counts=dict(rotary._COUNTS))
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        (out/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["status"]=="PASSED_PREFLIGHT_ONLY" else 1)

if __name__=="__main__":main()
