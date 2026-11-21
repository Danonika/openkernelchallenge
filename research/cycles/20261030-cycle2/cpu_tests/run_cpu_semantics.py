#!/usr/bin/env python3
"""Original, bounded CPU semantic witnesses; NOT a CUDA/bnb/kernel benchmark.

Uses only synthetic tensors and Torch operations; no models, data, trainer,
kernel source, challenge harness or challenge tests are imported or modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.manual_seed(20261030)
ROOT = Path(__file__).resolve().parent
RESULTS = []


def error(a, b):
    a, b = a.double(), b.double()
    return {"max_abs": (a-b).abs().max().item() if a.numel() else 0.0,
            "relative_l2": ((a-b).norm()/b.norm().clamp_min(1e-30)).item()}


def check_close(a, b, *, atol=2e-5, rtol=2e-5):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=True)


def case(name, ids, evidence, fn):
    try:
        details = fn()
        result = dict(name=name, candidates=ids, evidence=evidence,
                      status="PASS", details=details)
    except Exception as exc:
        result = dict(name=name, candidates=ids, evidence=evidence,
                      status="FAIL", error=repr(exc), traceback=traceback.format_exc())
    RESULTS.append(result)
    print(json.dumps(result, allow_nan=False), flush=True)


def shift(labels):
    return torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), 1)


def dense_stats(z16, labels, attention, denominator):
    z = z16.float()
    target = shift(labels)
    valid = target.ne(-100)
    ce = F.cross_entropy(z.flatten(0, 1), target.flatten(),
                         reduction="none", ignore_index=-100).view_as(target)
    lp = z.log_softmax(-1)
    ent = -(lp.exp()*lp).sum(-1)
    hits = (z.argmax(-1).eq(target) & valid).sum()
    counts = valid.sum(-1)
    return dict(loss=ce.sum()/denominator,
                entropy=(ent*attention).sum()/attention.sum(),
                accuracy=hits.float()/valid.sum().clamp_min(1),
                row_loss=ce.sum(-1)/counts.clamp_min(1), counts=counts,
                token_entropy=ent, target=target)


def centered_chunks(z16, labels, attention, denominator, chunk_size):
    # Independent row-reduction formulation. Counts and normalization are global.
    target = shift(labels).flatten()
    z = z16.flatten(0, 1).float()
    ce, ent, pred = [], [], []
    for start in range(0, z.shape[0], chunk_size):
        t = target[start:start+chunk_size]
        q = z[start:start+chunk_size]
        m = q.max(-1).values
        centered = q-m[:, None]
        exp = centered.exp()
        total = exp.sum(-1)
        logs = total.log()
        target_value = centered.gather(-1, t.clamp_min(0)[:, None]).squeeze(-1)
        ce.append(torch.where(t.ne(-100), logs-target_value, 0))
        ent.append(logs-(exp*centered).sum(-1)/total)
        pred.append(q.argmax(-1))
    ce = torch.cat(ce).view_as(labels)
    ent = torch.cat(ent).view_as(labels)
    pred = torch.cat(pred).view_as(labels)
    target = target.view_as(labels)
    valid = target.ne(-100)
    count = valid.sum(-1)
    return dict(loss=ce.sum()/denominator,
                entropy=(ent*attention).sum()/attention.sum(),
                accuracy=(pred.eq(target)&valid).sum().float()/valid.sum().clamp_min(1),
                row_loss=ce.sum(-1)/count.clamp_min(1), counts=count,
                token_entropy=ent, target=target)


def head_masks():
    max_error = 0.0
    trials = 0
    for t in (1, 7, 31, 127, 128, 129):
        # Strided logits exercise token/vocabulary views without any model.
        z = (torch.randn(3, t, 38)*3).half()[..., ::2]
        labels = torch.randint(0, 19, (3, t))
        labels[0] = -100
        labels[1, ::3] = -100
        labels[2, :max(1,t//2)] = -100
        attention = torch.ones(3, t)
        if t > 1:
            attention[1, -1] = 0
            labels[1, -1] = -100
        denominator = max(1, shift(labels).ne(-100).sum().item())+5
        ref = dense_stats(z, labels, attention, denominator)
        for chunk in (1, 7, 32, 128):
            got = centered_chunks(z, labels, attention, denominator, chunk)
            for key in ("loss", "entropy", "accuracy", "row_loss", "token_entropy"):
                check_close(got[key], ref[key])
                max_error = max(max_error, error(got[key], ref[key])["max_abs"])
            assert torch.equal(got["counts"], ref["counts"])
            assert torch.equal(got["target"], ref["target"])
            assert got["counts"][0].item() == 0 and got["row_loss"][0].item() == 0
            trials += 1
    # Explicit first-index tie and prompt/final entropy witnesses.
    z = torch.zeros(1, 4, 5).half()
    z[0, 1, 0] = 8
    labels = torch.tensor([[-100, -100, 0, -100]])
    a = torch.ones(1, 4)
    out = dense_stats(z, labels, a, 7)
    assert torch.equal(z.argmax(-1), torch.zeros(1, 4, dtype=torch.long))
    supervised_only_entropy = out["token_entropy"][shift(labels).ne(-100)].mean()
    assert abs(out["entropy"].item()-supervised_only_entropy.item()) > 1
    valid = shift(labels).ne(-100)
    ce = F.cross_entropy(z.float().flatten(0,1), shift(labels).flatten(),
                         reduction="none", ignore_index=-100)
    correct_global = ce.sum()/7
    illegal_local = ce[valid.flatten()].mean()
    assert abs(correct_global.item()-illegal_local.item()) > 1e-4
    ignored = torch.full((1, 4), -100)
    mean_loss = F.cross_entropy(z.float().flatten(0,1), ignored.flatten(), ignore_index=-100)
    assert torch.isnan(mean_loss)
    assert F.cross_entropy(z.float().flatten(0,1), ignored.flatten(),
                           reduction="sum", ignore_index=-100).item() == 0
    return dict(trials=trials, max_abs_error=max_error, exact_counts_and_row_mapping=True,
                first_argmax_index=0, full_attention_entropy=out["entropy"].item(),
                wrong_supervised_only_entropy=supervised_only_entropy.item(),
                correct_global_denominator_loss=correct_global.item(),
                wrong_local_denominator_loss=illegal_local.item(),
                all_ignored_mean="NaN preserved", all_ignored_sum=0,
                largest_logits_shape=[3,129,19])


def centered_reductions():
    examples = []
    for vals in ([65504.0]*19, [32768.0,32752.0], [0.0, -16.0], [4., 3., -9.]):
        z = torch.tensor([vals]).half().float()
        lp = z.log_softmax(-1)
        ref_ce = F.cross_entropy(z, torch.tensor([0]))
        ref_ent = -(lp.exp()*lp).sum(-1)
        lse = z.logsumexp(-1)
        unsafe_ce = lse-z[:,0]
        unsafe_ent = lse-(z.softmax(-1)*z).sum(-1)
        m = z.max(-1).values
        q = z-m[:,None]
        e = q.exp(); s = e.sum(-1); logs = s.log()
        safe_ce = logs-q[:,0]
        safe_ent = logs-(e*q).sum(-1)/s
        unsafe_p = (z-lse[:,None]).exp()
        safe_p = e/s[:,None]
        check_close(safe_ce.squeeze(), ref_ce, atol=1e-7, rtol=2e-6)
        check_close(safe_ent, ref_ent, atol=1e-6, rtol=2e-6)
        check_close(safe_p, z.softmax(-1), atol=1e-7, rtol=2e-6)
        examples.append(dict(logits=vals, baseline_ce=ref_ce.item(),
                             single_lse_ce=unsafe_ce.item(), centered_ce=safe_ce.item(),
                             baseline_entropy=ref_ent.item(),
                             single_lse_entropy=unsafe_ent.item(), centered_entropy=safe_ent.item(),
                             single_lse_probability_sum=unsafe_p.sum().item(),
                             centered_probability_sum=safe_p.sum().item()))
    assert abs(examples[0]["single_lse_entropy"]-examples[0]["baseline_entropy"]) > 1e-3
    assert examples[1]["single_lse_ce"] == 0 and examples[1]["baseline_ce"] > 0
    return dict(counterexample_established=True, examples=examples,
                decision="Reject single-LSE-only state for high-offset logits; retain max plus centered logsum state.")


def scaled_gradient():
    w = torch.tensor([[0.], [-20.]], dtype=torch.float32)
    values = []
    for scale in (.25,1.,3.,128.,65536.):
        h = torch.ones(1,1, requires_grad=True)
        loss = F.cross_entropy((h.half()@w.half().T).float(), torch.tensor([0]))
        (loss*scale).backward()
        z = h.detach().half()@w.half().T
        d = z.float().softmax(-1)-torch.tensor([[1.,0.]])
        correct = ((d*scale).half()@w.half()).float()
        wrong = (d.half()@w.half()).float()*scale
        check_close(correct, h.grad, atol=0, rtol=0)
        values.append(dict(incoming_scale=scale, autograd_hidden_grad=h.grad.item(),
                           scale_before_cast=correct.item(), scale_after_cast=wrong.item()))
    assert values[-1]["scale_before_cast"] != 0 and values[-1]["scale_after_cast"] == 0
    return dict(counterexample_established=True, cases=values,
                actual_cuda_gradscaler_executed=False,
                decision="Incoming gradient must scale FP32 dlogits before the FP16 cast.")


def compact_head():
    trials=[]
    for pattern in ("mixed", "all_supervised", "all_ignored"):
        for scale in (.25, 1., 3., 128., 65536.):
            h0 = torch.randn(2,7,13)*.1
            w = torch.randn(19,13)*.2
            labels = torch.randint(0,19,(2,7))
            if pattern == "mixed":
                labels[0,:4] = -100; labels[1,::2] = -100
            elif pattern == "all_ignored": labels[:] = -100
            target = shift(labels).flatten()
            use = target.ne(-100)
            denominator = max(1,use.sum().item())+5
            h = h0.clone().requires_grad_()
            logits = h.half()@w.half().T
            loss = F.cross_entropy(logits.float().flatten(0,1), target,
                                   reduction="sum", ignore_index=-100)/denominator
            (loss*scale).backward()
            # Only selected FINAL hidden rows need head-backward GEMM.
            selected = h0.flatten(0,1)[use]
            compact = selected.half()@w.half().T
            d = compact.float().softmax(-1)
            d[torch.arange(use.sum()), target[use]] -= 1
            d = ((d/denominator)*scale).half()
            scattered = torch.zeros_like(h0).flatten(0,1)
            scattered[use] = (d@w.half()).float()
            check_close(scattered.view_as(h0), h.grad, atol=0, rtol=0)
            assert torch.count_nonzero(h.grad.flatten(0,1)[~use]).item() == 0
            trials.append(dict(mask=pattern, scale=scale, selected_rows=use.sum().item(),
                               total_rows=target.numel(), denominator=denominator,
                               gradient_error=error(scattered.view_as(h0),h.grad)))
    # Tiny causal mixing proves earlier prompt activations may still have nonzero gradients.
    x=torch.tensor([[[.2,.3],[.1,.4],[.5,.2]]],requires_grad=True)
    causal=x.cumsum(1)
    loss=F.cross_entropy((causal@torch.tensor([[1.,-1.],[.5,.2]])).float()[:,1],torch.tensor([0]))
    causal.retain_grad(); loss.backward()
    assert causal.grad[0,0].abs().sum() == 0 and x.grad[0,0].abs().sum() > 0
    return dict(trials=trials, exact_match_all_trials=True,
                prompt_final_hidden_grad=causal.grad[0,0].tolist(),
                earlier_prompt_activation_grad=x.grad[0,0].tolist(),
                decision="Only compact head backward; retain all-token metric forward and complete transformer backprop.")


def norm_forward(x,w,eps=1e-6):
    xf=x.float()
    r=(xf.square().mean(-1,keepdim=True)+eps).rsqrt()
    return (xf*r).to(x.dtype)*w, r


def norm_backward():
    records=[]
    for width in (1,7,13,1536):
        for magnitude in (1e-8,1.,1e8):
            # Last-axis stride two is intentional.
            x=(torch.randn(3,width*2)*magnitude)[:,::2].detach().requires_grad_()
            w=torch.randn(width)
            dy=torch.randn_like(x)
            y,r=norm_forward(x,w)
            ref=torch.autograd.grad(y,x,dy)[0]
            g=dy*w
            got=r*g-x*r.pow(3)*(g*x).mean(-1,keepdim=True)
            check_close(got,ref,atol=4e-4,rtol=3e-5)
            records.append(dict(width=width,input_magnitude=magnitude,
                                input_stride=list(x.stride()),error=error(got,ref)))
    a=torch.randn(3,13,requires_grad=True)
    b=torch.randn(3,13,requires_grad=True)
    w=torch.randn(13); dn=torch.randn_like(a); dbypass=torch.randn_like(a)
    residual=a+b; y,r=norm_forward(residual,w)
    da,db=torch.autograd.grad((y,residual),(a,b),(dn,dbypass))
    g=dn*w
    own=r*g-residual*r.pow(3)*(g*residual).mean(-1,keepdim=True)+dbypass
    check_close(own,da);check_close(own,db)
    # Correct low-precision formula also rounds the incoming normalized-value gradient.
    x=(torch.randn(3,13)*.1).half().requires_grad_();w=torch.randn(13);dy=torch.randn(3,13)
    y,r=norm_forward(x,w);ref=torch.autograd.grad(y,x,dy)[0]
    g=(dy*w).half().float();xf=x.float()
    got=(r*g-xf*r.pow(3)*(g*xf).mean(-1,keepdim=True)).half()
    check_close(got,ref,atol=.001,rtol=.001)
    wrong_forward=((xf*r)*w).half().float()
    assert torch.count_nonzero(y.detach()-wrong_forward).item()>0
    x32=torch.randn(3,13);w32=torch.randn(13);y32,_=norm_forward(x32,w32)
    assert torch.count_nonzero(y32-y32.half().float()).item()>0
    return dict(fp32_trials=records, residual_bypass_error=error(own,da),
                fp16_backward_error=error(got,ref),
                wrong_cast_order_forward_error=error(wrong_forward,y.detach()),
                premature_fp16_output_error=error(y32.half().float(),y32),
                decision="Keep FP32 output for FP32 input; preserve cast-before-affine and both residual branches.")


def nf4_rounding():
    # Exact binary-representable synthetic code/scale/offset. This is NOT a real
    # bitsandbytes QuantState. Endpoint NF4 value +1 is sufficient for the witness.
    a=torch.tensor(-.5001220703125);b=torch.tensor(2.00048828125);offset=torch.tensor(1.501220703125)
    separate=a*b+offset
    fused=(a.double()*b.double()+offset.double()).float()
    assert separate != fused and separate.half() != fused.half()
    # Check all nibble identities, 64-element block boundaries and 256-scale
    # superblock boundaries, including odd packed tail. Synthetic lookup tables
    # intentionally avoid any assertion of real bnb quantizer compatibility.
    n=16451
    index=torch.arange(n)
    code=(index%16).to(torch.uint8)
    padded=F.pad(code,(0,n%2))
    packed=(padded[::2]<<4)|padded[1::2]
    unpacked=torch.where(index%2==0,packed[index//2]>>4,packed[index//2]&15).long()
    assert torch.equal(unpacked,code.long())
    lut=torch.linspace(-1,1,16);lut[7]=0
    scale_lut=torch.linspace(-1,1,256)
    block=index//64
    codes=torch.arange(math.ceil(n/64))%256
    group_scale=torch.tensor([.125,.25])
    centered=scale_lut[codes]*group_scale[torch.arange(len(codes))//256]
    scales=centered+1
    decoded=(lut[unpacked]*scales[block]).half()
    scalar=[]
    for i in range(n):
        byte=int(packed[i//2])
        nibble=byte>>4 if i%2==0 else byte&15
        block_i=i//64
        scale=scale_lut[codes[block_i]]*group_scale[block_i//256]
        scale=scale+torch.tensor(1.)
        scalar.append((lut[nibble]*scale).half())
    scalar=torch.stack(scalar)
    assert torch.equal(decoded,scalar)
    return dict(counterexample_established=True,
                synthetic_nested_code=a.item(), second_scale=b.item(), offset=offset.item(),
                separate_fp32_scale=separate.item(), single_round_fp32_scale=fused.item(),
                separate_fp16_weight=separate.half().item(), single_round_fp16_weight=fused.half().item(),
                tested_elements=n, all_16_nibbles=True, all_256_synthetic_scale_codes=True,
                boundaries=[63,64,16383,16384], odd_tail=True, nibble_decode_exact=True,
                actual_bnb_compatibility_validated=False,
                decision="Do not contract nested scale multiplication plus offset into FMA without real-state validation.")


def lora_adjoint():
    details=[]
    for mode in ("normal","subnormal","overflow"):
        s0=torch.randn(17,16)*.2;b0=torch.randn(23,16)*.2
        g=torch.randn(17,23)*.1
        if mode=="subnormal":
            g.fill_(3*2.**-27);s0.fill_(1);b0.fill_(1)
        elif mode=="overflow":g.fill_(40000.)
        s=s0.clone().requires_grad_();b=b0.clone().requires_grad_();base=torch.zeros(17,23,requires_grad=True)
        z=s.half()@b.half().T
        result=base+z*2.
        result.backward(g)
        d=(g.half()*2.).half()
        ds=(d@b0.half()).float();db=(d.T@s0.half()).float()
        check_close(ds,s.grad,atol=0,rtol=0);check_close(db,b.grad,atol=0,rtol=0)
        check_close(base.grad,g,atol=0,rtol=0)
        illegal_d=(g*2.).half()
        direct_fp32_db=d.float().T@s0.half().float()
        if mode=="normal":assert torch.count_nonzero(direct_fp32_db-db).item()>0
        if mode=="subnormal":assert torch.count_nonzero(illegal_d-d).item()>0
        details.append(dict(mode=mode, oracle_adjoint_exact=True, base_gradient_exact=True,
                            d16_nonfinite=torch.count_nonzero(~torch.isfinite(d)).item(),
                            grad_b_nonfinite=torch.count_nonzero(~torch.isfinite(db)).item(),
                            wrong_precast_scale_diff_count=torch.count_nonzero(illegal_d!=d).item(),
                            wrong_unrounded_fp32_grad_b_diff_count=torch.count_nonzero(direct_fp32_db!=db).item(),
                            grad_b_error=None if mode=="overflow" else error(direct_fp32_db,db)))
    return dict(shape=dict(M=17,N=23,rank=16), alpha_over_rank=2.,
                dropout="not invoked; deterministic no-dropout boundary fixture only", cases=details,
                decision="A shared D16 adjoint tile must preserve G32→G16→scale→D16 and FP16 parameter-gradient rounding.")


def pack_bool(mask):
    flat=mask.reshape(-1).to(torch.uint8)
    flat=F.pad(flat,(0,(-flat.numel())%8))
    shifts=torch.arange(8,dtype=torch.uint8)
    packed=(flat.view(-1,8)<<shifts).sum(-1).to(torch.uint8)
    return ("synthetic_bool_pack",packed,tuple(mask.shape))


def unpack_bool(packet):
    _,packed,shape=packet
    shifts=torch.arange(8,dtype=torch.uint8)
    flat=((packed[:,None]>>shifts)&1).flatten()
    return flat[:math.prod(shape)].bool().reshape(shape)


def native_dropout_mask_pack():
    records=[]
    for n in (1,7,8,9,257):
        for train in (True,False):
            data=torch.linspace(.1,1.,n)
            upstream_q=torch.linspace(.2,.6,n)
            upstream_v=torch.linspace(-.7,.3,n)
            runs=[]
            for packed_run in (False,True):
                torch.manual_seed(8100+n)
                x=data.clone().requires_grad_()
                masks=[]
                sizes=[]
                def pack(t):
                    if t.dtype != torch.bool: return t
                    packet=pack_bool(t)
                    sizes.append((t.numel(),packet[1].numel()))
                    # Clone is TEST instrumentation only, not memory-saving implementation.
                    masks.append(t.detach().clone())
                    return packet
                def unpack(t):
                    return unpack_bool(t) if isinstance(t,tuple) else t
                if packed_run:
                    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
                        q=torch.ops.aten.native_dropout.default(x,.05,train)[0]
                        v=torch.ops.aten.native_dropout.default(x,.05,train)[0]
                else:
                    q=torch.ops.aten.native_dropout.default(x,.05,train)[0]
                    v=torch.ops.aten.native_dropout.default(x,.05,train)[0]
                torch.autograd.backward((q,v),(upstream_q,upstream_v))
                runs.append(dict(q=q.detach(),v=v.detach(),grad=x.grad,
                                 rng=torch.get_rng_state(),sizes=sizes,masks=masks))
            base,got=runs
            for k in ("q","v","grad","rng"):
                assert torch.equal(base[k],got[k]), k
            assert len(got["sizes"])==2
            if train and n==257:
                assert not torch.equal(got["masks"][0],got["masks"][1])
            records.append(dict(length=n,training=train,outputs_gradients_rng_exact=True,
                                saved_bool_masks=len(got["sizes"]),
                                raw_mask_payload_bytes=sum(a for a,b in got["sizes"]),
                                packed_mask_payload_bytes=sum(b for a,b in got["sizes"])))
    m=(torch.arange(36).view(4,9)%3==0)[:,::2]
    assert not m.is_contiguous() and torch.equal(unpack_bool(pack_bool(m)),m)
    return dict(cases=records,strided_roundtrip_exact=True,
                dispatch="Explicit torch.ops.aten.native_dropout on CPU only",
                memory_measurement=False,
                debug_mask_clones_retained=True,
                actual_kaggle_nn_dropout_dispatch_validated=False,
                decision="Lossless bool-mask representation verified only; inspect real CUDA dropout dispatch and measure pack/unpack cost before promotion.")


case("head_causal_masks_global_denominator_metrics",["SOUP-NEW-001","SOUP-NEW-002"],"CPU behavioral fixture",head_masks)
case("centered_reductions_reject_single_lse",["SOUP-NEW-001","SOUP-NEW-002"],"CPU numerical counterexample",centered_reductions)
case("scaled_gradient_rounding_boundary",["SOUP-NEW-001","SOUP-NEW-002"],"CPU numerical counterexample, not CUDA GradScaler",scaled_gradient)
case("supervised_only_head_backward",["SOUP-TRAIN-001"],"CPU compact/scatter autograd fixture",compact_head)
case("cast_aware_norm_and_residual_bypass",["cast-aware-norm"],"CPU analytic vs autograd fixture",norm_backward)
case("nf4_separate_scale_offset_and_packing",["SOUP-NEW-003","SOUP-NEW-007"],"Synthetic CPU data-transform witness, not bnb validation",nf4_rounding)
case("lora_b_adjoint_cast_boundaries",["SOUP-LORA-001","LoRA-scale-add"],"CPU deterministic no-dropout autograd fixture",lora_adjoint)
case("native_dropout_lossless_saved_bool_mask",["SOUP-LORA-002"],"CPU explicit native_dropout saved-tensor fixture",native_dropout_mask_pack)

payload=dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
             scope="Cloud-only original toy-tensor numerical/semantic tests",
             environment=dict(python=sys.version.split()[0],torch=torch.__version__,torch_git_version=torch.version.git_version,
                              cuda_available=torch.cuda.is_available(),device="cpu",
                              threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                              platform=platform.platform()),
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             summary=dict(passed=sum(r["status"]=="PASS" for r in RESULTS),
                          failed=sum(r["status"]=="FAIL" for r in RESULTS)),
             not_validated=["actual GPU kernels","T4 performance","actual bnb QuantState or dispatch",
                            "CUDA GradScaler overflow/optimizer/scheduler behavior","CUDA/Kaggle dropout dispatch and RNG equivalence",
                            "full model training/quality","competition kernel-share or scoring"],results=RESULTS)
(ROOT/"results.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
summary={k:payload[k] for k in ("scope","generated_utc","environment","script_sha256","summary","not_validated")}
summary["outcomes"]=[dict(name=r["name"],candidates=r["candidates"],status=r["status"],
                          evidence=r["evidence"],
                          decision=r.get("details",{}).get("decision","Preserve causal masks, global normalization, row association and full-attention metrics.")
                          if r["status"]=="PASS" else r["error"]) for r in RESULTS]
(ROOT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
print(json.dumps(payload["summary"]))
sys.exit(bool(payload["summary"]["failed"]))
