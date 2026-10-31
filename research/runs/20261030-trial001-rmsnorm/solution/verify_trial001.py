"""Bounded GPU fixture gate. No training data/model weights or measurement changes.

The original expanded Torch graph is the oracle. Extreme arithmetic witnesses are
reported separately, not silently discarded and not universal-equivalence claims.
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch

from solution.rmsnorm import frozen_rms32, patch_qwen2_norm, supported


def reference(x, w, eps=1e-6):
    dtype = x.dtype
    z = x.float()
    z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)
    return w * z.to(dtype)


def gap(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    af, ef = actual.float(), expected.float()
    classes_equal = (torch.equal(torch.isnan(af), torch.isnan(ef))
                     and torch.equal(torch.isposinf(af), torch.isposinf(ef))
                     and torch.equal(torch.isneginf(af), torch.isneginf(ef)))
    finite = torch.isfinite(af) & torch.isfinite(ef)
    if finite.any():
        a, e = af[finite].double(), ef[finite].double()
        diff = (a - e).abs()
        max_abs = float(diff.max())
        ref_max = float(e.abs().max())
        rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(e).clamp_min(1e-30))
        exact = torch.equal(a, e)
    else:
        max_abs = ref_max = rel_l2 = 0.0
        exact = True
    return {'class_match': classes_equal, 'finite_exact': exact, 'max_abs': max_abs,
            'ref_max': ref_max, 'relative_l2': rel_l2,
            'pass': classes_equal and max_abs <= 2e-5 * max(1., ref_max) and rel_l2 <= 2e-5}


def pair(shape, magnitude, grad_scale, branched=False, zero=False, unit_weight=False):
    a = torch.randn(shape, device='cuda', dtype=torch.float32) * magnitude
    if zero:
        a.zero_()
    w = torch.randn(shape[-1], device='cuda', dtype=torch.float32) * .1 + 1
    if unit_weight: w.fill_(1)
    dy = torch.randn_like(a) * grad_scale
    bypass = torch.randn_like(a) * grad_scale if branched else None
    x0, x1 = a.clone().requires_grad_(), a.clone().requires_grad_()
    y0, y1 = reference(x0, w), frozen_rms32(x1, w)
    outputs0, outputs1 = (y0,), (y1,)
    grads = (dy,)
    if branched:
        outputs0, outputs1 = (y0, x0), (y1, x1)
        grads = (dy, bypass)
    dx0, = torch.autograd.grad(outputs0, x0, grads)
    dx1, = torch.autograd.grad(outputs1, x1, grads)
    forward, backward = gap(y1, y0), gap(dx1, dx0)
    forward['pass'] = forward['class_match'] and torch.allclose(y1,y0,rtol=1e-4,atol=1e-5)
    backward['pass'] = backward['class_match'] and torch.allclose(
        dx1/grad_scale,dx0/grad_scale,rtol=1e-3,atol=2e-5)
    return {'shape': list(shape), 'magnitude': magnitude, 'grad_scale': grad_scale,
            'branched': branched, 'zero': zero, 'unit_weight':unit_weight,
            'forward': forward, 'backward': backward,
            'inputs_unmodified': torch.equal(x0.detach(), a) and torch.equal(x1.detach(), a)}


def extreme_witnesses():
    results = []
    w = torch.ones(1536, device='cuda')
    for value, incoming, bypass_value in ((3e38, 1., None), (1024., 2.**34, 1.)):
        xs = [torch.full((2,1536), value, device='cuda', requires_grad=True) for _ in range(2)]
        ys = [reference(xs[0], w), frozen_rms32(xs[1], w)]
        derivatives = []
        for x,y in zip(xs,ys):
            if bypass_value is None:
                g, = torch.autograd.grad(y, x, torch.full_like(y, incoming))
            else:
                g, = torch.autograd.grad((y,x), x, (torch.full_like(y,incoming),torch.full_like(x,bypass_value)))
            derivatives.append(g)
        results.append({'x': value, 'dy': incoming, 'bypass': bypass_value,
                        'forward': gap(ys[1],ys[0]), 'backward': gap(derivatives[1],derivatives[0]),
                        'reference_first': str(derivatives[0].flatten()[0].item()),
                        'trial_first': str(derivatives[1].flatten()[0].item()),
                        'classification': 'diagnostic extreme; not a universal-bitwise gate'})
    return results


def routing_checks():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    original = Qwen2RMSNorm.forward
    patch_qwen2_norm()
    first = Qwen2RMSNorm.forward
    patch_qwen2_norm()
    assert Qwen2RMSNorm.forward is first, 'patch must be idempotent'
    records = []
    try:
        for dtype, width, trainable, strided, on_cpu in (
            (torch.float32,1536,False,False,False),
            (torch.float16,1536,False,False,False),
            (torch.float32,1536,True,False,False),
            (torch.float32,1535,False,False,False),
            (torch.float32,1536,False,True,False),
            (torch.float32,1536,False,False,True)):
            dev = 'cpu' if on_cpu else 'cuda'
            mod = Qwen2RMSNorm(width).to(device=dev,dtype=dtype)
            mod.weight.requires_grad_(trainable)
            x = torch.randn((2,width*2 if strided else width), device=dev,dtype=dtype)
            if strided: x=x[:,::2]
            x.requires_grad_()
            expected = original(mod,x)
            actual = mod(x)
            assert actual.dtype == expected.dtype
            accepted = supported(x,mod.weight) and width==1536
            if accepted:
                assert gap(actual,expected)['pass']
            else:
                assert torch.equal(actual,expected), 'fallback must retain original callable'
            (actual.float().sum()).backward()
            if trainable: assert mod.weight.grad is not None
            records.append({'dtype':str(dtype),'width':width,'trainable_weight':trainable,
                            'strided':strided,'cpu':on_cpu,'specialization':accepted,'pass':True})
    finally:
        Qwen2RMSNorm.forward = original
    # Saved-tensor version protection and no-grad state economy are real contracts.
    x=torch.randn(2,1536,device='cuda',requires_grad=True);w=torch.ones(1536,device='cuda')
    y=frozen_rms32(x,w)
    with torch.no_grad(): x.add_(1)
    try:
        y.sum().backward()
    except RuntimeError as exc:
        assert 'modified by an inplace operation' in str(exc)
    else:
        raise AssertionError('saved tensor mutation was not detected')
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        frozen_rms32(torch.randn(2,1536,device='cuda'),w)
        with torch.no_grad():
            y=frozen_rms32(torch.randn(2,1536,device='cuda',requires_grad=True),w)
            assert not y.requires_grad
    assert not saved, 'no-grad input must not save backward state'
    return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    start=time.monotonic();torch.manual_seed(73001)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert 'T4' in torch.cuda.get_device_name(0)
    cases=[]
    for shape in ((1,1536),(7,1536),(4,512,1536)):
        for magnitude in (.01,1.,10.):
            for grad_scale in (1.,128.,65536.):
                for branched in (False,True):
                    cases.append(pair(shape,magnitude,grad_scale,branched=branched,
                                      unit_weight=magnitude==.01))
    for h in (7,1535,1536):
        cases.append(pair((3,h),1.,1.,zero=True))
        cases.append(pair((3,h),1.,1.))
    realistic_ok=all(c['forward']['pass'] and c['backward']['pass'] and c['inputs_unmodified'] for c in cases)
    extremes=extreme_witnesses()
    routing=routing_checks()
    report={'status':'passed' if realistic_ok else 'failed','scope':'narrow GPU fixtures; not admission',
            'torch':torch.__version__,'cuda_device':torch.cuda.get_device_name(0),
            'tolerances':{'forward_allclose':{'rtol':1e-4,'atol':1e-5},
                          'backward_allclose_after_dividing_both_by_upstream_scale':{'rtol':1e-3,'atol':2e-5},
                          'finite_nonfinite_classes':'exact'},
            'realistic_cases':cases,'extreme_diagnostics':extremes,'routing':routing,
            'saved_tensor_version_protection':True,'no_grad_saves_nothing':True,
            'elapsed_s':time.monotonic()-start}
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('realistic_cases','routing')},indent=2),flush=True)
    assert realistic_ok, 'narrow realistic forward/backward oracle gate failed; no full comparison'


if __name__=='__main__':
    main()
