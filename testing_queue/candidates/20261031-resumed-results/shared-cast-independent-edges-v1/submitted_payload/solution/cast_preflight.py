"""Additional blocking target-only check on the already-loaded native MLP.

No model load, parameter mutation, installer replacement or relaxed comparison.
Reference is the verified parent composition's two independent input casts and
unchanged starter primitive. Candidate uses the real installed owner.
"""
import torch
from .mlp_half import starter_half as half
from .mlp_half.independent_cast import independent_half_inputs
from .compact_mlp_install import _InstalledGuard


def _exact(actual,expected,name):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
        raise RuntimeError('Cast-edge '+name+' shape/dtype changed')
    for pred in (torch.isnan,torch.isposinf,torch.isneginf):
        if not torch.equal(pred(actual),pred(expected)):
            raise RuntimeError('Cast-edge '+name+' nonfinite class changed')
    finite=actual.isfinite()&expected.isfinite()
    if not torch.equal(actual[finite].contiguous().view(torch.uint8),expected[finite].contiguous().view(torch.uint8)):
        raise RuntimeError('Cast-edge '+name+' finite bits changed')


def check_loaded_native_mlp(module,installation,scale):
    device=module.gate_proj.weight.device
    if device.type!='cuda':raise RuntimeError('Actual CUDA native MLP required')
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(device)
    parameters=tuple((id(p),p._version,p.data_ptr(),p.requires_grad) for p in module.parameters())
    x=((torch.arange(4*3*1536,device=device,dtype=torch.float32)%257)-128).reshape(4,3,1536)/512
    x.requires_grad_(True)
    reason=_InstalledGuard(module,installation).reason(x)
    if reason is not None:raise RuntimeError('Native cast-edge target guard declined: '+reason)
    a,b=independent_half_inputs(x)
    if (a is b or a.data_ptr()!=b.data_ptr() or a.grad_fn is None or b.grad_fn is None
            or a.grad_fn is b.grad_fn):
        raise RuntimeError('Cast-edge target storage/backward identity changed')
    del a,b
    answers=[]
    for candidate in (False,True):
        value=x.detach().clone().requires_grad_(True)
        with torch.autocast('cuda',dtype=torch.float16):
            if candidate:
                result=installation.call(module,value)
            else:
                gate=module.gate_proj(value.to(torch.float16))
                up=module.up_proj(value.to(torch.float16))
                result=module.down_proj(half.OwnedStarterHalf.apply(gate,up))
        dy=((torch.arange(result.numel(),device=device,dtype=torch.float32)%113)-56).reshape(result.shape)*(float(scale)/65536)
        gradient,=torch.autograd.grad(result,value,dy)
        answers.append((result.detach(),gradient.detach()))
        if not candidate:del gate,up
        del value,result,gradient,dy
    _exact(answers[1][0],answers[0][0],'output')
    _exact(answers[1][1],answers[0][1],'FP32 input gradient')
    if (parameters!=tuple((id(p),p._version,p.data_ptr(),p.requires_grad) for p in module.parameters())
            or not torch.equal(cpu_rng,torch.get_rng_state())
            or not torch.equal(cuda_rng,torch.cuda.get_rng_state(device))):
        raise RuntimeError('Cast-edge native probe mutated parameters or RNG')
    return {'status':'PASS_ACTUAL_NATIVE_MLP_CAST_EDGE_ONLY','scale':float(scale),
            'shape':[4,3,1536],'actual_pinned_native_guard':True,
            'reference':'verified_parent_two_independent_casts_same_starter_GEMM',
            'shared_half_storage_distinct_backward_edges':True,
            'output_and_FP32_input_gradient_exact_finite_bits_nonfinite_classes':True,
            'parameter_identity_version_pointer_and_RNG_unchanged':True,
            'manual_incoming_scaled_cotangent_not_separate_GradScaler_claim':True}
