"""Actual native BNB half producers feeding unchanged existing MEM001 owner.

Component-only: no model installation, hooks, global function replacement,
matrix replay, supplied activation derivative or original-kernel claim.
"""
import hashlib,inspect
import torch
from bitsandbytes.nn import Linear4bit,Params4bit
from bitsandbytes.autograd._functions import MatMul4Bit
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers.activations import SiLUActivation
from .owned_half import _OwnedHalfActivation
from torch.nn.modules import module as native_module_hooks

EXPECTED={
    'linear_init':('bitsandbytes.nn.modules','Linear4bit.__init__','8940ba823ebc8bd0a8ad52b24ad972eb98eba7fb0f03b0a07a2bcb9220c6190f'),
    'linear_set_compute_type':('bitsandbytes.nn.modules','Linear4bit.set_compute_type','b262e0b994403aec2b52651093fef8025b5e63fa81570413a8d539abf8f2bf80'),
    'activation':('transformers.activations','SiLUActivation.forward','f9eaaa437c9f969e5f5c6f347b85a09fefcbae8824433dff69e0e83faa098863'),
    'mlp':('transformers.models.qwen2.modeling_qwen2','Qwen2MLP.forward','03df2d7685cf8446594448cbd33ef676a2b088779d99d9803abe3bed2ec7ad9b'),
    'linear':('bitsandbytes.nn.modules','Linear4bit.forward','d52cf63457717cc718ca0a449dad0967171e679be90d5d2adcafe2e8a141093d'),
    'matmul':('bitsandbytes.autograd._functions','MatMul4Bit.forward','29f4e21a4fe99db3f6e0f94dbf6cbe8426220f9163c0501c86d33f362a54ccb6'),
    'backward':('bitsandbytes.autograd._functions','MatMul4Bit.backward','053e8719319a526c61f288a37221ba5200de9eaa20dae7c7e45386315f284cd2'),
}


def current_sources():
    return {'linear_init':Linear4bit.__init__,'linear_set_compute_type':Linear4bit.set_compute_type,'activation':SiLUActivation.forward,'mlp':Qwen2MLP.forward,'linear':Linear4bit.forward,'matmul':MatMul4Bit.forward,'backward':MatMul4Bit.backward}


def verify_sources():
    values=current_sources()
    for key,fn in values.items():
        module,name,digest=EXPECTED[key]
        if (not inspect.isfunction(fn) or hasattr(fn,'__wrapped__') or (fn.__module__,fn.__qualname__)!=(module,name)
                or hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()!=digest):
            raise RuntimeError('Native MLP source/owner changed: '+key)
    return values


def canonical_module_dispatch():
    # Verify the canonical inherited owner, not a potentially foreign class
    # method snapshot. A preexisting foreign Module owner prevents preparation.
    module=torch.nn.Module;source=inspect.getsourcefile(module);result={}
    for name,qualname in (('__call__','Module._wrapped_call_impl'),('_call_impl','Module._call_impl')):
        fn=getattr(module,name)
        if (not inspect.isfunction(fn) or hasattr(fn,'__wrapped__')
                or fn.__module__!='torch.nn.modules.module' or fn.__qualname__!=qualname
                or fn.__code__.co_filename!=source or inspect.getsourcefile(fn)!=source):
            raise RuntimeError('Foreign canonical Module dispatch owner')
        result[name]=fn
    return result


def canonical_instance_dispatch(module,dispatch):
    return (all(getattr(torch.nn.Module,name) is fn and getattr(type(module),name) is fn
                and name not in module.__dict__ for name,fn in dispatch.items())
            and getattr(module,'_compiled_call_impl',None) is None)


def _owned_native_mlp(module,x):
    # MUST remain two independent source->half edges. Sharing a cast would
    # combine two branch gradients in half, unlike native FP32 source fan-in.
    gate16=module.gate_proj(x.to(torch.float16))
    up16=module.up_proj(x.to(torch.float16))
    if (type(gate16) is not torch.Tensor or type(up16) is not torch.Tensor
            or gate16.dtype!=torch.float16 or up16.dtype!=torch.float16
            or gate16.shape!=up16.shape or not gate16.is_contiguous() or not up16.is_contiguous()):
        raise RuntimeError('Verified native producer unexpectedly violated half output contract')
    # These are actual native outputs, not rounded arbitrary FP32 tensors.
    # The existing mathematical forward/backward owner remains byte-identical.
    activation=_OwnedHalfActivation.apply(gate16,up16)
    return module.down_proj(activation)


class NativeMLPHalfOwner:
    """Prepared source-bound component callable; fallback is unchanged module call."""
    def __init__(self,module,*,first_order_only=False):
        self.module=module;self.first_order_only=first_order_only
        self.sources=verify_sources();self.dispatch=canonical_module_dispatch();self.counts={'owned':0,'native':0};self.reasons={}
        self.identities={name:id(getattr(module,name,None)) for name in ('gate_proj','up_proj','down_proj','act_fn')}

    def reason(self,x):
        module=self.module
        if not self.first_order_only:return 'first_order_contract_absent'
        if type(module) is not Qwen2MLP:return 'foreign_mlp'
        if torch.jit.is_tracing() or torch.jit.is_scripting() or torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active():return 'tracing_or_transforms'
        if not all(canonical_instance_dispatch(child,self.dispatch) for child in (module,module.act_fn,module.gate_proj,module.up_proj,module.down_proj)):return 'foreign_module_dispatch'
        if 'forward' in module.__dict__ or current_sources()!=self.sources:return 'foreign_forward_owner'
        if any(id(getattr(module,n,None))!=i for n,i in self.identities.items()):return 'module_identity_changed'
        if type(module.act_fn) is not SiLUActivation or 'forward' in module.act_fn.__dict__:return 'activation'
        if module.hidden_size!=1536 or module.intermediate_size!=8960:return 'geometry'
        if type(x) is not torch.Tensor or x.device.type!='cuda' or x.dtype!=torch.float32:return 'input_type_device_dtype'
        if x.ndim not in (2,3) or x.shape[-1]!=1536 or not 1<=x.numel()//1536<=2048:return 'input_geometry'
        if x.ndim==3 and x.shape[0]!=4:return 'batch_geometry'
        if not x.is_contiguous() or x.is_neg() or x.is_conj():return 'input_layout'
        if torch.is_inference_mode_enabled() or not torch.is_grad_enabled() or not x.requires_grad:return 'no_training_gradient'
        if torch.autograd.forward_ad.unpack_dual(x).tangent is not None:return 'forward_ad'
        if torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')!=torch.float16:return 'autocast_dtype'
        for name,shape in (('gate_proj',(8960,1536)),('up_proj',(8960,1536)),('down_proj',(1536,8960))):
            projection=getattr(module,name)
            if type(projection) is not Linear4bit or 'forward' in projection.__dict__:return 'foreign_projection'
            if projection.bias is not None or projection.compute_dtype!=torch.float16 or not projection.compute_type_is_set:return 'projection_compute_or_bias'
            weight=projection.weight;state=getattr(weight,'quant_state',None)
            if (type(weight) is not Params4bit or weight.requires_grad or weight.device!=x.device
                    or weight.dtype!=torch.uint8 or not weight.bnb_quantized or state is None
                    or tuple(state.shape)!=shape or state.quant_type!='nf4' or state.blocksize!=64):return 'weight_contract'
        for name in ('_global_forward_pre_hooks','_global_forward_hooks','_global_backward_pre_hooks','_global_backward_hooks'):
            if getattr(native_module_hooks,name):return 'global_module_hooks'
        for child in module.modules():
            if child._forward_hooks or child._forward_pre_hooks or child._backward_hooks or child._backward_pre_hooks:return 'module_hooks'
        if any(p.requires_grad for p in module.parameters()):return 'trainable_mlp_parameter'
        return None

    def __call__(self,x):
        why=self.reason(x)
        if why is not None:
            self.counts['native']+=1;self.reasons[why]=self.reasons.get(why,0)+1
            return self.module(x)
        self.counts['owned']+=1
        return _owned_native_mlp(self.module,x)
