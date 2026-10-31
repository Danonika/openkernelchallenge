"""Narrow raw-pointer normalization for native built-in Parameter bias.

Research only. The original Parameter remains the input to native MatMul4Bit;
only its no-grad forward's kernel operand becomes a plain storage-sharing view.
There is no custom backward and no generic tensor-subclass admission.
"""
import inspect
import torch


def normalized_native_bias(bias):
    if bias is None or type(bias) is torch.Tensor:
        return bias,None
    if type(bias) is not torch.nn.Parameter:
        return None,'custom_bias_subclass'
    if torch.is_grad_enabled():
        return None,'parameter_bias_outside_native_nograd_forward'
    if bias.is_neg() or bias.is_conj() or not bias.is_contiguous():
        return None,'lazy_or_strided_parameter_bias'
    detach=torch.Tensor.detach
    if (not inspect.ismethoddescriptor(detach) or getattr(detach,'__objclass__',None) is not torch._C.TensorBase
            or getattr(detach,'__name__',None)!='detach' or hasattr(detach,'__wrapped__')):
        return None,'foreign_detach_owner'
    raw=detach(bias)
    if type(raw) is not torch.Tensor:
        return None,'unexpected_detach_result'
    return raw,None
