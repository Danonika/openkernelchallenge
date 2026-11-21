"""Share one read-only forward half allocation, not its backward fan-in.

Only the source-pinned, hook-free, frozen native MLP owner calls this helper.
Its gate/up projections consume inputs without writing them. The first branch
keeps the ordinary native cast; the second has a distinct FP16->FP32 adjoint
edge. No retained activation, matrix replay, changed GEMM or kernel credit.
"""
import torch
_COUNTS={'forward_calls':0,'backward_calls':0,'shared_storage_calls':0}


class IndependentAliasCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, detached_half):
        if (type(source) is not torch.Tensor or type(detached_half) is not torch.Tensor
                or source.dtype!=torch.float32 or detached_half.dtype!=torch.float16
                or source.device!=detached_half.device or source.shape!=detached_half.shape
                or not source.is_contiguous() or not detached_half.is_contiguous()
                or detached_half.requires_grad):
            raise RuntimeError('Independent half-cast edge contract changed')
        # Returning a separate view leaves a separate autograd edge to source.
        # Do not save the buffer: native cast backward needs only the dtype.
        result=detached_half.view_as(detached_half)
        _COUNTS['forward_calls']+=1
        return result

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():raise RuntimeError('Independent cast is first-order only')
        if grad.dtype!=torch.float16:raise RuntimeError('Independent half adjoint required')
        result=grad.float()
        _COUNTS['backward_calls']+=1
        return result,None


def independent_half_inputs(source):
    first=source.to(torch.float16)
    second=IndependentAliasCast.apply(source,first.detach())
    if first is second or first.data_ptr()!=second.data_ptr():
        raise RuntimeError('Independent cast failed shared storage / separate Tensor contract')
    _COUNTS['shared_storage_calls']+=1
    return first,second


def status():
    return dict(_COUNTS,original_kernel_credit=0)
