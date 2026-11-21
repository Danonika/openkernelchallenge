"""Frozen own v4 kernel reference used only by isolated target preflight."""
import torch, triton
from .fused_kernel import soup_nf4_mma_tiled_reuse


def original_reference_forward(data):
    a,packed,state,bias=data
    n,k=state.shape;m=a.numel()//k
    result=torch.empty((*a.shape[:-1],n),device=a.device,dtype=torch.float16)
    soup_nf4_mma_tiled_reuse[(triton.cdiv(m,64),triton.cdiv(n,64))](
        a,packed,state.absmax,state.state2.code,state.state2.absmax,state.offset,
        a if bias is None else bias,result,m,n,k,bias is not None,
        num_warps=4,num_stages=1,enable_fp_fusion=False)
    return result
