from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class Summaries:
    entropy: torch.Tensor
    token_nll: torch.Tensor
    predictions: torch.Tensor

def _check(h,w,labels,chunk,denominator):
    assert type(h) is torch.Tensor and type(w) in (torch.Tensor,torch.nn.Parameter) and type(labels) is torch.Tensor
    assert h.device==w.device==labels.device and h.device.type in ('cpu','cuda')
    if h.device.type=='cuda':
        assert h.shape[-1]==1536 and w.shape==(151936,1536), 'Task-specific CUDA geometry'
        assert torch.cuda.get_device_capability(h.device)==(7,5), 'T4 component only'
        assert torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.float16, 'Native AMP FP16 boundary only'
    assert h.dtype==w.dtype==torch.float32 and labels.dtype==torch.int64
    assert h.ndim==3 and w.ndim==2 and labels.shape==h.shape[:2] and h.shape[-1]==w.shape[1]
    assert h.is_contiguous() and w.is_contiguous() and labels.is_contiguous()
    assert not any(x.is_neg() or x.is_conj() for x in (h,w,labels))
    assert not w.requires_grad,'Frozen head only: no vocabulary/weight-gradient accumulation proof'
    assert h.untyped_storage().data_ptr()!=w.untyped_storage().data_ptr()
    assert type(chunk) is int and chunk>0 and h.numel()>0 and w.shape[0]>0
    assert denominator is None or (type(denominator) is int and denominator>=0) or (type(denominator) is torch.Tensor and denominator.device==h.device and denominator.dtype==torch.int64 and denominator.ndim==0 and not denominator.requires_grad)

def shift_targets(labels):
    return F.pad(labels,(0,1),value=-100)[...,1:].contiguous()
