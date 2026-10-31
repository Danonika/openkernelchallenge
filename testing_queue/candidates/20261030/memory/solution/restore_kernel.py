"""Original, unaudited-on-T4 half-to-FP32 restoration, not SwiGLU."""
import torch
import triton
import triton.language as tl


@triton.jit
def soup_mem001_restore_half_to_fp32(source, destination, count,
                                    BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offset, mask=offset < count, other=0)
    tl.store(destination + offset, value.to(tl.float32), mask=offset < count)


def launch_restore(source):
    if source.dtype != torch.float16 or not source.is_contiguous():
        raise RuntimeError("MEM001 restore requires contiguous half payload")
    destination = torch.empty(source.shape, dtype=torch.float32, device=source.device)
    soup_mem001_restore_half_to_fp32[(triton.cdiv(source.numel(), 256),)](
        source, destination, source.numel(), BLOCK=256, num_warps=4,
        num_stages=1, enable_fp_fusion=False)
    return destination
