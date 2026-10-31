"""Independent grouped frozen-NF4 MLP input-gradient backward for Soup/T4.

Algorithms informed by the team's SOUP-NANO-002 + HERO4-022/023 research,
not copied from public competitor kernels. Existing input-only MLP *forward*
is unchanged. Backward reconstructs frozen gate/up weights into one contiguous
buffer, computes both projections in one cuBLAS GEMM, overwrites the gate/up
activation buffer with both gradients using one original Triton kernel, and
computes input gradients in one GEMM.

The final grouped FP16 dot has a different reduction order from two separate
dots; it must pass full Soup loss/eval checks. This is not bitwise-exact parity.
"""
from functools import wraps

import torch
import triton
import triton.language as tl

from .inputonly_mlp import _InputOnly

COUNTS = {"grouped": 0, "fallback": 0, "elements": 0}


@triton.jit(do_not_specialize=["TOTAL"])
def _soup_inplace_grouped_swiglu_vjp(
    GU, DH, TOTAL,
    INTER: tl.constexpr, BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = idx < TOTAL
    row = idx // INTER
    col = idx % INTER
    gate_ptr = GU + row * (2 * INTER) + col
    up_ptr = gate_ptr + INTER
    gate = tl.load(gate_ptr, mask=ok, other=0).to(tl.float32)
    up = tl.load(up_ptr, mask=ok, other=0).to(tl.float32)
    dy = tl.load(DH + idx, mask=ok, other=0).to(tl.float32)
    sig = tl.sigmoid(gate)
    act = gate * sig
    d_gate = ((dy * up) * sig) * (1.0 + gate * (1.0 - sig))
    d_up = dy * act
    # GU is saved only for this backward. Overwriting both halves avoids
    # another N*2*INTER FP16 allocation and a separate concatenate operation.
    tl.store(gate_ptr, d_gate.to(tl.float16), mask=ok)
    tl.store(up_ptr, d_up.to(tl.float16), mask=ok)


def _weight_shape(w):
    qs = getattr(w, "quant_state", None)
    if qs is None or getattr(qs, "quant_type", None) != "nf4":
        return None
    return tuple(getattr(qs, "shape", ()))


def patch_grouped_dx_runtime_n():
    """Install after patch_inputonly_mlp; fallback keeps original backward."""
    import bitsandbytes.functional as bnbf

    old = _InputOnly.backward
    if getattr(old, "_soup_grouped_dx_runtime_n", False):
        return True

    @wraps(old)
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        width = x.shape[-1]
        rows = x.numel() // width
        inter = _weight_shape(ctx.gate_w)
        up = _weight_shape(ctx.up_w)
        down = _weight_shape(ctx.down_w)
        valid = (isinstance(grad_out, torch.Tensor) and x.is_cuda
                 and grad_out.is_cuda and x.dtype == torch.float16
                 and grad_out.dtype in (torch.float16, torch.float32)
                 and x.is_contiguous() and grad_out.is_contiguous()
                 and x.ndim == 3 and grad_out.shape == x.shape
                 and 512 <= rows <= 2048 and width == 1536
                 and inter == up and inter is not None
                 and len(inter) == 2 and inter[1] == width
                 and inter[0] == 8960
                 and down == (width, inter[0])
                 and ctx.versions == (ctx.gate_w._version,
                     ctx.up_w._version, ctx.down_w._version))
        if not valid:
            COUNTS["fallback"] += 1
            return old(ctx, grad_out)

        intermediate = inter[0]
        with torch.autocast("cuda", enabled=False):
            # 26 MiB, then release before the larger packed projection.
            wd = bnbf.dequantize_4bit(
                ctx.down_w.data, ctx.down_w.quant_state).to(torch.float16)
            dh = torch.mm(grad_out.view(rows, width).to(torch.float16), wd)
            del wd

            # Two independent decoder weights share one contiguous allocation.
            wgu = torch.empty((2 * intermediate, width),
                              device=x.device, dtype=torch.float16)
            bnbf.dequantize_4bit(
                ctx.gate_w.data, ctx.gate_w.quant_state,
                out=wgu[:intermediate])
            bnbf.dequantize_4bit(
                ctx.up_w.data, ctx.up_w.quant_state,
                out=wgu[intermediate:])
            gu = torch.mm(x.view(rows, width), wgu.t())
            _soup_inplace_grouped_swiglu_vjp[
                (triton.cdiv(rows * intermediate, 1024),)
            ](gu, dh, rows * intermediate,
              INTER=intermediate, BLOCK=1024,
              num_warps=4, enable_fp_fusion=False)
            del dh

            # One packed GEMM replaces separate dgate@Wgate and dup@Wup.
            # Its FP16 rounding is checked against reference in our oracle.
            dx = torch.mm(gu, wgu)
        COUNTS["grouped"] += 1
        COUNTS["elements"] += rows * intermediate
        return dx.view_as(x), None

    backward._soup_grouped_dx_runtime_n = True
    _InputOnly.backward = staticmethod(backward)
    return True
