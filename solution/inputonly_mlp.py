"""Independent frozen NF4 MLP replay: save only decoder input, not gate/up/down.

Inspired by public QLoRA rematerialization research. Original code; retains
frozen base, adapter, data, optimizer, and exact starter SwiGLU kernel semantics.
The starter kernels do NOT count toward the challenge's own-kernel 5% share.
"""
from functools import wraps
import torch
import triton
from .swiglu import BLOCK, example_swiglu_fwd_kernel, example_swiglu_bwd_kernel

_STATS = {'active': 0, 'backward': 0, 'fallback': 0}

def counters():
    return dict(_STATS)

def _decode(weight):
    import bitsandbytes.functional as bnbf
    return bnbf.dequantize_4bit(weight.data, weight.quant_state).to(torch.float16).contiguous()

class _InputOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mlp):
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        act = torch.empty_like(gate)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate.contiguous(), up.contiguous(), act, n, BLOCK=BLOCK)
        out = mlp.down_proj(act)
        ctx.save_for_backward(x)
        ctx.gate_w = mlp.gate_proj.weight
        ctx.up_w = mlp.up_proj.weight
        ctx.down_w = mlp.down_proj.weight
        ctx.versions = (ctx.gate_w._version, ctx.up_w._version, ctx.down_w._version)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        if ctx.versions != (ctx.gate_w._version, ctx.up_w._version, ctx.down_w._version):
            raise RuntimeError('Frozen NF4 weights changed during MLP replay')
        width = x.shape[-1]
        rows = x.numel() // width
        with torch.autocast('cuda', enabled=False):
            wd = _decode(ctx.down_w)
            dh = torch.mm(grad_out.contiguous().view(rows, width).to(torch.float16), wd)
            del wd
            wg = _decode(ctx.gate_w)
            wu = _decode(ctx.up_w)
            x2 = x.contiguous().view(rows, width)
            g = torch.mm(x2, wg.t())
            u = torch.mm(x2, wu.t())
            dgate, dup = torch.empty_like(g), torch.empty_like(u)
            example_swiglu_bwd_kernel[(triton.cdiv(g.numel(), BLOCK),)](
                dh.float().contiguous(), g, u, dgate, dup, g.numel(), BLOCK=BLOCK)
            dx = torch.mm(dgate, wg)
            dx.add_(torch.mm(dup, wu))
        _STATS['backward'] += 1
        return dx.view_as(x), None

def patch_inputonly_mlp():
    import bitsandbytes as bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    prev = Qwen2MLP.forward
    if getattr(prev, '_soup_own_inputonly', False):
        return True
    linear = bnb.nn.Linear4bit
    @wraps(prev)
    def forward(self, x):
        good = (torch.is_grad_enabled() and self.training and x.is_cuda
                and x.requires_grad and x.dtype == torch.float16
                and x.ndim == 3 and x.is_contiguous()
                and x.numel() // x.shape[-1] > 448
                and getattr(self.config, 'hidden_act', 'silu') == 'silu')
        if good:
            for name in ('gate_proj','up_proj','down_proj'):
                m = getattr(self, name, None)
                st = getattr(getattr(m, 'weight', None), 'quant_state', None)
                if (type(m) is not linear or m.bias is not None
                        or m.weight.requires_grad or not m.weight.is_cuda
                        or m.weight.dtype != torch.uint8 or st is None
                        or st.quant_type != 'nf4' or st.dtype != torch.float16
                        or getattr(m, 'compute_dtype', None) != torch.float16
                        or m._forward_hooks or m._forward_pre_hooks or hasattr(m,'_hf_hook')):
                    good = False
                    break
        if not good:
            _STATS['fallback'] += 1
            return prev(self, x)
        _STATS['active'] += 1
        return _InputOnly.apply(x, self)
    forward._soup_own_inputonly = True
    Qwen2MLP.forward = forward
    return True
