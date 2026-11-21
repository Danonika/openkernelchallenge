"""Original out-of-place FP32 half-split RoPE; no normalization or attention patch.

Production dispatch is CUDA-only and fixed-task geometry. CPU interpretation is
an explicitly requested test path, never a performance or GPU validation claim.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
from collections import Counter

import torch
import triton
import triton.language as tl

EXPECTED_QWEN_BLOB = "b4fed98fafae8e4ee39be3b8cd766b1c85b079cd"
_COUNTS = Counter()


def _event(kind, **extra):
    _COUNTS[kind] += 1
    n = _COUNTS[kind]
    if n == 1 or n % 256 == 0:
        print("ROTARY_EVIDENCE " + json.dumps(dict(event=kind, count=n, **extra), sort_keys=True), flush=True)


def _save_codegen(compiled, name):
    # Opt-in preflight-only evidence. Measured runs never set this environment key.
    directory = os.environ.get("SOUP_ROTARY_CODEGEN_DIR")
    if directory and compiled is not None:
        ptx = compiled.asm.get("ptx", "")
        if not ptx:
            raise RuntimeError("No generated PTX available for required preflight inspection")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(ptx.encode()).hexdigest()
        (root / (name + "_" + digest[:16] + ".ptx")).write_text(ptx)


@triton.jit
def _signflip_f32(x):
    # Exact unary negation, including -0.0. Not the non-equivalent expression 0-x.
    return (x.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["T", "N"])
def rotary_apply_f32(X, C, S, Y, H: tl.constexpr, T,
                     D: tl.constexpr, CB: tl.constexpr, N,
                     BLOCK: tl.constexpr):
    # Flatten the dense B,T,H,(D/2) physical order of the projection storage.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    half = D // 2
    j = i % half
    h = (i // half) % H
    t = i // (half * H)
    b = tl.program_id(1)
    xp = ((b * T + t) * H + h) * D + j
    cp = ((b if CB != 1 else 0) * T + t) * D + j
    m = i < N
    a = tl.load(X + xp, m, other=0)
    z = tl.load(X + xp + half, m, other=0)
    c0 = tl.load(C + cp, m, other=0)
    c1 = tl.load(C + cp + half, m, other=0)
    s0 = tl.load(S + cp, m, other=0)
    s1 = tl.load(S + cp + half, m, other=0)
    # enable_fp_fusion=False at launch: two rounded products then rounded add.
    y0 = tl.add(tl.mul(a, c0), tl.mul(_signflip_f32(z), s0))
    y1 = tl.add(tl.mul(z, c1), tl.mul(a, s1))
    tl.store(Y + xp, y0, m)
    tl.store(Y + xp + half, y1, m)


@triton.jit(do_not_specialize=["T", "N", "GB", "GH", "GT", "GD"])
def rotary_adjoint_f32(G, C, S, DX, H: tl.constexpr, T,
                       D: tl.constexpr, CB: tl.constexpr, N,
                       GB, GH, GT,
                       GD, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    half = D // 2
    j = i % half
    h = (i // half) % H
    t = i // (half * H)
    b = tl.program_id(1)
    xp = ((b * T + t) * H + h) * D + j
    gp = b * GB + h * GH + t * GT + j * GD
    cp = ((b if CB != 1 else 0) * T + t) * D + j
    m = i < N
    g0 = tl.load(G + gp, m, other=0)
    g1 = tl.load(G + gp + half * GD, m, other=0)
    c0 = tl.load(C + cp, m, other=0)
    c1 = tl.load(C + cp + half, m, other=0)
    s0 = tl.load(S + cp, m, other=0)
    s1 = tl.load(S + cp + half, m, other=0)
    # Native slice-backward pads with +0 then accumulates two disjoint slices.
    # It canonicalizes a zero-valued rotated contribution to positive zero.
    z0 = tl.mul(g1, s1)
    z1 = _signflip_f32(tl.mul(g0, s0))
    z0 = tl.where(z0 == 0.0, 0.0, z0)
    z1 = tl.where(z1 == 0.0, 0.0, z1)
    dx0 = tl.add(tl.mul(g0, c0), z0)
    dx1 = tl.add(tl.mul(g1, c1), z1)
    tl.store(DX + xp, dx0, m)
    tl.store(DX + xp + half, dx1, m)


def _dense_projection(x):
    if x.ndim != 4:
        return False
    b, h, t, d = x.shape
    return b > 0 and h > 0 and t > 0 and d > 0 and d % 2 == 0 and x.stride() == (t*h*d, d, h*d, 1)


def _support_reason(q, k, cos, sin, unsqueeze_dim=1, *, cuda_only=True, task_only=True):
    xs = (q, k, cos, sin)
    if any(type(x) is not torch.Tensor for x in xs):
        return "tensor_subclass"
    if unsqueeze_dim != 1:
        return "axis"
    if any(x.dtype != torch.float32 or x.device != q.device or x.is_neg() or x.is_conj() for x in xs):
        return "dtype_device_or_lazy_view"
    if cuda_only and q.device.type != "cuda":
        return "not_cuda"
    if any(x.layout != torch.strided for x in xs):
        return "layout"
    if not _dense_projection(q) or not _dense_projection(k):
        return "projection_stride"
    b, hq, t, d = q.shape
    if k.shape[0] != b or k.shape[2:] != (t, d):
        return "qk_geometry"
    if task_only and not (b <= 4 and t <= 512 and hq == 12 and k.shape[1] == 2 and d == 128):
        return "outside_fixed_task_geometry"
    if cos.ndim != 3 or sin.shape != cos.shape or cos.shape[0] not in (1, b) or cos.shape[1:] != (t, d):
        return "coefficient_geometry"
    if cos.requires_grad or sin.requires_grad:
        return "trainable_coefficient"
    if not cos.is_contiguous() or not sin.is_contiguous():
        return "coefficient_stride"
    storage = [x.untyped_storage().data_ptr() for x in xs]
    if storage[0] == storage[1] or any(s in storage[2:] for s in storage[:2]):
        return "shared_activation_storage"
    return None


def _supported(*args, **kwargs):
    return _support_reason(*args, **kwargs) is None


def native_adjoint(g, cos, sin):
    """Differentiable fallback for higher-order requests; explicit native slice graph."""
    d = g.shape[-1] // 2
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    rotated_grad = g * s
    a, b = rotated_grad[..., :d], rotated_grad[..., d:]
    # Two slice-backwards really do add padded tensors; preserve their zero signs.
    za = torch.cat((torch.zeros_like(a), -a), dim=-1)
    zb = torch.cat((b, torch.zeros_like(b)), dim=-1)
    return g * c + (za + zb)


class _FrozenRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(cos, sin)
        ctx.shape, ctx.stride = tuple(x.shape), x.stride()
        y = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
        b, h, t, d = x.shape
        n = x.numel() // (2 * b)
        compiled = rotary_apply_f32[(triton.cdiv(n, 256), b)](x, cos, sin, y, h, t, d, cos.shape[0], n,
                                                BLOCK=256, num_warps=4, enable_fp_fusion=False)
        _save_codegen(compiled, "rotary_apply_f32")
        _event("forward_launch_returned", shape=list(x.shape), dtype=str(x.dtype), stride=list(x.stride()))
        return y

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None, None
        cos, sin = ctx.saved_tensors
        if torch.is_grad_enabled():
            _event("higher_order_native_adjoint")
            return native_adjoint(g, cos, sin), None, None
        # Pointer loads do not implement PyTorch's lazy negative/conjugate bits.
        g = g.resolve_neg().resolve_conj()
        dx = torch.empty_strided(ctx.shape, ctx.stride, dtype=g.dtype, device=g.device)
        b, h, t, d = ctx.shape
        n = h*t*d//2
        compiled = rotary_adjoint_f32[(triton.cdiv(n, 256), b)](g, cos, sin, dx, h, t, d, cos.shape[0], n,
                                                 *g.stride(), BLOCK=256, num_warps=4,
                                                 enable_fp_fusion=False)
        _save_codegen(compiled, "rotary_adjoint_f32")
        _event("backward_launch_returned", shape=list(ctx.shape), upstream_stride=list(g.stride()))
        return dx, None, None


def fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    interpreting = os.environ.get("TRITON_INTERPRET") == "1"
    reason = _support_reason(q, k, cos, sin, unsqueeze_dim,
                             cuda_only=not interpreting, task_only=not interpreting)
    if reason:
        raise ValueError("Unsupported frozen FP32 rotary boundary: " + reason)
    # Separate Functions preserve independently frozen/unused Q/K branches.
    return _FrozenRoPE.apply(q, cos, sin), _FrozenRoPE.apply(k, cos, sin)


def patch_qwen2_rotary():
    from transformers.models.qwen2 import modeling_qwen2 as module
    original = module.apply_rotary_pos_emb
    if getattr(original, "_original_rotary_candidate", False):
        return True
    raw = open(inspect.getfile(module), "rb").read()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != EXPECTED_QWEN_BLOB:
        _event("patch_refused_source_mismatch", observed=blob)
        return False

    @functools.wraps(original)
    def guarded(q, k, cos, sin, unsqueeze_dim=1):
        reason = _support_reason(q, k, cos, sin, unsqueeze_dim)
        if reason is not None:
            _event("native_fallback_" + reason)
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)

    guarded._original_rotary_candidate = True
    module.apply_rotary_pos_emb = guarded
    _event("patch_installed", source_blob=blob, owned_kernels=["rotary_apply_f32", "rotary_adjoint_f32"])
    return True
