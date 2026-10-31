"""Original byte-lane NF4 decoder with block-local scale broadcast.

Not a GEMM or autograd replacement. Geometry is flat 64-weight outer blocks,
256-scale nested blocks. FP32 scale product and offset addition do not contract.
The 16 constants below are public NF4 format data, not kernel source.
"""
import math
import torch
import triton
import triton.language as tl

QWEN_SHAPES = frozenset(((1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)))


@triton.jit
def _nf4_value(c):
    # Balanced nibble decisions; each lane decodes one packed byte, not one weight.
    low0 = tl.where(c == 0, -1.0, -0.6961928009986877)
    low1 = tl.where(c == 2, -0.5250730514526367, -0.39491748809814453)
    low2 = tl.where(c == 4, -0.28444138169288635, -0.18477343022823334)
    low3 = tl.where(c == 6, -0.09105003625154495, 0.0)
    high0 = tl.where(c == 8, 0.07958029955625534, 0.16093020141124725)
    high1 = tl.where(c == 10, 0.24611230194568634, 0.33791524171829224)
    high2 = tl.where(c == 12, 0.44070982933044434, 0.5626170039176941)
    high3 = tl.where(c == 14, 0.7229568362236023, 1.0)
    a = tl.where(c < 2, low0, low1)
    b = tl.where(c < 6, low2, low3)
    d = tl.where(c < 10, high0, high1)
    e = tl.where(c < 14, high2, high3)
    return tl.where(c < 8, tl.where(c < 4, a, b), tl.where(c < 12, d, e))


@triton.jit
def soup_nf4_blockpair_decode(PACKED, QABS, CODE2, ABS2, OFFSET, OUT,
                              N: tl.constexpr, SCALES: tl.constexpr,
                              PACK_HALF: tl.constexpr):
    # Scales load as a [SCALES] vector, then broadcast across their 32 bytes.
    # The block index is global: it never resets at a matrix row boundary.
    block = tl.program_id(0) * SCALES + tl.arange(0, SCALES)
    valid_block = block * 64 < N
    q = tl.load(QABS + block, valid_block, other=0).to(tl.int32)
    c = tl.load(CODE2 + q, valid_block, other=0).to(tl.float32)
    a = tl.load(ABS2 + block // 256, valid_block, other=0).to(tl.float32)
    product = c * a
    scale = product + tl.load(OFFSET).to(tl.float32)
    pair = block[:, None] * 32 + tl.arange(0, 32)[None, :]
    even = pair * 2
    byte = tl.load(PACKED + pair, even < N, other=0).to(tl.uint32)
    hi = _nf4_value(byte >> 4) * scale[:, None]
    lo = _nf4_value(byte & 15) * scale[:, None]
    if PACK_HALF:
        h = hi.to(tl.float16, fp_downcast_rounding="rtne")
        l = lo.to(tl.float16, fp_downcast_rounding="rtne")
        bits = h.to(tl.uint16, bitcast=True).to(tl.uint32)
        bits |= l.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
        # CUDA is little-endian. High nibble/even weight occupies low 16 bits.
        words = OUT.to(tl.pointer_type(tl.uint32))
        tl.store(words + pair, bits, even + 1 < N)
        if N % 2:
            tl.store(OUT + even, h, even == N - 1)
    else:
        # FP32 has no intermediate FP16 cast. Also handles unaligned FP16 views.
        tl.store(OUT + even, hi, even < N)
        tl.store(OUT + even + 1, lo, even + 1 < N)


def _tensor(t, dtype, count, device):
    return (type(t) is torch.Tensor and t.layout == torch.strided
            and t.device == device and t.dtype == dtype and t.numel() == count
            and t.is_contiguous() and not t.requires_grad)


def raw_supported(B, shape, qabs, code2, abs2, offset, dtype, *, model_only=True):
    # Metadata-only dispatch. No tensor values, benchmark state, or mutable cache.
    if (not isinstance(B, torch.Tensor) or not B.is_cuda
            or B.layout != torch.strided or B.dtype != torch.uint8
            or B.ndim != 2 or B.shape[1] != 1 or B.shape[0] <= 1
            or not B.is_contiguous() or B.requires_grad):
        return False
    if (not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2
            or any(type(d) is not int or d <= 0 for d in shape)
            or (model_only and tuple(shape) not in QWEN_SHAPES)
            or dtype not in (torch.float16, torch.float32)):
        return False
    n = math.prod(shape)
    outer = (n + 63) // 64
    if B.numel() != (n + 1) // 2:
        return False
    specs = ((qabs, torch.uint8, outer), (code2, torch.float32, 256),
             (abs2, torch.float32, (outer + 255) // 256), (offset, torch.float32, 1))
    return (all(_tensor(t, d, size, B.device) for t, d, size in specs)
            and torch.cuda.get_device_capability(B.device) == (7, 5))


def state_supported(B, state):
    if (state is None or getattr(state, "quant_type", None) != "nf4"
            or type(getattr(state, "blocksize", None)) is not int or state.blocksize != 64
            or getattr(state, "nested", False) is not True):
        return False
    nested = getattr(state, "state2", None)
    if (nested is None or getattr(nested, "blocksize", None) != 256
            or getattr(nested, "dtype", None) != torch.float32
            or getattr(nested, "nested", False)):
        return False
    return raw_supported(B, getattr(state, "shape", None), getattr(state, "absmax", None),
                         getattr(nested, "code", None), getattr(nested, "absmax", None),
                         getattr(state, "offset", None), getattr(state, "dtype", None))


def launch_into(B, qabs, code2, abs2, offset, out):
    """Internal validated launch; accepts CPU tensors only under Triton interpreter tests.

    Production wrappers own their fresh output; no decoded weight or scale cache.
    The alignment branch is only for an explicitly supplied test/output view.
    """
    n = out.numel()
    packed_half = out.dtype == torch.float16 and out.data_ptr() % 4 == 0
    soup_nf4_blockpair_decode[(triton.cdiv(n, 64 * 16),)](
        B, qabs, code2, abs2, offset, out, N=n, SCALES=16,
        PACK_HALF=packed_half, num_warps=4, enable_fp_fusion=False)
    return out


def decode(B, shape, qabs, code2, abs2, offset, dtype):
    # Native lifetime and row-major layout, not a persistent workspace. No stream
    # ownership/alias hazards or extra 2.44 GiB all-model dequantized-weight cache.
    with torch.cuda.device(B.device):
        out = torch.empty(tuple(shape), device=B.device, dtype=dtype)
        return launch_into(B, qabs, code2, abs2, offset, out)
