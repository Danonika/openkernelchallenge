"""Original, fused NF4 and nested-scale decoding for the fixed training stack.

This is a new packed-byte Triton implementation of the NF4 representation, not
a port of a library kernel. The 16 codebook values are the public NF4 format.
Nested scales follow bitsandbytes 0.50.2's public QuantState contract:
float32(code8[q] * scale2) + offset, then NF4[code4] * scale.

The optional patch covers both functional dequantization (training backward)
and the CUDA backend's existing dequantize+linear fallback (training forward).
It preserves bitsandbytes' decision to use its own fused GEMM on other shapes.
"""

from functools import wraps
from importlib.metadata import version
import math

import torch
import triton
import triton.language as tl


PACKED_BLOCK = 1024
VALID_BLOCKS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
_NF4_TABLES = {}


@triton.jit(do_not_specialize=["n_values"])
def soup_nf4_dequantize(
    packed, scales, nested_scales, nested_code, offset, nf4_code, output,
    n_values,
    QUANT_BLOCK: tl.constexpr,
    NESTED_BLOCK: tl.constexpr,
    NESTED: tl.constexpr,
    PACKED_OUTPUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One byte yields two consecutive weights; no scale temporary is written."""
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = 2 * byte < n_values
    packed_value = tl.load(packed + byte, active, other=0).to(tl.int32)
    scale_index = (2 * byte) // QUANT_BLOCK
    if NESTED:
        quantized_scale = tl.load(scales + scale_index, active, other=0).to(tl.int32)
        scale_code = tl.load(nested_code + quantized_scale)
        scale_maximum = tl.load(nested_scales + scale_index // NESTED_BLOCK, active, other=0.0)
        scale_offset = tl.load(offset)
        # Launch with enable_fp_fusion=False: the public reference materializes
        # the multiply as FP32 before adding the FP32 offset in a second kernel.
        scale = scale_code * scale_maximum + scale_offset
    else:
        scale = tl.load(scales + scale_index, active, other=0.0)
    high = tl.load(nf4_code + (packed_value >> 4)) * scale
    low = tl.load(nf4_code + (packed_value & 15)) * scale
    if PACKED_OUTPUT:
        # Put the even-position half in the low 16 bits on little-endian CUDA.
        # This coalesced 32-bit store avoids tl.interleave's shared-memory
        # layout conversion. The final odd value never writes past the output.
        high_bits = high.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        low_bits = low.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        both = high_bits | (low_bits << 16)
        tl.store(output.to(tl.pointer_type(tl.uint32)) + byte, both, 2 * byte + 1 < n_values)
        tl.store(output + 2 * byte, high, 2 * byte == n_values - 1)
    else:
        # Also supports FP32 output and contiguous FP16 views aligned to only
        # two bytes, without unsafe wider stores or a layout transpose.
        tl.store(output + 2 * byte, high, active)
        tl.store(output + 2 * byte + 1, low, 2 * byte + 1 < n_values)


def _nf4_table(device):
    """Only a 64-byte representation table is retained; never cache weights."""
    table = _NF4_TABLES.get(device)
    if table is None:
        from bitsandbytes.functional import get_4bit_type

        table = get_4bit_type("nf4", device=device)
        _NF4_TABLES[device] = table
    return table


def _components_supported(packed, shape, dtype, scales, blocksize,
                          nested_scales=None, nested_code=None, offset=None,
                          nested_blocksize=256, out=None, *, cuda_only=False):
    """Metadata-only checks; no device synchronization or data-dependent policy."""
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        return False
    if cuda_only and (not packed.is_cuda or torch.version.hip is not None):
        return False
    if packed.ndim != 2 or min(packed.shape) != 1 or not packed.is_contiguous():
        return False
    if not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2:
        return False
    if any(not isinstance(s, int) or s <= 0 for s in shape):
        return False
    n_values = math.prod(shape)
    if packed.numel() != (n_values + 1) // 2 or blocksize not in VALID_BLOCKS:
        return False
    if dtype not in (torch.float16, torch.float32):
        return False
    device = packed.device

    def valid(tensor, expected_dtype, count):
        return (isinstance(tensor, torch.Tensor) and tensor.device == device
                and tensor.dtype == expected_dtype and tensor.numel() == count
                and tensor.is_contiguous() and not tensor.requires_grad)

    scale_count = triton.cdiv(n_values, blocksize)
    nested = nested_scales is not None
    if not valid(scales, torch.uint8 if nested else torch.float32, scale_count):
        return False
    if nested:
        if nested_blocksize not in VALID_BLOCKS:
            return False
        if not valid(nested_scales, torch.float32, triton.cdiv(scale_count, nested_blocksize)):
            return False
        if not valid(nested_code, torch.float32, 256) or not valid(offset, torch.float32, 1):
            return False
    elif nested_code is not None or offset is not None:
        return False
    if out is not None and (not valid(out, dtype, n_values) or tuple(out.shape) != tuple(shape)):
        return False
    return True


def _execute(packed, shape, dtype, scales, blocksize,
             nested_scales=None, nested_code=None, offset=None,
             nested_blocksize=256, out=None):
    if out is None:
        out = torch.empty(shape, device=packed.device, dtype=dtype)
    nested = nested_scales is not None
    # Unused pointer operands have a valid type in the nonnested specialization.
    placeholder = _nf4_table(packed.device)
    n_values = math.prod(shape)
    with torch.cuda.device(packed.device) if packed.is_cuda else torch.device(packed.device):
        soup_nf4_dequantize[(triton.cdiv(n_values, 2 * PACKED_BLOCK),)](
            packed, scales, nested_scales if nested else placeholder,
            nested_code if nested else placeholder, offset if nested else placeholder,
            placeholder, out, n_values, QUANT_BLOCK=blocksize,
            NESTED_BLOCK=nested_blocksize, NESTED=nested,
            PACKED_OUTPUT=dtype == torch.float16 and out.data_ptr() % 4 == 0,
            BLOCK=PACKED_BLOCK,
            num_warps=4, enable_fp_fusion=False,
        )
    return out


def _state_components(packed, quant_state, out, *, cuda_only=False):
    if quant_state is None or getattr(quant_state, "quant_type", None) != "nf4":
        return None
    state2 = getattr(quant_state, "state2", None)
    nested = bool(getattr(quant_state, "nested", False))
    if nested != (state2 is not None):
        return None
    if nested and (getattr(state2, "nested", False) or state2.dtype != torch.float32):
        return None
    components = dict(
        packed=packed, shape=quant_state.shape, dtype=quant_state.dtype,
        scales=quant_state.absmax, blocksize=quant_state.blocksize,
        nested_scales=state2.absmax if nested else None,
        nested_code=state2.code if nested else None,
        offset=quant_state.offset if nested else None,
        nested_blocksize=state2.blocksize if nested else 256, out=out,
    )
    return components if _components_supported(**components, cuda_only=cuda_only) else None


def dequantize_nf4(packed, quant_state, out=None):
    """Decode a supported NF4 matrix, preserving the public transpose convention.

    Intended for CUDA FP16/FP32 matrices with ordinary or one-level nested
    FP32 scales. The direct entry point also supports Triton's CPU interpreter
    for mathematical checks. It never substitutes a CPU implementation.
    """
    components = _state_components(packed, quant_state, out)
    if components is None:
        raise ValueError("requires contiguous uint8 NF4 matrix with FP16/FP32 output and valid scales")
    decoded = _execute(**components)
    return decoded.t() if packed.shape[0] == 1 else decoded


def patch_bitsandbytes_nf4():
    """Patch exactly the reviewed 0.50.2 boundaries, with original fallbacks."""
    import bitsandbytes.functional as functional

    if version("bitsandbytes") != "0.50.2":
        return False
    if getattr(functional.dequantize_4bit, "_soup_candidate_nf4", False):
        return True
    # Importing the CUDA backend on a CPU installation may fail; functional
    # patching alone would silently miss forward. Install both or neither.
    if torch.version.cuda is None or torch.version.hip is not None:
        return False
    from bitsandbytes.backends.cuda import ops

    original = functional.dequantize_4bit
    original_fallback = ops._dequant_linear_fallback

    @wraps(original)
    def dequantize_4bit(A, quant_state=None, absmax=None, out=None,
                       blocksize=None, quant_type="fp4"):
        components = _state_components(A, quant_state, out, cuda_only=True)
        if components is None:
            return original(A, quant_state, absmax, out, blocksize, quant_type)
        decoded = _execute(**components)
        return decoded.t() if A.shape[0] == 1 else decoded

    @wraps(original_fallback)
    def dequant_linear_fallback(A, B, shapeB, absmax, blocksize, quant_type,
                               bias=None, absmax_8bit=None, absmax_code=None,
                               absmax_offset=None):
        nested = absmax_8bit is not None
        components = dict(
            packed=B, shape=shapeB, dtype=A.dtype,
            scales=absmax_8bit if nested else absmax, blocksize=blocksize,
            nested_scales=absmax if nested else None,
            nested_code=absmax_code, offset=absmax_offset, nested_blocksize=256,
        )
        if (quant_type != "nf4" or A.device != B.device
                or not _components_supported(**components, cuda_only=True)):
            return original_fallback(A, B, shapeB, absmax, blocksize, quant_type,
                                     bias, absmax_8bit, absmax_code, absmax_offset)
        decoded = _execute(**components)
        return torch.nn.functional.linear(A, decoded, bias)

    dequantize_4bit._soup_candidate_nf4 = True
    dequant_linear_fallback._soup_candidate_nf4 = True
    functional.dequantize_4bit = dequantize_4bit
    ops._dequant_linear_fallback = dequant_linear_fallback
    return True
