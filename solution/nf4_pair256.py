"""Original static 256-byte-pattern NF4 decoder for the team's own kernel stack.

Only the public sixteen NormalFloat4 constants are encoded. Each source literal
packs the exact FP32 bits of both possible four-bit values in one uint64.
All weights, nested 8-bit scales, offsets, gradient, optimizer and input
sequences remain dynamic and unchanged. No weight cache or compiled cache.
"""
import math
import torch
import triton
import triton.language as tl

# Expected bits of the public bitsandbytes NF4 table, in code order.
_CODE_BITS = (0xbf800000, 0xbf3239b1, 0xbf066b30, 0xbeca32a0, 0xbe91a24d, 0xbe3d353f, 0xbdba7871, 0x00000000, 0x3da2faff, 0x3e24cae3, 0x3e7c04dd, 0x3ead033a, 0x3ee1a4b8, 0x3f1007ab, 0x3f3913b3, 0x3f800000,)
_PAIR_BITS = (
    0xbf800000bf800000, 0xbf3239b1bf800000, 0xbf066b30bf800000, 0xbeca32a0bf800000, 0xbe91a24dbf800000, 0xbe3d353fbf800000, 0xbdba7871bf800000, 0x00000000bf800000, 0x3da2faffbf800000, 0x3e24cae3bf800000, 0x3e7c04ddbf800000, 0x3ead033abf800000, 0x3ee1a4b8bf800000, 0x3f1007abbf800000, 0x3f3913b3bf800000, 0x3f800000bf800000,
    0xbf800000bf3239b1, 0xbf3239b1bf3239b1, 0xbf066b30bf3239b1, 0xbeca32a0bf3239b1, 0xbe91a24dbf3239b1, 0xbe3d353fbf3239b1, 0xbdba7871bf3239b1, 0x00000000bf3239b1, 0x3da2faffbf3239b1, 0x3e24cae3bf3239b1, 0x3e7c04ddbf3239b1, 0x3ead033abf3239b1, 0x3ee1a4b8bf3239b1, 0x3f1007abbf3239b1, 0x3f3913b3bf3239b1, 0x3f800000bf3239b1,
    0xbf800000bf066b30, 0xbf3239b1bf066b30, 0xbf066b30bf066b30, 0xbeca32a0bf066b30, 0xbe91a24dbf066b30, 0xbe3d353fbf066b30, 0xbdba7871bf066b30, 0x00000000bf066b30, 0x3da2faffbf066b30, 0x3e24cae3bf066b30, 0x3e7c04ddbf066b30, 0x3ead033abf066b30, 0x3ee1a4b8bf066b30, 0x3f1007abbf066b30, 0x3f3913b3bf066b30, 0x3f800000bf066b30,
    0xbf800000beca32a0, 0xbf3239b1beca32a0, 0xbf066b30beca32a0, 0xbeca32a0beca32a0, 0xbe91a24dbeca32a0, 0xbe3d353fbeca32a0, 0xbdba7871beca32a0, 0x00000000beca32a0, 0x3da2faffbeca32a0, 0x3e24cae3beca32a0, 0x3e7c04ddbeca32a0, 0x3ead033abeca32a0, 0x3ee1a4b8beca32a0, 0x3f1007abbeca32a0, 0x3f3913b3beca32a0, 0x3f800000beca32a0,
    0xbf800000be91a24d, 0xbf3239b1be91a24d, 0xbf066b30be91a24d, 0xbeca32a0be91a24d, 0xbe91a24dbe91a24d, 0xbe3d353fbe91a24d, 0xbdba7871be91a24d, 0x00000000be91a24d, 0x3da2faffbe91a24d, 0x3e24cae3be91a24d, 0x3e7c04ddbe91a24d, 0x3ead033abe91a24d, 0x3ee1a4b8be91a24d, 0x3f1007abbe91a24d, 0x3f3913b3be91a24d, 0x3f800000be91a24d,
    0xbf800000be3d353f, 0xbf3239b1be3d353f, 0xbf066b30be3d353f, 0xbeca32a0be3d353f, 0xbe91a24dbe3d353f, 0xbe3d353fbe3d353f, 0xbdba7871be3d353f, 0x00000000be3d353f, 0x3da2faffbe3d353f, 0x3e24cae3be3d353f, 0x3e7c04ddbe3d353f, 0x3ead033abe3d353f, 0x3ee1a4b8be3d353f, 0x3f1007abbe3d353f, 0x3f3913b3be3d353f, 0x3f800000be3d353f,
    0xbf800000bdba7871, 0xbf3239b1bdba7871, 0xbf066b30bdba7871, 0xbeca32a0bdba7871, 0xbe91a24dbdba7871, 0xbe3d353fbdba7871, 0xbdba7871bdba7871, 0x00000000bdba7871, 0x3da2faffbdba7871, 0x3e24cae3bdba7871, 0x3e7c04ddbdba7871, 0x3ead033abdba7871, 0x3ee1a4b8bdba7871, 0x3f1007abbdba7871, 0x3f3913b3bdba7871, 0x3f800000bdba7871,
    0xbf80000000000000, 0xbf3239b100000000, 0xbf066b3000000000, 0xbeca32a000000000, 0xbe91a24d00000000, 0xbe3d353f00000000, 0xbdba787100000000, 0x0000000000000000, 0x3da2faff00000000, 0x3e24cae300000000, 0x3e7c04dd00000000, 0x3ead033a00000000, 0x3ee1a4b800000000, 0x3f1007ab00000000, 0x3f3913b300000000, 0x3f80000000000000,
    0xbf8000003da2faff, 0xbf3239b13da2faff, 0xbf066b303da2faff, 0xbeca32a03da2faff, 0xbe91a24d3da2faff, 0xbe3d353f3da2faff, 0xbdba78713da2faff, 0x000000003da2faff, 0x3da2faff3da2faff, 0x3e24cae33da2faff, 0x3e7c04dd3da2faff, 0x3ead033a3da2faff, 0x3ee1a4b83da2faff, 0x3f1007ab3da2faff, 0x3f3913b33da2faff, 0x3f8000003da2faff,
    0xbf8000003e24cae3, 0xbf3239b13e24cae3, 0xbf066b303e24cae3, 0xbeca32a03e24cae3, 0xbe91a24d3e24cae3, 0xbe3d353f3e24cae3, 0xbdba78713e24cae3, 0x000000003e24cae3, 0x3da2faff3e24cae3, 0x3e24cae33e24cae3, 0x3e7c04dd3e24cae3, 0x3ead033a3e24cae3, 0x3ee1a4b83e24cae3, 0x3f1007ab3e24cae3, 0x3f3913b33e24cae3, 0x3f8000003e24cae3,
    0xbf8000003e7c04dd, 0xbf3239b13e7c04dd, 0xbf066b303e7c04dd, 0xbeca32a03e7c04dd, 0xbe91a24d3e7c04dd, 0xbe3d353f3e7c04dd, 0xbdba78713e7c04dd, 0x000000003e7c04dd, 0x3da2faff3e7c04dd, 0x3e24cae33e7c04dd, 0x3e7c04dd3e7c04dd, 0x3ead033a3e7c04dd, 0x3ee1a4b83e7c04dd, 0x3f1007ab3e7c04dd, 0x3f3913b33e7c04dd, 0x3f8000003e7c04dd,
    0xbf8000003ead033a, 0xbf3239b13ead033a, 0xbf066b303ead033a, 0xbeca32a03ead033a, 0xbe91a24d3ead033a, 0xbe3d353f3ead033a, 0xbdba78713ead033a, 0x000000003ead033a, 0x3da2faff3ead033a, 0x3e24cae33ead033a, 0x3e7c04dd3ead033a, 0x3ead033a3ead033a, 0x3ee1a4b83ead033a, 0x3f1007ab3ead033a, 0x3f3913b33ead033a, 0x3f8000003ead033a,
    0xbf8000003ee1a4b8, 0xbf3239b13ee1a4b8, 0xbf066b303ee1a4b8, 0xbeca32a03ee1a4b8, 0xbe91a24d3ee1a4b8, 0xbe3d353f3ee1a4b8, 0xbdba78713ee1a4b8, 0x000000003ee1a4b8, 0x3da2faff3ee1a4b8, 0x3e24cae33ee1a4b8, 0x3e7c04dd3ee1a4b8, 0x3ead033a3ee1a4b8, 0x3ee1a4b83ee1a4b8, 0x3f1007ab3ee1a4b8, 0x3f3913b33ee1a4b8, 0x3f8000003ee1a4b8,
    0xbf8000003f1007ab, 0xbf3239b13f1007ab, 0xbf066b303f1007ab, 0xbeca32a03f1007ab, 0xbe91a24d3f1007ab, 0xbe3d353f3f1007ab, 0xbdba78713f1007ab, 0x000000003f1007ab, 0x3da2faff3f1007ab, 0x3e24cae33f1007ab, 0x3e7c04dd3f1007ab, 0x3ead033a3f1007ab, 0x3ee1a4b83f1007ab, 0x3f1007ab3f1007ab, 0x3f3913b33f1007ab, 0x3f8000003f1007ab,
    0xbf8000003f3913b3, 0xbf3239b13f3913b3, 0xbf066b303f3913b3, 0xbeca32a03f3913b3, 0xbe91a24d3f3913b3, 0xbe3d353f3f3913b3, 0xbdba78713f3913b3, 0x000000003f3913b3, 0x3da2faff3f3913b3, 0x3e24cae33f3913b3, 0x3e7c04dd3f3913b3, 0x3ead033a3f3913b3, 0x3ee1a4b83f3913b3, 0x3f1007ab3f3913b3, 0x3f3913b33f3913b3, 0x3f8000003f3913b3,
    0xbf8000003f800000, 0xbf3239b13f800000, 0xbf066b303f800000, 0xbeca32a03f800000, 0xbe91a24d3f800000, 0xbe3d353f3f800000, 0xbdba78713f800000, 0x000000003f800000, 0x3da2faff3f800000, 0x3e24cae33f800000, 0x3e7c04dd3f800000, 0x3ead033a3f800000, 0x3ee1a4b83f800000, 0x3f1007ab3f800000, 0x3f3913b33f800000, 0x3f8000003f800000,
)
assert len(_PAIR_BITS) == 256 and all(
    (_PAIR_BITS[x] & 0xffffffff) == _CODE_BITS[x >> 4]
    and (_PAIR_BITS[x] >> 32) == _CODE_BITS[x & 15]
    for x in range(256))
_TABLES = {}

@triton.jit(do_not_specialize=["n_values"])
def _nf4_pair256_decode(packed, scales, nested_scales, nested_code,
                        offset, pairs, output, n_values,
                        QUANT_BLOCK: tl.constexpr,
                        NESTED_BLOCK: tl.constexpr,
                        NESTED: tl.constexpr,
                        PACKED_OUTPUT: tl.constexpr,
                        BLOCK: tl.constexpr):
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = 2 * byte < n_values
    payload = tl.load(packed + byte, active, other=0).to(tl.int32)
    scale_index = (2 * byte) // QUANT_BLOCK
    if NESTED:
        quantized_scale = tl.load(scales + scale_index, active, other=0).to(tl.int32)
        scale_code = tl.load(nested_code + quantized_scale)
        scale_maximum = tl.load(nested_scales + scale_index // NESTED_BLOCK,
                                active, other=0.0)
        scale_offset = tl.load(offset)
        scale = scale_code * scale_maximum + scale_offset
    else:
        scale = tl.load(scales + scale_index, active, other=0.0)
    pair = tl.load(pairs + payload)
    high_f = (pair & 0xffffffff).to(tl.uint32).to(tl.float32, bitcast=True)
    low_f = (pair >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    high = high_f * scale
    low = low_f * scale
    if PACKED_OUTPUT:
        high16 = high.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        low16 = low.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        tl.store(output.to(tl.pointer_type(tl.uint32)) + byte,
                 high16 | (low16 << 16), 2 * byte + 1 < n_values)
        tl.store(output + 2 * byte, high, 2 * byte == n_values - 1)
    else:
        tl.store(output + 2 * byte, high, active)
        tl.store(output + 2 * byte + 1, low, 2 * byte + 1 < n_values)

def _table(device):
    device = torch.device(device)
    if device not in _TABLES:
        from . import nf4
        # An exact bits-only guard, performed once, avoids assuming other NF4 maps.
        reference = nf4._nf4_table(device)
        expected = torch.tensor(_CODE_BITS, device=device, dtype=torch.uint32)
        if not torch.equal(reference.view(torch.int32), expected.view(torch.int32)):
            _TABLES[device] = None
        else:
            _TABLES[device] = torch.tensor(_PAIR_BITS, device=device, dtype=torch.uint64)
    return _TABLES[device]

def patch_static_pair_nf4():
    from . import nf4
    old = nf4._execute
    if getattr(old, "_original_pair256_decode", False):
        return True

    def execute(packed, shape, dtype, scales, blocksize,
                nested_scales=None, nested_code=None, offset=None,
                nested_blocksize=256, out=None):
        if not packed.is_cuda:
            return old(packed, shape, dtype, scales, blocksize,
                       nested_scales, nested_code, offset, nested_blocksize, out)
        table = _table(packed.device)
        if table is None:
            return old(packed, shape, dtype, scales, blocksize,
                       nested_scales, nested_code, offset, nested_blocksize, out)
        if out is None:
            out = torch.empty(shape, device=packed.device, dtype=dtype)
        nested = nested_scales is not None
        n_values = math.prod(shape)
        # The decoder's own existing geometry and cast order are unchanged.
        with torch.cuda.device(packed.device):
            _nf4_pair256_decode[(triton.cdiv(n_values, 2 * nf4.PACKED_BLOCK),)](
                packed, scales,
                nested_scales if nested else table,
                nested_code if nested else table,
                offset if nested else table,
                table, out, n_values,
                QUANT_BLOCK=blocksize, NESTED_BLOCK=nested_blocksize,
                NESTED=nested,
                PACKED_OUTPUT=dtype == torch.float16 and out.data_ptr() % 4 == 0,
                BLOCK=nf4.PACKED_BLOCK,
                num_warps=4, enable_fp_fusion=False)
        return out

    execute._original_pair256_decode = True
    execute._previous = old
    nf4._execute = execute
    return True
