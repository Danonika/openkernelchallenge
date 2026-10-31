"""Experimental original Turing FP16 projection using the documented MMA ISA.

Stock Triton 3.6 lowers tl.dot to scalar FMA on SM75.  This module instead
emits Turing's explicit mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32.
It uses the public NVIDIA PTX fragment specification, not another kernel's code:
https://docs.nvidia.com/cuda/archive/11.2.0/parallel-thread-execution/index.html
Section 9.7.13.4.7, Matrix Fragments for mma.m16n8k8.

A CTA has four warps and computes a 32-by-64 output tile.  Each warp owns
16-by-32 outputs, issuing four MMA operations for every 8 hidden channels.
All lane-sensitive loads, arithmetic and stores occur INSIDE a single
side-effecting inline-assembly block using physical thread/lane IDs.  The code
does not assume a relationship between tl.arange indices and physical lanes.

FP16 inputs, FP32 accumulators, FP16 outputs.  Frozen weight; caller owns
autograd.  Every warp follows the same reduction loop, including edge tiles;
only loads and stores are predicated.  No autotuning or compiler patching.

Status: experimental.  Offline assembly and a CPU fragment-mapping model do
not validate CUDA execution, performance, race freedom, or contest eligibility.
Keep the PyTorch projection as the default until real T4 tests pass.
"""

import torch
import triton
import triton.language as tl


def _projection_assembly(even_depth):
    """Build our PTX text; repetition here keeps register families reviewable."""
    lines = [
        "{",
        ".reg .u32 tid, lane, warp, wm, wn, group, pair, k, channel, channel1;",
        ".reg .u32 ma0, ma1, wb0, wb1, wb2, wb3, outbase, outcol;",
        ".reg .u64 ax0, ax1, bw0, bw1, bw2, bw3, offset, ptr, outrow0, outrow1;",
        ".reg .b32 ar0, ar1, br0, br1, br2, br3;",
        ".reg .b16 h0, h1;",
        ".reg .pred am0, am1, bv0, bv1, bv2, bv3, pk0, pk1, pred, stop, outvalid;",
        ".reg .f32 c<16>;",
        "mov.u32 tid, %tid.x;",
        "mov.u32 lane, %laneid;",
        "shr.u32 warp, tid, 5;",
        "and.b32 wm, warp, 1;",
        "shr.u32 wn, warp, 1;",
        "shr.u32 group, lane, 2;",
        "and.b32 pair, lane, 3;",
        "shl.b32 pair, pair, 1;",
        # $1=X, $2=W, $3=Y, $4=M, $5=tile M start, $6=tile V start,
        # $7=D, $8=V.  $0 is an unused dummy output required by Triton.
        "mad.lo.u32 ma0, wm, 16, $5;",
        "add.u32 ma0, ma0, group;",
        "add.u32 ma1, ma0, 8;",
        "mad.lo.u32 outbase, wn, 32, $6;",
        "add.u32 wb0, outbase, group;",
        "add.u32 wb1, wb0, 8;",
        "add.u32 wb2, wb0, 16;",
        "add.u32 wb3, wb0, 24;",
        "setp.lt.u32 am0, ma0, $4;",
        "setp.lt.u32 am1, ma1, $4;",
        "setp.lt.u32 bv0, wb0, $8;",
        "setp.lt.u32 bv1, wb1, $8;",
        "setp.lt.u32 bv2, wb2, $8;",
        "setp.lt.u32 bv3, wb3, $8;",
    ]
    pairs = [
        ("ar0", "ax0", "ma0", "$1", "am0"),
        ("ar1", "ax1", "ma1", "$1", "am1"),
        ("br0", "bw0", "wb0", "$2", "bv0"),
        ("br1", "bw1", "wb1", "$2", "bv1"),
        ("br2", "bw2", "wb2", "$2", "bv2"),
        ("br3", "bw3", "wb3", "$2", "bv3"),
    ]
    for _, address, row, base, _ in pairs:
        lines.extend([
            f"mul.wide.u32 offset, {row}, $7;",
            "shl.b64 offset, offset, 1;",
            f"add.u64 {address}, {base}, offset;",
        ])
    lines.extend(f"mov.f32 c{i}, 0f00000000;" for i in range(16))
    lines.extend([
        "mov.u32 k, 0;",
        "L_depth:",
        "setp.ge.u32 stop, k, $7;",
        "@stop bra L_store;",  # Uniform across every lane in the CTA.
        "add.u32 channel, k, pair;",
        "add.u32 channel1, channel, 1;",
        "setp.lt.u32 pk0, channel, $7;",
        "setp.lt.u32 pk1, channel1, $7;",
        "mul.wide.u32 offset, channel, 2;",
    ])
    for packed, address, _, _, row_valid in pairs:
        lines.append(f"add.u64 ptr, {address}, offset;")
        if even_depth:
            # With aligned base pointers, even D and even channel guarantee
            # natural 4-byte alignment.  Either both halves are in bounds or
            # neither is in bounds.
            lines.extend([
                f"mov.b32 {packed}, 0;",
                f"and.pred pred, {row_valid}, pk0;",
                f"@pred ld.global.b32 {packed}, [ptr];",
            ])
        else:
            # Odd row stride or an odd FP16 storage offset may give only
            # 2-byte alignment.  Load halves separately and mask the final
            # hidden-channel tail independently.
            lines.extend([
                "mov.b16 h0, 0;",
                "mov.b16 h1, 0;",
                f"and.pred pred, {row_valid}, pk0;",
                "@pred ld.global.b16 h0, [ptr];",
                f"and.pred pred, {row_valid}, pk1;",
                "@pred ld.global.b16 h1, [ptr+2];",
                f"mov.b32 {packed}, {{h0, h1}};",
            ])
    for j in range(4):
        fragment = ", ".join(f"c{4*j+i}" for i in range(4))
        lines.append(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            f"{{{fragment}}}, {{ar0, ar1}}, {{br{j}}}, {{{fragment}}};"
        )
    lines.extend([
        "add.u32 k, k, 8;",
        "bra L_depth;",
        "L_store:",
        "mul.wide.u32 offset, ma0, $8;",
        "shl.b64 offset, offset, 1;",
        "add.u64 outrow0, $3, offset;",
        "mul.wide.u32 offset, ma1, $8;",
        "shl.b64 offset, offset, 1;",
        "add.u64 outrow1, $3, offset;",
        "add.u32 outbase, outbase, pair;",
    ])
    # C/D fragment: (group,2t), (group,2t+1), (group+8,2t), (group+8,2t+1).
    for j in range(4):
        for i in range(4):
            col_offset = 8*j + (i & 1)
            row = i // 2
            lines.extend([
                f"add.u32 outcol, outbase, {col_offset};",
                "setp.lt.u32 outvalid, outcol, $8;",
                f"and.pred pred, am{row}, outvalid;",
                "mul.wide.u32 offset, outcol, 2;",
                f"add.u64 ptr, outrow{row}, offset;",
                f"cvt.rn.f16.f32 h0, c{4*j+i};",
                "@pred st.global.b16 [ptr], h0;",
            ])
    lines.extend(["mov.u32 $0, 0;", "}"])
    return "\n".join(lines)


_PTX_EVEN = tl.constexpr(_projection_assembly(True))
_PTX_ODD = tl.constexpr(_projection_assembly(False))


@triton.jit
def soup_head_t4_projection(
    X, W, Y, TOKENS, DEPTH: tl.constexpr, VOCAB: tl.constexpr,
    PACKED_LOADS: tl.constexpr,
):
    start_m = tl.program_id(0) * 32
    start_v = tl.program_id(1) * 64
    # Scalar operands are uniform.  Physical thread/lane IDs and all memory
    # effects stay inside the asm; there are no distributed TL tensor layouts.
    if PACKED_LOADS and DEPTH % 2 == 0:
        tl.inline_asm_elementwise(
            _PTX_EVEN, constraints="=r,l,l,l,r,r,r,r,r,~{memory}",
            args=[X, W, Y, TOKENS, start_m, start_v, DEPTH, VOCAB],
            dtype=tl.int32, is_pure=False, pack=1,
        )
    else:
        tl.inline_asm_elementwise(
            _PTX_ODD, constraints="=r,l,l,l,r,r,r,r,r,~{memory}",
            args=[X, W, Y, TOKENS, start_m, start_v, DEPTH, VOCAB],
            dtype=tl.int32, is_pure=False, pack=1,
        )


def project_fp16(x, weight):
    """Compute [M,D] @ [V,D].T on CUDA using Turing-compatible FP16 MMA.

    A custom autograd Function (or no_grad) must own the gradient path.  Inline
    PTX cannot execute in Triton's CPU interpreter.  Callers must validate this
    optional variant on the actual GPU before choosing it for training.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("project_fp16 expects tensors")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("project_fp16 expects [M,D] and [V,D] with matching D")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("project_fp16 requires FP16 inputs")
    if x.device != weight.device or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("project_fp16 requires contiguous inputs on the same device")
    if weight.requires_grad:
        raise ValueError("project_fp16 requires a frozen weight")
    if torch.is_grad_enabled() and x.requires_grad:
        raise ValueError("project_fp16 requires caller-owned autograd or no_grad()")
    tokens, depth = x.shape
    vocab = weight.shape[0]
    y = torch.empty((tokens, vocab), device=x.device, dtype=torch.float16)
    if tokens == 0 or vocab == 0:
        return y
    if not x.is_cuda:
        raise ValueError("T4 inline PTX requires CUDA; CPU interpretation is unavailable")
    # A contiguous FP16 view can start at an odd storage offset.  Even row
    # stride alone does not make its base pointer naturally 4-byte aligned.
    packed_loads = depth % 2 == 0 and x.data_ptr() % 4 == 0 and weight.data_ptr() % 4 == 0
    soup_head_t4_projection[(triton.cdiv(tokens, 32), triton.cdiv(vocab, 64))](
        x, weight, y, tokens, depth, vocab, packed_loads, num_warps=4, num_stages=1,
    )
    return y
