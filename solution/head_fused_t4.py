"""Experimental Turing projection, sufficient statistics and scaled gradients.

This original kernel follows NVIDIA's documented m16n8k8 fragment mapping:
https://docs.nvidia.com/cuda/archive/11.7.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688

A 128-thread CTA projects 64 tokens against 128 vocabulary rows. Cooperative
global loads stage a K=32 slice into 15 KiB of STATIC shared memory. Four warps
perform explicit FP16-input / FP32-accumulator MMA. Every logit is rounded to
FP16, matching an autocast linear, before stable max/sum/centered-sum/argmax
statistics are formed. Only four scalars per token/vocabulary tile are written;
there is no global [tokens, vocabulary] score allocation.

The recomputation entrypoint uses the very same projection instruction sequence
and writes an FP16 logit-gradient chunk directly. It rounds logits to FP16,
forms exp((score - saved maximum) - saved log normalizer) in FP32, and applies
the actual upstream gradient divided by the loss denominator with a fused
multiply-add before rounding to FP16. Keeping the two normalization terms
separate avoids cancellation from a large common logit offset. For finite
projected logits and valid centered state, ignored rows are positive zero,
including with a nonfinite upstream scale. A caller can form d(hidden) with
its usual FP16 matrix multiplication afterward.

All lane-sensitive memory and arithmetic live in one side-effecting scalar
inline-PTX block. The 40-half shared row stride makes each MMA fragment load
conflict-free in the standard 32-bank, four-byte shared-memory model. Staging
stores are not claimed conflict-free. Both stage barriers are uniform, and the
shared region is reused for statistics only after the last consumption barrier.

Triton metadata.shared reports dynamic allocation (zero here). The cubin's
.nv.shared.soup_head_t4_fused_statistics section must report 15,360 bytes. This
static allocation is encoded in the cubin and needs no launch-time allocation.

These entrypoints require caller-owned autograd. Offline assembly and CPU
models cannot validate device execution, races, numerical drift or speed. Keep
this experiment disabled until actual T4 correctness, compute-sanitizer and
timing checks pass. Inputs must be finite, aligned contiguous FP16, and D must
be a positive multiple of 32; valid target IDs or -100 are required. The
projected FP16 logits must also be finite. The caller must route nonfinite or
numerically saturated rows through its native implementation; these low-level
kernels do not preserve native exceptional-value behavior on their own.
"""

import torch
import triton
import triton.language as tl


STATIC_SHARED_BYTES = (64 + 128) * 40 * 2
TILE_TOKENS = 64
TILE_VOCAB = 128


def _projection_assembly_lines():
    """One literally shared MMA sequence for forward and recomputation.

    Operand positions1..8 are X,W,targets,M,startM,startV,D,V in both kernels.
    The statistics and gradient epilogues use their remaining operands.
    """
    p = [
        "{",
        f".shared .align 16 .b8 stage[{STATIC_SHARED_BYTES}];",
        ".reg .u32 tid, lane, warp, wm, wn, group, pair, sm, idx, row, col, k, sk;",
        ".reg .u32 sr, sc, addr, localrow, globalrow, column, target, winner, candidate, tile;",
        ".reg .b32 a<4>, b<8>;",
        ".reg .u32 winner0, winner1;",
        ".reg .b16 half;",
        ".reg .u64 offset, ptr, target64;",
        ".reg .f32 c<64>, mx, total, centered_total, centered, expval, tmp;",
        ".reg .f32 mx0, mx1, sum0, sum1, cs0, cs1, delta0, delta1, factor0, factor1;",
        ".reg .f32 row_maximum, log_normalizer, scale, derivative, target_term;",
        ".reg .pred valid, rowvalid, match, stop, leader, firsthalf, nonempty0, nonempty1;",
        ".reg .pred supervised;",
        "mov.u32 tid, %tid.x;",
        "mov.u32 lane, %laneid;",
        "shr.u32 warp, tid, 5;",
        "and.b32 wm, warp, 1;",
        "shr.u32 wn, warp, 1;",
        "shr.u32 group, lane, 2;",
        "and.b32 pair, lane, 3;",
        "shl.b32 pair, pair, 1;",
        "mov.u32 sm, stage;",
        "shr.u32 tile, $6, 7;",
        "setp.eq.u32 leader, pair, 0;",
    ]
    p += [f"mov.f32 c{i}, 0f00000000;" for i in range(64)]
    p += ["mov.u32 k, 0;", "L_stage:", "setp.ge.u32 stop, k, $7;", "@stop bra L_epilogue;"]
    # Every source half2 is loaded by exactly one lane. A and B each have
    # D-major contiguous pairs; the shared layout pads each K32 row to K40.
    for source, count, rowstart, rowlimit, shared_offset in [
        ("$1", 8, "$5", "$4", 0),
        ("$2", 16, "$6", "$8", 64 * 80),
    ]:
        for iteration in range(count):
            p += [
                f"add.u32 idx, tid, {iteration * 128};",
                "shr.u32 sr, idx, 4;",
                "and.b32 sc, idx, 15;",
                f"add.u32 row, sr, {rowstart};",
                "shl.b32 col, sc, 1;",
                "add.u32 col, col, k;",
                f"setp.lt.u32 valid, row, {rowlimit};",
                "mul.wide.u32 offset, row, $7;",
                "cvt.u64.u32 ptr, col;",
                "add.u64 offset, offset, ptr;",
                "shl.b64 offset, offset, 1;",
                f"add.u64 ptr, {source}, offset;",
                "mov.u32 candidate, 0;",
                "@valid ld.global.b32 candidate, [ptr];",
                "mul.lo.u32 addr, sr, 80;",
                "mad.lo.u32 addr, sc, 4, addr;",
                f"add.u32 addr, addr, {shared_offset};",
                "add.u32 addr, addr, sm;",
                "st.shared.b32 [addr], candidate;",
            ]
    p += ["bar.sync 0;"]
    for inner in range(4):
        # Four packed A registers cover two m16 fragments. Eight B registers
        # cover eight n8 fragments. All 32 lanes enter every MMA instruction.
        p += [f"add.u32 sk, pair, {inner * 8};", "shl.b32 sk, sk, 1;"]
        for i in range(4):
            p += [
                "mad.lo.u32 row, wm, 32, group;",
                f"add.u32 row, row, {i * 8};",
                "mad.lo.u32 addr, row, 80, sk;",
                "add.u32 addr, addr, sm;",
                f"ld.shared.b32 a{i}, [addr];",
            ]
        for j in range(8):
            p += [
                "mad.lo.u32 row, wn, 64, group;",
                f"add.u32 row, row, {j * 8};",
                "mad.lo.u32 addr, row, 80, sk;",
                f"add.u32 addr, addr, {64 * 80};",
                "add.u32 addr, addr, sm;",
                f"ld.shared.b32 b{j}, [addr];",
            ]
        for mb in range(2):
            for j in range(8):
                registers = ", ".join(f"c{(mb * 8 + j) * 4 + i}" for i in range(4))
                p.append(
                    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                    f"{{{registers}}}, {{a{mb*2}, a{mb*2+1}}}, {{b{j}}}, {{{registers}}};"
                )
    # No lane can overwrite a stage while another warp still consumes it.
    p += ["bar.sync 0;", "add.u32 k, k, 32;", "bra L_stage;", "L_epilogue:"]
    return p


def _statistics_epilogue_lines():
    p = []
    for q in range(4):
        mb, r = divmod(q, 2)
        p += [
            "mad.lo.u32 localrow, wm, 32, group;",
            f"add.u32 localrow, localrow, {q * 8};",
            "add.u32 globalrow, localrow, $5;",
            "setp.lt.u32 rowvalid, globalrow, $4;",
            "mul.wide.u32 offset, globalrow, 8;",
            "add.u64 ptr, $3, offset;",
            "mov.u64 target64, 0xffffffffffffffff;",
            "@rowvalid ld.global.u64 target64, [ptr];",
            "cvt.u32.u64 target, target64;",
            "mov.f32 mx, 0fFF800000;",
        ]
        entries = [(f"c{(mb * 8 + j) * 4 + r * 2 + h}", j * 8 + h) for j in range(8) for h in range(2)]
        for c, v in entries:
            p += [
                "mad.lo.u32 column, wn, 64, pair;",
                "add.u32 column, column, $6;",
                f"add.u32 column, column, {v};",
                "setp.lt.u32 valid, column, $8;",
                f"cvt.rn.f16.f32 half, {c};",
                f"cvt.f32.f16 {c}, half;",
                f"selp.f32 {c}, {c}, 0fFF800000, valid;",
                f"max.f32 mx, mx, {c};",
                "setp.eq.u32 match, column, target;",
                "and.pred match, match, rowvalid;",
                "and.pred match, match, valid;",
                "mul.wide.u32 offset, globalrow, 4;",
                "add.u64 ptr, $13, offset;",
                f"@match st.global.f32 [ptr], {c};",
            ]
        for delta in (1, 2):
            p += [f"shfl.sync.bfly.b32 tmp, mx, {delta}, 31, 0xffffffff;", "max.f32 mx, mx, tmp;"]
        p += ["mov.f32 total, 0f00000000;", "mov.f32 centered_total, 0f00000000;", "mov.u32 winner, 2147483647;"]
        for c, v in entries:
            p += [
                "mad.lo.u32 column, wn, 64, pair;",
                "add.u32 column, column, $6;",
                f"add.u32 column, column, {v};",
                "setp.lt.u32 valid, column, $8;",
                f"sub.f32 centered, {c}, mx;",
                "selp.f32 centered, centered, 0f00000000, valid;",
                "mul.f32 tmp, centered, 0f3FB8AA3B;",  # log2(e)
                "ex2.approx.f32 expval, tmp;",
                "selp.f32 expval, expval, 0f00000000, valid;",
                "add.f32 total, total, expval;",
                "fma.rn.f32 centered_total, expval, centered, centered_total;",
                f"setp.eq.f32 match, {c}, mx;",
                "and.pred match, match, valid;",
                "selp.u32 candidate, column, 2147483647, match;",
                "min.u32 winner, winner, candidate;",
            ]
        for delta in (1, 2):
            p += [
                f"shfl.sync.bfly.b32 tmp, total, {delta}, 31, 0xffffffff;",
                "add.f32 total, total, tmp;",
                f"shfl.sync.bfly.b32 tmp, centered_total, {delta}, 31, 0xffffffff;",
                "add.f32 centered_total, centered_total, tmp;",
                f"shfl.sync.bfly.b32 candidate, winner, {delta}, 31, 0xffffffff;",
                "min.u32 winner, winner, candidate;",
            ]
        p += [
            "mad.lo.u32 addr, localrow, 2, wn;",
            "shl.b32 addr, addr, 4;",
            "add.u32 addr, addr, sm;",
            "@leader st.shared.f32 [addr], mx;",
            "@leader st.shared.f32 [addr+4], total;",
            "@leader st.shared.f32 [addr+8], centered_total;",
            "@leader st.shared.u32 [addr+12], winner;",
        ]
    p += [
        "bar.sync 0;",
        "setp.eq.u32 firsthalf, wn, 0;",
        "and.pred leader, leader, firsthalf;",
        "@!leader bra L_done;",
    ]
    for q in range(4):
        p += [
            "mad.lo.u32 localrow, wm, 32, group;",
            f"add.u32 localrow, localrow, {q * 8};",
            "add.u32 globalrow, localrow, $5;",
            "setp.lt.u32 rowvalid, globalrow, $4;",
            "shl.b32 addr, localrow, 5;",
            "add.u32 addr, addr, sm;",
            "ld.shared.f32 mx0, [addr];",
            "ld.shared.f32 sum0, [addr+4];",
            "ld.shared.f32 cs0, [addr+8];",
            "ld.shared.u32 winner0, [addr+12];",
            "ld.shared.f32 mx1, [addr+16];",
            "ld.shared.f32 sum1, [addr+20];",
            "ld.shared.f32 cs1, [addr+24];",
            "ld.shared.u32 winner1, [addr+28];",
            "max.f32 mx, mx0, mx1;",
            "setp.gt.f32 nonempty0, sum0, 0f00000000;",
            "setp.gt.f32 nonempty1, sum1, 0f00000000;",
            "sub.f32 delta0, mx0, mx;",
            "sub.f32 delta1, mx1, mx;",
            "selp.f32 delta0, delta0, 0f00000000, nonempty0;",
            "selp.f32 delta1, delta1, 0f00000000, nonempty1;",
            "mul.f32 factor0, delta0, 0f3FB8AA3B;",
            "mul.f32 factor1, delta1, 0f3FB8AA3B;",
            "ex2.approx.f32 factor0, factor0;",
            "ex2.approx.f32 factor1, factor1;",
            "mul.f32 total, factor0, sum0;",
            "fma.rn.f32 total, factor1, sum1, total;",
            "fma.rn.f32 cs0, delta0, sum0, cs0;",
            "fma.rn.f32 cs1, delta1, sum1, cs1;",
            "mul.f32 centered_total, factor0, cs0;",
            "fma.rn.f32 centered_total, factor1, cs1, centered_total;",
            "setp.eq.f32 match, mx0, mx;",
            "selp.u32 winner0, winner0, 2147483647, match;",
            "setp.eq.f32 match, mx1, mx;",
            "selp.u32 winner1, winner1, 2147483647, match;",
            "min.u32 winner, winner0, winner1;",
            "mad.lo.u32 idx, globalrow, $14, tile;",
            "mul.wide.u32 offset, idx, 4;",
        ]
        for pointer, value in [("$9", "mx"), ("$10", "total"), ("$11", "centered_total"), ("$12", "winner")]:
            p += [f"add.u64 ptr, {pointer}, offset;", f"@rowvalid st.global.b32 [ptr], {value};"]
    p += ["L_done:", "mov.u32 $0, 0;", "}"]
    return p


def _gradient_epilogue_lines():
    """Consume the common projection registers without materializing logits.

    Additional operands are saved FP32 maximum and log normalizer, FP32 scale,
    and FP16 output. No collective instruction occurs here, so masked rows
    cannot affect barriers. Projected scores and centered state must be finite.
    """
    p = ["ld.global.f32 scale, [$11];"]
    for q in range(4):
        mb, r = divmod(q, 2)
        p += [
            "mad.lo.u32 localrow, wm, 32, group;",
            f"add.u32 localrow, localrow, {q * 8};",
            "add.u32 globalrow, localrow, $5;",
            "setp.lt.u32 rowvalid, globalrow, $4;",
            "mul.wide.u32 offset, globalrow, 8;",
            "add.u64 ptr, $3, offset;",
            "mov.u64 target64, 0xffffffffffffff9c;",  # signed -100
            "@rowvalid ld.global.u64 target64, [ptr];",
            "cvt.u32.u64 target, target64;",
            "setp.ne.u32 supervised, target, 0xffffff9c;",
            "mul.wide.u32 offset, globalrow, 4;",
            "add.u64 ptr, $9, offset;",
            "mov.f32 row_maximum, 0f00000000;",
            "@rowvalid ld.global.f32 row_maximum, [ptr];",
            "add.u64 ptr, $10, offset;",
            "mov.f32 log_normalizer, 0f00000000;",
            "@rowvalid ld.global.f32 log_normalizer, [ptr];",
        ]
        for j in range(8):
            for h in range(2):
                c = f"c{(mb * 8 + j) * 4 + r * 2 + h}"
                p += [
                    "mad.lo.u32 column, wn, 64, pair;",
                    "add.u32 column, column, $6;",
                    f"add.u32 column, column, {j * 8 + h};",
                    "setp.lt.u32 valid, column, $8;",
                    "and.pred valid, valid, rowvalid;",
                    f"cvt.rn.f16.f32 half, {c};",
                    "cvt.f32.f16 tmp, half;",
                    "sub.rn.f32 centered, tmp, row_maximum;",
                    "sub.rn.f32 centered, centered, log_normalizer;",
                    "mul.rn.f32 tmp, centered, 0f3FB8AA3B;",
                    "ex2.approx.f32 expval, tmp;",
                    "setp.eq.u32 match, column, target;",
                    "neg.f32 target_term, scale;",
                    "selp.f32 target_term, target_term, 0f00000000, match;",
                    # Native log-softmax backward applies the scaled NLL
                    # gradient before its final multiply-add, including Inf.
                    "fma.rn.f32 derivative, expval, scale, target_term;",
                    # This guarantee requires finite scores/centered state.
                    # Exceptional ignored rows belong to the caller's native
                    # fallback because NaN probabilities times zero are NaN.
                    "selp.f32 derivative, derivative, 0f00000000, supervised;",
                    "cvt.rn.f16.f32 half, derivative;",
                    "mul.wide.u32 offset, globalrow, $8;",
                    "cvt.u64.u32 ptr, column;",
                    "add.u64 offset, offset, ptr;",
                    "shl.b64 offset, offset, 1;",
                    "add.u64 ptr, $12, offset;",
                    "@valid st.global.b16 [ptr], half;",
                ]
    return p + ["mov.u32 $0, 0;", "}"]


def _fused_assembly(mode="statistics"):
    if mode == "statistics":
        epilogue = _statistics_epilogue_lines()
    elif mode == "gradient":
        epilogue = _gradient_epilogue_lines()
    else:
        raise ValueError(f"unknown fused projection mode: {mode}")
    return "\n".join(_projection_assembly_lines() + epilogue)


_FUSED_PTX = tl.constexpr(_fused_assembly())
_GRADIENT_PTX = tl.constexpr(_fused_assembly("gradient"))


@triton.jit
def soup_head_t4_fused_statistics(
    X, W, targets, partial_max, partial_sum, partial_centered_sum,
    partial_argmax, target_score, TOKENS,
    DEPTH: tl.constexpr, VOCAB: tl.constexpr, TILES: tl.constexpr,
):
    start_m = tl.program_id(0) * 64
    start_v = tl.program_id(1) * 128
    tl.inline_asm_elementwise(
        _FUSED_PTX,
        constraints="=r,l,l,l,r,r,r,r,r,l,l,l,l,l,r,~{memory}",
        args=[X.to(tl.uint64), W.to(tl.uint64), targets.to(tl.uint64),
              TOKENS, start_m, start_v, DEPTH, VOCAB,
              partial_max.to(tl.uint64), partial_sum.to(tl.uint64),
              partial_centered_sum.to(tl.uint64), partial_argmax.to(tl.uint64),
              target_score.to(tl.uint64), TILES],
        dtype=tl.int32, is_pure=False, pack=1,
    )


@triton.jit
def soup_head_t4_fused_logit_gradient(
    X, W, targets, row_max, log_normalizer, loss_gradient_over_denominator, output, TOKENS,
    DEPTH: tl.constexpr, VOCAB: tl.constexpr,
):
    start_m = tl.program_id(0) * 64
    start_v = tl.program_id(1) * 128
    tl.inline_asm_elementwise(
        _GRADIENT_PTX,
        constraints="=r,l,l,l,r,r,r,r,r,l,l,l,l,~{memory}",
        args=[X.to(tl.uint64), W.to(tl.uint64), targets.to(tl.uint64),
              TOKENS, start_m, start_v, DEPTH, VOCAB,
              row_max.to(tl.uint64), log_normalizer.to(tl.uint64),
              loss_gradient_over_denominator.to(tl.uint64), output.to(tl.uint64)],
        dtype=tl.int32, is_pure=False, pack=1,
    )


@triton.jit
def soup_head_t4_merge_statistics(
    targets, target_score, partial_max, partial_sum, partial_centered_sum,
    partial_argmax, token_loss, entropy, predictions, logsumexp, row_max, log_normalizer,
    TILES: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    active = tile < TILES
    offset = row * TILES + tile
    local_max = tl.load(partial_max + offset, active, other=-float("inf"))
    maximum = tl.max(local_max, 0)
    delta = tl.where(active, local_max - maximum, 0.0)
    rescale = tl.exp(delta)
    local_sum = tl.load(partial_sum + offset, active, other=0.0)
    local_centered_sum = tl.load(partial_centered_sum + offset, active, other=0.0)
    total = tl.sum(rescale * local_sum, 0)
    centered_total = tl.sum(rescale * (local_centered_sum + delta * local_sum), 0)
    log_total = tl.log(total)
    lse = maximum + log_total
    local_winner = tl.load(partial_argmax + offset, active, other=2147483647)
    winner = tl.min(tl.where(active & (local_max == maximum), local_winner, 2147483647), 0)
    supervised = tl.load(targets + row) != -100
    correct_logit = tl.load(target_score + row)
    tl.store(token_loss + row, tl.where(supervised, -((correct_logit - maximum) - log_total), 0.0))
    tl.store(entropy + row, log_total - centered_total / total)
    tl.store(predictions + row, winner)
    tl.store(logsumexp + row, lse)
    tl.store(row_max + row, maximum)
    tl.store(log_normalizer + row, log_total)


def _validate_projection_inputs(x, weight, targets):
    """Shared metadata guards; inspecting them never synchronizes the device."""
    if not all(isinstance(t, torch.Tensor) for t in (x, weight, targets)):
        raise TypeError("x, weight and targets must be tensors")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected [M,D] and [V,D] with the same D")
    rows, depth = x.shape
    vocab = weight.shape[0]
    if depth <= 0 or depth % 32 or vocab <= 0:
        raise ValueError("D must be a positive multiple of 32 and V must be positive")
    if x.dtype != torch.float16 or weight.dtype != torch.float16 or targets.dtype != torch.int64:
        raise TypeError("x/weight must be FP16 and targets int64")
    if targets.shape != (rows,):
        raise ValueError("targets must contain one already shifted label per row")
    if x.device != weight.device or x.device != targets.device or not all(t.is_contiguous() for t in (x, weight, targets)):
        raise ValueError("all tensors must be contiguous and on the same device")
    if x.data_ptr() % 4 or weight.data_ptr() % 4:
        raise ValueError("FP16 base pointers must be four-byte aligned")
    if weight.requires_grad or (torch.is_grad_enabled() and x.requires_grad):
        raise ValueError("caller-owned autograd and frozen weights are required")
    return rows, depth, vocab


def fused_head_statistics_fp16(x, weight, targets):
    """Return CE, entropy, prediction, LSE, maximum and log normalizer.

    Research-only forward routine. The caller owns autograd/recomputation.
    Targets are int64 IDs in [0,V) or -100. Ignored CE is zero, while entropy
    and predictions still cover every input row. The FP32 maximum and centered
    log normalizer must be saved separately for recomputation; LSE is only a
    diagnostic output. Finite inputs and finite projected FP16 logits are
    required. The caller must detect exceptional or numerically saturated rows
    and use native arithmetic for them. No device-to-host validation or
    alternate training behavior is hidden inside this routine.
    """
    rows, depth, vocab = _validate_projection_inputs(x, weight, targets)
    if not x.is_cuda:
        raise ValueError("the fused T4 assembly requires actual CUDA execution")
    outputs = [torch.empty(rows, device=x.device, dtype=dtype) for dtype in
               (torch.float32, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)]
    if rows == 0:
        return tuple(outputs)
    tiles = triton.cdiv(vocab, TILE_VOCAB)
    partials = [torch.empty((rows, tiles), device=x.device, dtype=dtype) for dtype in
                (torch.float32, torch.float32, torch.float32, torch.int32)]
    # Exactly one matching logit writes each supervised target. Ignored rows
    # have no writer and use the initial zero; no atomic operation is needed.
    target_score = torch.zeros(rows, device=x.device, dtype=torch.float32)
    soup_head_t4_fused_statistics[(triton.cdiv(rows, TILE_TOKENS), tiles)](
        x, weight, targets, *partials, target_score, rows,
        DEPTH=depth, VOCAB=vocab, TILES=tiles, num_warps=4, num_stages=1,
    )
    soup_head_t4_merge_statistics[(rows,)](
        targets, target_score, *partials, *outputs,
        TILES=tiles, BLOCK=triton.next_power_of_2(tiles), num_warps=4,
    )
    return tuple(outputs)


def fused_head_logit_gradient_fp16(
    x, weight, targets, row_max, log_normalizer, loss_gradient_over_denominator, out=None,
):
    """Recompute logits and return their already-scaled FP16 gradient.

    `row_max` and `log_normalizer` are the corresponding saved FP32 forward
    results, indexed in the same order as x/targets. They must remain separate
    to preserve centered normalization. `loss_gradient_over_denominator` is one
    FP32 value containing the real upstream autograd gradient divided by the
    loss's denominator, including any AMP scale. Pass that value before any
    FP16 conversion. Projected scores and the saved centered state must be
    finite. Under that precondition, ignored rows are exactly positive zero,
    including when the scale is Inf or NaN for an all-ignored loss. Exceptional
    scores and numerically saturated rows require the caller's native fallback.

    The optional output is contiguous FP16 [M,V]; it may be two-byte aligned
    because stores are individual halfwords. It must not overlap any input.
    Caller owns autograd. This function does not calculate d(hidden).
    """
    rows, depth, vocab = _validate_projection_inputs(x, weight, targets)
    for name, tensor, count in [
        ("row_max", row_max, rows),
        ("log_normalizer", log_normalizer, rows),
        ("loss_gradient_over_denominator", loss_gradient_over_denominator, 1),
    ]:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be an FP32 tensor")
        if tensor.device != x.device or not tensor.is_contiguous() or tensor.numel() != count:
            raise ValueError(f"{name} must be contiguous, correctly sized and on the input device")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise ValueError("caller-owned autograd is required for saved centered state and scale")
    if row_max.shape != (rows,) or log_normalizer.shape != (rows,):
        raise ValueError("row_max and log_normalizer must have shape [M]")
    if out is not None:
        if (not isinstance(out, torch.Tensor) or out.dtype != torch.float16
                or out.device != x.device or out.shape != (rows, vocab)
                or not out.is_contiguous() or out.requires_grad):
            raise ValueError("out must be contiguous FP16 [M,V] on the input device without autograd")
        # All supported views are contiguous, so exact byte intervals suffice.
        # Prevent cross-CTA races if a caller aliases scores with x or weight.
        if out.numel():
            begin, end = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
            for tensor in (x, weight, targets, row_max, log_normalizer, loss_gradient_over_denominator):
                other_begin = tensor.data_ptr()
                other_end = other_begin + tensor.numel() * tensor.element_size()
                if tensor.numel() and begin < other_end and other_begin < end:
                    raise ValueError("out must not overlap any input tensor")
    if not x.is_cuda:
        raise ValueError("the fused T4 assembly requires actual CUDA execution")
    if out is None:
        out = torch.empty((rows, vocab), device=x.device, dtype=torch.float16)
    if rows:
        soup_head_t4_fused_logit_gradient[(triton.cdiv(rows, TILE_TOKENS), triton.cdiv(vocab, TILE_VOCAB))](
            x, weight, targets, row_max, log_normalizer, loss_gradient_over_denominator, out, rows,
            DEPTH=depth, VOCAB=vocab, num_warps=4, num_stages=1,
        )
    return out
