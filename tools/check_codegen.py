"""Reproduce offline SM75 compilation, without claiming GPU execution.

Run manually, outside the measured training process. Compiled artifacts stay
in a temporary directory outside the source package and are never submitted.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


def _static_shared_bytes(cubin):
    """Read static CUDA shared sections from ELF64 without external packages.

    Triton's metadata.shared is launch-time dynamic memory. Inline PTX static
    declarations are reflected in the cubin instead and must be counted too.
    """
    if cubin[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected a little-endian ELF64 cubin")
    table = struct.unpack_from("<Q", cubin, 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", cubin, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", cubin, table + i * entry_size) for i in range(count)]
    names_header = headers[names_index]
    names = cubin[names_header[4]:names_header[4] + names_header[5]]
    total = 0
    for header in headers:
        name = names[header[0]:].split(b"\0", 1)[0]
        if name.startswith(b".nv.shared."):
            total += header[5]
    return total


def _ptxas_resource_audit(ptx, directory, tag):
    """Ask the pinned assembler for register/spill facts without a CUDA device."""
    from triton.backends.nvidia.compiler import get_ptxas

    source = Path(directory) / f"{tag}.ptx"
    binary = Path(directory) / f"{tag}.cubin"
    source.write_text(ptx)
    result = subprocess.run(
        [get_ptxas(75).path, "-lineinfo", "-v", "--gpu-name=sm_75", str(source), "-o", str(binary)],
        capture_output=True, text=True, check=True,
    )
    registers = re.search(r"Used\s+(\d+)\s+registers", result.stderr)
    memory = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", result.stderr)
    barriers = re.search(r"used\s+(\d+)\s+barriers", result.stderr)
    if registers is None or memory is None:
        raise RuntimeError(f"unable to parse ptxas resource report for {tag}")
    return {
        "compiler": "the Triton-selected ptxas for SM75",
        "registers_per_thread": int(registers.group(1)),
        "stack_frame_bytes": int(memory.group(1)),
        "spill_store_bytes": int(memory.group(2)),
        "spill_load_bytes": int(memory.group(3)),
        "barrier_resources": int(barriers.group(1)) if barriers else 0,
        "physical_gpu_execution": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/codegen.json"))
    args = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") == "1":
        parser.error("run offline compilation without TRITON_INTERPRET=1")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    with tempfile.TemporaryDirectory(prefix="soup_codegen_") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        from solution.rmsnorm import soup_rmsnorm_forward, soup_rmsnorm_backward
        from solution.streamed_head import (
            soup_head_partial_statistics, soup_head_merge_statistics,
            soup_head_logit_gradient,
        )
        from solution.head_matmul import soup_head_projection
        from solution.head_matmul_t4 import soup_head_t4_projection
        from solution.head_fused_t4 import (
            soup_head_t4_fused_statistics, soup_head_t4_merge_statistics,
            soup_head_t4_fused_logit_gradient, _projection_assembly_lines,
        )
        from solution.nf4 import soup_nf4_dequantize
        from solution.frozen_storage import soup_frozen_embedding_forward
        from solution.rotary import soup_rotary_forward, soup_rotary_backward
        from solution.lora_epilogue import soup_lora_epilogue_forward, soup_lora_epilogue_backward
        from solution.swiglu import example_swiglu_fwd_kernel, example_swiglu_bwd_kernel

        cases = []
        for dtype in ("fp16", "fp32"):
            cases.append((f"rmsnorm_forward_{dtype}", soup_rmsnorm_forward,
                          {"X": f"*{dtype}", "W": "*fp32", "Y": "*fp32", "INV_RMS": "*fp32"},
                          {"WIDTH": 1536, "EPS": 1e-6, "BLOCK": 2048},
                          {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": False}))
            cases.append((f"rmsnorm_backward_{dtype}", soup_rmsnorm_backward,
                          {"DY": "*fp32", "X": f"*{dtype}", "W": "*fp32", "INV_RMS": "*fp32", "DX": f"*{dtype}"},
                          {"WIDTH": 1536, "BLOCK": 2048},
                          {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": False}))
        cases.extend([
            ("head_partial", soup_head_partial_statistics,
             {"scores": "*fp16", "partial_max": "*fp32", "partial_sum": "*fp32",
              "partial_centered_sum": "*fp32", "partial_argmax": "*i32"},
             {"VOCAB": 151936, "TILES": 38, "BLOCK": 4096}, {"num_warps": 4, "num_stages": 3}),
            ("head_merge", soup_head_merge_statistics,
             {"scores": "*fp16", "targets": "*i64", "partial_max": "*fp32", "partial_sum": "*fp32",
              "partial_centered_sum": "*fp32", "partial_argmax": "*i32", "token_loss": "*fp32",
              "token_entropy": "*fp32", "predictions": "*i64", "logsumexp": "*fp32",
              "row_max": "*fp32", "log_normalizer": "*fp32"},
             {"VOCAB": 151936, "TILES": 38, "MERGE_BLOCK": 64}, {"num_warps": 4, "num_stages": 3}),
            ("head_gradient", soup_head_logit_gradient,
             {"scores_and_gradient": "*fp16", "targets": "*i64", "row_max": "*fp32", "log_normalizer": "*fp32",
              "loss_gradient_over_denominator": "*fp32", "n_elements": "i32"},
             {"VOCAB": 151936, "BLOCK": 4096}, {"num_warps": 4, "num_stages": 3}),
            ("generic_dot_experiment", soup_head_projection,
             {"X": "*fp16", "W": "*fp16", "Y": "*fp16", "TOKENS": "i32"},
             {"DEPTH": 1536, "VOCAB": 151936, "TILE_M": 32, "TILE_V": 128, "TILE_D": 64},
             {"num_warps": 4, "num_stages": 2}),
        ])
        for tag, depth, vocab, packed in [
            ("full", 1536, 151936, True), ("odd", 35, 137, False), ("unaligned", 16, 9, False),
        ]:
            cases.append((f"explicit_t4_mma_{tag}", soup_head_t4_projection,
                          {"X": "*fp16", "W": "*fp16", "Y": "*fp16", "TOKENS": "i32"},
                          {"DEPTH": depth, "VOCAB": vocab, "PACKED_LOADS": packed},
                          {"num_warps": 4, "num_stages": 1}))

        for tag, depth, vocab in [("full", 1536, 151936), ("tail", 32, 129)]:
            cases.append((f"fused_t4_statistics_{tag}", soup_head_t4_fused_statistics,
                          {"X": "*fp16", "W": "*fp16", "targets": "*i64", "partial_max": "*fp32",
                           "partial_sum": "*fp32", "partial_centered_sum": "*fp32", "partial_argmax": "*i32",
                           "target_score": "*fp32", "TOKENS": "i32"},
                          {"DEPTH": depth, "VOCAB": vocab, "TILES": triton.cdiv(vocab, 128)},
                          {"num_warps": 4, "num_stages": 1}))
            cases.append((f"fused_t4_gradient_{tag}", soup_head_t4_fused_logit_gradient,
                          {"X": "*fp16", "W": "*fp16", "targets": "*i64", "row_max": "*fp32", "log_normalizer": "*fp32",
                           "loss_gradient_over_denominator": "*fp32", "output": "*fp16", "TOKENS": "i32"},
                          {"DEPTH": depth, "VOCAB": vocab}, {"num_warps": 4, "num_stages": 1}))
        cases.append(("fused_t4_merge", soup_head_t4_merge_statistics,
                      {"targets": "*i64", "target_score": "*fp32", "partial_max": "*fp32", "partial_sum": "*fp32",
                       "partial_centered_sum": "*fp32", "partial_argmax": "*i32", "token_loss": "*fp32",
                       "entropy": "*fp32", "predictions": "*i64", "logsumexp": "*fp32",
                       "row_max": "*fp32", "log_normalizer": "*fp32"},
                      {"TILES": 1187, "BLOCK": 2048}, {"num_warps": 4, "num_stages": 3}))

        for nested in (False, True):
            for dtype in ("fp16", "fp32"):
                cases.append((f"nf4_{'nested' if nested else 'flat'}_{dtype}", soup_nf4_dequantize,
                              {"packed": "*u8", "scales": "*u8" if nested else "*fp32",
                               "nested_scales": "*fp32", "nested_code": "*fp32", "offset": "*fp32",
                               "nf4_code": "*fp32", "output": f"*{dtype}", "n_values": "i32"},
                              {"QUANT_BLOCK": 64, "NESTED_BLOCK": 256, "NESTED": nested,
                               "PACKED_OUTPUT": dtype == "fp16", "BLOCK": 1024},
                              {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": False}))
        cases.append(("nf4_nested_fp16_unaligned", soup_nf4_dequantize,
                      {"packed": "*u8", "scales": "*u8", "nested_scales": "*fp32", "nested_code": "*fp32",
                       "offset": "*fp32", "nf4_code": "*fp32", "output": "*fp16", "n_values": "i32"},
                      {"QUANT_BLOCK": 64, "NESTED_BLOCK": 256, "NESTED": True, "PACKED_OUTPUT": False, "BLOCK": 1024},
                      {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": False}))
        cases.append(("frozen_embedding", soup_frozen_embedding_forward,
                      {"TOKEN_IDS": "*i64", "WEIGHT": "*fp16", "OUTPUT": "*fp32", "n_elements": "i32"},
                      {"WIDTH": 1536, "BLOCK": 512}, {"num_warps": 4, "num_stages": 3}))
        for batched in (False, True):
            constants = {"Q_HEADS": 12, "K_HEADS": 2, "WIDTH": 128,
                         "BATCHED_COEFFICIENTS": batched, "BLOCK": 512}
            cases.append((f"rotary_forward_{'batched' if batched else 'broadcast'}", soup_rotary_forward,
                          {"Q": "*fp32", "K": "*fp32", "COS": "*fp32", "SIN": "*fp32",
                           "Q_OUT": "*fp32", "K_OUT": "*fp32", "LENGTH": "i32", "Q_BLOCKS": "i32"},
                          constants, {"num_warps": 4, "enable_fp_fusion": False}))
            branches = [(True, True), (True, False), (False, True)] if batched else [(True, True)]
            for do_q, do_k in branches:
                signature = {"DQ_OUT": "*fp32", "DK_OUT": "*fp32", "COS": "*fp32", "SIN": "*fp32",
                             "DQ": "*fp32", "DK": "*fp32", "LENGTH": "i32", "Q_BLOCKS": "i32"}
                signature.update({name: "i64" for name in
                                  ("Q_SB", "Q_SH", "Q_ST", "Q_SD", "K_SB", "K_SH", "K_ST", "K_SD")})
                cases.append((f"rotary_backward_{int(batched)}_{int(do_q)}_{int(do_k)}", soup_rotary_backward,
                              signature, {**constants, "DO_Q": do_q, "DO_K": do_k},
                              {"num_warps": 4, "enable_fp_fusion": False}))
        cases.append(("lora_epilogue_forward", soup_lora_epilogue_forward,
                      {"BASE": "*fp16", "ADAPTER": "*fp16", "OUT": "*fp32", "N": "i32"},
                      {"BLOCK": 1024}, {"num_warps": 4, "enable_fp_fusion": False}))
        for write_base in (False, True):
            cases.append((f"lora_epilogue_backward_{int(write_base)}", soup_lora_epilogue_backward,
                          {"DOUT": "*fp32", "DBASE": "*fp16", "DADAPTER": "*fp16", "N": "i32"},
                          {"BLOCK": 1024, "WRITE_BASE": write_base},
                          {"num_warps": 4, "enable_fp_fusion": False}))
        # A new pointer-type specialization of the unchanged organizer example
        # is still the credited example, not an original eligible kernel.
        cases.append(("credited_starter_compact_saved_backward", example_swiglu_bwd_kernel,
                      {"grad_ptr": "*fp32", "gate_ptr": "*fp16", "up_ptr": "*fp16",
                       "dgate_ptr": "*fp32", "dup_ptr": "*fp32", "n_elements": "i32"},
                      {"BLOCK": 1024}, {"num_warps": 4, "num_stages": 3}))
        cases.append(("credited_starter_fp16_to_fp32_forward", example_swiglu_fwd_kernel,
                      {"gate_ptr": "*fp16", "up_ptr": "*fp16", "out_ptr": "*fp32", "n_elements": "i32"},
                      {"BLOCK": 1024}, {"num_warps": 4, "num_stages": 3}))
        cases.append(("credited_starter_fp32_to_fp16_backward", example_swiglu_bwd_kernel,
                      {"grad_ptr": "*fp32", "gate_ptr": "*fp16", "up_ptr": "*fp16",
                       "dgate_ptr": "*fp16", "dup_ptr": "*fp16", "n_elements": "i32"},
                      {"BLOCK": 1024}, {"num_warps": 4, "num_stages": 3}))

        results = []
        fused_projection_instructions = {}
        for tag, fn, signature, constants, options in cases:
            compiled = triton.compile(
                ASTSource(fn, signature, constexprs=constants),
                target=GPUTarget("cuda", 75, 32), options=options,
            )
            ptx, llvm = compiled.asm["ptx"], compiled.asm["llir"]
            static_shared = _static_shared_bytes(compiled.asm["cubin"])
            mma_count = len(re.findall(r"\bmma\.sync\.", ptx))
            if tag.startswith("explicit_t4_mma"):
                assert mma_count == 4, "explicit T4 kernel must retain four warp MMA operations"
                assert ".target sm_75" in ptx and re.search(r"\.reqntid\s+128", ptx)
                assert "%laneid" in ptx and "~{memory}" in llvm
            if tag.startswith(("fused_t4_statistics", "fused_t4_gradient")):
                assert mma_count == 64 and static_shared == 15360
                assert ".target sm_75" in ptx and re.search(r"\.reqntid\s+128", ptx)
                assert "%laneid" in ptx and "~{memory}" in llvm
                expected_barriers = 2 if tag.startswith("fused_t4_gradient") else 3
                assert len(re.findall(r"\bbar\.sync\b", ptx)) == expected_barriers
                assert not re.search(r"^\s*br\s", llvm.split("asm sideeffect", 1)[0], re.MULTILINE)
                instructions = re.findall(r"mma\.sync\.[^;]+;", ptx)
                shape = (constants["DEPTH"], constants["VOCAB"])
                if shape in fused_projection_instructions:
                    assert instructions == fused_projection_instructions[shape], "forward/recompute MMA order must match"
                else:
                    fused_projection_instructions[shape] = instructions
            if tag.startswith("fused_t4_gradient"):
                assert not re.search(r"\batom\.", ptx)
                assert ptx.count("sub.rn.f32 centered, tmp, row_maximum;") == 64
                assert ptx.count("sub.rn.f32 centered, centered, log_normalizer;") == 64
                assert ptx.count("fma.rn.f32 derivative, expval, scale, target_term;") == 64
                assert ptx.count("selp.f32 derivative, derivative, 0f00000000, supervised;") == 64
            if tag == "generic_dot_experiment":
                assert mma_count == 0 and "fma.rn.f32" in ptx, "re-evaluate generic projection for this compiler"
            if tag.startswith("nf4_"):
                assert not re.search(r"\bfma\.", ptx), "nested scale multiply and add must round separately"
            if tag.startswith(("rotary_", "lora_epilogue_")):
                assert not re.search(r"\bfma\.", ptx), "pointwise rounding boundaries must not contract"
                assert static_shared == 0 and not re.search(r"\batom\.", ptx)
                assert ".ftz" not in ptx, "preserve subnormal and signed-zero checks"
            source = Path(fn.fn.__code__.co_filename)
            item = {
                "case": tag, "kernel": fn.__name__, "status": "offline_compilation_passed",
                "source": str(source.relative_to(repo)),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "signature": signature, "constants": constants, "options": options,
                "shared_memory_bytes": compiled.metadata.shared + static_shared,
                "dynamic_shared_memory_bytes": compiled.metadata.shared,
                "static_shared_memory_bytes": static_shared,
                "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                "cubin_sha256": hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
                "cubin_bytes": len(compiled.asm["cubin"]), "mma_instruction_sites": mma_count,
                "physical_gpu_execution": False,
                "credited_starter_example": tag.startswith("credited_starter_"),
            }
            if tag.startswith(("fused_t4_statistics", "fused_t4_gradient")):
                item["projection_sequence_sha256"] = hashlib.sha256(
                    "\n".join(_projection_assembly_lines()).encode()
                ).hexdigest()
                item["ptxas_resource_audit"] = _ptxas_resource_audit(ptx, cache, tag)
            if tag.startswith(("rotary_", "lora_epilogue_")):
                item["ptxas_resource_audit"] = _ptxas_resource_audit(ptx, cache, tag)
            results.append(item)
            print(f"PASS {tag}: shared={item['shared_memory_bytes']} bytes, mma_sites={mma_count}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "triton_version": triton.__version__, "target": "cuda:sm75:warp32",
            "scope": "Offline compilation and instruction inspection only; no GPU execution or timing.",
            "compiled_artifacts_in_delivery": False, "cases": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
