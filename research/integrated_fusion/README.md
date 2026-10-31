# Soup Challenge — one integrated original fusion candidate

**Status: integrated source built and CPU-tested; physical Tesla T4 and official admission NOT verified.**

This is one actual `solution/` package, with one zero-argument `apply()`. It combines
algorithmic ideas from the two public competitor solutions (external contributor and external contributor)
using independently authored code plus our own previously tested NF4, streamed
CE, RMSNorm and frozen tied-weight optimization. **No competitor kernel source
is copied.** The original supplied starter `swiglu.py` is unchanged and is
never credited toward the 5% original-kernel share.

## Integrated training path

1. `solution.apply()` patches the original fused nested NF4 dequantization,
   exact FP16 tied frozen storage, original SwiGLU starter, compact frozen MLP,
   original masked-scatter fallback, direct FP16-output RMSNorm, streamed
   output head, and full decoder packing, in that order.
2. The *same guarded SFTTrainer.compute_loss* that invokes streamed CE sets
   `labelled_training(labels)` for the packed decoder. This fixes the earlier
   silent-fallback bug where packing was installed but not engaged.
3. The whole Qwen2 decoder computes only live rows of the unchanged input
   minibatch and uses **native causal SDPA per original sample**. There is no
   quadratic additive block-diagonal mask, no cross-example attention, and no
   reordering of user data. Rotary positions remain original per-row positions.
4. Q/V LoRA dropout draws native random masks in the original B×T×H geometry
   (FP16 or FP32 according to the actual projection input), then selects the
   live token rows. The CUDA RNG state advances once per original dropout site.
5. Each eligible frozen MLP saves only the FP16 `up` activation and the
   existing FP16 gate input; backward recomputes the frozen gate once instead
   of storing all 28 layers' wide gate activations. Non-NF4/unsupported calls
   retain their original unoptimized behavior.
6. FP32 RMSNorm arithmetic writes directly to FP16 only under the task's CUDA
   training/autocast contract, eliminating downstream FP32→FP16 copies.
7. The packed hidden state is scattered back to the **original B×T×H shape**
   before supervised CE/entropy/accuracy/Rewind, which are logged through the
   existing streamed-loss path. Dataset, labels, training steps, LoRA targets,
   optimizer and Soup switches remain unchanged.

## Tested locally

`pytest -q tests/test_fusion_cpu.py` (5 groups PASS): independent attention
forward/backward across four packed geometries, FP16/FP32 padded dropout +
RNG masks for ten seeds, FP16 RMSNorm analytical gradients over four shapes,
frozen MLP gate recomputation across 30 dtype/seed cases, integration/source AST
checks. This environment has no CUDA GPU / bitsandbytes / Triton, so **nothing
here is a physical T4 benchmark**.

## Official GPU notebook

Use `Soup_Integrated_Fusion_T4.ipynb` with the existing Google Sheets GPU
queue. The inline notebook is self-contained and installs only solution/ files
(not the protected starter SwiGLU or tests). It retains the service's official
`--only solution` 64-step harness and result-export cells verbatim. Inspect
training time, sampled VRAM, eval, loss curves, adapter parity and the 5%
original useful-kernel share; run **all 79 Soup regression tests**. Only then
compare it with paired public and hidden jury runs.

## Expected constraints and risks

- Native segmented SDPA per row can add four kernel launches per layer and
  may be slower than external contributor's custom varlen attention. This candidate does
  not copy their code or claim equivalent performance.
- Recomputing a frozen MLP gate costs a new GEMM per decoder layer during
  backward. The memory reduction is theoretical until measured.
- FP16-output RMSNorm may shift numerical rounding versus the original FP32
  residual path; quality and adapter gradients require a real T4 check.
- CPU parity **does not establish** real LoRA dropout Philox parity or T4
  output-loss/gradient equivalence. The saved queue's available worker was
  previously stale, so no result is assumed.
- The original pinned starter `solution/swiglu.py` and
  `solution/test_kernels.py` must remain untouched in the GPU VM.

Repository references (ideas only):
https://github.com/external-contributor/hackathon-decentrathon-6
https://github.com/external-contributor/soup-kernel-challenge
Original rules: https://github.com/MakazhanAlpamys/soup-kernel-challenge
