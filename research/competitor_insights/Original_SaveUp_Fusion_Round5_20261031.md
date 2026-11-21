# Round 5 — independently fused MLP save-up and segmented full-decoder packing

**Date 2026-10-31.** Goal <45 sec and <1,359 MiB on the pinned Soup Qwen2.5 NF4+LoRA 64-step single Tesla T4 run. Neither target is currently verified.

## New original component

`original_save_up_mlp_v2.py` saves exactly **the FP16 up projection output and the existing FP16 gate-projection input**, rather than storing the 8,960-wide FP16 gate result for all 28 decoder layers. Backward recomputes the **frozen** gate once and invokes our unchanged starter SwiGLU backward kernel. The original gate and up projection gradient paths keep separate FP16 cast-and-rounding boundaries. This changes neither model parameters nor LoRA dropout, dataset, optimizer, seed, schedule, or training steps.

At N=1427 valid tokens, H=1536, I=8960, 28 layers: gate output requires 24.387 MiB/layer, FP16 gate input 4.181 MiB/layer; theoretical reduction **565.783 MiB of logical saved activations**. This is NOT measured peak device memory; it may cost speed for a recomputed frozen gate GEMM.

Independent torch-only CPU reference: **20/20 forward and input-gradient parity cases passed exactly** for each of v1 and improved FP16-input v2. Local AST compile and patch-order validation passed. Real bitsandbytes NF4 on physical T4 is untested. Neither extra custom node nor starter SwiGLU is countable original kernel time.

## Published source and notebooks

- [Original enhanced module v2](project-repository/blob/master/soup-kernel-challenge/original_save_up_mlp_v2.py)
- [Team's original source v2](research/competitor_insights/original_save_up_mlp_v2.py)
- [Save-up v1 alone (79 test + own kernel gate)](project-repository/blob/master/soup-kernel-challenge/queue_20261031_original_saveup_mlp_lowmem_v1.ipynb)
- [Save-up + independently written segmented Qwen packing](project-repository/blob/master/soup-kernel-challenge/queue_20261031_original_fullpack_saveup_fusion_v1.ipynb)
- [Improved FP16-input save-up + segmented full decoder](project-repository/blob/master/soup-kernel-challenge/queue_20261031_fullpack_saveup_fp16input_v2.ipynb)

All three queued notebooks preserve the exact original `--only solution` 64-step command and the result-export cell. Their source is installed only in `solution/` before the measured subprocess starts. Physical GPU tiny Qwen/PEFT/dropout and own useful original-kernel share checks execute after the task in source-linked child processes; first notebook also runs all 79 Soup regressions.

## Google Sheets queue

| Row | ID | Status at report |
|---|---|---|
|501|`chatgpt-saveup-lowmem5005-20261031-v1`|Ready awaiting intake|
|502|`chatgpt-saveup-fullpack-lowmem-20261031-v1`|Ready awaiting intake|
|503|`chatgpt-saveup-fp16input-fullpack-20261031-v2`|Ready awaiting intake|

The VM Queue's last observed poll was still **2026-10-31 10:27 UTC**; **none had an assigned Job ID, training result, eval loss or full admission** when the report was authored. Do not invent or reuse score values.

Previous public-development Pareto: 66.33s/5,005MiB memory candidate (not 5% verified) and 58.78s/9,325MiB fastest candidate (later *failed* own-kernel gate at 1.967%). external contributor's public paired Kaggle report says 37.7s/3,461MiB/10 checks (not independently rerun). external contributor's README claims ~48s/2,850MiB without an independently verified saved result.

## Promotion gates

1. Get a terminal T4 job with real NF4 gate recompute parity; unsupported configs use the original path.
2. Require unchanged Soup output/adapters/curve and all 79 regressions.
3. Require **≥5% genuinely original useful Triton kernel time** from step3 onward, excluding starter SwiGLU (not inflated with synthetic work).
4. Paired Kaggle and jury hidden-row validation before calling it a win.
5. Avoid combining with our FP16-output RMSNorm branch by default: it changes the MLP input dtype; this save-up guard deliberately accepts FP32 MLP input only.

Idea sources: user's shared comparison, [external contributor](https://example.invalid/redacted-github), [external contributor](https://example.invalid/redacted-github). **No competitor kernel source copied.**
