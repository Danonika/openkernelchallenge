# Actual fused Soup candidate R27 / R28 — implementation status

**2026-10-31, Astana.** This is a real code integration, not a separate microbenchmark. **No copied competitor kernels**. Work remains in the team research `testing-queue` branch; do not confuse it with final competition submission.

## Source-bound completed foundation
- R19 original NF4 full-cache + MLP + q_proj + Qwen whole-decoder native varlen: 51.218 s, 7,897 MiB, public loss 2.04944. [GPU row39](https://example.invalid/redacted).
- R22 original NF4 low-memory + MLP + Qwen whole-decoder native varlen: 55.75 s, 4,767 MiB, public loss 2.04943. [GPU row45](https://example.invalid/redacted).
- The 45 s / 1,359 MiB **simultaneous target has not been met**.

## Repaired two exact installer bugs
- The original settings have `HEAD_NUMERICS = "native"`, not the previously asserted `HEAD_MODE = "cached"`. The task settings must be left unchanged; native integration dispatch is selected by a surgical function call substitution.
- Original `integration.py` includes `numerics=HEAD_NUMERICS` in the `streamed_linear_ce` call. Previously R20/R21/R23/R24 erroneously demanded a multiline call lacking this argument and failed **before training**. New integration replaces exactly one `streamed_linear_ce(` call and uses original `vocab_stream_linear_ce(..., **_legacy_options)` to accept the old optional arguments.

## Real full candidate notebooks (new queue rows; both pending physical T4 results)
- **R27 speed:** [source](project-repository/blob/master/soup-kernel-challenge/queue_20261031_r27_fullfusion_callsite_fixed.ipynb) [queue89](https://example.invalid/redacted), job `redacted-job`. R19 full NF4/MLP/qproj/varlen + own vocab-major Triton CE/entropy/accuracy/Rewind + FP16 RMSNorm + one-sync stats + startup. Physical T4 tiny head and varlen gradient gates precede unchanged full train.
- **R28 budgeted memory:** [source](project-repository/blob/master/soup-kernel-challenge/queue_20261031_r28_budgeted_fullfusion_callsite_fixed.ipynb) [queue90](https://example.invalid/redacted), job `redacted-job`. R22 low-memory+varlen + own exact NF4 decoded cache with 1,536 MiB hard budget + own vocab-major head + independent lossless FP16 tied frozen embedding + FP16 RMSNorm, stats and startup. Three physical T4 gates before unchanged training.
- Original standalone memory patch: [r25](project-repository/blob/master/soup-kernel-challenge/queue_20261031_r25_fullfusion_tied_frozen_fp16.ipynb), source module embedded, guarded PEFT prep wrapper preserves tied Parameter identity and exact float32 embedding output.

## Sources of IDEAS (NOT code)
- [external contributor](https://example.invalid/redacted-github) — genuine varlen padding removal, online vocab-sliced loss, layer-scoped NF4 dequantization, tied compact FP16 weights, startup amortization; self-reported 37.7 s /3,461 MiB at commit `redacted-revision`.
- [external contributor](https://example.invalid/redacted-github) — compact frozen MLP, frozen Linear4bit backward activation avoidance, online CE/entropy/accuracy; approximate, unverified ~48 s /2,850 MiB.

## Promotion checklist
Queue state `completed` is insufficient. Require 64 steps, original-kernel useful share >=5% steps3+, preserved PEFT q/v LoRA rank16/alpha32/dropout0.05, GradScaler, 79 regressions, loss +0.5% max, curve/token/entropy/accuracy/Rewind outputs and paired repeat. New attempts always use unique Request IDs in new rows, `Ready=TRUE` last.

## Final source-location caveat
This GitHub research branch and the GPU queue do not constitute the team’s final platform submission. The team must submit the platform-created repository, public Kaggle notebook with saved results, README/SUBMISSION.md, video and slides before Oct31 23:59 Astana.
