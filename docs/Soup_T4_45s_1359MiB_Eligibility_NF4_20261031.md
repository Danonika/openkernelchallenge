# Soup T4 research — 45 s / 1,359 MiB stretch goals (2026-10-31)

**Evidence classes:** measured = complete 64-step single public T4 service run; failed = no admissible completed notebook; CPU oracle = synthetic numerical test only; queued = no GPU result. The service reuses a saved public baseline. None is a paired hidden-row jury admission.

## Measured frontier and the missed gate

| Public candidate | Whole command | Sampled peak | Held-out public loss | Additional admission evidence |
| --- | ---: | ---: | ---: | --- |
| Saved public baseline | 71.85 s | 14,345 MiB | 2.04968 | Comparison reference |
| Fast Q-proj + O-proj | **58.78 s** | 9,325 MiB | 2.04966 | Speed-family follow-up **FAIL**: 1.967% own-kernel time |
| Full-cache + FP16 head reuse | 60.24 s | 8,827 MiB | 2.04969 | 5% own-share not established |
| One-GiB decoded NF4 + valid-row RMSNorm + fused residual | 62.82 s | 6,381 MiB | 2.04962 | 5% own-share not established |
| Compact MLP scatter + head chunk128 | 66.33 s | **5,005 MiB** | 2.04967 | 79 regressions and 5% own-share audit queued |

The follow-up of the fast family (request `chatgpt-fast5878-original-triton-share-20261031-v1`, row360) was failed by its **post-run 5% source attribution gate**, not by its completed training metric. Of 35.718 s of measured steps3+, countable original kernels totaled **702.596 ms, 1.967%**, principally original RMSNorm forward (303.477 ms), backward (390.524 ms), and frozen embedding gather (8.595 ms); original NF4 dequant ran for 0 ms because all frozen weights were cached. This is not eligibility. The protected starter SwiGLU kernels are excluded. Real kernels, not arbitrary padding operations, must supply additional useful work.

## Target constraints as arithmetic, not physical proof

At 58.78 s full-command and 41.07 s training-loop time, the nonloop overhead is **17.71 s**. If unchanged, a 45.00 s target requires a 27.29 s loop, a **33.55% cut** in loop time. There may be legitimate setup optimizations too; none is measured here.

From pinned Qwen2.5-1.5B shape (28 layers, hidden1536, MLP8960, tied vocabulary151936), a deliberately optimistic all-resident floor is:

- 1,310,195,712 nonembedding quantized weight values × 0.5 byte ≈ **624.750 MiB** (not including metadata).
- 233,373,696 tied embedding/head FP16 values ≈ **445.125 MiB**.
- 2,179,072 q/v LoRA parameters, gradients and two FP32 Adam states ≈ **33.250 MiB**.

Subtotal ≈ **1,103.125 MiB**, BEFORE quantized block metadata, norms, CUDA context, saved activations, scores, temporary workspaces or allocator reserve. A small unrelated GPU smoke reading was 143 MiB; treating it as a rough context allowance leaves **112.875 MiB** under 1,359 MiB, **not a proven lower bound**. Achieving the target likely requires nonresident frozen weights and a bounded-loss/backward strategy. One earlier CPU saved-activation offload hit the T4 80 s cutoff, so a simple synchronous host copy is insufficient.

## T4 numerical research done this turn

Local CPU script `nf4_t4_mma_oracle_20261031.py` passed five decoder-like shapes. Its independent scalar and vector decoders produce exactly equal FP16 words for synthetic NF4 packed bytes and separately rounded FP32 nested scales. The largest absolute forward mismatch between two CPU GEMM calculation paths is **6.103515625e−5**. Direct signed INT4 substitution correctly differs, proving the nonlinear 16-value NF4 format must be decoded before Tensor Core math.

This **does not** certify actual bitsandbytes QuantState, Turing inline PTX MMA fragments, GPU numerical parity, model quality or speed. PTX `mma.sync.m16n8k8.row.col.f32.f16.f16.f32` is supported on sm75; NVIDIA documents it, and the 16-value NF4 table is public. Next implement original packed decode into aligned 16-bit MMA operands and verify pinned bitsandbytes forward and dX against SM75 before patching training.

## New queued, immutable source experiments

1. **One-GiB NF4 + valid-row RMSNorm + fused original MLP residual + finite masked output head + strict 5% own-share audit**. Request `chatgpt-cache1g-validnorm-maskhead-share-20261031-v1`, To Run row440. Notebook: project-repository/blob/master/soup-kernel-challenge/queue_20261031_cache1g_validnorm_maskedhead_sharegate_v2.ipynb . It preserves original setup, solution-only harness and report exports. The original-kernel auditor fails the notebook if useful own-kernel share is below 5%; candidate's masked-head module has a separate CPU-interpreter numerical preflight.
2. **Exact 5,005 MiB low-memory candidate + own-share audit + offline bundled official 79 regressions**. Request `chatgpt-lowmem5005-ownshare79-20261031-v1`, row480. Notebook: project-repository/blob/master/soup-kernel-challenge/queue_20261031_lowmem5005_admission_gate_v1.ipynb . Original training command/export and source unchanged; post-run audit is separate. The test bundle installs no packages and does not affect the measured run.

Both queue rows were written A:H with Ready=TRUE last and have no accepted VM job ID at the time of this report. Do not treat Ready as completed, submit duplicates, overwrite result cells, or claim physical validation.

## Practical promotion policy

- Never choose a candidate on a 3-badge saved-baseline result alone.
- Require **all 79 pinned regressions**, output/adapter quality, finite exceptional-row parity, no prohibited changes, and useful original-kernel share **≥5% from steps3+ in every solution run**. No copied starter kernels or compiled-kernel credit.
- For fast cache variants, full caching eliminates the useful NF4 kernel's recurring work. Experiment with partial caching / fused true NF4→FP16 MMA work rather than padding no-op GPU kernels to inflate share.
- Before <45/1359 claims, complete physical T4 and paired jury-style runs. Today is Oct31, 2026; Decentrathon's stated deadline is **23:59 Kazakhstan time**.

References: https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/main/harness/check_rules.py ; https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/main/harness/score.py ; https://docs.nvidia.com/cuda/archive/11.2.0/parallel-thread-execution/index.html ; https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py ; https://example.invalid/redacted .
