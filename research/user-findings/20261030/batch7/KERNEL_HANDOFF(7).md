# Soup kernel challenge session handoff

Updated: 2026-10-30. Scope: accelerate the fixed LoRA training run for Decentrathon 6.0 Soup Kernel Challenge. This is the compact working state for the next session; load detailed sources only when needed.

## Current state

- Completed: brief review, starter/Soup source inspection, literature review, and a nine-page research PDF.
- Implementation: **not started**. No original solution kernels have been written or validated.
- Measurements: **none from our solution**. No T4 baseline, speedup, memory improvement, or quality result has been measured in this session.
- The platform-created team repository and an available T4 execution environment have not been identified. The public starter is not the team's submission repository.
- User prefers compact text handoffs for transfer between sessions; PDF is a review artifact.
- Proposed direction: preserve the existing dropout and NF4 path, fuse LoRA expansion/scale/add first, and investigate chunked loss plus required metrics in parallel. These are candidates, not demonstrated improvements.

## Fixed workload

| Item | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Base weights | NF4 4-bit; Soup default enables double quantization |
| Platform | One Kaggle T4; fp16 compute; no bf16 |
| Soup | 0.77.0 |
| LoRA | q_proj and v_proj; rank 16; alpha 32; scale 2 |
| LoRA dropout | **0.05**, inherited from Soup because task.yaml omits it |
| Training | Batch 4; sequence length <=512; 256 rows; 64 steps; one epoch |
| Optimizer | AdamW torch; learning rate 2e-4; preserve all fixed settings |
| Architecture | Hidden size 1536; 28 layers; q width 1536; v width 256 |
| Final evaluation | Hidden rows of the same dataset and a different seed |
| Stated deadline | 2026-10-31 23:59 Asia/Almaty, from the brief |

Actual installed PEFT, transformers, TRL, and bitsandbytes versions and runtime tensor dtypes remain unverified. Adapter parameters may be fp32 even with fp16 GEMMs under autocast.

## Rules that affect implementation

- Preserve model, data and batch order, steps, LoRA settings including dropout, optimizer, schedule, and required outputs.
- Do not change built-in modes such as fast_lora, use_cut_ce, gradient_checkpointing, activation_offloading, or packing.
- Original useful @triton.jit kernels must live under solution/. Borrow ideas with attribution; do not copy external kernels or import prohibited kernel libraries. The starter example and compiler-generated kernels do not count.
- Original kernels together must account for >=5% of measured training-step wall time from step 3 onward in each measured run. A useful small epilogue alone may fail this gate.
- Scored time includes loading, compilation, training, and saving. Do not ship precompiled caches or alter the harness.
- First-step loss: within 1%. Mean loss-curve difference: within 3%. Evaluation loss: <=0.5% worse. Required metric/rewind comparisons also apply; token counts must match. Time and peak memory: <=10% worse than baseline. Run official checks and required Soup tests.

## First candidate and mathematical contract

Keep the original base computation, defensive clone, dropout call, and A projection initially. Replace only the adapter expansion, scale, and addition with an original Triton operation and correct backward.

Use **PyTorch weight orientation** here: X[M,K], W[N,K], A[r,K], B[N,r], r=16.

```text
Z = dropout(X) @ A.T
Y = F(Ybase, Z, B) = Ybase + 2 * Z @ B.T

Given G = dL/dY:
dYbase = G
dZ = 2 * G @ B
dB = 2 * G.T @ Z
```

Preserve reference autocast, casts, gradient dtypes, and accumulation behavior. Mathematical equivalence does not imply bitwise equality. Test nonzero B and several updates: PEFT normally starts B at zero, which can hide adapter-path errors in a first-step forward check.

For later full-layer fusion, D is the forward dropout mask divided by (1-p), Xd=D*X:

```text
dA = dZ.T @ Xd
dX = G @ W + D * (dZ @ A)
```

Frozen W still contributes to input gradients. Training A requires Xd or enough information to reconstruct it; saving only Z is insufficient. q and v require separate dropout masks. Matching dropout probability or numerical seed does not establish identical random masks. Do not merge LoRA into W during training with nonzero dropout.

## Ranked pending experiments

1. Locate the authorized team checkout and T4 environment. Capture exact package versions and runtime dtypes. Run the untouched official baseline and inspect the actual bottlenecks.
2. Implement the narrow rank-16 expansion/scale/add candidate. Compare complete module forward/backward, then full command time and admission checks.
3. Improve adapter backward if repeated gradient reads are material. Compare fused and separate reductions.
4. In parallel, chunk the vocabulary projection, loss, entropy, accuracy, and per-row rewind reductions. Preserve shifts, masks, denominators, tie behavior, and row boundaries. Do not merely return loss and omit required metrics.
5. Combine successful candidates and rerun the official checks. Consider dropout/down-projection fusion or NF4/GEMM fusion only after measuring the simpler versions.

## Findings and boundaries

- LoRAFusion is the primary fusion reference. Its 1.27x average / 1.39x maximum standalone kernel gains were measured on H100, not Soup/T4. Its released dropout RNG does not demonstrate PyTorch same-seed equivalence.
- Unsloth provides manual-backward ideas. Axolotl's documented optimized LoRA path requires zero dropout and cannot be adopted unchanged.
- Liger's chunked loss principle is useful for memory. Apple's official CCE implementation requires Ampere or newer and defaults to gradient filtering; an independent T4 implementation must establish full-contribution correctness first.
- LoRA-FA freezes A and changes the prescribed training. Multi-adapter serving/scheduling gains do not directly transfer to this single-adapter training run.
- Derived adapter count: 2,179,072 parameters. At 2048 positions and vocabulary 151936, one fp16 logits tensor is 593.5 MiB; fp32 is 1187 MiB; a 128-position fp16 chunk is 37.1 MiB. These are tensor sizes, not measured peak-memory savings.
- Organizer figures (~80 s command, ~60 s training, ~14 GB peak; ~44% elementwise/casts) are external baselines, not our results. The 44% share is not all LoRA.

## Sources and durable artifacts

- [Detailed research PDF](https://chatgpt.com/library?file_id=libfile_65267149175c81919ab85d9819d7d8dc&account_id=personal)
- [Challenge brief](https://chatgpt.com/library?file_id=libfile_37b0abe044f88191b002d808e4d00570&account_id=personal)
- [Starter task](https://example.invalid/redacted-github); starter commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
- [Soup schema](https://example.invalid/redacted-github); inspected Soup publication commit `846888a045b31084452fa08a684a7a38c02f5f87`.
- [Official rule checks](https://example.invalid/redacted-github)
- [Official Qwen dimensions](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)
- [PEFT Linear4bit](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/bnb.py) - inspected upstream main, not an installed-version pin.
- [LoRAFusion paper](https://arxiv.org/html/2510.00206v1) and [repository](https://github.com/CentML/lorafusion)
- [Unsloth mathematics](https://unsloth.ai/introducing); [Axolotl requirements](https://docs.axolotl.ai/docs/lora_optims.html)
- [Liger paper](https://arxiv.org/html/2410.10989v3); [CCE repository](https://github.com/apple-aiml-research/ml-cross-entropy)

## Update before ending each working session

Replace stale current-state facts; record the team repo, branch/commit, implemented changes, actual measured results with their artifact paths, failed ideas with reasons, blockers, and the next concrete action. Keep this file compact. Once experiments exist, keep individual measured runs in experiments.jsonl and retain only the best verified results and pending decisions here. Do not label a hypothesis as a result or use scratch-only paths as the sole durable reference.
