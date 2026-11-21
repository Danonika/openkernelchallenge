# Hero1 — Soup source + competitor idea transfer (2026-10-31)

## Scope and allowed borrowing
Use public projects for *ideas, arithmetic identities, profiling patterns, and independently implemented kernels*, not their kernel source code. The competition disallows copied Triton kernels, forbidden ready-made libraries, changes to data/order/steps/LoRA/optimizer/flags, measurement manipulation, or missing Rewind/TRL outputs.

Pinned task: Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4, LoRA r16 alpha32 on q/v projections only, 64 steps/256 Kazakh public rows, T4 fp16, no bf16. Must pass entire suite and useful-original-kernel >=5% from step3.

## Source-read transfer
- Soup's `src/soup_cli/trainer/rewind_hf.py` reuses causal logits / token-loss, aligns `logits[:, :-1]` to `labels[:, 1:]`, and collects exact per-row mean CE and valid-token count. A reduced-loss path must still return per-token CE for Rewind, including ignored rows.
- Soup's `src/soup_cli/utils/stream_loss_chunk.py` defines the chunked loss/metric payload for streamed logits; don't enable `use_cut_ce` (prohibited configuration switch).
- Soup's `src/soup_cli/trainer/sft.py` builds the SFT trainer, saves adapters, checks finite metrics. Version-guard all patch boundaries; avoid touching harness/task/tests.

Upstream references:
- https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/trainer/rewind_hf.py
- https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/utils/stream_loss_chunk.py
- https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/trainer/sft.py

## Public competitors, independently screened
- https://github.com/ainiddin/Soup-Kernel-Challenge — README reports fused independently-written SwiGLU, CE, RMSNorm; self-reported own-original-Triton 9.7%, memory 1.217x, speed approximately tied. Opportunity: prove actual original-kernel timing, not just loss parity.
- https://github.com/FRIKT333/soup-kernel-challenge — README reports per-row/tiled stable CE, entropy/argmax simultaneously, original RMSNorm and full metrics; self-reported 1.21x speed /1.49x memory on *their* roughly 142.5-s baseline. These numbers are **not cross-run comparable** with our saved 71.85-s public T4 baseline.
- https://github.com/Khasan-012/soup-kernel-work — architecture inventory explains FP32 tied embedding/head, Q/V LoRA dropout, TRL entropy and Rewind row loss; only an example and a chunked head are installed in its visible `apply()` snapshot.

**No competitor Triton source code has been copied into Hero1.** The independent original kernel `soup_mlp_residual_scatter` came from our previously measured 60.52-s fusion lineage, with the own-code q/o, frozen NF4 and exact saved-head paths from our 58.78-s branch.

## Proven public-data frontier before Hero1
- Faster: 58.78 s, 9,325 MiB, eval 2.04966 (To Run row206, q/o padding, cached NF4 and retained logits).
- Leaner: 64.83 s, 5,345 MiB, eval 2.04965 (To Run row21, own MLP scatter).
- Smallest sampled in observed subset: 66.33 s, 5,005 MiB, eval 2.04967 (To Run row239, CE chunk128 + own scatter).
- Public baseline reused by service: 71.85 s / 14,345 MiB / eval 2.04968.
Different origins, single public screens and/or worker checks are **not** official repeated hidden-data or jury-admission results.

## New source-hashed test fusion
- `SOUP-FUSION-HERO1-011`, To Run row481: combine source-preserved q/o+head+full NF4 cache with our previously GPU-tested valid-row post-attention RMSNorm and Triton fused MLP residual scatter.
- `SOUP-FUSION-HERO1-012`, To Run row482: exact same fusion; change only NF4 frozen decoded-weight cache limit from 3 GiB to 1 GiB. This preserves FP16 math while exploring peak-memory/time Pareto.
- Both published under `Danonika/public/soup-kernel-challenge/` as reviewable notebooks and cloned into our testing-queue research branch. Their full training measurements are **pending**.

## Physical-memory feasibility
Research estimates `~644.78 MiB` for all28 compressed NF4 weights and metadata. A resident FP32 tied embedding table at `151936 × 1536 × 4 bytes` is `890.25 MiB`. Their approximate sum is `1535.03 MiB`, already above the stretch target `1359 MiB` before activations and CUDA runtime. This is an *accounting hypothesis*, not an observed peak or a proof the target is impossible in an offload regime. Achieving 1359 MiB likely requires honest residency scheduling or exact lossless frozen storage reduction, not just additional pointwise fusion. Soup's original output/quality checks must still hold.

## Stop/go
1. Check actual queue completion, runtime, peak, eval and unchanged training/rewind metrics.
2. For every candidate, audit source-classified original `@triton.jit` CUDA-event share (excluding shipped example or compile-only) >=5% and all 79 Soup regressions. Do not treat `status=completed` as jury admission.
3. Promote **only** qualified candidate under the user's hackathon team repository and pinned public Kaggle notebook; preserve all audit evidence and source credit. No hidden rows, alternate GPU, benchmark-only behavior, or precompiled kernels.
