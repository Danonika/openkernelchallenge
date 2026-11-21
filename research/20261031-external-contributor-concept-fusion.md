# 2026-10-31 competitor concept audit — external contributor and external contributor

**Scope:** Research of two user-named public repositories and independent implementation in our own project; **no competitor source code was imported into our solution**. The benchmark results cited for other teams are *their published claims*, not raw outputs independently checked in this ChatGPT environment.

Competitor sources:
- [external-contributor/hackathon-decentrathon-6](https://github.com/external-contributor/hackathon-decentrathon-6) — [SUBMISSION.md](https://github.com/external-contributor/hackathon-decentrathon-6/blob/main/SUBMISSION.md), [commit REDACTED](https://github.com/external-contributor/hackathon-decentrathon-6/commit/REDACTED), [final_numbers.json](https://github.com/external-contributor/hackathon-decentrathon-6/blob/main/docs/final_numbers.json), [Kaggle notebook (external, not directly inspected)](https://example.invalid/redacted).
- [external-contributor/soup-kernel-challenge](https://github.com/external-contributor/soup-kernel-challenge) — [README](https://github.com/external-contributor/soup-kernel-challenge/blob/main/README.md), [SUBMISSION.md](https://github.com/external-contributor/soup-kernel-challenge/blob/main/SUBMISSION.md).

## Verified source facts versus unverified claimed GPU scores

| Repository | Public-source technical claims | Measurement certainty |
|---|---|---|
| external contributor | Kaggle T4 full `soup train` baseline median **85.0 s** (88.4 / 81.5) versus solution **37.7 s** (37.8 / 37.7); **3,461 MiB** versus 14,345 MiB; eval 2.04983 vs 2.04968; original Triton share 12.4%; **10/10** pinned jury checks. | Present in repository's detailed SUBMISSION.md and JSON. Raw Kaggle outputs could **not be independently inspected** here; treat as credible published self-report, NOT verified first-hand. Different paired baseline / session from our GCP queue. |
| external contributor | NF4+LoRA T4 fused vocabulary CE and entropy, custom NF4 decoder and MLP input-only recomputation, fused RMSNorm and RoPE. README's ~48 s / ~2,850 MiB is an **estimated** benchmark table; SUBMISSION.md still says *fill from result.json*. | Implementation exists in repo, but no final measured run established by their source. Do not label as verified competition result. |

## external contributor's measured leave-one-out claims (source: SUBMISSION.md)

In its reported Kaggle session with **37.55 s** all-on, turning off a component caused:
- Fused vocabulary head + its connected unpadding path: **+19.0 s** and **+5,352 MiB**.
- Padding removal alone: **+13.2 s** and **+1,212 MiB**.
- RMSNorm with residual-add fusion: **+10.5 s** and **+4,556 MiB**.
- Startup import/model-loader changes: **+7.2 s** to the full command.
- RoPE fusion **+2.4 s**; on-process overlapping Triton compilation **+2.4 s**; grouped NF4 layer decoding **+1.7 s**, **+262 MiB**.
- Storing frozen tied embedding in its original FP16 representation: ~**440 MiB** saved without meaningful speed change.

These effects are **not additive** and were measured under their hardware, source and data; they cannot be grafted onto our speed metrics without tests. The largest conceptual opportunities are *full-decoder unpadding* and *host / residual+norm overhead*, not more CE token-chunk tuning.

## Concept transfer matrix (no copied code)

| Concept from research | Our status / independent path | Main numerical / rule risk | Priority |
|---|---|---|---|
| Attention only on contiguous live tokens with causal boundaries | **NEW independent original** [row 410](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_original_rowcropped_sdpa_v1.ipynb): keeps full Q/K/V and native LoRA dropout, trims SDPA per original example; CPU exact live-token forward+q/k/v-gradient tests passed. | Row-zero/padded outputs differ; required metrics unchanged; needs T4 full-run eval and loss curves. | High — safer ablation |
| Remove all padding *inside the decoder*, preserve q/v RNG draws on original B*T | **NEW independent original** [row 411](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_original_fullvalid_decoder_unpad_v1.ipynb): gather compact valid tokens in original row order; explicit per-sequence causal SDPA; generate native dropout masks on original geometry before compact gather; scatter last hidden states back to B*T. CPU 10x dropout/RNG and 10x segmented SDPA/fallback gradient oracles pass. | Most substantial source change. Must check T4 shapes, loss/entropy/accuracy/rewind and dropout RNG. Native source is **not** competitor code. | Highest speed upside |
| Frozen MLP saves only input; replays gate/up/down when needed for backward | **NEW independent original** [row 414](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_original_frozen_mlp_input_replay_v1.ipynb): custom autograd forward with transient graph, save x, replay original patched frozen MLP while restoring active-index context and autocast. 20 local CPU exact gradient cases. | Likely extra compute and possibly >80s T4 cutoff; forward numerics and gradient must be verified on actual NF4/LoRA stack. | High for memory |
| One NF4 unpack per decoder layer / fwd-bwd pass, 7 projections share tiles; boundary layers split at memory peak | **NOT IMPLEMENTED**. Our existing 3 GiB persistent FP16 decoded-weight cache helps speed but costs VRAM; a new layer-owned ephemeral buffer with strict lifetime is a high-potential compromise. | Must preserve bnb scales, quant states, version counters and exact fp16 rounding; avoid stale weight cache and aliasing. | Next big memory/speed investment |
| Stream vocabulary in chunks, accumulate dHidden immediately without keeping full logits | Partly represented by our streamed CE/retained-FP16 head, but **NOT** external contributor's online-vocabulary accumulator design. | Preserve native CE accuracy, full-position entropy, argmax ties, shifted target and exact loss denominator. | High, complex |
| Fused residual add + RMSNorm in one original row kernel | Partly implemented: row25 full cache+validnorm+Triton residual, measured **60.52s / 8,111 MiB** on GCP. Not same full-model across-layer fusion. | Dtype/cast boundaries, hidden padded rows. | Medium |
| Startup: defer unused heavy imports, reduce repetitive tokenizer/quant setup, overlap actual Triton compilation with model loading **within same process** | **Not yet implemented** in our winning source. external contributor claims +7.2s when disabled, but import paths/version-sensitive and background processes/work that escape measurement are prohibited. | Never prewarm outside measured run, never manipulate cutoff or timing; only make genuine task execution faster. | High host-wall-time |
| NF4+SwiGLU input-only recomputation with predequantized gate/up GEMM | external contributor code demonstrates idea; our input-only replay follows PyTorch-autograd semantics independently, but **gate+up GEMM fusion not yet implemented**. | Grouped weight decode, FP32/FP16 cast and native GEMM gradients. | Medium |

## Our verified GCP T4 reference frontier as of this source audit

Saved baseline full command **71.85s / 14345 MiB / eval 2.04968**.
- Current best speed: [row206](https://example.invalid/redacted), **58.78s / 9,325 MiB / eval2.04966**, own NF4 cache + MLP padding + exact FP16 head retention + q/o padding compaction.
- Low-memory complete winner: [row239](https://example.invalid/redacted), **66.33s / 5,005 MiB / eval2.04967**; 128-token head chunk + original Triton masked MLP scatter.
- Ambitious user target 45s / 1375 MiB: **not observed** for our complete 64-step run; external contributor's claimed 37.7s / 3461 MiB would beat the speed target but not the memory target. Different evaluation baselines and source authenticity require caution.

## Action: obtain actual GPU evidence, not conjecture

- Rows **410**, **411**, **414** are distinct submitted notebooks via Google Sheets queue; they do not copy competitor source code. Only public GitHub URLs in input column C, never credentials or both URL and JSON. Train command `--only solution` and report export cells remain identical.
- Compare each with saved baseline and existing full-run Pareto frontier. Inspect **job status, training time, GPU peak, public eval loss, original @triton.jit >=5% timed share**, exact loss curves and 79 original regressions when validator is equipped.
- If full-valid decoder substantially wins, iterate source from row411 under strict fallback/no-cross-sample-attention tests; otherwise retain independent cropped attention and MLP rematerialization as instructive negative/positive ablations.
- Do not declare jury admission based on service's three pass flags. Main team repository still requires exact winning `solution/` source files, official pinned Kaggle paired baseline/checks and submission artifacts before Oct 31 23:59 Asia/Almaty.

*Documentation of competitor techniques is attribution of **ideas and public evidence**, not code provenance. Any kernels in our implementation were authored independently from the mathematical behavior described above.*
