# Soup challenge, 2026-10-31 — GPU-verified 58.78s and new scale-only cache

**Evidence scope:** Public development GPU queue, one GCP Tesla T4. It reuses a saved baseline; this is **not** an admitted Kaggle/jury run.

## Latest measured Pareto frontier

Baseline: full command **71.85 s**, peak **14,345 MiB**, eval **2.04968**.

| Lane | Queue row / request | Full command | GPU peak | Public eval | Speedup | Evidentiary status |
|---|---|---:|---:|---:|---:|---|
| Fastest current | **206 / chatgpt-fast5950-qproj-oproj-20261031-v1** | **58.78 s** | 9325 MiB | 2.04966 | 1.222x | Complete, 3 development checks passed |
| Alternative speed/mem | 203 / chatgpt-fullcombo-validnorm-retain-qproj-20261031-v1 | 59.10 s | 8905 MiB | 2.04964 | 1.216x | Complete, 3 development checks passed |
| Lower-memory, retained speed | 22 / chatgpt-mlp-fused-residual-20261031-v3 | 64.03 s | 5405 MiB | 2.04960 | 1.122x | Complete, 3 development checks passed |
| Lowest-memory counted speed | 21 / chatgpt-mlp-padding-fused-scatter-20261031-v2 | 64.83 s | **5345 MiB** | 2.04965 | 1.108x | Complete, 3 development checks passed |
| 1GiB cache + valid-row norm | 194 / chatgpt-cache1g-validnorm-residual-20261031-v1 | 62.82 s | 6381 MiB | 2.04962 | 1.144x | Complete, 3 development checks passed |

All full-command times and memory figures are from the [live Sheets queue](https://example.invalid/redacted) and per-job records in VM Details. These are **independent single-run development screens**, not proof of jury entry.

**Race condition warning:** multiple agents edit this queue concurrently. Do not overwrite accepted A:H requests, ready flags, or I:BS results. Only add genuinely distinct new requests and set Ready boolean last after verifying the entire row.

## Submitted source-guarded follow-up experiments from this session

- **Row 239** — [Triton MLP masked-scatter plus 128-token native output-head chunks](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_mlp_scatter_head128_v1.ipynb). A one-variable low-memory ablation versus row21 (64.83s/5345MiB). Changes `CHUNK_TOKENS = 256` to 128, *not* optimizer, padding, dropout, model, data, trainer, or harness. Unmeasured until terminal queue report.
- **Row 245** — [2 GiB decoded NF4 cache + valid-row RMSNorm and fused residual](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_nf4cache2g_validnorm_residual_v1.ipynb). Maps the Pareto curve between already measured 1GiB (62.82s/6381MiB) and 3GiB (60.52s/8111MiB). Unmeasured until complete report.
- **Row 252** — [New original Triton cache of exact nested NF4 block scales only](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_fusedresidual_nf4_scale_cache_v1.ipynb). Against measured row22 low-memory fused residual. A new original `soup_nf4_reconstruct_nested_scales` kernel precomputes *only frozen FP32 block scales*, up to **128MiB** additional memory (not gigabytes of decoded FP16 matrices). The existing useful NF4 dequant, public model/dropout/optimizer, and original `--only solution` 64-step training/export cells remain unchanged. Precautions: tensor identity + pointer/version/dtype/shape/stride key, strong ownership references, default CUDA stream only, and `enable_fp_fusion=False` matching native separate FP32 multiply/add. Does **not** claim GPU numerical parity before test.

## High-value independent competition work

- Rows 203–208 compose q-projection/o-projection/valid-row RMSNorm/custom scatter changes. Row206 already measured 58.78s.
- **Row 330** — independent agent tests source-guarded head compaction that skips vocabulary projection for finite, unsupervised padding rows while preserving valid-token entropy and shifted labels. The first version (row270) failed at training startup because a wrapper omitted `numerics`; row330 fixes that interface. **Do not duplicate while queued.** Even if row330 completes, check exact metrics/rewind/full task parity and masking numerical contract before declaring admissibility.
- **Row 240** — a separate agent attempts vendored offline pytest for actual 79 pinned tests. Previous postrun regression checks failed because the GCP image lacks `pytest`: `/usr/local/bin/python: No module named pytest`. Therefore the 79 tests have not yet been shown passing and no candidate is admitted just because loss/time/memory passed.
- **Row 320** — profiling/hotspots instrumented by another agent; avoid duplicate profiler jobs, since earlier `--profile` exceeded the 80-second server training cutoff.

## Memory feasibility

Official Qwen2.5-1.5B family specs: about **1.54B total** and **1.31B non-embedding** parameters with tied embedding. Raw optimistic storage before metadata/activations/kernel runtime: 1.31B * 0.5 bytes for NF4 + (1.54-1.31)B * 2 bytes for FP16 tied embedding = approximately **1063 MiB**. A peak target of **1375 MiB** leaves about **312 MiB** for all quantization metadata, LoRA, saved activations, CUDA runtime/allocator, workspaces, etc. **This is not a mathematical impossibility proof**, but the memory target is an extremely difficult objective on a Tesla T4 under fixed 4×512 and 64 steps. CPU activation-offload ablation row27 reached partial 3607MiB and exceeded 80s without eval, so it is **not** a successful low-memory training result.

## Admission & submission are still blockers

Original pinned `harness/check_rules.py` requires **all 10 checks**, including own non-example `@triton.jit` launches, **>=5% training time in own kernels** over measured steps 3+, output/adapter metadata/entropy/accuracy/rewind and loss-curve parity, real public eval quality, clean one-T4 measurement, and **79 Soup regression passes**. GCP queue's 3 checks do **not** prove these. The user's 45s/1375MiB competitor goal has not been source-verified as a finished result.

Final Kaggle paired baseline/solution evidence plus exported exact `solution/` files (not merely notebook payload), README/SUBMISSION.md, short video and presentation must be finished before the local 2026-10-31 23:59 deadline. The team repository `main` historically contained only README; verify before claiming final submission.

## Scoring illustration, not actual jury points

If an admitted competitor really achieved **45s / 1375MiB** on the identical paired baseline, with those values as the best speed and memory denominators, the official formula's **speed+memory components** only would be roughly:
- 58.78s/9325MiB: **20.53 points**.
- 59.10s/8905MiB: **20.71 points**.
- 64.03s/5405MiB: **21.11 points**.

The ranking is conditional and excludes the 20 admission points + 15 documentation points. The best admitted competitor is not established. Thus, retaining a good low-memory candidate is strategically sensible even when it isn't the fastest.

## Exact 28-file source-base recovery

The 28-file archive embedded in the earlier local candidate (dab4c473 lineage) was extracted and independently compared with the SHA manifest from `queue_20261031_storage_f749d20.ipynb`. Exactly **two** settings text replacements reconstruct the measured storage base:

```text
FROZEN_STORAGE = False -> True
COMPACT_MLP_ACTIVATIONS = False -> True
```

This yielded settings.py SHA256 `17261889d0b049ddc41a2607e7106dc62e466069466344bcab885be84f9494de` and the exact official path+bytes whole-solution SHA256 **`f749d20c06ad60390bace705f443067f455e0b169b5441047c040ebb3f9bd885`**, matching the verified notebook payload. Thus the base 28 Python source files are now recoverable **byte-for-byte without relying on uninspected guessing**.

A local portable source-export ZIP and `EXPORT_SOUP_WINNER_SOURCES.py` were created for the user in this ChatGPT conversation. They need only the locally downloaded [measured 58.78s notebook](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_fast5950_qproj_oproj_v1.ipynb); the exporter applies its explicit allowlisted source-only overlay cells (`full-nf4-cache`, `mlp-padding-after-full-cache`, `triple-fullcache-mlppad-retainedfp16`, `qproj-on-fast6024`, `o-proj-on-fast5950-qproj`) and skips training and GPU-only oracle code. It emits readable `solution/*.py`, a SHA manifest, and an archive. **Actual final overlay source SHA is not yet locally cross-verified with the GPU execution source**; always compare that before final jury submission.

Current team repo `main` still requires the complete solution/ and submission assets. The development notebook itself is not proof of final competition entry.

## Latest checked results after source recovery

- **Row 239** `chatgpt-lowmem-triton-scatter-chunk128-20261031-v1`, completed **66.33 s / 5005 MiB / eval 2.04967**, all three GCP development checks passed. This is the **lowest peak of a full completed above-5%-speed-improvement run** visible in the Sheets queue, but slower than row21's 64.83s/5345MiB. The only source change is native CE chunk size 256->128 on original Triton masked-scatter MLP. Not proof of 79 tests or jury admission.
- **Row 208** shared Triton scatter combination completed **61.14 s / 8645 MiB / eval 2.04970** (regression versus 58.78s row206), and **row207** q_proj+k_proj completed **59.03 s / 9137 MiB / eval 2.04968** (not superior to 58.78s row206). Do not use those as the absolute speed leader.
- **Rows 245 / 252** 2GiB NF4 decoded-weight cache and original Triton nested-scale-only cache remain queued as of the last poll; both are genuine source-only ablations, not estimated scores.
- **Row 360** `chatgpt-fast5878-original-triton-share-20261031-v1` is queued: exact row206 measured source, unchanged 64-step train/export, then post-run compute original Triton useful step3+ share directly from official `harness.check_rules.source_kernels` and own-kernel timings. It will fail explicitly with share evidence if <5%. Other official checks (79 regressions, outputs/curves, paired Kaggle) still needed.
- **Row330** corrected finite padded-head optimization remains queued. Earlier row270 was an interface TypeError `numerics`, and has no completed training/eval; the initial 3039 MiB was *partial* and cannot substantiate claimed 1375 MiB.

The verified exact f749d20 28-file source plus portable exporter was packaged in a conversation artifact (`Soup_Exact_F749d20_Source_And_Exporter.zip`); its bytewise SHA is confirmed. The final overlays for row206 have not yet been source-hash compared to the GPU-tested unpacked package; exporter must be run on the exact commit-pinned notebook before final submission. Team repo `main` still contained only README on latest check.
