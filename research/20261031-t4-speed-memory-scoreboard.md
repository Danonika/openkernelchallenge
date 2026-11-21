# 2026-10-31: Soup T4 speed/memory development scoreboard

**Scope:** Actual Google Sheets notebook queue on a GCP Tesla T4, 4 usable CPUs, 30 GiB RAM, saved baseline reuse; **not Kaggle jury admission**. Timestamp is date-only; live status may change. No tokens or credentials in any linked notebook.

## Measured full 64-step GPU runs

Source: [Google Sheets `To Run`](https://example.invalid/redacted), with per-job JSON in **VM Details** and individual runs in **VM Runs**. Baseline from the service is 71.85s / 14345 MiB / public eval loss 2.04968.

| Development variant | Queue row | Job ID | Full command time | Peak MiB | Public eval loss | Speed vs saved base | Status |
|---|---:|---|---:|---:|---:|---:|---|
| Compact storage only | 7 | `redacted-job` | 72.51s | 5963 | 2.04963 | 0.99x | Complete; no speed gain |
| Cache decoded frozen NF4 weights (3 GiB cap), compact storage | 12 | `redacted-job` | 69.34s | 8529 | 2.04965 | 1.036x | Complete; faster but speed gain under 5% |
| **MLP-only padding compaction**, compact storage | **40** | `redacted-job` | **65.43s** | **5667** | **2.04967** | **1.098x** | Complete; fastest low-VRAM Pareto point |
| **NF4 cache + MLP-only padding compaction** | **41** | `redacted-job` | **61.74s** | **8227** | **2.04968** | **1.164x** | Complete; fastest verified service result |
| Exact FP16-logit reuse + storage | 11 | `redacted-job` | 71.97s | 6563 | 2.04965 | 0.998x | Complete; independent ablation |
| LoRA epilogue fusion + storage | 16 | `redacted-job` | 72.73s | 5963 | 2.04966 | 0.988x | Complete; no gain |
| RoPE fusion + storage | 14 | `redacted-job` | 73.54s | 5981 | 2.04965 | 0.977x | Complete; no gain |

All listed completed rows reported the development service's **time, memory, and eval-loss** checks passing. They do **not** prove all of the original starter's 10 official checks or jury admission.

## Published experimental notebooks / work in progress

- **Row 29:** [Full cache + MLP padding + retained FP16 logits](project-repository/blob/master/soup-kernel-challenge/queue_20261031_fullcache_mlppad_retain_v1.ipynb). This new triple combination is unmeasured as of the latest poll; 3 constituent paths have independent GPU evidence.
- **Row 42:** [Reconstructed R9 fusion](project-repository/blob/master/soup-kernel-challenge/Fusion_R9_Reconstructed_Queue.ipynb), source lineage not an independently verified old 61.61-second run. Test pending.
- **Rows 21–23:** independent original Triton scatter/residual fusions around MLP padding and its cache combination; test pending.
- **Row 71:** [Fast 61.74s candidate + *postrun* original Triton share / 79-test audit](project-repository/blob/master/soup-kernel-challenge/queue_20261031_fast6174_rules_attribution_audit_v1.ipynb). Its train and export cells remain unchanged; the added cell executes pinned `harness/check_rules.py` after training and exports the original check details. Since the queue uses `--only solution`, local `result.json` has no baseline run; checks requiring a paired baseline will predictably fail and must **not** be interpreted as jury disqualification. Original-kernel share and regression test checks remain independently useful.
- **Row 72:** [Partial 1 GiB decoded-NF4 cache + MLP padding](project-repository/blob/master/soup-kernel-challenge/queue_20261031_nf4cache_1g_mlppad_v1.ipynb), tests the memory/speed Pareto curve; only the cache cap differs from row 41.

Each queue request uses **Notebook URL in C, Notebook JSON empty in D, Ready written last in E**; source and task hashes are checked by the source installer. Never overwrite accepted queue rows or output I:BS.

## The admission gap (critical)

The pinned [`harness/check_rules.py`](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py) requires **all 10 checks**, including:
- At least one original `@triton.jit` kernel running; originals consume **>=5%** of training time from step 3 onward (starter-derived kernels excluded).
- Same task and Soup outputs (including adapter, tokens, rewind/metrics); close loss curve (first step <=1% relative; mean <=3% relative).
- Correct eval loss (<=0.5% above base), at most +10% time/VRAM, clean one-T4 measurement, and the **79 regression tests**.
- Final Kaggle baseline-versus-solution paired measurement and jury source review. A queue service's three passing summary checks do not prove the remaining gates.

**Specific semantic review:** the MLP-padding patch writes zero MLP output on padding rows while preserving original computation for nonpadding rows. This can affect logging/diagnostics or hidden quality checks even when supervised tokens are unaffected. Keep this risk open until official outputs and loss curves are checked.

## Objectives and decision

Target *aspiration* 45s / 1375 MiB has **no verified complete training report in the Sheet**. It is not a measured baseline or an achieved result.

There are two service-Pareto candidates:
1. Fastest: 61.74s / 8227 MiB (16.4% speedup, 42.6% lower VRAM).
2. Lower VRAM while already beating the 5%-speed threshold: 65.43s / 5667 MiB (9.8% speedup, 60.5% lower VRAM).

Official scoring weights are **35 speed / 30 memory / 20 correctness / 15 documentation** using log ratios against the best admitted team, with <5% improvement treated as a tie. Therefore **faster is not necessarily higher-scoring**. Under the purely hypothetical best-admitted 45s/1375MiB denominators, the memory-Pareto candidate has a slightly larger combined speed+memory score (~18.88 vs ~18.46). The actual best admitted team is not known.

## Follow-up in priority order

1. Read queue rows 21–23, 29, 42, 71, 72 results; keep fastest and memory-efficient options separate.
2. Verify own-kernel timed share, 79 tests, output/entropy/rewind parity and training loss curves before declaring a candidate eligible.
3. Run the fixed official Kaggle paired benchmark with the same prepared data/schedule and unmodified harness, then publish raw result and all 10 bound checks.
4. Extract the winning notebook's embedded 28-file `solution/` bundle into the platform-created repo, verify SHA/source contents and zero-argument `apply()`, then publish Kaggle notebook and SUBMISSION.md/video/presentation before deadline.
5. Do not transplant external ready-made kernel libraries: challenge forbids them. Research such as sm75 attention or fused dequant is inspiration only, with original implementation and numerical parity required.

Cross-reference source-only research: [Turing FA development benchmark](https://github.com/BhoumikPatidar/flash_attn_turing) and [NF4 standalone dequant experiments](https://github.com/Griffith-7/nf4-triton-kernel). They are not tests of this challenge.

## Later 2026-10-31 measured wave — verified GCP T4 queue, not jury admission

New complete 64-step development jobs (the service confirmed time/memory/eval checks; paired Kaggle / 10 original checks remain UNVERIFIED):

| Source candidate | Queue row | Job / identity | Wall time | T4 peak | Eval loss | Speedup |
|---|---:|---|---:|---:|---:|---:|
| Full NF4 cache + MLP padding + FP16 forward-logit reuse | 29 | `redacted-job` | **60.24s** | 8827 MiB | 2.04969 | **1.193x** |
| Full NF4 cache + fused compact-MLP residual + valid-row RMSNorm | 25 | `redacted-job` | **60.52s** | 8111 MiB | 2.04965 | 1.187x |
| Full NF4 cache + fused compact-MLP residual | 23 | `redacted-job` | 60.60s | 8071 MiB | 2.04966 | 1.186x |
| Original Triton compact-MLP residual scatter, no decoded cache | 22 | `redacted-job` | **64.03s** | **5405 MiB** | **2.04960** | 1.122x |
| Original Triton compact-MLP scatter, no decoded cache | 21 | `redacted-job` | 64.83s | **5345 MiB** | 2.04965 | 1.108x |
| 1 GiB decoded NF4 cache + original plain MLP padding | 72 | `redacted-job` | 62.90s | 6721 MiB | 2.04968 | 1.142x |
| Reconstructed R9 fusion (not identical old source) | 42 | `redacted-job` | 61.77s | 8623 MiB | 2.04963 | 1.163x |
| CPU activation offload with MLP padding | 27 | see To Run | **TIME_LIMIT_EXCEEDED at 80.0s** | 3607 MiB (partial run) | not evaluated | **FAILED, not admissible** |

**Claimed competitor target of 45s / 1375 MiB is not supported by a completed result in this queue.** Do not claim it as achieved. The offload example illustrates that a lower reported partial-run peak can come at the cost of exceeding the 80s training cutoff.

### Winning criteria and admission diagnostics

Two post-training test notebooks `chatgpt-speedwinner-79regressions-20261031-v1` and `chatgpt-guarded-mlppad-regressions-20261031-v1` **failed after training**, with generic `RuntimeError('Official Soup regressions failed')`; their complete `pytest` log was exported to GPU workspace but is not included in the Sheets job error message. This means the cause is **not yet identified**: do not say which source feature failed, and do not assume the source is cleared. `harness/check_rules.py` runs `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q -p no:cacheprovider` (with the harness environment) and requires exactly **79 passing tests**. GCP service's three pass flags are insufficient for the original **10 checks** (including timed original `@triton.jit` kernel share >=5%, training loss curves, entropy, accuracy, and rewind/output metadata).

Existing concurrent [row 180](https://example.invalid/redacted) `chatgpt-6052-regression-preflight-20261031-v1` is an **error-diagnostic test** for the 60.52s source. It runs a first-failure regression preflight and should surface the traceback, rather than a generic error. A separately published [backup debug notebook](project-repository/blob/master/soup-kernel-challenge/queue_20261031_official79_firstfailure_debug_v1.ipynb) is **not submitted** to avoid competing duplicate GPU work.

### New independent A/B candidates (submitted after this wave)

- **Row 188:** [60.52s full-cache valid-row RMSNorm / residual + retained native FP16 logits](project-repository/blob/master/soup-kernel-challenge/queue_20261031_fullcache_validnorm_residual_retain_v1.ipynb), source-guarded, pending training.
- **Row 194:** [1GiB cache cap + valid-row RMSNorm + fused residual](project-repository/blob/master/soup-kernel-challenge/queue_20261031_nf4cache1g_validnorm_residual_v1.ipynb), source-guarded; isolated reduction of the 3GiB cache budget; pending training.
- **Row 202:** [No-NF4-cache MLP fused residual + q_proj active-token compaction](project-repository/blob/master/soup-kernel-challenge/queue_20261031_fusedresidual_qproj_padding_v1.ipynb). Patches only q_proj plan tuple parsing to handle the fused-residual's extra mask/prefix fields; retains full dropout sampling geometry and unmodified task/harness. Pending combined GPU correctness and performance.

### Score-focused selection (conditional, not official points)

The official contest uses `35 * ln(speed_ratio) / ln(best_admitted_speed_ratio)` for speed and `30 * ln(memory_ratio) / ln(best_admitted_memory_ratio)` for memory, with improvements below 5% counted as ties and admission prerequisite for any performance points.

**If and only if** an admitted competitor actually achieved the user's 45s/1375MiB targets on a comparable paired baseline (not evidenced here), the projected speed+memory points *excluding* eligibility and documentation would be approximately:
- Fused residual low memory 64.03s/5405MiB: **21.11**.
- Fused scatter low memory 64.83s/5345MiB: **20.32**.
- Valid-row norm/residual full cache 60.52s/8111MiB: **20.13**.
- Absolute fastest 60.24s/8827MiB: **19.40**.

This result means the balanced low-memory candidate **can outrank** the pure fastest candidate under that hypothetical competitor; do not prematurely choose only by seconds.

### Final actions

1. Obtain first regression failure from row180. Repair root cause generically, **never disable behavior specifically in official tests or alter protected files**.
2. Inspect original Triton elapsed-ms / timed-loop from the raw output; ensure 5%-kernel-share threshold and all 79 regressions.
3. Decide speed-versus-memory frontier from actual full-run data. Avoid selecting CPU-offload jobs with incomplete training.
4. Export the **exact verified source files**, rather than only notebook links, into the real competition repo and run pinned Kaggle paired checks before deadline.

### New diagnostic outcome: test runner missing in GCP VM

The concurrent **row 180** first-failure regression preflight `chatgpt-6052-regression-preflight-20261031-v1` failed before any training because the GCP queue image does **not include pytest**:

```text
/usr/local/bin/python: No module named pytest
```

That is the first *confirmed* root cause. It does NOT prove that the current fast solution passes or fails the 79 regressions: previous postrun notebook failures had the same generic `pytest` nonzero status, plausibly the same missing dependency. The correct remedy is to run the unchanged 79 tests in a suitable pre-equipped validation runtime (e.g., the official Kaggle environment), or provision pytest in the separate queue **image** for diagnostic work. **Do not alter the competition's task/harness/tests, spoof test results, silently skip the 79 tests, or treat GCP's simple speed/memory/quality checks as official admission.**

A source-only diagnostic notebook `queue_20261031_official79_firstfailure_debug_v1.ipynb` was published but deliberately not queued, because another agent already submitted row180. It would have hit the same missing-test-runner blocker.

### Newly submitted and verified experiments

- **Row 188:** `chatgpt-validnorm-residual-retain-20261031-v1` — full decoded NF4 cache, fused MLP residual, valid-row norm + exact FP16 forward-logit reuse; last known status queued.
- **Row 194:** `chatgpt-cache1g-validnorm-residual-20261031-v1` — only reduce fullcache cap 3 GiB to 1 GiB on measured 60.52s/8111MiB source; last known queued.
- **Row 202:** `chatgpt-fusedresid-qproj-pad-20261031-v1` — measured lowmem 64.03s/5405MiB fused MLP residual + q_proj active-token compaction. Four-element compact MLP plan made compatible with q_proj's two-field ABI, preserving full original dropout geometry; last known queued.

All are public `.ipynb` files in [project-repository/soup-kernel-challenge](project-repository/tree/master/soup-kernel-challenge) and were added only in A:H, with Ready set last. No completion has been claimed for these three.

### Feasibility note: 1,375 MiB target

The completed CPU activation-offload experiment (row27) reached only a **partial** 3607 MiB peak and exceeded the 80-second training cutoff with no public eval. It cannot serve as a positive result. A 1.5B 4-bit model's quantized parameters, FP16 tied embedding/head, CUDA context, and live activations make 1375 MiB an extreme target on a single T4; meaningful additional VRAM reductions likely require CPU offload or different activation-lifetime management that risks the 80s cutoff. Do not claim an absolute impossible lower bound without a source-level tensor census.
