# Soup Kernel Challenge — competitor source review & original packed-decoder experiment

**Research date:** 2026-10-31. **Objective:** under 45 seconds whole training command and under 1,359 MiB sampled Tesla T4 peak, without altering the fixed task or violating the original-kernel rule. **Neither stretch target has been demonstrated.** Source-derived details, numerical experiments, and unverified claims are distinguished below.

## 1. Competitors: verified versus unverified evidence

| Public repository | Technical mechanism | Evidence quality |
| --- | --- | --- |
| [archived-source/archived-challenge-experiments](https://example.invalid/redacted-github) | Valid-token packing within each batch, block-diagonal causal attention, unchanged padded-shape LoRA dropout RNG, compact streaming CE, original fused MLP/RMSNorm/RoPE/LoRA kernels. | **Saved paired Kaggle result and checks.json:** base 86.36s/14,345MiB vs solution 75.74s/4,611MiB; eval 2.04960→2.05084 (+0.0605%); countable original-kernel share 9.566%; **10/10 admission checks** on public seed. Hidden jury unknown. |
| [archived-source/soup-kernel-challenge](https://example.invalid/redacted-github) | Original residual + RMSNorm fusion and next-layer norm fusion, compact vocabulary head, original NF4 decoding, selective MLP rematerialization. | README reports paired ~66.2s/5,419MiB and 10/10 checks; direct raw `results/` was not present at inspected repository root. Source code verified. |
| [archived-source/soup-kernel-challenge](https://example.invalid/redacted-github) | Original fused CE+entropy+first-index argmax, NF4 nested dequant, RMSNorm/add, right-pad packing, exact FP16 retained gate/up values. | Inspectable source; README's results table still contains `<BASE_S>` / `<OURS_S>` / `<CHECKS>` placeholders. **No numeric win verified.** |
| [external-contributor/soup-kernel-challenge](https://example.invalid/redacted-github) | Fused chunked cross-entropy; FP16 RMSNorm/attention data path; custom frozen NF4 MLP autograd; original NF4, RoPE and RMSNorm kernels. | README states roughly 48s/2,850MiB and 11.5% kernel share, but no `results/` directory or matching recorded measurements were found. **Treat as an unverified claim, not leaderboard evidence.** |
| [archived-source/soup-kernel-challenge-fork](https://example.invalid/redacted-github) | Original quantized head/logit LSE, scalar target NLL, row-wise RMSNorm, RoPE, narrow trim of padding edges. | Source/approach inspected; README's benchmark slots not filled. |

### Verified direct competitor source details

The real [Rufus saved checker](https://example.invalid/redacted-github) includes 10 checks, all `ok=true`, two base and two solution runs, 79 Soup tests, and 9.6% minimum useful kernel share. Their [independent verification](https://example.invalid/redacted-github) records source commit `cd06029bbc65ccb504924618230ecabf9acdc5f0`, baseline repeats 88.96/83.76s and solution repeats 75.82/75.66s. **Cross-session speed comparisons against our queue baseline are not paired or statistically fair.**

Rufus ablations show standalone within-batch packing reduced time by 16.51% yet passed only **8/10** tests; combining padding-aware streaming CE, frozen MLP activation compression, original norms and LoRA kernels produced **10/10**. Their cached NF4 weights raised peak memory from 4,611 to 7,131MiB and reduced their weighted balance versus no cache. These are source-recorded ablations, not our measurements: [experiment catalog](https://example.invalid/redacted-github), [NF4 cache trade-off](https://example.invalid/redacted-github).

## 2. Transferable ideas and what we already have

| Competitor idea | What our research has already done | Missing proof |
| --- | --- | --- |
| One-pass nested NF4 decode | Original fused NF4 code exists, with public T4 results. | Integrate actual decode directly with useful tensor-core GEMM to save a materialized full layer. Direct signed INT4 multiplication is invalid for nonlinear NF4. |
| Stream vocabulary CE, entropy, argmax, row losses | Original streamed head exists; native numerical fallback was needed for saturation/nonfinite rows. | Original gradient in the forward could avoid backward logits recomputation, **but FP16 GradScaler ordering and exceptional rows must be proven**. |
| Preserve FP16 frozen embedding and MLP activations | Our low-memory candidate already retains losslessly representable FP16 values. | Better lifetime control, no duplicated retained fp32 casts. |
| Fused residual / RMSNorm | Our original fused padding/scatter and valid-row norm experiments run in public T4 tests. | Real input-gradient/PEFT parity and official paired tests for each selected version. |
| Whole-batch valid-token packing | **New independent implementation below.** | Actual single-T4 Qwen/PEFT/LoRA-dropout test, 64-step quality and all official gates. |
| Cache frozen NF4 weights | 1GiB/2GiB/full-cache T4 data show a speed-memory Pareto curve. | Full cache can eliminate most recurring original NF4 kernel time and fail the 5% requirement. |

## 3. Our new original candidate: whole-decoder valid-token packing

**Implementation created this round (no copied competitor kernels):**

- [Original source, publicly inspectable](project-repository/blob/master/soup-kernel-challenge/research_original_full_decoder_pack.py)
- [Complete queued notebook](project-repository/blob/master/soup-kernel-challenge/queue_20261031_original_fullpack_lowmem_v1.ipynb)
- Google Sheets `To Run`, row 36: `chatgpt-competitor-original-fullpack-20261031-v1`.

The algorithm:

1. Confirm a standard right-padded `(B,T)` attention mask and no supervised label at a padding-only position. Keep batch composition, order, labels and optimizer unchanged.
2. Gather all real tokens, in original row-major order, into `(1,N)`. The additive attention mask allows `(i,j)` only if tokens belong to **the same original example** and `position(j) <= position(i)`. Qwen's original decoder, projections and SDPA remain in use.
3. Preserve **original Q/V LoRA dropout RNG**: invoke native dropout on an `(B,T,D)` tensor with exactly the original dtype/shape to consume the same Philox stream. Select only retained mask positions for the packed activations. The custom autograd bridge applies and differentiates those masks without changing trainable adapters.
4. Scatter final decoder hidden states back to `(B,T,H)` before the original streamed head, TRL metrics and Soup Rewind interface. Keep padded outputs non-contributing where the attention and supervised label masks exclude them.
5. The new notebook runs a separate real tiny Qwen+PEFT T4 forward/gradient/RNG preflight; only on success does it execute the **unchanged 64-step `--only solution` harness**, export reports, then compute countable original kernel share using `harness/check_rules.py`'s source classification.

CPU-only mathematical tests executed locally: **five packed-versus-dense causal attention forward/gradient shapes**, including `(4,512)`, and **four native-dropout mask/RNG tests**, all passed. These do **not** establish physical T4 forward/gradient or hidden-row equivalence. The remaining original NF4/RMSNorm kernels come from our own prior source. Do not count the supplied starter SwiGLU or padding masks as original kernel work.

### Current intake status

The new row was written A:H with **Ready TRUE last**, notebook URL public, no token embedded. **No VM job ID or GPU result was visible at the last check**. The snapshot of `VM Queue` still showed the last worker status update at 2026-10-31 10:27 UTC; the earlier far-down admission-audit rows 440/460/480 also remained `Ready — awaiting intake`. This may be a queue intake problem or delay. Do not invent a result or resubmit unchanged work with another ID while the first is unacknowledged.

## 4. Reality check: faster alone does not equal admission

Our saved public T4 baseline: **71.85 seconds / 14,345MiB**.

- Fastest completed public single solution: **58.78s / 9,325MiB**. But a source-family post-run checker rejected **1.967% original useful kernel share**, below the mandatory 5%.
- Lowest completed public single solution: **66.33s / 5,005MiB**; 79-regression and own-share gates are not yet proven for the exact source (a new audit is in the queue).
- Neither is a paired jury result. **Both stretch targets of <45s and <1,359MiB remain unmet.**

Rufus's directly inspected checked-paired 4,611MiB benchmark confirms that full decoder packing is worth our effort, but their 75.74s on a different baseline session cannot predict our new T4 result. Competitor source code is a design reference, **not code to copy**.

## 5. Next independent tests

First test the fullpack implementation's real Qwen2 PEFT integration, dropout RNG and gradients. If it passes, keep full loss/entropy/Rewind comparisons and 5% kernel-share gate. If not, preserve the exact failing artifact and diagnose the source, not the benchmark definition. Only then consider combining valid-row packing with a separately developed original FP16 `mma.sync` NF4 fusion to target substantial compute savings and a larger, legitimate counted-kernel share.

Relevant official rules: [source/own kernel checks](https://example.invalid/redacted-github), [scoring](https://example.invalid/redacted-github). The developer queue's saved baseline is not official Kaggle admission.