# Soup Kernel Challenge — Round 21 clean-room fusion research

**Date:** 31 October 2026 (Asia/Almaty). **Scope:** official fixed Qwen2.5-1.5B-Instruct NF4 q/v LoRA training on one Tesla T4. **Objective:** jointly reduce complete training-command latency and external GPU peak without changing model, data, batch order, optimizer, dropout probability, number of steps, metrics, adapter outputs, or measurement harness. **Status:** three fresh experiments were submitted via Google Sheets; VM intake and CUDA results are **pending** at this handoff.

## Source/attribution boundary

Read and synthesized the **algorithmic ideas only** from:
- https://github.com/external-contributor/hackathon-decentrathon-6 — self-reported paired Kaggle T4 85.0→37.7 s and 14,345→3,461 MiB, 10/10 checks, original kernel share 12.4%, code commit `redacted-revision`. **Not reproduced independently.**
- https://github.com/external-contributor/soup-kernel-challenge — approximate claims ~48 s / ~2,850 MiB in README; SUBMISSION.md still has placeholders. **Not independently validated.**
- User-supplied research notes, `Вставленный текст(2).txt`, grouping vocabulary-major loss, FP16 normalization, fused rotary, one-per-layer NF4 decode, compact LoRA autograd, full decoder unpadding, deferred host synchronization, startup, and testing.

**No rival kernel source was copied into the implemented candidates.** All candidates begin from our own, previously submitted and publicly documented `project-repository` development notebooks, and add our independently written PyTorch 2.11 native attention wrapper. Review the exact GitHub revisions linked below.

## Prior completed public development evidence — not official paired admission

| Source-bound public development run | Whole `soup train` | Peak GPU | Public eval loss |
| --- | ---: | ---: | ---: |
| Saved service baseline | 71.85 s | 14,345 MiB | 2.04968 |
| Our prior speed-focused q_proj/NF4/MLP candidate | 59.50 s | 9,125 MiB | 2.04967 |
| Our prior memory-focused MLP/scatter candidate | 64.83 s | 5,345 MiB | 2.04965 |

These fast and low-memory results are **different solutions**. Our 59.5-second result does not establish compliance with the 5% *own-original-kernel-share* rule on every hidden-data run. Other team agents are testing original-share and 79-regression checks; some profiling notebooks failed **after** completed training. Preserve provenance.

## New original implementation: R19 native variable-length whole-decoder packing

Source: `soup_r19/varlen_decoder.py` (SHA256 `9525a3eff33f6c2f0f9e29ff84c6e28e20087ce0bb51cc804b3034053bb82c49`); standalone CPU test `soup_r19/test_cpu.py` passes five mask patterns, shape/row-order, scatter and gradient checks. `soup_r19/gpu_gate.py` adds physical T4 native attention fwd/bwd and a tiny Qwen2+PEFT q/v-LoRA differential training test, including nonzero adapter B and nonzero dropout execution. All GPU test results pending.

**Method:**
1. Only at a guarded `Qwen2Model.forward` training boundary, identify strictly right-padded `[B,T]` input IDs and attention mask. Preserve each row's token order and position IDs.
2. Compress valid positions to `[1,N]`. Avoid constructing a dense block-diagonal attention matrix (`R17`'s weakness); pass the per-row cumulative lengths as public PyTorch ATen `cu_seqlens` to one native efficient-attention call.
3. In each decoder layer, keep the original three q/k/v projections and original Qwen rotary transform. Convert attention operands to T4-compatible FP16 only at the numerical boundary, materialize grouped keys/values conservatively for the first trial, and call `aten::_efficient_attention_forward` with native causal row isolation.
4. Custom autograd uses the corresponding pinned `aten::_efficient_attention_backward` and propagates dQ/dK/dV through the original PyTorch projection/LoRA graph. Native attention is used because PyTorch is an **allowed** library; it is NOT counted as an original Triton contribution.
5. Scatter the final hidden states to original `[B,T,H]` before any original head/loss/TRL metrics/Rewind consumer. Masked positions return zeros and no gradient. Model cache/inference/eval/unrecognized layouts use untouched native model **before consuming dropout RNG**. Once inside a packed step, exceptions raise rather than silently replay the model and duplicate dropout RNG.

**Critical risks:** Actual T4 ATen operator availability; error between native SDPA and FP16 attention after rotary; GQA repetition overhead; altered LoRA Philox RNG consumption due packed tensor shapes; possible shape-specific hooks in our own integration; native group count; any global-state patch affecting test regressions. Numerical gates enforce bounded errors but are not an official whole-training quality proof.

## New fused candidates

| Candidate | Our own existing modules + clean-room additions | Published notebook | Google Sheets |
| --- | --- | --- | --- |
| **R19 (speed control)** | Existing measured full frozen NF4 cache, compact MLP, q_proj padded-token optimization; new whole-decoder native-varlen attention | [R19](project-repository/blob/master/soup-kernel-challenge/queue_20261031_r19_fastbase_varlen_cleanroom_v1.ipynb) | [To Run row 39](https://example.invalid/redacted) |
| **R20 (head fusion)** | R19 family + our own padded attention `o_proj` compaction and original vocab-sliced/cuBLAS Triton online CE/entropy/accuracy/Rewind head | [R20](project-repository/blob/master/soup-kernel-challenge/queue_20261031_r20_fullfusion_vocab_varlen_v1.ipynb) | [row 43](https://example.invalid/redacted) |
| **R21 (maximal compatible fusion)** | R20 + our independently written FP16 output boundary for frozen RMSNorm, one-sync TRL metric values, and a one-shot package-distribution mapping reuse with pre-model `gc.freeze()` | [R21](project-repository/blob/master/soup-kernel-challenge/queue_20261031_r21_maxfusion_original_v1.ipynb) | [row 44](https://example.invalid/redacted) |

Each notebook preserves its original `--only solution` harness and final small-report export cell byte-for-byte. Every solution module is installed into `SOUP_REPO/solution/`, with a zero-argument `apply()` call before native model/trainer construction. The new gate code runs in a distinct process before measured training, with no usable persisted compiled cache. No Soup config switches, task files, harness files or protected tests are changed. New rows were written in `To Run!A:H`, and `Ready` was set **TRUE last** after readback. No secrets are embedded.

### Checks already completed

- Python3 syntax parse: `varlen_decoder.py`, `gpu_gate.py`, `startup_once.py`, CPU test scripts.
- CPU packing/scatter: five shape/mask patterns, preserved row identity and exact gradient scattering, PASS.
- Startup one-call mapping / no repeat installation local Python test PASS; new module SHA256 `315c107a5d34fce2bd4d2369bbce837a9836987e0aa5044b2f20b830982a609e`.
- GitHub notebook round-trip and source cell inventory: verified R19/R20/R21; base measurement/report-export preserved; no hardcoded access token.
- Google Sheets queue request inputs and `Ready=True` readback: verified R19/R20/R21.
- **No actual R19/R20/R21 physical GPU test or 64-step training result available at this handoff.** Do not infer quality, kernel share, peak, training speed or victory from queue submission.

### Included versus postponed competitor-inspired concepts

- **Included conceptually via our independent code:** whole-decoder valid-token packing, native row-isolated varlen attention, own vocabulary-sliced CE+metrics, existing NF4 cache, own MLP/input activation compaction, Q and o projection optimization, FP16 RMSNorm boundary, one-sync metrics and safe startup mapping.
- **Not enabled in R21:** foreign fused attention or RoPE kernel code; reference SwiGLU kernel as **ours**; competitor source implementation; cross-layer NF4 grouped kernel with lifecycle unproven in our candidate; LoRA dropout RNG state-replay autograd; offloading model weights/activations; external forbidden libraries; precompiled cache; background compilation; compiler build flag modifications. These require separate independent implementations and numerical/quality proof. Our original NF4 cache is an alternative to one-decode-per-layer, not a claim of implementing that identical mechanism.
- **Risk:** R21's one-sync metrics and FP16 normalization overlays are independently authored development additions but may interact. A fresh actual trainer integration and 79 regression checks are mandatory even if tiny-Qwen passes.

## Memory/time reasoning and remaining targets

- **Competing claims:** external contributor reported 37.7s /3,461MiB; external contributor approximate 48s /2,850MiB. Our strict target from the user is **<45s AND <1,359MiB**, but to beat external contributor's *reported* speed we also need <37.7s.
- NF4 packed frozen decoder weights ideally ~624.75 MiB; exactly representable tied FP16 embedding ~445.125 MiB. Together ~1,069.88 MiB **before NF4 codebooks, scales, adapter parameters, CUDA runtime, activations, scratch and transient copies**. The 1,359MiB target therefore likely requires further memory-lifetime control or moving some frozen tensors to CPU; do not describe the 1,070MiB figure as full-model feasibility.
- Published CPU activation offload public test reached a **partial** peak of 3,607MiB but exceeded the 80-second cutoff. It did not prove a completed run. Avoid naive blanket offload.
- More compute savings require reducing true attention pairs `sum_i L_i^2` rather than merely the dense packed `N^2` mask; this motivates the R19 native efficient-varlen module. No assumption of the T4 operator's support is made until a physical gate passes.
- Reducing setup cost matters because the jury scores **the entire Soup command** including cold import/setup/save; however no precompilation cache or background measurement bypass is used. The R21 startup optimization is deliberately conservative.

## Decision gate for promotion

For each request read `To Run!I:U`, `X:Z`, `BS` and the matching `VM Runs` / `VM Details`. If CUDA gate or tiny-model PEFT comparison fails, preserve error and adjust independently; **do not mark a training speed result**. If it completes, verify the actual Qwen64 updates, eval loss <=1.005*paired base, time/VRAM <1.10*paired base, all logged metrics, Rewind row order and token counts, adapter names/dtypes, 79 regression tests, and useful original `solution/` Triton kernels >=5% of timed steps 3+. Then do a second clean run under another public seed/dataset slice and let the jury use paired hidden rows. Do not rely on an unpaired service badge alone.

**Final competition submission remains a separate platform team repository, public Kaggle notebook with saved output, README, SUBMISSION.md, video, presentation by 2026-10-31 23:59 Astana.** The current GitHub/Sheets results are research transport and do not replace that final submission.

## R22 memory-first counterpart (added later in the same research session)

**R22** uses the team's **lower-memory** existing frozen storage + compact MLP path (no persistent full NF4 decoded cache) with the same independent R19 whole-Qwen native variable-length attention. It preserves the old `--only solution` harness and export cells and runs the same physical T4/q-v LoRA numerical gate first. It therefore isolates whether true whole-decoder unpadding helps the memory frontier, rather than assuming an NF4 cache optimized for speed is also memory optimal.

- Source: [original R22 low-memory varlen notebook](project-repository/blob/master/soup-kernel-challenge/queue_20261031_r22_lowmem_varlen_original_v1.ipynb).
- [Google Sheets row 45](https://example.invalid/redacted), request `chatlora-r22-lowmem-varlen-20261031`, `Ready=TRUE` set last and verified. At last poll: **Ready — awaiting intake**, not a measured result.
- R22 is the **memory-first** branch; R19/R20/R21 are **speed/broad fusion** branches. No score can be transferred between them, and no measured joint result exists yet.