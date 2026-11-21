# Qwen-Inspired Kernel Optimization for the Soup Kernel Challenge

Research date: 30 October 2026 — updated after research round 6  
Target: Soup 0.77.0, Qwen2.5-1.5B LoRA training, one Kaggle T4  
Inspected starter commit: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`

## Session handoff

**Goal:** Improve full-command training time and peak GPU memory for the fixed Soup Kernel Challenge on one Kaggle T4.

**Status:** Six research rounds completed. Exact dependency/compiler sources inspected; CPU checks cover streaming reductions, supervised-only head gradients, LoRA algebra, all 63,488 finite FP16 encodings, compact MLP state, attention ordering/layout, and interface contracts. The latest audit identifies a conditional baseline AMP compatibility problem, corrects final adapter-storage assumptions, and supplies concrete trainer and layout integration designs. No candidate Triton kernels implemented; no GPU, PyTorch, or Triton runtime is available in this research environment. No T4 speed, peak-memory, or competition-quality result is claimed.

**Next:** Run the unchanged baseline through its first optimizer update on the actual scoring image, recording package versions, final adapter/gradient dtypes, scaler state, and any failure. Section 13.1 explains the audited BF16/GradScaler incompatibility and includes an isolated diagnostic probe. After baseline compatibility is established, develop **L1 diagnostics and M1 compact MLP saved state in parallel**, then the streamed head with G0 and library GEMMs. Use the guarded trainer integration, explicit AMP boundaries, stride-aware K1 backward, and R1/P1 contracts in sections 12–13. Compare the P1 query-layout variant only on the verified math backend. Select through the official harness.

**Latest findings — section 13:**

- **Check the baseline before kernel tuning.** TRL 0.29.1 casts trainable quantized-model parameters to BF16 after PEFT setup. The examined Torch 2.11 CUDA GradScaler unscale dispatch excludes BF16. This predicts a compatibility problem for that exact source combination; the unpinned scoring Torch must be checked directly.
- The final expected adapter storage is **4.15625 MiB in BF16**, not the earlier 8.3125 MiB FP32 estimate. Effective adapter GEMMs still use FP16 under the task's autocast. A1 backward must preserve `RN16(2 * RN16(dY32))`.
- Assigning `outputs.logits = None` can leave the dense tensor in the `ModelOutput` mapping. Construct a fresh compact output, coordinate trainer/model fallback before forward, and require exactly one current summary. Eighteen stdlib semantic mock checks passed.
- Efficient attention's apparent output reorder is already a view. A different P1 query layout may remove a **12 MiB FP32 reshape clone and 24 MiB logical traffic per maximum-shape math-attention layer**. This is source-derived, not a GPU result.
- K1 backward must accept token-major gradients even when its forward writes head-major outputs. The first K branch can omit our external backward while the efficient backend still computes its internal dK.
- Default Triton NVIDIA autotuning uses a **256 MiB cache-flush buffer** and repeated launches. Keep the scored path to a small, already selected configuration set; changing the launch grid alone does not require another compiled specialization.

Section 13 contains the compatibility evidence, scoped integration design, layout tables, new CPU checks, and an explicitly unexecuted CUDA diagnostic. Earlier plans are conditional on the actual baseline passing its preparation and update path.

**Established findings from round 5 — section 12:**

- Frozen-gamma RMSNorm already retains its input and one row statistic, rather than all its forward intermediates. R1 mainly removes temporary traffic and launches. Start with the native operation-order backward; residual fusion must preserve the residual output needed by the skip branch.
- Source analysis predicts **57 RMSNorm forwards and 56 backwards**. The first frozen K branch also has no gradient path. Combined autograd outputs must preserve those flags and avoid materializing unused zero gradients.
- A conservative new P1 candidate fuses **query RoPE and the following FP16 cast**. Casting K before the existing masked-path repetition changes grouped backward accumulation; the key extension needs a different boundary.
- Native TRL entropy already uses stable log-softmax. Our reducer targets duplicate passes and temporaries. Triton 3.6 `tl.exp` is approximate; disabling multiply-add fusion does not turn it into libdevice exponential.
- A source-based traffic/launch model shows why the old 44% elementwise profile cannot establish the final candidate's 5% share. The report now separates logical bytes, launched grids, arithmetic programs, and measured event time.

Section 12 contains the new implementation contracts, CPU evidence, and the next focused experiments. The conclusions remain subject to actual T4 measurements and official gates.

**Established findings from round 4 — section 11:**

- **Triton 3.6's ordinary `tl.dot` path falls back to scalar FMA below SM80, including T4/SM75.** Existing Turing instruction strings deeper in the compiler do not establish that ordinary `tl.dot` reaches them. Keep cuBLAS GEMMs initially; a custom SM75 tensor-core path is a separate feasibility project. The scoring image's actual versions still need verification.
- Reconstruct the FP16 logit gradient **once**, then feed it to the head input-gradient GEMM. Naively reconstructing inside every hidden-dimension tile repeats exponentiation 12 or 24 times for the examined tile widths.
- Full offload of the 1,960 MiB maximum-shape compact MLP state moves **3,920 MiB per step** out and back. Across 64 maximum-shape steps, a hypothetical 12 GB/s link needs **21.92 seconds of copy service before overlap**. Start with a small prefix of layers and bounded backward prefetch.
- Memory gains must be evaluated by phase. L1 still retains native CE state; M1 and its offload savings overlap; the head compute weight must be counted once. Allocated bytes, reserved bytes, and the externally sampled peak differ.
- The 5% requirement uses **summed synchronized step intervals for steps 3–64**, and the lowest qualifying share across repeats. Per-launch event instrumentation and cold compilation are real costs; fewer substantive launches can matter more than isolated kernel speed.

Section 11 gives the kernel feasibility constraints, transfer budgets, memory-lifetime rules, and command-time economics. Its estimates are calculations, not GPU predictions.

**Established findings from round 3 — section 10:**

- The frozen MLP's gate/up outputs are expected to be exact FP16 results widened to FP32. Saving only their FP16 representations and recomputing native FP32 SiLU can reduce calculated maximum retained MLP state from **5,880 MiB to 1,960 MiB**, with no extra large GEMM. This is a storage estimate, not a GPU-peak measurement.
- Save **`m` and `log_a` separately** for head backward. Collapsing them into one FP32 `logZ` loses normalizer precision at large common offsets.
- Chunked/recomputed GEMMs can round differently. Pair each normalizer with its logits; saved supervised logits are the conservative first policy.
- SDPA already receives FP16 inputs under baseline autocast. A conditional T4 opportunity instead comes from **maskless native GQA selecting the math path** under the examined PyTorch 2.11 backend rules.
- Fusing K/V repetition with casting can avoid expanded FP32 temporaries on the existing masked path. Its backward must promote before the six-head sum.

Appendix B contains the round-3 verification scripts. The excluded SwiGLU work and a pure Torch attention adapter do not establish the original-Triton runtime requirement.

**Established findings from round 2:**

- Metrics consume FP32-converted values of FP16-rounded head logits. Match both stages.
- PyTorch 2.11 autocast does not cache casts of frozen weights. A naive head-chunk loop can repeatedly cast the entire head; one FP16 representation is 445.125 MiB.
- Only supervised prediction positions need the terminal head's backward matmul. Prompt positions still need forward logits for entropy, and prompt decoder gradients must remain intact.
- bitsandbytes 0.50.2 already has a T4 fused NF4 forward family, but its source dispatch falls back for ordinary longer padded training batches. Its input-gradient backward always dequantizes before matmul when that gradient is needed.
- Dense NF4 projection weights are temporary, with a largest individual FP16 matrix of 26.25 MiB. Caching all of them would retain 2,499 MiB; reconstructed FP32 block scales total about 78.1 MiB.

These are source findings and calculations, with the proposed optimizations still subject to runtime verification. Sections 3–6 contain the implementation decisions; section 8 and Appendix A preserve the CPU-check evidence.

**Preserve:** Shifted supervised labels, normalization, FP16 AMP scaling, prompt-inclusive entropy, accuracy, rewind values, and independent q/v dropout at 0.05. Keep all fixed training settings and batches unchanged. Original qualifying Triton kernels must meet at least 5% runtime share; rewritten starter SwiGLU does not qualify. All official correctness and regression gates remain mandatory.

## Executive overview


> **Recommended first build**
> Begin with shared diagnostics and compact MLP saved state. Then develop the streamed output head with explicit weight-cast reuse, centered normalizers, and supervised-only backward. Retain library GEMMs until compiler inspection and a T4 test establish an efficient custom path. Pursue original normalization, logit-gradient, LoRA epilogue, and attention cast/repeat kernels alongside it. Select further work from the measured full-command bottleneck.

### Decision and evidence

Qwen's published optimization experiment offers useful ideas about parallel work, data reuse, stable reductions and sustained measurement. This report adapts those principles to the fixed Soup training workload, the competition's actual scoring, and the requirements of FP16 training on a Kaggle T4. [1-3]

**Evidence level:** source inspection establishes control flow and eligibility; arithmetic establishes tensor sizes; the CPU checks establish limited algebraic behavior. None establishes GPU performance, the final adapter's quality, or a winning rank.

| Fixed target | Value |
| --- | --- |
| Model and training | Qwen2.5-1.5B-Instruct; Soup 0.77.0; NF4 4-bit weights; LoRA rank 16, alpha 32, q/v adapters |
| Workload and hardware | 256 instruction rows; 64 steps; batch 4; sequence length up to 512; one Kaggle T4; FP16 |
| Scoring emphasis | Speed 35 points; peak memory 30; correctness/admission 20; README and reproducibility 15 |
| Submission deadline | 31 October 2026, 23:59 Astana / Almaty time (UTC+5) |

The target is the attached Soup challenge. The earlier H200 / Open Kernel Challenge project uses different hardware and workloads; its optimization priorities should be evaluated separately. [2-4]

### Dependency versions: what is actually fixed

The inspected Kaggle installation cell pins Soup **0.77.0**, Transformers **5.16.1**, PEFT **0.20.0**, TRL **0.29.1**, bitsandbytes **0.50.2**, and Accelerate **1.14.0**. It does **not** explicitly pin Torch or Triton. The brief's Torch 2.11 / Triton 3.6 environment is the reference used for the corresponding source audits; record and recheck the installed versions on the scoring image. The starter HEAD was rechecked and remains the commit above. [13]

## 1. What transfers from Qwen

Published evidence and our workload-specific interpretation

Qwen reports a **10.0x geometric-mean speedup** for SGLang Extend Attention on T-Head ZW-M890 PPUs after about **35 hours, 432 kernel evaluations and 1,158 tool calls**. This is a vendor-reported kernel result, not a prediction for Soup training. [1]

| Published change | Cumulative result | Our adaptation |
| --- | --- | --- |
| Split KV work; merge partial results | 2.58x | Tile large reductions and merge stable summaries. |
| Reduce allocation and synchronization | 5.37x | Reduce temporary tensors and avoid unnecessary host waits. |
| Choose splits by workload size | 6.85x | Use a small dispatch table based on real tensor shapes. |
| Tune registers, buffers and reductions | 8.50x | Balance occupancy, data reuse and register pressure on T4. |
| Reuse K/V across four queries | 10.0x | Combine consumers of the same data, including loss and diagnostics. |

The reported milestones are cumulative. They cannot be multiplied together. The adaptation column is our engineering proposal, not a Qwen benchmark result. [1]

### Soup's larger opportunity is outside attention

| Operation | Approximate share of GPU time |
| --- | --- |
| Elementwise operations / casts | 44% |
| Matrix multiplication | 36% |
| Vocabulary softmax / cross-entropy | 7% |
| Attention | 5% |
| NF4 unpacking | 5% |

Organizer profile: approximate shares of GPU time within a training step. The listed categories total about 97%; the table does not imply that they cover all work. These are not fractions of full command time. [2,3]

Using that profile, a hypothetical 10x attention improvement gives about **1 / (0.95 + 0.05/10) = 1.047x** across GPU work. Hypothetically halving the whole 44% elementwise/cast portion gives about **1.28x**. Whole-command gains would be smaller once startup, compilation and saving are included.

> **Priority:** first reduce repeated tensor traffic and large live intermediates. Reconsider attention only if the combined implementation's measured profile makes it a meaningful bottleneck.

## 2. Ranked experiment portfolio

Research priorities; no speed or memory gains have yet been measured

| Order / experiment | Hypothesis | Specific acceptance condition |
| --- | --- | --- |
| **1. Shared diagnostics, then streamed head** | Remove redundant reductions/copies; bound logits residency; reuse head casts; restrict head backward to supervised positions. | Correct labels, denominators, entropy, accuracy, rewind, AMP rounding and gradient flow. |
| **2. Compact MLP saved state** | Save proven FP16-origin gate/up values compactly; recompute only native FP32 activation work. | Verify producer dtypes and unchanged arithmetic; do not count a starter SwiGLU derivative as original qualifying work. |
| **3. Original RMSNorm / residual fusion** | Reduce reads, writes and launches. | Preserve FP32 reduction and input/output dtype behavior; validate gradients; do not count no-op FP32 casts as savings. |
| **4. LoRA epilogue / backward fusion** | Reduce rank-16 temporaries and scale/add/cast operations. | Preserve dropout, q/v mask independence, A/B gradients and adapter layout. |
| **5. Attention cast/repeat and conditional dispatch repair** | Remove expanded FP32 copies; allow an efficient backend if maskless native GQA causes math fallback. | Trace the actual branch; preserve FP32 grouped-gradient accumulation; validate backend-change variants separately. |
| **6. Projection recomputation / NF4 input-gradient fusion** | Trade retained activations for replay, or combine packed-weight decode with GEMM. | Preserve dtype boundaries and double quantization; satisfy the complete-command time gate. |
| **7. Within-batch unpadding** | Avoid padding work without changing the batch. | Preserve sequences, positions, masks, row identities, token counts and dropout. |

### Optimize the scoring objective

For admitted teams, define **s = baseline time / candidate time** and **m = baseline peak / candidate peak**. The published improvement terms are: [2,8]

```text
speed points = 35 * ln(s) / ln(best admitted s)
memory points = 30 * ln(m) / ln(best admitted m)
```

Ratios below 1.05 count as a tie with the baseline. Memory breaks equal-total ties. Keep several candidates that offer different time/memory tradeoffs: a modestly slower candidate can still be valuable if its memory improvement is large and all admission gates pass. [2,8]

### A practical selection policy

- First establish a valid combined candidate with substantive original kernels.
- Retain alternatives unless another candidate is at least as good in both time and memory.
- Spend additional work on the largest measured bottleneck, accounting for compilation and validation time.

## 3. Stream the head and diagnostics

The main architectural experiment

At the maximum padded shape, logits contain **4 x 512 x 151,936 = 311,164,928 values**. One FP32 tensor is **1,244,659,712 bytes, about 1.16 GiB**. Actual batches may be shorter. This is one tensor, not the total memory peak or a promised saving. [4,9]

Changing only cross-entropy after full logits have already been built has a smaller memory ceiling. The head computation itself must be chunked or tiled to avoid the complete batch-by-sequence-by-vocabulary allocation. Apple's Cut Cross-Entropy is useful algorithmic background; its ready-made implementation is not our proposed submission. [11]

### First isolate duplicated diagnostics

The pinned stack performs overlapping vocabulary work: training CE, TRL entropy, TRL argmax accuracy, and Soup's per-row CE. TRL's accuracy path calls `.contiguous()` on the logits with the final sequence position removed. With batch size greater than one, this slice can require another almost-full vocabulary-sized copy. [10,15,17]

**Experiment L1:** retain the ordinary model loss and its backward; use one original tiled reducer for entropy, argmax, and row losses. Run it inside the model wrapper on FP16 logits, with FP32 arithmetic, and return compact summaries instead of logits. This can also avoid Accelerate's full FP32 returned-logits copy. L1 still materializes internal dense logits and preserves the original CE backward state, so its memory benefit is bounded. The integration contract is below.

### Proposed forward data flow

- **Compute a bounded token chunk.** Initially retain efficient library GEMMs using the same FP16 compute operands; reuse an explicitly cast head weight across chunks.
- **Run original Triton reductions.** Accumulate loss, entropy, argmax/correctness and per-row statistics over vocabulary tiles.
- **Keep compact state.** Compare saving supervised logits with saving supervised hidden rows for recomputation; release prompt logits after their diagnostics are reduced.
- **Continue through the batch.** Preserve original row identity and aggregation semantics.

### Stable sufficient statistics

For each token with logits z, a tile reduction can maintain a maximum m, an exponential sum a, and a centered weighted sum b. Compatible partial states can be merged after rescaling to a common maximum. In real arithmetic:

```text
m = max_j(z_j)
a = sum_j exp(z_j - m)
b = sum_j exp(z_j - m) * (z_j - m)
logZ = m + log(a)
CE = logZ - z_target
entropy = log(a) - b / a
```

Track the argmax index alongside these reductions. This replaces repeated full-vocabulary passes with compact summaries; floating-point agreement still needs to be tested against the baseline.

**Round-3 numerical correction:** the identities above are in real arithmetic. In an FP32 implementation, retain `(m, log_a)` with `log_a=log(a)` separately. Use `CE=-((z_target-m)-log_a)` and `p=exp((z-m)-log_a)`. A rounded `logZ=m+log_a` should not be the only saved backward state. Section 10 shows the stress-test evidence and the pinned native log-softmax arithmetic. [31]

For two states `(m1, a1, b1)` and `(m2, a2, b2)`, the correct merge is:

```text
M = max(m1, m2)
w1 = exp(m1 - M)
w2 = exp(m2 - M)
A = w1*a1 + w2*a2
B = w1*(b1 + a1*(m1-M)) + w2*(b2 + a2*(m2-M))
```

The offset terms in `B` are necessary because each tile used a different center. Preserve the lowest vocabulary index on equal maxima, and guard masked lanes so an expression such as `0 * -inf` cannot contaminate the entropy sum. CPU tests of the merge are recorded in section 8; they do not establish a Triton reduction's rounding behavior.

> **T4 design choice:** use manageable vocabulary tiles. A simple power-of-two full-row program would pad 151,936 entries to 262,144. Triton's introductory fused-softmax design assumes rows fit on-chip; that assumption must not be carried over blindly. [12]

### Avoid recasting the frozen head for every chunk

PyTorch 2.11's autocast cache requires the FP32 source tensor to have `requires_grad=True`, along with other conditions. The frozen tied head fails that condition. Repeated calls to the original head can therefore repeat its whole FP32-to-FP16 conversion. This is a source-based prediction to confirm with a cast trace on the actual image. [14]

The head contains `151936 * 1536 = 233,373,696` values. Its FP32 source occupies **890.25 MiB** and one FP16 compute representation occupies **445.125 MiB**. Keep the original tied FP32 parameter and embedding behavior; introduce only a non-trainable runtime compute tensor for the head. [9; calculated]

For L2/L3 custom head execution that controls saved-weight lifetime, compare:

| Representation lifetime to test | Expected tradeoff, not a measured result |
| --- | --- |
| Once per head-forward phase; recast once for head backward | Adds another full cast. A lower peak requires an actual intervening allocation peak; adjacent forward/backward phases alone provide no useful release window. |
| Create at head forward; retain until head backward completes | Preferred initial policy: one cast supplies both phases. Route all consumers through the same representation, then release when backward no longer needs it. |
| Retain across training steps | Avoids more casts while keeping 445.125 MiB live during decoder work; test only if the memory tradeoff is favorable. |

This representation must remain separate from the source `Parameter`, preserve tying and serialization, and be recreated within each measured run. The runtime tensor is unrelated to precompiled kernel caches. Its peak-memory impact depends on lifetime overlap; the full 445.125 MiB is not automatically an incremental peak because the baseline also needs a compute weight for head work. Native linear backward already retains its compute weight: creating a shadow while leaving that native cast path active can duplicate storage. Section 11 separates this reuse policy from the later, conditional experiment of offloading the unchanged FP32 source itself.

### Choose chunks in flattened tokens

Define `C` as tokens across the whole batch, rather than positions per example. Soup's built-in helper chunks the per-example sequence dimension, so its chunk argument has a different interpretation. [10]

| C | FP16 logits | One FP32 copy | Both buffers together |
| ---: | ---: | ---: | ---: |
| 64 | 18.546875 MiB | 37.09375 MiB | 55.640625 MiB |
| 128 | 37.09375 MiB | 74.1875 MiB | 111.28125 MiB |

These are `C * 151936 * bytes_per_value` calculations, not GPU peaks. A tiled reducer can read FP16 logits and perform FP32 arithmetic in registers without allocating the full FP32 copy. Start with a small set of chunk choices; the extra GEMM launches and cold compilation count toward the score.

## 4. Backward and metric fidelity

Preserve the training problem, not only its scalar loss

| Output / dependency | Required behavior |
| --- | --- |
| Training loss | Shift labels so position t predicts the next token; exclude ignored labels and preserve the baseline denominator. |
| Entropy | Cover non-padding attention-mask positions, including prompt tokens. |
| Token accuracy | Compare predictions with the shifted supervised labels using the same counting semantics. |
| Rewind row losses | Keep original row identities, supervised-token counts and per-row aggregation; preserve zero-supervision row handling. |
| Gradient flow | The head is frozen, but hidden-state gradients must reach earlier trainable adapters. |

These contracts were checked across Soup 0.77.0, Transformers 5.16.1, TRL 0.29.1, and Accelerate 1.14.0. In particular, entropy includes the final non-padding position; CE and accuracy use shifted supervised targets. A zero-supervision row must keep the baseline `(loss=0.0, count=0)` rewind behavior. [10,15-17]

### Correct dtype boundary: FP16 computation, FP32 diagnostics

Accelerate wraps model forward with autocast and then recursively converts returned outputs to FP32. Inside Qwen2, the head produces FP16 logits and the causal-LM loss casts its logits to FP32. Outside that model call, TRL entropy and Soup rewind receive the FP32-converted returned logits. [15-17]

Therefore, the required reference is **FP32 statistics computed from FP16-rounded logits**. Computing statistics directly from unrounded FP32 GEMM accumulators changes that reference. Soup's chunk-helper comments about head-native metric dtypes are insufficient to establish the prepared baseline's behavior; the external wrapper matters. Verify the boundary on the prepared runtime before implementing it.

### Supervised-only terminal-head backward

Let `S` be prediction positions whose shifted target is not `-100`. Direct terminal-head gradients are zero outside `S`. Compute the head input-gradient matmul only for `S`, then scatter its result into a full-shaped zero-initialized terminal gradient. This follows from the loss definition; it does not require changing the decoder batch.

The last prompt position can predict the first supervised answer token, so derive `S` from shifted labels, not a naive answer-span mask. Earlier prompt representations still receive gradients through the decoder's causal dependencies. Detaching prompt decoder computation would be a different, incorrect operation. Forward head evaluation must continue to cover entropy-required prompt positions.

### AMP-aware backward

For supervised rows, reconstruct probabilities from saved or recomputed head logits and their matching centered `(m, log_a)` state. Form the logit gradient using the actual upstream gradient scale before the relevant low-precision rounding:

```text
dZ = (P - one_hot(target)) * grad_output / denominator
dH = dZ @ W
# Ignored-label positions contribute zero direct loss gradient.
```

No head-weight gradient is needed for this fixed frozen head. However, skipping its input gradient would stop the learning signal from reaching the decoder. Similarly, frozen NF4 base projections still need their input-gradient path.

Consume the provided `num_items_in_batch` denominator. If it is `None`, reproduce the baseline mean over non-ignored shifted targets; otherwise divide summed supervised loss by the supplied value exactly once. The pinned Transformers causal-LM path counts shifted labels; do not independently assume a batch size, sequence length, or unshifted valid-label count. Match the baseline gradient cast sequence: FP32 probability/scaling arithmetic, FP16 head-gradient operands, appropriate matmul accumulation, FP16 output rounding, then the original hidden dtype where required. A different FP32-return policy is a separately validated variant. [16]

### Compare two storage policies

| Policy | Retained state | Main tradeoff |
| --- | --- | --- |
| Save supervised FP16 logits | `2 * abs(S) * vocabulary` bytes, plus compact summaries | Preserves forward's rounded logits and avoids a repeated head-forward GEMM. |
| Recompute supervised logits | FP16 head-input rows for `S`, plus compact summaries | Much less retained state, with one additional supervised head-forward GEMM and possible shape-dependent rounding differences. |

For an illustrative **512 supervised positions**, saved logits occupy **148.375 MiB**. That count is not measured task data. Keep both candidates until actual supervised fractions, full-command time, and peak lifetime overlap are known.

### FP16 risk to test explicitly

Precomputing an unscaled hidden gradient in FP16 and applying the upstream scale later can round or underflow small values too early. The safer initial design above avoids assuming that those two orders are numerically equivalent. This is an identified design risk, not a demonstrated failure of a candidate.

### Recomputation has a real cost

An extra head pass may reduce memory while increasing time. Evaluate a small set of chunk sizes and backward policies on the full command. Preserve candidates with useful time/memory tradeoffs, but reject any combined candidate that fails an official admission gate.

Start with saved supervised FP16 logits. Compaction or a different chunk shape can change GEMM rounding; a recomputed logit row may then disagree with its saved forward normalizer. Section 10 describes checking that pairing and a conservative shape-preserving replay alternative. [32]

> **Easy-to-miss constraint:** prompt tokens can be excluded from supervised CE, but their vocabulary distributions still contribute to entropy. Skipping all prompt-token head evaluation would change a required output. [10]

### Concrete integration plan without enabling a built-in flag

The fixed training path has no identified mandatory full-logits consumer beyond loss, entropy, accuracy, and rewind. This follows from the inspected call chain; runtime and regression tests still need to verify the proposed wrappers. The task's `val_split` is zero, and ordinary prediction/evaluation calls should keep their original behavior. [4,6,10,16,17]

Install narrowly guarded original wrappers through `solution.apply()`:

1. **Model boundary:** after Soup's `SFTTrainerWrapper._setup_transformers` creates the final PEFT model, obtain its actual inner causal LM and wrap that instance's forward. Continue invoking it through the outer PEFT/Accelerate model. Restrict the new path to the supported model, training mode, and labelled calls; preserve the forwarded signature and loss-kwargs detection.
2. **L1 forward:** invoke the original causal-LM forward, reduce diagnostics from its internal logits, then construct a fresh output with the original loss and applicable fields but no logits. **L2 forward:** invoke the unchanged decoder and our custom chunked-head/autograd implementation, returning loss and compact state.
3. **Trainer boundary:** supply an original subclass of the base SFT trainer through Soup's `_build_rewind_trainer`. Soup can then retain its original outer `RewindTrainer`, sampler, row fingerprints, recorder, callbacks, and serialization.
4. **Loss/metric handling:** on the supported path, call `transformers.Trainer.compute_loss` with the original `num_items_in_batch` and `return_outputs=True`. Reproduce TRL's existing entropy, accuracy, and token-counter updates from the new summaries, including gathering and averaging semantics. Remove `_prediction_loss_only` and set `use_cache=False` as the pinned path does. Return loss or `(loss, outputs)` as requested.
5. **Rewind interface:** provide an original compact object at `trainer._soup_chunked_step` containing `token_loss`. Soup's existing outer wrapper already uses this interface when returned logits are absent. This reuses a data contract, without enabling `stream_loss_chunk`, `use_cut_ce`, or other protected optimization flags. [10]

The compact `token_loss` is FP32 `[B,T]`. Entry `t` holds CE for `labels[:,t+1]`; ignored targets and the final entry are zero. Training loss uses the global provided denominator. Row means belong only to rewind, which retains its own per-row counts. Never introduce a second normalization or an average of row means into training. Section 13 tightens this interface with a scoped trainer/model ticket, metadata guards, and a source-proven stale-mapping failure case; training mode plus labels alone is not a sufficient guard.

Clear stale summary state before each forward and consume each new summary once. Keep compact row-loss state alive until the outer rewind wrapper has read it; clear it afterward through a small outer cleanup wrapper, or before the next guarded forward. Unsupported calls, including evaluation, unsupported adapter variants, and regression-test fake modules, delegate to the original method before the optimized path executes. If an optimized forward raises, clear transient summary state and propagate the exception. Do not silently rerun a forward after dropout or other stateful work has already executed.

## 5. Fusion and later experiments

Start with contained changes; isolate the higher-risk branches

### Original RMSNorm and residual fusion

Implement both forward and backward, preserving FP32 norm reduction and baseline input/output types. Fuse adjacent residual or cast operations only where dependencies and rounding permit. Treat broader precision changes as a separate experiment so numerical deviations remain attributable.

Round 5 specifies R1's frozen-weight backward and R2's residual-output contract in section 12. Native autograd already retains only the necessary input/statistic/gamma references; the custom norm must not claim removal of nonexistent saved intermediates. Both casts in the expected FP32 norm path are no-ops. [16,25,40]

### LoRA: keep the stochastic behavior

The task uses rank 16, alpha 32 and dropout 0.05 on q/v adapters. Those projections have separate dropout operations. Begin by retaining the existing dropout calls and masks, then fuse downstream low-rank epilogues, scaling, base-result addition and casts. A shared dropped-out input must not silently replace independent q/v masks. [3,4]

Backward must preserve both adapter-matrix gradients and all input contributions. Full trajectory comparison matters: matching a dropout probability distribution alone does not establish agreement with the tightly checked seeded run.

#### First LoRA candidate: a fresh-output epilogue

PEFT 0.20.0 computes the base projection, defensively clones its output, computes the two adapter linears, scales the adapter result, then adds it. The clone protects against problematic view manipulation. Our proposed original epilogue should read the base result and write a **fresh output**, preserving that protection while removing a separate clone and scaled-adapter temporary. Retain native dropout, both linears, and the NF4 base path initially. [18]

On the expected FP32 residual / FP16 AMP path, preserve the operation:

```text
U16 = existing adapter B(A(dropout(X))) output
Y32 = base32 + float32(float16(2 * U16))
```

bitsandbytes returns its result to the original input dtype; the adapter GEMMs follow autocast. Confirm these dtypes at runtime. A fused accumulator added directly to `base32` can change the rounding sequence, as can a custom backward that changes when its gradient is cast or scaled. Guard the vanilla active-adapter case and delegate merged, disabled, mixed, or unsupported variants. [18,20]

**Round-6 backward clarification:** the adapter edge is `dU16 = RN16(2 * RN16(dY32))`; the base edge keeps `dY32`. Casting only after multiplication changes subnormal rounding. Later TRL conversion of adapter Parameters to BF16 does not remove this earlier FP16 activation-gradient boundary. See section 13.2.

#### Keep rank-sized state; target full-width traffic

The LoRAFusion research proposes splitting at the small rank intermediate so memory-oriented fusion can coexist with efficient large GEMMs. Its H100/L40S results are not T4 predictions. We use the graph-splitting principle as an idea with attribution, not its implementation or scheduler. [19]

At the maximum 2,048 flattened tokens, our calculated sizes are:

| LoRA object | Dimensions / size |
| --- | --- |
| q matrices | A `[16,1536]`; B `[1536,16]` |
| v matrices | A `[16,1536]`; B `[256,16]` |
| One FP16 rank intermediate | `[2048,16]`, **64 KiB** |
| One FP16 dropped input | `[2048,1536]`, **6 MiB** |
| One FP32 original input | `[2048,1536]`, **12 MiB** |
| All trainable adapter parameters | **2,179,072**; expected **4.15625 MiB in BF16 after TRL initialization**. Earlier PEFT-stage FP32 storage would be 8.3125 MiB; see section 13.1. |

The rank intermediate is 96 times smaller than the dropped input. Preserve it initially and investigate fusion of mask/rescale into the adapter input-gradient epilogue, followed by shared loading of the upstream gradient for `dH` and `dB`. Those two gradients reduce over different dimensions; their combined tiling must earn its cost. Tune q and v separately because their output widths differ sixfold. These are proposals derived from the fixed model geometry. [4,9,18]

Use the original native dropout calls, input shape/dtype, and call order. A matching seed with a different random-number kernel does not establish the same masks. Soup's grouped QKV helper supports its concatenated path only with zero dropout; the fixed 0.05 case cannot share a single dropped input across q/v. Current external implementations vary, so a blanket claim that all fast-LoRA systems require zero dropout would also be inaccurate. [10,24]

**Optional, unverified memory experiment:** with both native masks preserved, a shared `FP16(X * c)` representation could reconstruct the two dropped-half inputs, where `c` exactly matches native dropout's FP32 scaling. A theoretical upper saving is one 6 MiB input per layer, or 168 MiB across 28 layers at maximum length. Check signed zeros, nonfinite values, scaling precision, actual saved tensors, reconstruction cost, and gradient rounding. Saving an extra 12 MiB FP32 input merely to replace two 6 MiB half inputs provides no raw byte saving. This experiment follows the safe epilogue in priority.

### Selective activation recomputation

Identify large saved intermediates that are inexpensive to recreate. Recompute them inside original autograd code, retaining masks or the correct RNG state when replaying stochastic operations. Custom recomputation and offloading are permitted, while changing the built-in configuration switches is prohibited. [2,3]

Round 3 makes this concrete for the frozen, dropout-free MLP: first pack gate/up saved values losslessly and recompute native SiLU; only then test replay of one or both projection GEMMs. The maximum-shape storage inventory and replay costs are in section 10. Replaying a full decoder layer is a broader change with attention and LoRA RNG concerns.

### NF4 decode fused with matrix multiplication

The exact bitsandbytes 0.50.2 source already dispatches to fused NF4 forward code on T4 for eligible shapes, including gradient-enabled execution. Its `MatMul4Bit.backward`, however, dequantizes a full weight and uses `torch.matmul` for the input gradient. It retains packed weights and metadata, not all dense projections, between forward and backward. [20]

#### Which training shapes take the fused forward?

The following is static source evaluation for a 40-SM T4, FP16 compute, standard 64-element NF4 blocks, nested second-level blocksize 256, and canonical weight orientation `[N,K]`. It is not a runtime trace. Here `M` is the flattened input-token count. [21]

| Projection | N | K | Fused forward eligible M |
| --- | ---: | ---: | --- |
| q, o | 1,536 | 1,536 | 1–4 or 9–64 |
| k, v | 256 | 1,536 | 1–64 |
| gate, up | 8,960 | 1,536 | 1–128 |
| down | 1,536 | 8,960 | 1–64 |

With unchanged padded batch size four, `M=4*L`. Under these conditions, **length 33 or greater falls back for every listed projection**. Actual batch lengths and executed kernels must be traced before assigning a measured time fraction. The q/o exclusion at M=5–8 is an occupancy-heuristic detail, not a typo.

#### Target the backward gap and preserve the representation

An original fused `dX = dY @ dequant(W)` kernel could consume packed tiles directly. It should decode the original NF4 codebook and double-quantized scales, preserve FP16 operand rounding, and use appropriate matmul accumulation. No frozen base-weight gradient is needed. This targets a concrete source-level gap, but larger cuBLAS GEMMs may still outperform a fused implementation. **Round-4 prerequisite:** ordinary Triton 3.6 `tl.dot` does not select tensor cores on SM75; establish a viable custom MMA implementation before investing in this fused GEMM. [33,34]

NF4 nibbles index a nonuniform codebook. They cannot be sent directly to signed-INT4 multiplication. The first packed element uses the high nibble. The native SM75 kernel multiplies centroid and scale in FP32 before casting to FP16. Match nested-scale reconstruction and its rounding; transposing and **requantizing** changes the original values and quantization groups. An exact repack must preserve every original code-to-scale association. [22]

#### Memory arithmetic changes the cache strategy

| Calculated storage, excluding activations and allocator effects | MiB |
| --- | ---: |
| Largest single dense FP16 projection | 26.25 |
| All seven dense FP16 projections in one layer | 89.25 |
| All dense FP16 projections across 28 layers | **2,499** |
| Reconstructed FP32 first-level scales for those weights | **78.09375** |

The baseline's dense fallback weights are temporary. Removing one 26.25 MiB buffer does not establish multi-gigabyte savings. Keeping every layer's dense weights until its backward would accumulate the full 2,499 MiB because all layers run forward before backward begins. [9,20; calculated]

A separate hypothesis is to reuse the frozen **FP32 block scales** while retaining the original packed NF4 weights. This has a much smaller nominal footprint and can support repeated fallback/backward decoding. Reconstruct and compare scales exactly; include construction time and residency in the official measurements. A startup-only scale kernel contributes nothing to the original-kernel share measured from step three.

Using the organizer's approximate profile only as an illustration: if training occupies 60 seconds of an 80-second command, eliminating a 5% training-time slice saves 3 seconds, giving `80/77 = 1.039x`. That would remain below the 1.05 speed-scoring threshold. Treat NF4 as a combined decode/GEMM/backward experiment, with measured end-to-end justification. [2; illustrative arithmetic]

### Remove padding within each existing batch

Custom unpadding is allowed, but rows cannot be mixed across batches. Preserve independent causal sequences, position IDs, shifted labels, row mapping, token accounting and dropout behavior. Start with isolated operations before redesigning the attention layout. [2,3]

> **Originality trap:** the starter already supplies fused SwiGLU forward and backward. Its kernels do not count as ours, even if renamed or rewritten. Retain it only as a supporting optimization; establish eligibility through substantive original work elsewhere. [3,6]

### Priority rule

Develop L1 diagnostics, M1 compact MLP state, and original normalization/LoRA or cast/repeat kernels in parallel. Keep one integrated candidate passing all gates. Advance the full head, projection replay, NF4, or padding branches according to measured potential and time left for validation.

## 6. Experiment and integration plan

Parallel development, one uncontended GPU measurement queue

| Workstream | Concrete output |
| --- | --- |
| Head, loss and metrics | Original streamed path; compact diagnostics; AMP-correct backward. |
| Normalization and LoRA | Original fusion kernels; forward/backward parity checks. |
| Memory experiments | Selective recomputation, then padding or NF4 candidates if justified. |
| Integration and evaluation | Known-good combined candidate; official results; regression checks and source attribution. |

### A short, repeatable loop

- **Establish the baseline.** Confirm the pinned environment and run the official harness on the scoring hardware.
- **Make one attributable change.** State the expected saving and the semantic behavior that must remain identical.
- **Screen correctness.** Check forward and backward, masks and edge cases before spending a complete training run.
- **Measure the complete candidate.** Recheck combined performance, memory and original-kernel share after integration.
- **Keep evidence.** Record the commit, time, peak, loss deviations, diagnostics, quality, kernel share and rejection reason.

### Official commands after environment/data preparation

```bash
python -m pytest solution -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

The starter also provides solution-only iteration and profiling modes. Their partial output is for development; final admission requires the complete base/solution comparisons. [3,5]

### Cold compilation changes tuning economics

Each scored run starts with empty compiler caches. Downloading model files is warmed up before timing, but local model loading, compilation/autotuning, training and saving remain in the timed command. Avoid a large search during the scored run. [5]

Illustration: saving 5 ms per step saves only 0.32 s over 64 steps. An extra 2 s of compilation would make that candidate slower overall. Prefer a small number of reusable shape families and measure the complete cold run.

Triton `tl.constexpr` values still specialize. Keep changing token-row counts as runtime values where possible; use fixed architecture dimensions and a small number of intentional block configurations. `do_not_specialize` is not a way to make a `constexpr` argument dynamic. Verify the actual installed Triton API, record compile events, and avoid per-length autotuning inside the scored command. [23]

### Experiment matrix established in round 2

These are separate comparisons, followed by integration; no row has a measured result yet.

| ID | Isolated change | Main evidence needed |
| --- | --- | --- |
| L1 | Shared diagnostics inside forward; compact outputs; original loss/backward | Same metrics and row records; removed copy/reduction traffic; complete time and peak. |
| L2 | Chunked head; supervised-only terminal backward | Full-shaped terminal gradient and earlier adapter gradients; saved-logit versus recomputed-logit comparison; training-quality gates. |
| L3 | Explicit FP16 head representation | Cast count/bytes; exact FP16 weight values; phase-only versus retained lifetime; unchanged ties and save format. |
| A1 | Fresh-output LoRA epilogue with native dropout and GEMMs | Same masks and dtype boundaries; forward/backward agreement; actual kernel share. |
| N1 | Original RMSNorm and justified residual/cast fusion | Dtype and reduction agreement; gradient checks; interaction with LoRA and head branches. |
| F1 | NF4 input-gradient fusion; optional reconstructed-scale reuse | Exact decode and input gradient; actual forward dispatch; throughput/peak versus the installed backend. |

A1 and N1 are useful parallel branches while L1–L3 progress. F1 stays behind them until a trace establishes enough headroom. Saved-logit and recomputed-logit candidates should both remain available if neither dominates time and memory.

Round 3 adds M1/M2 activation-state candidates and K1/K2 attention candidates in section 10. L2 now starts with saved supervised logits and centered normalizer state; recomputation is a separately checked policy.

### Minimal measurements before choosing a winner

Capture the actual package versions and GPU; every padded length; supervised and non-padding counts; head and returned-logits dtypes; frozen/trainable flags; head-cast count/bytes; NF4 executed path; and largest live allocations. Collect detailed traces in development runs, then use the unchanged official measurement procedure for scored comparisons.

For each candidate, retain commit, seed, complete command time, maximum memory peak, first/mean loss deviation, entropy/accuracy/rewind checks, final adapter evaluation loss, original-kernel share, compile time, and regression status. Do not add independent microbenchmark speedups together: combined lifetime overlap, rounding, compilation, and bottleneck shifts can change the result.

## 7. Admission gates and final review

Requirements to pass, not claims that the proposed work already passes

| Gate | Official requirement / measurement |
| --- | --- |
| Final quality | Adapter evaluation loss at most 0.5% above baseline. Evaluation reloads the adapter without solution code. |
| Loss trajectory | First-step relative difference at most 1%; mean per-step absolute relative difference at most 3%. |
| Soup outputs | Required metric names and counts preserved; entropy, accuracy and rewind row losses within the harness's 3% checks. |
| Original Triton work | Qualifying original kernels run in every candidate run and jointly occupy at least 5% of training-step time from step 3 onward. |
| No performance regression | Whole-command time and peak GPU memory each at most 10% above baseline. |
| Same task | Unchanged model, rows/batches/order, steps, optimizer/schedule, trainable parameters and protected settings. |
| Execution and regressions | Clean one-T4 run, unchanged measurement setup and all 79 required Soup regression tests. |

### Exact measurement nuances

The inspected harness uses median wall time across repeats and the **maximum GPU memory peak** across repeats. With two runs, median time equals the arithmetic mean. Memory is sampled externally every 100 ms across the whole run. Optimize the worst peak, including loading, rather than only a steady-state allocation. [5]

The original-kernel share uses CUDA-event time divided by the sum of synchronized `on_step_begin` to `on_step_end` wall intervals from step 3. The lowest share across candidate runs must pass, using qualifying kernel names launched in every candidate repeat. This denominator differs from the first-to-last training-loop span and from full-command time. Section 11 explains event overhead, callback placement, and the cold-cost budget. Cover several real bottlenecks and leave headroom for hidden rows; do not add useless work to increase the share. [6,7,16]

### Submission package

Use the hackathon platform's team repository. Include `solution/`, a clear README, completed submission metadata, the final commit, and a saved public Kaggle notebook that runs without edits. Supply the required video of at most 3 minutes and presentation. Attribute borrowed ideas and describe observed limits. [2,3]

> **Release decision:** select a fully validated candidate, keep the official task and evaluator intact, and submit by **31 October 2026, 23:59 Astana / Almaty**. A measured, reproducible improvement is the goal; ranking remains unknown until competitors are evaluated.

## 8. What was verified in research round 2

### Source findings versus executed checks

The dependency audits used the exact pinned package sources. The NF4 dispatch table was evaluated from the official pure dispatch logic with fixed T4 properties on CPU. That establishes conditional eligibility; it does not prove which branch a real batch took.

The research environment has NumPy 2.3.5, no installed Torch, no `nvidia-smi`, and no exposed NVIDIA device. Consequently, the executed work below is **CPU-only algebra verification**. No CUDA kernel, training trajectory, full adapter, regression suite, or official benchmark was executed here.

### Executed numerical results

Seed: **20261030**. Appendix A contains the complete original script, with a portable output path. Errors below are maximum absolute differences in the stated CPU checks.

| Check | Coverage | Observed result |
| --- | --- | --- |
| Streaming sufficient statistics | Ten cases: FP64/FP32; irregular vocabulary; common offset 10,000; actual vocabulary width 151,936; equal logits; extreme peak | Worst FP64 entropy difference **1.78e-15**; worst FP32 LSE and entropy differences each **9.54e-7**. |
| Tile merge order | Forward and reverse order across the same five cases in each dtype | Worst FP32 difference **1.43e-6**. |
| Supervised-only head backward | Toy 21-position batch, six supervised targets, one zero-supervision row | Dense versus selected-row terminal gradient difference **0.0**. |
| Earlier prompt gradient | Toy causal cumulative-sum decoder with finite-difference checks | Worst difference **8.66e-11**; a prompt-input gradient remained nonzero. |
| LoRA derivatives | Float64 toy q/v branches, distinct fixed dropout masks; 11 finite differences across X/A/B | Worst difference **1.97e-10**. |
| FP16 scale-order illustration | Scalars 1e-9, 1e-8, 1e-7; scale 65,536 | Casting first erased the first two; scaling before casting preserved nonzero values after unscaling. |

The dropout check uses rank 3 toy shapes and fixed masks to test the equations. It does not check the real rank-16 CUDA kernels, native RNG stream, autocast graph, or optimizer. The scale illustration identifies a numerical risk; it is not evidence that baseline or candidate training actually underflows.

These checks justify progressing to implementation, not skipping GPU verification. Before comparing performance, confirm the prepared baseline's actual dtype/shape path; test the original kernels on representative tensors; then run the complete official quality, metrics, originality, and regression gates. Broaden validation only when a changed behavior or an official gate requires it.

### Decisions at the end of round 2

The current priorities in the session handoff and section 10 incorporate the later findings. These entries preserve the reasoning that led to the round-2 plan.

1. Start from **L1**, which isolates diagnostics and the returned-logits copy while preserving the baseline training-loss graph.
2. Build **L2 and L3 together**: chunking without explicit frozen-weight cast reuse risks repeating a very large conversion.
3. Compare saved-supervised-logit and supervised-recompute policies using observed supervision fractions and lifetime traces.
4. Use original RMSNorm and the conservative LoRA epilogue as parallel opportunities and additional qualifying kernel work.
5. Prioritize NF4 backward or combined decode/GEMM work only after a trace; do not assume dense weights persist or that every forward is unfused.
6. Keep Qwen's measurement-and-reuse methodology as the transferable idea. No sourced headline speedup predicts this task's final ranking.

## 9. Sources and provenance

Primary sources checked for the research; hyperlinks are clickable

**[1]** Qwen Team. Qwen3.7: The Agent Frontier. Published optimization narrative and cumulative kernel results.
[qwen.ai/blog?id=qwen3.7&locale=en](https://qwen.ai/blog?id=qwen3.7&locale=en)

**[2]** Decentrathon 6.0: Soup Kernel Challenge. User-supplied technical brief. Workload, profile, rules, scoring and deadline.
[Open the supplied challenge brief](https://chatgpt.com/library?file_id=libfile_2f89bbb621c081918a68369a0fe4df4b&account_id=personal)

**[3]** Soup Kernel Challenge README. Starter behavior, allowed modifications, originality exclusion and submission.
[github.com/archived-source/soup-kernel-challenge/blob/c7c2fcd.../README.md](https://example.invalid/redacted-github)

**[4]** Fixed task configuration. Model, training parameters and batch/sequence settings.
[github.com/archived-source/soup-kernel-challenge/blob/c7c2fcd.../task/task.yaml](https://example.invalid/redacted-github)

**[5]** Measurement harness. Cold caches, warm-up, run timing and summary aggregation.
[github.com/archived-source/soup-kernel-challenge/blob/c7c2fcd.../harness/run_task.py](https://example.invalid/redacted-github)

**[6]** Admission checks. Quality, trajectory, outputs, original-kernel share and regression limits.
[github.com/archived-source/soup-kernel-challenge/blob/c7c2fcd.../harness/check_rules.py](https://example.invalid/redacted-github)

**[7]** Training instrumentation. Per-launch CUDA events and step-time measurement.
[github.com/archived-source/soup-kernel-challenge/blob/c7c2fcd.../harness/_train.py](https://example.invalid/redacted-github)

**[8]** Scoring implementation. Ratios, improvement terms and noise threshold.
[github.com/archived-source/soup-kernel-challenge/blob/c7c2fcd.../harness/score.py](https://example.invalid/redacted-github)

**[9]** Qwen2.5-1.5B-Instruct model configuration. Vocabulary, dimensions and tied embeddings.
[huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)

**[10]** Soup CLI 0.77.0 release. Pinned wheel inspected as source: trainer/sft.py, trainer/rewind_hf.py, utils/stream_loss_chunk.py, and utils/fast_lora*.py. The outer prepared stack must also be considered when interpreting helper dtype comments.
[pypi.org/project/soup-cli/0.77.0/](https://pypi.org/project/soup-cli/0.77.0/)

**[11]** Apple. Cut Your Losses in Large-Vocabulary Language Models. Algorithmic background for avoiding full logits residency.
[github.com/apple-aiml-research/ml-cross-entropy](https://github.com/apple-aiml-research/ml-cross-entropy)

**[12]** Triton documentation. Fused Softmax. Data-movement motivation and the on-chip row-size condition.
[triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

**[13]** Actual Kaggle notebook installation cell, at the inspected challenge commit. Explicit dependency pins and the absence of explicit Torch/Triton pins.
[templates/kaggle_notebook.ipynb](https://example.invalid/redacted-github)

**[14]** PyTorch 2.11 autocast source. `cached_cast` requires `requires_grad()` for the FP32 weight-cast cache; outermost autocast exit clears its cache.
[aten/src/ATen/autocast_mode.cpp](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.cpp) · [torch/amp/autocast_mode.py](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/amp/autocast_mode.py)

**[15]** Accelerate 1.14.0. Prepared forward uses autocast followed by recursive FP32 output conversion. Exact pinned wheel inspected.
[accelerator.py](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py) · [utils/operations.py](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py)

**[16]** Transformers 5.16.1. Qwen2 head invocation, causal-LM loss and its denominator, Trainer loss handling and shifted-label counting. Exact pinned wheel inspected.
[modeling_qwen2.py](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py) · [loss_utils.py](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py) · [trainer.py](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py)

**[17]** TRL 0.29.1. Entropy, accuracy, token counting, and logging. Exact pinned wheel inspected.
[sft_trainer.py](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py) · [utils.py](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/utils.py)

**[18]** PEFT 0.20.0. Four-bit LoRA forward, defensive clone, adapter scaling/addition, and k-bit preparation.
[lora/bnb.py](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py) · [utils/other.py](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py)

**[19]** Zhu et al. LoRAFusion: Efficient LoRA Fine-Tuning for LLMs. Primary paper, section 5.1, and authors' repository. Source of the graph-splitting idea; no implementation is copied into a candidate.
[arXiv:2510.00206v1](https://arxiv.org/html/2510.00206v1) · [CentML/lorafusion](https://github.com/CentML/lorafusion)

**[20]** bitsandbytes 0.50.2, resolved commit `08a9956b939857ae2c4fbd56753960cc370fa7cf`. Forward dispatch invocation, saved backward state, input-gradient dequantization, and original-input-dtype return.
[autograd/_functions.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/autograd/_functions.py) · [nn/modules.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/nn/modules.py)

**[21]** bitsandbytes CUDA dispatch heuristics and dense fallback, at the same exact release commit.
[backends/cuda/ops.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/backends/cuda/ops.py)

**[22]** bitsandbytes NF4 representation, nested scales, nibble ordering, and SM75 computation precision, at the same release commit.
[functional.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/functional.py) · [csrc/kernels.cu](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/csrc/kernels.cu) · [csrc/gemm_4bit_sm75.cu](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/csrc/gemm_4bit_sm75.cu)

**[23]** Triton JIT documentation and the 3.6.0 runtime source. Specialization and cache-key behavior; recheck against the installed version.
[triton.jit documentation](https://triton-lang.org/main/python-api/generated/triton.jit.html) · [v3.6.0/runtime/jit.py](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/jit.py)

**[24]** PyTorch 2.11 native CUDA dropout and a pinned current Unsloth implementation. Evidence for preserving the actual native RNG/mask path and avoiding blanket claims about other systems' dropout support.
[native/Dropout.cpp](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Dropout.cpp) · [native/cuda/Dropout.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu) · [Unsloth fast_lora.py, commit 5acda0f](https://github.com/unslothai/unsloth/blob/5acda0f669a55f80ea6dfef0cb3afbd685edfe09/unsloth/kernels/fast_lora.py)

**[25]** PyTorch 2.11 derivative definitions. Saved inputs for SiLU/multiplication and metadata requirements for dtype conversion.
[tools/autograd/derivatives.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml)

**[26]** PyTorch 2.11 autograd storage and recomputation contracts. Packed/unpacked content, hook lifetime and nesting, reference-cycle constraints, and checkpoint/RNG behavior.
[Autograd saved-tensor hooks](https://docs.pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.saved_tensors_hooks) · [Autograd mechanics](https://docs.pytorch.org/docs/2.11/notes/autograd.html) · [graph.py](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/autograd/graph.py) · [Checkpoint documentation](https://docs.pytorch.org/docs/2.11/checkpoint.html)

**[27]** PyTorch 2.11 native CUDA SiLU forward and backward arithmetic.
[ActivationSiluKernel.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/ActivationSiluKernel.cu)

**[28]** The exact supplied SwiGLU example: saved gate/up state, its formula, and its explicit exclusion from original-kernel credit.
[solution/swiglu.py](https://example.invalid/redacted-github)

**[29]** Transformers 5.16.1 SDPA integration and causal-mask construction, release revision `93c8b7b485963a10800c91f55304db6be211c2bd`. GQA predicate, explicit K/V repetition, and mask-omission conditions.
[integrations/sdpa_attention.py](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/integrations/sdpa_attention.py) · [masking_utils.py](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/masking_utils.py)

**[30]** PyTorch 2.11 SDPA implementation and dispatch, release revision `70d99e998b4955e0049d13a98d77ae1b14db1f45`. Hardware/dtype/head-count checks, math GQA/FP32 promotion, CUDA autocast registration, and the eligibility diagnostic.
[cuda/sdp_utils.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp) · [sdp_utils_cpp.h](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/transformers/sdp_utils_cpp.h) · [attention.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/transformers/attention.cpp) · [autocast_mode.h](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/autocast_mode.h) · [torch/backends/cuda](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/backends/cuda/__init__.py)

**[31]** PyTorch 2.11 integer-target CE, NLL backward, and CUDA log-softmax epilogues. Centered normalizer arithmetic and gradient operation order.
[LossNLL.cpp](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/LossNLL.cpp) · [cuda/Loss.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Loss.cu) · [cuda/SoftMax.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/SoftMax.cu)

**[32]** PyTorch 2.11 numerical-accuracy guidance. Mathematically equivalent batched/sliced computations can differ in floating-point arithmetic; the math SDPA path normally promotes half inputs for its internal calculations.
[Numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)

**[33]** Triton 3.6.0 ordinary-dot acceleration pass and upstream change. `BlockedToMMA` rejects compute capability below 80 before MMA lowering; the README's supported NVIDIA hardware starts at CC8.0.
[AccelerateMatmul.cpp](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp) · [README](https://github.com/triton-lang/triton/blob/v3.6.0/README.md) · [Upstream PR 5066](https://github.com/triton-lang/triton/pull/5066)

**[34]** Triton 3.6.0 NVIDIA backend and residual Turing lowering. Architecture-dependent pipeline passes; Turing instruction definitions alone do not prove reachability from ordinary `tl.dot`.
[compiler.py](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/backend/compiler.py) · [MMAv2.cpp](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp)

**[35]** NVIDIA Turing/PTX specifications and Triton inline-assembly contract. Hardware MMA capability, register/shared-memory limits, lane fragments, and the limits of elementwise assembly layout assumptions.
[Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html) · [PTX warp-level MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma) · [Triton inline assembly](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html) · [Exact 3.6 language implementation](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/core.py)

**[36]** PyTorch 2.11 transfer and stream source, release revision `70d99e998b4955e0049d13a98d77ae1b14db1f45`. Current-stream asynchronous copy, temporary buffers, event dependencies, and allocator lifetime protection.
[cuda/Copy.cu](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/Copy.cu) · [cuda/streams.py](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/cuda/streams.py) · [record_stream](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.record_stream.html)

**[37]** PyTorch 2.11 CUDA memory accounting. Live allocations, reserved allocator memory, and visibility through `nvidia-smi`.
[CUDA semantics: memory management](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management)

**[38]** NVIDIA T4 hardware and host-transfer guidance. PCIe Gen3 x16 versus GDDR6 bandwidth; pinned-memory transfer examples are not measurements of Kaggle.
[T4 specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/) · [CUDA Best Practices: host/device transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device)

**[39]** Transformers 5.16.1 loading sources, revision `93c8b7b485963a10800c91f55304db6be211c2bd`. Requested dtype, per-module FP32 overrides, parameter materialization, and bitsandbytes exclusions. Used with Soup's exact 0.77.0 load-dtype resolver and PEFT 0.20.0 preparation, already listed above.
[modeling_utils.py](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/modeling_utils.py) · [core_model_loading.py](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/core_model_loading.py) · [quantizer_bnb_4bit.py](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/quantizers/quantizer_bnb_4bit.py)

**[40]** PyTorch 2.11 native arithmetic used by RMSNorm's unfused graph. Mean/power backward, scalar reciprocal multiplication, forward reductions, and rsqrt dispatch; read together with the derivative definitions in [25].
[FunctionsManual.cpp](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/FunctionsManual.cpp) · [BinaryDivTrueKernel.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu) · [PowKernel.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/PowKernel.cu) · [ReduceMomentKernel.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/ReduceMomentKernel.cu) · [SharedReduceOps.h](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/SharedReduceOps.h)

**[41]** PyTorch 2.11 custom-autograd output metadata and gradient materialization; PEFT 0.20.0 checkpoint-related input-gradient preparation. Sources for preserving first-layer no-gradient branches.
[mark_non_differentiable](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html) · [set_materialize_grads](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html) · [function.py](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/autograd/function.py) · [PEFT peft_model.py](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py)

**[42]** Triton 3.6 elementary-function lowering, libdevice mappings, and max/index reductions, compared with PyTorch 2.11 CUDA exponential. Approximate exponential, explicit library functions, and first-index ties are separate choices.
[ElementwiseOpToLLVM.cpp](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp) · [cuda/libdevice.py](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/language/cuda/libdevice.py) · [language/math.py](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/math.py) · [language/standard.py](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/standard.py) · [PyTorch UnaryOpsKernel.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/UnaryOpsKernel.cu)

**[43]** NVIDIA floating-point guidance. Fused multiply-add uses a different rounding sequence from separate multiplication and addition; mathematical accuracy and agreement with a particular execution path are distinct questions.
[Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)

**[44]** Milakov and Gimelshein. Online normalizer calculation for softmax. Primary background for maximum/rescaled-sum merging and combining normalization with selection. The report's centered entropy moment is an algebraic extension; no kernel implementation is copied.
[arXiv:1805.02867v2](https://arxiv.org/abs/1805.02867v2)

**[45]** PyTorch 2.11 AMP source, revision `70d99e998b4955e0049d13a98d77ae1b14db1f45`. Custom-autograd context handling, GradScaler grouping, and the CUDA dtype-dispatch mismatch; read with TRL's constructor [17] and Accelerate [15].
[autocast_mode.py](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/amp/autocast_mode.py) · [grad_scaler.py](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/amp/grad_scaler.py) · [AmpKernels.cu](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/AmpKernels.cu) · [Dispatch.h](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/Dispatch.h) · [AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html) · [Accelerate optimizer.py](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/optimizer.py)

**[46]** Transformers 5.16.1 ModelOutput mapping semantics and quantization flags; PEFT 0.20.0 attribute forwarding. These supplement the trainer, loss, and conversion sources in [15–18,39,41].
[utils/generic.py](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/utils/generic.py) · [quantizer_bnb_4bit.py](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/quantizers/quantizer_bnb_4bit.py) · [tuners/tuners_utils.py](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py)

**[47]** PyTorch 2.11 attention storage, backward allocation, dense conversion, reshape, and matmul sources, revision `70d99e998b4955e0049d13a98d77ae1b14db1f45`. Used with the exact SDPA math/HF sources already listed in [29–30].
[CUDA attention.cu](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/transformers/cuda/attention.cu) · [attention_backward.cu](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/transformers/cuda/attention_backward.cu) · [TensorIterator.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/TensorIterator.cpp) · [TensorConversions.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/TensorConversions.cpp) · [LinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/LinearAlgebra.cpp) · [TensorShape.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/TensorShape.cpp)

**[48]** Triton 3.6 runtime specialization, autotuning, and NVIDIA benchmarking. Grid handling, scalar/tuple specialization, repeated benchmark calls, and the 256 MiB flush allocation; version-specific source evidence, not a measurement of the scoring image.
[runtime/jit.py](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/jit.py) · [python/src/specialize.cc](https://github.com/triton-lang/triton/blob/v3.6.0/python/src/specialize.cc) · [runtime/autotuner.py](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/autotuner.py) · [testing.py](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/testing.py) · [NVIDIA driver.py](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/backend/driver.py) · [Autotune documentation](https://triton-lang.org/main/python-api/generated/triton.autotune.html)

### Provenance and interpretation

Starter sources were inspected at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. The report distinguishes publisher/organizer measurements from our calculations and untested design proposals. No winning kernel release, competition ranking or candidate T4 performance is asserted.

The live harness is more precise than the brief on repeated-run aggregation: it takes maximum memory peak rather than an average. The implementation details and numeric gates in this report follow the inspected harness. [5-8]

Round 2 added the prepared AMP dtype audit, frozen-head cast analysis, compact-metrics integration plan, supervised-only head-backward derivation, exact NF4 dispatch analysis, LoRA epilogue proposal, and CPU checks. Round 3 added compact MLP saved state, projection-replay costs, centered normalizer corrections, head-recompute consistency, T4 SDPA/GQA dispatch analysis, and cast/repeat fusion. Round 4 added the Triton/SM75 compiler constraint, bounded logit-gradient production, selective-offload economics, combined lifetime accounting, exact event attribution, and a conditional tied-source offload idea. Round 5 added first-layer gradient topology, explicit RMSNorm and RoPE fusion contracts, head elementary-function choices, additional CPU rounding checks, and a conditional traffic/launch model for qualification. Round 6 adds final adapter dtype and baseline AMP compatibility analysis, precise autograd/trainer contracts, a query-layout opportunity, backward-stride corrections, and runtime specialization/autotuning costs. The older PDF is an earlier snapshot; this Markdown file is the updated research handoff.


## 10. Research round 3: activation memory, attention dispatch, and stable backward

**Main decision:** add compact MLP saved state as an early memory experiment alongside L1 diagnostics. Keep the original arithmetic while changing what backward retains. The head design also needs centered normalizers and an explicit policy for forward/recomputed-logit agreement. Attention work should begin with the actual backend branch and K/V copy traffic.

Everything in this section is identified as a source finding, a calculated bound, a CPU check, or a proposed experiment. No T4 candidate was executed.

### 10.1 Compact MLP state without another large GEMM

The pinned Qwen MLP has three bias-free projections:

```text
G = gate(X)
U = up(X)
A = native_silu(G)
Y = down(A * U)
```

Only q/v projections receive adapters in this challenge. The MLP weights are frozen NF4, and this block contains no dropout. With the expected prepared FP32 residual stream and explicit FP16 bitsandbytes compute dtype, gate/up results follow `FP16 result -> FP32 widening`. An explicit compute dtype prevents the module from choosing FP32 compute merely because its input is FP32. [4,10,16,18,20]

This gives a narrow representation opportunity: **G and U contain only FP16-representable values, even though their tensors are FP32.** Their FP16 saved representations can reconstruct those FP32 values exactly for finite values under the checked conversion behavior. Native SiLU output, its product, norm outputs, adapted q/v sums, and arbitrary residuals do not share this guarantee.

PyTorch's derivative definitions require the SiLU input and both multiplication operands. Its cast backward needs tensor options rather than the original activation values. The frozen bitsandbytes matmul saves packed weights and quantization metadata rather than its input activation. Thus the baseline's wide MLP saved state is `G32`, `native_silu(G32)`, and `U32`. [20,25]

At `N=4*512=2048`, `H=1536`, `I=8960`, one `[N,I]` FP32 tensor is **70 MiB**, its FP16 representation is **35 MiB**, and `[N,H]` in FP16 is **6 MiB**. The calculated inventory is:

| Policy | Saved activation values per layer | MiB per layer | MiB across 28 layers |
| --- | --- | ---: | ---: |
| Native baseline | G32, SiLU(G32), U32 | 210 | **5,880** |
| Supplied SwiGLU | G32, U32 | 140 | **3,920** |
| M1: packed gate/up, pointwise replay | G16, U16 | 70 | **1,960** |
| M2a: replay one projection | X16 and one of G16/U16 | 41 | **1,148** |
| M2b: replay both projections | X16 | 6 | **168** |

The supplied kernel already removes one FP32 tensor per layer. Its calculated 1,960 MiB reduction agrees approximately with the organizer's reported 2,000 MiB improvement; that is supporting evidence for the inventory. [3,28]

**These totals describe retained activation values at maximum shape.** They exclude gradients, current-layer workspaces, other model state, the head, allocator reservation, and loading peaks. Shorter batches scale with their token count. The difference between table rows is not a measured reduction in `nvidia-smi` peak.

#### M1 implementation contract

Insert a small custom autograd function between the unchanged gate/up and down projections. Keep its inputs and output in their original FP32 dtype.

Forward computes the original native FP32 `silu(G) * U`, but saves only G16 and U16. Backward restores G32 and U32, then performs:

```text
A32  = native_silu(G32)
dU32 = mul_fp32(dZ32, A32)
dA32 = mul_fp32(dZ32, U32)
dG32 = native_silu_backward(dA32, G32)
```

Return the two FP32 gradients to the original projection graph. This replays pointwise work, with **zero additional large matrix multiplications**. Packing and restoration still consume time and temporary memory.

Using native SiLU initially separates the storage idea from arithmetic changes. The pinned CUDA forward evaluates `x / (1 + exp(-x))`; the starter's `x * sigmoid(x)` expression is algebraically equivalent but can round differently. A later kernel that promotes packed values in registers needs its own numerical comparison. [27,28]

Guard the exact frozen FP16-compute producer path, absence of gate/up adapters or extra FP32 epilogues, expected activation, and supported tensor layout. A broad FP32-to-FP16 compression policy is invalid. Likewise, casting the custom function's inputs to FP16 before doing SiLU would change the computation rather than only its storage.

Saved-tensor hooks are another mechanism, but unpacking must restore the original content, dtype, device, and size. Their packed objects must not retain the original FP32 tensors, or the memory saving disappears. Hooks can nest with only the innermost pair taking effect and can unpack more than once. An explicit small autograd function has a clearer state contract here. [26]

**Originality:** treat M1 as supporting memory work. The starter SwiGLU exclusion remains in force; a revised saved-state policy does not establish qualifying original-kernel time. Use independently developed kernels for the required runtime share. [3,6,28]

### 10.2 Replay expensive projections only when the time budget permits

A custom frozen-MLP backward can retain X16, replay gate/up, and compute the required input gradients. It does not need to replay down's forward: the down weight is frozen and its input gradient depends on its weight and incoming gradient. A hybrid saves one projection output and replays the other.

The original cast boundaries remain part of the contract:

```text
dY32 -> dY16 -> down-input-gradient GEMM -> dZ16 -> dZ32
replay gate/up using original X16, weights, shapes, and quantization
native FP32 SiLU/multiply backward -> dG32, dU32
round dG and dU separately to FP16
gate/up input-gradient GEMMs -> two FP16 dX branches
widen each branch to FP32, then add
```

Summing the two branches in FP16 and widening once changes the baseline. Keep original shapes first; chunking replay can change library dispatch and rounding. Release temporary dense weights after use rather than retaining them across all layers. The MLP has no stochastic operation; broadening replay to the attention block introduces separate dropout/RNG obligations. [16,20,26]

For multiplication and addition counted separately, each maximum-shape projection costs `2*N*H*I = 56.37144576 GFLOP`. Frozen MLP training has three forward and three input-gradient GEMMs.

| Policy | Additional GEMMs per layer | Extra work across 28 layers at max shape | Increase in MLP GEMM FLOPs |
| --- | ---: | ---: | ---: |
| M1 pointwise replay | 0 | 0 large-GEMM FLOPs | 0% |
| M2a one-projection replay | 1 | 1.5784 TFLOP/step | 16.7% |
| M2b gate/up replay | 2 | 3.1568 TFLOP/step | 33.3% |

These are operation counts, not latency estimates. Replay also adds dequantization, launches, and allocation traffic. If `T_mlp` is measured time in the six original MLP GEMMs, the rough extra GEMM term is `T_mlp/6` for M2a or `T_mlp/3` for M2b, before other costs and saved traffic. Compare the complete candidate with its remaining admission budget:

```text
additional_time_budget = 1.10 * baseline_command_time - current_candidate_command_time
```

Select M1, M2a, M2b, or a measured mixture from the time/memory tradeoff. Do not infer that the smallest retained-state row wins the competition.

### 10.3 Keep the head normalizer centered

PyTorch 2.11's native log-softmax computes `(z-m)-log(a)`. Ordinary integer-target CE uses log-softmax followed by NLL loss. Combining the maximum and log sum into a single FP32 `logZ` can discard low-order normalizer bits before subtraction. [31]

The custom head should save two FP32 values per supervised row and use:

```text
log_a = log(sum(exp(z - m)))
CE = -((z_target - m) - log_a)
p = exp((z - m) - log_a)
```

For two identical FP16-rounded logits at 60,000, our CPU check obtained a probability sum of **1.00174236** using collapsed FP32 `logZ`, versus **1.0** with the separate state. CE error against FP64 was about **0.00174093** versus **1.90e-9**. This is a stress case, not an observed task activation.

Across six cases and vocabulary tiles of 1,024, 2,048, and 4,096, the stable CPU implementation's worst CE error was **3.50e-6**, entropy error **6.68e-7**, and every checked argmax chose the correct lowest index. Appendix B reproduces these checks.

Use the actual upstream `grad_output` and provided denominator. The real-arithmetic expression `(p-one_hot)*scale` and the native sparse NLL/log-softmax backward can round differently when expanded into separate FP32 operations. A CPU illustration found 12 different FP16 stores among 10,025 selected probabilities. It establishes operation-order sensitivity, not which form matches CUDA FMA/exp behavior. Compare against the actual prepared baseline before choosing fused arithmetic. [31]

#### Pair normalizers with the logits that produced them

Changing chunk size, final-chunk shape, compaction, or strides can change GEMM rounding. PyTorch explicitly does not guarantee bitwise identity between mathematically equivalent full and sliced computations. [32]

The saved-logit policy preserves the candidate forward's exact FP16 logits. A recompute policy must compare its logits with those produced in forward. Reusing a forward normalizer with materially different recomputed logits can produce incorrectly normalized probabilities. Recomputing the normalizer restores internal consistency, but does not recreate the original forward's rounded values.

A conservative intermediate policy replays only original forward chunks that contain supervised positions, retaining their shapes and row positions. Count all replayed rows in the cost, including any unsupervised rows retained for that geometry.

#### When supervised recomputation can still save arithmetic

Let P be baseline padded positions, T the positions evaluated in head forward, S supervised prediction positions, and D/V hidden/vocabulary dimensions. The frozen head has no weight-gradient GEMM.

| Policy | Approximate head GEMM FLOPs |
| --- | --- |
| Dense baseline | `4*P*D*V` |
| Save supervised logits | `2*D*V*(T+S)` |
| Recompute supervised logits | `2*D*V*(T+2*S)` |
| Replay R rows to preserve chunk geometry | `2*D*V*(T+R+S)` |

When T=P, supervised recomputation uses fewer total head GEMM FLOPs than the dense baseline's forward plus input-gradient backward if **S/T < 0.5**. For the illustrative case P=T=2,048 and S=512, the counts are 1.9118 TFLOP for dense baseline, 1.1949 for saved logits, and 1.4338 for supervised replay. Actual supervision fractions and GEMM efficiency decide runtime.

### 10.4 A concrete two-stage vocabulary reducer

This is an original design proposal for L1/L2, not a copied implementation.

**Stage A:** one program per token and vocabulary tile. Load FP16 logits, promote to FP32, and write four 32-bit fields: partial maximum, exponential sum, centered weighted sum, and lowest-index argmax. Mask padded vocabulary lanes before they can affect statistics or indices.

**Stage B:** one program per token merges those partials with the stable formula in section 3. It produces centered `(m, log_a)`, entropy, CE, and correctness. Load the target logit once here, only when its target is valid. Ignored targets retain zero CE but can still require entropy. Batch/row aggregation then processes only small summaries.

| Vocabulary tile | Tiles per token for V=151,936 | Partial state for C=64 | Partial state for C=128 |
| ---: | ---: | ---: | ---: |
| 1,024 | 149 | 149 KiB | 298 KiB |
| 2,048 | 75 | 75 KiB | 150 KiB |
| 4,096 | 38 | 38 KiB | 76 KiB |

At C=64 and tile 2,048, the first stage has 4,800 independent programs and 75 KiB of partial state. These are layout calculations; occupancy, register pressure, spills, and launch cost require T4 measurement. Start with a small fixed configuration set to control cold compilation.

Resolve ties by original vocabulary index, guard masked `0 * -inf` expressions, and retain failure visibility for real NaN/Inf values. Avoid floating-point atomic aggregation in the first deterministic implementation. A large full-vocabulary program is unnecessary.

### 10.5 Attention: distinguish Python dtype from effective SDPA dtype

Python-visible Q/K/V can be FP32 after the projection and RoPE paths. Nevertheless, PyTorch registers SDPA under CUDA autocast's lower-precision policy, so normal FP16 AMP presents FP16 inputs to its backend. The math backend can then promote them to FP32 internally. Simply adding another FP16 cast is not evidence of removing an FP32 attention bottleneck. [14-16,30]

In the examined PyTorch 2.11 source, T4/SM75 cannot use its Flash or cuDNN SDPA families, whose hardware checks begin at SM80. The memory-efficient family supports SM75, including FP16 and FP32, but requires equal Q/K/V head counts. Native GQA with Q12/K2/V2 fails that check. Other build, mask, stride, and alignment checks still apply. [30]

Transformers 5.16.1 enables native GQA when the explicit attention mask is absent and K/V dimensions meet its condition. It does not include a T4-specific check. In ordinary uncached training, an all-true padding mask can be omitted; batches with actual padding retain a mask and explicitly repeat K/V. [29]

| Batch path | Call shape | Conditional T4 implication under examined source |
| --- | --- | --- |
| No padded positions; mask omitted | Q12/K2/V2, native GQA | Standard fused options reject it; math is expected when enabled. |
| At least one padded position | Q12/K12/V12 after explicit repeat | Memory-efficient attention can be eligible if the other checks pass. |

The frequency of these cases is unknown. Four rows all truncated to 512 can produce the first case, as can equal shorter lengths. Hidden rows may change the distribution.

#### K2: guarded dispatch candidate

For the supported maskless T4 training shape, explicitly repeat K/V and call SDPA with `enable_gqa=False`, preserving causal behavior, scale, dropout, mask, and output layout. Check the installed backend's eligibility and executed kernels before claiming a benefit.

This is a **numerically distinct candidate**. Switching math to memory-efficient attention changes reduction order. It also moves K/V gradient rounding: native-GQA math accumulates replicated-head gradients before the compact FP16 boundary, while an explicit-repeat path crosses FP16 separately for expanded heads. Both changes require full training-quality validation. A pure Torch adapter provides no original-Triton attribution. [30]

For a development eligibility check on the examined API:

```python
from torch.backends.cuda import SDPAParams, can_use_efficient_attention

params = SDPAParams(q, k, v, mask, dropout_p, is_causal, enable_gqa)
eligible = can_use_efficient_attention(params, debug=True)
```

The constructor has seven arguments, including `enable_gqa`, and no scale argument. Supply effective post-autocast dtypes when investigating dispatch. Eligibility does not prove selection; profiler kernel names establish which path actually ran. Keep this diagnostic outside scored timing. [30]

### 10.6 K1: fuse K/V repetition with casting on the existing masked path

The current masked path expands FP32 K/V from two heads to twelve, then SDPA autocast converts the expanded values to FP16. An original kernel can read compact FP32 K/V and write expanded FP16 K/V directly. Forward values are unchanged by this reordering of copying and per-element conversion. [29,30]

Its backward must reproduce the existing dtype sequence:

```text
expanded incoming FP16 gradient
-> promote each element to FP32
-> sum six replicated heads in FP32
-> return compact FP32 gradient
```

Casting compact K/V first and relying on ordinary FP16 repeat backward moves the grouped sum into FP16. Our CPU counterexample `[1, 2^-12, 0, 0, 0, 0]` sums to **1.000244140625** in FP32 but **1.0** when the result is rounded to FP16 first. Even with the correct dtype, a new reduction tree still needs numerical comparison.

At the maximum shape, the expanded FP32 K/V pair occupies **24 MiB**; the required expanded FP16 pair occupies **12 MiB**. Fusion avoids writing and rereading the expanded FP32 pair. This is transient allocation and memory traffic, not evidence of 24 MiB times 28 layers of retained-state savings.

Keep K1's first test on the already-masked path, where it can preserve the selected SDPA backend and effective input values/shapes. Combining it with K2's maskless dispatch change must be evaluated as the separate numerical variant above.

### 10.7 Experiment additions and evidence gates

| ID | New candidate | First comparison | Acceptance evidence |
| --- | --- | --- | --- |
| M1 | Packed gate/up state; native FP32 pointwise replay | Native baseline and supplied SwiGLU | Verified producer chain; restored values; output/input gradients; actual saved-state and full peak/time. |
| M2a/M2b | Replay one/both frozen projections | M1 | Original cast boundaries and shapes; extra replay/dequant time; complete training-quality and time gates. |
| K1 | Original cast-plus-repeat with FP32 grouped backward sum | Existing masked attention path | Same effective SDPA inputs; gradient/reduction comparison; removed traffic; qualifying runtime attribution. |
| K2 | Explicit-repeat adapter for maskless GQA on T4 | Actual maskless math path | Backend selection; causal/mask/dropout fidelity; loss trajectory and final adapter quality. |

L1 now uses the two-stage reducer. L2 begins with saved supervised logits and centered state; L3 retains the phase-scoped weight representation experiments. M1 is an early parallel memory branch. M2 and K2 depend on measured time headroom and branch frequency.

The kernel-share requirement applies to the **combined final candidate**, whose denominator changes as other work becomes faster. Pure Python/PyTorch memory policies, existing library kernels, and the excluded starter do not count. An independently original, substantive memory-traffic kernel can qualify under the same source, execution, and review requirements. Add useful original work where the remaining profile justifies it, and measure its share; do not add unnecessary work to inflate attribution. [6,7]

### 10.8 Executed round-3 checks and remaining uncertainty

NumPy 2.3.5 was used. The storage/cast checks used seed 20261030; the head numerical audit used seed 864213. Both scripts are in Appendix B.

| CPU check | Result |
| --- | --- |
| Exhaustive finite binary16 widening/packing/restoration | All **63,488** encodings preserved both packed FP16 bits and restored FP32 bits, including signed zeros. |
| Packed-state native-style pointwise arithmetic | Forward, dGate, and dUp were bitwise equal in the NumPy check over 8,192 values. |
| Counterexamples to general FP32 compression | Arbitrary FP32 values and native-style SiLU outputs lost information when packed to FP16. |
| Cast/repeat forward | Bitwise-equal FP16 values in both copy/cast orders for the tested tensor. |
| Cast/repeat backward | The FP32-versus-FP16 six-head-sum counterexample differed by **0.000244140625**. |
| Stable head reduction, six cases and three tile sizes | Worst CE error **3.50e-6**, entropy error **6.68e-7**, correct checked argmax ties. |

These checks establish representation and algebra behavior on CPU. They do not establish CUDA conversion of subnormals or NaN payloads, native GPU exp/FMA behavior, custom autograd correctness, runtime tensor provenance, backend selection, memory peaks, or training convergence.

The next T4 development run should capture metadata without retaining extra activation tensors: producer/output dtypes, shapes and strides; unique storage sizes; actual mask paths; effective SDPA inputs and selected kernels; and supervised-token fractions. Saved-tensor hooks cover `save_for_backward` and native saves, but do not inventory every tensor kept in arbitrary context attributes or runtime caches. Use that inventory together with a memory trace and the official whole-process measurement. Avoid letting the instrumentation itself keep the tensors alive. [20,26]

## 11. Research round 4: compiler feasibility, combined memory, and the real time budget

**Main decision:** retain the large library GEMMs while implementing original reductions, logit-gradient generation, and useful memory-traffic kernels. Combine M1 with the streamed head before committing to offload or expensive replay. The original-kernel share must be measured on that combined candidate; no proposed component has established admission yet.

### 11.1 Check the compiler path before writing a large custom GEMM

The exact Triton 3.6.0 `BlockedToMMA.matchAndRewrite` implementation rejects `computeCapability < 80` and takes the FMA path. This includes T4's SM75. The release README lists supported NVIDIA hardware from CC8.0; upstream PR 5066 documents the earlier change. **An ordinary FP16 `tl.dot` in this release is not a T4 tensor-core implementation.** [33]

Turing MMA instruction strings remain in `MMAv2.cpp`, but the ordinary acceleration pass stops before that path. The NVIDIA backend also places SM75 outside its normal architecture-8/9/10+ loop-pipelining branches. Treating a `num_stages` sweep as standard Ampere pipeline tuning would therefore be unjustified. [34]

The immediate development gate is to record the installed Torch/Triton versions, target architecture, generated PTX, and an actual T4 smoke test. The template does not explicitly pin these two packages. Reduction and elementwise candidates still need runtime validation on the installed stack. [13,33]

Turing hardware itself supports FP16 matrix operands and FP32 accumulation. A separately authored SM75 inline-PTX MMA implementation is conceivable, but requires correct warp/lane fragments, packing, layout, masking and synchronization. Triton's elementwise assembly interface does not specify which input groups reach each invocation; inserting an MMA instruction into arbitrary elementwise code is insufficient. Keep this as a later feasibility project. [35]

Ordinary `@triton.experimental.gluon.jit` and `@gluon.jit` are not recognized by the current originality classifier. Use the supported ordinary `@triton.jit` form for candidate work; the existing cuBLAS GEMMs remain non-qualifying library work. [6]

### 11.2 G0: generate the logit gradient once, then use cuBLAS

Implement the following original producer for a bounded chunk of supervised prediction rows:

1. Read the saved FP16 logits and the matching FP32 `(m, log_a)` values.
2. Reconstruct centered log probabilities/probabilities in FP32.
3. Apply shifted targets, the correct denominator, and the actual upstream gradient using the validated native NLL/log-softmax operation order.
4. Round the completed logit gradient to FP16 **after applying the upstream scale**.
5. Run `dH16 = dZ16 @ W16` through the original library path and precision settings.
6. Widen to the original hidden dtype and scatter into the full terminal hidden-gradient shape, with zero direct head contributions outside supervised rows.

This preserves the important FP16 boundary at both sides of the GEMM. Directly returning an unrounded FP32 accumulator would be a different numerical path. The native operation-order audit and the round-3 counterexample remain relevant; the mathematical expression `(p - one_hot) * scale` alone does not establish identical rounding. [14,16,31,32]

For `V=151936`, FP16 gradient scratch is `2*C*V` bytes:

| Supervised rows per chunk C | FP16 dZ scratch | FP16 supervised dH output |
| ---: | ---: | ---: |
| 64 | 18.546875 MiB | 0.1875 MiB |
| 128 | 37.093750 MiB | 0.3750 MiB |

Reuse scratch only after its dependent GEMM has completed; same-stream ordering is the first implementation. These amounts exclude saved logits, the head weight, full hidden-gradient storage, allocator reservation, and library workspace.

#### Why not reconstruct probabilities inside every GEMM tile?

A conventional output-tiled `dH` kernel repeats reconstruction once per hidden-column tile. With D=1536, `BLOCK_D=128` gives 12 repeats and `BLOCK_D=64` gives 24. At S=512 supervised rows, one producer evaluates 77,791,232 probabilities; the two fused layouts evaluate 933,494,784 or 1,866,989,568 respectively. The GEMM itself costs `2*S*V*D = 238.974664704 GFLOP`.

Fusion removes a gradient-buffer read/write stage, but adds repeated exponentials and live FP32 state. For an equal tiled schedule, logical FP16 logit/gradient traffic is `2*S*V*ceil(D/B_D)` for inline reconstruction versus `2*S*V*(2+ceil(D/B_D))` for the separate producer and tiled consumer. At S=512 and B_D=128, those are 1,780.5 and 2,077.25 MiB. **Logical load counts are not measured DRAM traffic**; cache reuse and the actual cuBLAS schedule matter.

Vocabulary slabs could shrink scratch further: 64 rows by 8,192 columns is 1 MiB in FP16. They also add partial GEMMs and launches. A valid first comparison needs FP32 partial outputs and accumulation with one final FP16 rounding, rather than summing already-rounded FP16 partial outputs. Defer slabs until row chunks show a measured problem.

#### G1: conditional split-K research

Only after an efficient original SM75 MMA path is proven, consider deterministic vocabulary splitting. A `64x128` row/hidden tile yields only 12 unsplit blocks for S=64, versus 96 for S=512. Splitting may help the smaller grid, but also adds scratch and a reduction launch. FP32 partial storage is `4*J*S*D` bytes: four/eight splits need 1.5/3 MiB at S=64 or 12/24 MiB at S=512. A fixed-order merge avoids floating-point atomics while still changing the reduction tree relative to cuBLAS. Quality checks remain necessary. [32; calculated]

The same output tile has 32 KiB of FP32 accumulators, or 64 values per thread with four warps, before operands and temporaries. Turing's register/shared-memory limits make additional inline probability state a concrete concern. These resource counts motivate a small configuration set; they do not predict achieved occupancy. [35]

### 11.3 O1: selective offload after compact storage

M1 retains 70 MiB per layer at N=2048, or 1,960 MiB across 28 layers. Offload can remove some of that **remaining** state; it cannot claim the same bytes already removed by compact storage. The T4's host interface is PCIe Gen3 x16. Its advertised 320+ GB/s describes GDDR6, not the CPU link. NVIDIA gives roughly 12 GB/s as a pinned-transfer example; use it below only as an explicit hypothetical effective rate. [38]

Let k be the number of first layers offloaded. With maximum shape in every step, payload in each direction is `70*k MiB`, and 64-step copy service is:

`64 * 2 * 70 * k * 2^20 / (12 * 10^9)` seconds.

| First layers offloaded k | Host payload per step, one direction | Remaining M1 state at a drained head boundary | Copy service across 64 maximum-shape steps at hypothetical 12 GB/s |
| ---: | ---: | ---: | ---: |
| 0 | 0 MiB | 1,960 MiB | 0 s |
| 7 | 490 MiB | 1,470 MiB | 5.481 s |
| 14 | 980 MiB | 980 MiB | 10.961 s |
| 21 | 1,470 MiB | 490 MiB | 16.442 s |
| 28 | 1,960 MiB | 0 MiB | 21.922 s |

The full case moves 3,920 MiB per step and **263.06674688 decimal GB** over 64 steps. These are transfer-service totals before overlap, not wall-time penalties. More generally, service is `payload/B_D2H + payload/B_H2D`; compute and, where supported, opposite-direction transfers can overlap. Real padded lengths, available bandwidth, contention, launch-ahead allocation and dependencies change elapsed time. Zero resident state at the idealized head boundary does not mean zero activation memory during forward/backward.

A prefix of early layers is a useful initial hypothesis because those values are produced early and consumed late in reverse-order backward, allowing a longer transfer window. Start with k=7 and a small bounded prefetch window; increase only when the actual time/peak tradeoff is favorable. Persisting a growing queue of restored layers can erase the benefit.

#### The offload implementation needs separate readiness and lifetime guarantees

PyTorch 2.11 `save_on_cpu(pin_memory=True)` uses a blocking `packed.copy_(tensor)` for packing; only unpack requests a nonblocking transfer. It does not by itself provide the proposed asynchronous D2H schedule. Keep Soup's fixed offloading switch unchanged and implement any candidate through original selective code. [26]

For a contiguous same-dtype transfer, nonblocking copy uses the current stream; dtype conversion or layout changes can introduce temporary allocations. Preallocate a bounded pool of same-dtype pinned buffers, and account for those buffers separately from GPU memory. [36]

The required ordering is:

- The copy stream waits for production, then issues D2H. GPU source storage remains valid until the copy and every other consumer finish.
- A slot's pinned payload remains valid and unmodified through its later H2D completion; it cannot be recycled just because D2H completed.
- Backward prefetch waits for the relevant D2H event, restores a bounded number of slots, and records H2D completion. Compute waits for that completion before use.
- Restored device storage remains valid through its backward consumer. Use `record_stream` or explicit GPU-side stream/event ordering back to the allocation's creation stream before reuse or release; avoid a device-wide synchronization per transfer.

`record_stream` protects allocator reuse; it does not make a consumer wait for its producer. Stream waits establish readiness; they do not eliminate all retained references. Avoid retaining original CUDA tensors in context objects or closures after offload, and explicitly drain/clean up on error. [26,36]

A two-buffer GPU staging pool still needs enough CPU slots to retain every offloaded layer until backward: seven maximum-shape layers require 490 MiB of host payload. Reuse that bounded pool across steps instead of retaining prior batches. Start decoder prefetch after the head weight is released when the head phase determines the peak; moving prefetch earlier trades more overlap for another 70 MiB per restored layer.

Offloading the smaller M2b X16 state would move only 168 MiB in each direction across all layers, or 1.879 seconds of aggregate service over 64 maximum-shape steps at hypothetical 12 GB/s. That option inherits M2b's extra 3.157 TFLOP of projection replay per maximum-shape step. It is not an independent cheap replacement for M1.

### 11.4 Combine memory changes by lifetime, not by adding advertised savings

Count unique storage in each phase, then take the maximum over phases. A lower retained-state subtotal is evidence of a useful target, not a prediction of the competition's externally sampled peak.

Three corrections matter:

1. **L1 preserves native CE state.** Suppressing returned logits removes output conversion/diagnostic traffic, but native log-softmax backward still retains its FP32 result. At P=2048, one full `[P,V]` FP32 tensor is 1,187 MiB. The native backward can overlap its saved result with dense gradient temporaries. [25,31]
2. **Count the head compute weight once.** Native linear backward already saves the weight needed for input gradients. A separate shadow that leaves the original cast path in place can duplicate the 445.125 MiB representation. [14,25]
3. **Forward and head backward are adjacent in the ordinary single-loss path.** In L2/L3 custom backward, releasing the weight after head forward and recasting before head backward does not reduce their shared maximum unless a real intervening allocation peak makes the gap useful. Start by retaining one representation through head backward, then release it before decoder backward. L1's native autograd owns its saved weight; deleting an external reference does not release that storage. This is a lifetime deduction, not a measured speed result.

The following deliberately counts only M1 saved values, one head compute weight, and the indicated retained loss state. It assumes N=P=2048, S=512, completed offloads, and that the selected saved tensors are still live at head backward entry:

| Selected resident components | No offload | First 7 layers offloaded | First 14 layers offloaded |
| --- | ---: | ---: | ---: |
| M1 + FP16 head weight + native FP32 log-softmax result | 3,592.125 MiB | 3,102.125 MiB | 2,612.125 MiB |
| M1 + FP16 head weight + saved supervised FP16 logits | 2,553.500 MiB | 2,063.500 MiB | 1,573.500 MiB |

**These are incomplete component inventories.** Add the FP32 source weight, quantized base, other decoder state, gradients, workspaces, chunk temporaries, prefetch buffers, and allocator behavior before interpreting a total. For example, a C64 head gradient chunk adds 18.546875 MiB while live; a reference implementation that also materializes C64 FP32 logits adds another 37.09375 MiB in that phase. The original tiled reducer should avoid that whole FP32 chunk.

PyTorch distinguishes live tensor allocations from memory reserved by its caching allocator; unused reserved blocks still appear in `nvidia-smi`. The harness's fixed expandable-segment setting does not make these quantities identical. Track allocated, reserved, and external memory together, including model loading. Do not assume deleting tensors lowers the score's peak, or add per-step `empty_cache()` calls as a substitute for understanding reuse. [5,37]

A CPU event ledger checked fifteen simplified schedules and found equal tracked peaks for releasing/recasting the head weight versus retaining it through adjacent head forward/backward. The ledger assumes drained D2H, bounded prefetch, and an explicitly illustrative 350 MiB current-layer workspace. It omits real allocator/transfer behavior and illustrates the lifetime argument only.

### 11.5 Budget the official attribution and cold command together

Let L be the qualifying original kernel names launched in **every** successful solution repeat. For repeat r, let K_r be their summed `own_kernel_ms`, and let D_r be the recorded `timed_loop_s`, calculated as `round(sum(step_s[2:]), 3)` in seconds. The official gate is:

`min_r K_r / (1000 * D_r) >= 0.05`.

The normal denominator has 62 timed steps. At hypothetical mean step durations of 500, 750, or 1,000 ms, the corresponding average original event-time budgets are 25, 37.5, or 50 ms. Improving a qualifying kernel can reduce its share while improving overall time; evaluate the final combination instead of optimizing the ratio itself. Use unique kernel entry-point names so launch/file provenance is unambiguous. [6,7; calculated]

The callback interval contains forward/backward, clipping, optimizer and scheduler work. The examined Transformers loop fetches batches before `on_step_begin` and performs normal post-step logging/save/evaluation after `on_step_end`; computations inside `compute_loss`, including diagnostics, can still be inside. `train_loop_s` also includes intervening gaps and is a different metric. All scored-process work still matters to command time. [7,16]

Every timed `solution/` launch receives two current-stream events, including starter kernels later excluded from credit. The harness retains the pairs and queries them after training. Launch bookkeeping adds step overhead, and final event processing adds command overhead. A late compilation inside `JITFunction.run` can also contaminate an event interval if the GPU waits between event records. Overlapping stream intervals are summed, not merged into exclusive GPU busy time. Call the reported value **official event-time share** and use the profiler separately to explain execution. [7,23,36]

For the whole command, define additional cold/non-step overhead DeltaC and net step savings s_i:

`T_candidate = T_baseline + DeltaC - sum(s_i, i=1..64)`.

Break-even occurs at `sum(s_i) = DeltaC`; a positive time improvement requires `sum(s_i) > DeltaC`. Reaching the 1.05 speed ratio requires `sum(s_i) >= DeltaC + T_baseline/21`.

| Illustrative extra cold overhead | Average step saving to break even | Average step saving for 1.05x on an illustrative 80 s baseline |
| ---: | ---: | ---: |
| 0 s | 0 ms | 59.52 ms |
| 1 s | 15.63 ms | 75.15 ms |
| 2 s | 31.25 ms | 90.77 ms |

The amortization horizon is 64 steps; the attribution horizon is 62. Fresh caches, loading, compilation, saving and instrumentation processing count. The optional profiler run is separate, with 6 waiting, 2 warm-up and 16 active steps; its GPU-time table cannot substitute for the official share. Keep one or two justified configurations initially, then compare cold runs through the unmodified harness. [5,7,8]

### 11.6 E1: conditional offload of the unchanged tied FP32 source

This is a later experiment, separate from the recommended head-cast reuse policy. Soup's T4 resolver requests FP16 loading; the examined Qwen2 class has no special FP32-loading override for the embedding/head; PEFT subsequently widens non-quantized half parameters to FP32. This source chain suggests the frozen tied table's FP32 values are exactly recoverable from FP16. It is a runtime property to verify across the complete loaded tensor, including signed zero and finite-value checks. [10,16,18,39]

If verified, an own-code offload design could preserve the original tied FP32 Parameter on CPU while retaining one shared FP16 runtime representation on the GPU. Head computation consumes that representation; a custom frozen embedding lookup widens gathered values back to FP32 before decoder use. The Parameter's values, dtype, tying, trainable set and serialized adapter must remain unchanged.

The GPU source is 890.25 MiB. Relative to a candidate already retaining source plus half shadow, relocating the source removes that amount of live GPU storage. Relative to source-only residency outside head work, replacing 890.25 MiB by a persistent 445.125 MiB shadow gives a 445.125 MiB component difference. Neither calculation predicts the full-command peak: startup copies, cached allocations and the loading peak may dominate.

Mixed-device model preparation, `model.device`, device hooks, embedding behavior, reload and serialization all need explicit integration tests. Do not silently change a Parameter to half or assume a partial tensor sample proves exact recoverability. Own offload is allowed in the rules, but this specific representation is not expressly adjudicated; unchanged computation and the jury's review remain the basis for eligibility. The default branch continues to retain the original source on GPU. [3,6,15,39]

### 11.7 Revised implementation sequence and evidence

| Stage | Build or measure | Advance when |
| --- | --- | --- |
| 0 | Actual versions; original reduction smoke test; dtype/backend/allocation trace | The installed stack executes the intended kernels and reference behavior is recorded. |
| 1 | L1 diagnostics and M1 compact MLP state | Each passes local numerical checks, then their combination preserves task outputs. |
| 2 | L2/L3 saved-logit head with G0 producer and cuBLAS | Correct AMP boundaries, supervised gradients, metrics, cold time, and actual memory. |
| 3 | Original RMSNorm, LoRA epilogue, K1 cast/repeat as justified by the remaining profile | The complete candidate satisfies the original event-time gate with useful work. |
| 4 | O1 prefix offload, starting at 7 layers | A measured peak reduction is worth its exposed transfer/launch cost and host footprint. |
| 5 | M2 replay, K2 dispatch change, E1 tied-source offload, or G1 custom MMA | The specific numerical/integration/feasibility prerequisite is established. |

Record each repeat's command time, timed-step denominator, common qualifying names, per-name event totals and launches, external/allocated/reserved peaks, late compilations, padded lengths and supervision fractions. Keep quality and regression results with the same candidate identity. A policy selected from public shapes must continue to work on unseen rows and the hidden seed.

Round 4 executed CPU arithmetic for head tiles, scratch, transfer service and cold budgets, plus the simplified lifetime ledger. It inspected the exact compiler and dependency sources. **No Triton compilation, CUDA execution, measured speedup, measured memory saving, or candidate-quality pass occurred in this research environment.**

## 12. Research round 5: precise fusion contracts and useful qualifying work

**Main decision:** the next original kernels should remove identifiable passes while preserving the prepared model's dtype and gradient boundaries. R1 frozen RMSNorm, P1 query RoPE/cast, A1 LoRA epilogues, and K1 masked K/V cast/repeat are concrete candidates. Their combined event-time share remains unmeasured. The head reducer now has an explicit conservative math mode and a more parallel second stage.

### 12.1 Preserve the first layer's gradient topology

With the fixed checkpointing setting disabled, the inspected Soup/PEFT/TRL path does not install an embedding input-gradient hook for this quantized LoRA model. The frozen embedding output therefore is expected to have `requires_grad=False`. The first input norm and frozen K projection have no input-gradient path; q/v LoRA branches introduce trainable dependencies. Later residual streams do require gradients. This is an inference from exact sources, pending a trace of the prepared runtime. [10,16–18,41]

| Proposed work across 28 layers | Forward invocations | Expected backward invocations |
| --- | ---: | ---: |
| All RMSNorm sites, including final norm | 57 | 56 |
| q/v LoRA epilogues, counted separately | 56 | 56 |
| K/V cast/repeat, counted separately when its branch applies | 56 | 55 |

The first normalized activation still supplies LoRA parameter-gradient computation. Its lack of a norm backward does not make that activation unnecessary.

A combined custom Q/K or K/V autograd function can accidentally make its first K output differentiable because another input requires gradients. Preserve the original metadata with `ctx.mark_non_differentiable` for that output. If using `ctx.set_materialize_grads(False)`, accept `None` output gradients and use `ctx.needs_input_grad` before launching each backward path. Marking an output non-differentiable alone does not disable the default materialization of undefined gradients as zero tensors. This matters when the output is a large expanded K tensor. [41]

Record the actual flags before optimizing; do not add a gradient hook merely to make custom backward code run. The resulting SDPA backend's internal work still needs profiling—graph metadata alone does not prove how much of its K-gradient arithmetic it skips.

### 12.2 R1 and R2: RMSNorm traffic, arithmetic, and the residual boundary

#### R1: freeze the implementation contract, not just the formula

The expected norm input, frozen gamma, and output are FP32. The two explicit dtype conversions in Qwen's norm are then no-ops. Native autograd already retains the original input, row reciprocal-square-root, and references to the existing gamma. Squared input, mean, variance-plus-epsilon, and normalized intermediate are temporary rather than separate full-sized retained activations. **The main opportunity is fewer launches and temporary passes, not another large saved-state compression.** [16,25,40]

Use one program per row, `D=1536`, and a masked `BLOCK_D=2048`; padded lanes contribute zero and the denominator remains **1536**. Preserve the existing epsilon, expected to be `1e-6`, and the existing Parameter object. Start with four warps and compare eight only after correctness. No matrix multiplication, gamma-gradient reduction, or atomics are needed. [9,16; proposed design]

Forward computes the rounded FP32 square, mean, epsilon addition, reciprocal-square-root, normalization, and gamma multiplication. The inspected mean scales its completed FP32 accumulator by a rounded `FP32(1/D)` factor; keep that multiplication separate from the epsilon addition. Preserve those operation boundaries initially with `enable_fp_fusion=False`; a custom reduction tree can still differ from the native tree. Save the computed row statistic `r` for backward. At N=2048 it occupies only 8 KiB per site, so recomputing it has little memory upside.

For incoming gradient `dy`, let `RN` denote a separate FP32 rounding boundary. A backward that follows the source's staging is:

```text
u_i       = RN(dy_i * gamma_i)
direct_i  = RN(u_i * r)
dr        = sum_FP32(RN(u_i * x_i))
r3        = RN(RN(r * r) * r)
dv        = RN(RN(-0.5 * dr) * r3)
dq        = RN(dv * FP32(1 / D))
via_var_i = RN(dq * RN(2 * x_i))
dx_i      = RN(direct_i + via_var_i)
```

Here `sum_FP32` specifies an abstract FP32 reduction, not a particular grouping or a promise to reproduce native bits. CUDA's inspected scalar division path multiplies by a rounded reciprocal, which motivates `FP32(1/D)` in the mean backward. The compact mathematical identity `r * (u - x*r*r*mean(u*x))` changes staging and belongs in a separate numerical variant. Preserve GradScaler's actual upstream scale. The native `rsqrt` call, Triton's ordinary primitive, and the explicit libdevice primitive also need comparison on the installed stack; none is assumed bitwise interchangeable. [25,34,40,42]

#### R2: a local two-output residual fusion

Start at the 28 attention-output additions immediately before post-attention RMSNorm. In the expected path, SDPA supplies FP16 to the frozen bnb output projection, which returns its original input dtype. The attention output is therefore FP16 while the skip stream is FP32. The proposed forward is:

```text
S32 = FP32(residual32 + widen(attention16))
Y32 = RMSNorm(S32)
return S32, Y32
```

Return **both fresh outputs**: S feeds the MLP residual branch, and Y feeds the unchanged MLP. Save S and r, sharing S's returned storage. Backward combines the direct S gradient and the norm contribution in FP32, then returns the combined gradient in each add input's original dtype. Cast the attention-branch gradient to FP16 only after that combination. The MLP-down output follows a different FP32-input/FP32-output path; do not assume both block additions have the same dtype contract. [16,20,29,30]

Relative to standalone R1 plus a separate add, this fusion can remove the forward reread of S: `4*N*D`, or **12 MiB per maximum-shape site**. It must still write S for the skip branch. Native add backward can forward an existing gradient, so a separate add-backward kernel is not a guaranteed saving. Grouping direct, variance, and external skip contributions differently can also change rounding. These are reasons to validate the local R2 variant after R1, before attempting interlayer fusion.

Executed NumPy 2.3.5 checks, seed `281536`, used FP64 finite differences and explicit FP32 staging, with a CPU sqrt/reciprocal surrogate rather than CUDA rsqrt:

| Check | Observed CPU result |
| --- | --- |
| Norm and two-output residual Jacobians | Maximum directional finite-difference absolute error `2.1474e-9` |
| Ordinary 16×1536 input, upstream scale 1: staged versus compact backward | 8,602 of 24,576 FP32 values differ; maximum `4.7684e-7` |
| Near-radial cancellation case | Maximum difference `9.8395e-7`, against staged gradient maximum `5.2452e-6` |
| Regrouping three residual-gradient terms | Maximum difference `1.2654e-7` |
| Cast before versus after a cancelling sum | Sum in FP32 then cast gives `5.00083e-5`; separately casting the branches first gives zero |

The test set also varies input magnitude and upstream scales through 65,536. These examples justify preserving operation order; they are neither CUDA-equivalence tests nor estimates of training drift.

### 12.3 P1: fuse query RoPE and its final cast first

Qwen already computes position-dependent cosine/sine tensors once per model forward and shares them across layers. Reuse the supplied tensors, including their padding/position-ID semantics. The original rotary expression is two FP32 products followed by an FP32 addition; SDPA autocast later rounds its inputs to FP16. [16,29,30]

**P1 should first handle Q only:** take the expected FP32 query and supplied cos/sin, compute RoPE with separate FP32 rounding, and store the FP16 query needed by the existing FP16-autocast SDPA path. Keep the original FP32 K RoPE output and its existing repetition/cast path. Guard the dtype/autocast/layout assumptions, and fall back when they are absent. Preserve the effective attention mask, scale, causal/dropout arguments, output layout, and backend selection.

Simply returning both Q and K as FP16 from `apply_rotary_pos_emb` is unsafe on the masked branch. That places K's cast **before** head repetition. Backward would then reduce replicated gradients in FP16, whereas the baseline casts those gradients to FP32 before the grouped reduction. Forward values alone cannot catch this error. [29,30; section 10.6]

For a split input `x=(a,b)`, cosine halves `(c0,c1)`, sine halves `(s0,s1)`, and incoming rotated gradient `(g0,g1)`, the transpose is:

```text
da = g0*c0 + g1*s1
db = g1*c1 - g0*s0
```

Widen the incoming FP16 gradient first, then preserve separate FP32 products and additions. This form does not assume the two cosine or sine halves are identical. Frozen positional tensors suffice for backward; retaining a new full Q/K input is unnecessary. Native RoPE also does not retain that full input solely to differentiate against it, so this is principally a launch/traffic opportunity. [16,25]

**P2, a later masked-K extension, must own the whole boundary:** FP32 K rotation, logical head repetition, and final FP16 storage. Backward widens the expanded gradient, sums its six groups in FP32, and then applies the rotary transpose in FP32. Rotating each group before summing is algebraically equivalent but changes rounding. A custom attention integration must skip the original repetition after producing expanded K; feeding it unchanged to the existing helper would repeat it again. Keep the first K output non-differentiable where appropriate. P2 overlaps K1's work and must replace that component in cost accounting.

The initial P1/P2 kernels should disable FMA contraction. FMA usually improves accuracy relative to exact arithmetic, but it rounds differently from the existing multiply/add chain; agreement with that chain is the relevant comparison. Disabling contraction does not settle the reduction tree or other compiler choices. [43]

Executed CPU check: seed `202610305`, NumPy 2.3.5, 2,048 head vectors of dimension 128. Inputs model FP32 sums of half-origin values; positional values are FP32. The FMA comparison uses the host's scalar `fmaf`, not a GPU emulator.

| Variant compared with the staged reference | Observed difference |
| --- | --- |
| Cast the input to FP16 before rotation | 74,305 of 262,144 final FP16 outputs change |
| Contract one multiply/add into FMA | 18 final FP16 outputs change; 64,981 intermediate FP32 outputs change |
| Rotate six gradient groups before summing | Maximum FP32 difference `1.4305e-6`; 102 eventual FP16 gradient stores change |
| Rotary transpose validation | Adjoint error `1.78e-15`; nine finite differences have maximum absolute error `8.57e-11` |

These are constructed counterexamples to moving casts or regrouping arithmetic, not failure rates for real training batches.

### 12.4 Head reduction: choose elementary functions explicitly

The pinned TRL entropy baseline is already stable: it computes `log_softmax`, then `exp(logps)`, then the negative sum of `exp(logps)*logps`, over flattened chunks of 128 rows. The prepared stack supplies FP32-widened FP16 logits. Our opportunity is to share vocabulary scans and avoid large intermediates; it does not rely on correcting an unstable TRL formula. [15,17,31]

In Triton 3.6's NVIDIA FP32 lowering, ordinary `tl.exp(x)` uses an approximate base-two exponential after scaling by `log2(e)`. Explicit libdevice functions map to `__nv_expf` and `__nv_logf`. Setting `enable_fp_fusion=False` does **not** turn ordinary `tl.exp` into libdevice exp. [42]

Use a small, interpretable comparison:

| Variant | Purpose |
| --- | --- |
| Explicit libdevice exp/log, FP32 state, contraction disabled | Initial diagnostic mode that limits simultaneous changes |
| Ordinary approximate exp in Stage A, same merge structure | Separately test the throughput/accuracy tradeoff |
| FMA enabled or changed division primitive | Only after the first two modes isolate the remaining numerical sensitivity |

Even the first mode is not a Torch bitwise-equivalence promise. Reduction trees, compiled library versions, division, and subnormal/flush-to-zero behavior can differ. If needed, isolate division with a correctly rounded primitive rather than changing several math options at once.

#### Parallel Stage B, with centered entropy throughout

For vocabulary tiles i, each Stage A state is `(m_i,a_i,b_i,argmax_i)`. Reduce the partial maxima in parallel, then reduce the two rescaled sums:

```text
M   = max_i m_i
w_i = exp(m_i - M)
A   = sum_i w_i * a_i
B   = sum_i w_i * (b_i + a_i*(m_i-M))
log_a = log(A)
entropy = log_a + (-B / A)
CE = -((z_target - M) - log_a)
```

This avoids a sequential loop of 75 state merges. In exact finite-input arithmetic `A>=1` and `B<=0`, so the entropy's final addition combines nonnegative terms. Padded vocabulary lanes and padded partial slots must be neutralized before products; zero times negative infinity is not safe. Keep nonfinite detection rather than silently converting invalid rows into plausible metrics. [31,42,44; derived design]

A max/index reduction can preserve the lowest vocabulary index on ties. With ordered contiguous tiles, Stage B can select the earliest winning tile and load only that tile's stored argmax. Relative to reading all four partial fields, this optional refinement saves `4*(75-1)=296` bytes per token—about 0.58 MiB at 2,048 tokens. Correct ties matter more than this small traffic difference. [42; calculated]

For T=2048 active tokens, `V=151936`, and 75 tiles, the entropy formulas entail about **622,329,856** exponential arguments for log-softmax followed by entropy exp, versus **311,318,528** for the centered two-stage reducer. These are mathematical evaluation counts, not GPU instruction counts or a predicted twofold speedup. L1 still keeps native CE work, and G0 still reconstructs supervised probabilities during backward. L2 can share forward normalizers across loss and diagnostics.

The expanded CPU audit executed **207 comparisons**: 23 cases × three vocabulary tile sizes × three arithmetic variants, using NumPy 2.3.5 and seed `20261030`. Twenty-two cases begin with FP16-origin logits; one additional case uses arbitrary FP32 values. All argmax results matched their references. The ordinary-exp comparison produced:

| Comparison with a NumPy FP32 version of TRL's formula | Maximum observed difference |
| --- | ---: |
| Entropy, FP16-origin cases | `2.86102294921875e-6` absolute |
| Entropy, all cases including arbitrary FP32 | `5.7220458984375e-6` absolute |
| Relative entropy, all cases | `0.08314263%` |
| Cross-entropy, all cases | `1.9073486328125e-6` absolute |

The alternative base-two calculation is a rounding proxy, not an implementation of the GPU's approximate instruction. NumPy exp, reduction trees, subnormals, and FMA proxies do not establish CUDA agreement.

A concentrated distribution illustrates the metric distinction: one logit at zero and the other 151,935 at −32 gives about `6.1572e-8` entropy in the tested FP32 baseline expression, versus `6.3496e-8` in FP64—a 3.125% token-level difference. The centered reducer matched the FP32 reference in this case. Combining that token with an ordinary token reduced the relative difference between their mean entropies to `5.3245e-8`. The actual harness compares **logged step aggregates**, then averages relative differences across the complete finite series with denominator floor `1e-12`; it does not average per-token relative errors or discard nonfinite steps. Use both concentrated stress cases and the real aggregation path. [6,17; executed CPU illustration]

### 12.5 Qualification model: remaining work, not inherited profile share

For this section, N means padded decoder tokens, T means non-padding head rows, and S means supervised prediction positions. Set `N=2048`, `D=1536`, `D_KV=256`, `V=151936`, vocabulary tiles `J=75`, and head chunk `C=64`. T and S vary by batch. This notation makes the decoder/head distinction explicit. [4,9]

The following model counts **major logical bytes still moved by the proposed kernels**. It excludes library GEMMs/dropout, small metadata, allocator effects, compaction gather/scatter, and other existing operations. R1 assumes gamma cache reuse; its loads are not a guaranteed DRAM count. The components are standalone R1, A1, K1, streamed metrics, and G0—not R2 or P2 variants whose work overlaps them.

| Component | Logical bytes per step | Example size | Original grid launches |
| --- | --- | ---: | ---: |
| R1 norm forward/backward | `57*8*N*D + 56*12*N*D` | 3,384 MiB | 113 |
| A1 q/v epilogues/backward conversion | `28*N*(D+D_KV)*(10+6)` | 1,568 MiB | 112 |
| K1 when the masked repeat path applies | `(56+55)*16*N*D_KV` | 888 MiB | 111 separate K/V; 56 paired |
| Two-stage metrics | `2*T*V + 32*T*J` | 598.1875 MiB at T=2048 | `2*ceil(T/C)` with active-row scheduling |
| G0 logit-gradient producer | `4*S*V` | 296.75 MiB at S=512 | `ceil(S/C)` |

For example, A1 forward reads base32 and adapter16 and writes output32: ten bytes per element. Its backward reads the FP32 gradient and writes the correctly scaled FP16 adapter gradient: six bytes. The identity base-gradient path need not add a copy. K1 forward reads compact FP32 and writes sixfold expanded FP16; backward reads expanded FP16 and writes compact FP32. First-layer K has no backward in the expected topology. These assumptions must match the eventual implementation.

In the ordinary uncached baseline, a full non-padding batch can omit its mask and take the native-GQA branch; K1's existing masked branch is then absent. A padded batch has fewer active metric rows but can add K1. Therefore adding maximum metric traffic and masked-path work without describing a batch is misleading. Nearly full padded batches can still approach both maxima. [29,30]

| Batch and scheduling, S=512 | Major logical traffic | Grids at C=64 |
| --- | ---: | ---: |
| T=2048, maskless, K1 absent | 5,846.9375 MiB | 297 |
| T=1536, masked, active rows compacted, separate K/V | 6,585.390625 MiB | 392 |
| Same padded batch, paired K/V grids | Same reference traffic | 337 |

At C=128, the corresponding grid counts are 261, 364, and 309. Chunk changes can alter GEMM selection, scratch, and rounding, so launch count alone cannot choose C. Scheduling padded N rows instead of compacted T rows would give 408 grids in the second case, assuming inactive programs skip logits and partial-state accesses. Do not claim the compacted launch count while retaining padded scheduling.

**Programs are not launches.** At T=2048 the first metric stage has 153,600 programs across the step, but C=64 uses 32 Stage A grids plus 32 Stage B grids. At T=1536 compact scheduling uses 115,200 Stage A programs and 24 grids per stage. Dense L1 can use two metric grids over its already available logits; the chunked counts describe L2. Separately implemented final aggregation would add launches. The watcher charges event bookkeeping per grid launch, not per program. [7; calculated]

For a limited bandwidth sanity check, divide those byte totals by an assumed rate. NVIDIA advertises 320+ GB/s GDDR6 bandwidth for T4; 320 is used as a rounded optimistic reference, while the other rates below are hypothetical effective rates. [38]

| S=512 scenario | At 320 GB/s | At 200 GB/s | At 100 GB/s |
| --- | ---: | ---: | ---: |
| Full, maskless | 19.16 ms | 30.65 ms | 61.31 ms |
| T=1536, masked | 21.58 ms | 34.53 ms | 69.05 ms |

These are conditional memory-service calculations, **not measured latency, speedup, or unconditional lower bounds**. Caches can eliminate DRAM transfers; reductions, exponentials, register spills, launch costs, and overlap are not represented. A measured 500 ms step would need 25 ms of eligible original event time for a 5% share, whereas a 250 ms step would need 12.5 ms. The actual repeat-level denominator and common qualifying names in section 11.5 control admission.

The organizer's 44% broad elementwise/cast category cannot be inherited as our original-kernel share. Fusion removes some of that work; retained library operations and the excluded starter SwiGLU do not become original credit. Optimize useful work, then measure the complete candidate. A theoretical traffic total is not a reason to add unnecessary operations or retain inefficient kernels. [2,3,6,7]

### 12.6 Concrete next experiments and evidence to collect

| Experiment | Isolate first | Evidence needed before combining |
| --- | --- | --- |
| R1 | Frozen FP32 gamma/input, source-staged arithmetic, one row per program | Forward/backward errors at actual scaler values; first norm no-grad; rstd and reduction primitive; time/traffic |
| P1 | Query-only rotary/cast with provided positional tensors | Exact cast position, forward/backward agreement, strides and effective SDPA backend |
| L1 reducer | Libdevice exp/log and parallel centered Stage B | Entropy/CE/argmax, nonfinite behavior, per-row rewind, step aggregation, cold cost |
| A1 and K1 | Native q/v dropout/GEMMs; original masked branch | Independent stochastic paths, six-group FP32 gradient reduction, first K no-grad |
| Combined L2 + M1 + G0 | Shared head state and original gradient producer with library GEMMs | Actual memory lifetime, command time, complete quality gates, original event-time share |
| R2, P2, approximate math | One change at a time after the preceding contracts pass | Additional benefit survives numerical, backend, cold-time, and integration checks |

Log N/T/S, actual `requires_grad` flags and dtypes, mask/backend branch, original entry-point names and launches, per-name event totals, command time, allocated/reserved/external peaks, and late compilations. Use the official repeated-run gate and full quality trajectory for the same candidate identity. This is the information that can distinguish a competitive implementation from an attractive list of fusions.

**Round-5 evidence boundary:** the source audits, NumPy arithmetic/finite differences, and byte/launch calculations above were completed. No Torch/CUDA runtime, original Triton compilation, T4 benchmark, measured memory saving, or competition admission result was available. The report supplies concrete implementation hypotheses and failure cases; it does not establish a rank or promise to beat every team.

## 13. Research round 6: baseline compatibility and integration that preserves the gains

**Main decision:** establish the actual baseline's final parameter and gradient dtypes before implementing the candidate. The new source audit finds a compatibility contradiction in the reference dependency combination, strengthens the compact-head integration contract, and identifies a useful query-layout variant on the math attention path. Several apparent copy savings disappear once actual storage order is considered.

### 13.1 The final adapter dtype changes after PEFT setup

Soup first constructs the quantized PEFT model. Later, TRL 0.29.1's SFTTrainer constructor converts every trainable Parameter to BF16 when `is_loaded_in_4bit` or `is_loaded_in_8bit` is true. The ordinary path satisfies this condition: the Transformers BNB quantizer sets the inner model's flag, and PEFT delegates that attribute through its wrappers. The conversion occurs before the base Trainer constructor. It is not gated by the task's BF16 autocast setting. [10,17,18,39,41,46]

| Object or boundary | Expected state after the complete setup |
| --- | --- |
| Trainable q/v LoRA Parameters | BF16 storage |
| Adapter GEMM operands/results under the fixed FP16 autocast | FP16 compute operands/results |
| Leaf adapter gradients | Returned to BF16 Parameter dtype |
| Frozen norm weights and head/embedding source | Existing FP32 storage; excluded from TRL's trainable-only conversion |

The 2,179,072 adapter values therefore occupy **4.15625 MiB in BF16**. The earlier 8.3125 MiB figure describes hypothetical FP32 storage, not the expected post-Trainer state. This correction does not halve the large frozen-head or decoder-activation estimates. It also does not authorize changing the Parameters' dtype in our solution.

#### A source-level compatibility contradiction to resolve first

The examined PyTorch 2.11 GradScaler groups gradients by device and dtype and sends BF16 gradients unchanged to CUDA unscale. Both CUDA unscale implementations—the foreach path and its TensorIterator fallback—use a dispatch macro that admits Double, Float, and Half, but **not BFloat16**. Accelerate's FP16 path creates a GradScaler and unscales before gradient clipping. Together with TRL's adapter conversion, this source chain predicts an unsupported BF16 unscale operation in that exact reference combination. [15,17,45]

**This is not an observed failure of the scoring image.** Torch/Triton are not explicitly pinned in the template, and their installed builds have not been executed here. Record `torch.__version__`, build information, final `is_loaded_in_4bit`, scaler state, and actual leaf gradient dtypes; run the unchanged baseline through its first unscale, clip, and optimizer update. If it fails, retain the exact traceback and resolve the baseline environment before comparing optimizations. Do not conceal the problem by upcasting adapters, disabling scaling/clipping, or changing protected settings in the candidate.

A successful baseline remains the reference even if its actual dtype chain differs from this source inference. All numerical comparisons must use that prepared state. Earlier inspection immediately after `get_peft_model` is insufficient because it precedes the TRL conversion.

### 13.2 Custom backward needs an explicit mixed-precision contract

#### A1 must preserve two rounding stages

The native forward scales the adapter's FP16 output before adding it to the FP32 base result. Its backward first crosses the add's FP32-to-FP16 boundary, then differentiates the FP16 scalar multiplication:

```text
Y32     = base32 + widen(RN16(2 * U16))
dbase32 = dY32
dU16    = RN16(2 * RN16(dY32))
```

Replacing the final line with `RN16(2*dY32)` can change gradients. An executed NumPy 2.3.5 example uses `dY32=2^-25`: native staged rounding produces zero, while one final cast produces `2^-24 = 5.960464477539063e-8`. This is a CPU IEEE-half counterexample, not a measured CUDA discrepancy. BF16 leaf Parameter storage occurs later in the gradient chain and does not erase this activation-gradient boundary. [18,25; executed check]

#### Preserve outer autocast; scope explicit math inside it

Paired `torch.amp.custom_fwd(device_type="cuda", cast_inputs=None)` and `custom_bwd(device_type="cuda")` can preserve the forward autocast state for a custom backward. The backward decorator relies on metadata installed by the forward decorator. A blanket `cast_inputs=torch.float16` recursively casts eligible floating inputs, potentially changing FP32 norm inputs, gamma, or head normalizers. Blanket FP32 casting can replace intended half GEMM operands. Select precision per operand and operation. [45]

The unchanged decoder must still run through the original PEFT/Accelerate autocast path. PEFT's BNB wrapper branches on whether autocast is enabled; disabling it casts X to the adapter weight dtype before dropout. With BF16 adapters that changes both activation/dropout dtype and subsequent computation. Disable autocast only around custom internals whose operands have already been explicitly prepared. [15,18]

| Candidate | Explicit contract to verify |
| --- | --- |
| L2/G0 | FP32 loss and actual incoming `grad_loss`; validated FP32 probability/scaling work; store dZ16; library GEMM with explicit dZ16/W16; retain its FP16 result boundary before widening dH. |
| M1 | Restore packed half-origin values to FP32 before the original pointwise replay; keep incoming/output gradients at the original boundaries. |
| R1/R2 | FP32 norm/residual math and gradients; R2 casts the combined attention-branch gradient only after accumulation. |
| P1 | Intentional FP16 output means its incoming gradient has crossed that boundary; widen before the rotary transpose. |
| A1 | Keep the FP32 base-gradient identity and both FP16 adapter-gradient rounding stages above. |

For the head's explicitly typed `dZ16 @ W16`, a narrowly scoped autocast-disabled region makes the operand contract clear. Keep the native GEMM precision settings and avoid passing the FP32 source weight to every backward chunk, which can recreate the frozen-weight casting problem. Triton itself does not infer PyTorch AMP casts for its pointer operands. [14,45]

#### Optimizer skips are part of the training trajectory

Accelerate continues to own scaling, unscale, clipping, optimizer stepping, and scaler updates. Custom backward consumes the provided upstream gradient; it must not multiply by another copy of the scaler or unscale a partial gradient early. [15,45]

When the native path detects overflow, it skips the optimizer and normal scheduler updates. The Trainer still clears gradients, increments `global_step`, and invokes the step-end callback. The batch, dropout RNG, metrics, and rewind have already advanced. Do not retry or erase that step. Release per-batch saved state normally. A frozen head representation remains value-valid; cached activations and trainable weight representations have different lifetimes and invalidation rules. [16,45]

Also test nonfinite ignored rows. Supervised-only direct head gradients are zero under the usual finite-logit assumptions. Native log-softmax backward can nevertheless evaluate `0 - exp(logp)*0`; a NaN log probability can propagate even from a row with zero upstream loss gradient. A shortcut that emits zeros can hide that event. Preserve the baseline's nonfinite/skip behavior or reject the unsupported case explicitly; do not use clamping or `nan_to_num` to manufacture agreement. [31,45; derived validation case]

### 13.3 A concrete compact-head integration boundary

#### Fresh output means a fresh mapping

In the pinned `ModelOutput`, assigning a field updates its dictionary only when the new value is non-None. Thus `outputs.logits = None` can leave `outputs["logits"]` holding the dense tensor. Accelerate traverses the mapping and can still create the full FP32 conversion. Construct a **new output object containing the loss and supported compact fields**, with no dense logits entry. Do not assume attribute inspection proves the tensor was removed. [15,46]

Saved logits, half head weights, and autograd state also belong outside returned metric/output mappings. The external FP32 converter can otherwise widen them again. Compact metric side channels are not automatically converted, so explicitly produce detached FP32 summaries and integer counts.

#### The trainer and model must agree before one forward starts

Use a per-model, one-call ticket armed only by the supported trainer override. Without the ticket, the inner causal LM delegates to its original forward—even if it is in training mode and has labels. Preflight the trainer/model configuration together; after an optimized forward, require exactly one summary matching the ticket. A missing or stale summary is an error, not a reason to silently skip metrics or run the decoder again.

The intended call path remains: Soup's outer RewindTrainer, our SFT subclass, the original `Trainer.compute_loss`, prepared outer PEFT/Accelerate model, then the inner causal-LM hook. The decoder invocation belongs inside that hook; calling the decoder directly from the trainer bypasses the intended wrapper boundary. [10,15–18]

Preserve both `model_accepts_loss_kwargs` and `_loss_shifts_labels`, along with the bound forward signature, model class, loss type, and config. The first controls denominator forwarding and accumulation normalization; the second controls shifted-label counting. A wrapper that merely accepts `**kwargs` can affect this introspection.

Delegate before model work when label smoothing, custom loss functions, virtual tokens, auxiliary router loss, context/sequence-parallel shifted labels, tuple outputs, nonstandard logits slicing, extra hidden/attention outputs, evaluation, or prediction are requested. In particular, the base Trainer removes labels for smoothing/custom-loss paths; a labels-only model guard would then disagree with a trainer that already bypassed SFT. Keep regression-test fake modules on their original path. [16,17,46]

Design pseudocode for the supported trainer branch:

```text
clear any previous compact step
if the complete call is unsupported:
    return original SFT behavior before starting model work
arm one per-model forward ticket
try:
    loss, outputs = original Trainer.compute_loss(..., return_outputs=True)
    summary = require_and_consume_exactly_one_summary(ticket)
    update metrics using pinned TRL aggregation
    publish detached token_loss for Soup's outer rewind recorder
    return loss or (loss, outputs), as requested
on exception:
    clear all transient state and propagate; never rerun forward
always:
    disarm the forward ticket
    retain successful token_loss until the outer recorder reads it
```

The row-loss side channel must outlive the inner trainer return. Clear it after outer recording or at the beginning of the next call, including fallback calls. Clearing it in the inner method's successful `finally` is too early. Soup can catch recorder exceptions and disable rewind without failing training, so successful process exit is insufficient evidence; verify complete row records and metric entries. [10]

#### Preserve each aggregation contract

| Quantity | Required semantics |
| --- | --- |
| Training CE | Supplied `num_items_in_batch` exactly once; it can describe the accumulation window. Otherwise retain native mean behavior. |
| Rewind token CE | Unnormalized detached FP32 `[B,T]`; position t predicts label t+1; ignored/final entries zero. |
| Empty supervised row | Rewind loss/count are zero; this does not justify clamping an entirely ignored training batch's denominator. |
| Entropy | All attention-mask positions, including prompts and the final valid position; retain TRL's gathered-local-quotient averaging. |
| Accuracy | Shifted supervised targets; gather numerator and denominator separately before division. |
| Logging | Cumulative token counter; inherited SFT logging averages its accumulated lists and clears them. |

Eighteen executed standard-library semantic mock checks cover stale mappings, fresh outputs, signatures, ticket scoping, exactly-once summaries, denominator/rewind separation, fallback, and failure cleanup. **They do not import or execute Transformers, Torch, CUDA, or autograd.** L1 still retains native dense CE and head backward; these interface checks do not turn it into L2 or establish a memory result.

### 13.4 Layout-aware attention: distinguish a view from a real copy

The source-derived shapes below use logical `[B,H,L,d]` with `B=4`, `L=512`, `H=12`, and `d=128`. Strides count elements, not bytes. The ordinary projection stores tokens contiguously, then views/transposes into heads. Native RoPE and dense dtype conversion preserve that physical order. [16,29,47]

| Tensor layout | Element strides in logical BHLD order |
| --- | --- |
| Native Q after RoPE/autocast, token-major storage | `[786432,128,1536,1]` |
| Proposed P1 head-major query output | `[786432,65536,128,1]` |
| Explicitly repeated K/V forward output | `[786432,65536,128,1]` |
| Efficient SDPA's ordinary public output | `[786432,128,1536,1]` |

#### Efficient output already matches the next consumer

The CUDA memory-efficient implementation allocates output contiguous in `[B,L,H,d]`, then returns its BHLD transpose. HF immediately transposes it back, so `.contiguous()` is a no-op for that expected path. Qwen's following flattening is also a view. **Do not count a 6 MiB output allocation or 12 MiB reorder traffic as removable work on this branch.** A `.contiguous()` call in Python does not establish that a copy executed. [29,47]

#### P1 layout variant: remove the math backend's query reshape clone

The native-GQA math path widens/scales Q, then batched matmul flattens B and H. For native token-major Q, `stride_B=786432` differs from `H*stride_H=1536`; these dimensions cannot merge as a view. The inspected matmul/reshape path therefore clones the FP32 query. If P1 writes the same FP16 rotary values directly in head-major order, the merge condition holds and that reshape becomes a view. [30,47]

At maximum shape, the candidate targets a **12 MiB FP32 clone allocation and 24 MiB logical read/write traffic per math layer**. Across 28 such layers, 672 MiB is cumulative traffic—not a prediction of simultaneous memory savings. This extends the original P1 proposal as a separately validated layout variant. It has no corresponding proven clone saving on the efficient branch.

Write the chosen forward layout directly; adding a separate `.contiguous()` merely moves the copy. Backward must read actual incoming strides, widen its FP16 gradient before rotary arithmetic, and directly emit the original token-major FP32 query gradient. Otherwise an upstream view/transpose backward can introduce another copy. Compare native-layout P1 and head-major P1 with actual kernel traces and backend checks.

#### K1's backward layout is not its forward layout

Efficient attention normally returns token-major dQ/dK/dV even if its forward K/V were head-major. K1 backward must therefore address its incoming gradient using strides, map each expanded head h to compact head `h//6`, widen each value to FP32, and reduce the six groups before writing the compact producer layout. Assuming contiguous BHLD gradients is incorrect for the inspected path. [47]

The expected 27 K and 28 V external backward reductions remain valid. However, efficient backward allocates/calculates all three internal gradients; the wrapper only passes the bias-gradient mask into its internal routine. Preserving first K's no-grad status does not remove the backend's first dK buffer or computation. It removes our downstream K cast/repeat/RoPE backward path. [41,47]

Q-only P1 also relies on FP16 autocast converting its Python-visible Q16/K32/V32 inputs before native SDPA's equal-dtype check. Outside that guarded autocast path, fall back. BF16 autocast would create a different rounding sequence. [14,29,30]

A later math-output experiment could fuse the internal FP32-to-FP16 cast with output layout conversion: 30 MiB of reference forward traffic becomes 18 MiB at maximum shape. A wrapper after public SDPA is too late to eliminate its internal cast. That larger integration change is lower priority and overlaps K2 if K2 removes the math branch.

Executed NumPy checks, seed `202610306`, verify the index mappings, repetition order, cast/reorder value identity, and the corresponding affine reshape/view relationships. They do not demonstrate actual CUDA copy kernels, backend selection, timing, or peak memory. The source predicts where to look in a trace.

### 13.5 Keep specialization and autotuning out of the critical path

Triton 3.6 computes its JIT cache key before canonicalizing the launch grid. Changing only the number of programs does not require a distinct compiled kernel. Keep token counts runtime where possible; reserve constexpr values for genuine algorithm choices and fixed dimensions. [48]

Ordinary unannotated runtime integers are not necessarily specialized by every value: the inspected native path distinguishes value one, divisibility by 16, and ordinary values within the same integer type. `do_not_specialize` can remove those value/alignment classes, but cannot override an explicit constexpr. Version 3.6's tuple handler also recurses with default specialization flags, so suppressing specialization on a tuple does not necessarily suppress its members; flat runtime arguments are easier to audit. [48]

A CPU model of this exact subset, using illustrative tail lengths 1–64, produces 64 key classes for constexpr, three for ordinary runtime integers, two when suppressing alignment alone, and one with flat `do_not_specialize`. A tuple with that flag still produces three under the examined recursion. This is key-class arithmetic, not executed Triton compilation. Dtypes, pointer alignment, target and compiler options can add variants, and less specialization can reduce optimization opportunities.

Live autotuning adds more than compilation. The default NVIDIA benchmarker in this release uses ordinary `do_bench`, allocates a **256 MiB flush buffer**, clears it around measurements, synchronizes, and repeats the kernel. Multiple configurations and keys repeat that process. The allocation can affect reserved/external memory, but it is not automatically 256 MiB added to the final peak; lifetimes and allocator reuse matter. Default tuning is not CUDA-graph tuning, although a separately requested graph benchmark would conflict with the challenge's tracing constraints. [3,37,48]

Autotuned kernels also execute repeatedly on the supplied buffers. Reset/restore hooks matter for accumulators and mutable state; a tuning function must not consume fresh dropout RNG or update a persistent metric repeatedly. The official watcher observes repeated JIT launches as well. These are tuning overheads, not useful workload improvements. Select a small configuration set in development and dispatch once per required operation in the scored path. This is a concrete extension of the cold-command budget in section 11.5.

### 13.6 Runtime preflight and evidence from this round

The first gates are now:

1. **Unchanged baseline:** actual package/build identifiers, final quantization/adapter/gradient metadata, scaler enabled state, and successful native unscale/clip/update or its exact failure.
2. **Supported integration path:** trainer/model agree before forward; every required metric and rewind record is produced; evaluation and unsupported calls retain their original outputs.
3. **Boundary comparison:** A1's two casts, R1/R2 FP32 gradients, P1 strides/casts, and G0's scaled FP16 GEMM operands match the prepared reference.
4. **Controlled overflow and corner cases:** preserve skip/backoff, scheduler/global-step behavior, RNG consumption, zero-supervision semantics, and nonfinite visibility.
5. **Combined fixed run:** measure complete cold command, actual memory timeline, original event share, and all official quality/regression gates.

#### Standalone CUDA dtype probe — supplied, not executed here

This development-only probe isolates GradScaler's BF16 unscale support from the model and from native BF16 matrix multiplication. It creates fresh disposable state for each dtype. It does not alter task Parameters/settings and must not be inserted into the scored training command. A successful probe is narrower evidence than a successful baseline update.

```python
import importlib.metadata
import json
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Run this diagnostic on the intended CUDA image")

try:
    triton_version = importlib.metadata.version("triton")
except importlib.metadata.PackageNotFoundError:
    triton_version = None

result = {
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "triton": triton_version,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "cases": [],
}
for dtype in (torch.float32, torch.bfloat16):
    p = torch.nn.Parameter(torch.ones(1, device="cuda", dtype=dtype))
    optimizer = torch.optim.SGD([p], lr=0.0)
    scaler = torch.amp.GradScaler("cuda")
    case = {"parameter_dtype": str(dtype), "scaler_enabled": scaler.is_enabled()}
    try:
        # FP32 math with a finite scaled gradient; no BF16 GEMM is required.
        loss = (p.float().square() * 1e-3).sum()
        scaler.scale(loss).backward()
        case["gradient_dtype"] = str(p.grad.dtype)
        scaler.unscale_(optimizer)
        torch.cuda.synchronize()
        case["unscale"] = "passed"
    except Exception as exc:
        case["unscale"] = "failed"
        case["error"] = f"{type(exc).__name__}: {exc}"
    result["cases"].append(case)
print(json.dumps(result, indent=2))
```

Do not skip the storage/unscale probe merely because a GPU lacks native BF16 matrix-multiplication support; those are distinct questions. Record the real model's final dtypes separately.

**Executed evidence:** exact source audits, 18 stdlib semantic mock checks, NumPy A1 rounding and attention index/layout checks, and the specialization-key calculation. **Unexecuted:** the CUDA probe, real dependency integration/autograd, original Triton compilation, T4 performance, memory peaks, and official admission. The source-level baseline contradiction takes priority over another speculative speed claim.

## Appendix A. Portable CPU algebra checks from round 2

Save the Python block below as `check_research_math.py` and run it with NumPy installed:

```bash
OPENBLAS_NUM_THREADS=1 python check_research_math.py
```

It prints the detailed results and writes `research_math_checks.json` to the working directory. The recorded run used NumPy 2.3.5 and seed 20261030. The code is original verification work from this research, with no Torch, Triton, or CUDA dependency. Its scope and limitations are stated in section 8.

```python
"""CPU-only algebra checks. These are not GPU or competition benchmarks."""
from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(20261030)


def stats(z):
    m = z.max(axis=1)
    d = z - m[:, None]
    e = np.exp(d)
    return m, e.sum(axis=1), (e * d).sum(axis=1)


def merge(left, right):
    m1, a1, b1 = left
    m2, a2, b2 = right
    m = np.maximum(m1, m2)
    w1, w2 = np.exp(m1 - m), np.exp(m2 - m)
    a = w1 * a1 + w2 * a2
    b = w1 * (b1 + a1 * (m1 - m)) + w2 * (b2 + a2 * (m2 - m))
    return m, a, b


def finish(s):
    m, a, b = s
    return m + np.log(a), np.log(a) - b / a


stats_results = []
for dtype in (np.float64, np.float32):
    for name, z in [
        ('irregular_vocab', rng.normal(0, 4, (11, 1031))),
        ('large_common_offset', 10000 + rng.normal(0, 8, (7, 1031))),
        ('actual_vocab_width', rng.normal(0, 3, (4, 151936))),
        ('ties', np.zeros((5, 1031))),
        ('extreme_peak', np.tile(np.r_[0.0, np.full(1030, -1000.0)], (5, 1))),
    ]:
        z = z.astype(dtype)
        ref_lse, ref_h = finish(stats(z))
        tile = 4096 if z.shape[1] > 4096 else 127
        chunks = [stats(z[:, i:i+tile]) for i in range(0, z.shape[1], tile)]
        state = chunks[0]
        for chunk in chunks[1:]:
            state = merge(state, chunk)
        lse, ent = finish(state)
        reverse = chunks[-1]
        for chunk in reversed(chunks[:-1]):
            reverse = merge(reverse, chunk)
        rev_lse, rev_h = finish(reverse)
        lse_error = float(np.max(np.abs(lse - ref_lse)))
        entropy_error = float(np.max(np.abs(ent - ref_h)))
        order_error = float(max(np.max(np.abs(lse-rev_lse)), np.max(np.abs(ent-rev_h))))
        tol = 1e-10 if dtype is np.float64 else 0.002
        assert max(lse_error, entropy_error, order_error) < tol
        stats_results.append(dict(dtype=np.dtype(dtype).name,case=name,rows=z.shape[0],
                                  vocab=z.shape[1],tile=tile,lse_max_abs=lse_error,
                                  entropy_max_abs=entropy_error,merge_order_max_abs=order_error))

# The head sees all rows for entropy. Direct CE gradients only need labelled rows.
B, S, D, V = 3, 7, 5, 13
x = rng.normal(size=(B,S,D)) * 0.3
w = rng.normal(size=(V,D)) * 0.2
labels = np.full((B,S),-100,dtype=np.int64)
labels[0,4:] = rng.integers(0,V,size=3)
labels[2,4:] = rng.integers(0,V,size=3)
targets = np.concatenate([labels[:,1:],np.full((B,1),-100)],axis=1).reshape(-1)
valid = targets != -100
denom = int(valid.sum())


def causal_hidden(v):
    return np.cumsum(v,axis=1)/np.sqrt(S)


def head_loss(v):
    h = causal_hidden(v).reshape(-1,D)
    z = h @ w.T
    lse,_ = finish(stats(z))
    rows = np.flatnonzero(valid)
    return float((lse[rows]-z[rows,targets[rows]]).sum()/denom)


h = causal_hidden(x).reshape(-1,D)
z = h @ w.T
lse,_ = finish(stats(z))
dz = np.exp(z-lse[:,None])
rows = np.flatnonzero(valid)
dz[rows,targets[rows]] -= 1
dz[~valid] = 0
dz /= denom
dh_dense = dz @ w

z_selected = h[valid] @ w.T
lse_selected,_ = finish(stats(z_selected))
dz_selected = np.exp(z_selected-lse_selected[:,None])
dz_selected[np.arange(denom),targets[valid]] -= 1
dh_selected = np.zeros_like(h)
dh_selected[valid] = (dz_selected/denom) @ w
head_error = float(np.max(np.abs(dh_selected-dh_dense)))
assert head_error < 1e-12

dh = dh_selected.reshape(B,S,D)
dx = np.cumsum(dh[:,::-1],axis=1)[:,::-1]/np.sqrt(S)
eps = 1e-6
finite_difference_errors = []
for idx in [(0,0,0),(0,2,3),(0,5,2),(1,0,1),(2,1,4),(2,6,2)]:
    plus, minus = x.copy(),x.copy()
    plus[idx] += eps
    minus[idx] -= eps
    numeric = (head_loss(plus)-head_loss(minus))/(2*eps)
    finite_difference_errors.append(abs(numeric-dx[idx]))
assert max(finite_difference_errors) < 1e-8
assert abs(dx[0,0,0]) > 1e-8

# LoRA algebra with separate, fixed q/v dropout masks and both input-gradient paths.
N,I,R = 6,5,3
lx = rng.normal(size=(N,I)) * 0.2
blocks=[]
for O in (6,2):
    mask=(rng.random((N,I))>=0.05).astype(np.float64)
    blocks.append(dict(W=rng.normal(size=(O,I))*.2,A=rng.normal(size=(R,I))*.2,
                       B=rng.normal(size=(O,R))*.2,G=rng.normal(size=(N,O)),mask=mask))
blocks[0]['mask'][0,0]=0
blocks[1]['mask'][0,0]=1
scale=2.0
keep_prob=0.95


def objective(xx):
    total=0.0
    for block in blocks:
        xd=xx*block['mask']/keep_prob
        low=xd@block['A'].T
        y=xx@block['W'].T + scale*low@block['B'].T
        total += float(np.sum(y*block['G']))
    return total


dlx=np.zeros_like(lx)
analytical=[]
for block in blocks:
    xd=lx*block['mask']/keep_prob
    low=xd@block['A'].T
    dlow=scale*block['G']@block['B']
    dlx += block['G']@block['W'] + (dlow@block['A'])*block['mask']/keep_prob
    analytical.append(dict(A=dlow.T@xd,B=scale*block['G'].T@low))
lora_errors=[]
for idx in [(0,0),(1,2),(5,4)]:
    plus,minus=lx.copy(),lx.copy()
    plus[idx]+=eps
    minus[idx]-=eps
    lora_errors.append(abs((objective(plus)-objective(minus))/(2*eps)-dlx[idx]))
for bidx,block in enumerate(blocks):
    for key in ('A','B'):
        for idx in [(0,0),(block[key].shape[0]-1,block[key].shape[1]-1)]:
            old=float(block[key][idx])
            block[key][idx]=old+eps
            plus=objective(lx)
            block[key][idx]=old-eps
            minus=objective(lx)
            block[key][idx]=old
            lora_errors.append(abs((plus-minus)/(2*eps)-analytical[bidx][key][idx]))
assert max(lora_errors)<1e-8

# Scalar rounding demonstration, not evidence of underflow in this workload.
g=np.array([1e-9,1e-8,1e-7],dtype=np.float32)
amp_scale=np.float32(65536)
round_early=g.astype(np.float16).astype(np.float32)
scale_first=(g*amp_scale).astype(np.float16).astype(np.float32)/amp_scale

result={
    'kind':'CPU NumPy algebra checks, not a GPU/training benchmark',
    'seed':20261030,
    'numpy_version':np.__version__,
    'streaming_stats_cases':stats_results,
    'supervised_head_backward':{'positions':B*S,'supervised':denom,
         'dense_vs_selected_max_abs':head_error,
         'causal_decoder_finite_difference_max_abs':max(finite_difference_errors),
         'prompt_input_gradient_example':float(dx[0,0,0]),
         'note':'Toy causal cumulative-sum decoder; does not validate Transformer, CUDA, AMP or harness.'},
    'lora_fixed_independent_masks':{'finite_difference_checks':len(lora_errors),
         'max_abs_error':max(lora_errors),'rank':R,'note':'Float64 toy shapes; no GPU/dtype-performance claim.'},
    'fp16_rounding_demo':{'unscaled':g.tolist(),'round_before_scale':round_early.tolist(),
         'scale_before_round_then_unscale':scale_first.tolist()},
}
out=Path('research_math_checks.json')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
```


## Appendix B. Portable CPU checks from round 3

These two independent scripts require NumPy. Save each block using the indicated filename and run it in a writable directory. They print and save their detailed JSON results. They do not import PyTorch or execute CUDA; the exact scope is described in section 10.8.

### B1. Saved-state representation and attention cast/repeat

Save as `check_storage_and_repeat.py` and run:

```bash
OPENBLAS_NUM_THREADS=1 python check_storage_and_repeat.py
```

```python
"""CPU representation/algebra checks; not a CUDA or training benchmark."""
from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(20261030)

# Exhaustively cover every finite IEEE binary16 encoding, including signed zero.
all_bits = np.arange(65536, dtype=np.uint16)
all_half = all_bits.view(np.float16)
finite = np.isfinite(all_half)
half = all_half[finite]
widened = half.astype(np.float32)
packed = widened.astype(np.float16)
restored = packed.astype(np.float32)
half_bits_equal = np.array_equal(packed.view(np.uint16), all_bits[finite])
float_bits_equal = np.array_equal(restored.view(np.uint32), widened.view(np.uint32))
assert half_bits_equal and float_bits_equal
assert int(finite.sum()) == 63488

# Keep arithmetic in FP32; only represent proven half-origin saved inputs compactly.
gate = rng.normal(size=(128, 64)).astype(np.float16).astype(np.float32)
up = rng.normal(size=gate.shape).astype(np.float16).astype(np.float32)
grad = rng.normal(size=gate.shape).astype(np.float32)

def native_style_pointwise(g, u, dy):
    one = np.float32(1)
    e = np.exp(-g)
    activated = g / (one + e)
    y = activated * u
    dactivated = dy * u
    sigmoid = one / (one + e)
    dg = dactivated * sigmoid * (one + g * (one - sigmoid))
    du = dy * activated
    return y, dg, du

reference = native_style_pointwise(gate, up, grad)
candidate = native_style_pointwise(gate.astype(np.float16).astype(np.float32),
                                  up.astype(np.float16).astype(np.float32), grad)
pointwise_bits = [np.array_equal(a.view(np.uint32), b.view(np.uint32))
                  for a, b in zip(reference, candidate)]
assert all(pointwise_bits)

# These counterexamples prevent overgeneralizing the representation proof.
arbitrary = np.array([np.nextafter(np.float32(1), np.float32(2)),
                      np.float32(1) + np.float32(2**-12)], dtype=np.float32)
arbitrary_roundtrip = arbitrary.astype(np.float16).astype(np.float32)
assert np.any(arbitrary != arbitrary_roundtrip)
g = np.array([1.0, 2.0, -1.0], dtype=np.float32)
silu = g / (np.float32(1) + np.exp(-g))
silu_roundtrip = silu.astype(np.float16).astype(np.float32)
assert np.any(silu != silu_roundtrip)

# Cast and repeat commute in forward; moving a gradient reduction into FP16 does not.
k32 = rng.normal(size=(2, 2, 3, 4)).astype(np.float32)
repeat_then_cast = np.repeat(k32, 6, axis=1).astype(np.float16)
cast_then_repeat = np.repeat(k32.astype(np.float16), 6, axis=1)
assert np.array_equal(repeat_then_cast.view(np.uint16), cast_then_repeat.view(np.uint16))
rep_grad16 = np.array([1, 2**-12, 0, 0, 0, 0], dtype=np.float16)
baseline_grad32 = rep_grad16.astype(np.float32).sum(dtype=np.float32)
naive_grad32 = rep_grad16.sum(dtype=np.float16).astype(np.float32)
assert baseline_grad32 != naive_grad32

M, D, I, layers = 2048, 1536, 8960, 28
mib = 2**20
sizes = {
    'one_mlp_fp32_mib': M*I*4/mib,
    'one_mlp_fp16_mib': M*I*2/mib,
    'baseline_mlp_saved_state_mib': 3*M*I*4*layers/mib,
    'starter_mlp_saved_state_mib': 2*M*I*4*layers/mib,
    'packed_gate_up_saved_state_mib': 2*M*I*2*layers/mib,
    'one_projection_recompute_state_mib': (M*D*2 + M*I*2)*layers/mib,
    'both_projection_recompute_state_mib': M*D*2*layers/mib,
    'two_projection_recompute_flop_per_max_step': 4*M*D*I*layers,
    'expanded_kv_fp32_mib': 2*4*12*512*128*4/mib,
    'expanded_kv_fp16_mib': 2*4*12*512*128*2/mib,
}

result = {
    'scope': 'CPU NumPy representation and algebra checks only; no autograd/CUDA/runtime proof',
    'numpy_version': np.__version__,
    'seed': 20261030,
    'exhaustive_binary16': {
        'finite_encodings': int(finite.sum()),
        'packed_half_bits_identical': half_bits_equal,
        'restored_fp32_bits_identical': float_bits_equal,
        'includes_signed_zeros': True,
        'nan_payloads_and_cuda_subnormal_behavior': 'not verified',
    },
    'packed_saved_state_pointwise': {
        'shape': list(gate.shape),
        'forward_dgate_dup_bitwise_equal_in_numpy': pointwise_bits,
    },
    'arbitrary_fp32_counterexample': {
        'original': arbitrary.tolist(), 'roundtrip': arbitrary_roundtrip.tolist(),
    },
    'silu_not_half_representable': {
        'original': silu.tolist(), 'roundtrip': silu_roundtrip.tolist(),
    },
    'cast_repeat': {
        'forward_bits_identical': True,
        'baseline_fp32_reduction': float(baseline_grad32),
        'naive_fp16_reduction_then_widen': float(naive_grad32),
        'difference': float(baseline_grad32-naive_grad32),
    },
    'calculated_storage_and_flops': sizes,
}
out = Path(__file__).with_name('storage_and_repeat_results.json')
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
```

### B2. Centered normalizers, tiled statistics, and head cost calculations

Save as `numerical_audit.py` and run:

```bash
OPENBLAS_NUM_THREADS=1 python numerical_audit.py
```

```python
"""NumPy algebra/rounding checks; not a PyTorch or GPU kernel benchmark."""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).parent
rng = np.random.default_rng(864213)
V, D = 151936, 1536


def split_stats(z, block):
    parts = []
    for start in range(0, z.size, block):
        q = z[start:start+block]
        m = q.max()
        centered = q-m
        e = np.exp(centered)
        parts.append((m, e.sum(dtype=np.float32),
                      (e*centered).sum(dtype=np.float32),
                      start+int(np.argmax(q))))
    m = np.float32(max(x[0] for x in parts))
    pm = np.array([x[0] for x in parts], np.float32)
    pa = np.array([x[1] for x in parts], np.float32)
    pc = np.array([x[2] for x in parts], np.float32)
    w = np.exp(pm-m)
    a = (w*pa).sum(dtype=np.float32)
    c = (w*(pc+(pm-m)*pa)).sum(dtype=np.float32)
    index = min(x[3] for x in parts if x[0] == m)
    return m, np.log(a), np.log(a)-c/a, index


def audit(name, raw):
    # The reference consumes exactly these FP16-rounded values in FP32.
    z = raw.astype(np.float16).astype(np.float32)
    assert np.isfinite(z).all()
    zd = z.astype(np.float64)
    ref_m = zd.max()
    ref_la = np.log(np.exp(zd-ref_m).sum())
    ref_lp = (zd-ref_m)-ref_la
    ref_p = np.exp(ref_lp)
    ref_entropy = -(ref_p*ref_lp).sum()
    target = int(rng.integers(0, z.size))
    m = z.max()
    la = np.log(np.exp(z-m).sum(dtype=np.float32))
    split_lp = (z-m)-la
    collapsed_lp = z-np.float32(m+la)
    split_p, collapsed_p = np.exp(split_lp), np.exp(collapsed_lp)
    alpha = np.float32(np.float32(65536)/np.float32(511))
    onehot = np.zeros_like(z); onehot[target] = 1
    split_grad = (split_p-onehot)*alpha
    collapsed_grad = (collapsed_p-onehot)*alpha
    tile = []
    for block in (1024, 2048, 4096):
        tm, tla, ent, idx = split_stats(z, block)
        ce = -((z[target]-tm)-tla)
        tile.append({"block":block,"entropy_error":float(abs(ent-ref_entropy)),
                     "ce_error":float(abs(ce+ref_lp[target])),
                     "argmax_matches":idx==int(np.argmax(z))})
    return {"name":name,"vocab":z.size,"target":target,
            "split_probability_sum":float(split_p.sum(dtype=np.float64)),
            "collapsed_probability_sum":float(collapsed_p.sum(dtype=np.float64)),
            "split_ce_abs_error":float(abs(-split_lp[target]+ref_lp[target])),
            "collapsed_ce_abs_error":float(abs(-collapsed_lp[target]+ref_lp[target])),
            "split_probability_max_abs_error":float(np.max(np.abs(split_p-ref_p))),
            "collapsed_probability_max_abs_error":float(np.max(np.abs(collapsed_p-ref_p))),
            "collapsed_vs_split_scaled_gradient_max_abs":float(np.max(np.abs(collapsed_grad-split_grad))),
            "tiles":tile}


cases = [
    ("two_equal_60000", np.full(2,60000.0,np.float32)),
    ("large_vocab_equal_60000",np.full(V,60000.0,np.float32)),
    ("large_vocab_offset10000",np.float32(10000)+rng.normal(0,8,V).astype(np.float32)),
    ("large_vocab_offset60000",np.float32(60000)+rng.normal(0,24,V).astype(np.float32)),
    ("large_vocab_no_offset",rng.normal(0,3,V).astype(np.float32)),
    ("irregular_ties",np.array([0,1,3,3,-4,0,3]*37,np.float32)),
]
results = [audit(name,z) for name,z in cases]
assert all(t["argmax_matches"] for r in results for t in r["tiles"])
assert max(t["entropy_error"] for r in results for t in r["tiles"]) < 2e-5
assert max(t["ce_error"] for r in results for t in r["tiles"]) < 2e-5

# These are ordinary separate NumPy FP32 operations, not CUDA exp/FMA results.
alpha=np.float32(np.float32(65536)/np.float32(511))
p=np.concatenate([np.linspace(0,1,10001,dtype=np.float32),
                  np.float32(1)-np.exp2(-np.arange(1,25,dtype=np.float32))])
via_subtract_then_multiply=(p-np.float32(1))*alpha
via_sparse_vjp=(-alpha)-p*(-alpha)
rounding={"alpha":float(alpha),"separate_fp32_target_form_max_abs_difference":float(np.max(np.abs(via_subtract_then_multiply-via_sparse_vjp))),
          "fp16_store_disagreements":int(np.count_nonzero(via_subtract_then_multiply.astype(np.float16)!=via_sparse_vjp.astype(np.float16))),
          "note":"Illustrates operation-order sensitivity; does not model CUDA FMA contraction or exp."}
tables=[]
for ratio in (.1,.25,.5,1.0):
    tables.append({"S_over_T":ratio,"saved_vs_dense_head_flops":(1+ratio)/2,
                   "recompute_vs_dense_head_flops":(1+2*ratio)/2,
                   "recompute_vs_saved_head_flops":(1+2*ratio)/(1+ratio)})
layout=[]
for block in (1024,2048,4096):
    k=(V+block-1)//block
    layout.append({"vocab_tile":block,"tiles_per_token":k,"partial_bytes_per_token":16*k,
                   "partial_KiB_C64":64*16*k/1024,"partial_KiB_C128":128*16*k/1024})
report={"scope":"NumPy-only arithmetic checks; no GPU timing, PyTorch execution or T4 correctness claim",
        "numpy_version":np.__version__,"cases":results,"rounding_order":rounding,
        "flop_ratios":tables,"flops_per_head_token_gemm":2*D*V,
        "example_P2048_T2048_S512":{"baseline":4*2048*D*V,
                                    "save_logits":2*(2048+512)*D*V,
                                    "recompute_logits":2*(2048+2*512)*D*V},
        "partial_layout":layout}
(OUT/'numerical_audit_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
```
