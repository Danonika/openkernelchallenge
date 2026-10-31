# Qwen-Inspired Kernel Optimization for the Soup Kernel Challenge

Research date: 30 October 2026 — updated after research round 2  
Target: Soup 0.77.0, Qwen2.5-1.5B LoRA training, one Kaggle T4  
Inspected starter commit: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`

## Session handoff

**Goal:** Improve full-command training time and peak GPU memory for the fixed Soup Kernel Challenge on one Kaggle T4.

**Status:** Rules, starter, and exact training-dependency sources inspected. CPU algebra checks passed for streamed statistics, supervised-only terminal-head gradients, and LoRA gradients with independent fixed masks. No candidate Triton kernels implemented; no GPU or PyTorch runtime is available in this research environment. No T4 speed, peak-memory, or competition-quality result is claimed.

**Next:** On the actual T4, record runtime versions, tensor dtypes, padded lengths, supervised-token counts, and allocation/cast traces. First isolate a shared diagnostics reducer with the baseline loss/backward retained. Then build the chunked head with an explicitly reused FP16 compute weight and supervised-only backward. Develop original RMSNorm and a conservative LoRA epilogue in parallel; use the official harness for selection.

**Most consequential round-2 findings:**

- Metrics consume FP32-converted values of FP16-rounded head logits. Match both stages.
- PyTorch 2.11 autocast does not cache casts of frozen weights. A naive head-chunk loop can repeatedly cast the entire head; one FP16 representation is 445.125 MiB.
- Only supervised prediction positions need the terminal head's backward matmul. Prompt positions still need forward logits for entropy, and prompt decoder gradients must remain intact.
- bitsandbytes 0.50.2 already has a T4 fused NF4 forward family, but its source dispatch falls back for ordinary longer padded training batches. Its input-gradient backward always dequantizes before matmul when that gradient is needed.
- Dense NF4 projection weights are temporary, with a largest individual FP16 matrix of 26.25 MiB. Caching all of them would retain 2,499 MiB; reconstructed FP32 block scales total about 78.1 MiB.

These are source findings and calculations, with the proposed optimizations still subject to runtime verification. Sections 3–6 contain the implementation decisions; section 8 and Appendix A preserve the CPU-check evidence.

**Preserve:** Shifted supervised labels, normalization, FP16 AMP scaling, prompt-inclusive entropy, accuracy, rewind values, and independent q/v dropout at 0.05. Keep all fixed training settings and batches unchanged. Original qualifying Triton kernels must meet at least 5% runtime share; rewritten starter SwiGLU does not qualify. All official correctness and regression gates remain mandatory.

## Executive overview


> **Recommended first build**
> Begin with one shared diagnostics pass, then a streamed output head with explicit weight-cast reuse and supervised-only backward. Develop original normalization and LoRA epilogue kernels alongside it. Select further work from the measured full-command bottleneck.

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
| **2. Original RMSNorm / residual fusion** | Reduce reads, writes, casts and launches. | Preserve FP32 reduction and input/output dtype behavior; validate gradients. |
| **3. LoRA epilogue / backward fusion** | Reduce rank-16 temporaries and scale/add/cast operations. | Preserve dropout, q/v mask independence, A/B gradients and adapter layout. |
| **4. Selective recomputation** | Exchange inexpensive repeated compute for lower activation memory. | Keep settings unchanged; replay stochastic operations; remain inside time gates. |
| **5. NF4 input-gradient fusion / scale reuse** | Target the unfused backward and actual forward fallback; reuse small frozen scale metadata if useful. | Preserve double quantization and dtype boundaries; beat the real dispatched baseline. |
| **6. Within-batch unpadding** | Avoid padding work without changing the batch. | Preserve sequences, positions, masks, row identities, token counts and dropout. |

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

| Representation lifetime to test | Expected tradeoff, not a measured result |
| --- | --- |
| Once per head-forward phase; recast once for head backward | Reuses every chunk within a phase and permits an earlier release, at the cost of an additional full cast. |
| Create at head forward; retain until head backward completes | One cast supplies both phases, but remains live between them. |
| Retain across training steps | Avoids more casts while keeping 445.125 MiB live during decoder work; test only if the memory tradeoff is favorable. |

This representation must remain separate from the source `Parameter`, preserve tying and serialization, and be recreated within each measured run. The runtime tensor is unrelated to precompiled kernel caches. Its peak-memory impact depends on lifetime overlap; the full 445.125 MiB is not automatically an incremental peak because the baseline also needs a compute weight for head work.

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

For supervised rows, reconstruct probabilities from saved or recomputed head logits and form the logit gradient using the upstream gradient scale. Apply that scale before the relevant low-precision rounding:

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

> **Easy-to-miss constraint:** prompt tokens can be excluded from supervised CE, but their vocabulary distributions still contribute to entropy. Skipping all prompt-token head evaluation would change a required output. [10]

### Concrete integration plan without enabling a built-in flag

The fixed training path has no identified mandatory full-logits consumer beyond loss, entropy, accuracy, and rewind. This follows from the inspected call chain; runtime and regression tests still need to verify the proposed wrappers. The task's `val_split` is zero, and ordinary prediction/evaluation calls should keep their original behavior. [4,6,10,16,17]

Install narrowly guarded original wrappers through `solution.apply()`:

1. **Model boundary:** after Soup's `SFTTrainerWrapper._setup_transformers` creates the final PEFT model, obtain its actual inner causal LM and wrap that instance's forward. Continue invoking it through the outer PEFT/Accelerate model. Restrict the new path to the supported model, training mode, and labelled calls; preserve the forwarded signature and loss-kwargs detection.
2. **L1 forward:** invoke the original causal-LM forward, reduce diagnostics from its internal logits, then construct a fresh output with the original loss and applicable fields but no logits. **L2 forward:** invoke the unchanged decoder and our custom chunked-head/autograd implementation, returning loss and compact state.
3. **Trainer boundary:** supply an original subclass of the base SFT trainer through Soup's `_build_rewind_trainer`. Soup can then retain its original outer `RewindTrainer`, sampler, row fingerprints, recorder, callbacks, and serialization.
4. **Loss/metric handling:** on the supported path, call `transformers.Trainer.compute_loss` with the original `num_items_in_batch` and `return_outputs=True`. Reproduce TRL's existing entropy, accuracy, and token-counter updates from the new summaries, including gathering and averaging semantics. Remove `_prediction_loss_only` and set `use_cache=False` as the pinned path does. Return loss or `(loss, outputs)` as requested.
5. **Rewind interface:** provide an original compact object at `trainer._soup_chunked_step` containing `token_loss`. Soup's existing outer wrapper already uses this interface when returned logits are absent. This reuses a data contract, without enabling `stream_loss_chunk`, `use_cut_ce`, or other protected optimization flags. [10]

The compact `token_loss` is FP32 `[B,T]`. Entry `t` holds CE for `labels[:,t+1]`; ignored targets and the final entry are zero. Training loss uses the global provided denominator. Row means belong only to rewind, which retains its own per-row counts. Never introduce a second normalization or an average of row means into training.

Clear stale summary state before each forward and consume each new summary once. Keep compact row-loss state alive until the outer rewind wrapper has read it; clear it afterward through a small outer cleanup wrapper, or before the next guarded forward. Unsupported calls, including evaluation, unsupported adapter variants, and regression-test fake modules, delegate to the original method before the optimized path executes. If an optimized forward raises, clear transient summary state and propagate the exception. Do not silently rerun a forward after dropout or other stateful work has already executed.

## 5. Fusion and later experiments

Start with contained changes; isolate the higher-risk branches

### Original RMSNorm and residual fusion

Implement both forward and backward, preserving FP32 norm reduction and baseline input/output types. Fuse adjacent residual or cast operations only where dependencies and rounding permit. Treat broader precision changes as a separate experiment so numerical deviations remain attributable.

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
| All trainable adapter parameters | **2,179,072**, or **8.3125 MiB** in FP32 |

The rank intermediate is 96 times smaller than the dropped input. Preserve it initially and investigate fusion of mask/rescale into the adapter input-gradient epilogue, followed by shared loading of the upstream gradient for `dH` and `dB`. Those two gradients reduce over different dimensions; their combined tiling must earn its cost. Tune q and v separately because their output widths differ sixfold. These are proposals derived from the fixed model geometry. [4,9,18]

Use the original native dropout calls, input shape/dtype, and call order. A matching seed with a different random-number kernel does not establish the same masks. Soup's grouped QKV helper supports its concatenated path only with zero dropout; the fixed 0.05 case cannot share a single dropped input across q/v. Current external implementations vary, so a blanket claim that all fast-LoRA systems require zero dropout would also be inaccurate. [10,24]

**Optional, unverified memory experiment:** with both native masks preserved, a shared `FP16(X * c)` representation could reconstruct the two dropped-half inputs, where `c` exactly matches native dropout's FP32 scaling. A theoretical upper saving is one 6 MiB input per layer, or 168 MiB across 28 layers at maximum length. Check signed zeros, nonfinite values, scaling precision, actual saved tensors, reconstruction cost, and gradient rounding. Saving an extra 12 MiB FP32 input merely to replace two 6 MiB half inputs provides no raw byte saving. This experiment follows the safe epilogue in priority.

### Selective activation recomputation

Identify large saved intermediates that are inexpensive to recreate. Recompute them inside original autograd code, retaining masks or the correct RNG state when replaying stochastic operations. Custom recomputation and offloading are permitted, while changing the built-in configuration switches is prohibited. [2,3]

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

An original fused `dX = dY @ dequant(W)` kernel could consume packed tiles directly. It should decode the original NF4 codebook and double-quantized scales, preserve FP16 operand rounding, and use appropriate matmul accumulation. No frozen base-weight gradient is needed. This targets a concrete source-level gap, but larger cuBLAS GEMMs may still outperform a fused implementation.

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

Develop L1 diagnostics, the streamed-head branch, and original normalization/LoRA epilogues in parallel. Keep one integrated candidate passing all gates. Select recomputation, NF4, or padding work based on measured potential and time left for validation.

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

### Round-2 experiment matrix

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

The original-kernel share uses CUDA-event time divided by wall time of training steps from step 3. The lowest share across candidate runs must pass. Cover several real bottlenecks and leave headroom for hidden rows; do not add useless work to increase the share. [6,7]

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

### Decisions carried into the next session

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
[github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd.../README.md](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md)

**[4]** Fixed task configuration. Model, training parameters and batch/sequence settings.
[github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd.../task/task.yaml](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/task/task.yaml)

**[5]** Measurement harness. Cold caches, warm-up, run timing and summary aggregation.
[github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd.../harness/run_task.py](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/run_task.py)

**[6]** Admission checks. Quality, trajectory, outputs, original-kernel share and regression limits.
[github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd.../harness/check_rules.py](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py)

**[7]** Training instrumentation. Per-launch CUDA events and step-time measurement.
[github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd.../harness/_train.py](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/_train.py)

**[8]** Scoring implementation. Ratios, improvement terms and noise threshold.
[github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd.../harness/score.py](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/score.py)

**[9]** Qwen2.5-1.5B-Instruct model configuration. Vocabulary, dimensions and tied embeddings.
[huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)

**[10]** Soup CLI 0.77.0 release. Pinned wheel inspected as source: trainer/sft.py, trainer/rewind_hf.py, utils/stream_loss_chunk.py, and utils/fast_lora*.py. The outer prepared stack must also be considered when interpreting helper dtype comments.
[pypi.org/project/soup-cli/0.77.0/](https://pypi.org/project/soup-cli/0.77.0/)

**[11]** Apple. Cut Your Losses in Large-Vocabulary Language Models. Algorithmic background for avoiding full logits residency.
[github.com/apple-aiml-research/ml-cross-entropy](https://github.com/apple-aiml-research/ml-cross-entropy)

**[12]** Triton documentation. Fused Softmax. Data-movement motivation and the on-chip row-size condition.
[triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

**[13]** Actual Kaggle notebook installation cell, at the inspected challenge commit. Explicit dependency pins and the absence of explicit Torch/Triton pins.
[templates/kaggle_notebook.ipynb](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/templates/kaggle_notebook.ipynb)

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

### Provenance and interpretation

Starter sources were inspected at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. The report distinguishes publisher/organizer measurements from our calculations and untested design proposals. No winning kernel release, competition ranking or candidate T4 performance is asserted.

The live harness is more precise than the brief on repeated-run aggregation: it takes maximum memory peak rather than an average. The implementation details and numeric gates in this report follow the inspected harness. [5-8]

Round 2 adds the prepared AMP dtype audit, frozen-head cast analysis, compact-metrics integration plan, supervised-only head-backward derivation, exact NF4 dispatch analysis, LoRA epilogue proposal, and CPU checks. The older PDF is an earlier snapshot; this Markdown file is the updated research handoff.


## Appendix A. Portable CPU algebra checks

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
