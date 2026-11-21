# LoRA kernel research — round 3

Date: 2026-10-30  
Scope: the fixed Decentrathon 6.0 Soup Kernel Challenge workload  
Evidence: versioned source inspection, a reproduced CPU workload audit, and analytical kernel designs  
GPU status: no T4 benchmark or submitted solution kernel yet

## Decision

**Test an fp16 logits cache before committing to full output-head recomputation. In parallel, focus new LoRA kernels on backward traffic and small grouped matrix products.**

This round adds actual public-workload counts. It also identifies a middle design between retaining the ordinary fp32 loss graph and recomputing the head in backward: retain the already-rounded fp16 logits privately, stream fp32 reductions, and apply the real upstream loss scale when backward arrives.

The compact state is in KERNEL_HANDOFF.md. PUBLIC_WORKLOAD_AUDIT.json contains the machine-readable audit summary. PUBLIC_WORKLOAD_AUDIT_REPRO.zip contains the audit script, pinned-source checks, per-row and illustrative batch results, and reproduction instructions.

“Measured” in this report refers only to CPU token counts. Capacity, arithmetic, and traffic figures are calculations. None is a measured GPU speedup or peak-memory reduction.

## 1. We now know the public token workload

### The masking path is established

Soup defaults to training on responses only. It pretokenizes the formatted messages with masked labels, adds assistant masks derived from those labels, and passes the pretokenized data to TRL. TRL preserves those labels and applies the supplied assistant mask. Its separate default assistant-only flag does not turn this into full-sequence training. [S01], [S02], [S03], [S04]

The pinned Qwen chat template lacks generation markers. Soup consequently uses its incremental-render fallback. That fallback trains the assistant turn’s prefix/content/suffix token delta, including some role and boundary tokens; “response-only” should not be interpreted as natural-language answer text alone. [S05], [S06]

### CPU method

The audit used public dataset rows 0–255, the same Alpaca-to-messages conversion, the unmodified pinned Soup label-construction function, and truncation to 512 positions. A thin adapter rendered the exact Qwen Jinja template and encoded it with the downloaded tokenizer JSON. An independent prefix-delta check matched the labels for every row.

The final script was executed again after review and reproduced the totals. It used tokenizers 0.23.3 and Jinja2 3.1.6; it did not load model weights, torch, a GPU, or a full Trainer.

| Quantity | CPU audit result |
|---|---:|
| Public examples | 256 |
| Real, non-padding positions, Nr | **52,890** |
| Supervised next-token targets, Ns | **33,608** |
| Token-weighted Ns/Nr | **63.5432%** |
| Ns/(Nr−256), excluding each row’s final prediction position | 63.8523% |
| Rows truncated to 512 | 16 |
| Median row length | 140 |
| Median per-row supervised ratio | 42.6771% |
| Rows whose individual ratio is below 50% | 138 |

The median example is misleading for compute estimates: long responses dominate the token-weighted total. Use summed token counts, not the median of row ratios.

These numbers reproduce the specified formatting/masking/truncation through a thin tokenizer adapter. They remain distinct from a full installed Trainer run. The actual torch sampler order has not been reproduced.

### Immutable provenance

- Dataset: AmanMussa/kazakh-instruction-v2.
- Dataset revision: c641407ee50723c84a43ea2ec867b464d3893c12.
- Tokenizer: Qwen/Qwen2.5-1.5B-Instruct.
- Tokenizer revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- Soup source: 846888a045b31084452fa08a684a7a38c02f5f87.
- Public selection: rows 0–255 inclusive, exactly the starter selection. [S07]

The script checks SHA-256 hashes of tokenizer files, the Soup masking source, and the selected public-data serialization. The summary records the hashes and explicit limitations. Hidden rows and different seeds can have different distributions.

## 2. Padding changes the head-recomputation decision

The previous round derived the condition:

\[
N_r+2N_s<2N_p,
\]

where Np is the number of positions the baseline processes after batch padding. For these public rows:

\[
N_r+2N_s=120{,}106,
\qquad
N_p>60{,}053
\]

is the recomputation break-even point in a simplified head-GEMM operation count.

Although Ns/Nr exceeds one half, dynamic batch padding can make recomputation attractive. We calculated example groupings without changing any actual training run.

| Ordering or bound | Padded positions Np | Real/padded utilization | Recompute head work / baseline |
|---|---:|---:|---:|
| Source order, groups of four; illustrative | 83,700 | 63.19% | 0.7175 |
| Python shuffle seed 0; illustrative | 94,152 | 56.18% | 0.6378 |
| Python shuffle seed 1; illustrative | 93,880 | 56.34% | 0.6397 |
| Python shuffle seed 42; illustrative | 93,196 | 56.75% | 0.6444 |
| Python shuffle seed 2026; illustrative | 89,564 | 59.05% | 0.6705 |
| Length-sorted minimum-padding grouping; analytical bound | 53,580 | 98.71% | 1.1208 |

Across 1,000 Python shuffles, padded positions ranged from 86,516 to 98,088; the ordinary median was 92,400. The maximum possible grouping of these rows yields 105,728 padded positions. These are ordering analyses, not the harness’s actual permutation.

**Do not reorder the task’s batches.** Sorted grouping is only a lower-bound calculation. The optimization under consideration selects positions at the final output head within each unchanged batch. It preserves transformer attention, data order, and dropout calls.

The ratios omit GEMM efficiency, launch overhead, gathers/scatters, reduction work, and memory traffic. They are not latency predictions.

## 3. A stronger middle option: private fp16 logits caching

### Why this differs from both earlier choices

The ordinary path returns logits, builds its loss graph, and permits large fp32 allocations in loss and diagnostics. The fully recomputing design avoids retaining logits but calculates the head projection again during backward.

A cache design retains only the actual fp16 projection values needed for backward. It streams fp32 loss/statistic calculations and does not retain their full vocabulary-sized intermediates.

| Design, at 2,048 positions without filtering | Head-equivalent GEMMs | Approximate head arithmetic | Main retained activation |
|---|---:|---:|---|
| Ordinary frozen head | 2 | 1.912 TFLOPs | Ordinary logits/loss graph |
| Fully recomputed head | 3 | 2.868 TFLOPs | Hidden states |
| Private fp16 logits cache | 2 | 1.912 TFLOPs | 593.5 MiB fp16 logits |

The cache avoids approximately 0.956 TFLOPs of extra projection work at the maximum shape. Chunked backward GEMMs may be less efficient than the original large GEMM; equal arithmetic does not establish equal latency.

### Forward and backward contract

Let H be hidden states, W the frozen head, and Z16 the actual rounded fp16 head result:

\[
Z_{16}=\operatorname{GEMM}_{16}(H_{16},W_{16}^{\top}).
\]

Compute FP32 log-sum-exp, CE, entropy, and argmax from those rounded values. Retain Z16 and small per-token statistics privately.

When backward supplies its actual upstream scalar g:

\[
dZ_{32,i,j}=
\frac{g}{D_{\mathrm{loss}}}
\left[
e^{\operatorname{FP32}(Z_{16,i,j})-\mathrm{LSE}_i}
-\mathbf{1}[j=y_i]
\right].
\]

Ignored targets contribute zero. Only after applying g and the exact baseline denominator should dZ be rounded to the head GEMM dtype:

\[
dH=\operatorname{GEMM}_{16}(dZ_{16},W_{16}),
\]

followed by the reference-compatible conversion to the original hidden-state dtype.

This receives GradScaler’s real multiplier during backward. It avoids computing unscaled fp16 gradients in forward and does not need to predict the loss scale. Preserve the normal scaler and optimizer. The mathematical derivative is the same, but reduction order and GEMM tiling can still change numerical results; validate against the pinned autocast path. [S08], [S09]

### What to retain

For a frozen head, backward needs logits, the shared compute-dtype head weight, targets/maps, LSE, denominator, shape, and dtype metadata. It does not need hidden-state values to calculate dH; those would be needed for dW, which is absent. This is not an incremental saving over every baseline linear implementation: a normal frozen-head linear can also omit H. [S10]

Keep the cache in autograd saved tensors, outside returned model-output fields. Accelerate recursively upcasts returned half tensors; returning the cache could turn it into the very fp32 allocation we intended to avoid. Share one transient head cast across chunks and preserve the original tied fp32 parameter. [S11], [S12]

Initially use a separate dZ scratch buffer. Overwriting the saved logits can break retained-graph or repeated-backward behavior. Fall back when the head is trainable or callers require ordinary logits.

### Three implementation stages

1. **Full cache, original forward shape.** Keep the original head GEMM shape; cache its half result and stream reductions/backward. This gives the cleanest forward comparison.
2. **Cache supervised rows only.** Compute entropy for all required real positions, then retain logits only where shifted labels require gradients. Preserve all metrics and original row maps.
3. **Select head rows before projection.** Run the head only for the union of entropy-required and loss-required positions. Backward visits supervised rows and scatters dH to the original hidden-state layout.

Stage 3 changes GEMM shapes and introduces gather/scatter costs. It does not remove decoder tokens or repack attention.

A model-only patch returning logits=None remains insufficient. The trainer and rewind integration must consume genuine computed summaries and retain ordinary output behavior where necessary. [S04], [S13]

## 4. The public audit quantifies the cache opportunity

A supervised-only cache occupies:

\[
M_{\mathrm{cache}}=2N_sV\ \text{bytes},
\quad V=151{,}936.
\]

Across the 64 public batches, the mean is **152.18 MiB per batch**, independent of the permutation. The peak depends strongly on which rows share a batch.

| Cache quantity | Calculated fp16 storage |
|---|---:|
| Mean supervised-only cache | **152.18 MiB** |
| Largest source-order batch | 432.95 MiB |
| Largest illustrative Python seed-42 batch | 334.71 MiB |
| Maximum possible four-row grouping in this public set | 534.09 MiB |
| Full maximum-size 2,048-position cache | 593.5 MiB |

The average is not a peak-memory promise. The full head weight’s compute-dtype copy is another 445.125 MiB, and scratch buffers and existing activations have their own lifetimes.

When both forward and backward head rows are selected, cache arithmetic is proportional to Nr+Ns, rather than the recomputed design’s Nr+2Ns:

\[
\frac{F_{\mathrm{cache}}}{F_{\mathrm{baseline}}}
=\frac{N_r+N_s}{2N_p}.
\]

| Illustrative ordering | Selected-row cache head arithmetic | Selected-row recompute head arithmetic |
|---|---:|---:|
| Source order | 51.67% of baseline | 71.75% |
| Python shuffle seed 42 | 46.41% of baseline | 64.44% |

The seed-42 arithmetic reduction is approximately 53.6% **for the head**. It is not a 53.6% training speedup. The head is about 15.1% of a simplified count of frozen dense-linear forward/backward arithmetic at this architecture, so that example corresponds to roughly 8.1% less dense-linear arithmetic overall, before additional overhead and without modeling attention, dequantization, reductions, or launch costs.

For a first implementation, retain runtime-dependent masks. Do not hard-code public row identities or these ratios; the jury changes data and seed.

## 5. LoRA fusion should target launches and backward traffic

### Arithmetic is small; execution overhead may not be

Using M=2048, hidden size 1536, rank 16, q width 1536, v width 256, and 28 layers, a straightforward count gives:

\[
F_{\mathrm{LoRA}}
=6Mr(2H+N_q+N_v)L
\approx26.776\ \mathrm{GFLOPs}.
\]

The corresponding simplified frozen dense-linear forward/backward count, including the head and the 8960-wide MLP, is approximately 12.645 TFLOPs. LoRA is about **0.21% of that arithmetic**, while a straightforward graph can issue up to **336 adapter GEMMs per step**, plus casts, masks, scaling, and additions. Some early branches may omit input gradients, so this is an approximate upper count. [S14], [S15]

This does not establish LoRA’s runtime share. Narrow GEMMs can have poor utilization, and temporary traffic and launch costs can dominate them.

### Candidate L1: expansion plus dropout backward

Use the original independent q/v masks. Let Ui be the actual fp16 gradient entering A’s backward, after the reference B-gradient and scaling/casting operations.

For each branch:

\[
H_i=\operatorname{round}_{16}(U_iA_{i,16}),
\]

\[
dX_i=\operatorname{dropout\_backward}
(\operatorname{FP32}(H_i),\mathrm{mask}_i,0.05).
\]

An original kernel can compute the expansion and immediately perform the cast and dropout backward, avoiding intermediate materialization.

At maximum shape, each branch otherwise has a 6 MiB half expansion and a 12 MiB upcast. For two branches, eliminating those intermediate arrays avoids approximately **36 MiB of writes and 36 MiB of subsequent reads per layer**, conditional on the eager path materializing them as expected.

Compare two individually fused kernels with a paired q/v kernel. Pairing may reduce launches but increases accumulator pressure. Preserve each branch’s fp16 rounding and separate mask before any fp32 accumulation, and preserve frozen-base input gradients. Combining branches can change fp32 summation order, so exact bitwise agreement is not established. [S15], [S16]

### Candidate L2: group parameter-gradient GEMMs

The four independent adapter-gradient outputs are:

| Gradient | Output shape | Reduction dimension |
|---|---:|---:|
| dAq | 16 × 1536 | M |
| dAv | 16 × 1536 | M |
| dBq | 1536 × 16 | M |
| dBv | 256 × 16 | M |

With suitable orientation and 64×16 output tiles, these expose approximately 24+24+24+4=**76 tiles** in one grouped launch. The v B-gradient alone has only four such tiles.

A four-way split along the token reduction exposes 304 partial tiles. FP32 partials for all 77,824 adapter parameters occupy **1.1875 MiB**. This adds a reduction launch and changes accumulation order; test unsplit, two-way, and four-way variants rather than assuming more splitting is better.

Use a small fixed mapping for these four known problems. Keep v’s actual width; padding it to q’s width would introduce unnecessary work. Joint scheduling can also prolong operand lifetimes or delay a branch, so profile the entire backward.

### New literature support, with limits

ALTO’s §6.1 and Appendix A.1 describe separating base GEMMs from grouped adapter work and grouping backward calculations. Its heterogeneous multi-job H100 setting is different from this single-adapter T4 task. Use its decomposition as an idea, not its headline speedups or early-exit policy. Its LoRAFusion reference names a different paper from the CentML LoRAFusion reviewed earlier, so its comparison is not evidence against that implementation. [S17]

Triton’s Group GEMM tutorial supplies a scheduling reference, while NVIDIA’s NIM article explains splitting a long reduction to improve skinny LoRA GEMM utilization. The latter concerns inference. Neither establishes the best tile or split for this training workload. [S18], [S19]

## 6. Sharing a cast can change backward arithmetic

For fp32 residual inputs, separate q/k/v casts allow each branch’s half input gradient to be upcast before accumulation:

\[
dX_{\mathrm{separate}}=\sum_i\operatorname{FP32}(h_i).
\]

One ordinary shared fp16 input can instead accumulate the gradients in fp16:

\[
dX_{\mathrm{shared}}=
\operatorname{FP32}(\operatorname{FP16sum}_i h_i).
\]

A verified scalar counterexample is:

| Branch values | Upcast separately, then sum | Sum in fp16, then upcast |
|---|---:|---:|
| 2048 and 1 | **2049** | **2048** |

Forward values can match while backward changes. Reusing one cast requires explicitly preserving the backward conversion/accumulation structure; a shared half tensor alone is insufficient.

At maximum shape, removing two redundant fp32→fp16 casts could avoid 36 MiB of forward traffic per layer. That is a traffic estimate, not persistent-memory savings. It also does not justify moving dropout after the cast, sharing masks, or merging LoRA into frozen weights. [S15], [S16]

Materializing dense BA is unattractive here: forming and applying both dense q/v updates costs approximately 35.6 times the arithmetic of the factorized adapter forward at maximum shape, while changing operation ordering. Preserve the low-rank factorization.

## 7. Attention is another profiling lead, with a backend constraint

The inspected defaults select SDPA. PyTorch 2.11’s CUDA Flash and cuDNN attention hardware checks require SM80 or newer, excluding T4’s SM75. Efficient attention remains a possible backend; the actual choice has not been observed. [S20], [S21]

The transformers SDPA adapter expands K/V heads when an attention mask prevents its native GQA path. Qwen has 2 KV heads and 12 query heads, so that expansion is sixfold. Unequal-length right-padded batches make this a concrete layout/cast/traffic lead. [S14], [S22]

Do not simply force enable_gqa: the source warns that masked GQA can select the math backend. First record the actual backend, mask shape, and repeat/cast allocations. A future independent layout or rotary/cast fusion may be lower scope than replacing attention itself.

## 8. Which speedrun ideas transfer?

The nanochat and modded-nanogpt projects mix execution improvements with changes to models, losses, optimizers, and training schedules. Their objectives and hardware differ from this fixed Soup run. [S23], [S24]

| Idea | Treatment in this challenge |
|---|---|
| Group independent skinny GEMMs | Useful design idea; implement original kernels and preserve math/dtypes |
| Fuse loss/metric reductions and reduce intermediate traffic | Useful design idea; preserve every metric and loss contract |
| Explicit precision and cast placement | Useful analysis method; retain this task’s precision boundaries |
| Add n-gram embeddings, change model depth, or alter attention structure | Changes the prescribed model |
| Sample vocabulary entries or change the loss/softcap | Changes the prescribed objective |
| Replace AdamW with Muon/ANVIL or change update frequency | Changes the prescribed optimizer/training |
| Capture qualifying kernels only inside CUDA graphs | Conflicts with the starter’s launch-observation requirement |
| Assume speedrun compilation/warm-up is outside the score | Wrong for this harness; complete command time counts |

The starter’s rules determine eligibility. Existing kernels can inform ideas, but ready-made imports or copied kernels do not become our original contribution. [S25]

## 9. Concrete next experiments

| ID | Experiment | Main decision |
|---|---|---|
| M0 | Fused entropy and strided accuracy, original loss graph | Establish a small integration baseline |
| H0 | Private full fp16 logits cache, original forward GEMM shape | Validate precision, masks, summaries, and scaled backward |
| H1 | Supervised-only cache; then select real forward rows | Measure the opportunity quantified by the CPU audit |
| L1 | Adapter expansion + cast + dropout backward | Test the strongest new LoRA traffic hypothesis |
| L2 | Group four adapter parameter gradients; limited reduction splitting | Test launch/utilization improvement |
| H2 | Recompute instead of caching | Choose additional memory saving only if its total-run tradeoff wins |
| A0 | Observe SDPA backend and K/V expansion | Decide whether an attention-layout kernel is worthwhile |

On T4, record actual package versions, runtime dtypes, original batch order and masks, peak allocated/reserved memory, complete command time, and original-kernel share. The CPU audit is a reproducible expectation to compare with the trace, not permission to assume the trace matches.

For H0/H1, compare forward losses/metrics, complete hidden gradients, and several ordinary scaled optimizer steps. For L1/L2, include nonzero B, independent q/v masks, parameter-gradient casts, and base-path gradients. Preserve the normal optimizer/scaler and fallback paths. Then run the unchanged harness and required tests.

The original-kernel share must still reach 5% of measured step wall time in every measured run. A good memory change alone does not satisfy that condition. Keep compilation/autotuning costs small for the 64-step command. [S25]

## 10. Completed and pending

Completed this round: public CPU token/mask audit, illustrative padding analysis, immutable input hashes, a reusable audit bundle, the fp16-cache design, grouped-backward candidates, a shared-cast counterexample, and an attention/backend source audit.

Pending: the actual team checkout and T4 execution environment, full runtime verification, original kernel implementation, and all performance and quality measurements.

**Next concrete action:** obtain the untouched T4 trace and build H0 plus M0 as the first head/loss experiment, while developing L1 as the independent LoRA contribution. H1’s row selection and cache compaction follow after H0 agrees numerically.

## Primary sources

- [S01] — Soup configuration defaults
- [S02] — Soup formatting selection
- [S03] — Soup pretokenization and mask preservation
- [S04] — Pinned TRL trainer and collator
- [S05] — Pinned Soup label construction
- [S06] — Immutable Qwen tokenizer configuration
- [S07] — Starter public-data selection
- [S08] — Transformers loss normalization
- [S09] — PyTorch AMP examples
- [S10] — PyTorch custom autograd
- [S11] — Accelerate output conversion
- [S12] — PyTorch autocast caching
- [S13] — Soup rewind
- [S14] — Official Qwen dimensions
- [S15] — Pinned PEFT LoRA
- [S16] — PyTorch derivative definitions
- [S17] — ALTO
- [S18] — Triton grouped GEMM
- [S19] — NVIDIA LoRA reduction splitting
- [S20] — Transformers attention selection
- [S21] — PyTorch CUDA SDPA dispatch
- [S22] — Transformers SDPA adapter
- [S23] — nanochat README
- [S24] — modded-nanogpt README
- [S25] — Fixed challenge rules and execution contract

[S01]: https://example.invalid/redacted-github
[S02]: https://example.invalid/redacted-github
[S03]: https://example.invalid/redacted-github
[S04]: https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py
[S05]: https://example.invalid/redacted-github
[S06]: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/tokenizer_config.json
[S07]: https://example.invalid/redacted-github
[S08]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py
[S09]: https://docs.pytorch.org/docs/2.11/notes/amp_examples.html
[S10]: https://docs.pytorch.org/docs/2.11/notes/extending.html
[S11]: https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py
[S12]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.cpp
[S13]: https://example.invalid/redacted-github
[S14]: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
[S15]: https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py
[S16]: https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml
[S17]: https://arxiv.org/html/2604.05426v1
[S18]: https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
[S19]: https://developer.nvidia.com/blog/seamlessly-deploying-a-swarm-of-lora-adapters-with-nvidia-nim/
[S20]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py
[S21]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
[S22]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/integrations/sdpa_attention.py
[S23]: https://example.invalid/redacted-github
[S24]: https://example.invalid/redacted-github
[S25]: https://example.invalid/redacted-github

