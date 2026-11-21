# Qwen Kernel Optimization: Research and Experiment Review

- **Project:** Decentrathon 6.0 / Soup Kernel Challenge
- **Research date:** 30 October 2026
- **Edition:** Third pass: official starter, pinned dependency audit and saved-state encoding
- **Status:** Starter/harness and exact source versions inspected; two CPU numerical studies completed. Contest kernels, T4 benchmarks and official eligibility remain unverified.
- **Purpose:** A self-contained Markdown reference for review and continuation.

## Session handoff

**Objective:** improve complete training runtime and sampled peak GPU memory while preserving the fixed recipe, gradients, adapter quality and required outputs.

**Recipe now resolved:** Qwen2.5-1.5B-Instruct, NF4 double quantization, Q/V LoRA rank 16/alpha 32, **dropout 0.05**, batch 4, sequence length up to 512, 64 steps, **checkpointing off**, one Kaggle T4 with FP16 autocast. Final trainable LoRA storage is FP32. [S1] [S24] [S25] [S10]

**Major gap closed:** the official starter is inspected at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. Alongside Soup 0.77.0, its notebook pins Transformers 5.16.1, PEFT 0.20.0, TRL 0.29.1, bitsandbytes 0.50.2 and Accelerate 1.14.0. The supplied example is **SwiGLU**; its kernels remain excluded under any rewrite. [S26] [S30]

**Leading original experiments:** A0, strided metric reductions avoiding TRL's large accuracy copy; R0, FP32 RMSNorm fusion; M0, exact encoding of selected saved MLP gate/up tensors. Keep supplied SwiGLU separately identified as an uncredited optimization. N0 dequantization and B0 LoRA epilogues remain useful candidates.

**Concrete targets:** maximum-shape FP32 accuracy copy **1,184.682 MiB**; possible reduction in retained gate/up storage **1,960 MiB** across 28 layers. These are tensor-size calculations, not measured peak savings, and must not simply be added together.

**Compiler constraint:** exact upstream Triton 3.6.0 disables ordinary Tensor Core MMA lowering below compute capability 8.0. Inspect the actual image's generated code before relying on `tl.dot` for a fast GEMM. [S19] [S20] [S21]

**Measurement resolved:** every run uses empty compiler caches. Eligibility uses CUDA-event time for original kernels divided by synchronized step wall time from step 3 onward, taking the lowest solution run. Whole-command time still includes compilation and the first steps. [S27] [S28] [S29]

**Executed checks:** 54 CPU vocabulary-reduction cases; exhaustive finite binary16 round trips and a sampled FP32 SwiGLU derivative check for M0. No downloaded package code was executed.

**Remaining evidence:** actual T4 profile, first-training-step dtypes, saved-storage ownership, generated PTX, full training correctness, cold-run speed/memory and official checks. Source access is resolved; device validation is outstanding.

## 1. Recommendation and rationale

Public implementations establish that the relevant Qwen operations are established optimization targets. Unsloth publishes the exact model family and links T4 training notebooks; Liger lists Qwen2.5 support. The useful research question is which remaining work can be independently implemented, integrated and measured under this fixed training recipe. [S3] [S4]

The organizers provide this approximate GPU step profile: [S1]

| Work category | Reported share | Revised interpretation |
| --- | ---: | --- |
| Elementwise operations and dtype casts | 44% | Primary speed target; obtain an operation-level breakdown before choosing kernels |
| Matrix multiplication | 36% | Preserve efficient existing GEMMs initially; custom Triton GEMM has a T4 lowering risk |
| Vocabulary softmax | 7% | Share reductions across loss and metrics; consider bounded head storage |
| Attention | 5% | Lower initial priority for this workload |
| NF4 unpacking | 5% | Fuse nested-scale decode and unpacking; distinguish forward from backward |

These are approximate GPU-time shares, not a precise end-to-end decomposition. The 44% category is not the cost of any one normalization, activation or LoRA operation.

**Working arrangement:** one track owns A0 and later A1; another owns R0 and M0; a third evaluates N0/B0 against the actual profile. The starter and pinned dependency audit in sections 4–5 now make these concrete. Combine only candidates that remain correct, eligible and beneficial.

### 1.1 Evidence levels

| Label used in this report | Meaning |
| --- | --- |
| Pinned source finding | Read in the exact downloaded release, with provenance recorded in section 11 |
| Current upstream finding | Observed in the linked public source on the research date; installed contest version still needs confirmation |
| Calculation or hypothesis | Our derivation or proposed implementation; no measured GPU benefit implied |
| CPU validation | Original numerical experiment in NumPy; does not validate Triton, FP16 AMP or Soup integration |
| Contest result | None established in this research |

## 2. Fixed workload and submission requirements

### 2.1 Target configuration

The following values come from the brief and inspected starter, not a locally reproduced T4 run. [S1] [S24] [S25]

| Setting | Required value |
| --- | --- |
| Training application | Soup 0.77.0 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Base weights | 4-bit NF4 |
| Adapter | LoRA, `r=16`, `alpha=32` |
| Adapter targets | `q_proj`, `v_proj`; YAML uses `target_modules: auto` |
| LoRA dropout | 0.05 |
| Gradient checkpointing | Disabled |
| Gradient accumulation | 1 |
| Optimizer / schedule | `adamw_torch` / cosine, learning rate `2e-4` |
| Batch size | 4 |
| Sequence length | Up to 512 tokens |
| Training workload | 256 examples, one epoch, 64 steps |
| Scored hardware | One free Kaggle T4 |
| Execution precision | FP16 autocast; storage dtype varies across parameters and activations |
| Specified environment | Python 3.13, PyTorch 2.11, Triton 3.6.0 |
| Integration | Original kernels in `solution/`, connected through `apply()` |

Final evaluation uses hidden training/evaluation rows and another seed with the remaining task settings fixed. Specialization must depend on legitimate model dimensions and input shapes, not memorized examples. [S1]

The notebook's additional installation pins are Transformers **5.16.1**, PEFT **0.20.0**, TRL **0.29.1**, bitsandbytes **0.50.2** and Accelerate **1.14.0**. Record actual installed versions/builds as a reproducibility check. [S26]

### 2.2 Originality and allowed reuse

- Attributed ideas from other implementations are allowed.
- Copied kernel code and ready-made optimization libraries are forbidden, including Unsloth, Liger, xFormers, FlashAttention, torchao and similar libraries; the full list is in the harness.
- Useful original `@triton.jit` kernels must collectively account for **at least 5% of the solution's training time from step 3 onward, in every measured run**.
- This threshold concerns their runtime **after optimization**, not the baseline fraction they replace.
- Compiler-generated kernels, unused results and the supplied **SwiGLU forward/backward** do not count. Rewriting the example does not make it eligible. [S30]
- Open-source licensing and attribution do not override the competition's restrictions on copied implementations. [S1]

An extremely fast but small custom kernel may fail the participation threshold. Establish sufficient useful original work without adding artificial computation.

The starter permits retaining its SwiGLU alongside original work. Its README reports roughly 2,000 MiB lower peak memory and speed within a few percent for that example; these are organizer observations, not measurements reproduced here. Keep it labeled as supplied and uncredited. [S24]

### 2.3 Task invariants

Preserve the model, data within the supplied run, batch contents and order, number of steps, optimizer, learning-rate schedule, LoRA settings including dropout, trainable-parameter set and required outputs. [S1]

Do not switch built-in modes such as `gradient_checkpointing`, `activation_offloading`, `fast_lora`, `use_cut_ce` or packing. Custom removal of padding inside an existing batch is allowed; mixing examples across different batches is not. Do not alter the measurement/evaluation scripts or their allocator settings. [S1]

### 2.4 Material correctness and admission conditions

| Check | Requirement from the brief |
| --- | --- |
| Hidden-set adapter evaluation loss | No more than 0.5% above baseline |
| First-step training loss | Within 1% |
| Average training-curve deviation | Within 3% |
| Logged entropy/accuracy and per-row losses in `rewind.jsonl` | Within 3% |
| Token counts per step | Match exactly |
| Whole-command runtime | No more than 10% above baseline |
| Peak GPU memory | No more than 10% above baseline |
| Existing Soup tests | All 79 required tests in the specified five files pass |
| Reproducibility | Clean Kaggle notebook `Run All` works without edits |
| Documentation | Required README and submission artifacts are provided |

The inspected harness uses `abs(solution-reference) / max(abs(reference), 1e-12)` and averages these relative gaps over paired observations. Entropy/accuracy use logged series; rewind losses are flattened across rows. Near-zero references therefore require care. [S1] [S28]

### 2.5 Measurement and score

- **Time:** the entire `soup train` command, including model loading, kernel compilation and adapter saving.
- **Order:** baseline, solution, solution, baseline; average the two measurements for each side.
- **First two steps:** excluded only from original-kernel participation. Its denominator is synchronized step wall time from step 3, not total GPU kernel time; the lowest solution run must pass. They still count in complete-command timing. [S28] [S29]
- **Cache behavior:** every run removes/recreates its run directory and uses new Triton/Inductor cache directories. Compilation is charged again in every measured run. [S27]
- **Memory:** whole-run GPU peak sampled through `nvidia-smi` every 0.1 seconds. Lower PyTorch allocated memory alone does not establish a lower scored peak. [S1]

| Criterion | Available points |
| --- | ---: |
| Eligibility and correctness | 20 |
| Speed | 35 |
| Memory | 30 |
| README and reproducibility | 15 |

Performance scores are logarithmic relative to the best admitted team. A ratio below 1.05 earns zero for that metric. Memory breaks total-score ties. Failing an admission requirement excludes the submission. [S1]

## 3. Verified public work

| Source | Verified finding | Interpretation for this challenge |
| --- | --- | --- |
| Unsloth Qwen2.5-1.5B 4-bit [S3] | Exact model family and size; links to Qwen2.5 fine-tuning notebooks for Tesla T4 | A model repository does not itself activate optimized kernels. Generic speed claims are not measurements against Soup. |
| Liger Kernel [S4] | Explicit Qwen2/Qwen2.5 support for RoPE, RMSNorm, SwiGLU, cross-entropy and fused linear cross-entropy | Useful algorithm references. The library and copied implementations cannot be used in the submission. |
| HF agent-written Qwen RMSNorm [S2] | Qwen3-8B RMSNorm: reported 1.94x average isolated speedup over a PyTorch baseline; 1.58x-2.47x range; H100 80 GB, BF16 | Evidence of an optimized operation, not whole-model training acceleration or demonstrated T4 performance. |
| HF standalone RMSNorm [S5] | Forward and backward entry points are published; no benchmark on the retrieved card | Reference for a training interface. Availability alone does not establish performance. |
| Cut Cross-Entropy [S6] | Avoids full logits materialization. Published Triton implementation requires Ampere or newer. Default CCE includes gradient filtering; an unfiltered mode is documented | Study the memory-saving algorithm. T4 support, numerical agreement and Soup integration still require independent work. |

### 3.1 What the Qwen3.7 article establishes

Qwen3.7-Max was the **agent performing the optimization**. Qwen reports a **10.0x geometric-mean speedup** for SGLang Extend Attention on T-Head ZW-M890 PPUs after about **35 hours and 432 evaluations**. The baseline, hardware and serving workload differ from the contest. These are the authors' results, not measurements reproduced here. [S8]

The transferable ideas are iterative profiling, eliminating avoidable allocation and synchronization, sharing memory loads, and selecting parallelism for actual workload shapes. The attention multiplier should not be used as a forecast for Qwen2.5 LoRA training on T4.

### 3.2 What the second pass established

The exact `soup-cli==0.77.0` wheel was downloaded, its SHA-256 checked against PyPI metadata, and its Python files inspected statically. It was not installed or imported. The release notes independently identify the chunked head/loss and optional scale-cache features. [S10] [S11]

| Component | Verified source finding | Consequence |
| --- | --- | --- |
| `stream_loss_chunk.py` | Existing chunked output-head/loss and metric summaries | Bounded loss is relevant prior art; enabling the built-in option is not our kernel contribution |
| `rewind_hf.py` | Reuses original logits; row CE runs one example at a time | Do not claim rewind performs another model forward or explicitly creates a whole-batch FP32 logits copy |
| `fast_lora_qkv.py` | Groups the present adapter A matrices and portions of backward | Shared Q/V adapter projection may already exist in the chosen execution path |
| `fast_lora.py` | Adapter matmul followed by dtype promotion, scaled add and cast | Remaining epilogue passes are a concrete fusion candidate |
| `layer_stream_runtime.py` | Optional decoded-scale cache; dense NF4 input-gradient fallback | Caching is known prior art; backward dequantization is a specific research target |
| `loss_summary.py` | Extracts scalar loss summaries from logs | It is not the numerical training-loss hotspot |

These describe available package implementations. The fixed task uses ordinary PEFT with Fast-LoRA and streamed loss inactive; their code remains prior art rather than an enabled baseline path. [S10] [S24] [S25]

The earlier pass knew only Soup's broad dependency ranges. The official notebook now resolves that uncertainty with exact pins. The following analysis uses those release-tag sources, while actual runtime/build verification remains a separate requirement. [S26]

## 4. Track A: loss and metrics with bounded memory

**Hypothesis:** share vocabulary reductions and reduce temporary storage while preserving every consumer of the model's scores.

### 4.1 The exact output contract matters

The pinned TRL/Transformers code and Soup recorder establish the following contract. Validate the implemented replacement against actual baseline outputs. [S10] [S35] [S37]

| Output | Positions | Reduction or denominator |
| --- | --- | --- |
| Training CE | Logit at `t` predicts label at `t+1`; ignore `-100` | Sum divided by `num_items_in_batch` when supplied, otherwise supervised-token count |
| Token accuracy | Same shifted supervised positions | Correct count / supervised count |
| Entropy | All `attention_mask` positions, including prompt and final valid position | Mean over those positions; with `position_ids` instead, mean over all positions |
| Total tokens | Attention-mask positions | Accumulated token count, including prompts |
| Rewind row loss | Shifted supervised positions within each example | Per-example mean; empty supervised row yields zero |
| Rewind row tokens | `labels[:, 1:] != -100` | Exact count for that example |

**Design consequence:** removing prompt logits because their labels are ignored is insufficient. Entropy still needs their distributions or equivalent summaries. Removing padding inside a batch can be useful only if masks, indices, counts and reconstructed outputs retain the required meaning.

Rewind's ordinary path already reuses forward logits under `no_grad`, processes one example at a time and casts the token-loss vector to FP32. Repeated CE work and scalar transfers remain candidates to profile. In the expected native-AMP path, Accelerate converts returned logits to FP32 before metrics and rewind consume them. Source-level allocations still do not establish their simultaneous lifetime or measured peak. [S38] [S39] [S10]

### 4.2 Why bounded output-head storage is attractive

The official vocabulary contains 151,936 entries. At the maximum batch shape: [S7] [S1]

```text
4 × 512 × 151,936 = 311,164,928 logits
FP16: 622,329,856 bytes = 593.5 MiB
FP32: 1,244,659,712 bytes = 1,187 MiB
```

These are calculated sizes of one tensor, not observed peak VRAM. A chunk containing 128 total flattened token positions would hold 37.09375 MiB of FP16 logits, or 74.1875 MiB in FP32. “128 total positions” is different from a sequence chunk of 128 for every member of a batch of four.

The head still has to perform the necessary arithmetic. Smaller GEMMs, repeated weight reads and recomputation can trade speed for memory. The scored peak also includes loading, allocator reservations and other live tensors.

### 4.3 Stage A0: avoid TRL's accuracy copy

TRL 0.29.1 makes `outputs.logits[..., :-1, :]` contiguous before argmax. With batch size four, the shifted view has gaps between sequences; its maximum FP32 copy is:

```text
4 × 511 × 151,936 × 4 bytes = 1,184.681640625 MiB
```

An original strided reduction can read the existing logits and compute the same predictions without that buffer. **Intercept the metric path before the copy.** Replacing argmax after the copy has occurred does not remove it. [S35]

Entropy is already processed in **128 flattened-token-row chunks**. One FP32 vocabulary intermediate for that chunk is 74.1875 MiB. Target per-chunk log-softmax/exp/product intermediates and repeated reads; do not claim this pinned implementation creates full-batch entropy log-probabilities. [S36]

Keep ordinary model forward and differentiable CE initially. Original metric reductions should preserve masks, first-index argmax ties, logging/gather conventions and exact counts. Leave rewind unchanged for the first experiment; share its row CE only after equivalence is verified.

| Consumer | Pinned behavior |
| --- | --- |
| Transformers CE | Full logits `.float()` inside forward; shifts labels rather than copying shifted logits |
| Accelerate model wrapper | Converts returned low-precision tensors to FP32 after forward |
| TRL entropy | Computes vocabulary intermediates for 128 token rows at a time |
| TRL accuracy | Copies shifted logits to contiguous storage, then argmax |
| Soup rewind | Calculates row CE from returned logits |

The internal CE cast and outward cast are distinct logical operations. Their sizes cannot simply be added with every other temporary and presented as simultaneous peak memory. [S37] [S38] [S39]

### 4.4 Stage A1: an original bounded-memory head

After A0 establishes the contract:

1. Process bounded token chunks with an efficient allowed output-head GEMM.
2. Produce required statistics with original Triton reductions.
3. Retain only hidden states and compact summaries needed for backward; ensure autograd does not retain every chunk's logits.
4. Recompute scores or gradients in backward when necessary.
5. Supply the original metric consumers and rewind recorder coherently.

The output head is frozen in this recipe, so it requires hidden-state gradients but no trainable head-weight gradient. Preserve the actual trainable set and tied weights. Return compact detached metrics on the optimized training path: returning full low-precision logits still invites Accelerate's FP32 conversion, and concatenating chunks restores the allocation. Keep ordinary output behavior where an evaluation/prediction consumer needs logits. [S38] [S39]

The existing Soup implementation offers concrete integration references: [S10]

| File / entry point | Role to understand |
| --- | --- |
| `trainer/sft.py::SFTTrainerWrapper.setup` | Trainer construction and wrapper ordering |
| `trainer/sft.py::_build_rewind_trainer` | Attaches rewind row/sampler state |
| `trainer/rewind_hf.py::make_rewind_trainer_class` | Records per-row results after parent loss computation |
| `utils/stream_loss_chunk.py::make_chunked_loss_trainer_class` | Demonstrates how logits-dependent metrics need an alternative consumer |
| `_soup_chunked_step` interface | Existing route for per-token summaries when logits are absent |

Use these to understand the contract. Copying the implementation or turning on `stream_loss_chunk` is not the proposed contest solution.

### 4.5 Stable online vocabulary reduction

For each vocabulary block, define a maximum `m` and two centered sums:

```text
s = Σ exp(z_j − m)
q = Σ exp(z_j − m) × (z_j − m)
```

Merge two block states without storing the full vocabulary:

```text
M = max(m_a, m_b)
a = exp(m_a − M)
b = exp(m_b − M)
s = a × s_a + b × s_b
q = a × (q_a + (m_a − M) × s_a)
  + b × (q_b + (m_b − M) × s_b)

Entropy = log(s) − q / s
CE      = log(s) + (M − z_target)
```

This is an original mathematical formulation for the proposed reduction, not a claim of a novel algorithm. Track argmax with first-index tie behavior, target logits and the appropriate masks. Initialize from a nonempty valid block or handle empty/tail blocks explicitly. Avoid creating `0 × (-infinity)` when masking vocabulary tails.

The CE ordering above avoids forming a large absolute log-sum-exp and then subtracting a nearby target score. The CPU check in section 10 found a material difference on large common offsets.

For supervised positions, the mathematical gradient is:

```text
dZ = incoming_loss_gradient × (softmax(Z) − one_hot(target)) / denominator
```

Ignored positions have zero CE gradient. Entropy and accuracy are logging outputs, not extra objectives.

Keep reduction state and normalization in FP32. Uniform scores give `s=151936`, above FP16's largest finite value 65,504, even though final entropy is only `log(151936) ≈ 11.9312`. A finite final result does not make low-precision intermediate accumulation safe. [S41]

### 4.6 FP16 gradient scaling needs explicit design

The pinned chunked implementation calculates hidden gradients in forward and multiplies them by the incoming scalar gradient later. For our proposed FP16 implementation, rounding a small gradient before applying GradScaler can lose information that later multiplication cannot recover. This is a numerical risk inferred from operation order; it is not a reproduced Soup defect. [S10] [S22]

Apply the incoming gradient and supplied denominator in FP32 before low-precision gradient materialization. The normal GradScaler initial scale is 65,536, itself beyond finite FP16 range: do not cast that scalar to FP16 first. Preserve `num_items_in_batch` across chunks and actual autocast/rounding behavior. The fixed task accumulates one microbatch, but normalizing each head chunk separately would still be wrong. [S22] [S37] [S41]

## 5. Track B: pointwise operations, casts and LoRA

**Hypothesis:** reduce measured memory traffic and launch overhead while retaining the fixed training semantics.

### 5.1 Architecture and shape specialization

The official configuration gives hidden width 1536, MLP width 8960, 28 layers, 12 query heads, two KV heads, vocabulary 151,936, RMSNorm epsilon `1e-6` and tied embeddings. Head width is therefore 128. [S7]

At `batch × sequence = 2048` positions:

| Projection | Frozen weight shape | Rank-16 expansion |
| --- | --- | --- |
| Q | `[1536, 1536]` | `[2048,16] × [16,1536]` |
| V | `[256, 1536]` | `[2048,16] × [16,256]` |

Q's output is six times wider than V's. Dispatch/tune by actual projection and token count. The model's stored BF16 metadata does not override the contest's FP16 requirement.

### 5.2 Identify work that is still present

Soup's optional Fast-LoRA is inactive in this recipe, and its grouped path rejects the task's nonzero dropout. Its graph grouping remains prior art; new LoRA experiments must target the actual ordinary-PEFT path. [S10] [S24] [S25]

Current upstream PEFT's 4-bit path also includes a defensive clone of the base result, followed by adapter linears, scaling, possible conversions and addition. PEFT explains that clone in terms of backward/view safety. Removing it blindly is not a valid optimization argument. [S12]

**B0:** keep existing efficient matmuls; fuse compatible final scale, cast and addition into a correctly owned output. This preserves a manageable correctness boundary and avoids depending on automatic Tensor Core lowering.

**B1:** consider fusing rank-16 expansion with its output operations. Benchmark against B0 and the original path. A `tl.dot` implementation inherits the T4 issue in section 6; a direct small-rank reduction is another experiment, not an assumed win. Preserve intermediate rounding and promote/add/cast semantics when the baseline requires them.

RMSNorm, RoPE-adjacent casts and compatible input-gradient accumulation remain candidates. SwiGLU is the supplied, uncredited example. Sections 5.6–5.8 refine the remaining boundaries.

### 5.3 Preserve all gradients

For `j` equal to Q or V, with dropout probability `p`, binary mask `M_j` and scaling `s=32/16=2`:

```text
X_j = M_j × X / (1 − p)
H_j = X_j A_j^T
Y_j = X W_j^T + bias_j + s H_j B_j^T

dB_j = s G_j^T H_j
dH_j = s G_j B_j
dA_j = dH_j^T X_j
dX_j = G_j W_j + M_j × (dH_j A_j) / (1 − p)
```

Here `G_j` is the upstream output gradient. With dropout inactive, use an identity mask and an effective probability of zero. Sum Q/V input contributions with the unadapted K contribution and other graph branches as required.

Frozen base weights remove base-weight gradients, not input gradients. Keeping only `H_j` cannot recover the full input needed for `dA_j`; freezing A would change the training method.

These equations describe real arithmetic. A fused implementation still has to respect storage dtype, accumulation dtype, casts, dropout and the optimizer's expectations.

### 5.4 Fixed dropout and parameter-dtype lifecycle

The task uses **dropout 0.05** on Q and V. Preserve their independent masks and RNG consumption. Axolotl's shared-mask grouped API illustrates why a published fusion is not automatically compatible. [S14] [S24]

The pinned setup has a phase-dependent dtype:

| Phase | LoRA parameter storage |
| --- | --- |
| Relevant PEFT promotion | FP32 |
| TRL trainer construction | BF16 for trainable parameters of the four-bit model |
| Soup immediately before `trainer.train()` | FP32 restored for trainable `lora_*` parameters in FP16 mode |

The last step is `align_trainable_dtype_for_fp16` in Soup. Final training storage is FP32, although a hook after trainer construction can observe BF16 temporarily. Nonzero initialized A values have already been rounded by the BF16 conversion; restoring FP32 does not recover those bits. Do not remove the round trip, cache stale storage or change the precision recipe. Inspect dtypes at the real training boundary. [S31] [S35] [S10]

### 5.5 Arithmetic and buffer accounting

Our calculations use `M=2048`, input width 1536, rank 16 and 28 layers; one multiply-add counts as two FLOPs. They exclude dequantization, dropout, casts and launch overhead. The per-layer table assumes that layer's input requires gradients; the aggregate below accounts for the first frozen prefix.

| Quantity | Q | V | Q + V |
| --- | ---: | ---: | ---: |
| Adapter parameters / layer | 49,152 | 28,672 | 77,824 |
| Adapter forward, GFLOPs / layer | 0.2013 | 0.1174 | 0.3188 |
| Adapter forward + backward, GFLOPs / layer | 0.6040 | 0.3523 | 0.9563 |
| Frozen base forward + input backward, GFLOPs / layer | 19.3274 | 3.2212 | 22.5486 |

Across 28 layers, adapters contain 2,179,072 parameters, or 8.3125 MiB as FP32 weights. Counting input gradients in every layer gives 26.776 GFLOPs for adapters and 631.360 GFLOPs for Q/V base projections. Accounting for the already-pruned first-prefix input gradients gives **26.575 GFLOPs** and **620.086 GFLOPs**, respectively, a ratio of about **4.286%**. These are arithmetic estimates, not runtime shares; utilization and copies can make small operations consume disproportionate time.

| Per-layer buffer at the maximum token count | FP16 | FP32 |
| --- | ---: | ---: |
| Input X | 6 MiB | 12 MiB |
| Both low-rank intermediates H | 0.125 MiB | 0.25 MiB |
| Combined Q/V adapter outputs | 7 MiB | 14 MiB |

Two boolean dropout masks occupy 6 MiB; two materialized FP16 dropped inputs occupy 12 MiB. These are individual buffer sizes, not guaranteed live-memory savings. Shared storage must be deduplicated. Checkpointing changes which tensors coexist.

Removing one full read/write round trip over the combined FP16 Q/V outputs avoids 14 MiB of traffic per layer. That is a bandwidth calculation, not a 14 MiB peak-memory reduction. Recomputing both H tensors could avoid 3.5 MiB of saved storage in FP16, or 7 MiB in FP32, across all 28 layers if they were simultaneously retained. These are theoretical storage amounts; larger copies deserve attention first.

RunLoRA is relevant training prior art for selecting matrix-association orders by shape and measured cost. Its published setting does not establish benefits for this fixed NF4/dropout recipe. [S23]

### 5.6 R0: FP32 RMSNorm fusion

The pinned producer chain explains the FP32 traffic: PEFT promotes applicable frozen parameters, embeddings/residuals are FP32, and bitsandbytes promotes FP16 projection results back to its input dtype. Qwen normalizes in FP32, converts the normalized value to its original input dtype, then multiplies gamma. Returning FP16 from a generic replacement would change this boundary. [S31] [S32] [S33]

For FP32 input and frozen gamma:

```text
r  = rsqrt(mean(x*x) + epsilon)
y  = gamma * (x*r)
v  = dy * gamma
dx = r*v − x*r*r*r*mean(v*x)
```

Fuse reductions and full-tensor passes while keeping the FP32 output and input-gradient contract. Preserve epsilon `1e-6`. When padding width 1536 to a convenient block, mask invalid lanes and divide by **1536**, not padded width. Frozen gamma needs no gradient; eager autograd already omits it.

At 2,048 positions, one FP32 hidden tensor is 12 MiB. A simple forward model counts about 84 MiB for separate square/reduce/normalize/gamma passes versus about 24 MiB for a fused read/write, excluding small row/gamma traffic and cache effects. The modeled difference is 60 MiB per norm, or **3,420 MiB across 57 norm forwards**. This is logical traffic, not measured speed or allocated-memory savings.

The first frozen preattention norm should already need no input backward; the other 56 norm input-gradient paths remain relevant. Multiple saved references to one input are not separate dense allocations.

Residual-plus-norm fusion must retain both the residual sum and normalized result. At the attention boundary, FP32 residual plus FP16 attention output gives FP32. Writing both outputs reduces modeled traffic from about 54 to 42 MiB relative to separate add plus already-fused norm: only 12 MiB of extra saving per occurrence. Preserve both output gradients and their dtype boundaries.

### 5.7 M0: exact encoding of selected saved MLP activations

The MLP gate/up projections are bias-free and unadapted here. bitsandbytes computes FP16 outputs and promotes them to the FP32 input dtype. Their finite FP32 values therefore remain exactly representable in FP16. [S32] [S33] [S42]

Proposed saved-state path:

```text
FP16 projection output → ordinary FP32 gate/up used by forward
→ compact FP16 saved payload
→ the same FP32 gate/up values restored for backward
```

Keep supplied SwiGLU unchanged. A narrowly scoped saved-tensor hook around its autograd invocation can pack only the known gate/up pair. PyTorch supports packed saved objects; unpack must restore values, shape, dtype and device. Avoid in-place mutation, input-reference cycles and assumptions that nested hooks compose. [S30] [S40]

The payload must not retain the original FP32 allocation. Unpack must restore **FP32** before the existing backward, whose behavior depends on the saved dtype. Guard producer, architecture, device, dtype and layout; leave unrelated tensors on the original path.

| Maximum-shape retained object | FP32 | Encoded FP16 |
| --- | ---: | ---: |
| One `[2048,8960]` gate/up tensor | 70 MiB | 35 MiB |
| Pair per layer | 140 MiB | 70 MiB |
| All 28 pairs | 3,920 MiB | 1,960 MiB |

Nominal retained-byte reduction: **1,960 MiB**. Actual peak reduction depends on aliases, backward unpack buffers, other live allocations and sequence lengths.

Separate packing and restoration add approximately **11.484375 GiB of logical traffic per maximum-shape step** across 28 layers. Retaining the producer's original FP16 output could avoid the pack pass but requires more invasive ownership/integration work. Measure the conversion cost.

CPU exactness results are in section 10.2. GPU conversions, full training and live-memory release remain untested. GPU round-trip checks must include half subnormals and both signed zeros; the finite-value proof does not establish preservation of NaN payloads. The starter's arithmetic stays excluded; eligibility and runtime share of any original packing kernels still need to be established.

### 5.8 Boundaries that can silently change backward

**Sharing a cast:** independent projections can upcast their FP16 input gradients separately before accumulating in FP32. One shared input cast may instead accumulate branch gradients in FP16. For example, adding half values 2048 and 1 after promotion gives FP32 2049; adding in FP16 first rounds to 2048. Forward identity alone does not prove backward identity.

**Blanket compression:** FP32 residuals, adapter sums, normalized states, SiLU outputs and gradients are not generally exact promotions of FP16. M0 applies only to the verified producers.

**Frozen layers:** block 0's postattention MLP already depends on its trainable Q/V adapters; its input gradient is required. Most later frozen projections also carry gradients to earlier adapters. The frozen embedding/first preattention prefix should already have unnecessary backward work pruned. There is no broad “skip frozen backward” shortcut.

**RoPE:** a later P0 experiment can fuse FP32 rotation with the FP16 conversion consumed by SDPA. Qwen rotates contiguous halves of 128-wide heads, with 12 Q and two K heads. Scope to the actual no-cache attention call; preserve arithmetic/cast order, masks, layouts and backward. Before KV-head replication, Q+K occupy 14 MiB in FP32 versus 7 MiB in FP16; this is transient storage, not 28 layers of saved activation. [S32] [S43] [S44]

When HF explicitly repeats KV heads before SDPA, preserve the FP32 accumulation of repeated-head gradients. Moving K's cast before `repeat_kv` can instead make that accumulation occur in FP16. Place the cast after that boundary or implement equivalent FP32 accumulation. A no-cache call alone does not establish equivalence. [S44]

## 6. NF4 and the fixed T4 compiler

### 6.1 Existing fusion narrows the opportunity

bitsandbytes PR #1949 introduced fused four-bit forward GEMMs with an FP16 Turing path. Its author reports that calls above 1,536 tokens use the earlier fallback, with lower cutoffs for many shapes; QLoRA tests saw unchanged peak memory and only a small host-overhead-related timing improvement. Its inference headline is not a training forecast. [S15]

Training microbatch size is not GEMM's token dimension: four sequences of length 512 give `M=2048`. Actual shorter or padded shapes need recording. The starter pins bitsandbytes 0.50.2; its relevant source is inspected, while runtime dispatch/timing still needs a device profile. [S26] [S42]

Pinned bitsandbytes 0.50.2 computes requested base input gradients using dequantization plus matmul; it retains packed weights/metadata rather than the projection input. Soup independently records the transposed four-bit `dX` gap. [S34] [S10]

For frozen `W:[N,K]`:

```text
Y  = X W^T
dX = dY W
```

An original kernel could decode weight tiles during `dX` and avoid a dense weight buffer. Its access pattern differs from forward: transposing packed bytes without preserving logical element-to-scale mapping is incorrect. This is a meaningful gap, but fast GEMM execution is a prerequisite.

### 6.2 Preserve the actual NF4 format

NF4 is a nonuniform 16-value codebook format. Conceptually:

```text
decoded_weight[i] = NF4_codebook[packed_code[i]] × block_scale[i // block_size]
```

It is not interchangeable with uniform signed INT4 or GPTQ/AWQ packing. For nested quantization, decode scales using the actual second-level codebook, scales and offset; do not reinterpret those bytes as arbitrary IEEE FP8. The official starter confirms double quantization for this task. [S17] [S24]

The official bitsandbytes Triton dequantizer is a useful semantic reference. In its inspected layout, the high nibble maps to the first output element and the low nibble to the next. Preserve block indexing, output order, tail masks and cast order; independently implement rather than copy its kernels. [S18]

**N0:** fuse nested-scale reconstruction, NF4 unpacking and final store. Compare against the complete baseline operation, including scale expansion, rather than timing only a nibble lookup. Specialize by actual Qwen matrix shapes after correctness.

### 6.3 Upstream Triton 3.6.0 disables ordinary T4 MMA lowering

The exact `v3.6.0` tag resolves to commit:

```text
7c56a5e40f7fd928dfd5c72902d5def0097db73a
```

In `AccelerateMatmul.cpp`, the ordinary `triton::DotOp` rewrite returns failure for `computeCapability < 80` and identifies the FMA fallback. This guard occurs before MMA-version selection. The NVIDIA compiler pipeline invokes the pass. Thus, the helper's apparent ability to select an MMA version for capability 75 does not make the ordinary rewrite reachable on T4. [S19] [S20]

The earlier upstream change explicitly describes this policy. [S21]

**Implication:** do not assume that `tl.dot` produces Tensor Core instructions on the mandatory T4/Triton combination. This affects NF4 GEMM, custom head GEMM and LoRA matmul proposals. It does not imply that elementwise kernels and reductions cannot run.

Before a GEMM project, use the supplied image to compile a representative FP16 operation, inspect PTX for `mma.sync` versus scalar FMA, and measure throughput. The actual organizer build could differ from upstream; no such device experiment has been run here.

Keep the fixed environment. Explicit inline PTX is a separate, more demanding implementation and eligibility question, not permission to replace Triton or switch to a CUDA-only solution. Efficient allowed existing GEMMs plus original reductions/epilogues remain a practical starting point.

### 6.4 Cache costs and novelty

Caching decoded scales or repacking fixed NF4 codes can be investigated if the rules permit the exact change. Do not requantize the model. Charge preparation time and memory to the complete run.

Soup already contains an optional decoded-scale cache for streamed weights, so scale caching is established prior art. [S10]

Calculated costs, retaining the packed weights and assuming block size 64:

| Scope | Additional FP16 dense-weight cache | FP32 decoded-scale cache |
| --- | ---: | ---: |
| All Q/V matrices | 147 MiB | 4.59375 MiB |
| All seven decoder linear matrices across 28 layers | 2,499 MiB | 78.09375 MiB |

These exclude allocator overhead, second-level metadata and other tensors. They are not measured peak savings or confirmed legal modifications. Use true weight identity/version for caches; pointer-only identity can be wrong when storage is reused.

## 7. Experiments and acceptance decisions

All GPU experiments below are proposed. The only executed numerical work is the CPU validation in section 10.

### 7.1 Capture the remaining runtime evidence

The recipe, pins and excluded example are now known. On the T4, confirm loaded builds, flags, first-training-step dtypes, tensor strides and quantization metadata. Inspect PTX before custom GEMM work.

Capture baseline outputs and profile metric copies, casts, saved storage, synchronization and peaks. Deduplicate storage aliases and inspect lifetimes. Source-level byte counts are not peak measurements.

### 7.2 Revised experiment matrix

| ID | Proposed change | Intended benefit | Main decision |
| --- | --- | --- | --- |
| A0 | Strided accuracy and compatible entropy reduction | Avoid accuracy copy and metric intermediates | Same metrics/counts and measured complete-run gain |
| R0 | FP32 RMSNorm forward/input backward | Fewer passes and launches | Same precision boundary and gradients |
| M0 | Exact saved gate/up encoding around unchanged example | Lower retained storage | GPU exactness, original storage released, added traffic acceptable |
| N0 | Nested-scale decode and NF4 unpack fusion | Less decoding and launch overhead | Complete format/cast equivalence and speed |
| B0 | LoRA scale/cast/add after existing GEMMs | Lower traffic and launch count | Dropout, ownership, dtypes and gradients preserved |
| A1 | Bounded head with compact statistics | Lower logits-related live storage | All consumers/AMP correct; time admissible |
| B1 / N1 / N2 | Adapter expansion / NF4 backward / forward GEMM | More extensive fusion | Competitive generated code on fixed T4 compiler |
| P0 | RoPE plus consumed FP16 conversion | Avoid transient output/cast | Exact attention and backward boundary |

Keep supplied SwiGLU separately identified as uncredited. A0/R0/M0 lead the new hypotheses; none is a measured winner or guaranteed to meet the 5% requirement. Test useful combined work across every run.

### 7.3 Correctness checks that can reveal real failures

- **Vocabulary path:** independent attention/label masks, causal shifting, last positions, empty supervised rows, argmax ties, partial vocabulary tiles, actual denominators and logits dtypes.
- **Backward:** incoming gradient scales, AMP underflow/overflow behavior, A/B/input gradients, noncontiguous gradients and accumulation across branches.
- **LoRA:** task dropout masks and checkpoint replay; nonzero adapter weights or a warmed state. With zero-initialized B, some adapter gradients vanish and can conceal errors. Preserve zero gradients versus absent (`None`) gradients because the optimizer may handle them differently.
- **NF4:** nibble order, second-level scale codebook/offset, logical transpose indexing, tails, cast sequence and input gradients through unadapted frozen layers.
- **Integration:** exact row/token bookkeeping, required metric keys/cadence, serialization and normal evaluation behavior.

Do not substitute a small local check for the required Soup tests or full training trajectory. Run the relevant gates when the implementation is available; avoid unrelated broad testing.

### 7.4 Comparison and keep/reject rule

| Variant | Whole-command time | Sampled peak memory | Quality/logs | Original-kernel share |
| --- | --- | --- | --- | --- |
| Baseline | Not measured here | Not measured here | Reference to establish | N/A |
| A0 / B0 / N0 individually | To measure | To measure | Check | Measure |
| A1, if warranted | To measure | To measure | Check | Measure |
| Best valid combination | To measure | To measure | Check | Measure every run |
| Additional GEMM variants | Conditional | Conditional | Check | Measure |

Retain a candidate only when it remains correct, eligible and beneficial under official measurements. A memory improvement can justify modest extra work only within admission limits and the scoring tradeoff.

For delegated kernel work, specify operation, GPU, dtype, exact shapes, required gradients, integration point and acceptance evidence. Hugging Face's agent workflow reinforces the need for precise tasks and correctness/performance iteration. [S9]

Reserve time for a clean notebook run and required submission artifacts. Patches also run before regression tests using fake modules; guard real-model/CUDA assumptions and preserve fallback behavior. [S1] [S24]

## 8. Performance expectations and scoring

### 8.1 Illustrative whole-command model

The brief reports approximately **80 seconds total**, including **60 seconds training**. For illustration, treat 20 seconds as unaffected overhead and approximate GPU profile shares as training-time shares. [S1]

```text
T_new = 20 + 60 * ((1 - f) + f / k)
whole-command speedup = 80 / T_new

f = fraction accelerated
k = local speed multiplier
```

| Hypothetical intervention | Illustrative time | Illustrative speedup |
| --- | ---: | ---: |
| Attention's 5% share becomes 10x faster | 77.3 s | 1.035x |
| The entire 44% elementwise/cast share becomes 2x faster | 66.8 s | 1.198x |

Neither row is a prediction. GPU time is not identical to elapsed time; overhead can change; the 44% share covers many operations. The illustration explains why a spectacular attention microbenchmark may remain below the 1.05x speed-scoring threshold.

### 8.2 Score model after measurement

```text
s = baseline time / solution time
m = baseline peak / solution peak

speed points  = 35 * ln(s) / ln(best admitted s)
memory points = 30 * ln(m) / ln(best admitted m)
```

Apply the brief's threshold and floor rules: ratios below 1.05 receive zero for that metric, and points do not go below zero. Eligibility contributes 20 points and README/reproducibility contributes up to 15. Relative scores depend on other admitted teams, so the formulas do not establish our rank. [S1]

Compilation/loading count, and caches start empty each run. Limit shape variants and autotuning because only 64 steps amortize them. A saving of 10 ms per step yields only 0.64 seconds of total budget for added one-time costs before the runtime benefit vanishes. This is arithmetic, not measured compilation cost. [S27]

Memory-oriented M0 may add time and still be useful within the admission limit. Keep speed and sampled peak as separate outcomes.

### 8.3 The participation gate can change which combination is viable

Suppose an eligible custom kernel replaces baseline training fraction `p` and speeds it up by `k`, with all other work unchanged. Its new runtime share is:

```text
custom_share = (p / k) / (1 − p + p / k)
```

For `p=7%` and `k=2`, the new share is about 3.63%, below the required 5%. This is an illustrative calculation, not a claim that the entire reported vocabulary category maps to one implementation.

The solution may therefore need several useful original kernels. Artificially slowing a kernel or adding unused computation is not a legitimate way to satisfy the rule. Measure useful eligible work after optimization. The denominator is synchronized step wall time, and the harness warns when the lowest share is between 5% and 6%. Seek measured margin across legitimate batches without adding artificial work. [S28] [S29]

## 9. Resolved questions and remaining decisions

| Item | Current evidence | Remaining action |
| --- | --- | --- |
| Starter/harness | Immutable commit inspected | Run official files unchanged |
| Excluded operation | Supplied SwiGLU, including rewrites | Separate it from original credit |
| Configuration/pins | Exact recipe and installation versions known | Confirm supplied runtime/build |
| Compiler | Upstream 3.6.0 guard verified | Inspect PTX and throughput on T4 |
| Metric allocation | Accuracy copy and entropy chunking identified | Measure lifetime and A0 benefit |
| FP32 activation provenance | Exact producer chain inspected | Confirm real boundary dtypes |
| M0 representation | Exhaustive finite CPU round trip and sampled derivatives pass | GPU/hook/lifetime/time/peak checks |
| Loss mathematics | CPU cases and finite differences pass | AMP and full training validation |
| T4 profile and results | Organizer estimates only | Reproduce baseline and candidates |
| Original-kernel share | Unmeasured | Measure every required run |
| Official correctness/ranking | Unmeasured | Required tests, trajectories and evaluation |

Start A0/R0/M0 after baseline capture. Advance A1 or custom GEMMs when profile/compiler evidence justifies them. Preserve dropout, mode flags, initialization rounding, optimizer and logging. Compare complete cold runs; nominal bytes and CPU equivalence are not GPU performance claims.

## 10. CPU mathematical validation performed

### 10.1 Online vocabulary reductions

This was an original NumPy experiment, not a contest kernel or benchmark. Python 3.12.14 and NumPy 2.3.5 were used in the research environment; they are not asserted to be the contest environment.

- **Seeds:** input 20261030; labels 20261130; gradient check 20261031. Masks were constructed deterministically.
- **Coverage:** 54 dtype/chunk combinations in float64 and float32, including the actual 151,936-entry vocabulary on small token batches.
- **Cases:** random logits, large positive/negative shared offsets, wide dynamic range, tied maxima, vocabulary tails, ignored labels and differing attention/supervised masks.
- **Reference:** dense stable reductions evaluated in float64 on the same already-rounded input values. Input quantization error is separate from reduction error.
- **Gradient check:** tiny `[2,3,5]` logits, vocabulary chunk 2, central differences with step `1e-6`, three supervised positions.

| Fixture | Shape | Vocabulary chunk sizes |
| --- | --- | --- |
| Random small | `[2,3,17]` | 1, 3, 7, 64 |
| Random medium | `[2,4,1009]` | 1, 7, 64, 257 |
| Qwen vocabulary width | `[2,4,151936]` | 127, 1024, 4096, 65536 |
| Positive / negative common offset | `[2,3,257]` each | 1, 7, 64 |
| Ties and uniform distributions | `[2,3,65]` | 1, 3, 16, 64 |
| Finite extreme-score tails | `[2,3,131]` | 1, 3, 16, 128 |
| Single class | `[2,3,1]` | 1 |

Pass thresholds for the checked stable reductions were `1e-9` in float64 and `5e-5` in float32.

| Check | Maximum absolute error |
| --- | ---: |
| Online entropy, float64 | 3.5527e-15 |
| Online entropy, float32 vs float64 same-input reference | 2.3044e-6 |
| Stable CE, float64 | 1.7764e-15 |
| Stable CE, float32 vs float64 same-input reference | 3.9127e-6 |
| Analytical CE gradient vs float64 central differences | 1.4639e-10 |
| Ignored-position gradient | Exactly zero in the checked case |

All checks of the merged entropy, stable CE, argmax and masks passed, including first-index ties, exact ignored losses and mask/count separation. The less stable CE ordering discussed next exposed a numerical failure and is not included in that pass claim.

**Finding that changed the design:** for common offsets of approximately ±1e6, FP32 evaluation of `(M + log(s)) − target` had maximum CE error about 0.0188867. The rearranged `log(s) + (M − target)` reduced the maximum across the test suite to 3.9127e-6 against the same-input float64 reference.

The experiment does not recover detail already lost when logits are converted to FP32. It also does not validate causal alignment in Soup: the synthetic targets were already aligned. No GPU, FP16, AMP, head GEMM, optimizer, logger, performance or official eligibility claim follows from these checks.

This section records the mathematical specification, fixture shapes and observed results. Exact fixture reproduction additionally requires the original distribution and mask-generation code. Research scratch files were `validate_online_vocab_reduction.py` and `online_vocab_reduction_results.json`.

### 10.2 Exact encoding of half-origin saved values

An original NumPy 2.3.5 experiment enumerated all 65,536 binary16 bit patterns. All **63,488 finite values** survived FP16 → FP32 → FP16 → FP32 with bitwise-identical FP32 results. Positive and negative zero were preserved. Two infinities and 2,046 NaN patterns were excluded from that finite-value assertion.

A `[19,37]` sample used half-origin gate/up and general FP32 upstream gradients. FP32 SwiGLU forward, gate gradient and up gradient were bitwise identical after encoding/restoration, with maximum absolute difference zero.

Counterexamples show the scope:

```text
FP32 residual: 1 + 2^-11       = 1.00048828125
FP32 LoRA sum: 1 + 2 * 2^-12   = 1.00048828125
Either after FP16 storage and FP32 restoration = 1.0
```

M0 therefore depends on the producer, not merely dtype or shape. Research files: `saved_fp16_encoding_check.py` and `saved_fp16_encoding_check.json`. This establishes CPU representation and sampled derivatives, not GPU conversion behavior, live-memory release, performance, training quality or eligibility.

## 11. Pinned-source provenance

### 11.1 Soup

| Field | Value |
| --- | --- |
| Package | `soup-cli==0.77.0` |
| Artifact | `soup_cli-0.77.0-py3-none-any.whl` |
| Size | 2,891,737 bytes |
| SHA-256 | `f1346e79cecb451211b714df06759b8b7dc9a6fd2fc086883297c35d1fe60ebb` |
| Verification | Compared downloaded wheel with PyPI JSON digest |
| Inspection | Static extraction/read; no import, installation or execution |

The [package artifact][S10] and [version metadata](https://pypi.org/pypi/soup-cli/0.77.0/json) allow independent reconstruction. Key files are `utils/stream_loss_chunk.py`, `trainer/rewind_hf.py`, `trainer/sft.py`, `utils/fast_lora.py`, `utils/fast_lora_qkv.py`, `utils/peft_wiring.py`, `utils/layer_stream_runtime.py`, `config/schema.py` and distribution `METADATA`.

### 11.2 Triton

| Field | Value |
| --- | --- |
| Tag | `v3.6.0` |
| Commit | `7c56a5e40f7fd928dfd5c72902d5def0097db73a` |
| `AccelerateMatmul.cpp` SHA-256 | `bc73e8363a12c3cb050dc330bfe2c10bc3f3696cd245727f46405952edbeae68` |
| NVIDIA `compiler.py` SHA-256 | `eea55698bab00c726b3d1d495b390b72a63aaa8d167781a65ea5212d394d76f6` |
| Inspection | Tag API and two immutable source files; no compiler/GPU execution |

The capability guard is in the `DotOp` rewrite before MMA selection; the compiler file shows the pass is invoked. These are upstream-source findings, not a measured description of a potentially patched organizer image. [S19] [S20]

### 11.3 Official starter and pinned dependency identities

The starter was inspected at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb` through read-only GitHub API/raw-file access. README, task YAML, notebook cell, example, run/check/score harness, evaluator and common helpers were read; no downloaded code was executed.

| Release/file | Git blob SHA |
| --- | --- |
| Transformers 5.16.1 Qwen2 | `b4fed98fafae8e4ee39be3b8cd766b1c85b079cd` |
| Transformers 5.16.1 loss | `92c0bd0637e58ddf33275abe68e16d6b221bb729` |
| TRL 0.29.1 trainer | `3692b454a60beaf0dd85a162ff7369db3ff679d4` |
| TRL 0.29.1 utilities | `ca653ce681566207f54a03ccfefafad074fd9fcf` |
| PEFT 0.20.0 preparation | `ef435add45da04067c152219ed74926b622a26b2` |
| bitsandbytes 0.50.2 modules | `ebc0b09439c4620ef015b18b242328aa65f14c18` |
| bitsandbytes 0.50.2 autograd | `8a069bd101a089afc59b7d4d3d48e5dd399bc967` |
| Accelerate 1.14.0 utilities | `869b500f3452f809753586a435c6ea8d910b3f28` |

Release-tag links follow. Local manifests record additional retrieved files and hashes. These identities support the source audit, not a reproduced training environment.

## 12. Source ledger

Sources were checked on 30 October 2026. Preserve the version, hardware and workload attached to each result. Rolling `main` documentation describes current upstream behavior, not an installed dependency pin.

| ID | Primary source | Supports |
| --- | --- | --- |
| [S1] | Supplied challenge technical specification | Fixed workload, profile, restrictions, correctness, measurement and score; Russian pages 1–7, English 16–22 |
| [S2] | Hugging Face: Custom Kernels for All from Codex and Claude | Qwen3-8B RMSNorm microbenchmark on H100/BF16 |
| [S3] | Unsloth exact-model card | Qwen2.5-1.5B 4-bit model and T4 notebook links |
| [S4] | Liger Kernel README | Qwen2.5 operations and chunked linear CE |
| [S5] | Hugging Face RMSNorm kernel card | Forward/backward interface |
| [S6] | Apple AIML Research CCE repository | Memory-saving method, hardware requirement and filtering modes |
| [S7] | Official Qwen configuration | Architecture and arithmetic inputs |
| [S8] | Qwen3.7 article | Author-reported SGLang attention experiment on T-Head hardware |
| [S9] | HF kernel-agent guide | Precise task definition and iterative validation |
| [S10] | Hash-verified Soup 0.77.0 wheel | Pinned Python implementation and dependency metadata |
| [S11] | Soup v0.77.0 release | Chunked head/loss and optional scale-cache features |
| [S12] | PEFT 4-bit LoRA source, current main | Clone, adapter linears, casts and output addition |
| [S13] | PEFT dtype troubleshooting | Default adapter precision and AMP considerations |
| [S14] | Axolotl LoRA API | Grouped projection and shared-dropout semantics |
| [S15] | bitsandbytes PR #1949 | Turing fused forward path, dispatch scope and author's QLoRA observations |
| [S16] | bitsandbytes autograd source, current main | Dense dequantization in four-bit input backward |
| [S17] | bitsandbytes Linear4bit/LinearNF4 reference | NF4 codebook semantics |
| [S18] | bitsandbytes Triton four-bit source, current main | Packed layout and dequantization operations |
| [S19] | Triton immutable AccelerateMatmul.cpp | Exact v3.6.0 capability guard |
| [S20] | Triton immutable NVIDIA compiler.py | Compiler pass invocation |
| [S21] | Triton PR #5066 | Upstream reason/history of pre-Ampere FMA fallback |
| [S22] | PyTorch 2.11 AMP examples | Custom autograd and autocast handling |
| [S23] | Run LoRA Run, ACL Industry 2025 | Shape-dependent LoRA training computation orders |
| [S24] | Official starter README | Baseline and supplied example |
| [S25] | Official task YAML | Fixed training settings |
| [S26] | Official notebook | Installation pins |
| [S27] | Official run harness | Empty per-run compiler caches |
| [S28] | Official check harness | Gap formulas and lowest-run eligibility |
| [S29] | Official training probe | CUDA-event numerator and step-time denominator |
| [S30] | Supplied SwiGLU | Saved pair and excluded arithmetic |
| [S31] | PEFT 0.20.0 preparation | Freezing, promotions, conditional gradient hooks |
| [S32] | Qwen2, Transformers 5.16.1 | Norm/MLP/residual/RoPE/head |
| [S33] | bitsandbytes 0.50.2 modules | Input/compute/output dtypes |
| [S34] | bitsandbytes 0.50.2 autograd | Requested input gradients and saved state |
| [S35] | TRL 0.29.1 SFTTrainer | Accuracy copy, masks, interim adapter cast |
| [S36] | TRL 0.29.1 utilities | 128-row entropy chunks |
| [S37] | Transformers 5.16.1 loss | FP32 scores, label shift, denominator |
| [S38] | Accelerate 1.14.0 Accelerator | AMP forward wrapping |
| [S39] | Accelerate 1.14.0 utilities | FP32 output conversion |
| [S40] | PyTorch 2.11 autograd | Saved-tensor hook contract |
| [S41] | PyTorch 2.11 AMP reference | FP16 range and GradScaler default |
| [S42] | bitsandbytes 0.50.2 CUDA backend | FP16 producer and dispatch |
| [S43] | PyTorch 2.11 autocast source | SDPA lower-precision registration |
| [S44] | Transformers 5.16.1 SDPA | Attention layout/interface |

[S1]: https://chatgpt.com/library?file_id=libfile_0f170950ca54819186db99d6c3bad927&account_id=personal "Supplied Soup Kernel Challenge brief"
[S2]: https://huggingface.co/blog/custom-cuda-kernels-agent-skills "Custom Kernels for All from Codex and Claude"
[S3]: https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit "Unsloth Qwen2.5-1.5B-Instruct-bnb-4bit"
[S4]: https://github.com/linkedin/Liger-Kernel/blob/main/README.md "Liger Kernel README"
[S5]: https://huggingface.co/kernels-community/rmsnorm "HF RMSNorm kernel"
[S6]: https://github.com/apple-aiml-research/ml-cross-entropy "Cut Cross-Entropy implementation"
[S7]: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json "Official Qwen configuration"
[S8]: https://qwen.ai/blog?id=qwen3.7&locale=en "Qwen3.7 blog"
[S9]: https://huggingface.co/docs/kernels/builder/agents-guide "Develop kernels with agents"
[S10]: https://files.pythonhosted.org/packages/e8/ba/30337aeb7d1d565789699caa1164c2dd62dde104715caf6f2a317c56ca45/soup_cli-0.77.0-py3-none-any.whl "Verified Soup 0.77.0 wheel"
[S11]: https://example.invalid/redacted-github "Soup v0.77.0 release"
[S12]: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/bnb.py "PEFT bitsandbytes LoRA implementation"
[S13]: https://huggingface.co/docs/peft/developer_guides/troubleshooting "PEFT dtype troubleshooting"
[S14]: https://docs.axolotl.ai/docs/api/kernels.lora.html "Axolotl LoRA API"
[S15]: https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1949 "Fused four-bit forward GEMM"
[S16]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py "bitsandbytes autograd"
[S17]: https://huggingface.co/docs/bitsandbytes/main/reference/nn/linear4bit "bitsandbytes four-bit layers"
[S18]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/backends/triton/kernels_4bit.py "bitsandbytes Triton dequantization"
[S19]: https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp "Triton 3.6.0 immutable matmul pass"
[S20]: https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/third_party/nvidia/backend/compiler.py "Triton 3.6.0 immutable NVIDIA pipeline"
[S21]: https://github.com/triton-lang/triton/pull/5066 "Pre-Ampere FMA fallback"
[S22]: https://docs.pytorch.org/docs/2.11/notes/amp_examples.html "PyTorch 2.11 AMP"
[S23]: https://aclanthology.org/2025.acl-industry.15/ "Run LoRA Run"
[S24]: https://example.invalid/redacted-github "Official starter"
[S25]: https://example.invalid/redacted-github "Fixed task"
[S26]: https://example.invalid/redacted-github "Pinned install"
[S27]: https://example.invalid/redacted-github "Run harness"
[S28]: https://example.invalid/redacted-github "Checks"
[S29]: https://example.invalid/redacted-github "Training probe"
[S30]: https://example.invalid/redacted-github "Supplied example"
[S31]: https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py "Pinned preparation"
[S32]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py "Pinned Qwen2"
[S33]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py "Pinned Linear4bit"
[S34]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py "Pinned autograd"
[S35]: https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py "Pinned SFTTrainer"
[S36]: https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/utils.py "Pinned entropy"
[S37]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py "Pinned CE"
[S38]: https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py "Pinned AMP wrapping"
[S39]: https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py "Pinned output conversion"
[S40]: https://docs.pytorch.org/docs/2.11/autograd.html "Saved-tensor hooks"
[S41]: https://docs.pytorch.org/docs/2.11/amp.html "FP16 range and gradient scaling"
[S42]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py "Pinned CUDA backend"
[S43]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.h "Pinned autocast registrations"
[S44]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/integrations/sdpa_attention.py "Pinned SDPA"
