# Qwen Kernel Optimization: Research and Experiment Review

- **Project:** Decentrathon 6.0 / Soup Kernel Challenge
- **Research date:** 30 October 2026
- **Edition:** Seventh pass: compact CE before head replay, conditional T4 attention dispatch, and shared input storage with separate gradient edges
- **Status:** Starter/harness and exact source versions inspected; twelve original CPU studies completed. Contest kernels, real PyTorch integration, T4 benchmarks and official eligibility remain unverified.
- **Purpose:** A self-contained Markdown reference for review and continuation.

## Session handoff

**Objective:** improve complete training runtime and sampled peak GPU memory while preserving the fixed recipe, gradients, adapter quality and required outputs.

**Fixed recipe:** Qwen2.5-1.5B-Instruct, NF4 double quantization, Q/V LoRA rank 16/alpha 32, dropout 0.05, batch 4, sequence length up to 512, 64 steps, checkpointing off, one T4 with FP16 autocast. Final trainable LoRA storage is FP32. Starter commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb` pins the inspected stack. The supplied SwiGLU and its rewrites remain excluded from original-kernel credit. [S1] [S24] [S25] [S26] [S30]

**Leading new experiment — A0b:** keep the full FP16 head logits and the original head GEMMs, but replace eager CE's full FP32 log-probability and backward buffers with compact row statistics. This avoids A1's additional projection replay. A0b is a proposed original kernel design, not an implemented or measured result.

**Why A0 alone may not lower the peak much:** its maximum accuracy copy is 1,184.682 MiB, but source inspection predicts a separate native CE-backward phase with three coexisting 1,187 MiB FP32 tensors: **3,561 MiB**. A0 leaves that phase unchanged. A0 + A0b has a smaller calculated head-local budget; with ordinary Rewind still enabled, its listed largest phase is about **2,076.670 MiB**. These are phase-specific payload calculations, excluding body state, head weight, workspaces and allocator reservations; they are not scored-memory forecasts. See section 4.3.4. [S35] [S55] [S74] [S75] [S89] [S90]

**Attention now has a concrete dispatch question:** a maskless HF call can pass 12 Q heads and two K/V heads with GQA enabled. Pinned PyTorch rejects that head mismatch for its memory-efficient backend, while T4 is below the Flash/cuDNN hardware gates. A masked call explicitly repeats K/V and can qualify for memory-efficient attention. Record the actual backend per batch before changing masks or RoPE packing. This is conditional source analysis; the installed build has not been profiled. [S44] [S63] [S94] [S95]

**New cast proposal — I0:** share a detached FP16 input payload across Q/K/V or gate/up, while retaining a distinct backward edge for each projection. Separate promotion of each incoming gradient preserves the intended FP32 fan-in boundary. The maximum-shape calculation removes 84 forward conversions and **1,512 MiB of logical traffic** across 28 blocks. It does not remove a retained activation pool. Native custom-Function aliasing, accumulation order and overhead remain untested. [S33] [S34] [S92]

**Other leading work:** A0 strided metrics, R0 FP32 RMSNorm and M1 reuse of the producer's half gate/up storage remain useful independent experiments. M1's nominal retained-pair difference is 1,960 MiB before reconstruction. B0 and N0 have source-backed integration proposals and original CPU rounding checks. Keep each optimization's ownership and arithmetic boundaries explicit.

**Validation status:** twelve original NumPy/Python studies. The new compact-CE study performs 72 gradient comparisons; reconstruction from identical statistics matches its CPU oracle, while a different reduction tree changes 20 half-gradient entries. The I0 study passes eight manual value/gradient/ownership cases and exposes why prematurely casting LoRA's dropout input is unsafe. PyTorch is not installed. No native autograd, model training, CUDA kernel or GPU benchmark has been executed.

**Compiler and measurement constraints:** upstream Triton 3.6.0 disables ordinary Tensor Core MMA lowering below compute capability 8.0. Every scored run has empty compiler caches. The harness uses median wall time but the maximum sampled memory across each side's runs. Original-kernel participation is measured from step 3 onward and the lowest solution run must pass. [S19] [S20] [S27] [S28] [S29]

**Next decisive evidence:** baseline phase lifetimes and attention dispatch on the real T4, then isolated A0/R0, A0b paired with A0, and native M1 ownership checks. Compare A1 only if remaining head storage justifies replay. I0 and the conditional attention experiments follow their specific correctness gates. No candidate has established contest eligibility or a speed/memory win.

## 1. Recommendation and rationale

Public implementations establish that the relevant Qwen operations are established optimization targets. Unsloth publishes the exact model family and links T4 training notebooks; Liger lists Qwen2.5 support. The useful research question is which remaining work can be independently implemented, integrated and measured under this fixed training recipe. [S3] [S4]

The organizers provide this approximate GPU step profile: [S1]

| Work category | Reported share | Revised interpretation |
| --- | ---: | --- |
| Elementwise operations and dtype casts | 44% | Primary speed target; obtain an operation-level breakdown before choosing kernels |
| Matrix multiplication | 36% | Preserve efficient existing GEMMs initially; custom Triton GEMM has a T4 lowering risk |
| Vocabulary softmax | 7% | Share reductions across loss and metrics; consider bounded head storage |
| Attention | 5% | Verify mask-dependent backend first; the aggregate share can hide a math-fallback branch |
| NF4 unpacking | 5% | Fuse nested-scale decode and unpacking; distinguish forward from backward |

These are approximate GPU-time shares, not a precise end-to-end decomposition. The 44% category is not the cost of any one normalization, activation or LoRA operation.

**Working arrangement:** profile CE lifetimes and attention dispatch first. Establish A0 and R0 independently; then test A0b with A0 before paying A1's head-replay cost. Validate M1's native storage ownership separately before combining it with the head changes. I0 needs a native separate-edge check before performance work. Evaluate N0/B0 and conditional attention changes when attribution justifies them. Combine only candidates that remain correct, eligible and beneficial.

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
- **Order:** baseline, solution, solution, baseline. The harness aggregates wall time with `statistics.median`; for two runs this equals their arithmetic mean. It aggregates GPU peak memory with the **maximum**, not the average. Training-loop time is also a median. [S27]
- **First two steps:** excluded only from original-kernel participation. Its denominator is synchronized step wall time from step 3, not total GPU kernel time; the lowest solution run must pass. They still count in complete-command timing. [S28] [S29]
- **Cache behavior:** every run removes/recreates its run directory and uses new Triton/Inductor cache directories. Compilation is charged again in every measured run. [S27]
- **Memory:** whole-run GPU peak sampled through `nvidia-smi` every 0.1 seconds, with the worse peak retained across each side's runs. Unused caching-allocator reservations remain visible to this tool. Lower tensor payload or PyTorch allocated memory alone does not establish a lower scored peak. Keep the prescribed allocator configuration. [S1] [S27] [S100]

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

### 3.3 Sixth-pass public evidence: useful algorithms, different contracts

The Hub now provides a directly relevant `trl-lib/fused-linear-ce` kernel, beyond the earlier RMSNorm examples. Its v1 card documents compact accuracy and entropy, both summed over non-ignored targets. This differs from the challenge's attention-mask entropy, which includes prompt and final valid positions. Its broad CUDA hardware catalog is not a T4 benchmark. [S76]

The inspected v1 tree identifies commit `00e29bdf4e142fafa8f558be414c7a9f344d61d0`. Source uses FP32 projection accumulators directly in softmax, rather than the challenge's explicit FP16 score rounding. Backward applies upstream scaling before storing low-precision score gradients, uses a 256 MiB workspace budget, and accumulates low-precision vocabulary-chunk matmul results into FP32 hidden-gradient storage. That destination does not undo rounding of each partial result. Its configuration comment names an H100 tuning workload. These are design references, not a verified compatible implementation. [S77]

| Additional reference | Source-backed finding | Consequence here |
| --- | --- | --- |
| TRL PR 5575, merged 27 April 2026 [S78] [S79] | Published synthetic head benchmarks report 1.74–1.87× the baseline forward/backward time for the selected memory-priority configurations. Merge source rejects PEFT. | Later upstream work, absent from fixed TRL 0.29.1. The benchmark uses BF16 storage, a trainable classifier and an FP32 projection reference without autocast; it does not predict this T4 workload. |
| Pinned CCE threshold/backward [S80] [S81] | Default filtering uses dtype epsilon/32 on unscaled per-logit derivatives, before incoming loss scaling. | Many small terms can collectively matter to the required hidden gradient. Independent A1 should retain every vocabulary contribution. |
| Pinned CCE presets [S82] | `cce_kahan_full_c` still filters hidden gradients; `cce_exact` removes filtering and enables FP32 accumulation. | A full classifier-gradient option does not address the active gradient when the classifier is frozen. Unfiltered does not promise bitwise cuBLAS equivalence. |
| Pinned Liger fused loss [S83] | Its default shape heuristic gives 32 token rows and 64 chunks for this maximum shape; one half-logit chunk is 9.2734375 MiB. | This calculation exposes a launch/weight-read tradeoff. Do not adopt a chunk size without a T4 comparison. |

The CCE mechanism has a simple analytic counterexample. For zero hidden state and identical nonzero classifier rows, all scores are zero and the exact hidden gradient cancels. With `V=151936`, each non-target probability is about `6.5817e-6`, below the default FP16 threshold `3.0517578125e-5`. If only an illustrative target-containing vocabulary tile of 128 entries survives, its coefficient is `128/V - 1 = -0.99915754`, rather than zero. This is our construction, not an execution of CCE, its autotuner or real checkpoint weights. [S80] [S81]

All of these implementations remain research references. No ready implementation, copied kernel, package upgrade or built-in loss-mode change is proposed for the submission.

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


#### 4.3.1 Install the replacement before the metric allocation

Pinned TRL inherits `SFTTrainer → BaseTrainer → transformers.Trainer`; its BaseTrainer does not override `compute_loss`. A narrow replacement can therefore delegate to the existing Transformers loss implementation exactly once, then calculate metrics independently. Forward `num_items_in_batch` unchanged, preserve `model_accepts_loss_kwargs`, and return the same differentiable loss and model outputs. No custom backward is needed for A0. [S35] [S45] [S46]

There is an installation-order trap: Soup imports `SFTTrainer` at the start of setup, before calling its model-loading helper. Replacing only the exported class after model loading is too late for the locally captured class. Install a subclass during `solution.apply()` before setup, or idempotently patch the original class method with a strict applicability guard. Preserve Soup's Rewind subclass ordering. [S10]

The initial supported path should be the actual single-process CUDA training recipe: ordinary NLL, rank-two labels and attention mask, no custom loss function, label smoother, Liger, auxiliary loss, virtual prompt tokens, distributed sequence/context/tensor parallelism, or pre-shifted labels. Evaluation, prediction, fake test objects and other configurations use the original method.

For an applicable call, retain labels before delegation, consume the prediction-only marker, preserve `use_cache=False`, delegate once with `return_outputs=True`, then calculate detached metrics and honor the caller's requested return form. The existing loss path retains its denominator and AMP behavior. [S35] [S45]

**Two different fallback points are necessary.** Before model execution, delegate to the saved original SFT method. If a returned dtype, shape or stride is unsupported after execution, use an ordinary metric helper on the already-produced outputs. Calling the original loss method at that point would run the model and dropout twice. Do not conceal a failed CUDA launch by assuming the device context is safe for fallback.

#### 4.3.2 A concrete two-stage metric reduction

This is an original design proposal, not implemented contest code or a measured launch configuration.

Read `logits[B,T,V]` through its actual element strides. For a flattened token number `n`, use `b=n//T` and `t=n%T`, with vocabulary addresses `b*s_B + t*s_T + v*s_V`. There is no need to copy or reshape the shifted vocabulary tensor.

**Stage 1:** one program per token and vocabulary tile. Start development with tile width 4096 and four warps. For `V=151936` there are 38 tiles; the final tile contains 384 valid entries. Each program writes FP32 `m,s,q` from section 4.5 and an INT32 index of the first tile maximum. Explicitly choose the lowest vocabulary index among equal maxima.

**Stage 2:** one program per token merges the 38 partials, with a masked 64-lane reduction as an initial layout. Let `M=max(m_p)` and `a_p=exp(m_p-M)`. Calculate:

```text
S = sum(a_p * s_p)
Q = sum(a_p * (q_p + (m_p - M) * s_p))
entropy = log(S) - Q / S
prediction = minimum i_p among partials with m_p == M
optional token CE = log(S) + (M - target_logit)
```

Mask both vocabulary tails and partial-record tails before intermediate products. In particular, avoid evaluating zero times negative infinity as a contribution to `q`. The target for time `t` is `labels[b,t+1]`; the final time position has no supervised next-token target.

| Proposed maximum-shape workspace | Bytes | Binary size |
| --- | ---: | ---: |
| Three FP32 statistics plus INT32 index for `[2048,38]` | 1,245,184 | 1.1875 MiB |
| Per-token FP32 entropy and INT32 prediction | 16,384 | 16 KiB |
| Optional per-token FP32 CE | 8,192 | 8 KiB |

Use ordinary small tensor reductions initially to turn those outputs into batch/row summaries. A third custom kernel is not justified solely by processing a few thousand scalar values. This design still retains the original full logits and differentiable CE allocations; it removes the extra accuracy copy and changes metric intermediates.

The arithmetic above assumes finite logits. Metadata guards do not prove finiteness. An implementation must preserve reference behavior for NaN/infinity or detect unsupported values and compute fallback metrics from the same outputs. Extra status storage is not included in the 16-byte partial-record calculation. The CPU check in section 10.3 does not settle nonfinite GPU semantics.

#### 4.3.3 Preserve masks, logs and the Rewind boundary

For finite logits, metric arithmetic is needed when a token's attention mask is nonzero **or** its shifted label is supervised. This union preserves prompt entropy, the final valid position's entropy, and accuracy when the two masks differ. Accuracy uses exact integer correct/supervised counts and returns zero for no supervised targets. The token counter includes attention-mask positions. An empty attention denominator follows the baseline undefined `0/0` result; silently replacing it with zero changes the contract. [S35]

Skipping fully masked rows needs a separate nonfinite qualification: the baseline computes their entropy before multiplying by the mask, so a nonfinite result can propagate through `0 * NaN`. Any integrated nonfinite detector must visit those rows too, before skipping metric arithmetic, or equivalent nonfinite behavior remains unsupported. This is an inference from the inspected operation order, not an executed GPU check. [S35]

Append one entropy and accuracy value per call; update the cumulative training token count with a Python integer; retain the same metric keys, list averaging, clearing and per-step availability. Keep `SFTTrainer.log` unchanged. The initial single-process guard also avoids changing distributed semantics: TRL averages per-rank entropy means, whereas accuracy combines correct/total counts. [S35]

One compact CPU transfer of already-computed scalar results can replace several individual scalar transfers. The current per-batch counts are at most 2048 and exactly representable in FP32; recover Python integers and maintain the cumulative count separately. This is a synchronization-reduction hypothesis, not a measured time saving.

Leave Rewind unchanged for the first A0 experiment. A later extension can reuse the same partial statistics for detached per-token CE and row means. Soup gives existing `outputs.logits` priority, so setting `_soup_chunked_step` alone will not redirect it. A scoped handoff to `rewind_hf.row_losses` must match the exact logits/labels objects, preserve the original sampler FIFO and writer, fall back on mismatch, and clear state in `finally`. No persistent pointer cache is required. [S10]

#### 4.3.4 A0b: compact CE state while keeping the original head

This seventh-pass proposal is an intermediate experiment between metrics-only A0 and the replayed head in A1. It preserves the existing full FP16 logits and both baseline head GEMMs. It changes the loss's saved state and backward, with no additional head projection.

**The native backward floor.** For ordinary hard targets with no smoothing or class weights, pinned CE decomposes to log-softmax plus NLL. NLL and log-softmax both retain references to the same FP32 log-probability storage; count that storage once. NLL backward then produces a full FP32 gradient tensor, zeroing it and scattering the target entries. Log-softmax backward allocates a distinct full FP32 result while consuming the saved log-probabilities and incoming gradient. Its three vocabulary-shaped tensors coexist. [S89] [S90] [S55] [S74] [S75]

At `B=4, T=512, V=151936`, this is `3 × 1187 = 3561 MiB`. The baseline accuracy phase's saved log-probabilities, returned logits and shifted copy total `1187 + 1187 + 1184.681640625 = 3558.681640625 MiB`. Removing the accuracy copy therefore leaves a CE-backward payload phase of almost the same size. These are source-derived lifetimes and arithmetic, not observed allocator peaks.

The outward FP32 logits are **not automatically a fourth backward tensor**. In the ordinary loss-only training call, Rewind requests outputs temporarily, records row losses and returns only the loss; the Trainer calls backward afterward. Cast derivatives retain metadata rather than the original forward tensor values. External hooks or instrumentation can prolong ownership, so validate the actual run, but do not charge ordinary backward for outputs that have already expired. [S10] [S45] [S55]

**A narrow installation point.** Qwen2ForCausalLM calls `self.loss_function(...)` with logits before the Transformers loss utility applies `.float()`. The pinned model base class exposes a per-instance loss property backed by `_loss_function` and a setter. An A0b installer can identify the actual Qwen causal-LM instance beneath PEFT, save its original callable and install one guarded wrapper. It need not globally replace `F.cross_entropy` or duplicate Qwen's whole forward. Preserve existing user loss overrides, mode, keyword handling, output structure and idempotent installation. [S32] [S37] [S91]

Start with the exact supported training case: FP16 CUDA logits, hard labels with causal shift and ignore index -100, no label smoothing/weights/custom loss, known layouts, first-order backward, and the original supplied denominator. Unsupported inputs call the saved loss on those **same already-produced logits**. Never rerun the model or dropout to fall back.

**Proposed forward state and backward.** Retain the immutable original FP16 scores `Z16` by reference; do not clone them. Compute the loss using FP32 reductions and save two FP32 scalars per token row: maximum `m` and log of the shifted exponential sum `log_s`. At 2048 rows these normalizers occupy 16 KiB. Shifted target indices, denominator and metadata are small additional state. The full retained `Z16` still occupies 593.5 MiB.

Reconstruct each tile's FP32 log-probability in backward in the intended order:

```text
logp32 = (float32(Z16) - m32) - log_s32
target_term = -g / D at the supervised target, otherwise 0
row_sum = -g / D for a supervised row, otherwise 0
dZ32 = target_term - exp(logp32) * row_sum
dZ16 = float16(dZ32)
```

Here `g` is the actual incoming loss gradient, including dynamic scaling; `D` is the original denominator, not a per-tile or per-chunk count. Construct the sparse target term in registers rather than as a full tensor. Write the final full `dZ16` directly and return it to the unchanged head backward. The native-shaped formula above helps preserve arithmetic order, but a new reduction tree, exponential implementation or FMA can still change rounding. No bitwise CUDA claim follows from it. [S37] [S74] [S75]

Keeping `Z16` and `dZ16` means A0b is **not** a fully bounded-vocabulary head: their backward payload alone is 1187 MiB. Its advantage over A1 is avoiding an extra projection while removing full FP32 CE buffers. The original head compute-weight snapshot and ordinary hidden-gradient GEMM remain required.

**Nonfinite and denominator behavior.** An ignored row is not generally safe to zero in backward. A nonfinite log-probability can propagate through `exp(logp) * 0`, affecting overflow detection even though NLL ignores the target. A first finite-only implementation could detect unsupported nonfinite rows, including ignored rows, and use the original loss from the same logits before attaching its custom backward. That fallback may require synchronization and restores ordinary allocations for that call; measure it. Empty supervision, a zero denominator and unsupported higher-order differentiation should also preserve original behavior rather than inventing a finite result. The CPU counterexample appears in section 10.11.

**Pair it with A0.** Retaining the original metrics would preserve the accuracy copy, reducing A0b's benefit. Initially combine A0b with the separate A0 metric implementation. A later coordinated reduction can also emit detached entropy, predictions and per-token CE. The loss wrapper does not itself receive the attention mask, so produce per-row statistics and let the trainer apply the exact entropy and supervision masks.

Accelerate's returned FP32 tensor differs in identity from the internal FP16 logits. Bind any shared compact metrics through an invocation-scoped forward/output record, then validate outputs and labels at the consumer boundary. Clear it in `finally`. Avoid a persistent pointer cache. Keep Rewind's original row computation until its handoff is independently verified; it prioritizes `outputs.logits`. [S10] [S35] [S38] [S39]

**Phase-specific payload ledger.** Let `L32` be saved log-probabilities, `O32` outward logits, `C32` the shifted accuracy copy, and `R32` the one-example Rewind log-softmax. With 511 predicted positions, `R32 = 296.17041015625 MiB`. Values below exclude model/optimizer/body state, the 445.125 MiB FP16 head-weight representation, labels, small normalizers, workspaces and allocator behavior. Do not sum rows.

| Variant / phase | Included vocabulary payload | Calculated MiB |
| --- | --- | ---: |
| Baseline CE forward | Z16 + Z32 + L32 | 2967.5 |
| Baseline output conversion | Z16 + L32 + O32 | 2967.5 |
| Baseline entropy chunk | L32 + O32 + three 128-row FP32 temporaries | 2596.5625 |
| Baseline accuracy | L32 + O32 + C32 | 3558.681641 |
| Baseline Rewind | L32 + O32 + R32 | 2670.170410 |
| Baseline or A0-only CE backward | L32 + incoming FP32 gradient + output FP32 gradient | 3561 |
| A0b without A0, accuracy | Z16 + O32 + C32 | 2965.181641 |
| A0 + A0b metrics | Z16 + O32 + A0 partial workspace | 1781.6875 |
| A0 + A0b with ordinary Rewind | Z16 + O32 + R32 | 2076.670410 |
| A0b CE backward | Z16 + dZ16 | 1187 |

The ledger identifies what must be measured and prevents a false conclusion from eliminating one large allocation. A0b's forward reductions, reconstruction cost, synchronization, gradient differences, allocator reuse and full training quality remain unmeasured. Test this intermediate design before accepting A1's replay cost.

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

#### 4.4.1 Preserve the actual forward and backward precision boundaries

The ordinary frozen, bias-free Qwen head receives FP32 hidden states and weights in this recipe. CUDA autocast supplies half operands to its GEMM; Transformers then casts the half scores to FP32 for CE. The native input-gradient GEMM returns half before the cast edge promotes it to the original hidden dtype. This is derived from the pinned model, linear, derivative and autocast sources, not observed GPU execution. [S32] [S37] [S55] [S56] [S70] [S71]

```text
Forward:
  H16 = half(H32)
  W16 = half(W32)
  Z16 = GEMM(H16, W16.T)
  loss32 = native_CE(float(Z16), shifted_targets) / global_denominator

Backward:
  g32 = actual incoming loss gradient
  dZ32 = FP32 CE derivative with g32 and the same global denominator
  dZ16 = half(dZ32)
  dH16 = GEMM(dZ16, W16)
  dH32 = float(dH16)
```

The displayed denominator applies to a summed CE numerator; do not divide a mean CE again. If `num_items_in_batch` is absent, preserve the reference mean reduction. Keep the actual incoming gradient dynamic; it is not always the initial GradScaler value.

The familiar `(p-one_hot)*g/D` is mathematically correct, but it is not a proof of bitwise native CE equivalence. Native log-softmax stores rounded FP32 log probabilities and its backward uses their exponential; a direct normalized exponential, multiplication ordering or fused multiply-add can differ. Preserve the major dtype boundaries and validate the proposed reduction against actual native CUDA outputs. [S74] [S75]

Avoid a blanket `custom_fwd(cast_inputs=float16)` on a function that also receives FP32 scalar/reduction state. The decorator casts eligible floating inputs, then disables autocast inside. Explicitly select H/W compute dtypes and keep normalization and incoming scaling in FP32; `custom_bwd` does not independently enforce the required arithmetic order. [S72] [S73]

#### 4.4.2 A frozen head is excluded from the normal autocast weight cache

PyTorch 2.11's cached-cast predicate requires a FP32, requires-grad, leaf, non-view tensor. This frozen head fails the requires-grad condition. Repeated chunk calls with the original FP32 weight therefore need not reuse one half conversion even within one autocast scope. [S70]

At `V=151936` and `H=1536`:

| Head representation or operation | Calculated payload / logical traffic |
| --- | ---: |
| Original FP32 tied weight | 890.25 MiB |
| One FP16 compute-weight snapshot | 445.125 MiB |
| One FP32 read plus FP16 write for conversion | 1,335.375 MiB |
| Sixteen separate conversions, 128 total rows per chunk | 20.865234375 GiB |
| The same naïve conversions in forward and replay | 41.73046875 GiB |

These are tensor sizes and conversion-traffic calculations, not measured DRAM traffic or latency. Reuse one detached half snapshot throughout a single forward/backward invocation. Do not convert `lm_head` in place or assign a replacement Parameter: Qwen ties its weight to the token embedding, so either action can change embedding arithmetic or break the weight contract. [S32]

The half weight is **not automatically 445.125 MiB of new overhead**: ordinary head backward already needs and normally saves that compute weight for `dH`. Count its real lifetime in both implementations. A1 can additionally save the exact half hidden operands for replay: 6 MiB at maximum shape. The custom Function still takes the original graph-connected hidden input and returns its gradient in the original dtype. No persistent cross-step weight cache is needed for this first design.

#### 4.4.3 Prefer token-row chunks before splitting the vocabulary gradient

Rows have independent CE terms. With a frozen classifier there is no head-weight gradient to accumulate between row chunks, and each hidden-gradient GEMM can reduce across the entire vocabulary once. A vocabulary split instead creates partial hidden gradients; if each library GEMM returns FP16, adding those partials into FP32 does not undo their earlier rounding.

The original CPU study supplies a small witness: `dZ=[1,1,-2]` and one weight column `[2048,1,1024]`. Ideal accumulation followed by one half rounding yields 1. Splitting after the second vocabulary entry produces half partials 2048 and -2048; even their FP32 sum gives 0. This proves a boundary difference in the explicit oracle, not how a particular cuBLAS algorithm accumulates.

Use the same row-chunk geometry and H16/W16 snapshots in forward and replay initially. Compacting only supervised rows in backward changes GEMM shape and can change score rounding. In particular, do not normalize recomputed scores with saved forward statistics unless their consistency is verified; recomputing the statistics from replayed scores is a separate possible choice. PyTorch does not promise bitwise equality between mathematically identical sliced and full operations. [S87]

| Total token rows per chunk | Chunks for 2048 rows | One FP16 scores or dZ tile | Two simultaneously live FP16 tiles / one FP32 tile |
| --- | ---: | ---: | ---: |
| 64 | 32 | 18.546875 MiB | 37.09375 MiB |
| 128 | 16 | 37.09375 MiB | 74.1875 MiB |
| 256 | 8 | 74.1875 MiB | 148.375 MiB |
| 512 | 4 | 148.375 MiB | 296.75 MiB |

Start device comparisons with 128 and 256 total rows, holding the rest fixed. These are proposed comparison points, not tuned choices. Persistent model storage, the saved weight/hidden snapshots, reduction records, returned hidden gradients and allocator behavior are outside this tile table.

#### 4.4.4 Backward-time replay has a real compute price

One maximum-shape head projection has `2*2048*1536*151936 = 955,898,658,816` multiply/add FLOPs. The frozen baseline needs a forward projection and an input-gradient projection, approximately 1,911.797 GFLOP. Replaying scores in backward adds another projection: approximately 2,867.696 GFLOP in total, a **50% increase in this head GEMM arithmetic** before any valid-row specialization.

This is not a 50% increase in total training time. Chunk efficiency, avoided memory traffic, other layers and launch overhead all matter. It is nevertheless too much work to dismiss as free recomputation. A1 must pass the whole-command time constraint, including cold compilation, and cannot count library GEMMs as original Triton participation.

Precomputing an unscaled half gradient in forward does not generally avoid this cost while preserving AMP behavior. In a synthetic uniform-vocabulary row with denominator 2044, a non-target derivative is about `3.2200e-9` and rounds to half zero. Applying incoming scale 65536 first produces a representable half value `0.0002110004425`. All 151,935 non-target entries disappear under the early-rounding construction. This is a numerical counterexample, not a measured failure rate in Qwen or an execution of Soup's implementation.

#### 4.4.5 Use a native per-chunk backward as the integration reference

Before writing the original score-gradient kernel, a proposed correctness bridge can replay one row chunk inside `torch.enable_grad` and the matching autocast state. Make a detached half hidden chunk a new requires-grad leaf, use the saved half weight with ordinary `F.linear`, and calculate native FP32 summed CE divided by the single global denominator. Request its gradient with `torch.autograd.grad(..., grad_outputs=actual_incoming_loss_gradient)`. The result is half; promote it when returning the custom gradient for the original FP32 hidden input.

This sequence places upstream scaling inside native differentiation, keeps native per-chunk CE ordering, and lets each temporary graph die before the next chunk. Do not call `backward()` internally or scale/divide the returned gradient again. The bridge uses extra native chunk intermediates and does not itself constitute an original kernel contribution. It is a proposed comparator for an independently written dZ kernel, not an implemented or GPU-validated solution. [S55] [S56] [S73] [S74] [S75]

A1 installation also has to precede or deliberately preserve Accelerate's stored forward/autocast/output-conversion wrapper. A late class-method replacement may miss an already installed instance wrapper. Keep normal evaluation/prediction output behavior, and coordinate A1's compact metrics with the trainer and Rewind consumers rather than letting A0 assume a full `outputs.logits` tensor. [S38] [S39]

#### 4.4.6 Ignored targets do not automatically make nonfinite rows harmless

Native NLL backward initializes its input gradient to zero and skips ignored targets. Log-softmax backward then combines that zero gradient with the saved log probabilities. If those probabilities are NaN, the expression can still propagate NaN through multiplication by zero. This can affect the upstream gradients and GradScaler's overflow decision. [S74] [S75]

An optimized path that simply drops every ignored row can therefore change nonfinite behavior. Detect unsupported nonfinite score states across all rows, including those excluded from ordinary metrics, or preserve the original behavior. A pre-backward fallback can recompute the original head/loss from the already produced decoder hidden states; it must not rerun the decoder and dropout. The fallback may sacrifice A1's memory saving in exceptional cases. Real CUDA validation remains necessary.

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


#### 5.6.1 Initial kernel and saved-state design

Start with one program per row, a 2048-lane logical block, and masking for the 1536 real features. Four warps is a development starting point; compare two/eight only when compiled resource use or measured performance warrants it. This is a reduction kernel and does not depend on Tensor Core lowering.

An alternative backward expression avoids forming a standalone tiny `r^3`:

```text
z  = x * r
v  = dy * gamma
c  = mean(v * z)
dx = (v - z * c) * r
```

Both formulas are mathematically equivalent. They have different floating-point behavior; neither has yet been checked against the exact PyTorch CUDA backward. Preserve FP32 loads, reductions, stores, epsilon and forward multiplication boundaries. Initially constrain optional contraction/reassociation and compare the actual compiled result. Do not add clipping, change epsilon, unscale gradients, or silently replace nonfinite values.

| State for an active input-gradient path | Maximum-shape size | Ownership |
| --- | ---: | --- |
| Original input | 12 MiB | Saved reference to existing FP32 storage |
| Frozen gamma | 6 KiB | Reference to existing parameter |
| Inverse RMS | 8 KiB | One FP32 scalar per row |
| Shape/layout metadata | Small | Host metadata |

No cross-row workspace, weight-gradient partials, atomics or locks are needed. Across 56 active norm backward paths, the inverse-RMS vectors total at most 448 KiB if simultaneously live. This is not automatically incremental memory: eager normalization also produces row statistics. Saving the input remains necessary; eliminating forward intermediates does not eliminate its lifetime.

Use outer grad mode and input requirements to decide whether the forward needs saved state. Grad mode inside a custom autograd Function's forward is normally disabled, so that check alone is insufficient. Consult `ctx.needs_input_grad[0]` before launching backward. Fall back when gamma is trainable, preserve the incoming loss scale, and never overwrite input, gamma or `dy`.

Unsupported forward layouts should take the original path rather than incur an unnoticed reshape copy. A contiguous forward does not guarantee contiguous upstream gradients: handle backward strides or explicitly materialize a copy and measure it.

At maximum shape, row-level parallelism already supplies 2048 programs. A persistent row loop does not eliminate 2048 kernel launches: the ordinary row-program grid is itself one launch. Multirow/persistent variants trade scheduling and gamma reuse against registers and occupancy. Turing supports up to 32 resident warps per SM and has 64K 32-bit registers per SM; inspect actual register use and spills before choosing a configuration. [S50]

Keep width/dtype/epsilon specializations narrow and allow variable row count. A forward with statistics, forward without statistics and input-backward variant are a small initial set. Record actual compilation count. Broad runtime autotuning is especially costly because every official run begins with empty caches. [S27]

The CPU study in section 10.5 validates the real-arithmetic derivative and exposes cancellation/underflow edge cases. It does not establish PyTorch or GPU equivalence.

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

Separate packing and restoration add approximately **11.484375 GiB of logical traffic per maximum-shape step** across 28 layers. M1 below avoids the pack half of that traffic by retaining the original producer storage. Its integration is more involved; compare both only after correct ownership is demonstrated.

CPU exactness results are in section 10.2. GPU conversions, full training and live-memory release remain untested. GPU round-trip checks must include half subnormals and both signed zeros; the finite-value proof does not establish preservation of NaN payloads. The starter's arithmetic stays excluded; eligibility and runtime share of any original packing kernels still need to be established.


#### 5.7.1 M1: retain the producer's existing FP16 storage

**Preferred retention design to test:** capture the existing FP16 gate/up result before its normal FP32 promotion, keep forward arithmetic unchanged, and use that half storage as the saved payload for SwiGLU backward.

The exact interception point exists in pinned `Linear4bit.forward`: the result of its delegated `bnb.matmul_4bit` call is converted to the original input dtype. Capture at that return boundary, then let the original method complete its FP32 conversion. The bias-free gate/up producers and compute dtype must match the audited recipe. [S32] [S33] [S42]

One proposed integration design uses a scoped capture record for each MLP invocation:

1. Identify the exact eligible gate/up modules and establish separate producer tickets. Require frozen four-bit weights, no bias, FP16 compute, FP32 input/output, supported layouts and the unchanged supplied SwiGLU.
2. A delegating matmul wrapper records a detached view of the matching FP16 result. **Return the original graph-connected result to the caller.** A detached view shares storage without its autograd history; neither view may be modified. [S47]
3. Let the original Linear4bit perform its FP32 promotion. Bind a weak reference to that exact returned tensor object to the strongly owned half storage and simple restoration metadata.
4. Around only the SwiGLU autograd invocation, substitute the compact payload at the saved-tensor hook if object identity and provenance match. Otherwise retain an ordinary detached full representation.
5. Clear temporary capture records in `finally`. Each saved payload independently owns its half storage and can restore FP32 without consulting any later capture frame.

A module-level context variable can scope producer tickets and restore the previous state on exceptions or nesting. A weak reference permits identity checks without keeping the promoted allocation alive. These are Python ownership mechanisms; the design still needs a real PyTorch implementation check. [S48] [S49]

Do not match only by tensor shape, raw device address or an old `id`. Pointer/object identifiers may be reused, and module hooks or contiguity conversions may return a different tensor object. Exact identity mismatch is a reason to fall back, even if a less strict shortcut might appear to work.

The packed payload must hold the half tensor strongly, without holding the promoted FP32 tensor, an entire module graph or the temporary frame. A weak half reference or raw pointer does not keep its storage alive. Unpack is stateless, returns the original FP32 shape/device/values and supports repeated calls. Nested saved-tensor hooks require explicit compatibility; only the innermost pair applies. [S40]

**Gradient argument:** ordinary FP32 inputs still travel through their original FP16 conversion, four-bit projection, FP32 promotion and unchanged SwiGLU. Restoration supplies saved values, not new forward graph edges. SwiGLU backward returns FP32 gate/up gradients, and the original promotion/input-cast edges perform their original conversions and accumulation. Do not return the detached shadow from the projection or merge branch cast edges. This is a first-order argument awaiting autograd validation.

Pinned bitsandbytes saves packed weights and metadata for its input backward, rather than the projection output. This supports the possibility that a correctly scoped payload can release the promoted FP32 allocations; aliases or other hooks in the actual runtime still need inspection. [S34]

| Gate/up retention comparison | M0: fresh packing | M1: producer reuse |
| --- | ---: | ---: |
| Retained pair per maximum-shape layer | 70 MiB | 70 MiB |
| Extra packing traffic per layer | 210 MiB | None |
| Extra restoration traffic per layer | 210 MiB | 210 MiB |
| Extra conversion traffic across 28 layers | 11.484375 GiB | 5.7421875 GiB |
| Nominal retained-byte reduction across 28 layers | 1,960 MiB | 1,960 MiB |

The ordinary forward promotion remains in both paths and is excluded from the additional-traffic rows. M1 extends the existing half allocation's lifetime; FP32 forward values still exist temporarily and FP32 values are reconstructed during backward. The result is the same nominal retained-state reduction as M0, with less conversion work.

M1 also removes a possible original packing-kernel launch. Its credited kernel time may decrease even while complete runtime improves. Do not preserve redundant packing solely to inflate credit. An independently implemented restoration kernel must actually supply backward values to be considered useful original work; the supplied SwiGLU remains excluded. [S30]

The NumPy ownership/manual-gradient study in section 10.4 supports the value and lifetime reasoning. Actual autograd hooks, CUDA allocation release, conversion behavior, cold-run time, sampled peak and eligibility remain open.

#### 5.7.2 Saved-hook callables can defeat the compact payload

PyTorch 2.11's saved-variable hook object retains three owners: the pack callable, unpack callable and packed data. A small half payload is insufficient if a callback closure, default argument, partial or bound object also retains the promoted gate/up tensors. Clearing the invocation frame does not clear a separate captured tuple. [S84]

Use module-level non-capturing callbacks. Pack may look up the current invocation record while saving; unpack must work solely from its payload after that record has expired. Retain only detached half storage and restoration metadata in the payload. Keep weak promoted-object references in the capture record and restore the previous ContextVar token in `finally`.

The new Python ownership model gives a concrete result: callbacks that capture the promoted pair retain **56/56** original arrays; module-level callbacks retain **0/56**. Both have identical half payload size, release all capture frames, and restore the same values. At maximum model shape, the mistake could retain 3,920 MiB of original FP32 values plus 1,960 MiB of halves, or **5,880 MiB** of this payload. That is 1,960 MiB above the original representation, rather than the intended 1,960 MiB saving. These are calculations and a CPU ownership mechanism, not GPU peak measurements.

Keep both the hook context and invocation record alive around the **complete `SwiGLU.apply` call**. Native SavedVariables are constructed after custom forward returns and its outputs are wrapped. A hook context confined to the body of custom forward can expire too early. [S86]

Do not require `torch.is_grad_enabled()` inside pack: native packing uses a no-grad guard. Custom Function forward also disables recording. Capture training applicability outside, then use provenance and metadata inside; `ctx.needs_input_grad` identifies requested derivatives. [S85] [S86]

Hooks also bypass the usual saved-variable version-counter check, because PyTorch cannot track arbitrary packed representations. The pack call itself must not mutate its input, and the half shadow or any alias must remain immutable through all unpacks. Use fresh restoration output rather than one global scratch buffer shared by live states. These ownership requirements need a native autograd test before M1 can be trusted. [S85]

### 5.8 Boundaries that can silently change backward

**Sharing a cast:** independent projections can upcast their FP16 input gradients separately before accumulating in FP32. One shared input cast may instead accumulate branch gradients in FP16. For example, adding half values 2048 and 1 after promotion gives FP32 2049; adding in FP16 first rounds to 2048. Forward identity alone does not prove backward identity. Section 5.11 proposes a narrower alternative: share only the detached forward storage, with a separate gradient edge for each original branch.

**Blanket compression:** FP32 residuals, adapter sums, normalized states, SiLU outputs and gradients are not generally exact promotions of FP16. M0 applies only to the verified producers.

**Frozen layers:** block 0's postattention MLP already depends on its trainable Q/V adapters; its input gradient is required. Most later frozen projections also carry gradients to earlier adapters. The frozen embedding/first preattention prefix should already have unnecessary backward work pruned. There is no broad “skip frozen backward” shortcut.

**RoPE:** a later P0 experiment can fuse FP32 rotation with the FP16 conversion consumed by SDPA. Qwen rotates contiguous halves of 128-wide heads, with 12 Q and two K heads. Scope to the actual no-cache attention call; preserve arithmetic/cast order, masks, layouts and backward. Before KV-head replication, Q+K occupy 14 MiB in FP32 versus 7 MiB in FP16; this is transient storage, not 28 layers of saved activation. [S32] [S43] [S44]

When HF explicitly repeats KV heads before SDPA, preserve the FP32 accumulation of repeated-head gradients. Moving K's cast before `repeat_kv` can instead make that accumulation occur in FP16. Place the cast after that boundary or implement equivalent FP32 accumulation. A no-cache call alone does not establish equivalence. Initially restrict expanded-KV P0 to the original mask-present branch. On a maskless call, expanding KV can also change the backend; section 7.7 treats that as a distinct P2 experiment. [S44] [S94] [S95]


### 5.9 B0: fuse the exact active LoRA epilogue

The fifth pass identifies a narrow original operation after the existing base projection, dropout and adapter GEMMs. In pinned PEFT 0.20.0's vanilla four-bit LoRA path under autocast, the relevant sequence is: clone the FP32 base result; multiply the FP16 adapter output by scaling; add that FP16 result to the FP32 base. The explicit conversion to the base dtype is guarded by autocast being disabled, so it is inactive in the training path examined here. [S51]

For the fixed scaling `alpha/r = 32/16 = 2`, let `B32` be the base result, `U16` the raw adapter output and `G32` the incoming output gradient:

```text
forward:
    S16 = half(float32(U16) * 2)
    Y32 = B32 + float32(S16)

backward:
    dB32 = G32
    dU16 = half(float32(half(G32)) * 2)
```

These intermediate half conversions are part of the reference behavior. Add's generated derivative supplies the incoming gradient, then PyTorch's engine validates/converts it to the next edge's gradient dtype before scalar-multiplication backward runs. The default gradient dtype is the tensor's scalar dtype. Thus the adapter branch receives `half(G32)` before multiplication by two. [S52] [S53] [S54] [S55] [S56]

An original forward epilogue can read `B32` and `U16` once and produce a **fresh FP32 output**, preserving `S16` rounding without materializing that whole tensor. A backward kernel can perform the two required half boundaries and produce a separate `dU16`. The base branch can receive the unchanged upstream gradient when required. Never scale or overwrite `G32` in place: the base branch also needs it.

Keep base projection, original Q/V dropout calls, adapter A/B GEMMs, and their autograd edges outside this small custom Function. Guard one active vanilla adapter, expected scaling/dtypes/device/layout, unmerged enabled adapters and the actual autocast path; use the original method otherwise. Other scaling values, LoRA variants and multi-adapter behavior require their own equivalence check.

The defensive PEFT clone is motivated by view/backward concerns. A fresh out-of-place result avoids editing the base tensor, but actual view and autograd behavior must still be tested. Clone backward is identity, and these clone/add/scalar-multiply derivatives do not require retaining their full forward values. This is principally a traffic/launch optimization, not a large saved-activation reduction. [S51] [S55]

| Logical traffic at `M=2048`, Q and V together per layer | Existing sequence | Proposed B0 |
| --- | ---: | ---: |
| Forward clone, scale and add | 77 MiB | 35 MiB |
| Adapter-gradient cast and scale | 35 MiB | 21 MiB |
| Total | 112 MiB | 56 MiB |

Nominal traffic avoided: **56 MiB per layer, or 1,568 MiB across 28 layers**. The calculation excludes GEMMs, dropout, caches, gradient accumulation and unrelated traffic. Q and V do not need to be simultaneously live for this traffic sum. It is not a peak-memory or timing prediction.

Two explicit counterexamples show why algebraic simplification is insufficient:

| Case | Reference order | Reordered calculation |
| --- | ---: | ---: |
| `B32=0`, `U16=32768`, scale 2 | Half scaling overflows to `+inf` | Promoting before scaling produces FP32 65536 |
| `G32=2^-25`, scale 2 | Half conversion rounds to zero before scaling | Scaling first produces the smallest positive half subnormal |

Soup's inactive Fast-LoRA helper promotes its update before an alpha-scaled addition. It is useful prior art but not the numerical reference for active PEFT AMP behavior. Do not enable that prohibited mode or copy its implementation. The original CPU oracle in section 10.6 checks the specified rounding sequence; actual CUDA/Triton cast preservation remains open. [S10]

### 5.10 O0: optimizer fusion is a lower-priority hypothesis

Pinned Transformers selects `torch.optim.AdamW` for `adamw_torch` and supplies `fused=True` only for the distinct fused optimizer name. In PyTorch 2.11, ordinary compatible CUDA parameters with unspecified foreach/fused flags select the multi-tensor foreach path. Therefore, the expected baseline already batches adapter updates; it should not be modeled as a wholly separate optimizer-kernel sequence for every matrix. This is a source-derived expectation to verify in the runtime. [S64] [S65] [S66] [S67] [S68]

Foreach still has multiple arithmetic passes, so an original vertically fused update could leave some work to remove. Its state-memory opportunity is much smaller than the activation candidates:

| Fixed Q/V LoRA storage | Calculated size |
| --- | ---: |
| 2,179,072 FP32 parameters in 112 tensors | 8.3125 MiB |
| FP32 gradients, if all materialized | 8.3125 MiB |
| Two FP32 Adam moments | 16.625 MiB |
| Parameters, gradients and moments combined | 33.25 MiB |
| One full-size foreach intermediate | 8.3125 MiB |

Moments are lazily initialized like their parameters. These figures exclude small counters and other runtime objects. Small byte volume does not prove negligible time because launches can matter; obtain attribution before investing in a replacement. [S65]

Preserve Soup's gradient-clipping default 1.0 and weight decay 0.01, the existing clipping-before-update order, overflow-skipped optimizer/scheduler behavior, zero-versus-None gradients, state dtype, bias corrections and checkpoint state. Do not switch the fixed optimizer name or reduce moment precision. A built-in fused optimizer also supplies no original Triton credit. [S10] [S45] [S68]

### 5.11 I0: one input payload, separate projection gradients

The earlier shared-cast warning remains valid. A single graph-connected FP16 cast can accumulate branch gradients in half before promotion. The new proposal shares physical forward storage while keeping independent backward edges.

**Source opportunity.** Each frozen bitsandbytes base projection converts its FP32 normalized input to its configured FP16 compute dtype. Q, K and V consume the same input object; gate and up consume another shared input. With the inspected configuration, there are five such conversions per block. MatMul4Bit retains packed weight/quantization state and input metadata rather than saving its input activation. A same-dtype/device `.to()` with default copy behavior returns the existing tensor. [S32] [S33] [S34] [S93]

For each exact group invocation, create one detached half payload. Give each original base projection a distinct custom-Function output view of that payload, connected separately to the original FP32 input:

```text
payload16 = float16(detach(X32))
for each original base-projection call in its original position:
    branch16 = SeparateCastView.apply(X32, payload16)
    base16 = saved_original_Linear4bit_forward(module, branch16)
    base32 = float32(base16)

SeparateCastView.backward(incoming_half_gradient):
    return float32(incoming_half_gradient) for X32
    return no gradient for payload16
```

This is design pseudocode, not implementation. There must be a distinct node/view for every branch, not one multi-output function that combines all base gradients internally. The latter could change their accumulation relative to adapter gradients. The bridge needs only metadata for this first-order cast derivative; it must not save the FP32 input or keep the half payload alive solely for backward.

Pinned native custom-Function handling can wrap an unmodified returned input as a view and attach a new gradient edge. That supports the proposed alias arrangement, but also imposes read-only/version-counter constraints. Native behavior, exact accumulation order and graph overhead still need direct tests. First-block inputs without a required input gradient must retain their ordinary no-gradient path. [S92] [S86]

**Keep the boundary below PEFT.** The outer Q/V LoRA wrapper must receive the original FP32 input. Its dropout and A/B linears stay in their original sequence. Prematurely casting that input changes dropout scaling: one CPU witness produces half `0.5283203125` for FP32 dropout scaling followed by the normal cast, versus `0.52783203125` when the input is rounded to half first. Section 10.12 gives the exact construction; it simulates no RNG.

An initial base-class wrapper should require an initialized `compute_type_is_set`, `compute_dtype == float16`, FP32 source input, exact module/input identity, compatible device/layout, frozen base parameters, and no unsupported subclass or hook behavior. Delegate once to the saved base implementation with the branch half tensor, then restore its ordinary FP32 output. In this pin the bias cast follows the compute-dtype input cast, so the supported original and proposed base paths both use half bias. Preserve quantization-state repair and every other delegated behavior. [S33] [S51]

**One owner must coordinate I0 and M1.** If I0 moves the final FP32 promotion outside the delegated base forward, M1 cannot continue binding its weak-reference record to an intermediate returned by a different wrapper layer. Coordinate input delegation, half output capture and FP32 promotion in one wrapper. Match the original module ticket and FP32 output identity. Release the shared input payload after V or up consumes it, with exception/reentrancy cleanup. Do not retain it for a whole decoder forward or cache by data pointer.

At `M=2048,H=1536`, one FP32 input is 12 MiB, its half payload is 6 MiB, and a conversion's logical read/write volume is 18 MiB:

| Quantity | Separate casts | Proposed shared payloads |
| --- | ---: | ---: |
| Input conversions per block, QKV plus gate/up | 5 | 2 |
| Conversions across 28 blocks | 140 | 56 |
| Logical forward cast traffic across 28 blocks | 2520 MiB | 1008 MiB |

The difference is **84 conversions and 1512 MiB of logical forward traffic**. Backward still promotes each branch separately; no backward traffic reduction is claimed. These temporary casts are not 28 layers of retained memory. Shared payloads can live longer across intervening adapter work, and Python custom-Function and wrapper overhead can outweigh launch savings.

A later R0 integration could emit both its required FP32 normalized output and a detached half shadow rounded from that exact FP32 result. Separate I0 gradient edges would still be necessary. This could avoid another 56 reads of a 12 MiB FP32 tensor, or **672 MiB** of logical traffic at maximum shape. Establish the simpler I0 design first; neither figure predicts speed, peak VRAM or original-kernel participation.

## 6. NF4 and the fixed T4 compiler

### 6.1 Existing fusion narrows the opportunity

bitsandbytes PR #1949 introduced fused four-bit forward GEMMs with an FP16 Turing path. Its author reports that calls above 1,536 tokens use the earlier fallback, with lower cutoffs for many shapes; QLoRA tests saw unchanged peak memory and only a small host-overhead-related timing improvement. Its inference headline is not a training forecast. [S15]

Training microbatch size is not GEMM's token dimension: four sequences of length 512 give `M=2048`. **The fifth-pass shape-specific audit in section 6.5 tightens the relevant fallback bounds to 64 or 128 tokens for this model.** The global 1536 cap alone is too loose. Runtime dispatch/timing still needs a device profile. [S26] [S42]

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
| All K/V matrices | 42 MiB | 1.3125 MiB |
| All Q/V matrices | 147 MiB | 4.59375 MiB |
| All attention Q/K/V/O matrices | 294 MiB | 9.1875 MiB |
| One selected gate/up/down matrix | 26.25 MiB | 0.8203125 MiB |
| All seven decoder linear matrices across 28 layers | 2,499 MiB | 78.09375 MiB |

These exclude allocator overhead, second-level metadata and other tensors. They are not measured peak savings or confirmed legal modifications. Use true weight identity/version for caches; pointer-only identity can be wrong when storage is reused.


### 6.5 N0: fuse nested decoding at both dense entry points

For FP16 and a standard 40-SM T4, applying the pinned dispatch conditions gives these **sufficient fallback bounds**. They are source-derived decisions, not observed launches. [S42] [S69]

| Projection | Weight shape `[N,K]` | Separate decode plus linear selected when |
| --- | --- | --- |
| Q, O | `[1536,1536]` | `M > 64` |
| K, V | `[256,1536]` | `M > 64` |
| Gate, up | `[8960,1536]` | `M > 128` |
| Down | `[1536,8960]` | `M > 64` |

All seven therefore take the fallback for `M>128`, equivalent to padded sequence length greater than 32 at batch four. Below those bounds dispatch is not globally monotone: small-M special cases exist. Preserve upstream dispatch rather than replacing it with a simplified threshold rule.

There are 196 base projection forwards and 193 requested base input backwards per ordinary step: the first block's Q/K/V base input gradients are absent with the frozen embedding prefix. When every forward falls back, this gives **389 decoded-weight uses per step**. A fused forward does not remove its requested backward decode. [S32] [S34]

Two integration points are required:

| Consumer | Actual pinned path | Consequence |
| --- | --- | --- |
| Dense forward fallback | Backend `_dequant_linear_fallback` reconstructs scales and calls `_dequantize_4bit_impl` directly | Replace/guard the composite fallback before scale expansion; a public-only patch misses it and a low-level-only patch is too late |
| Base input backward | `MatMul4Bit.backward` calls `functional.dequantize_4bit` and then matmul | Scope a replacement to the matching metadata/dtype contract |

Keep existing matrix multiplications and the four-bit autograd wrapper. In the registered CUDA path, the composite fallback is resolved as a module global; wrapping that helper can retain the upstream dispatch decision. Replacing only `_dequantize_4bit_impl` cannot remove nested-scale decode or offset addition, which have already happened and whose original metadata is no longer passed at that boundary. N0 needs only an independently written nested decoder, so it does not rely on Triton MMA lowering. [S34] [S42]

For an original row-major element `i`, block size 64, nested group size 256, FP32 scale metadata and FP16 decoded output, the dense path's arithmetic is:

```text
packed_byte = packed[i // 2]
code4       = high nibble if i is even, low nibble otherwise
block       = i // 64
group       = block // 256
product32   = fp32(code8[scale_codes[block]] * scale2[group])
scale32     = fp32(product32 + offset)
weight32    = fp32(NF4[code4] * scale32)
weight16    = half(weight32)
```

The product and offset addition have **separate FP32 rounding boundaries** in the dense path. Contracting them into one fused multiply-add can change the final FP16 weight. Converting centroids/scales to half early also changes the result. The original experiment in section 10.7 provides exact arithmetic witnesses. [S57] [S58]

The outer CUDA NF4 decoder selects its canonical 16-value table by quantization type; its API does not consume `QuantState.code`. The nested 256-entry `state2.code` is actual metadata and must be honored. For transpose gathers, derive the scale index from the original source index `n*K+k`, not a transposed flattened index. Guard odd tails and retain the public API's legacy transposed-packed behavior through fallback where unsupported. [S57] [S58]

Restrict the first implementation to the audited storage format, block/group sizes, dtypes, device and output contract. Preserve separate arithmetic boundaries in generated code. The existing fused SM75 GEMM reconstructs scales inside a multiply/add expression; its compiled contraction was not inspected. Do not globally replace its nested metadata with cached dense-path scales. Soup's own scale-cache documentation likewise restricts that optimization to the dense path. [S59] [S10]

Under complete fallback, the usual nested decode comprises scale decode, offset addition and NF4 decode. The logical launch model is approximately **1,167 decode-related launches reduced to 389** by one decoder per use, excluding unrelated conversions. This is a model of source-level launches, not a measured time saving.

Across those uses, FP16 decoded-weight output payload totals **4,992 MiB per step**, and expanded FP32 scale payload totals **156 MiB**. N0 still writes the decoded weights. Its initial benefit is removal of scale intermediates and repeated launches, not elimination of the large dense-weight output.

### 6.6 Caching: preserve packed-state backward and bound real ownership

Frozen base weights make derived scale/weight caches plausible, but the cache must preserve the fixed quantized model, exact consumer arithmetic and original metadata. Charge preparation and retained storage to the complete run. Do not enable Soup's existing streaming cache or copy it; it establishes prior art only. [S10]

**The main lifetime trap:** existing `MatMul4Bit` retains packed weights and metadata for backward. Replacing its module call with ordinary differentiable `F.linear` over a decoded weight can make autograd retain that dense weight to compute `dX`. Eviction from an LRU then removes only one reference; the graph still owns the storage. A cache labeled small can accumulate dense weights throughout forward. This follows from the input-gradient dependency and ordinary matrix backward rules; it has not been reproduced on the GPU here. [S34] [S55]

Preserve the current packed-state backward, or implement an equivalent custom backward that fetches/redecodes from immutable packed metadata. A fixed selected subset is easier to bound than a global LRU scanning all layers forward and then backward. Do not mistake cache dictionary size for actual live GPU storage.

The 42 MiB all-K/V subset has **110 full-fallback uses per step**, as does the 147 MiB Q/V subset. One 26.25 MiB MLP weight has only two. These alternatives trade launch count against bytes decoded; profiling must determine which cost dominates. All attention matrices together have 221 uses. The cache budgets in section 6.4 retain the original packed representation.

Use model/weight/metadata identity, shape, dtype, device and appropriate mutation checks. Pointer identity alone is insufficient, and a version counter alone need not detect arbitrary data replacement. Invalidate on movement or replacement and avoid strong references that unnecessarily retain an unloaded model.

Restrict shared cached buffers to known read-only GEMM consumers. Returning the same storage from an unrestricted decoder changes fresh-output/alias behavior for callers that mutate the result or provide `out=`. Preserve the existing fused forward path unless separately validated.

Cache-fill work limited to the first two steps contributes no recurring timed decoder work from step three. Dense caching can remove original N0 calls from that window. A cache is therefore not automatically helpful for the 5% participation gate, even if it improves runtime. Evaluate the final combination's actual share. [S29]

**Order to test:** establish N0 without persistent caching, then compare decoded-scale reuse and one deliberately bounded weight subset. Avoid a full decoded-model cache or general LRU before real autograd ownership and sampled peak are established.

## 7. Experiments and acceptance decisions

All GPU experiments below are proposed. The only executed numerical work is the CPU validation in section 10.

### 7.1 Capture the remaining runtime evidence

The recipe, pins and excluded example are now known. On the T4, confirm loaded builds, flags, first-training-step dtypes, tensor strides and quantization metadata. Inspect PTX before custom GEMM work.

Capture baseline outputs and profile metric copies, native CE backward, casts, saved storage, synchronization and peaks. Record attention-mask presence, Q/K/V head counts and the selected SDPA backend per batch. Deduplicate storage aliases and inspect lifetimes. Source-level byte counts are not peak measurements.

### 7.2 Revised experiment matrix

| ID | Proposed change | Intended benefit | Main decision |
| --- | --- | --- | --- |
| A0 | Two-stage strided entropy/argmax reduction | Avoid accuracy copy and metric intermediates | Same metrics/counts; single forward; measured complete-run gain |
| A0b | Full FP16 logits with compact FP32 CE state and direct half dZ | Remove full FP32 CE forward/backward buffers without head replay | Pair with A0; preserve scale, denominator, nonfinite behavior and consumers |
| R0 | FP32 RMSNorm forward/input backward | Fewer passes and launches | Same precision boundary and gradients |
| M0 | Fresh saved gate/up packing around unchanged example | Lower retained storage; simpler comparison | GPU exactness, original storage released, added traffic acceptable |
| M1 | Reuse original half producer storage for the same saved values | Retain M0's storage reduction without its packing pass | Actual capture/autograd ownership, exact fallback and measured benefit |
| N0 | Nested NF4 fusion at forward fallback and backward decode | Remove scale intermediates and repeated launches | Exact format/rounding; preserve GEMMs and dispatch |
| B0 | Fresh LoRA output plus exact half-rounded scale/add and backward | Remove clone/temporary traffic | Preserve two half boundaries and shared upstream gradient |
| I0 | Detached shared base-input payload with separate cast-gradient edges | Remove repeated forward input conversions | Native view/edge behavior, original dropout, FP32 accumulation and wrapper overhead |
| A1 | Bounded head with compact statistics | Lower logits-related live storage | All consumers/AMP correct; time admissible |
| B1 / N1 / N2 | Adapter expansion / NF4 backward / forward GEMM | More extensive fusion | Competitive generated code on fixed T4 compiler |
| P0 | RoPE plus consumed FP16 conversion, initially on masked repeated-KV calls | Avoid transient output/cast | Preserve backend and repeated-head gradient accumulation |
| P1 | Valid-row compaction only inside deterministic MLP work | Avoid padded MLP computation/saved rows | Actual padding, original dropout calls, row mapping and gradients |
| P2 | Explicit KV repetition for observed maskless math-fallback calls | Permit the existing T4 memory-efficient SDPA backend | Actual backend evidence, changed gradient rounding, full training and eligibility |
| P3 | One immutable aligned additive mask per forward | Avoid repeated per-layer boolean-mask conversions | Exact mask semantics, dtype, row pitch and ownership |
| C0 / C1 | Derived scale cache / fixed dense-weight subset | Avoid repeated decode work | Consumer arithmetic, real ownership, peak and participation |
| O0 | Original AdamW update fusion | Reduce remaining optimizer passes | Profile first; existing foreach and all optimizer semantics |

Keep supplied SwiGLU separately identified as uncredited. Test A0/R0 and M1 independently, then A0 + A0b before A1. I0 requires native edge validation; P2 requires observed maskless math dispatch and a separate numerical comparison. P3 is a smaller secondary opportunity. B0/N0 retain their source-backed designs and CPU checks; M0 is a simpler retention comparison. Defer P1 until actual padding is measured and O0 until attribution justifies it. No candidate is a measured winner or guaranteed to meet the 5% requirement.

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
| A0 / R0 / M0 / M1 individually | To measure | To measure | Check | Measure |
| A0 + A0b, then with validated body changes | To measure | To measure | Check | Measure |
| A1, if A0b leaves a material head bottleneck | To measure | To measure | Check | Measure |
| Best valid combination | To measure | To measure | Check | Measure every run |
| N0 / B0 / I0 / conditional P2 or P3 | Conditional | Conditional | Check | Measure |

Retain a candidate only when it remains correct, eligible and beneficial under official measurements. A memory improvement can justify modest extra work only within admission limits and the scoring tradeoff.

For delegated kernel work, specify operation, GPU, dtype, exact shapes, required gradients, integration point and acceptance evidence. Hugging Face's agent workflow reinforces the need for precise tasks and correctness/performance iteration. [S9]

Reserve time for a clean notebook run and required submission artifacts. Patches also run before regression tests using fake modules; guard real-model/CUDA assumptions and preserve fallback behavior. [S1] [S24]


### 7.5 P1: padding research rules out a simple trim

The ordinary data path already pads dynamically. Soup converts the Alpaca-style fields into messages, applies the responses-only formatter/chat template, truncates each example to at most 512 tokens, and returns an all-one attention mask for its retained tokens. TRL then right-pads to the longest example in that batch. `pad_to_multiple_of` defaults to `None` and is not overridden by this recipe. [S10] [S35] [S36] [S61]

Consequently, with `T=max(L_b)`, at least one example is valid at the last column. **There is no common all-padding suffix to trim.** Accepted examples are nonempty, so there is no common all-padding prefix either. A prefix whose labels are ignored is generally prompt/context; removing it changes conditioning and entropy.

The public preparation script specifies dataset `AmanMussa/kazakh-instruction-v2` at revision `c641407ee50723c84a43ea2ec867b464d3893c12`, training indices 0–255 and public evaluation indices 40000–40255. It writes instruction/input/output fields without tokenization, length sorting or padding. No dataset rows were downloaded in this pass. [S60]

Actual padding depends on tokenization, truncation and the trainer's shuffled batch grouping. Measure row IDs and final lengths in each real batch:

```text
valid positions N = sum_b L_b
padded positions P = B * max_b(L_b) - N
aggregate padding fraction = sum_steps(P) / sum_steps(B*T)
```

No actual padding percentage or expected compaction speedup is established here.

#### 7.5.1 Keep the original dropout mapping

Q and V apply separate LoRA dropout calls. CUDA dropout's launch geometry, vectorization and Philox counter reservation depend on element count, layout and device properties. Removing padded rows can change which random draws reach later valid positions and can change subsequent calls' generator offset. Keeping seed and probability alone does not prove identical masks. [S51] [S62]

A more bounded hypothesis is **MLP-only compaction**: retain the original rectangular attention and Q/V paths; gather valid attention-mask rows in original order for gate/up/activation/down; scatter their outputs back before the next rectangular layer. Backward gathers/scatters the corresponding gradients. Preserve shape, dtype, layout, native dropout path and call order at each Q/V dropout boundary.

This can reduce deterministic MLP work and saved intermediate rows if padding is material. It does not justify changing the entire decoder to one compact token dimension. Gather/scatter overhead, changed GEMM shapes and rounding, invalid-row/nonfinite behavior, zero-gradient routing and interaction with M1 all require testing.

Head-only compaction occurs after decoder dropout, but belongs with A1's compact loss/metric consumers. Recreating a full vocabulary tensor afterward restores the allocation. Entropy still requires prompt and final valid-token positions; supervised labels alone are not a sufficient selection mask.

#### 7.5.2 Attention mask changes need a separate experiment

For finite valid queries in strictly right-padded causal attention, padded keys occur later and are already excluded by causality. That mathematical observation does not establish equivalent implementation behavior.

Pinned Transformers generally keeps mixed-length padding masks. Its SDPA wrapper explicitly repeats KV heads when the mask is present, while a supported no-mask path can request GQA. Removing the mask can therefore change dispatch, temporary storage, padded-query outputs and backward accumulation. Pinned backend selectors now explain why these calls can differ: maskless GQA has no supported fused T4 route in the inspected stack, while repeated-KV masked input can qualify for memory-efficient attention. The installed build and actual calls still need profiling; mask elision is not part of initial P1. Section 7.7 provides the conditional map. [S44] [S63] [S94] [S95]

Naively concatenating examples changes attention context unless sequence boundaries are enforced. Even a correct block mask does not prove that the backend performs sparse or variable-length attention; a flattened tensor may still expose quadratic attention dimensions.

The original synthetic study in section 10.8 checks row mapping and illustrates RNG/context hazards. It provides no measurement of actual dataset lengths or native CUDA random sequences.

### 7.6 Compose patches around clear owners

The following is a proposed integration arrangement, derived from pinned call sites. It has not been implemented or tested as a combined solution. [S30] [S32] [S33] [S34] [S35] [S42] [S51] [S88]

| Owner | Intended boundary | Composition rule |
| --- | --- | --- |
| A0 | SFTTrainer loss/metric method before Soup captures the class | Exactly one model forward; post-output fallback consumes existing outputs |
| A0b | Per-model Qwen loss callable before the internal FP32 cast | Keep head GEMMs and outputs; share invocation-scoped compact statistics only with validated consumers |
| R0 | Qwen2RMSNorm forward and input backward | Preserve FP32 residual/norm semantics; do not include its saved values in M1's narrow hook scope |
| M1 invocation | One coordinated Qwen2MLP wrapper around supplied SwiGLU | Exact gate/up module tickets and an invocation record; unchanged supplied arithmetic |
| M1 producer / I0 base wrapper | Actual bnb base projection, matmul alias and final promotion | One owner for input payload delegation, half output capture and FP32 output identity; delegate once |
| B0 | PEFT's four-bit LoRA wrapper | Distinct from bitsandbytes' base class; retain base/dropout/A/B calls and use fresh epilogue output |
| N0 | CUDA composite dense fallback plus functional backward decoder | Stay inside original MatMul4Bit autograd and preserve dispatch, signatures and dtype/alias fallbacks |
| Future A1 | Head/loss plus compact consumer contract | Replace A0/A0b's full-logit assumption with one coordinated consumer; preserve AMP wrapper and Rewind |
| P2 / P3 | One scoped attention owner | Keep mask-presence dispatch explicit; do not silently fold backend changes into P0 |

The package imports `matmul_4bit` into its own namespace. Rebinding only the definition in `bitsandbytes.autograd._functions` after import does not update the package attribute used by `Linear4bit.forward`. Verify the actual call site before choosing a patch slot. [S33] [S88]

Give each slot one owner, save its true original once, and make installation idempotent by checking the actual target object. A single global installed flag is insufficient if a regression test replaces modules. Unsupported fake objects and metadata must fall back before accessing CUDA-only fields. Do not swallow a failed device launch and pretend ordinary fallback is safe.

M1 and N0 can be structurally compatible because one captures the original half projection output while the other changes weight decoding below that projection. M1 and B0 target different modules. I0 must be coordinated with M1 because it changes the base-forward promotion boundary. These relationships do not establish numerical equivalence or additive memory savings.

Avoid blanket FP16 input-casting decorators on R0 or B0: R0 deliberately uses FP32 residuals, and B0 has a mixed FP32-base/FP16-update contract. Keep the original LoRA GEMMs and trainable parameters outside B0's narrow custom Function. Internal no-grad execution is normal for M1 packing and N0's custom-Function context. [S72] [S85] [S86]

### 7.7 The pinned T4 attention backend is conditional

This static audit closes part of the earlier attention uncertainty. It does not establish what a particular installed binary actually selects. The relevant maximum-shape call has `B=4,T<=512,Hq=12,Hkv=2,D=128`, FP16 autocast, no cache and zero attention dropout. [S7] [S25] [S32] [S43]

**HF prepares two different calls.** With an attention mask, the pinned SDPA wrapper explicitly repeats K/V to 12 heads. With no mask and supported head width, it can enable GQA and pass the original two K/V heads. Mask construction can elide an all-valid mask even when the collator supplied one; mixed-length padded batches generally retain a mask. Actual batch lengths therefore matter to backend attribution. [S44] [S63]

**PyTorch's source gates on T4.** Pinned Flash and cuDNN attention require SM80 or newer. The memory-efficient route supports older devices, including an SM75 half path, but its dense head check passes `supports_grouped_query_attention=false`. It rejects the 12/2/2 GQA head mismatch. The math path remains the ordinary fallback when enabled. [S94] [S95] [S101]

| Call presented to SDPA | Q/K/V heads | Conditional result on the inspected T4 stack |
| --- | --- | --- |
| HF mask present, explicit repetition | 12 / 12 / 12 | Memory-efficient backend can qualify |
| HF mask absent, GQA enabled | 12 / 2 / 2 | Fused routes rejected; math fallback when enabled |
| Mask absent, K/V explicitly repeated in a separate experiment | 12 / 12 / 12 | Memory-efficient backend can become eligible |
| Memory-efficient support absent, disabled or another constraint fails | Depends on call | Math fallback or error according to enabled backends |

Build options, dtype, strides and enabled-backend settings still matter. Do not claim every masked call uses the efficient path. The existing SM75 CUTLASS attention route also does not depend on writing a new Triton `tl.dot` kernel; the Triton 3.6 lowering restriction is a separate issue. [S94] [S97] [S102]

**P2 is a distinct numerical experiment.** If actual maskless calls use math attention, explicit K/V repetition may make the existing efficient backend available. Record the unchanged baseline first, retain model masks/flags, and compare this branch separately. It changes backend arithmetic and can change the location of gradient rounding.

In the masked repeated-KV path, FP32 K/V are repeated before autocast; backward promotes each repeated half gradient before the repeat's FP32 sum. In the original maskless GQA math path, K/V are cast to half before math's promotion and internal repetition; repeat gradients accumulate in FP32 inside math before returning through the original half boundary. These are different accumulation schedules. A forward equality check or real-arithmetic argument cannot establish identical gradients. [S44] [S43] [S96]

Do not silently include this change in P0's RoPE packing. First restrict P0 to calls that already repeat K/V. Reusing an existing PyTorch backend may be useful to the complete candidate, but this report establishes neither an original-kernel contribution for P2 nor official eligibility.

**Saved-state and layout findings.** The memory-efficient derivative retains Q, K, V, output, bias, log-sum-exp and small RNG state. At maximum shape, one half 12-head tensor is 6 MiB; Q/K/V/O total 24 MiB per layer. FP32 log-sum-exp uses query length rounded to 32 and occupies 0.09375 MiB at T512. It does not save a full T-by-T probability matrix. In ordinary math attention the half inputs are promoted to FP32 unless the reduced-precision math option is enabled; one FP32 attention matrix is 48 MiB per layer, or **1344 MiB for 28 probability matrices** if that full branch is active. These are conditional payload components, not competing total peaks. [S55] [S96] [S97] [S101]

Efficient Q/K/V require unit last-dimension stride, not universal full contiguity. The native efficient output is allocated in B,T,H,D order and presented through a transposed view; HF transposes it back, so its final `.contiguous()` is normally a no-op on that route. Do not budget an unavoidable 6 MiB output copy for every attention layer. Math output can have a different layout; whether backward's incoming `grad_out.contiguous()` allocates depends on its layout. [S44] [S94] [S97] [S98]

Another apparent target disappears under specialization: the generic backward source contains an external delta reduction, but the SM75 half kernel computes delta internally. The predicate in `kernel_backward.h` selects that internal path for half on architectures other than SM70. Optimizing the generic fallback would not remove that work from this expected T4 specialization. [S98] [S99]

**P3: share one aligned additive mask.** A boolean B,1,T,T mask is normally shared across decoder layers, but native SDPA converts it to an additive mask for each call. At T512, each half mask occupies 2 MiB; 28 independent copies total 56 MiB. A single immutable per-forward additive mask would retain 2 MiB, a nominal **54 MiB payload difference**, assuming it follows the same efficient branch. [S32] [S96] [S97]

Preserve true-to-zero, false-to-negative-infinity semantics and the effective FP16 compute dtype. Keep mask presence and causal flags unchanged. Cache only by exact mask identity inside one invocation. Preserve head expansion as a view, rather than materializing a B,12,T,T mask.

The native preprocessing alignment is **8 elements** in the inspected code, even though an adjacent comment mentions 16. For T not divisible by 8, a plain contiguous half mask can still trigger a padded copy in each call. A proposed reusable buffer needs row pitch `ceil(T/8)*8`, logical shape B,1,T,T, aligned leading strides and unit last stride. Verify the actual bias alias saved by backward. P3 is a secondary memory/cast experiment; it is not a stand-alone argument for the 5% participation gate. [S96]

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


### 8.4 Concrete budgets for the first combined candidate

Rearranging the simplified participation expression, a kernel accelerated by `k` must have replaced baseline fraction `p >= k/(19+k)` to retain 5% by itself:

| Local speed multiplier | Required baseline fraction in this simplified model |
| --- | ---: |
| 2× | 9.52% |
| 4× | 17.39% |
| 8× | 29.63% |

The approximate 44% pointwise/cast category does not reveal RMSNorm's individual fraction. The required time must come from useful eligible kernels in the **combined candidate**, measured under the official denominator.

For intuition, the following are arithmetic examples using hypothetical candidate step times:

| Candidate step wall time | Eligible time for 5% | Eligible time for 6% |
| --- | ---: | ---: |
| 0.50 seconds | 25 ms | 30 ms |
| 0.75 seconds | 37.5 ms | 45 ms |
| 1.00 second | 50 ms | 60 ms |

These are not expected T4 times. A0 may contribute its two vocabulary reductions; R0 contributes active forward/backward kernels; M1 may contribute actual restoration. Do not count Python wrappers, ordinary PyTorch operations or excluded starter arithmetic as original kernels. Do not add percentages measured in separate candidates.

Track complete cold time, timed-step wall time, eligible event time and sampled peak separately. An extra second of startup work needs more than **15.625 ms average saving over 64 steps** merely to break even. Conversely, eliminating a few milliseconds of steady-state work may not repay broad autotuning or many sequence-length specializations.

The staged combined comparisons should test A0 + A0b against A0 alone, then add independently validated R0/M1 while checking eligibility margin. Summing the 1,184.682 MiB accuracy copy and 1,960 MiB retained-state difference is not a valid peak prediction; native CE backward, output ownership and reconstruction buffers interact. Section 4.3.4 separates the head phases.

### 8.5 Avoid counting transient buffers as retained pools

The head ledger in section 4.3.4 excludes body activations deliberately. The following ownership audit identifies distinct body components at maximum shape. It is source-derived under the expected dtypes and gradient requirements, not a GPU allocator trace. [S30] [S32] [S33] [S34] [S51] [S55]

| Component | Calculated payload / lifetime | Consequence for proposals |
| --- | --- | --- |
| Supplied SwiGLU gate/up saved pairs | 140 MiB FP32 per layer; 3920 MiB across 28 layers | M1's intended half representation is 1960 MiB before reconstruction |
| First active MLP backward under ideal M1 ownership | 1960 MiB half pairs plus 140 MiB current restored pair = 2100 MiB | At that phase the pair-pool difference is 1820 MiB, not 1960 MiB |
| SwiGLU output and down-projection cast | 70 MiB FP32 output plus 35 MiB half input for one active forward | MatMul4Bit does not save its A input; do not invent another 28-layer retained-output pool |
| RMSNorm backward inputs | 56 needed FP32 inputs total 672 MiB; inverse-RMS values total 0.4375 MiB | These inputs/statistics are required state, not automatically eliminated by R0 |
| LoRA A-weight-gradient inputs | Half dropout inputs total 336 MiB across Q/V and 28 layers | Adapter weight gradients still require them even when the base is frozen |
| Expected dropout input-gradient masks | 6 MiB per layer for 27 layers = 162 MiB | First frozen preattention input has no input gradient; its adapter weight gradients remain required |
| LoRA B-weight-gradient inputs | Rank-16 half inputs total 3.5 MiB | Small separate pool; do not double-count with base cast payloads |
| I0's shared base inputs | Temporary 6 MiB payload per QKV or gate/up invocation | Traffic hypothesis; not a retained-memory subtraction |

The norm input count excludes the first preattention normalization whose input gradient is unnecessary, and includes the final norm. Frozen gamma does not require saving a separate normalized output for a gamma gradient. R0 targets passes, launches and transient intermediates; it does not remove the original residual input required by input backward. The exact native graph still needs checking.

Dropout's FP32 output is another transient in the inspected path; the A weight gradient retains the half value consumed by its GEMM. Preserve the dropout computation and original input dtype. Distinct logical objects may alias or have nonoverlapping lifetimes, so this table is not an additive total-model budget.

Finally, the official memory score observes whole-process GPU residency. PyTorch's caching allocator can keep freed blocks reserved, and the prescribed expandable-segments setting does not make tensor payload equal `nvidia-smi` usage. Loading, compilation, asynchronous execution and worst-run aggregation also matter. Use allocation/lifetime evidence to explain a measured result; keep official measurement settings unchanged. [S27] [S100]

## 9. Resolved questions and remaining decisions

| Item | Current evidence | Remaining action |
| --- | --- | --- |
| Starter/harness | Immutable commit inspected | Run official files unchanged |
| Excluded operation | Supplied SwiGLU, including rewrites | Separate it from original credit |
| Configuration/pins | Exact recipe and installation versions known | Confirm supplied runtime/build |
| Compiler | Upstream 3.6.0 guard verified | Inspect PTX and throughput on T4 |
| Metric allocation/integration | Two-stage design, exact masks and early class-capture boundary identified | Implement guarded single-forward path; measure A0 |
| Native CE backward / A0b | Three-buffer phase identified; per-model loss slot and compact-state design; 72 CPU gradient comparisons | Native loss/gradient/fallback checks, Rewind ownership and paired A0 + A0b profile |
| FP32 activation provenance | Exact producer chain inspected | Confirm real boundary dtypes |
| M0 representation | Exhaustive finite CPU round trip and sampled derivatives pass | GPU/hook/lifetime/time/peak checks |
| M1 producer reuse | Producer, complete-apply hook scope and retained-callback ownership audited; two CPU ownership studies | Native hooks/version/alias checks, release, fallback and T4 cost |
| R0 mathematics | FP64 finite differences pass; FP32 stress sensitivity exposed | Exact eager CUDA comparison and full training |
| B0 epilogue | Active forward/backward rounding order audited; CPU oracle passes | Fresh-output autograd and GPU cast preservation |
| I0 input sharing | Separate-edge design, native view source and eight CPU mechanism cases | Actual custom-Function aliasing, accumulation order, dropout, lifetime and overhead |
| N0 decode | Both entry points, shape dispatch and exact rounding identified | Actual checkpoint byte checks, original kernel and profile |
| Cache ownership | Packed-state versus dense-autograd retention distinguished | Real lifetime and bounded subset measurements |
| Padding | Ordinary collator already removes common tail | Measure actual row lengths; test deterministic-only compaction |
| Attention dispatch | Pinned mask/GQA and SM75 backend gates traced | Record actual backend per batch; separate P2 from P0 |
| Attention mask/layout | Efficient saved state, output view, internal delta and 8-element mask alignment inspected | Verify P3 ownership and real copies; measure only observed costs |
| Optimizer | Pinned path predicts foreach; state-size calculation complete | Profile before O0 development |
| A1 head/AMP | Frozen-weight cast-cache exclusion, precision chain, row-chunk design and 48 CPU schedule cases | Native backward comparator, exact consumer integration and T4 memory/time tradeoff |
| Loss mathematics | CPU cases, finite differences and precision counterexamples | Native CE/GEMM rounding, nonfinite behavior and full training |
| T4 profile and results | Organizer estimates only | Reproduce baseline and candidates |
| Original-kernel share | Unmeasured | Measure every required run |
| Official correctness/ranking | Unmeasured | Required tests, trajectories and evaluation |

After baseline capture, establish A0/R0 and native M1 ownership independently. Compare A0 + A0b before A1's replay, then consider I0 and the observed attention branch. Advance custom GEMMs only with compiler and profile evidence. Preserve dropout, mode flags, initialization rounding, optimizer and logging. Compare complete cold runs with median time and worst sampled memory; nominal bytes and CPU equivalence are not GPU performance claims.

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


### 10.3 A0 strided metric reduction

Original script: `tmp/research/a0_stride_metrics_validation.py`; results: `tmp/research/a0_stride_metrics_results.json`. Python 3.12.14, NumPy 2.3.5, seed 20261031.

**Eighteen shape/tile checks passed** using FP32 partial and merge arithmetic against a dense FP64 reference on identical quantized inputs. The test's absolute tolerance was `5e-5`.

| Check | Observed result |
| --- | ---: |
| Maximum per-token entropy absolute error | 4.602182333002247e-7 |
| Maximum supervised token CE absolute error | 4.147331464920967e-7 |
| Maximum row CE absolute error | 3.402167951804813e-7 |
| Argmax on active positions | Exact in all cases |
| Supervised/correct counts | Exact in all cases |
| Original/shifted-view affine indexing | Exact in all cases |

Coverage includes vocabulary stride two, transposed batch/time storage, nonzero storage offset, row gaps, actual vocabulary 151936, cross-tile and final-tail ties, negative uniform logits, common offsets, peaked distributions, independent masks, empty supervised rows, a single time position and an empty attention denominator.

The actual-vocabulary case uses a small number of rows; it is not an allocation or timing test of the full `[4,512,151936]` workload. The 1.1875 MiB full-workload partial size is calculated separately.

These tests establish finite-input indexing and mathematical consistency in NumPy. They do not establish PyTorch CUDA rounding, GPU exponential/reduction behavior, nonfinite parity, training integration, runtime or eligibility.

### 10.4 M1 producer retention and manual gradients

Original script: `tmp/research/half_shadow_retention_check.py`; results: the corresponding `.json`. Python 3.12.14, NumPy 2.3.5, seed 61031.

The toy uses explicit half-input/half-output projection rules, FP32 SwiGLU arithmetic, separate branch gradient conversions and an upstream parameter derivative. It compares full FP32 saved arrays, M0 fresh packing and M1 retained half producers.

| Check | Observed result |
| --- | --- |
| Six outstanding states accessed in order 3, 0, 5, 2, 1, 4 | Passed |
| Forward, local gate/up derivatives, input derivative and upstream parameter derivative | Bitwise equal for both compact modes |
| Repeated unpack/manual backward | Identical |
| Promoted arrays released in compact modes | 12/12 weak references cleared |
| Full-representation control | Its 12 saved arrays remained alive |
| Original half-storage sharing | M1 true; M0 false |
| Same-valued different object | Strict identity rejected it |
| Unrelated FP32 fallback value | Preserved |
| Negative zero and smallest positive half subnormal | Restored in the CPU case |
| Exception during capture | Scoped state reset |

This is a Python ownership and explicit chain-rule toy, not an autograd-engine test. It does not execute bitsandbytes, PyTorch, Triton or CUDA. NumPy array collection is not evidence of real GPU allocator release, and the toy does not model all PyTorch alias/version behavior.

### 10.5 R0 derivative and FP32 sensitivity

Original script: `tmp/research/rmsnorm_cpu/check_rmsnorm_math.py`; results: `tmp/research/rmsnorm_cpu/rmsnorm_math_results.json`. NumPy 2.3.5, seed 20261030, width 1536, epsilon `1e-6`.

The study includes 21 cases: row counts 1, 3, 7, 33, 257 and 2048; zero/tiny/large and mixed-scale inputs; sparse spikes; zero/signed gamma; upstream scales 65536 and `1e-6`; strided arrays; cancellation and overflow stress.

**All 15,360 coordinate finite differences across ten FP64 cases passed.** Perturbations use `1e-5 * max(abs(x_j), row_RMS, sqrt(epsilon))`. The relative-L2 pass threshold is `2e-6`; the largest observed error is `6.004002104198291e-10`. These validate the real-arithmetic vector-Jacobian formula, not a derivative of discrete FP32 rounding.

| Ordinary-case FP32 versus FP64 statistic | Observed maximum |
| --- | ---: |
| Forward max absolute error / max reference magnitude | 1.2364149897651381e-7 |
| Normalized-form backward max absolute error / max reference magnitude | 1.8666829222075708e-7 |
| Normalized-form backward relative L2 error | 9.61717098458153e-8 |

The ordinary summary excludes the explicitly reported cancellation/extreme cases:

- **Radial cancellation:** the normalized-form gradient error was at most approximately `7.13e-7` in absolute value, but relative L2 error was about **7.5%**, because the true gradient was tiny. Small absolute error alone is not a guarantee of relative agreement.
- **Standalone `r^3`:** input scales `1e15`–`1e17` caused underflow in that intermediate and roughly 1–2% relative L2 error. Normalized/left-associated forms avoided that particular failure in this CPU experiment.
- **Scale `1e20`:** FP32 squaring overflowed, so the result no longer matched finite FP64 normalization. The prototype reports the mismatch rather than changing the normalization formula.

NumPy uses its own reduction order and reciprocal-of-square-root computation. This is not a comparison with pinned PyTorch eager autograd, GPU `rsqrt`, FMA contraction, Triton or AMP. The initial implementation must compare against the actual FP32 CUDA reference and preserve the accepted training trajectory.


### 10.6 B0 epilogue rounding oracle

Script: `tmp/research/lora_pass5/check_epilogue_order.py`; results: the corresponding `.json`. Python 3.12.14, NumPy 2.3.5, seed 51030.

The original oracle checked **522 forward entries and 522 backward entries**, comparing the proposed explicit-rounding formulas with the source-derived reference specification. Both matched bitwise, including overflow infinities. Inputs remained unchanged; output and adapter-gradient arrays had independent storage; the base gradient retained the unchanged upstream value.

The two reordering counterexamples in section 5.9 were reproduced. For `G32=2^-25`, converting before scaling gives zero, while scaling first gives `5.960464477539063e-08`. For `U16=32768`, scaling in half overflows, while early promotion gives finite FP32 65536.

This checks NumPy rounding and alias construction. It does not execute native PyTorch scalar promotion, autograd, CUDA conversions, Triton, dropout or full training. The reference is a specification derived from source, not a captured GPU trace.

### 10.7 Nested-NF4 indexing and rounding

Script: `tmp/research/nf4_pass5/check_nested_nf4_semantics.py`; results: `tmp/research/nf4_pass5/nested_nf4_results.json`. NumPy 2.3.5, seed 20261030, block size 64 and nested group size 256.

**Fourteen cases covering 2,850,640 logical elements passed all byte comparisons** for staged versus directly indexed decode, cached versus uncached scales, transpose-source indexing, odd-tail handling, block/group boundaries and code-index coverage.

Coverage includes every packed byte pattern and every nested code index; lengths around 63/64/65 and 16383/16384/16385; cross-row blocks; zero scales/signed zero; and representative Q/K/V dimensions. The scale table and metadata were synthetic, not loaded from the Qwen checkpoint.

A separate search over one million synthetic operand sets found eight final-half differences when scale reconstruction was modeled with a single rounding. Three witnesses were verified using exact rational nearest-FP32 rounding. One produces:

| Scale arithmetic | FP32 scale | Final FP16 weight |
| --- | ---: | ---: |
| Separate multiplication/addition rounding | 0.6028099060058594 | -0.05487060546875 |
| Single-round multiply-add | 0.6028099656105042 | -0.054901123046875 |

Early half conversion of both centroid and scale changed 273,345 synthetic outputs in that search. This is not an error frequency for the checkpoint. The exact witness proves an arithmetic hazard; it does not establish whether any inspected CUDA binary contracts the expression.

No upstream code was imported or executed. The legacy transposed-packed wrapper shim, actual quantization metadata, GPU conversion, Triton, cache ownership, performance and official correctness remain untested.

### 10.8 Padding, sequence isolation and illustrative RNG

Script: `tmp/research/padding_pass5/padding_index_rng_toy.py`; results: `tmp/research/padding_pass5/padding_index_rng_toy_results.json`. Python 3.12.14, NumPy 2.3.5, seed 20261030.

Synthetic shape `[4,8,16]` and lengths `[8,3,6,2]` give 19 valid and 13 padded token positions, with zero common all-pad columns. All assertions passed:

- Valid gather/scatter values, original positions and shifted per-row target counts were preserved exactly.
- Causal-only and causal-plus-right-padding masks agreed on valid queries.
- Naive concatenation changed the second example's first attention value from 10 to 4, demonstrating cross-example leakage.
- An illustrative sequential RNG changed 13 valid Q-mask and 32 valid V-mask elements when calls were compacted from 512 to 304 draws.
- Keeping both original full-shape calls and gathering afterward preserved the toy masks.

The RNG is deliberately **not a simulation of CUDA Philox**. Synthetic lengths are not dataset statistics. This is an indexing/context illustration, not proof of SDPA dispatch, native RNG parity, model gradients or performance.

### 10.9 A1 chunk schedule and gradient-precision study

Script: `tmp/research/head_pass6/check_head_amp_schedule.py`; results use the same stem with `.json`. Python 3.12.14, NumPy 2.3.5, seed 61030.

The explicit oracle uses FP64 dot accumulation followed by FP16 rounding, and an original FP32 softmax. It deliberately isolates dtype boundaries; it does **not** simulate cuBLAS, the CUDA reduction tree or bitwise native CE.

All **48** combinations passed: three shapes with vocabularies 257, 4097 and 151936; chunk lengths 1, 3, 8 and 20; incoming gradients `2^-10`, 1, 128 and 65536. Per-token loss and hidden-gradient bits matched the unchunked explicit oracle. Tests include a supplied denominator different from the supervised count, ignored rows/chunks, tails and final positions with no cross-example label shift.

Additional exact constructions:

| Question | Result in the original CPU construction |
| --- | --- |
| Can incoming scaling recover previously rounded-away values? | No: at V151936, D2044 and g65536, all 151935 non-target values become zero if materialized in half before scaling; correct placement yields half `0.0002110004425` |
| Does FP32 summation repair half-rounded vocabulary partials? | No: the section 4.4.3 witness gives full result 1, split result 0 |
| Can chunks normalize independently? | For token losses [1,4,9,16,25], one global mean is 11; the sum of means for chunks [4,1] is 32.5 |
| Is default per-logit filtering an identity? | The illustrative target-tile-only coefficient is -0.99915754 instead of real-arithmetic zero |

For the uniform example, the sum of correctly scaled half derivatives is approximately -0.00414777, already reflecting half rounding; early rounding followed by scaling gives -32.0625. Thus even the correct-boundary oracle should not be confused with exact-real cancellation.

The script also calculates the head weight/tile/traffic/FLOP tables in section 4.4. It does not measure memory, throughput, overflow frequency or checkpoint distributions. Native AMP, nonfinite fallback, training quality and participation remain untested.

### 10.10 Saved-hook callable ownership

Script: `tmp/research/integration_pass6/check_hook_callable_ownership.py`; results: `tmp/research/integration_pass6/hook_callable_ownership_results.json`. Seed 61030.

An original Python saved-node model retains both callbacks and payload, reflecting the ownership established by native source. It uses 28 pairs of small half-origin arrays, rather than allocating model-size tensors.

| Observation | Callback captures promoted pair | Module-level callbacks |
| --- | ---: | ---: |
| Saved-node models | 56 | 56 |
| Original FP32 arrays alive after scope | 56 | 0 |
| Capture frames alive after scope | 0 | 0 |
| Half-payload bytes | 59,024 | 59,024 |
| Restored values and repeated unpacks | Bitwise equal | Bitwise equal |
| Original FP32 arrays alive after releasing nodes | 0 | 0 |

A nested-scope exception also restored the previous ContextVar, preserved outer state, released the inner frame and left the final context empty. This supports the callback-ownership design; it does not execute native saved hooks, test PyTorch Tensor identity/versioning, or establish GPU allocator release.

### 10.11 A0b compact CE state and nonfinite rows

Script: `tmp/research/loss_pass7/check_compact_ce_state.py`; results use the same stem with `.json`. Python 3.12.14, NumPy 2.3.5, seed 71030. The dense oracle expresses the inspected log-softmax/NLL scalar order in original NumPy code. It is not an executed native PyTorch reference.

The finite study uses five rows, vocabularies 257, 4097 and 151936, common logit offsets 0/+1000/-1000, vocabulary tiles 512/4096, and incoming gradients 1, 128, 65536 and -2. Its denominator 11 deliberately differs from three supervised rows. Across 72 gradient comparisons, it compares 18,754,800 entries, including repeated comparisons under different settings.

| Result | Observed CPU value |
| --- | ---: |
| Reconstruction using identical saved row statistics | Bitwise equal in all 72 comparisons |
| Different tiled reduction tree: maximum token-loss absolute error | 9.5367431640625e-7 |
| Different tiled reduction tree: maximum half-gradient absolute error | 0.0009765625 |
| Different tiled reduction tree: maximum gradient relative L2 error | 1.973609837496578e-7 |
| Different tiled reduction tree: changed half-gradient entries across comparisons | 20 |

The first row isolates compact-state reconstruction. The remaining rows show why independent normalization is not automatically bitwise equal, even when its aggregate error is small.

A separate three-row construction places NaN, positive infinity and all-negative-infinity values in ignored rows. Both dense and compact scalar-order oracles produce nine NaN gradient entries; wrongly zeroing ignored rows removes all nine. Finite positions match. This supports the nonfinite warning, not a claim about native GPU NaN/FMA behavior.

The study validates neither CUDA/Triton exponentials and reduction trees, native autograd storage ownership, actual loss-scaler decisions, training trajectory, peak VRAM, speed nor participation.

### 10.12 I0 cast payloads, separate manual gradients and dropout scaling

Script: `tmp/research/projection_casts_pass7/check_cast_storage_edges.py`; results use the same stem with `.json`. Seed 71030. Eight cases cover two/three branches and shapes 1×5, 3×13, 17×37 and 33×1536.

The model gives each branch a distinct NumPy view of shared half storage. Forward values match separate casts under an explicit FP64-dot-then-half oracle. Manual gradients match when each half branch is promoted independently and added in the same FP32 order. The deliberately shared half-gradient edge fails the 2048+1 witness. Input arrays remain unchanged; payloads release while metadata-only edge models survive. Exact identity rejects a different input with the same shape.

This is a manual values/ownership model, not native custom-Function or autograd-engine execution. The fixed accumulation order is part of its oracle, not something established about the real engine.

The LoRA boundary counterexample uses FP32 input `0.5016999840736389` and keep probability 0.95. The CPU FP32 keep-mask scaling construction followed by half conversion gives `0.5283203125`; rounding the input to half before scaling gives `0.52783203125`. Of 10,001 constructed finite inputs, 2,554 differ between these two schedules. No random generator or native dropout mask was simulated.

The script also derives the 84 removed forward casts, 1512 MiB logical traffic difference and optional additional 672 MiB R0-read reduction. Those are arithmetic counts, not timing or retained-memory results.

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
| TRL 0.29.1 BaseTrainer | `8d70ef2f971cca8ee4d8c8319ccc221f491d1c52` |
| Transformers 5.16.1 Trainer | `ccb1110bd6897f548e288f56ec97c6473b453426` |
| TRL 0.29.1 utilities | `ca653ce681566207f54a03ccfefafad074fd9fcf` |
| PEFT 0.20.0 preparation | `ef435add45da04067c152219ed74926b622a26b2` |
| bitsandbytes 0.50.2 modules | `ebc0b09439c4620ef015b18b242328aa65f14c18` |
| bitsandbytes 0.50.2 autograd | `8a069bd101a089afc59b7d4d3d48e5dd399bc967` |
| Accelerate 1.14.0 utilities | `869b500f3452f809753586a435c6ea8d910b3f28` |

Release-tag links follow. Local manifests record additional retrieved files and hashes. These identities support the source audit, not a reproduced training environment.


### 11.4 Original CPU study identities

These scripts were authored for this research. They execute NumPy/Python experiments, not downloaded model or kernel implementations. Paths are relative to the research workspace; hashes identify the exact scripts behind section 10.

| Script | SHA-256 |
| --- | --- |
| `tmp/research/validate_online_vocab_reduction.py` | `6cc78607796f542028760c2f04eb566b622da1b9e29500962d55021866b665ea` |
| `tmp/research/saved_fp16_encoding_check.py` | `286761e51db233555f3a73f9c34efce93b41ec00d501b3890063ff52456f5832` |
| `tmp/research/a0_stride_metrics_validation.py` | `46d04c2edd69507d8d43b8ff8f8151206e1d78e4f0cf3079e49623627078cfbe` |
| `tmp/research/half_shadow_retention_check.py` | `f5e154168279e6d7ac9f91df16f9b1dd7bd7e3e959b912064d9891e96f45324c` |
| `tmp/research/rmsnorm_cpu/check_rmsnorm_math.py` | `f42e7d0a63defb8305ea17885e479e9d2108f3363cbfca4a09ad9dd06625e585` |
| `tmp/research/lora_pass5/check_epilogue_order.py` | `bd6c924ee9ee0a48ed02c34b75cf40d3f017310fded846e0d9f07a02f41ccb5b` |
| `tmp/research/nf4_pass5/check_nested_nf4_semantics.py` | `e2c9f8e0af1d9bb38efe3c4aa9fd9b2260986e0ec68add4fd5aed8cb057e82b0` |
| `tmp/research/padding_pass5/padding_index_rng_toy.py` | `8ab1206ddda774ceb4e6d9568780f8447a843850b51d11e845132a4546aa1f11` |
| `tmp/research/head_pass6/check_head_amp_schedule.py` | `2f66151dcd05309fab6e89e51645cd3b2d6264bf0ec38a30b9323948e6599e70` |
| `tmp/research/integration_pass6/check_hook_callable_ownership.py` | `43118d77b9b4f2246ba5eaca8877eefcdd21e38558ff0a589e8c25e2c7a084ca` |
| `tmp/research/loss_pass7/check_compact_ce_state.py` | `a24369878faf26d1a06b744210810d47872b2a12b2b6ebae5ec43efc890e9bd4` |
| `tmp/research/projection_casts_pass7/check_cast_storage_edges.py` | `2385d02285810b9550e38f5df43260bfd7677261f503acbb20db532747a5fe5f` |

The report is the review deliverable; the scripts/results are research intermediates. No contest implementation or GPU benchmark is implied by these identities.


### 11.5 Fifth-pass source snapshots

New release-pinned snapshots cover PEFT/engine derivative semantics, nested NF4 CUDA/functional decoding, public data preparation, collator configuration, dropout, mask construction and optimizer dispatch. The primary source links are S51–S69. Original package code was inspected as text only.

Local manifests: `tmp/research/lora_pass5/provenance.json`, `tmp/research/nf4_pass5/source_provenance.json`, `tmp/research/padding_pass5/provenance.json` and `tmp/research/optimizer_pass5/provenance.json`. They retain exact URLs and content hashes.

| Additional pinned file | Git blob SHA |
| --- | --- |
| Official public data preparation | `2a1378e968de7b6af53f08c14059894773498206` |
| TRL 0.29.1 SFTConfig | `3234b6d76a4133745441b0f81b0fdb0edeebefed` |
| Transformers 5.16.1 masking utilities | `7bd46c8bb287ad855b2a3f5af3fb78e3dcac152c` |
| PyTorch 2.11 CUDA dropout | `538b32a71278da071b29256847e220174016b65f` |

No public or hidden dataset rows, model weights, GPU timing or native dropout sequences were collected in this pass. Actual token lengths remain a runtime measurement.

### 11.6 Sixth-pass source snapshots

New snapshots trace the head's native linear/CE/autocast/backward path and saved-hook ownership. Local manifests are `tmp/research/head_pass6/provenance.json`, `tmp/research/integration_pass6/provenance.json` and `tmp/research/hf_head_pass6/provenance.json`. They distinguish newly fetched files from reused pinned sources.

| Additional reference | Identity / retrieval qualification |
| --- | --- |
| PyTorch native head, CUDA loss and saved-hook internals | v2.11.0 release-tag files; content hashes in the manifests |
| HF fused linear CE | Observed v1 tree commit `00e29bdf4e142fafa8f558be414c7a9f344d61d0`; source read via a web extraction with line numbers, not a byte-exact downloaded snapshot |
| TRL PR 5575 merge | `9bcf7294343014eb1ab7036811a908346e4b9097` |
| CCE source audit | `3de376c106a1916bc5e1b619f9c77c87a461ee1c` |
| Liger source audit | `781a8aef64bf6105bdefb5f4c991e0d14c08e793` |

The HF pinned URL is provided for identification; the successful source view used its observed v1 reference. No third-party code was imported or executed. Two new original CPU studies bring the total to ten; no model weights, dataset rows or GPU results were collected.

### 11.7 Seventh-pass source snapshots and validation scope

New release-pinned source reads cover native CE allocation/decomposition, the model loss-property override, custom-Function output views, SDPA selection and CUTLASS attention layouts/backward, plus CUDA allocator semantics. Primary links S89–S102 supplement the earlier pinned dependency sources.

| Audit | Local provenance manifest | Scope |
| --- | --- | --- |
| CE and body-memory ownership | `tmp/research/memory_pass7/provenance.json` | New LossNLL/SoftMax/allocator text plus reused pinned trainer, derivatives and harness |
| Per-model loss installation | `tmp/research/loss_pass7/provenance.json` | Transformers 5.16.1 model base class; hash identifies the local text snapshot |
| I0 views and projection boundaries | `tmp/research/projection_casts_pass7/provenance.json` | PyTorch 2.11 custom-Function source, pinned bnb/PEFT, original study identities |
| T4 attention routes | `tmp/research/attention_pass7/provenance.json` | Nine PyTorch 2.11 selector/CUDA/CUTLASS files and reused HF source |

All downloaded implementation files were inspected as text. Neither PyTorch nor downloaded model/kernel code was imported or executed. The arithmetic-only `memory_pass7/account_memory.py` is not counted as a numerical study. Two new original studies in sections 10.11–10.12 bring the total to twelve.

Representative local-text SHA-256 identities:

| File | SHA-256 |
| --- | --- |
| PyTorch `LossNLL.cpp` | `de1759a872fb3e5dd34e040ebb904712db343ace504c7c0a27292b82eb38af2e` |
| PyTorch `custom_function.cpp` | `32f6236fadae2a8c0b480511f3a7e31ee22fee42eecd0a5896c2794e6a4ea715` |
| PyTorch CUDA `sdp_utils.cpp` | `cafe7ea84e68f9c0b480ed7dd5d62f73e3263c5e46e05e4e19cb580d25b10627` |
| Transformers `modeling_utils.py` | `0afe22f1d7f7aa4946991ef6dcd83f22034802f59cb849e8ba73f00a5879a6e5` |

These identities support source reconstruction. They do not identify an installed or benchmarked contest environment. Actual model weights, dataset tokenization, GPU kernels, allocator traces and official tests remain unexecuted in this research.

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
| [S45] | Transformers 5.16.1 Trainer | Loss delegation, item-count and scaling behavior |
| [S46] | TRL 0.29.1 BaseTrainer | Inheritance without a compute-loss override |
| [S47] | PyTorch 2.11 Tensor.detach | Detached view and shared-storage contract |
| [S48] | Python 3.13 contextvars | Scoped state and restoration |
| [S49] | Python 3.13 weakref | Non-owning object identity references |
| [S50] | NVIDIA Turing Tuning Guide | Warp/register occupancy constraints |
| [S51] | PEFT 0.20.0 four-bit LoRA | Active clone/AMP scale/add path |
| [S52] | PyTorch 2.11 autograd engine | Gradient dtype conversion on edges |
| [S53] | PyTorch 2.11 input metadata | Gradient dtype recorded from tensor |
| [S54] | PyTorch 2.11 variable metadata | Default gradient dtype |
| [S55] | PyTorch 2.11 derivative definitions | Add/clone/multiply/matrix backward dependencies |
| [S56] | PyTorch 2.11 manual derivatives | Scalar multiplication backward |
| [S57] | bitsandbytes 0.50.2 functional API | Nested scales, dtype and transpose wrapper |
| [S58] | bitsandbytes 0.50.2 CUDA kernels | Nibble order, indexed scale decode, canonical NF4 table |
| [S59] | bitsandbytes 0.50.2 SM75 GEMM | In-kernel nested scale reconstruction |
| [S60] | Official public data preparation | Pinned dataset revision and row ranges |
| [S61] | TRL 0.29.1 SFTConfig | Padding multiple default |
| [S62] | PyTorch 2.11 CUDA dropout | Shape/layout-dependent RNG mapping and reservation |
| [S63] | Transformers 5.16.1 mask construction | Conditions for mask retention/elision and sequence isolation |
| [S64] | Transformers 5.16.1 optimizer selection | Ordinary versus fused AdamW arguments |
| [S65] | PyTorch 2.11 Adam implementation | Foreach dispatch and state allocation |
| [S66] | PyTorch 2.11 AdamW implementation | Adam delegation with decoupled weight decay |
| [S67] | PyTorch 2.11 optimizer helpers | Compatible foreach parameter/device selection |
| [S68] | PyTorch 2.11 AdamW reference | Foreach defaults, memory tradeoff and zero/None semantics |
| [S69] | NVIDIA CUDA introduction | T4 has 40 SMs |
| [S70] | PyTorch 2.11 autocast cache | Frozen-weight exclusion from cached casts |
| [S71] | PyTorch 2.11 native linear | Linear-to-matmul head path |
| [S72] | PyTorch 2.11 AMP decorators | Explicit input casting and backward autocast state |
| [S73] | PyTorch 2.11 GradScaler | FP32 dynamic incoming scale |
| [S74] | PyTorch 2.11 CUDA NLL | Ignored target and native backward behavior |
| [S75] | PyTorch 2.11 CUDA softmax | Stored log-probability rounding and backward expression |
| [S76] | HF fused linear CE card | Compact metric masks and API |
| [S77] | HF fused linear CE observed v1 | Projection precision, bounded backward workspace and H100 tuning |
| [S78] | TRL PR 5575 | Published chunk-size memory/time study |
| [S79] | TRL PR 5575 merge source | Later chunked loss implementation and PEFT restriction |
| [S80] | Pinned CCE threshold helper | Automatic filtering epsilon |
| [S81] | Pinned CCE backward | Filtering, upstream scaling and logit rounding |
| [S82] | Pinned CCE presets | Classifier versus hidden-gradient filtering options |
| [S83] | Pinned Liger fused linear CE | Token-chunk heuristic and precomputed gradients |
| [S84] | PyTorch 2.11 Python saved hooks | Retained pack/unpack callables and payload |
| [S85] | PyTorch 2.11 SavedVariable | Hook packing grad mode and mutation/version checks |
| [S86] | PyTorch 2.11 custom Function internals | Deferred saved-variable creation and disabled recording |
| [S87] | PyTorch 2.11 numerical accuracy | Shape/reduction changes and lack of bitwise guarantees |
| [S88] | bitsandbytes 0.50.2 package aliases | Actual exported matmul_4bit binding |
| [S89] | PyTorch 2.11 native NLL/CE | Hard-label log-softmax decomposition and full NLL gradient shape |
| [S90] | PyTorch 2.11 native softmax | Distinct full-shaped log-softmax backward result |
| [S91] | Transformers 5.16.1 model base | Per-model loss_function property and setter |
| [S92] | PyTorch 2.11 custom-Function output handling | Returned-input view wrapping and independent gradient edges |
| [S93] | PyTorch 2.11 Tensor.to reference | Same dtype/device conversion returns self by default |
| [S94] | PyTorch 2.11 CUDA SDPA selector | SM gates, efficient-attention constraints and dense GQA rejection |
| [S95] | PyTorch 2.11 SDPA head checks | Dense head-count compatibility and GQA gate |
| [S96] | PyTorch 2.11 native attention | Math promotion/GQA, mask conversion and 8-element bias alignment |
| [S97] | PyTorch 2.11 CUDA attention forward | Efficient output/LSE layout and bias handling |
| [S98] | PyTorch 2.11 CUDA attention backward | Gradient layout and conditional delta reduction |
| [S99] | PyTorch 2.11 CUTLASS attention backward | SM75 half specialization computes delta internally |
| [S100] | PyTorch 2.11 CUDA notes source | Caching allocator visibility and expandable segments |
| [S101] | PyTorch 2.11 SDPA reference | Backend-dependent numerics, math precision and experimental GQA support |
| [S102] | PyTorch 2.11 CUTLASS attention forward | Architecture and efficient-kernel specialization |

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

[S45]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py "Pinned Transformers Trainer"
[S46]: https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/base_trainer.py "Pinned TRL BaseTrainer"
[S47]: https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.detach.html "PyTorch detached shared-storage tensor"
[S48]: https://docs.python.org/3.13/library/contextvars.html "Python context variables"
[S49]: https://docs.python.org/3.13/library/weakref.html "Python weak references"
[S50]: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html "NVIDIA Turing tuning"

[S51]: https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py "PEFT 0.20.0 four-bit LoRA"
[S52]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/engine.cpp "PyTorch 2.11 autograd engine"
[S53]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/input_metadata.cpp "PyTorch 2.11 input metadata"
[S54]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/variable.cpp "PyTorch 2.11 variable metadata"
[S55]: https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml "PyTorch 2.11 derivative definitions"
[S56]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/FunctionsManual.cpp "PyTorch 2.11 manual derivatives"
[S57]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py "bitsandbytes 0.50.2 functional API"
[S58]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/kernels.cu "bitsandbytes 0.50.2 CUDA kernels"
[S59]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/gemm_4bit_sm75.cu "bitsandbytes 0.50.2 SM75 GEMM"
[S60]: https://example.invalid/redacted-github "Official public data preparation"
[S61]: https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_config.py "TRL 0.29.1 SFTConfig"
[S62]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu "PyTorch 2.11 CUDA dropout"
[S63]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/masking_utils.py "Transformers 5.16.1 mask construction"
[S64]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer_optimizer.py "Transformers 5.16.1 optimizer selection"
[S65]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/adam.py "PyTorch 2.11 Adam implementation"
[S66]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/adamw.py "PyTorch 2.11 AdamW implementation"
[S67]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/optimizer.py "PyTorch 2.11 optimizer helpers"
[S68]: https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html "PyTorch 2.11 AdamW reference"
[S69]: https://developer.nvidia.com/blog/even-easier-introduction-cuda/ "NVIDIA CUDA introduction"

[S70]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.cpp "PyTorch 2.11 autocast cache"
[S71]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Linear.cpp "PyTorch 2.11 native linear"
[S72]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/amp/autocast_mode.py "PyTorch 2.11 AMP decorators"
[S73]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/amp/grad_scaler.py "PyTorch 2.11 GradScaler"
[S74]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Loss.cu "PyTorch 2.11 CUDA NLL"
[S75]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/SoftMax.cu "PyTorch 2.11 CUDA softmax"
[S76]: https://huggingface.co/kernels/trl-lib/fused-linear-ce "HF fused linear CE card"
[S77]: https://huggingface.co/kernels/trl-lib/fused-linear-ce/blob/00e29bdf4e142fafa8f558be414c7a9f344d61d0/build/torch-cuda/fused_linear_ce.py "HF fused linear CE observed v1"
[S78]: https://github.com/huggingface/trl/pull/5575 "TRL PR 5575"
[S79]: https://github.com/huggingface/trl/blob/9bcf7294343014eb1ab7036811a908346e4b9097/trl/trainer/sft_trainer.py "TRL PR 5575 merge source"
[S80]: https://github.com/apple-aiml-research/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/cut_cross_entropy/utils.py "Pinned CCE threshold helper"
[S81]: https://github.com/apple-aiml-research/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/cut_cross_entropy/cce_backward.py "Pinned CCE backward"
[S82]: https://github.com/apple-aiml-research/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/cut_cross_entropy/cce_utils.py "Pinned CCE presets"
[S83]: https://github.com/linkedin/Liger-Kernel/blob/781a8aef64bf6105bdefb5f4c991e0d14c08e793/src/liger_kernel/ops/fused_linear_cross_entropy.py "Pinned Liger fused linear CE"
[S84]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/python_saved_variable_hooks.cpp "PyTorch 2.11 Python saved hooks"
[S85]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/saved_variable.cpp "PyTorch 2.11 SavedVariable"
[S86]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/python_function.cpp "PyTorch 2.11 custom Function internals"
[S87]: https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html "PyTorch 2.11 numerical accuracy"
[S88]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/__init__.py "bitsandbytes 0.50.2 package aliases"
[S89]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/LossNLL.cpp "PyTorch 2.11 native NLL/CE"
[S90]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/SoftMax.cpp "PyTorch 2.11 native softmax"
[S91]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py "Transformers 5.16.1 model base"
[S92]: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/custom_function.cpp "PyTorch 2.11 custom-Function output handling"
[S93]: https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.to.html "PyTorch 2.11 Tensor.to reference"
[S94]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp "PyTorch 2.11 CUDA SDPA selector"
[S95]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h "PyTorch 2.11 SDPA head checks"
[S96]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/attention.cpp "PyTorch 2.11 native attention"
[S97]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/attention.cu "PyTorch 2.11 CUDA attention forward"
[S98]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/attention_backward.cu "PyTorch 2.11 CUDA attention backward"
[S99]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h "PyTorch 2.11 CUTLASS attention backward"
[S100]: https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/cuda.rst "PyTorch 2.11 CUDA notes source"
[S101]: https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html "PyTorch 2.11 SDPA reference"
[S102]: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h "PyTorch 2.11 CUTLASS attention forward"
