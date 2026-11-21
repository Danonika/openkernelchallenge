# Qwen Kernel Optimization: Research and Experiment Review

- **Project:** Decentrathon 6.0 / Soup Kernel Challenge
- **Research date:** 30 October 2026
- **Edition:** Markdown version of the research review
- **Status:** Research complete; proposed implementation strategy awaits review. No contest kernels have been implemented or benchmarked on T4 in this work.
- **Purpose:** A self-contained reference for team review and continuation in another session.

## Session handoff

**Objective:** improve the complete training runtime and peak GPU memory of the fixed Soup workload while preserving the task, required metrics, gradients and adapter quality.

**Actual target:** Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4 weights, LoRA rank 16 and alpha 32 on `q_proj` and `v_proj`, batch 4, sequences up to 512, 64 training steps, one Kaggle T4 with FP16. This is a separate workload from the earlier H200/Open Kernel discussion. [S1]

**Proposed priorities:** develop two original tracks in parallel: **A. loss plus entropy, accuracy and per-example logging with bounded temporary memory; B. fusion of profiled elementwise, casting and LoRA operations, including backward propagation.** Treat NF4 unpacking plus GEMM as a separate, higher-risk experiment. These are recommendations, not measured results.

**Critical rules:** ideas from published kernels are allowed with attribution; copied code and ready-made optimization libraries are forbidden. Useful original Triton kernels must collectively occupy at least 5% of the solution's measured training time from step 3 onward in every measured run. Starter example kernels are excluded even when rewritten. [S1]

**Unresolved:** the starter repository and harness could not be inspected. Its example operation, exact tolerance-normalization formulas, baseline implementation details and cache handling remain unverified. A failed retrieval does not establish whether the repository is public or private.

**Next evidence needed:** inspect the starter example and harness; establish a baseline on the target T4; obtain a detailed breakdown of the reported 44% elementwise/casting category. Do not invent runtime, memory, speedup, eligibility or leaderboard results.

**Source of truth:** the supplied [challenge brief][S1] and, once accessible, its official [starter repository](https://example.invalid/redacted-github). Published benchmarks below remain attributed to their original hardware and baselines.

## 1. Recommendation and rationale

Existing public work confirms that the relevant Qwen operations are established optimization targets. Unsloth has an exact Qwen2.5-1.5B quantized-model repository linked to T4 training notebooks. Liger explicitly supports optimized operations for Qwen2.5. The competition opportunity is to independently implement suitable algorithms, integrate them correctly into Soup and measure their effect under the fixed workload. [S3] [S4]

The organizers provide this approximate GPU step profile: [S1]

| Work category | Reported share | Proposed priority |
| --- | ---: | --- |
| Elementwise operations and dtype casts | 44% | Primary speed target; obtain a more detailed breakdown first |
| Matrix multiplication | 36% | Investigate selectively; a new GEMM must outperform the existing implementation |
| Vocabulary softmax | 7% | Memory opportunity together with the output projection and metric consumers |
| Attention | 5% | Lower initial priority for this workload |
| NF4 unpacking | 5% | Higher-risk opportunity when fused with its consumer |

These shares are approximate GPU-time measurements. They are not a precise decomposition of end-to-end elapsed time. The 44% category includes multiple operations; it is not the measured cost of RMSNorm, SwiGLU or LoRA individually.

**Recommended working arrangement:** pursue Track A and Track B independently, then test their combination. Keep NF4/GEMM experiments isolated until they pass correctness checks and demonstrate a useful whole-run improvement.

## 2. Fixed workload and submission requirements

### 2.1 Target configuration

The following values come from the brief, not from a locally reproduced environment. [S1]

| Setting | Required value |
| --- | --- |
| Training application | Soup 0.77.0 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Base weights | 4-bit NF4 |
| Adapter | LoRA, `r=16`, `alpha=32` |
| Adapter targets | `q_proj`, `v_proj` |
| Batch size | 4 |
| Sequence length | Up to 512 tokens |
| Training workload | 256 examples, one epoch, 64 steps |
| Scored hardware | One free Kaggle T4 |
| Execution precision | FP16; no BF16 |
| Specified environment | Python 3.13, PyTorch 2.11, Triton 3.6.0 |
| Integration | Original kernels in `solution/`, connected through `apply()` |

Final evaluation uses hidden training/evaluation rows and another seed with the remaining task settings fixed. Specialization must depend on legitimate model dimensions and input shapes, not memorized examples. [S1]

### 2.2 Originality and allowed reuse

- Attributed ideas from other implementations are allowed.
- Copied kernel code and ready-made optimization libraries are forbidden, including Unsloth, Liger, xFormers, FlashAttention, torchao and similar libraries; the full list is in the harness.
- Useful original `@triton.jit` kernels must collectively account for **at least 5% of the solution's training time from step 3 onward, in every measured run**.
- This threshold concerns their runtime **after optimization**, not the baseline fraction they replace.
- Compiler-generated kernels, unused results and starter-example kernels do not count. Rewriting the starter example does not make it eligible.
- Open-source licensing and attribution do not override the competition's restrictions on copied implementations. [S1]

An extremely fast but small custom kernel may fail the participation threshold. Establish sufficient useful original work without adding artificial computation.

### 2.3 Task invariants

Preserve the model, data within the supplied run, batch contents and order, number of steps, optimizer, learning-rate schedule, LoRA settings including dropout, trainable-parameter set and required outputs. [S1]

Do not switch built-in modes such as `gradient_checkpointing`, `activation_offloading`, `fast_lora`, `use_cut_ce` or packing. Custom removal of padding inside an existing batch is allowed; mixing examples across different batches is not. Do not alter the measurement/evaluation scripts or their allocator settings. [S1]

### 2.4 Material correctness and admission conditions

| Check | Requirement from the brief |
| --- | --- |
| Hidden-set adapter evaluation loss | No more than 0.5% above baseline |
| First-step training loss | Within 1% |
| Average training-curve deviation | Within 3% |
| Entropy, accuracy and per-row losses in `rewind.jsonl` | Within 3% |
| Token counts per step | Match exactly |
| Whole-command runtime | No more than 10% above baseline |
| Peak GPU memory | No more than 10% above baseline |
| Existing Soup tests | All 79 required tests in the specified five files pass |
| Reproducibility | Clean Kaggle notebook `Run All` works without edits |
| Documentation | Required README and submission artifacts are provided |

Source: brief, especially sections 5, 7 and 8. The harness was not inspected, so this report does not invent the detailed normalization formulas used by its comparisons. [S1]

### 2.5 Measurement and score

- **Time:** the entire `soup train` command, including model loading, kernel compilation and adapter saving.
- **Order:** baseline, solution, solution, baseline; average the two measurements for each side.
- **First two steps:** excluded only from the original-kernel participation calculation, not from whole-command timing.
- **Cache behavior:** the brief does not establish that caches are cleared before every repeat.
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

## 4. Track A: loss and metrics with bounded memory

**Hypothesis:** reduce large temporary tensors while preserving every consumer of model scores.

### 4.1 Why the output head matters

The official model vocabulary has **151,936 entries**. At the maximum configured batch shape, one complete logits tensor contains: [S7] [S1]

```text
4 * 512 * 151,936 = 311,164,928 elements
FP16: 622,329,856 bytes, approximately 0.62 GB
FP32: 1,244,659,712 bytes, approximately 1.24 GB
```

These are decimal-unit tensor-size calculations, not observed Soup peaks. Additional copies, gradients and intermediates can increase memory use.

Liger documents chunked linear cross-entropy; CCE evaluates vocabulary reductions without retaining all logits globally. Both motivate limiting temporary output-head storage. The proposal below also handles Soup's metrics. [S4] [S6]

### 4.2 Proposed first implementation

1. **Inspect all consumers.** Locate training loss, entropy, accuracy and per-row rewind losses. Record masks, causal-label shifting, reductions, normalization and output requirements.
2. **Chunk the output head.** Process manageable token chunks with allowed GEMMs. Use original Triton reductions for loss and metrics; avoid retaining all batch logits simultaneously.
3. **Preserve summaries.** Keep the required token/row statistics and baseline averaging conventions. Handle ignored labels exactly.
4. **Bound backward storage.** Use an explicit gradient/recomputation design so autograd does not retain every chunk's logits. Preserve input gradients and every required trainable-weight gradient.
5. **Tune after validation.** Sweep a small number of chunk sizes on T4. Smaller GEMMs and recomputation can cost time, so assess the measured memory benefit together with runtime and admission limits.

### 4.3 More ambitious vocabulary streaming

For a token, let `z_j` be vocabulary scores, `y` its target and `p_j` the softmax probability. Required summaries can be expressed as:

```text
L          = log(sum_j exp(z_j))
CE         = L - z_y
Entropy    = L - sum_j p_j * z_j
Prediction = argmax_j z_j
```

Accumulate these quantities over vocabulary tiles with stable maximum-shifted reductions. Recompute tiles in backward where needed. The equations specify the mathematical objective; they do not establish bitwise agreement with a particular FP16 execution.

**Main integration risk:** loss-memory savings can disappear if metric logging independently materializes the same huge tensors. Validate loss, entropy, accuracy, per-row outputs and gradients as one integrated change.

## 5. Track B: elementwise operations, casts and LoRA

**Hypothesis:** reduce memory passes and kernel-launch overhead in the actual measured bottlenecks.

Qwen2.5 uses operations already targeted by Liger. Start with a detailed profiler breakdown to identify the expensive normalization, activation, casting or LoRA paths in Soup. No individual operation has been assigned the full 44% category. [S4] [S1]

| Candidate | Proposed experiment | Required checks |
| --- | --- | --- |
| SwiGLU | Fuse activation and multiplication, including backward | Casting order, gradients, saved intermediates and recomputation cost |
| RMSNorm | Fuse row reduction, scaling and appropriate conversions | Epsilon, dtypes, accumulation and input gradients; starter eligibility must be checked |
| RoPE and nearby casts | Combine compatible transformations or layout work where the call graph permits | Positions, strides, precision and whether the baseline is already optimized |
| LoRA branch | Specialize small-rank operations and fuse compatible scaling, addition or casts | Fixed rank/alpha/dropout, RNG behavior, optimizer, forward results and gradients; tune q/v projections separately |

### 5.1 Fixed model dimensions

The official model configuration provides: [S7]

| Property | Value |
| --- | ---: |
| Hidden width | 1536 |
| MLP intermediate width | 8960 |
| Layers | 28 |
| Query heads | 12 |
| KV heads | 2 |
| Derived head width | 128 |
| RMSNorm epsilon | `1e-6` |
| Vocabulary | 151,936 |
| Tied embeddings | Yes |

Confirm live module layouts before specializing. The stored BF16 dtype metadata does not override the challenge's FP16 requirement.

### 5.2 LoRA semantics to preserve

A schematic adapter projection is:

```text
Y = X W^T + (alpha / r) * Dropout(X) A^T B^T
```

Preserve baseline bias terms where present. Here `W` is the frozen base projection and `A`, `B` form the trainable low-rank update. Fusion may remove intermediate passes, but it must preserve dropout and gradient behavior.

Frozen base weights do not require base-weight gradients; upstream input gradients remain necessary. This is an implementation consideration, not evidence that the current baseline performs unnecessary gradient work or that LoRA is the principal bottleneck.

### 5.3 Higher-risk branch: NF4 unpacking plus GEMM

The goal is to unpack weights where they are consumed and avoid a full dense intermediate. Unpacking alone accounts for approximately 5% of the supplied profile. Benefits involving the broader 36% GEMM category require improving that work too. [S1]

Correct codebooks, quantization scales, layouts and backward behavior are essential. A naive replacement GEMM can erase any benefit from fusion. Keep this branch separate until it passes correctness and improves the actual task measurements.

## 6. Experiment plan

The following stages are proposed work; they have not been executed in this research session.

| Stage | Action | Evidence needed to proceed |
| --- | --- | --- |
| 0. Establish baseline | Inspect starter/harness, run the supplied task on T4 and capture profiler/output artifacts | Full time, sampled peak memory, losses, logs, environment and resolution of the starter-example exclusion |
| 1. Split work | Track A for loss/metrics; Track B for profiled pointwise/cast/LoRA fusion; optional separate NF4 branch | Original code, identified integration sites and forward/backward comparisons |
| 2. Integrate one change | Connect through `apply()` and run relevant correctness checks plus the complete task | Measured benefit, preserved semantics and acceptable quality/log outputs |
| 3. Combine winners | Measure the combined solution and original-kernel contribution; tune a small justified configuration set | Gains survive interaction; useful original kernels meet the 5% gate after optimization |
| 4. Reproduce and submit | Clean notebook run; finalize README, commit, notebook, presentation and demo | All required checks pass; saved results match claims; sources and limitations are documented |

The brief requires the platform-created team repository, submission metadata, Kaggle notebook, presentation and demo video. Finalize artifacts according to its exact instructions. [S1]

### 6.1 Experiment comparison table

| Variant | Whole-command time | Sampled peak memory | Quality and logs | Original-kernel participation |
| --- | --- | --- | --- | --- |
| Baseline | Not measured here | Not measured here | Reference to establish | N/A |
| Track A only | To measure | To measure | Check | Measure |
| Track B only | To measure | To measure | Check | Measure |
| Track A + B | To measure | To measure | Check | Measure |
| A + B + NF4/GEMM | Optional experiment | Optional experiment | Check | Measure |

**Keep/reject rule:** retain a candidate only when it remains correct, eligible and beneficial under official measurements. A memory improvement can justify modest extra computation only within the runtime admission limit and the overall scoring tradeoff.

For agents, specify the exact model, operation, GPU, dtype, representative shapes, required gradients and integration point. Require correctness evidence and whole-run impact alongside isolated timings. Hugging Face documents this development workflow. [S9]

## 7. Performance expectations and scoring

### 7.1 Illustrative whole-command model

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

### 7.2 Score model after measurement

```text
s = baseline time / solution time
m = baseline peak / solution peak

speed points  = 35 * ln(s) / ln(best admitted s)
memory points = 30 * ln(m) / ln(best admitted m)
```

Apply the brief's threshold and floor rules: ratios below 1.05 receive zero for that metric, and points do not go below zero. Eligibility contributes 20 points and README/reproducibility contributes up to 15. Relative scores depend on other admitted teams, so the formulas do not establish our rank. [S1]

Compilation and loading count in total time. Limit compilation variants and autotuning overhead because only 64 training steps amortize them. Keep measured speed and measured memory as separate outcomes.

## 8. Open questions and review decisions

| Item | Current state | What resolves it |
| --- | --- | --- |
| Starter example operation | Unknown | Read current starter source |
| Exact harness behavior and tolerance definitions | Not inspected | Review harness code and run its checks |
| Current Soup operation implementations | Not inspected | Profile and inspect the installed pinned version |
| T4 baseline time/memory | Only organizer-reported approximations available | Reproduce the supplied baseline |
| Detailed 44% category breakdown | Unknown | Operation-level profile |
| Loss/metric consumer integration | Proposed design | Inspect call graph and validate all outputs |
| Eligible original-kernel runtime share | Unmeasured | Measure solution kernels in each required run |
| End-to-end improvement and ranking | Unmeasured | Official workload runs and competition comparison |

Recommended decisions for team review:

1. Prioritize loss-plus-metrics and pointwise/casting fusion; retain NF4/GEMM as a separate experiment.
2. Resolve the starter exclusion and obtain a T4 baseline before treating any named kernel as an eligible contribution.
3. Begin with implementations that preserve task semantics and required gradients/metrics. Treat approximation as a separate research question, not an assumed allowance.
4. Accept improvements only after eligibility, quality, whole-command time and sampled peak-memory checks.
5. Preserve time for the final reproducible notebook, source attribution, README, presentation and demo.

**Evidence boundary:** public implementations establish useful algorithmic precedents. This work has not established a competition speedup, memory reduction, eligible implementation or winning rank.

## 9. Source ledger

Web sources were checked during the research on 30 October 2026. Preserve the hardware, workload and baseline attached to each reported result.

| ID | Primary source | What it supports |
| --- | --- | --- |
| [S1] | Supplied Soup Kernel Challenge technical specification | Target, organizer profile, restrictions, correctness, measurement, scoring and deliverables. Russian pages 1-7; English equivalent pages 16-22. |
| [S2] | Hugging Face: Custom Kernels for All from Codex and Claude, 13 February 2026 | Qwen3-8B isolated RMSNorm benchmark on H100 80 GB, BF16. |
| [S3] | Unsloth: Qwen2.5-1.5B-Instruct-bnb-4bit model card | Model-family/size match and Qwen2.5 T4 notebook links. |
| [S4] | LinkedIn: Liger Kernel README | Explicit Qwen2.5 operation support and chunked fused linear cross-entropy. |
| [S5] | Hugging Face Kernel Hub: kernels-community/rmsnorm | Forward/backward entry points; no benchmark on the retrieved card. |
| [S6] | Apple AIML Research: Cut Cross-Entropy implementation | Memory-saving method, Ampere-or-newer requirement and gradient-filtering modes. |
| [S7] | Qwen: Qwen2.5-1.5B-Instruct official configuration | Architecture dimensions, vocabulary, epsilon and tied embeddings. |
| [S8] | Qwen3.7 blog | Author-reported SGLang attention optimization by Qwen3.7 on T-Head hardware. |
| [S9] | Hugging Face: Develop kernels with agents | Precise task definition, integration and correctness/performance validation. |

[S1]: https://chatgpt.com/library?file_id=libfile_0f170950ca54819186db99d6c3bad927&account_id=personal "Supplied Soup Kernel Challenge brief"
[S2]: https://huggingface.co/blog/custom-cuda-kernels-agent-skills "Custom Kernels for All from Codex and Claude"
[S3]: https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit "Unsloth Qwen2.5-1.5B-Instruct-bnb-4bit"
[S4]: https://github.com/linkedin/Liger-Kernel/blob/main/README.md "Liger Kernel README"
[S5]: https://huggingface.co/kernels-community/rmsnorm "HF RMSNorm kernel"
[S6]: https://github.com/apple-aiml-research/ml-cross-entropy "Cut Cross-Entropy implementation"
[S7]: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json "Qwen2.5-1.5B-Instruct config"
[S8]: https://qwen.ai/blog?id=qwen3.7&locale=en "Qwen3.7 blog"
[S9]: https://huggingface.co/docs/kernels/builder/agents-guide "Develop kernels with agents"
