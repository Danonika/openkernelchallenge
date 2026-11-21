# Qwen-Inspired Kernel Optimization for the Soup Kernel Challenge

Research date: 30 October 2026  
Target: Soup 0.77.0, Qwen2.5-1.5B LoRA training, one Kaggle T4  
Inspected starter commit: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`

## Session handoff

**Goal:** Improve full-command training time and peak GPU memory for the fixed Soup Kernel Challenge on one Kaggle T4.

**Status:** Rules and pinned Soup 0.77.0 source inspected; no candidate kernels implemented or benchmarked. Qwen's reported 10x attention result does not predict our speedup.

**Next:** Implement original chunked LM-head loss/metrics and RMSNorm/LoRA fusion; benchmark with the official harness.

**Preserve:** Shifted supervised labels, normalization, FP16 AMP scaling, prompt-inclusive entropy, accuracy, rewind values, and independent q/v dropout at 0.05. Keep all fixed training settings and batches unchanged. Original qualifying Triton kernels must meet at least 5% runtime share; rewritten starter SwiGLU does not qualify. All official correctness and regression gates remain mandatory.

## Executive overview


> **Recommended first build**
> Develop a streamed output-head, loss and metrics path alongside original normalization kernels. Add LoRA fusion, then choose the next experiment from the measured end-to-end bottleneck.

### Decision and evidence

Qwen's published optimization experiment offers useful ideas about parallel work, data reuse, stable reductions and sustained measurement. This report adapts those principles to the fixed Soup training workload, the competition's actual scoring, and the requirements of FP16 training on a Kaggle T4. [1-3]

**Status:** the sources, starter code, harness and pinned Soup release were inspected. The proposed kernels have not been implemented or benchmarked in this research. All candidate benefits remain hypotheses; the report does not claim a speedup or a guaranteed ranking.

| Fixed target | Value |
| --- | --- |
| Model and training | Qwen2.5-1.5B-Instruct; Soup 0.77.0; NF4 4-bit weights; LoRA rank 16, alpha 32, q/v adapters |
| Workload and hardware | 256 instruction rows; 64 steps; batch 4; sequence length up to 512; one Kaggle T4; FP16 |
| Scoring emphasis | Speed 35 points; peak memory 30; correctness/admission 20; README and reproducibility 15 |
| Submission deadline | 31 October 2026, 23:59 Astana / Almaty time (UTC+5) |

The target is the attached Soup challenge. The earlier H200 / Open Kernel Challenge project uses different hardware and workloads; its optimization priorities should be evaluated separately. [2-4]

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
| **1. Streamed head, loss and metrics** | Lower full-vocabulary residency and repeated reduction traffic. | Correct labels, denominators, entropy, accuracy, rewind and AMP backward. |
| **2. Original RMSNorm / residual fusion** | Reduce reads, writes, casts and launches. | Preserve FP32 reduction and input/output dtype behavior; validate gradients. |
| **3. LoRA epilogue / backward fusion** | Reduce rank-16 temporaries and scale/add/cast operations. | Preserve dropout, q/v mask independence, A/B gradients and adapter layout. |
| **4. Selective recomputation** | Exchange inexpensive repeated compute for lower activation memory. | Keep settings unchanged; replay stochastic operations; remain inside time gates. |
| **5. NF4 decode + matmul** | Consume decoded tiles without a large expanded-weight intermediate. | Match double quantization, layout and both forward and input-gradient paths. |
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

### Proposed forward data flow

- **Compute a bounded token chunk.** Initially retain the existing matrix-multiplication path for the head.
- **Run original Triton reductions.** Accumulate loss, entropy, argmax/correctness and per-row statistics over vocabulary tiles.
- **Keep compact state.** Retain hidden activations and normalizers needed for backward; release large chunk intermediates.
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

> **T4 design choice:** use manageable vocabulary tiles. A simple power-of-two full-row program would pad 151,936 entries to 262,144. Triton's introductory fused-softmax design assumes rows fit on-chip; that assumption must not be carried over blindly. [12]

## 4. Backward and metric fidelity

Preserve the training problem, not only its scalar loss

| Output / dependency | Required behavior |
| --- | --- |
| Training loss | Shift labels so position t predicts the next token; exclude ignored labels and preserve the baseline denominator. |
| Entropy | Cover non-padding attention-mask positions, including prompt tokens. |
| Token accuracy | Compare predictions with the shifted supervised labels using the same counting semantics. |
| Rewind row losses | Keep original row identities, supervised-token counts and per-row aggregation; preserve zero-supervision row handling. |
| Gradient flow | The head is frozen, but hidden-state gradients must reach earlier trainable adapters. |

These contracts were checked in the pinned Soup 0.77.0 source, including `stream_loss_chunk.py` and `rewind_hf.py`. The implementation must also preserve the actual runtime dtype and rounding behavior. [10]

### A safer initial backward

Recompute each required head chunk during backward and construct the supervised logit gradient using the upstream gradient scale. Apply that scale before the relevant low-precision rounding:

```text
dZ = (P - one_hot(target)) * grad_output / denominator
dH = dZ @ W
# Ignored-label positions contribute zero direct loss gradient.
```

No head-weight gradient is needed for this fixed frozen head. However, skipping its input gradient would stop the learning signal from reaching the decoder. Similarly, frozen NF4 base projections still need their input-gradient path.

### FP16 risk to test explicitly

Precomputing an unscaled hidden gradient in FP16 and applying the upstream scale later can round or underflow small values too early. The safer initial design above avoids assuming that those two orders are numerically equivalent. This is an identified design risk, not a demonstrated failure of a candidate.

### Recomputation has a real cost

An extra head pass may reduce memory while increasing time. Evaluate a small set of chunk sizes and backward policies on the full command. Preserve candidates with useful time/memory tradeoffs, but reject any combined candidate that fails an official admission gate.

> **Easy-to-miss constraint:** prompt tokens can be excluded from supervised CE, but their vocabulary distributions still contribute to entropy. Skipping all prompt-token head evaluation would change a required output. [10]

## 5. Fusion and later experiments

Start with contained changes; isolate the higher-risk branches

### Original RMSNorm and residual fusion

Implement both forward and backward, preserving FP32 norm reduction and baseline input/output types. Fuse adjacent residual or cast operations only where dependencies and rounding permit. Treat broader precision changes as a separate experiment so numerical deviations remain attributable.

### LoRA: keep the stochastic behavior

The task uses rank 16, alpha 32 and dropout 0.05 on q/v adapters. Those projections have separate dropout operations. Begin by retaining the existing dropout calls and masks, then fuse downstream low-rank epilogues, scaling, base-result addition and casts. A shared dropped-out input must not silently replace independent q/v masks. [3,4]

Backward must preserve both adapter-matrix gradients and all input contributions. Full trajectory comparison matters: matching a dropout probability distribution alone does not establish agreement with the tightly checked seeded run.

### Selective activation recomputation

Identify large saved intermediates that are inexpensive to recreate. Recompute them inside original autograd code, retaining masks or the correct RNG state when replaying stochastic operations. Custom recomputation and offloading are permitted, while changing the built-in configuration switches is prohibited. [2,3]

### NF4 decode fused with matrix multiplication

Decode weight tiles and consume them immediately, avoiding an expanded tensor where possible. Preserve the NF4 codebook, nested scales, layout, transpose and bias behavior. Validate forward and input-gradient paths separately. Reduced memory movement may be outweighed by unpacking cost, register pressure or less efficient matrix multiplication; the T4 result decides.

### Remove padding within each existing batch

Custom unpadding is allowed, but rows cannot be mixed across batches. Preserve independent causal sequences, position IDs, shifted labels, row mapping, token accounting and dropout behavior. Start with isolated operations before redesigning the attention layout. [2,3]

> **Originality trap:** the starter already supplies fused SwiGLU forward and backward. Its kernels do not count as ours, even if renamed or rewritten. Retain it only as a supporting optimization; establish eligibility through substantive original work elsewhere. [3,6]

### Priority rule

Develop the streamed head/metrics path and original normalization kernels in parallel. Add LoRA fusion next. Select recomputation, NF4 or padding work based on measured potential and the time left for integration and validation.

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

## 8. Sources and provenance

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

**[10]** Soup CLI 0.77.0 release. Pinned wheel inspected as source; stream_loss_chunk.py and rewind_hf.py define metric contracts.
[pypi.org/project/soup-cli/0.77.0/](https://pypi.org/project/soup-cli/0.77.0/)

**[11]** Apple. Cut Your Losses in Large-Vocabulary Language Models. Algorithmic background for avoiding full logits residency.
[github.com/apple-aiml-research/ml-cross-entropy](https://github.com/apple-aiml-research/ml-cross-entropy)

**[12]** Triton documentation. Fused Softmax. Data-movement motivation and the on-chip row-size condition.
[triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

### Provenance and interpretation

Starter sources were inspected at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. The report distinguishes publisher/organizer measurements from our calculations and untested design proposals. No winning kernel release, competition ranking or candidate T4 performance is asserted.

The live harness is more precise than the brief on repeated-run aggregation: it takes maximum memory peak rather than an average. The implementation details and numeric gates in this report follow the inspected harness. [5-8]
