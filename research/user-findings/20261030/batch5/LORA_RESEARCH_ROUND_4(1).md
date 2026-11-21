# LoRA kernel research — round 4

**Date:** 2026-10-30. **Workload:** the fixed Decentrathon 6.0 Soup Kernel Challenge.  
**Outcome:** numerical evidence now supports a narrower first implementation: preserve the original head GEMMs, privately cache their rounded fp16 logits, and replace loss/metric storage first. For LoRA, begin with a separate input-gradient/dropout fusion for each adapter branch. A new attention opportunity is to prepare the shared additive mask once per forward.

This round produced original CPU research programs, an independently repeated numerical audit, source-based integration contracts, and a revised experiment order. It produced **no submission kernel, T4 benchmark, complete Soup training run, or official challenge pass**. The authorized submission checkout and target T4 environment remain unidentified.

## 1. What changed this round

| Finding | Evidence | Consequence |
|---|---|---|
| Private fp16 logits can reproduce the controlled reference loss, statistics, and gradients | CPU PyTorch 2.11.0; 174 main comparisons plus 24 empty-supervision checks | Keep this as the leading head-memory experiment. |
| Algebraically equivalent loss gradients are not numerically interchangeable | Saturation and large-offset cases fail several reconstruction formulas | Use the native backward as the initial numerical control; choose a Triton formula only after target-GPU checks. |
| LoRA backward has additional rounding boundaries | Seven gradients, 9,840 elements, match the preserved CPU graph exactly; altered operations expose differences | Preserve casts before scaling, parameter-gradient output rounding, masks, and per-branch dropout products. |
| Rewind already supports genuine no-logits summaries | Pinned Soup reads `_soup_chunked_step.token_loss` | A coordinated model/trainer path can preserve Rewind without rewriting its recorder. |
| The same attention mask is repeatedly converted | Qwen reuses one mask; native SDPA converts it inside each layer | Convert and align it once, retaining the native attention backend. |
| A seeded `randperm` does not establish the training order | Rewind's wrapper prevents Accelerate's RandomSampler type check from matching | Preserve CPU RNG state and capture real sampler indices before reporting exact batch costs. |
| Compilation must fit a 64-step budget | Fresh-cache scoring plus dozens of illustrative sequence lengths | Limit specialization and runtime autotuning. |

Numerical records are in `CPU_NUMERICS_ROUND_4.json`. The complete runnable programs, inputs, individual results, and detailed design notes are in `KERNEL_ROUND_4_REPRO.zip`. Read `KERNEL_HANDOFF.md` first in the next session.

## 2. Private logits cache: what the experiment establishes

The ordinary-autograd reference and custom first-order operation use the **same token-chunked fp16 head projection**. Hidden states and frozen head storage are fp32; each is converted for the projection; each output is rounded to fp16. Cross entropy and diagnostics operate in fp32 over those rounded values. The custom operation saves supervised fp16 logits, one fp16 frozen-head copy, positions, labels, the denominator, and small statistics. It saves no hidden-state values. This is not claimed as an additional hidden-state saving over a normal frozen linear, which also does not need its input values for its input gradient.

The main suite covers ten tensor/masking scenarios and 29 scenario/scale combinations. Six backward formulas give **174 comparisons**. Cases include causal label shifting, prompt masking, unequal padding, an individually unsupervised row, argmax ties, awkward chunk sizes, an external loss denominator, saturated logits, a common logit offset of 1,000, and the actual vocabulary size of 151,936. The true-vocabulary case uses hidden dimension 16, not Qwen's full 1,536, with five supervised, eight real, and ten padded positions. Its twelve comparisons pass the research gradient gate.

Scales include 0, 0.1, 1, 1/1024, 1,000, 65,535, 65,536, and 131,072 where applicable. The actual upstream scalar is applied **before** the logits gradient is rounded to fp16. This matches the need to preserve scaled backward described in PyTorch's AMP guidance. [1–4]

### Results

Each formula sees 27 finite-gradient comparisons and two deliberately overflowing comparisons. Overflow cases are reported separately; matching overflow is not counted as a finite correctness pass.

| Formula | Finite pass / fail | Matching nonfinite cases | Largest absolute hidden-gradient error |
|---|---:|---:|---:|
| Native ATen log-softmax backward diagnostic | 27 / 0 | 2 | 0 |
| Softmax minus onehot | 27 / 0 | 2 | 0 |
| Manual log-softmax VJP | 21 / 6 | 2 | 0.005859375 |
| Saved uncentered log-sum-exp | 19 / 8 | 2 | 261.8828125 |
| Saved centered statistics | 21 / 6 | 2 | 0.046875 |
| Deliberately wrong scale-after-half control | 20 / 7 | 2 | 10.7421875 |

All 174 comparisons retain exactly the reference's supervised fp16 logit values and integer metrics. Entropy and per-row loss differences are zero. Maximum scalar loss difference is **4.76837158203125e-7**, reflecting reduction organization. NaN, positive-infinity, and negative-infinity positions match; the checker explicitly fails a nonfinite-pattern mismatch and also checks remaining finite elements.

The exploratory gradient gate accepts absolute error at most 2e-6, **or** both maximum error divided by the reference peak and relative L2 error at most 0.002. Loss and floating metrics use `atol=rtol=1e-6`. These are stated research criteria, not the challenge's acceptance thresholds. Raw errors remain in the JSON, including small errors that pass through the absolute floor.

### Why this changes the implementation

Let `a = grad_loss / D`, where D is the exact loss denominator. For a target token, `-a - p*(-a)` and `(p-1)*a` are equal in real arithmetic. Different intermediate rounding, exponential evaluation, and operator fusion can produce different floating-point results.

At a saturated logit margin of 18 and upstream scale 65,536, the manual VJP reaches **12.5% peak-normalized hidden-gradient error**. The scale-after-half negative control reaches **50%**. Some saved-statistics variants lose a gradient component large enough to reach 100% peak-normalized error. With a common logit offset of 1,000, uncentered `exp(z-LSE)` reaches **18.885 times the reference gradient peak**. Saving centered statistics removes that offset problem in the tested case but does not remove all saturation failures.

The native backward matches because it calls the same backend operation as the reference. The CPU softmax-minus-onehot result is useful evidence, but **does not select a CUDA-safe formula**. PyTorch's CUDA backward source uses `gradOutput - exp(output) * sum`; target-device reductions, exponentials, and generated instructions still need testing. The experiment does not identify FMA as the sole cause. [5]

### Entire-batch zero supervision

An additional 24 comparisons cover all targets ignored, with no supplied denominator or a supplied positive denominator of 17, at two upstream scales for all six formulas.

| Condition | Loss | Hidden gradient | Accuracy | Per-row diagnostic |
|---|---|---|---|---|
| No supplied denominator; zero supervised count | NaN | Zero | Zero | Zero |
| Supplied denominator 17 | Zero | Zero | Zero | Zero |

All checks match. The loss denominator is never clamped. The per-row diagnostic's zero-count handling is a separate contract. A training integration should preserve these results or route the case to the original loss, rather than silently replace a NaN training loss. [1–2]

## 3. H0 now has a concrete integration route

The first GPU experiment should be narrower than the CPU custom-head prototype: **leave Qwen's native head forward and native head backward intact**. Put a custom loss/statistics autograd operation on the actual fp16 logits. Return its fp16 gradient to the original head backward, keeping the original GEMM shape and autocast weight-copy behavior. This isolates loss/metric storage changes from projection and row-compaction changes.

The reference projection should run once. The loss operation can recompute pointwise probability information from its private fp16 cache during backward; it does not need to recompute the head GEMM. Full-row logits and the original full-row gradient schedule are the initial control. Supervised-only storage and compacted GEMMs belong to later experiments.

The pinned source exposes a useful existing interface: when `outputs.logits` is absent, Rewind already reads `trainer._soup_chunked_step.token_loss`. Its existing reducer computes the per-row mean and token count, then uses the normal sampler FIFO and step bookkeeping. No recorder rewrite is needed. [6–7]

A minimal implementation contract is:

1. Install a guarded SFTTrainer subclass before Soup constructs its existing Rewind subclass. Run original model setup first.
2. On the expected real Qwen/PEFT model, install an instance-specific `loss_function` callback and forward wrapper. Keep parameters, module names, and checkpoint structure intact.
3. Only a supported labeled training call from the custom trainer establishes a per-call context authorizing private-logits handling. Ordinary direct calls, evaluation, prediction, and unsupported modes use the original path.
4. Delegate outer loss handling to `transformers.Trainer.compute_loss`, forwarding the exact `num_items_in_batch` and preserving its device scaling.
5. The loss callback consumes the actual native-head logits, computes genuine loss and detached summaries, and privately retains its cache through autograd.
6. Before Accelerate's output conversion, reconstruct a genuine `CausalLMOutputWithPast` with the real loss, preserved auxiliary fields, and `logits=None`. Reconstructing matters: setting an attribute to `None` can leave the previous tensor in a ModelOutput mapping.
7. Update TRL metrics from those exact summaries and set `_soup_chunked_step`. The enclosing Rewind override consumes the per-token losses itself.
8. Clear per-call temporary state on completion or exception, invalidate stale summaries before every eligible call, and retain the current summary until Rewind has consumed it.

`return_outputs=True` alone is not a request for a logits tensor: Rewind uses it to retrieve loss/output metadata. The explicit per-call context is the appropriate distinction. The prototype remains first-order only; any unsupported higher-order or trainable-head use requires fallback. [3,6–10]

The summary contract remains exact:

| Field | Required meaning |
|---|---|
| Training loss | Sum of shifted supervised CE divided by the exact supplied denominator, or original mean behavior |
| `token_loss[b,t]` | CE predicting `labels[b,t+1]`; zero for ignored targets and the final position |
| Entropy | All attention-mask positions, including prompts and the final real token |
| Accuracy | Correct shifted supervised tokens divided by supervised count; zero when the count is zero |
| Token count | The original cumulative attention-mask count |
| Rewind | Original row identity, per-row CE mean, supervised count, FIFO, and step/micro accounting |

Baseline returned logits are upcast by Accelerate before TRL/Rewind statistics. The replacements therefore use fp32 reductions over the actual fp16 values. Returning the large half cache through model outputs would defeat this purpose. The existing summary interface is evidence for an integration route; enabling built-in fast loss modes or copying their kernels is not part of the proposal. [2–3,6–10]

At maximum shape, the fp16 logits tensor alone is 593.5 MiB and an fp32 copy is 1,187 MiB. These tensor sizes explain the opportunity; they do not establish a peak-memory saving, because lifetimes and scratch buffers overlap differently. The initial native-operator H0 is a correctness/control experiment and does not itself satisfy the original-Triton participation requirement.

## 4. LoRA backward: preserve the complete dtype graph

For one branch, let X and master parameters A/B be fp32, adapter GEMMs use fp16 operands and outputs, scale be 2, and dropout probability be 0.05. Define `R16` as rounding to fp16, `C32` as the exact upcast, and `MM16` as the selected GEMM including its fp16 output boundary. Its internal reduction tree remains unspecified.

Forward uses the native independent dropout mask, then:

    Xd16 = R16(dropout(X32))
    Z16  = MM16(Xd16, A16.T)
    E16  = MM16(Z16, B16.T)
    Y32  = Ybase32 + C32(R16(2 * E16))

With incoming fp32 gradient G, the preserved backward boundaries are:

| Value | Operation |
|---|---|
| Up-projection input gradient | `T16 = R16(2 * R16(G32))` |
| Down-projection output gradient | `U16 = MM16(T16, B16)` |
| B gradient returned to master parameter | `dB32 = C32(MM16(T16.T, Z16))` |
| Pre-dropout input gradient | `H16 = MM16(U16, A16)` |
| A gradient returned to master parameter | `dA32 = C32(MM16(U16.T, Xd16))` |
| Branch input gradient | Native fp32 mask multiplication and dropout scaling applied to `C32(H16)` |

The input cast comes **before** LoRA's factor of two. The weight-gradient dot products have a half output boundary before their fp32 master-gradient accumulation. The base branch still contributes its own input gradient. These properties follow the pinned PEFT forward and PyTorch autograd/cast/dropout implementations. Actual target-runtime dtypes remain to be observed. [11–14]

### CPU evidence

The preserved expressions match ordinary autograd exactly across **seven gradient tensors and 9,840 elements**. A separate CPU fp16-autocast graph also matches across four gradients and 1,848 elements. All ten explicit audit gates pass.

| Deliberate change | Differing finite elements | Maximum absolute difference |
|---|---:|---:|
| Remove half rounding before dropout backward | 3,114 / 3,120 | 0.0001459122 |
| Return fp32 dAq dot product directly | 768 / 768 | 0.0002405643 |
| Return fp32 dBq dot product directly | 768 / 768 | 0.0002436042 |
| Reuse q's dropout mask for v | 292 / 3,120 | 0.1881167889 |
| Factor dropout scale outside the q/v sum | 1,017 / 3,120 | 5.9604645e-8 |

These differences establish that the operations changed; a differing bit alone does not prove failure of an official challenge tolerance. The mask substitution is a more substantial change than the tiny rounding drift in the last row.

Specific witnesses clarify the risks:

- Casting `2^-25` to half and then multiplying by two gives zero; scaling first and then casting gives `2^-24`.
- Separate half cast edges with gradients 2,048 and 1 produce fp32 2,049; a shared half edge produces 2,048. Two gradients of 40,000 produce 80,000 through separate edges but infinity through a shared half edge.
- Dropout backward multiplies by the mask. A false mask times infinity produces NaN; selecting a literal zero suppresses it and can change overflow detection.
- With branch values 65,504 and -65,504, separately rounded fp32 dropout products sum to zero. A numerical simulation of one contracted FMA produces 0.003288269. This is an arithmetic counterexample, not executed CUDA code.
- Half operands can yield fp32 products `[2^24, 1, -2^24]`; different fp32 reduction trees can return final half values zero or one.

For the fixed dropout probability, the native fp32 multiplier is **1.0526316165924072**. Preserve branch products before addition if pairing branches. An fp32-to-fp32 cast does not prevent contraction; the pinned Triton compiler enables scalar floating-point fusion by default. Local precision controls and generated-code inspection are needed for the paired experiment. Do not change global training precision flags to manufacture agreement. [12,15–16]

### Revised L1 and L2

**L1 first:** one independently fused `U @ A -> half round -> fp32 cast -> native-mask dropout backward` kernel per branch. Leave q/v/shared-input accumulation to existing autograd initially. This targets intermediate traffic while limiting changes to summation order. A paired q/v kernel is a later comparison.

**L2 has a dependency boundary:** dB is ready from T and Z, but dA also needs `U=T@B`. Compare grouping Uq/Uv followed by all four parameter gradients against grouping the four independent jobs Uq/dBq/Uv/dBv followed by dAq/dAv. The barrier does not by itself prove a slowdown on one stream. A normal grouped kernel cannot let dA thread blocks consume U while unrelated blocks are still writing it without valid synchronization or explicitly costed recomputation.

T4 resource arithmetic also favors restrained initial tiles. With four warps and two simultaneously live fp32 accumulators, paired 32×64, 32×128, and 64×128 tiles require respectively 32, 64, and 128 accumulator registers per thread before other values. A paired 64×256 tile would require 256 just for accumulators; Turing's per-thread limit is 255. Compiler spilling or alternative scheduling changes that calculation, but it is a poor initial all-register design. These are analytic budgets, not measured occupancy. [17]

## 5. New attention lead: share the converted mask

Qwen creates one causal/padding mask and passes it to all 28 layers. For the bool-mask path, native SDPA converts it to additive query-dtype values inside every call. Efficient attention then pads/copies a mask whose preceding strides are not multiples of eight. [18–21]

An original small kernel can consume the existing bool mask once and emit fp16 values 0 or negative infinity. Allocate physical storage `[B,1,T,Tpad]`, with `Tpad=ceil(T/8)*8`, and expose the logical last dimension of length T. The final stride stays one and preceding strides are aligned. This changes storage pitch, not the sequence, positions, mask values, or batch data.

Preserving the original mask's presence also preserves Transformers' repeat-KV/GQA and causal-flag branches. The source accepts an additive mask with query dtype, and efficient CUDA attention explicitly expects that dtype and a contiguous final dimension. Its eligibility does not branch on bool versus additive mask. On the audited T4 path, Flash/cuDNN hardware guards still reject SM75. Thus the proposal preserves source-level dispatch conditions under the same q/k/v, build, autocast dtype, and flags; actual backend execution remains to be traced. [19–22]

At B=4,T=512, the bool mask is 1 MiB and a shared fp16 mask is 2 MiB. If efficient backward retains one separately converted bias per layer, the original path can retain 56 MiB of such biases versus a shared 2 MiB: a **potential 54 MiB reduction at maximum length**. This is a source-based lifetime inference, not an allocator measurement. The small once-per-forward kernel is unlikely to be a sufficient participation strategy on its own; its measured share is unknown.

The repeat-KV/cast idea remains another candidate. On the existing masked path, directly write expanded fp16 K/V from the unexpanded inputs, avoiding expanded fp32 intermediates. Backward must restore each expanded half gradient to fp32 **before** summing its six replicas. Optional RoPE fusion must retain the original fp32 rotation and reused cos/sin values before the half store. Do not initially change the no-mask path: native GQA can cause a different backend and math-precision path on T4. [19–22]

## 6. Sampler and compilation: two execution constraints

### The actual batch order still has to be captured

Transformers enables its seedable-sampler option by default, and Accelerate converts a sampler only when it is an instance of `RandomSampler`. However, Soup's Rewind subclass wraps it first in `IndexRecordingSampler`, whose base is `Sampler`. Forwarding attributes does not make that wrapper a `RandomSampler`. Accelerate's conversion check therefore does not match this wrapper. [6,23–25]

The inner PyTorch RandomSampler, with no supplied generator, draws its own seed from the global CPU RNG at iteration time. An original minimal diagnostic reproduces the relevant type behavior: the wrapper fails `isinstance(RandomSampler)`, still forwards `generator`, and one added CPU random draw changes the resulting row permutation. [26]

This does **not** recover the full Trainer RNG history or prove an error in the baseline. It means neither Python shuffle nor `torch.randperm(256, seed=1)` should be described as the observed training order. Preserve native sampler/recorder behavior. Avoid consuming the global training RNG for kernel setup or synthetic tuning inputs; use independent generators or properly restore every affected RNG state. Capture real row IDs and sequence lengths in the untouched trace.

### Do not spend the short run compiling shape variants

The public source-order grouping has 48 different sequence lengths across 64 batches. The four prior illustrative Python shuffles have 42–47. A direct seed-one torch permutation has 49; the minimal wrapped-loader diagnostics have 44–47. These are measured counts on the existing public-row audit, **not observed harness batch shapes**.

Making every T/M a constexpr or an autotuning key can multiply compilation or benchmarking work. Triton's autotuner reevaluates configurations when its key changes; constexpr arguments specialize compiled code. Runtime dimensions with masks, a small number of fixed structural configurations, and selective `do_not_specialize` controls are more appropriate starting points for this short fresh-cache run. Preserve valid alignment assumptions; do not disable specialization indiscriminately. [27–29]

If added compilation/tuning costs C seconds over 64 steps, the average step saving must exceed `C/64` just to break even, ignoring other changes:

| Added setup/compilation | Required average saving per step |
|---:|---:|
| 1 second | 15.625 ms |
| 5 seconds | 78.125 ms |
| 10 seconds | 156.25 ms |
| 30 seconds | 468.75 ms |

These are arithmetic examples, not measured compile times. The official score includes command startup, apply, loading, compilation, training, and saving, with a fresh Triton cache. Offline selection of source configurations is different from shipping a forbidden precompiled cache. [30]

## 7. Revised experiment order and stopping rules

| Stage | Change | Evidence required before keeping it |
|---|---|---|
| T0 | Untouched T4 run | Runtime/package pins, real tensor dtypes, SDPA backend, sampler IDs, shapes, peak allocations, whole-command time |
| M0 | Genuine summary instrumentation with original execution | Exact masks/counts/row order; normal evaluation and direct-output behavior |
| H0 | Private full fp16 logits; original head GEMMs; native loss-backward control | Scaled gradients, nonfinite behavior, metrics, Rewind, memory lifetimes, official quality checks |
| L1 | Independent branch expansion/cast/dropout fusion | Nonzero-B gradients, multiple AMP optimizer steps, original masks, target-GPU numerical agreement |
| A0 | One shared aligned additive mask | Equal logical mask/output/gradients, same backend, observed allocation and time changes |
| H1 / L2 | Supervised cache/row compaction or grouped weight gradients | Improvement over accepted controls; no new numerical or compile-cost regression |

The new CPU programs are useful numerical references. They are not valid replacements for target-GPU tests, the full 64-step training comparison, evaluation, and official rules checks. The original-kernel time share still must meet the challenge's at-least-5% requirement in every measured run; no proposed combination has demonstrated that share yet. [30]

Once the authorized checkout and T4 runtime are available, the next work should follow this experiment order rather than restart the paper survey. Keep the original masks, sampler, training flags, loss semantics, and harness. Record actual experiments and rejected variants in a machine-readable log alongside the compact handoff.

## Primary sources

1. [Transformers 5.16.1 causal loss and denominator](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py)
2. [TRL 0.29.1 SFT loss and metrics](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py)
3. [Accelerate 1.14.0 output conversion](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py)
4. [PyTorch 2.11 AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html)
5. [PyTorch 2.11 CUDA softmax forward/backward](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/SoftMax.cu)
6. [Pinned Soup Rewind implementation](https://example.invalid/redacted-github)
7. [Pinned Soup genuine chunked-summary interface](https://example.invalid/redacted-github)
8. [Transformers 5.16.1 Trainer loss handling](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py)
9. [Transformers 5.16.1 Qwen model](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py)
10. [Transformers loss-function property](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py)
11. [PEFT 0.20.0 quantized LoRA forward](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py)
12. [PyTorch 2.11 CUDA dropout](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu)
13. [PyTorch autograd derivatives](https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml)
14. [PyTorch manual cast/multiply/GEMM derivatives](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/FunctionsManual.cpp)
15. [Triton 3.6 NVIDIA compiler options](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/backend/compiler.py)
16. [PyTorch CUDA GEMM dispatch](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/cuda/CUDABlas.cpp)
17. [NVIDIA Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
18. [Transformers 5.16.1 mask construction](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/masking_utils.py)
19. [Transformers SDPA repeat/GQA branches](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/integrations/sdpa_attention.py)
20. [PyTorch SDPA mask conversion and alignment](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/attention.cpp)
21. [PyTorch CUDA attention bias requirements](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/attention.cu)
22. [PyTorch SDPA backend eligibility](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
23. [Transformers sampler configuration](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer_pt_utils.py)
24. [Accelerate 1.14.0 sampler preparation](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/data_loader.py)
25. [Pinned task and default seed](https://example.invalid/redacted-github)
26. [PyTorch 2.11 RandomSampler](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/data/sampler.py)
27. [Triton 3.6 JIT specialization](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/jit.py)
28. [Triton JIT documentation](https://triton-lang.org/main/python-api/generated/triton.jit.html)
29. [Triton autotune documentation](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
30. [Pinned challenge rules](https://example.invalid/redacted-github) and [whole-command harness](https://example.invalid/redacted-github)

Detailed subsystem notes in the reproduction bundle add source links for individual gradient edges, autocast cache eligibility, and attention backward. Empirical statements in this report refer to the accompanying original CPU results, independently rerun before delivery.
