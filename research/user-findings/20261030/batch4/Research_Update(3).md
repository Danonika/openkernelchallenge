# Soup Kernel Challenge: second research and implementation pass

**Date:** 30 October 2026, Astana/Almaty. **Target:** the platform's fixed Soup 0.77.0 / Qwen2.5-1.5B-Instruct NF4 training task on one Kaggle Tesla T4.

## What changed

This pass produced an original fused NF4 decoder, two lossless storage experiments, a new shared-memory T4 head experiment, and a reproducible GPU screening tool. The best new memory lead is **approximately 2.783 GiB less retained storage at the maximum batch shape** from the two optional storage transformations. This number is allocation arithmetic, not a measured device-memory peak.

The expanded source passed **79 official regressions**, **99 additional CPU/interpreter checks**, and **21 offline SM75 compilation cases**. **Twenty-two physical-CUDA tests remain skipped.** No speedup, peak-memory ratio, quality result, admission, or leaderboard rank has been measured.

The strategic priority is now: establish the default NF4/head candidate on T4, measure compact saved MLP activations, then measure shared frozen weight storage. The new fused head deserves a controlled experiment, but replacing cuBLAS prematurely is the riskier direction.

## 1. The largest new opportunity: save values in the precision they already have

### MLP gate and up activations

The important detail is the boundary between bitsandbytes and the Qwen MLP. In the pinned stack, ordinary frozen `Linear4bit` modules compute in configured FP16 and cast their output back to the input dtype. Qwen's normalized inputs are FP32. Therefore the gate and up arrays arrive at SwiGLU as FP32 tensors, even though their values came from FP16 computation. The pinned implementation is [bitsandbytes 0.50.2, `nn/modules.py`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py).

These values can be stored in FP16 without losing information. This is different from assuming that arbitrary FP32 activations tolerate rounding. The new wrapper only enables this transformation when the module types, frozen NF4 weights, precision configuration, forward implementation, hooks and dimensions establish the required dataflow.

One version-specific detail mattered: bitsandbytes 0.50.2 sets `compute_type_is_set = compute_dtype is not None` during construction. Soup explicitly requests FP16 on T4, and Transformers passes that setting into the module. Consequently the first FP32 input does not reset compute precision. A guard based on an older bitsandbytes implementation could have reached the wrong conclusion.

At the maximum task shape, the retained pair of activations across all layers is:

```text
2 tensors × 2048 tokens × 8960 intermediate width × 4 bytes × 28 layers
= 4,110,417,920 bytes
```

Saving the same values in FP16 requires **2,055,208,960 bytes**, a reduction of **1.914 GiB** in those retained arrays. Dimensions come from the [Qwen architecture config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) and the fixed task.

Forward still performs the starter's FP32 SwiGLU computation. Backward still consumes FP32 upstream gradients and produces FP32 gate/up gradients. The existing kernel loads the saved half values into FP32 registers; it does not recreate full FP32 activation arrays. This distinction is necessary: casting a scaled upstream gradient to half before differentiation would change the arithmetic.

The optimization adds two forward cast passes per layer. Smaller backward reads offset some traffic, but the net effect on end-to-end runtime is unknown. Actual sequence lengths and tensor lifetimes also determine the reduction in peak GPU memory. The option is wired as `COMPACT_MLP_ACTIVATIONS=False` until measured.

**Originality:** this uses the unchanged, credited starter SwiGLU kernels. It introduces no new eligible kernel, and its kernel time must remain excluded from the 5% original-work gate.

### Frozen tied embedding/output weight

PEFT preparation freezes base parameters and upcasts non-quantized FP16/BF16 parameters to FP32. The relevant source is [PEFT 0.20.0 preparation](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py). Our current streamed head also retains a half-precision cast for projection.

For the tied `[151936,1536]` parameter, the storage arithmetic is:

| Representation | Bytes | MiB |
| --- | ---: | ---: |
| FP32 tied parameter | 933,494,784 | 890.25 |
| Separate FP16 head cache | 466,747,392 | 445.125 |
| Current combined storage | 1,400,242,176 | 1,335.375 |
| Optional shared FP16 allocation | 466,747,392 | 445.125 |
| Potential reduction | **933,494,784** | **890.25** |

The optional hook retains the actual FP16 allocation before unmodified PEFT preparation. After preparation, it verifies that the frozen FP32 parameter contains exactly those values, using bounded temporary chunks. Only then does it restore the smaller allocation. The same Parameter remains tied, and the head cache aliases it.

An original Triton embedding gather still returns FP32 activations. Under normal FP16 autocast the head consumes the half storage directly. Outside that context, it reconstructs an FP32 weight operand to preserve ordinary computation.

The tiny-model checks preserve exact logits, loss, LoRA gradients, dropout RNG state, saved adapter tensors and adapter configuration. Nevertheless, full base `state_dict` storage dtype and `model.dtype` change. The audited Qwen, PEFT and Soup task paths derive their computation dtype from other relevant inputs/settings, but broader consumers could depend on `model.dtype`. This is why `FROZEN_STORAGE` remains false pending actual T4 verification.

Together, the two optional changes remove **2,988,703,744 bytes, approximately 2.783 GiB**, from these retained representations at maximum shape. That sum must not be advertised as a measured whole-run saving.

## 2. Original NF4 work now covers the actual forward and backward paths

Patching only `bitsandbytes.functional.dequantize_4bit` is incomplete for the pinned release. Training forward enters `torch.ops.bitsandbytes.gemm_4bit`; its CUDA dispatch selects `_dequant_linear_fallback` above 1,536 input rows. The maximum task batch has 2,048 rows. That private fallback decodes through backend helpers and bypasses the public function. Backward uses the public function. Sources: [CUDA dispatch](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py) and [autograd](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py).

The new original Triton decoder covers both boundaries without changing the native decision to use a fused GEMM on other shapes. It reconstructs a nested scale and decodes a packed byte into two weights in one pass:

```text
scale = round_fp32(scale_code[q] * secondary_absmax) + offset
weight_even = NF4[high_nibble] * scale
weight_odd  = NF4[low_nibble]  * scale
```

The multiply and offset add remain separate FP32 operations, matching the materialized reference. The implementation uses the canonical NF4 representation table, not copied kernel code. [QLoRA](https://arxiv.org/abs/2305.14314) is the primary background for NF4 and double quantization.

Code generation exposed an avoidable shared-memory conversion in the first vector-interleave design. Replacing it with aligned packed FP16 stores removed that conversion. Odd tails use one masked half store; unaligned output views use individual half stores. All five final decoding configurations compile with **zero shared memory** and preserve the separate scale-rounding operations.

CPU interpreter checks match the native bitsandbytes CPU representation exactly for FP16/FP32, flat/nested scales, every nibble combination, scale-block boundaries, transposed API inputs, odd tails and output sentinels. A wrapper test exercises both integration boundaries. Native CUDA parity and real `Linear4bit` scaled backward remain tests to execute on the T4.

`FUSED_NF4=True` is now part of the candidate. It is version-gated to the reviewed 0.50.2 implementation. It caches only a 64-byte format table. Caching every dequantized base weight would instead require about **2.44 GiB** for this architecture, undermining the memory objective; that is not implemented.

NF4 provides more useful original training work, but its inclusion alone does not prove the required 5% share. Making a kernel faster can reduce its percentage of wall-step time. The share must be checked for the combined useful original kernels in every repeated solution run.

## 3. A more credible T4 fused-head experiment

The previous ordinary Triton-dot experiment exposed the pinned compiler's SM75 limitation: Triton 3.6 declines MMA lowering for compute capability below 80. The prior explicit PTX projection bypassed that limitation but loaded fragments directly from global memory. The primary compiler source is [Triton v3.6.0 `AccelerateMatmul.cpp`](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp).

The new experiment stages a 64-token by 128-vocabulary tile through shared memory with a K=32 slice. Four warps execute the documented Turing `mma.sync.m16n8k8` operation, round each logit to FP16, and form max, exponential sum, centered moment and first-index argmax without writing a global score tile. Those sufficient statistics preserve loss, entropy and prediction requirements. The instruction mapping follows [NVIDIA's PTX specification](https://docs.nvidia.com/cuda/archive/11.7.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688).

| Offline full-target resource | Result |
| --- | ---: |
| Threads per CTA | 128 |
| Registers per thread | 128 |
| Static shared memory | 15,360 bytes |
| Dynamic shared memory | 0 bytes |
| Stack/spill stores/spill loads | 0 bytes |
| MMA instruction sites per K=32 body | 64 |

The static allocation is visible in the cubin, while Triton's usual launch metadata reports dynamic shared memory separately. The updated verification script checks both, avoiding the misleading conclusion that this kernel uses no shared memory.

For 256 tokens, the four partial statistics occupy **4,861,952 bytes / 4.64 MiB**, versus **77,791,232 bytes / 74.19 MiB** for the current FP16 score chunk. This is a 16-fold reduction in that intermediate. A full 2,048-token forward would use **37.09 MiB** of partials, still below the current 256-token score buffer, and could replace eight outer forward iterations with one projection/statistics launch and one merge.

Requested projection input loads fall fourfold compared with our earlier direct-load MMA experiment, before caching. That comparison is not against cuBLAS, and requested loads are not measured DRAM traffic. The T4's resource limits, synchronous staging, and vendor GEMM efficiency still make performance uncertain. The [Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/) is the relevant hardware reference.

This routine is **forward-only and unintegrated**. CPU fragment models, unique-writer checks and merge tests support its design; actual assembly execution, race checking, near-tie numerics and runtime remain open. It should not become the default merely because it emits Tensor Core instructions.

## 4. A useful correction to the sparse-head cost model

The streamed head recomputes scores for backward. With `r` denoting supervised shifted-label rows divided by all rows, a simple GEMM-work model is:

```text
Dense reference:  1 forward + 1 hidden-gradient GEMM = 2
Streamed sparse:  1 forward + r recomputation + r hidden-gradient = 1 + 2r
```

| Supervised output fraction | Streamed projection-equivalents | Fraction of dense GEMM work |
| ---: | ---: | ---: |
| 25% | 1.5 | 75% |
| 50% | 2.0 | 100% |
| 75% | 2.5 | 125% |
| 100% | 3.0 | 150% |

This is arithmetic, not a speed prediction. Small GEMMs, different cuBLAS algorithms, a synchronized compaction step and saved memory all affect runtime. It explains why sparse backward can be valuable without being universally faster. The new benchmark deliberately spans different supervised fractions instead of assuming a favorable one.

Forward cannot simply discard prompt rows: required entropy includes all attention-mask positions. Only mathematically zero direct output gradients are skipped. Causal attention continues propagating gradients into earlier prompt representations.

## 5. Verification and measurement are now separate, reproducible steps

The current local verification is:

| Evidence | Result |
| --- | --- |
| Official regression suite with patches | 79 passed |
| Additional CPU/interpreter suite | 99 passed; 22 CUDA-only skipped |
| Offline SM75 configurations | 21 compiled, including one excluded starter specialization |
| Static original kernel candidates | 11, including disabled experiments |
| Protected files changed | None |
| Banned imports, source parse failures or starter lookalikes | None detected |
| Physical GPU execution and official admission | Not run |

The new `tools/benchmark_candidates.py` plans 28 cases: 12 head, 4 projection, 8 NF4 and 4 fused-statistics comparisons. Fused rows are 256 and 2,048. Every case runs in a separate process with a fresh compiler cache. A short-lived preflight avoids keeping an extra parent CUDA context resident beside workers.

The first operation's synchronized wall time includes its compilation. Warm CUDA-event intervals and synchronized wall times are separate. Correctness is checked before warm measurement; NF4 requires identical FP16 output. Head checks include the real TRL entropy helper, GradScaler at 65,536, CE, predictions, supervised accuracy, unscaled hidden gradients and exact zeros for ignored direct output gradients. The runner records source hashes, versions, device details, raw samples and failures.

Those screens omit decoder training, optimizer, adapter evaluation and Rewind. Their allocator peaks are not externally sampled device peaks. They cannot establish the competition's performance, quality or original-kernel share. On this CPU-only workspace the runner correctly reports `blocked_no_cuda` with no measured results.

There is also a protocol correction worth preserving: the official harness summarizes time by **median**, which equals mean with two repeats, but memory by the **maximum sampled peak** across each side's runs. It evaluates the first successful adapter per side. Source: [the unchanged official harness](https://example.invalid/redacted-github).

## 6. The next measured candidates

1. **Default:** streamed head + original RMSNorm + new fused NF4, with the existing credited SwiGLU.
2. **Memory candidate:** default plus compact saved MLP activations, after its native CUDA tests pass.
3. **Additional memory candidate:** previous candidate plus verified frozen tied storage, after native CUDA and adapter compatibility checks.
4. **Research experiment:** full-batch fused T4 forward statistics. Adopt only after numerical/performance evidence and a valid backward integration.

The settings are source-controlled and fixed across measured runs. Do not modify the protected task, allocator, batch order, dropout, optimizer or logging. Do not combine speculative component speedups into a claimed whole-run result. The final benchmark must use the unchanged repeated harness and pass every admission check on the exact submitted source.

The platform-created team repository and a runnable Kaggle T4 remain the missing execution resources. The implementation, tests, experiment runner and detailed handoff are ready for that next step. The deadline is **31 October 2026, 23:59 Astana/Almaty (18:59 UTC)**.
