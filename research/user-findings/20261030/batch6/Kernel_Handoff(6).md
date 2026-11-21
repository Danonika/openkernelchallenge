# Kernel: current implementation handoff

**Updated:** 30 October 2026. **Goal:** a competitive Soup Kernel Challenge submission for Decentrathon 6.0. **Current state:** code implemented, CPU correctness/integration checks passed, offline SM75 compilation passed. **No physical T4 benchmark or winning result exists yet.**

## 1. Resume here

The source overlay contains the new/modified implementation, its tests, README, submission-status file, reproducible code-generation script, and validation records. Apply it to the official starter **inside the team repository created by the hackathon platform**. The actual team repository URL and Kaggle notebook link were not supplied in this session.

The unchanged starter is [archived-source/soup-kernel-challenge](https://example.invalid/redacted-github), inspected at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. Its original `harness/`, `task/`, `tests/`, notebook, and SwiGLU example are not duplicated in the overlay. They are required to run it.

Source hash after applying the overlay to that starter, using the official harness's `solution_sha256` function:

```text
8a27a18906c5cf05d21cc479dc84d92a31278c85a436e120ffe03fe6c9e1975e
```

## 2. Fixed target and admission gates

| Item | Required |
| --- | --- |
| Workload | Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4 |
| Adapter | q/v LoRA, rank 16, alpha 32, dropout 0.05 |
| Training | Batch 4, up to 512 tokens, 256 rows, 64 steps |
| Hardware | One Kaggle Tesla T4, FP16 |
| Own kernels | Useful original Triton kernels >=5% of wall training-step time from step 3, in every solution run |
| Quality | Held-out adapter loss increase <=0.5% |
| Training agreement | First-step loss within 1%; mean curve and required metric/row-loss deviations within 3%; token counts exact |
| Performance admission | Runtime and peak memory each <=110% of baseline |
| Compatibility | Original 79 Soup regression tests with patches |
| Measurement | Full command, loading and cold compilation included; sampled GPU peak; base/solution/solution/base |
| Deadline | 31 October 2026, 23:59 Astana/Almaty / 18:59 UTC |

Do not change the task, optimization switches, data/batch order, LoRA settings, optimizer, metrics, or harness. Copying kernel code from outside projects is forbidden; attributed ideas are permitted. The sample SwiGLU never counts toward the original-kernel share.

## 3. What now exists

| File | Implementation |
| --- | --- |
| `solution/streamed_head.py` | Three original Triton kernels: tiled softmax sufficient statistics, stable CE/entropy/argmax merge, and FP16 logit gradients. Custom autograd bounds score storage and preserves AMP scaling. |
| `solution/rmsnorm.py` | Original Qwen2 RMSNorm forward/backward with reference casting order and frozen FP32 scales. |
| `solution/integration.py` | Scoped training-head replacement retaining Trainer -> PEFT -> Accelerate -> Qwen2, exact metric masks, denominator forwarding, and the existing Rewind token-loss interface. |
| `solution/head_matmul_t4.py` | Experimental explicit Turing tensor-core projection; genuine `mma.sync.m16n8k8` instructions, physical lane addressing, tail masks, pointer-alignment fallback. |
| `solution/head_matmul.py` | Disabled generic Triton-dot experiment that exposed an SM75 compiler limitation. |
| `solution/settings.py` | Fixed candidate choices, identical across measured runs. |
| `tools/check_codegen.py` | Reproduces offline SM75 compilation/instruction checks with a disposable external cache. |
| `validation/summary.json` | Exact test counts, package versions, source hashes, static screening, and explicitly unmeasured competition metrics. |

The starter's `solution/swiglu.py` remains unchanged and explicitly credited.

Current settings:

```python
CHUNK_TOKENS = 256
SPARSE_BACKWARD = True
HEAD_PROJECTION = "torch"
```

The optional values are `"t4_mma"` and the known scalar-FMA experiment `"triton"`. Neither is enabled by default. Do not enable an experimental projection based only on offline compilation.

## 4. Decisions and correctness details to preserve

**Chunk the output head.** A `[4,512,151936]` FP32 logits tensor is 1,244,659,712 bytes. One 256-token FP16 chunk is 77,791,232 bytes, approximately 74.19 MiB. These are arithmetic buffer sizes, not observed whole-run memory savings. The frozen FP16 head cache, decoder activations and other allocations remain.

**Keep the metric masks distinct.** Loss and accuracy use shifted supervised labels. Entropy includes every attention-mask position, including prompts and the last valid token. Rewind receives `[B,T]` token CE, with position `t` predicting label `t+1`, ignored targets zero, and the final position zero. Row identities remain under Soup's sampler/recorder.

**Apply the true GradScaler value before rounding gradients to FP16.** Backward recomputes scores, forms `(softmax - target) * grad_output / denominator` in FP32, then casts for the head-gradient GEMM. Precomputing an unscaled FP16 hidden gradient in forward would lose small contributions before scaling.

**Skip only mathematically zero output-gradient rows.** Sparse backward gathers supervised shifted-label positions and scatters their gradients into a zero full output gradient. Forward still computes all statistics. The transformer continues propagating causal-attention dependencies into earlier prompt tokens. Compaction adds synchronization and gather/scatter costs, so its net benefit needs measurement.

**Preserve Qwen normalization casts.** Normalization occurs in FP32, the normalized value is cast to input dtype before multiplication by FP32 weights, and backward respects the corresponding intermediate cast.

**Keep normal wrappers.** Trainer invokes the wrapped model normally; a scoped record targets only the inner Qwen causal model. This preserves AMP and PEFT/dropout. Context is reset on failure and the model is never retried after consuming dropout RNG.

## 5. The T4 compiler trap found in this session

In [Triton 3.6 AccelerateMatmul.cpp](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp), the `computeCapability < 80` branch declines MMA lowering and falls back to FMA. Our ordinary `tl.dot` projection compiled for SM75 but emitted scalar `fma.rn.f32` and no `mma.sync` instructions. It is disabled.

The explicit T4 alternative follows [NVIDIA's m16n8k8 FP16 fragment specification](https://docs.nvidia.com/cuda/archive/11.7.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688). A 128-thread CTA covers a 32x64 output tile. All lane-sensitive operations stay inside a side-effecting assembly block. Generated PTX/LLVM shows full-warp entry, four unconditional MMA instructions per reduction iteration, and a memory clobber. Full dimensions, odd tails and unaligned-pointer variants compile. This is static evidence, **not a physical GPU correctness or speed result**.

## 6. Actual validation evidence

| Check | Result |
| --- | --- |
| Original Soup regressions with `solution.apply()` | **79 passed** |
| Additional numerical/integration suite | **50 passed; 14 CUDA-only tests skipped** |
| Offline SM75 configurations | **11 compiled successfully** |
| Original kernel candidates recognized by official static scanner | **7**, including two disabled projection alternatives |
| Starter examples excluded | **2**, correctly credited |
| Banned imports / copied-example lookalikes / parse failures | **None detected** |
| Modifications to protected harness/task/tests | **None** |
| Physical T4 execution, runtime, peak memory, adapter quality, >=5% share | **Not run / unknown** |

The integration suite uses a real tiny Qwen2 + PEFT q/v LoRA, nonzero adapter B factors, dropout 0.05, Accelerate output conversion and gradient scale 65,536. Test-only CPU capability overrides exercise the streamed path; production guards remain CUDA-only. This does not reproduce NF4 T4 training.

Local environment: Python 3.12.14, CPU PyTorch 2.11.0, Triton 3.6.0; pinned transformers 5.16.1, peft 0.20.0, TRL 0.29.1, bitsandbytes 0.50.2, Accelerate 1.14.0, Soup 0.77.0. The competition's Kaggle environment remains the acceptance target.

## 7. Next actions

1. Obtain the actual platform-created team repository URL and Kaggle notebook access/link. Put the source overlay there after the starter is present.
2. Run the unchanged notebook's CUDA correctness tests and the default candidate's official repeated benchmark:

   ```bash
   python task/prepare_data.py
   python -m pytest solution -q
   SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
   python harness/run_task.py --repeats 2
   python harness/check_rules.py
   ```

3. Read the real failures and per-kernel timings. The default may miss the 5% share even if it saves memory. Do not claim admission until all checks pass.
4. If needed, validate `t4_mma` on the GPU, then compare it using the same official harness. Compare chunk sizes or sparse backward only to address measured bottlenecks. Cold compilation is part of the score.
5. Use measured results to finalize README, the submitted commit, saved public notebook, presentation, and <=3-minute demo. Those submission links/results are currently incomplete in `SUBMISSION.md`.

No performance results, benchmark JSON, quality result, rank, or victory were fabricated. No code was pushed to a guessed team repository. All compiled caches and binaries are excluded from the source overlay.
