# Kernel: current implementation handoff

**Updated:** 30 October 2026. **Goal:** a competitive Soup Kernel Challenge entry for Decentrathon 6.0. **State:** an expanded source candidate, exact CPU/reference checks and offline SM75 compilation. **No physical T4 benchmark, admission result, or winning result exists.**

## 1. Resume here

Apply `Soup_Kernel_Candidate_Overlay.zip` to the official starter **inside the platform-created team repository**. The actual team repository URL and Kaggle notebook access were not supplied. No code has been pushed or submitted.

Starter: [archived-source/soup-kernel-challenge](https://example.invalid/redacted-github), inspected at `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. The overlay omits unchanged `harness/`, `task/`, `tests/`, `templates/`, `solution/swiglu.py`, `solution/test_kernels.py`, `LICENSE` and `NOTICE`. They must remain present from the starter. No compiled cache or binary is included.

Current solution hash, using the official harness's `solution_sha256` after applying the overlay:

```text
eb067e19031ce7a845b3366c30c91c6009717070938890d2bd79e7ccd84499f6
```

Read `Research_Update.md` for the new reasoning and tradeoffs, `README.md` for reproduction, and `validation/summary.json` for exact evidence and source hashes.

## 2. Fixed target and gates

| Item | Required |
| --- | --- |
| Workload | Soup 0.77.0; Qwen2.5-1.5B-Instruct; NF4 with nested quantization |
| Adapter | q/v LoRA, rank 16, alpha 32, dropout 0.05 |
| Training | Batch 4, up to 512 tokens, 256 rows, 64 steps |
| Hardware | One Kaggle Tesla T4, FP16 |
| Own kernels | Useful original Triton kernels >=5% of summed wall training-step time from step 3, in every solution run |
| Quality | Held-out adapter loss increase <=0.5% |
| Training agreement | First-step loss within 1%; mean curve and required metric/row-loss deviations within 3%; token counts exact |
| Performance admission | Runtime and peak memory each <=110% of baseline |
| Compatibility | Original 79 Soup regression tests with patches |
| Measurement | Full command including loading/cold compilation/saving; external sampled GPU peak; base/solution/solution/base |
| Deadline | 31 October 2026, 23:59 Astana/Almaty; 18:59 UTC |

Preserve task, optimizer, dropout, data/order, logging, Rewind, and protected files. Activation compression and weight repacking are permitted directions in the brief. The supplied SwiGLU never becomes eligible original work through renaming, different pointer types or a new wrapper.

## 3. Candidate settings

```python
CHUNK_TOKENS = 256
SPARSE_BACKWARD = True
HEAD_PROJECTION = "torch"
FUSED_NF4 = True
FROZEN_STORAGE = False
COMPACT_MLP_ACTIVATIONS = False
```

These are fixed source options, identical across measured runs. No production branch detects a benchmark, seed, dataset row or test. The new NF4 patch is enabled; the two storage experiments are wired but disabled. The fused T4 statistics routine is forward-only and unintegrated. `HEAD_PROJECTION="t4_mma"` selects the older explicit-MMA projection, not the fused statistics routine. Generic `"triton"` remains a known SM75 scalar-FMA experiment.

## 4. Implementation map

| File | Purpose and status |
| --- | --- |
| `solution/streamed_head.py` | Original stable vocabulary statistics, merge and scaled-logit-gradient kernels; bounded score chunks; sparse supervised-row backward. Enabled. |
| `solution/integration.py` | Scoped Trainer -> PEFT -> Accelerate -> Qwen call; exact masks/denominator/Rewind handoff; frozen FP16 head cache. Enabled. |
| `solution/rmsnorm.py` | Original normalization kernels preserving reference casting order. Enabled. |
| `solution/nf4.py` | Original packed-byte NF4 and nested-scale decoding; public backward and private forward-fallback patches. Enabled for bitsandbytes 0.50.2 CUDA. |
| `solution/activation_storage.py` | Saves exact gate/up values in FP16; uses unchanged starter arithmetic with FP32 gradients. Optional; adds no original kernel. |
| `solution/frozen_storage.py` | Lossless previously-FP16 tied weight storage, original gather to FP32 activations. Optional. |
| `solution/head_fused_t4.py` | Shared-memory 64x128 Turing MMA tile plus loss/entropy/argmax statistics. Forward-only experiment. |
| `solution/head_matmul_t4.py` | Older direct-global-load explicit Turing MMA projection. Optional experiment. |
| `solution/head_matmul.py` | Generic dot experiment exposing the pinned compiler limitation. Disabled. |
| `solution/swiglu.py` | Unchanged organizer example; excluded from original time. |
| `tools/benchmark_candidates.py` | 28-case synthetic GPU plan covering NF4, dense/sparse head, projection and fused statistics. |
| `tools/check_codegen.py` | Reproduces 21 offline compilation/instruction/resource checks with disposable caches. |

## 5. Critical details to preserve

**Labels and metrics.** Position `t` predicts `labels[t+1]`; ignored/final token CE is zero. Entropy covers every attention-mask position, including prompts and the final valid token. Accuracy uses shifted supervised labels. Rewind gets real `[B,T]` token CE; its sampling and row identity stay untouched.

**Gradient scaling.** Recompute scores in backward and apply the actual upstream loss scale before FP16 logit-gradient rounding. Do not precompute unscaled FP16 hidden gradients in forward.

**Sparse backward.** Only ignored output rows have zero direct head gradient. The decoder still propagates attention paths into prompt tokens. Compaction adds synchronization; more supervised rows can make recomputation costlier than the dense reference. Forward statistics remain complete.

**RMSNorm.** Normalize in FP32, cast the normalized result to input dtype, then multiply by the FP32 scale. Backward preserves the corresponding intermediate rounding.

**NF4 boundaries.** bitsandbytes 0.50.2 forward uses `torch.ops.bitsandbytes.gemm_4bit`; at more than 1,536 rows its backend calls `_dequant_linear_fallback`, bypassing the public decoder. Backward calls public `functional.dequantize_4bit`. Both are patched without changing GEMM dispatch. Nested scale multiply and offset add must remain separate FP32 operations. Decode high nibble first; packed FP16 output places that first value in the low 16 bits. Preserve odd-tail and unaligned-output stores.

**Saved activations.** The optional wrapper proves FP16 representability from ordinary frozen NF4 projections configured for FP16. In pinned bitsandbytes 0.50.2, explicit compute dtype makes `compute_type_is_set=True` at construction, so first FP32 input does not reset it. Save gate/up in half, but retain FP32 forward and backward arithmetic and gradient outputs. Do not claim its starter kernels as ours.

**Frozen storage.** Capture actual FP16 storage before unmodified PEFT preparation, then verify exact FP32 equality before restoring it. Never round arbitrary FP32 weights. Tying and adapter files are preserved, but base storage dtype and `model.dtype` change; this remains a GPU experiment.

**Wrapper/RNG behavior.** Call the real wrapped model once, keep AMP/PEFT/dropout, and reset scoped state on failure. Never retry a forward after consuming dropout RNG.

## 6. New research results

| Retained/intermediate allocation | Current bytes | Optional bytes | Meaning |
| --- | ---: | ---: | --- |
| Saved gate/up over 28 layers at batch 4x512 | 4,110,417,920 | 2,055,208,960 | About 1.914 GiB less retained storage |
| Tied head/embedding plus head cache | 1,400,242,176 | 466,747,392 | About 890.25 MiB less retained storage |
| 256-token head score versus fused partials | 77,791,232 | 4,861,952 | 16x smaller corresponding intermediate |

The two storage reductions add to **2,988,703,744 bytes, approximately 2.783 GiB**, at the maximum shape. This is arithmetic on allocations, not a measured reduction in external whole-run peak. Actual sequence lengths, lifetimes, backward temporaries and the allocator affect that peak.

The fused head can also process all 2,048 rows in one forward launch using about 37.09 MiB of partials. Its full target compiles with 128 registers/thread, 15,360 static shared bytes and no spills. It still needs physical correctness, race and performance checks.

Triton 3.6 ordinary `tl.dot` declines MMA lowering below SM80. Both explicit Turing experiments avoid this via documented `mma.sync.m16n8k8`; neither has physical GPU execution evidence.

## 7. Actual verification

| Check | Result |
| --- | --- |
| Official regressions with `solution.apply()` | **79 passed** |
| Additional CPU/interpreter suite | **99 passed; 22 CUDA-only tests skipped** |
| Offline SM75 cases | **21 compiled**, including one credited starter specialization |
| Original static kernel candidates | **11**, including disabled experiments |
| Starter kernels excluded | **2** |
| Banned imports / example lookalikes / parse errors | **None detected** |
| Changes to protected harness/task/tests or original example | **None** |
| Physical T4 runtime, peak, quality and original-kernel share | **Not run / unknown** |

New CPU checks include native bitsandbytes representation parity; exact saved-activation values/gradients; real PEFT preparation, tiny Qwen2 LoRA/dropout/RNG and saved adapter parity; and independent fragment/statistics models. Test-only device overrides do not establish actual NF4 GPU training.

Environment: Python 3.12.14, CPU PyTorch 2.11.0, Triton 3.6.0, transformers 5.16.1, peft 0.20.0, TRL 0.29.1, bitsandbytes 0.50.2, Accelerate 1.14.0, Soup 0.77.0. The Kaggle environment remains the acceptance target.

## 8. GPU experiment order

1. In the actual platform-created team repo, install the unchanged starter notebook's pinned environment. Set one visible T4.
2. Run `python -m pytest solution -q` and `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q`. The physical CUDA tests must actually execute.
3. Run standalone `--suite nf4` and `--suite head` screens. Keep their synthetic measurements separate from official results. The runner does not call `solution.apply()` and does not alter settings.
4. Measure the current default through the unchanged harness. Inspect its true original-kernel share and whole-command time.
5. Test `COMPACT_MLP_ACTIVATIONS=True`, then additionally `FROZEN_STORAGE=True`, as fixed recorded candidates. Retain only improvements satisfying all gates. Their predicted bytes are not measured results.
6. Screen `--suite fused` (default rows 256 and 2,048) only after its actual CUDA checks pass. It needs backward integration and full quality testing before becoming a training candidate.

Example screen:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/benchmark_candidates.py --suite nf4 --output validation/nf4_t4.json
```

Final unchanged protocol:

```bash
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2 --out runs/final_candidate
python harness/check_rules.py --result runs/final_candidate/result.json
python harness/score.py runs/final_candidate/result.json
```

The order is base, solution, solution, base. Wall time is summarized by **median** (equal to mean for two); GPU peak by **maximum** across each side's runs. Evaluation uses the first successful adapter per side. Checks bind exact result bytes and current solution source. Do not change source after final measurements without rerunning them.

Finish README results, final commit, saved public notebook, presentation and <=3-minute video only from real results. Their missing links remain explicit in `SUBMISSION.md`.
