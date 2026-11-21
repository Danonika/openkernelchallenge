# Soup cycle: training head, normalization and compiler controls

Research lane 2 · 2026-10-30 · cloud-only · first bounded cycle.

## Useful results first

1. **One genuinely narrower candidate survives deduplication:** `SOUP-TRAIN-001`, recomputing/backpropagating the frozen head only for causally supervised rows. All attention-mask positions still receive forward logits/entropy. The jury lane conditionally retained it at P1; 15 compact/scatter CPU cases now exactly match the tested dense autograd gradient. This is not permission to prune transformer prompt backpropagation.
2. **Strengthen existing `SOUP-NEW-002`, do not add a duplicate row:** preserve the maximum and centered log-normalizer separately. Storing only their sum can erase tiny CE and dominant-class gradient corrections, even for finite FP16-representable logits. Use centered entropy state rather than subtracting two large raw-logit quantities. CPU counterexamples already exist; compiled T4 behavior remains untested.
3. **Hugging Face kernel packaging is not an approved shortcut.** At the new inspected pin, `rmsnorm` lists CPU/XPU only; `layer-norm` lists CUDA capabilities beginning at 8.0, not T4's 7.5, and derives from FlashAttention. FLA fused-linear-CE precomputes gradients in forward and scales saved gradients later—precisely the FP16 GradScaler boundary that requires rejection as a drop-in Soup oracle.
4. Nanochat's current default-branch head is still the already-inspected July commit with a chunked-CE TODO. No fresh nanochat head optimization or Soup speed result was found.

No kernel was copied, imported, installed or executed by this research lane. No GPU, model download, authentication change or external write occurred. The 118-row master and all five requested prior reports were read for deduplication. Source-supported hypotheses and CPU witnesses below are not Kaggle measurements.

## Source authority and fresh pins

- **R — unchanged challenge authority:** [rules and permitted recomputation](https://example.invalid/redacted-github). Own Triton/autograd/patches and independently written recomputation are allowed methods; fixed task, precision boundaries, metrics, rewind, batches, steps and dropout remain obligations. Named-source ideas do not authorize copying or importing their implementations.
- **T — actual pinned trainer behavior:** [TRL 0.29.1 entropy and accuracy paths, lines 1238–1320](https://github.com/huggingface/trl/blob/617406a4e69551c115c145114497999618d24f3c/trl/trainer/sft_trainer.py#L1238-L1320), [entropy utility, lines 572 onward](https://github.com/huggingface/trl/blob/617406a4e69551c115c145114497999618d24f3c/trl/trainer/utils.py#L572-L609).
- **L — actual pinned loss:** [Transformers 5.16.1 causal shift, ignore index and normalization](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/loss/loss_utils.py#L35-L73).
- **P — numerical reference implementation, not a Kaggle version claim:** [PyTorch log-softmax epilogue](https://github.com/pytorch/pytorch/blob/e9a0032ac4a867ce57c079de379ece80e230be5d/aten/src/ATen/native/cuda/SoftMax.cu#L39-L49), current upstream commit dated 2026-10-30 10:55:22Z. It subtracts the row maximum before the log of the centered exponential sum. Actual Kaggle Torch is unpinned by the notebook and must be inventoried.
- **H — fresh inspected kernel collection:** [HF kernels-community commit 4498ed972091eb7d087fdb3853417f706193df28](https://github.com/huggingface/kernels-community/commit/4498ed972091eb7d087fdb3853417f706193df28), dated 2026-10-29 14:47:57Z. This is a provenance pin, not proof that every contained kernel changed yesterday.
- **N — freshly rechecked but unchanged nanochat:** [gpt.py head, July 3 pin](https://example.invalid/redacted-github).
- **C — compiler API evidence:** [Triton 3.8 Config controls](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/python/triton/runtime/autotuner.py#L324-L377). This does not establish the installed Kaggle version.

## New candidate: SOUP-TRAIN-001

### Mechanism and why it is distinct

For unchanged final hidden rows `X[B,T,H]`, let N=B*T and let U be the number of valid **shifted** targets. Preserve the prior dataflow correction: the head emits FP16 logits while Accelerate upcasts returned logits to FP32 for trainer-visible metrics; the hook position is part of the contract. The forward pass must still calculate entropy for every attention-mask position, including prompt and final positions. However, T computes entropy and accuracy under no-grad, and L ignores shifted label -100. With finite values and the fixed CE-only objective, the head's loss gradient is zero on all other final-hidden rows.

During the streamed-head backward:

1. Use an original-row index map for the same batch's shifted supervised positions only.
2. Recompute their FP16-rounded head logits and FP32 CE derivative using the unchanged global denominator.
3. Apply the incoming autograd scalar, including GradScaler, **before** the FP16 dlogits cast.
4. Compute FP16 dHidden for those gathered rows, cast at the original boundary, and scatter into a full-shape zero buffer.
5. Return that full-shape gradient to the unchanged decoder. Do not compact/remove prompt tokens in transformer attention/backward, where prompt representations can receive gradient from later supervised tokens.

This refines head lifetime/recomputation but is distinct from `SOUP-NEW-001` (generic token chunking) and `SOUP-NEW-006` (whole-model padding compaction with dropout/RNG obligations). It exploits a **loss gradient mask after the decoder**, not a new data batch or dropout mask.

### Derived work budget, not a speed forecast

For a frozen head and equal per-row GEMM work, dense head training needs one forward and one hidden-gradient matmul: approximately 2N row-units. Recomputing every row needs approximately 3N. This candidate needs approximately N+2U: full forward, supervised recompute and supervised hidden-gradient matmul. Consequently U/N<1/2 is the arithmetic crossover below the dense 2N count. This is an algebraic opportunity bound only: gathering, scattering, small-M GEMM efficiency, compilation, synchronization and reduction work can reverse it. No actual U/N distribution or throughput was measured.

### Separate classifications

| Axis | Verdict |
|---|---|
| Idea legality | Conditional semantic-preserving implementation of explicitly permitted own recomputation; not a task-setting change. Final jury review still controls admission. |
| Runtime dependency | Only baseline torch/triton plus a narrowly scoped own autograd patch. No external kernels. |
| Hardware fit | Ordinary FP16 GEMM and gather/scatter concepts fit SM75; compiled implementation and shape-specific rounding untested. |
| Feasibility | P1 experiment after materialized-logit reductions; benefits depend on U/N and actual GEMM routing. |
| Own-share eligibility | Only genuinely useful solution-owned Triton launches count, not cuBLAS work, Python indexing, or setup. Share cannot be inferred from work saved. |
| Evidence | Primary-source inference; jury conditional keep; 15 exact CPU compact/scatter cases; no GPU or end-to-end result. |

### Required disproof tests

- Shift before selecting rows; T=1, all ignored, entirely ignored sequence, first/final label, strided and ragged layouts; exact row IDs and token counts.
- Global `num_items_in_batch` rather than selected chunk count; preserve zero-sum/global-denominator versus all-ignored mean NaN behavior.
- Dense versus gathered FP16 GEMM output rounding and dHidden, because changing M can select a different matmul algorithm.
- Incoming scaling before both FP16 gradient boundaries; finite/overflow cases and baseline handling of nonfinite logits/upstream gradients. A zero-scatter shortcut must not silently suppress NaNs that would change GradScaler skip behavior. A generic nonfinite fallback requires its own proof and cannot be assumed free.
- Keep full forward entropy and rewind, causal targets and accuracy. Only the backward path is sparse; all original trainer hooks remain semantically current.
- Full normal decoder backward, LoRA gradients, clipping, optimizer/scaler/scheduler steps, saved adapter quality and unchanged harness gates.

## Refinement of SOUP-NEW-002: centered sufficient statistics

The existing proposal stores LSE and a weighted raw-logit sum. In real arithmetic that suffices; in floating point it can lose the value of interest.

For finite FP16-rounded logits converted to FP32, keep:

- m = max(z)
- s = sum exp(z-m)
- c = sum exp(z-m)*(z-m)
- target delta z_target-m, and first-index argmax.

Then evaluate CE in the baseline-style order `-((z_target-m)-log(s))`, log probabilities as `(z-m)-log(s)`, and entropy as `log(s)-c/s`. Do **not** form `m+log(s)` just to subtract m again. Backward probabilities should preserve the same centered subtraction instead of using only a rounded single LSE.

For two nonempty tile states a,b, let m=max(m_a,m_b), alpha=exp(m_a-m), beta=exp(m_b-m). The centered merge is:

`s = alpha*s_a + beta*s_b`

`c = alpha*(c_a + (m_a-m)*s_a) + beta*(c_b + (m_b-m)*s_b)`

This is an algebraic derivation, not copied source. Empty/masked tiles need explicit neutral handling: naively multiplying zero by an infinite maximum difference creates NaNs. All real vocabulary elements must remain; no tail or probability pruning is allowed.

Why this matters: logits such as [32768,32752] are finite and exactly representable in FP16. A tiny `log(s)` can vanish when added to m in FP32, turning positive CE into zero and changing the target-class gradient. The separate CPU lane also found that 19 equal logits at 65504 produce different CE/entropy under uncentered formulas. These are **adversarial correctness witnesses**, not claims that the actual model emits these logits or that the whole-run gate already fails.

The centered entropy identity still does not reproduce every rounding operation of TRL's explicit log-softmax→exp→product reduction. Near-zero entropy, ties, extreme gaps and vocabulary tails require tests; a two-pass implementation following the reference may be preferable. First validate reductions on already materialized baseline logits, then integrate head recomputation. Retain `SOUP-NEW-002` rather than inflating the table.

## Hugging Face kernel collection: narrow negative findings

| Inspected source at H | What source actually establishes | Soup disposition |
|---|---|---|
| [rmsnorm build manifest](https://github.com/huggingface/kernels-community/blob/4498ed972091eb7d087fdb3853417f706193df28/rmsnorm/build.toml), [autograd wrapper](https://github.com/huggingface/kernels-community/blob/4498ed972091eb7d087fdb3853417f706193df28/rmsnorm/torch-ext/rmsnorm/layers.py) | Build backends CPU/XPU, not CUDA. Wrapper separately requests input/weight gradients. | Not a T4 runtime candidate. Frozen-weight dx-only specialization is already in existing norm plans; no new ID. |
| [layer-norm manifest](https://github.com/huggingface/kernels-community/blob/4498ed972091eb7d087fdb3853417f706193df28/layer-norm/build.toml), [README provenance](https://github.com/huggingface/kernels-community/blob/4498ed972091eb7d087fdb3853417f706193df28/layer-norm/README.md), [wrapper](https://github.com/huggingface/kernels-community/blob/4498ed972091eb7d087fdb3853417f706193df28/layer-norm/torch-ext/layer_norm/layers.py) | CUDA capability list starts at 8.0 and omits 7.5; native CUDA source originates from FlashAttention. LlamaRMSNorm wrapper directly calls forward op. | Reject packaged substitution: source provenance/own-kernel legality and listed hardware coverage are independent blockers. A forward wrapper alone is not proof of full training autograd or Soup cast parity. Do not claim underlying mathematics inherently requires SM80. |
| [FLA fused-linear CE](https://github.com/huggingface/kernels-community/blob/4498ed972091eb7d087fdb3853417f706193df28/fla/torch-ext/fla/modules/fused_linear_cross_entropy/ops.py#L115-L247) | Saved dx has input dtype; CE derivatives and dx are computed in forward; backward multiplies saved gradients by incoming scalar. It also clamps ignored-label count and computes weight gradients whenever a weight object exists. | Not a drop-in FP16 GradScaler oracle or frozen-head specialization. Supports the prior recompute-in-backward caution, not a new candidate. It also names Liger ancestry. No import/copy. |
| [FlashAttention-derived CE adapter](https://github.com/huggingface/kernels-community/blob/4498ed972091eb7d087fdb3853417f706193df28/flash-attn-ops/torch-ext/flash_attn_ops/cross_entropy.py) | Different forward/backward tiling and compact LSE, with a labels-address alignment workaround. | Tiling overlaps existing head-reduction ideas. Alignment workaround is a useful adversarial fixture, not proof that actual Kaggle Triton has that bug. No library transfer or new ID. |

HF provenance does not remove a banned ancestor, authorize a ready-made package, or prove SM75 support. Conversely, rejected runtime packages do not ban their mathematical ideas when independently implemented and cited.

## Compiler/register lane: improve existing experiments, no duplicate rows

The inspected Triton Config source exposes maxnreg as a per-thread register ceiling; it separately identifies block clusters as SM90-only and software-pipeline stages as principally useful for SM80+ matmuls. Therefore:

- Add explicit maxnreg as an **existing occupancy-tuning refinement**, not a new mechanism beside L1-CUDAPY-02/SOUP-COMP-002. A lower cap can increase occupancy or instead force expensive spills; it is not automatic register optimization.
- Test `disable_licm`, accumulator multibuffering and maxnreg individually with identical math and tile shape. Require actual compiled register count, spills/local-memory traffic and complete cold-command time before keeping any setting.
- Keep `num_ctas=1` for SM75. Do not import Hopper register-reallocation or warp-specialized TMA machinery under the label of register tuning.
- For `SOUP-COMP-001`, test labels/masks/metadata views with real nonzero offsets, including int64 labels at addresses not 16-byte aligned. Do not assert false alignment. An old source workaround is not evidence of a present compiler defect, and turning off specialization is not proof of correctness.
- Existing runtime-scalar, pointer-alignment and loop-live-range controls remain the retained compiler ideas. No new compiler row or performance assertion is justified without actual compilation/profile evidence.

## CPU evidence and handoff

Final CPU results were inspected from [cpu_tests/results.json](cpu_tests/results.json), generated 2026-10-30 11:15:45Z; the original [test script](cpu_tests/run_cpu_semantics.py) was inspected too. Environment: Python 3.12.14, Torch 2.15.0+cpu, CUDA unavailable. Whole CPU suite: 8 passed, 0 failed. Relevant evidence:

- **SOUP-TRAIN-001:** 15 compact/scatter cases across mixed/all-supervised/all-ignored patterns and incoming scales 0.25, 1, 3, 128, 65536; all tested final-hidden gradients matched dense autograd exactly. A causal-mixing witness has zero ignored final-hidden gradient but nonzero earlier prompt-activation gradient, confirming why decoder prompt backprop must remain.
- **Centered state:** FP16-representable logits [32768,32752] give baseline CE 1.192092824e-7, single-LSE CE 0, centered CE 1.192092824e-7; baseline entropy 1.919772103e-6, single-LSE entropy 0, centered entropy 1.919771876e-6. The tiny centered entropy difference demonstrates that centered algebra is not a bitwise identity to the explicit reference.
- **Scaled boundary:** at incoming scale 65536, baseline/scale-before-cast hidden gradient was -0.002700805664; scale-after-cast was 0. This is a CPU counterexample, not a run of CUDA GradScaler.
- **Norm:** FP32 formula and residual-bypass fixtures passed selected tolerances, with width1536 included. Moving cast-before-affine or prematurely emitting FP16 produced nonzero differences. Do not label these tests full normalization equivalence.

The script SHA256 recorded by the results is `9e65cf8f1ab4bc5c2787d303fc7684ef89c3d45ae7c237d9afb4af8b0217f004`. A CPU pass establishes only the tested arithmetic/indexing contract, not cuBLAS/Triton rounding, real-model distributions, actual GradScaler CUDA execution, own-kernel share or speed. The compact-head fixture did not settle nonfinite/overflow parity.

Next narrow question: **For the observed unchanged batch shapes and supervised fraction U/N, does N+2U head work plus gather/scatter beat materialized dense or full-recompute head paths while passing scaled-backward and metric gates?** Evidence needed: actual Kaggle versions, a baseline head/metric profile and shape histogram, then one compiled boundary fixture. No GPU run is authorized for this lane. Further generic source searches should not manufacture more rows while those facts are missing.
