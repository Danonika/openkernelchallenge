# Head cycle 4 — a guarded full-contract CPU oracle, not a streamed implementation

2026-10-30 · existing `SOUP-NEW-001`, `SOUP-NEW-002`, `SOUP-TRAIN-001` · cloud-only source/CPU work.

## Findings that change the implementation contract

1. **The three token domains are different.** Main causal CE evaluates all B×T logits after appending an ignored final target; accuracy and rewind use only T−1 positions; entropy evaluates all B×T positions before multiplying by the attention mask. An exceptional final or padded row cannot be discarded by one common “valid token” mask.
2. **Masked entropy is multiplication, not selection.** A padded NaN entropy multiplied by zero still produces NaN. Likewise the explicit reference expression exp(logp)×logp does not substitute the mathematical zero convention for 0×−Inf. Exceptional fallback must preserve the executed expression, not sanitize it.
3. **A dense-forward, guarded-backward CPU oracle is a useful integration gate but no memory optimization.** It intentionally materializes exact rounded logits and native scalar CE. A passing oracle does not establish that streamed reductions, gathered GPU GEMMs, or a trainer patch are equivalent.
4. Retain the complete hidden tensor until backward dispatch: a nonfinite incoming scalar is only known then. Saving only supervised rows cannot implement a full original-geometry exceptional fallback.

No new catalog ID, kernel, benchmark point or candidate speedup is proposed. The observed official baseline is **77.92 s, 14,345 MiB, public evaluation loss 2.04965**, solely a reference from the [baseline evidence](kaggle/v3_verified_report.md), not a comparison against this CPU design.

## 1. Pinned source contract

### Boundary and loss

[Accelerate output conversion](https://github.com/huggingface/accelerate/blob/beb0672aa8444ea7647aee056f624effe5996346/src/accelerate/utils/operations.py#L889-L948) converts returned FP16/BF16 tensors to FP32. For the fixed frozen head, the narrow expected path is:

`hidden FP32 → autocast hidden FP16 / effective weight FP16 → head output FP16 → loss and trainer-visible logits FP32`.

The CPU oracle makes these casts explicit; it does not claim CPU half GEMM has the same reduction algorithm as T4. The prepared model's dtype, wrapper, weight/version, layout and bias assumptions remain runtime routing checks.

[Transformers causal loss](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/loss/loss_utils.py#L35-L73) upcasts logits, appends −100 to the labels and shifts left. Consequently the loss consumes all B×T rows, including each final ignored row. Supplied `num_items_in_batch` means native summed CE followed by division by that supplied denominator; absent it, native mean CE is used. Do not replace either scalar with a sum of independently reduced token/chunk losses without testing changed reduction order. Do not normalize by selected chunk count.

[Accelerate backward](https://github.com/huggingface/accelerate/blob/beb0672aa8444ea7647aee056f624effe5996346/src/accelerate/accelerator.py#L2818-L2850) applies configured accumulation normalization and loss scaling before backward. The incoming scalar therefore must reach FP32 CE backward **before** dLogits is rounded to FP16 and before the FP16 head backward produces dHidden. This CPU cycle passes explicit scalars; it does not execute CUDA GradScaler or validate optimizer/scheduler skip behavior.

### Metrics and rewind

[TRL entropy utility](https://github.com/huggingface/trl/blob/617406a4e69551c115c145114497999618d24f3c/trl/trainer/utils.py#L572-L609) processes flattened token rows in chunks of 128 and computes `−sum(exp(log_softmax(z))*log_softmax(z))`. [Trainer aggregation](https://github.com/huggingface/trl/blob/617406a4e69551c115c145114497999618d24f3c/trl/trainer/sft_trainer.py#L1238-L1320) multiplies this full-shaped token entropy by attention_mask before summing/dividing. Prompt and final positions participate. Padding is not permission to avoid computing a row whose exceptional entropy could contaminate this expression.

Accuracy instead compares first-index argmax of logits[:, :-1] with labels[:, 1:], masks −100, and returns 0 when the supervised count is zero. Preserve original vocabulary indices for finite ties; exceptional argmax is delegated to the native reference, not a guessed NaN ordering.

[Soup rewind](https://example.invalid/redacted-github) computes per-row token CE on T−1 positions, under no_grad, then applies the valid-label mask and divides that row's sum by max(row_count,1). Its values are per-sequence means, not training-loss contributions already normalized by the global denominator. Averaging row means is generally not the global token-weighted CE. Preserve original sampler row identities; token compaction must not reorder the sequence FIFO.

The [rewind wrapper](https://example.invalid/redacted-github) records training batches, not evaluation batches. The [existing chunk protocol](https://example.invalid/redacted-github) consumes its metrics packet once; [trainer wiring](https://example.invalid/redacted-github) distinguishes train/eval metrics and cumulative training tokens. These sources motivate an original synthetic lifecycle test, not permission to enable Soup layer streaming or copy its helper.

## 2. Guarded CPU integration design

### Reference and scope

Use tiny original tensors with frozen FP32 weight and explicit FP16 head operands/output. Keep batch/time coordinates and causal labels unchanged. Produce native scalar CE and a detached metric packet containing token entropy, scalar entropy, integer correct/count totals, per-sequence losses/counts and original sequence IDs. All metric work is non-differentiable. Reference and candidate intentionally share the native metric helper at this stage; exact metric agreement is an interface/oracle check, not a test of separately optimized metric arithmetic. Diagnostic retention of selected forward logits is allowed solely for comparing recomputation and is not a proposed memory-saving layout.

The dense reference is a normal PyTorch graph. The independent guarded custom boundary computes the same dense forward for this proof stage. During its backward it rebuilds one of two native graphs from saved unmodified hidden/weight references. This separation isolates dispatch, original-row mapping, casts, normalization, and metric lifecycle before introducing custom reductions or streaming.

### Finite fast path

Require the effective FP16 weight and **every actual FP16 forward logit** to be finite; require positive finite effective denominator/count, finite incoming scalar, and finite normalized upstream in the reference arithmetic dtype. A finite scalar loss alone is insufficient. Conservatively use the reference fallback for all-ignored batches and unsupported routes.

Gather only original head rows whose shifted labels are not −100. Recompute their FP16 head output and FP32 native CE, preserving native mean when the original denominator is absent, or native sum followed by the unchanged supplied denominator otherwise. Feed the incoming scalar through autograd so dLogits16 and dHidden16 round in the original order. In particular, division after native summed CE means backward forms the normalized upstream `grad_loss / denominator` before NLL and log-softmax backward. A future manual formula such as `((p−onehot)/D)*grad_loss` or even `(p−onehot)*(grad_loss/D)` changes finite arithmetic association relative to the native target subtraction and is not validated by this oracle. Native-autograd compaction deliberately avoids claiming that analytic-kernel equivalence. Scatter into a full-shaped hidden gradient; this does not prune earlier decoder prompt gradients.

The CPU fixture must record equality of the recomputed selected logits to original forward logits and compare the resulting positional dHidden values/classes. Even a successful CPU equality check is not a proof of GPU equivalence: changing GEMM M or layout can choose different algorithms. Actual selected-versus-dense rounded GEMM equivalence remains a separate T4 gate. Finiteness alone is not a proof of rounding parity.

### Reference-ordered exceptional fallback

If a guard fails, rebuild the **full original B×T head geometry**, including the last ignored timestep, and the native CE graph with the original scalar/denominator. Do not trim ignored rows, use `(p−onehot)*scale` as a universal nonfinite formula, sanitize values, clamp denominators, or manipulate GradScaler state. Preserve complete hidden state and correct saved-tensor/version handling until this decision is made.

The earlier nonfinite argument also matches the release corresponding to the observed baseline package version: [PyTorch v2.11.0 resolves to 70d99e9](https://github.com/pytorch/pytorch/commit/70d99e998b4955e0049d13a98d77ae1b14db1f45). Its [CE composition](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/LossNLL.cpp#L628-L661) applies log-softmax before NLL; [NLL backward](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/Loss.cu#L458-L485) skips writing ignored targets; [log-softmax backward](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/SoftMax.cu#L52-L62) still evaluates its expression over those rows. This is release-source evidence, not confirmation of the wheel's git hash or a CUDA fixture. CPU execution remains Torch 2.15.0+cpu at d0f2af8, distinct from observed Kaggle Torch 2.11.0+cu128 / Triton 3.6.0.

### Lifecycle and unsupported calls

Associate a detached metric packet with one exact forward call, its mode, original sequence IDs and unmodified labels/mask. Consume once; do not reuse a stale training packet after evaluation, an unlabelled call or an exception. Training updates cumulative train-token metrics and rewind; evaluation metrics do not consume the training sampler FIFO. A synthetic protocol proves only these local invariants, not successful HF/TRL/Soup integration.

Trainable head weights, bias/layout/model variants outside the narrow contract, generation/unlabelled calls, label smoothing, custom losses or requested logits unsupported by the new interface must retain their original path. Do not emit dummy logits merely to satisfy consumers, and do not silently stop returning requested outputs.

## 3. Bounded proof plan and limits

The CPU lane owns new original fixtures only; the previous 23 evidence groups are immutable and not rerun. New groups combine formerly separate claims:

- Finite complete head + CE + all metrics + guarded compact backward; shifted masks, ignored sequence, finite ties, odd vocabulary tail, global denominator different from local token count, and incoming scales.
- Exceptional dispatch matrix with reason recording and positional finite/NaN/+Inf/−Inf comparison, including backward-only scalar failures and actual FP16 conversion/output overflow.
- Discriminating final/padded-token and explicit entropy-expression counterexamples.
- Detached per-call packet, train/eval/FIFO lifecycle, no duplicate/stale consumption, and distinction between row means and global loss.

No CPU tolerance is a challenge tolerance. No measured memory saving follows from this intentionally dense oracle. Passing uses of native CE/autograd do not validate an independently implemented analytic gradient or vocabulary merge. The prior near-zero relative-error and residual-graph warnings remain open.

### Separate decisions

- **Legality:** conditional independent own-kernel idea transfer; no copied/imported source kernels, changed tasks, batches, precision, metrics, rewind, dropout, optimizer or scaler. Existing IDs and priorities only.
- **Hardware:** no GPU compiler or execution in this lane. T4/SM75 shape-dependent GEMM, CUDA CE and scalar handling remain to be tested on the actual stack.
- **Feasibility:** this is a correctness-oracle prerequisite, not a working streamed implementation. Guard synchronization, extra retained state, fallback memory, reductions, recompilation, useful own-kernel share and complete-run performance remain unmeasured.

## Final bounded CPU evidence

Inspected the independently written [head oracle](cpu_tests/cycle4/head_oracle.py) and its [results](cpu_tests/cycle4/head_results.json), generated **2026-10-30 12:03:17Z**. All **four head evidence groups passed**, meaning their stated positive and negative claims were established—not challenge admission. Torch 2.15.0+cpu, git d0f2af8, two intra-op/two inter-op threads. This research lane did not run a GPU or install dependencies. The prior 23 groups were not rerun; a hash check of their 43 recorded artifact/report files returned exit 0.

- **25 finite integration cases:** mixed supervision, ignored sequence, finite ties, V=19/31, absent versus deliberately different supplied denominator, and incoming scales .25/65536. One extra case used a scalar int64 tensor denominator retained with save_for_backward. Tested scalar loss, metric packet, selected recomputed FP16 logits, FP16 hidden-gradient boundary and FP32 returned gradient matched the native dense reference numerically. The metric helper and dense forward are deliberately shared, so this does not validate an optimized metric or streamed arithmetic implementation.
- **18 exceptional cases:** guard failure dispatched to the original full-geometry reference for ignored/padded/final NaNs, ignored +Inf, finite head operands overflowing FP16 output, weight conversion overflow/NaN, NaN/+Inf incoming scalar, zero/NaN/Inf denominator, normalized-upstream overflow, all-ignored batches, T=1 and a connected zero incoming scalar with final-position NaN. Tested finite values and positional NaN/+Inf/−Inf classes matched; no sanitizer or forced scaler decision was used.
- **Metric discriminator:** masking by multiplication gives NaN entropy for a padded NaN row, while incorrectly selecting only nonpadding rows produces 0.965503931. Mixed finite/−Inf log probabilities likewise give NaN under the explicit reference entropy product. Removing an ignored final head row changed two gradient positions from native NaN to zero, despite finite scalar loss 0.394229591 and finite per-row rewind loss 1.379803538.
- **Lifecycle discriminator:** two train packets preserved their original sequence ordering; evaluation did not enter the train FIFO or increment cumulative train tokens. Duplicate, out-of-order and repeated consumption were rejected in this synthetic protocol. Each packet had 15 attention tokens but only 6 shifted supervised tokens. Native global mean CE was 1.945910096; averaging per-row means instead gave 1.297273397. This is not actual Soup/TRL integration or a claim to change native recorder failure handling.
- **Scope correction:** actual recorded finite leaf/head strides were both [91,13,1], contiguous for shape [3,7,13]. A non-dense source slice had been cloned; it does not establish strided head support. Diagnostics retain selected forward logits solely to compare recomputation. Neither storage inventory nor peak-memory improvement is claimed.

Exact here means tested finite numerical values plus nonfinite classes, not signed-zero bit identity or NaN payload identity. Tensor-denominator mutation/device behavior, broad layouts, the actual prepared model, V=151936 reductions, CUDA GradScaler, optimizer state, quality, own-kernel share and performance remain unproved. Native CE/autograd in this oracle is not a substitute for validating the arithmetic order of an independent analytic kernel.

Script SHA256: `4a5ad5e41f56be6eaf61b5b9ba5360be02debfeb26d88300c773a11c3d403513`.
Results SHA256: `1f758a534a3328664768a3e2bb4a696880ce83aca8f73f6b69cda10ceb2bb762`.

**Next concrete proof:** on the authorized installed T4 stack, compare the same original full-head reference against gathered/recomputed FP16 head logits and gradients, then independently validate any custom CE/entropy arithmetic before touching the trainer interface. Keep the existing materialized-logit-reduction phase separate from streamed forward and its exceptional fallback allocation cost.
