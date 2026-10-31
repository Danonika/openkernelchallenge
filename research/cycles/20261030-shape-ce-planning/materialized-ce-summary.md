# Materialized CE cycle7 — smaller boundary, bounded CPU evidence

**Existing SOUP-NEW-001/002; planning and CPU semantics only.** Keep normalization outside the proposed custom node: return SUM CE and retain the original native division by the supplied denominator. The incoming adjoint then already contains native normalization and upstream scaling. This avoids dividing twice during a late native fallback.

The smaller scalar-reduction proposal is native one-column NLL over already-selected log-probabilities, preserving true labels/ignore positions and native scalar aggregation—not another softmax. The expensive vocabulary reduction would be original work; head GEMM, returned logits and trainer metrics remain native. **Thin NLL, independent centered forward reductions and actual hook integration have not been tested.** [Pinned causal loss](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/loss/loss_utils.py#L32-L71), [native NLL aggregation](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/Loss.cu#L237-L411).

Two bounded CPU proof groups established their stated claims, including negative witnesses:

- Using native log-softmax state, the staged adjoint matched tested final FP16 numerical values and positional nonfinite classes in 13 selected cases, **but three finite cases differed in FP32**. This is not FP32 bit identity or universal FP16/CUDA equivalence.
- Simplified arithmetic, ignored-NaN suppression and premature FP16 rounding produced counterexamples. Finite FP32 gradients overflowing on the final FP16 cast were retained, not clamped. No real CUDA GradScaler/optimizer was exercised.
- A native SUM-recompute oracle with external division matched three tested upstream cases. It proves that narrow first-order oracle topology, not an independently implemented analytic CE.
- In the small logical storage inventory, deduplicated saved-plus-returned storage was 1292 B native versus 984 B oracle; proposed row state would add 64 B. This is retention accounting, not live-memory or peak measurement.

At maximum task geometry, the earlier 593.5 MiB FP32-versus-FP16 retained-state difference must additionally account for 16 KiB of proposed row states and other lifetimes. It remains conditional arithmetic, **not a peak-memory saving**. Payload ratios likewise cannot predict useful ownership or speed.

Required future proof includes actual hook arguments/layout, full-vocabulary centered reductions, thin-NLL scalar behavior, staged gradients, exceptional/full-geometry fallback, version/higher-order behavior and complete unchanged task checks. The native head and precision boundaries stay intact. No extra broad CPU campaign, GPU code, launch, admission or performance point follows from these two groups. [Pinned native CUDA stages](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/SoftMax.cu#L39-L63).
