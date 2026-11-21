# Independent materialized-CE CPU prototype

Existing IDs: **SOUP-NEW-001 / SOUP-NEW-002**. CPU engineering evidence only; priorities and measured GPU outcomes are unchanged.

## Result and tested mechanism

An original independent centered forward plus analytic backward passed **28/28 predeclared engineering checks**: 24 functional comparisons and four controls. It uses selected log-probabilities with native thin-NLL SUM aggregation, preserves native external denominator division, and casts the already-scaled gradient to FP16 once. Explicit native fallbacks remain part of the tested implementation. This advances beyond the earlier native-log-softmax-state algebra oracle; it does not erase that oracle's FP32 association counterexamples.

The independent reference constructs causal targets separately and uses native cross-entropy, rather than the prototype's oracle helper. Tests cover B4/T1,3,7 at vocabulary 151936 and B4/T512 at vocabulary 19. The full [4,512,151936] tensor and real model were **not** tested. Input/label bytes and RNG state remained unchanged; expected routes and native fallback gradients were checked.

## Frozen tolerances and retained discrepancies

The jury fixed these CPU gates before execution; source and gates were not tuned afterward:

- Loss: rtol 1e-4 plus atol 1e-6; nonzero reference magnitude below 1e-3 additionally requires relative error at most 1%; an exact zero reference requires zero.
- Final FP16 gradient: exact positional NaN/+Inf/-Inf classes; finite rtol 2e-3 plus atol 2^-24, relative L2 at most 1e-3, and exact zeros on ignored finite rows. Gradients are compared without undoing upstream scaling.

| Case | Differing FP16 entries | Maximum absolute gradient error | Relative L2 |
|---|---:|---:|---:|
| Full vocabulary, T3, upstream 3 | 48 | 9.53674e-7 | 2.11288e-6 |
| Full vocabulary, T3, upstream 65536 | 4096 | 0.03125 | 4.01527e-6 |
| Full vocabulary, T7, upstream 3 | 75 | 4.76837e-7 | 1.89419e-6 |
| T512, vocabulary 19, upstream 65536 | 2 | 3.05176e-5 | 1.48486e-8 |

**Not bit-exact:** the worst observed per-entry relative gradient discrepancy is approximately 1.923% on tiny tails. The frozen absolute subnormal-ULP allowance admits those differences; it is incorrect to claim every gradient agrees within 0.2%. Nonfinite-class mismatches were zero in the tested cases. Largest SUM-loss discrepancy was 9.15527e-5, relative 3.42489e-7.

The deliberately wrong scale-after-FP16 control was **rejected**: eight entries exceeded tolerance, relative L2=1. A tiny real **CPU** GradScaler overflow control preserved both native skipped updates and scale reduction 65536→32768; this is not CUDA GradScaler validation.

## Limits and public contract

Real-model integration, CUDA reductions/rounding, the full large shape, higher-order behavior, allocator lifetime, GPU speed/peak memory, useful own-share, hidden quality and admission remain unproved. CPU passes cannot promote the separate measured entropy-only trial003 failure.

Public contract references: [causal shift and scalar division](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/loss/loss_utils.py#L32-L71), [native log-softmax epilogues](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/SoftMax.cu#L39-L63).
