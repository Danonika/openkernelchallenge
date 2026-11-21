# Trial001 jury contract — original frozen-FP32 RMSNorm

Decision at 2026-10-30 12:03 UTC: **conditional go to one bounded T4 trial**, existing catalog ID `liger_castaware_norm`. This is an independent implementation, not permission to use the cited library. No gain, numerical admission, official jury approval, or own-share success is presumed.

## Scope and ownership

Replace only the installed Qwen2 RMSNorm source path when input and frozen affine weight are contiguous FP32 CUDA tensors on the same device, hidden width 1536, and the expected module epsilon is available. All matching norms may participate; do not restrict to the final norm simply to avoid the known decoder-branch counterexample. Unsupported dtype/layout/device/trainable-weight/module paths use the original callable. Preserve task, data, dropout, optimizer, precision boundaries, parameters, gradients-needed state and ordinary JIT launches. No residual-add fusion, example SwiGLU activation, reference-kernel copying, caches, offload, forced gradients, or GradScaler manipulation.

Forward is FP32 square, sum over the true H divided by H, add epsilon, reciprocal square root, x*r, then affine weight. Backward is g=dy*w; direct=g*r; dr=sum(g*x); dv=(-0.5*dr)*r^3; variance=(dv/H)*(2*x); dx=direct+variance. Save only needed x/r/w with `save_for_backward`; no mutation. No higher-order-gradient claim. Installed Triton contraction-disable support must actually compile; no compiler installation/change.

## Bounded preflight

Use actual T4, installed Torch/Triton, normal eager reference Qwen norm arithmetic, H1536, rows 1/7/2048, ordinary x scales 0.01/1/10, all-zero controls, weight one and bounded nontrivial FP32 weight, upstream scales 1/128/65536. Compare standalone gradients and the original expanded norm plus an independent bypass (not a reference already containing a custom norm). Record raw maximum errors and finite/nonfinite classes.

Engineering go/no-go for ordinary finite cases, fixed before results: forward `allclose(rtol=1e-4, atol=1e-5)`; gradients after dividing both observed results by the known test upstream scale `allclose(rtol=1e-3, atol=2e-5)`. This normalization is diagnostic only, never a training change. Include patched-module routing, fallback for trainable weights/unsupported dtype/layout, no-grad, input non-mutation and saved-version protection. Run the unchanged 79-test regression suite with the patch. Failure of an ordinary-case gate stops the training trial and is retained as evidence, not tuned away by silently widening tolerances.

The width1536 extreme cancellation witness (a=b512, incoming norm gradient2^34, bypass1; CPU native0 versus atomic norm boundary1), overflow/nonfinite controls, reduction association, rsqrt lowering and native pow3 versus multiplication-chain differences remain explicit diagnostics. Their existence disproves universal floating-point identity, not automatically the challenge's tolerance-based full-run gates. Do not suppress or mislabel these outcomes. No unbounded microbenchmark/autotuning search before the first run.

## Measured-run and claim gates

After realistic GPU preflight passes, run one bounded original frozen-harness solution trial; keep cold-cache measurement behavior intact. Verify nested train/eval success, 64 steps, physical T4 / one visible GPU, idle and sampled memory, unchanged adapter/trainer/token/rewind structure, finite same-length curves, and actual eligible own kernels. An outer COMPLETE status or process0 alone is insufficient.

Official source thresholds are unchanged: first training loss relative gap <=1%; mean per-element relative gaps over loss, entropy, accuracy and rewind losses <=3%; public eval loss increase <=0.5%; total useful eligible own kernel share >=5%; time and memory growth <=10%; all regression and source/rule gates. These are not the preflight tolerances above. The harness uses median whole-command time and maximum sampled memory across repeats, not a made-up combined statistic.

A solution-only run compared with prior baseline v3 is useful diagnostic measurement, but is cross-session/single-repeat and cannot be represented as the official same-session paired/repeated admission or a score. Preserve the verified baseline as the only graph point until actual candidate evidence is reviewed and parent explicitly permits a measured point. No invented gain or candidate coordinate.

## Source review

Reviewed `trial001/solution/rmsnorm.py` and `solution/__init__.py` at the timestamp above. No source-level blocker to the stated conditional trial found. GPU compilation/execution and immutable final source hash remain pending.

Rules: [pinned README](https://example.invalid/redacted-github); [numeric and resource checks](https://example.invalid/redacted-github). Source/CPU constraints: `research_norm_increment3.md`, `cpu_cycle3_results.md`, approved `jury_cycle3_review.md`.
