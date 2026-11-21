# Shared MLP input-cast storage, independent adjoint edges

One source-bound delta to the verified batched-CE/storage composition: compute the
MLP input's FP16 bytes once, but keep separate gate/up backward cast edges. The
first branch uses the ordinary native cast; the second uses a separate autograd
view that returns only its own FP16 cotangent widened to FP32. Naive shared cast
fan-in is explicitly rejected by a numerical witness.

Parent source ZIP: `7f9882870836adf2829da353997091f31ec21e9333534cf91ba5fa54f0040eaa`
(parent's completed single screen: 70.872 s, 6,503 MiB, public loss 2.04963).
No new numerical kernel, matrix replay, changed GEMM, changed dropout, or original
kernel credit. All CE, full-M head, NF4, RMSNorm, frozen storage, native MLP owner,
and starter JIT/arithmetic stay unchanged. Production source/module/hook guards
remain binding; the input aliases cannot escape the source-pinned read-only
frozen projections. First-order-only behavior is explicit.

The hypothesis is that avoiding one known forward input cast per MLP offsets the
new small Python backward node. No attribution to the old 17.76% aggregate profiler
bucket, target latency, peak-memory or admission claim is made before execution.

18 small CPU checks passed using the actual new helper, including scaled cotangent
and external FP32 fan-in cases, finite/nonfinite/signed-zero cases, two explicit
frozen dense CPU surrogates, a naive-sharing counterexample, guards, source/policy
identity, and forward-only CPU cast count 2 to 1. They are not native BNB/CUDA proof.

The unchanged inherited native preflight assertions remain blocking. A narrow
additional check uses the already-loaded real pinned BNB MLP at scales 1/65536:
compare the verified parent's two independent casts against this installed owner,
requiring exact output/FP32-input-gradient finite bits and nonfinite classes,
separate backward edges/shared physical buffer, unchanged parameter identities,
versions/pointers, and CPU/CUDA RNG. It performs no extra model load or parameter
update and occurs before inherited route-count snapshots. This actual target check
has NOT run locally; missing native dependencies are not mocked away. The existing
warning-only allowlist changes only its derived strict-helper SHA, not its contents
or any threshold. Training requires all 1,792 new forward and backward edge routes.

Candidate-only 80-second single screen, one repeat, seed 1; no baseline or automatic
submission. If the new native check fails, stop before training and report that
specific integration failure. If completed runtime/peak do not improve materially,
stop this version rather than rerun unchanged. One independent reviewer owns this
new delta; prior source reviews are reused.
