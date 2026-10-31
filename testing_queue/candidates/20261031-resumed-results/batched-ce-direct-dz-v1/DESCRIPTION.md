## Hypothesis and exact delta

The full-M corrected head already retains all FP16 logits and performs one native full-M adjoint GEMM, but still classifies and summarizes CE in 128-row Python groups. Batch those unchanged row-local Triton kernels across up to 2,048 rows and return the single final FP16 dZ directly to the native adjoint GEMM. Do not change GEMM geometry, cast order, causal shift, global NLL reduction, denominator, real GradScaler upstream, or any numerical kernel.

At M=2,048, the structural all-safe route changes 16 forward CE groups to one, 32 classifier host reads to two, and 16 backward CE launches plus temporary-dZ copies to one backward launch with no assembled-dZ copy. These are source-derived operation counts, NOT measured timings or a whole-model speedup. Native exceptional rows are replayed in at most 128-row chunks, preventing the full batch from creating a larger full-vocabulary native temporary. Mixed native rows replace disjoint positions; their subset scalar losses are never combined.

This is an UNRUN composed derivative of the held compact-MLP/frozen-storage packet, not an isolated gain over a tested storage baseline. The reviewed frozen-storage implementation is byte-identical (SHA256 a3edfdfe568595e183ce57d858db4032d2e0e0b6cfe5bac8468dda3b0c299338). Existing CE/NF4/RMS kernels are reused unchanged; no new numerical-kernel or starter/storage original credit is claimed. Original packet remains immutable.

## Evidence and limits

Bounded CPU Torch 2.11.0+cpu tests passed native replay and disjoint scatter at finite, zero, negative, large, and nonfinite upstream values; native NaN/Inf/ignored labels; the 128-row native temporary frontier; loss/hidden-gradient equivalence of 128-row versus 2,048-row head scheduling with a native injected tile; and the M=2,048 single-forward/single-backward schedule. This is explicitly standalone changed-boundary evidence, NOT Triton execution or a fake HF/PEFT/Trainer fixture. The real native integration preflight is retained and must run in the supported environment. Triton, pinned real-HF/PEFT/bitsandbytes integration, target memory/time, task quality, and jury admission remain unestablished.

Batching changes numerical classifier scope. An all-ignored or near-only 128-row group may now join a mixed full batch; any invalid/nonfinite logit sends the full batch through bounded native replay. Existing native metric/gradient gates and explicit finite-LoRA-warning policy are retained unchanged; no threshold is loosened. Full-batch route telemetry must show 64 forward/backward/direct-dZ calls, 64 CE groups, zero assembled-dZ calls, and at least one call above 128 rows.

## Execution boundary

One candidate-only screen: native 64-step task and native public evaluation, `--only solution --repeats 1 --seed 1 --timeout 80`. No baseline, paired run, install, download, server/API request, or GPU action occurred during preparation. Server execution remains held until the user explicitly resumes it. Exact-source independent review is pending in this draft; reuse that single review and only repair concrete blockers.
