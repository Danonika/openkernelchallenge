---
name: soup-clean-benchmark
description: Prepare and verify an authorized clean Kaggle T4 Soup benchmark with unchanged task and harness, cold measurement, nested success checks, and reproducible artifacts.
---

# Soup clean benchmark

Read the shared [canonical rules and evidence schema](../soup-jury-admission/references/rules-v1.json) and [checklist guide](../soup-jury-admission/references/CHECKLIST.md). Do not duplicate or loosen thresholds here.

- Confirm current authority and quota scope before external execution. This project is cloud-only; never run on the user's PC. A skill is not permission for installation, a new GPU run, extra agents, paid resources, public visibility, or hidden-data access.
- Use one bounded candidate per isolated Kaggle session/GPU. Current user authority allows up to5 concurrent tests only when the verified account limit, free quota and isolation permit; do not assume5 slots exist. Prefer faster wall time within those limits, without duplicate runs, shared GPU contention or measurement interference.
- Freeze the official commit, solution hash and package environment. Only solution code is copied for judging; do not alter `harness/`, `tests/`, `task/`, allocator settings, batch/order/seed policy or adapter generation. Unsupported paths must preserve the baseline, not behave specially because a test or measurement is detected.
- Use the template's clean Run All path. Pin the intended final commit. The formal comparison is paired in one session in the prescribed order; single-side/single-repeat experiments are provisional. Keep normal JIT launches and cold-cache behavior; compilation/autotuning costs count. Do not upload precompiled caches, use CUDA graphs, leave GPU work behind, or count warm microbenchmarks as command time.
- Preflight only the selected candidate with fixed declared engineering tolerances, realistic full shapes and source-defined boundaries. Preserve extreme diagnostics. Such tolerances are not official quality gates. Do not expand into repeated speculative tuning before the first measured trial.
- Run patched unchanged Soup regressions and distinguish them from the starter's GPU tests. Verify actual T4, one visible GPU, idle start, unchanged allocator and externally sampled peak against internal reserved peak. The physical machine may contain two GPUs; the training process must use one and the sampler sums physical GPU memory.
- Inspect nested runs, evaluations, steps, outputs, source/launch provenance and exact hashes. A notebook COMPLETE state or orchestrator exit0 alone is insufficient. Recompute harness-native aggregates and show the PDF mean-versus-code-max memory discrepancy without changing either source. Evaluation uses the first successful adapter per side, not an invented repeated-eval mean.
- Run unchanged official checks and preserve their output/result/solution hashes. Feed the evidence to the jury checklist. Report measured failures and unknowns without scores or candidate points until evidence is reviewed. No leaderboard score can be invented without admitted-comparator information.

If access is denied or monitoring is blocked, record the last confirmed run state separately from observation failure. Do not label blocked observation a GPU failure, switch routes to evade denial, or submit a duplicate run. Use only supported authorized recovery; otherwise hand off the exact blocker.
