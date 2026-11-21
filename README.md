# Soup Kernel Challenge

[Русская версия](README.ru.md)

An optimization project for LoRA fine-tuning on a single NVIDIA Tesla T4. This repository contains the final solution and paired benchmark evidence, the earlier experiments that led to it, and the tools built to run, check, and explain those experiments.

## Final result

The saved Kaggle v1 session used the fixed Soup workload and ran **baseline → solution → solution → baseline** on one T4. It passed **10/10 automated checks** and **79/79 regression tests** in that session.

| Metric | Baseline | Solution |
|---|---:|---:|
| End-to-end command time, median of two runs | 86.39 s | **47.73 s (1.81× faster)** |
| Training-loop time, median of two runs | 62.95 s | **33.63 s** |
| Peak GPU memory, maximum of two runs | 14,345 MiB | **2,263 MiB (84.2% less)** |
| Held-out evaluation loss | 2.04966 | 2.04984 (+0.009%) |

[Raw measurements](docs/evidence-final-v1/result.json) · [Automated checks](docs/evidence-final-v1/checks.json) · [Measurement rules](docs/CHALLENGE_CONTRACT.md)

These are historical measurements, not a fresh test of the public cleanup. The automated pass does not establish jury approval or placement. The final run and the development runs below use different comparisons.

## What we built

- **Output head and loss:** process the 151,936-token vocabulary in blocks instead of retaining the full logits tensor, while still computing loss and training metrics across the vocabulary.
- **MLP activation memory:** recompute gate/up projections during backward and use grouped or packed operations on supported paths instead of retaining wide activations.
- **Quantized projections:** share NF4 dequantization across compatible projections and group QKV operations without changing the stored quantization format.
- **Normalization and positional operations:** optimize RMSNorm and RoPE, reduce intermediate buffers and casts, and avoid unnecessary work on padding rows in supported paths.
- **Integration and validation:** install the patches via `solution.apply()`, use the supplied workload and regression suite, and record wall time, GPU memory, output parity, evaluation loss and original-kernel activity. The saved LoRA adapter remains in the standard format.

## Research ledger and evidence

The project kept three different kinds of records: **ideas and source reviews**, **candidate submissions and queue attempts**, and **measured runs**. A catalog row or submitted notebook is not a benchmark. Retries, rejected ideas, incomplete runs, and results under different baselines are kept separate rather than added to the final benchmark.

| Ledger section | Recorded coverage |
|---|---|
| Soup Ideas | **147 entries:** 131 deferred, eight duplicates, five rejected, three undecided. These are catalog decisions, not run outcomes. |
| Fusion Ideas | **67 distinct ideas; nine completed solo screens.** A completed screen is not a paired benchmark. |
| Soup Runs | **133 records:** 90 need evidence, 39 failed and four passed. The four passing records are baseline references, not winning candidates. |
| Execution queue | **261 unique candidate/request IDs:** 88 completed, 172 failed and one was rejected. Among the 88 completed, 83 failed checks, three lack a complete comparison, one showed no gain and one beat its saved optimized comparator. |

These are aggregate counts from the research ledger, not four disjoint sets. The ledger also preserves abandoned approaches and retries rather than counting them as new verified results.

| Research stage | What we investigated | Public evidence |
|---|---|---|
| Source catalog and shortlist | Reviewed hardware and workload fit, legality of transferring an idea, and the proof needed before implementation. The first snapshot has **118 unmeasured ideas**; a closed snapshot has **121** (68 for memory/data movement/compilation, 25 for NF4/GEMM, 13 for normalization/activations, eight for head/loss/metrics, and seven for LoRA). A later research index records **131 ranked ideas**. These are successive snapshots, not additive counts. | [118-row catalog](research/soup-20261030/MASTER.md) · [Shortlist](research/soup-20261030/SHORTLIST.md) · [121-row snapshot](research/cycles/20261030-cycle2/README.md) · [131-row index](research/README.md) |
| Correctness and counterexamples | Checked centered cross-entropy, entropy/accuracy/rewind, compact head backward, RMSNorm and residual cast order, RoPE, LoRA gradient boundaries, and NF4 decode/dequantization. The closed CPU series retained **30 passing fixture groups and one native CPU NF4 parity failure**. A separate materialized CE prototype passed its engineering checks but was not bit-exact. None of this establishes GPU speed or admission. | [CPU prototypes](research/cycles/20261030-cpu-prototypes/README.md) · [Norm and RoPE](research/cycles/20261030-cycle3/README.md) · [Head and NF4](research/cycles/20261030-cycle4/README.md) |
| Early paired Kaggle T4 runs | Four candidates tested original RMSNorm, backward-capable NF4, entropy-only reduction and forward-only NF4. All four fell below the useful original-kernel activity gate; only two had full saved raw results and checks in that snapshot. A separate **77.55 s** accuracy-only run met its engineering screen but used no original Triton path and had no paired baseline. | [RMSNorm](research/runs/20261030-trial001-rmsnorm/README.md) · [NF4 backward and entropy](research/runs/20261030-trials002-003/README.md) · [NF4 forward](research/runs/20261030-trial004-nf4-forward/README.md) · [Accuracy-only](research/runs/20261030-trial006-accuracy-screen/README.md) |
| Component probes and controls | Screened sparse and exact-shape NF4, direct CE gradients (dZ) and loss batching, RoPE component versus whole-task controls, and attention with variable lengths or offload. A component pass or microbenchmark does not demonstrate end-to-end speed. | [Shape and CE plans](research/cycles/20261030-shape-ce-planning/README.md) · [Wave 2](testing_queue/candidates/20261030-wave2/) · [Wave 3](testing_queue/candidates/20261030-wave3/) · [Batched CE and shared-cast tests](testing_queue/candidates/20261031-resumed-results/README.md) |
| Full-step T4 development and profiling | Tested compact and padding-aware MLP, frozen-weight and NF4-scale cache budgets, FP16-logit reuse, LoRA epilogues, fused residual/RMSNorm, grouped QKV, RoPE and activation offload. Two examples from the development queue: **58.78 s / 9,325 MiB** (speed-focused) and **66.33 s / 5,005 MiB** (memory-focused). These completed development runs used different evidence and checks from the final paired Kaggle result. | [Speed/memory scoreboard](research/20261031-t4-speed-memory-scoreboard.md) · [Pareto results and source recovery](research/20261031-best-5878-and-scale-cache-handoff.md) |
| Integration and negative results | Explored grouped and packed backward paths, runtime-shape dispatch, exact frozen-weight reuse, fused output-head/MLP/QKV variants and source-bound fusion controls. Saved failed admission checks, out-of-memory or time-limited attempts, and a one-bit storage hypothesis: for the tested codec, corrections erased the storage benefit on sampled decoder blocks and the first-batch quality screen failed. | [Fusion prototypes](research/integrated_fusion/) · [Separate optimized-control experiments](research/soup-20261031/lowmemory_bestbaseline_fusion_r30_r31.md) · [One-bit diagnostics](research/runs/20261031-onebit-hypothesis/README.md) · [Notebook archive](soup-kernel-challenge/) |
| Runner, queue and records | Built candidate packaging, queued execution, result collection, checks and smoke tests; tracked CPU-master leases and replay prefixes separately from new full runs; kept provenance manifests and a static baseline graph. No live endpoint is offered by this repository. | [Queue tooling](testing_queue/) · [Service code](service/) · [Baseline graph](graph/baseline/) · [Demo script](presentation/Demo-Script-RU.md) |

The replay and output-head notes also record these narrower results:

- **LoRA saved-state replay:** 66/66 comparisons passed, with **12 MiB per pair in CPU accounting**. Later checks passed 41/41 interpreter cases and 9/9 CPU autograd cases. Neither set proves end-to-end GPU behavior or device-memory savings.
- **Exact-scale output-head fallback:** **72.87 s / 5,363 MiB** and 79 regression tests passed; no paired baseline was recorded for this run.
- **Packed-word NF4:** the regression variant was rejected. Passing 79 tests on nearby variants did not validate the rejected variant.

One queue experiment combined packed T448 replay with a 4,096-wide vocabulary-head chunk at N=1,024. It recorded **38.12 s for the training command, 25.65 s in the loop, 94.09 s for the full job, and 2,261 MiB peak memory**. It beat a **saved optimized comparator** of 38.49 s at 2,261 MiB on internal time, loss and memory checks. This was not a fresh paired comparison and does not replace or outrank the final 47.73 s paired submission.

The [fast-v2 screen](research/runs/20261031-allresearch-approved/) recorded **62.71 s / 7,955 MiB** without a fresh paired baseline. The lower-memory attempt in the same archive stopped at **46 of 64 steps**; its observed peak cannot be used as a full-run result.

## Reproduce the benchmark

Start with the [Kaggle notebook](notebooks/Soup-Duo-Final-Submission-v1.ipynb) and select a single T4. It checks the integrity of 38 bundled source files; package, model and dataset downloads still require internet access. A GitHub token and additional branches are not required.

From a compatible CUDA environment and the repository root:

```bash
python -m pip install -r requirements.txt
python task/prepare_data.py
python harness/run_task.py --repeats 2 --timeout 900
python harness/check_rules.py
python harness/score.py runs/result.json
```

The fixed workload uses Qwen2.5-1.5B-Instruct, Soup 0.77.0, double-quantized NF4 with FP16 compute, and LoRA on `q_proj` and `v_proj` (`r=16`, `alpha=32`). It runs 64 steps on 256 instructions, with batch size 4 and sequences up to 512 tokens. End-to-end time includes setup, compilation, training and adapter save.

The [solution](solution/), [runner](harness/), [task configuration](task/) and [regression tests](tests/) are at the repository root. Historical evidence and the pinned submission notebook preserve original artifacts; some archived notebooks rely on unavailable services and are not part of this reproduction path. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
