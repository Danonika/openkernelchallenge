# Soup Kernel Challenge

[Русская версия](README.ru.md)

An optimization project for LoRA fine-tuning on a single NVIDIA Tesla T4. This repository contains the final solution and paired benchmark evidence, the earlier experiments that led to it, and the tools built to run, check, and explain those experiments.

**Second contributor:** `Ami-nur`.

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

## Research and engineering archive

| Workstream | What the archive contains |
|---|---|
| [CPU prototypes](research/cycles/20261030-cpu-prototypes/) and [early T4 trials](research/README.md) | Centered cross-entropy and NF4 reference tests, four early paired T4 trials that did not meet the original-kernel activity gate, and a separate quality screen. |
| [Development T4 scoreboard](research/20261031-t4-speed-memory-scoreboard.md) and [GPU research rounds](research/runs/20261031-allresearch-approved/) | Ablations for decoded-weight cache budgets, compact MLP, FP16 logits, LoRA epilogues, RoPE and activation offload. The archive includes completed, failed and time-limited runs alongside candidate checks. |
| [Fusion prototypes](research/integrated_fusion/) and [experiment notebooks](soup-kernel-challenge/) | Head/MLP/QKV combinations, runtime-shape and packed backward variants, diagnostic notebooks, and earlier attempts kept for comparison. |
| [Notebook runner](testing_queue/) and [service tooling](service/) | Candidate packaging, queued execution, result collection, smoke tests and deployment adapters. This repository does not provide a live endpoint. |
| [Research graph](graph/baseline/) and [demo script](presentation/Demo-Script-RU.md) | A static graph for the baseline and idea catalog, plus presentation materials describing the project. |

The [fast-v2 screen](research/runs/20261031-allresearch-approved/) recorded 62.71 s / 7,955 MiB, but was not paired with a fresh baseline. A separate low-memory attempt stopped at 46 of 64 steps; its observed peak is not a complete result. See the individual run records before comparing numbers.

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
