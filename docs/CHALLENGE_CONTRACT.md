# Soup challenge contract audit — 30 October 2026

The official starter's current `main` resolves to `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`, the same commit used by the user's research. All 24 files were fetched through the GitHub connector and materialized exactly into `starter/`. Their Git blob hashes match the repository tree; see `starter_integrity.json`. This is a source snapshot, not an authorized team repository or a submitted commit.

## Measurement and target

- The official README reports approximately **79–83 s for the whole command**, of which **56–61 s** cover the 64 training steps. The user's **77 s baseline and 33 s record** are user-reported targets until their matching harness records are supplied.
- The scored time is process wall time including Python import/apply, model setup, fresh Triton compilation/autotuning, training and final adapter save. Downloads are warmed outside timing. Each measured run has a fresh compiler-cache directory.
- `--repeats 2` runs base, solution, solution, base. The time ratio uses median whole-command times; the memory ratio uses each side's **maximum** sampled peak, not its average.
- GPU memory is sampled externally with `nvidia-smi` every 100 ms and summed over all GPUs. The harness restricts CUDA to device 0. The device must be a Tesla T4, the GPUs must start below 1000 MiB total, and sampled peak must be at least 97% of Torch's reserved peak.
- To beat 33 s from an observed 77 s baseline requires more than 2.3333× whole-command speedup and more than 57.1429% time reduction. The required training-loop acceleration may be much larger because setup/save/compilation time also counts.

## Exact fixed task

Qwen/Qwen2.5-1.5B-Instruct, double-quantized NF4 with FP16 compute on T4; frozen norms, biases and tied embedding/head remain FP32 after PEFT preparation. LoRA q_proj and v_proj only, rank 16, alpha 32, dropout 0.05. One epoch, 256 Kazakh instruction rows, 64 steps, batch 4, max sequence 512, accumulation 1, lr 2e-4, AdamW Torch and cosine schedule. No checkpointing or packing switch changes. Public training rows 0–255, eval rows 40000–40255; jury changes rows and seed.

Install pins in the official notebook: `soup-cli[train]==0.77.0`, `transformers==5.16.1`, `peft==0.20.0`, `trl==0.29.1`, `bitsandbytes==0.50.2`, `accelerate==1.14.0`, plus pytest. Torch and Triton are recorded by the harness but not explicitly pinned by that install cell.

## Integration and permitted changes

- Only `solution/` is copied into the jury's clean starter. All implementation and runtime resources must live there. Do not depend on a developer script or notebook changing task, harness or tests.
- `solution.apply()` is called before Soup constructs model/trainer. Model methods, PEFT, bitsandbytes, TRL and Soup can be narrowly patched.
- Useful hooks: `SFTTrainerWrapper._setup_transformers` after its original call; `SFTTrainerWrapper.train` before its original call; Qwen2 module methods; `trl.SFTTrainer` before Soup constructs its Rewind subclass.
- Regression tests use fake modules/models. Guard setup hooks with a real `torch.nn.Module` and specific supported model/dtype/quantization conditions; call the original for unsupported inputs.
- Native GEMMs are allowed. Original Triton kernels/autograd/supporting patches are allowed. Sources of ideas must be attributed. Own recomputation/offloading is allowed with its time included; changing Soup's feature switches is forbidden.
- The supplied SwiGLU may be retained for performance, but never contributes own-kernel credit, including renamed or rewritten variants.
- No ready-made prohibited kernel packages, copied third-party kernels, installation during measured runs, added trainable parameters, changed task/data/batch order/settings/outputs, CUDA-graph launches invisible to the harness, artificial work, alternate GPUs, precompiled caches or measurement-dependent behavior.

## Admission gates

All ten checks and manual review must pass. Own `@triton.jit` kernels in `solution/` must execute in every measured run and together cover at least 5% of **synchronized callback step-wall time from step 3**, using CUDA-event timings from the same measured runs. Common launched kernel names across the runs determine eligibility; names should be unique. This is not the percentage of GPU profiler time. A 5–6% result is explicitly flagged as marginal.

Same global steps, adapter configuration/tensor names/shapes/dtypes, recorded trainer settings, logged metric names, exact token counts per step, Rewind rows and token counts. Mean relative deviations for entropy, accuracy and Rewind row losses ≤3%. First loss within 1%, mean relative full-curve loss deviation ≤3%. Held-out adapter loss at most 0.5% above baseline. Time and peak memory each at most 110% of baseline. All 79 Soup regression tests must pass with patches applied.

The quality evaluator runs plain Transformers/PEFT **without solution patches**, so the saved artifact must remain a standard compatible adapter. Keep genuine assistant-token loss, entropy, token accuracy and per-row Rewind losses; efficient calculation is permitted.

## Commands

```bash
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2 --out runs/final
python harness/check_rules.py --result runs/final/result.json
python harness/score.py runs/final/result.json
```

`check_rules.py` takes `--result`, not `--out`. `score.py` takes positional result paths. `--skip-tests` deliberately makes check 10 fail. The optional `--profile` adds a shortened separate profiling run; its numbers are not the scored run. `--only solution` is useful during iteration but does not satisfy comparison gates.

## Executor status

Observed here: `nvidia-smi` absent, no `/dev/nvidia0` or `/dev/nvidiactl`; default Python has no Torch, Triton, Transformers, PEFT, TRL, bitsandbytes, Soup or pytest. No target CUDA/T4 measurement is available. Direct sandboxed git failed because its proxy was unavailable; connected GitHub reads succeeded. No baseline, candidate runtime, memory, gradient correctness, kernel-share or official test pass was produced by this audit.

## Primary sources

- [Official starter README](https://example.invalid/redacted-github)
- [Fixed task](https://example.invalid/redacted-github)
- [Run harness](https://example.invalid/redacted-github)
- [Measurement instrumentation](https://example.invalid/redacted-github)
- [Rule checks](https://example.invalid/redacted-github)
- [Shared environment](https://example.invalid/redacted-github)
- [Notebook and dependency pins](https://example.invalid/redacted-github)
