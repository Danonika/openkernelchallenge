# Soup CPU cycle 2b — LoRA ownership and selective saved-mask hooks

**3 fixture groups passed; 0 unexpected failures.** This increment preserves both earlier result sets. These are original cloud-CPU reference tests, not GPU kernels, timings, measured memory savings or challenge qualification.

## SOUP-LORA-001: B + scale + base-add autograd ownership

- Six paired cases: normal, subnormal and overflow incoming gradients, each in train/eval mode; shared X32 with separate q/v dropout invocations; M=17, H=13, rank=16, qN=23, vN=11.
- A GEMMs and dropout remain outside the custom function. The baseline defensive base clones remain present. The frozen base is a small independent FP32 linear stand-in, **not bitsandbytes emulation**.
- Outputs and RNG endstates matched exactly. X, A_q/A_v, B_q/B_v, base_q/base_v and S16_q/S16_v gradients had **zero finite-value error and zero nonfinite-class mismatches** in every fixture. Overflow NaN/Inf locations were preserved; they were not suppressed or treated as equal merely because both sides contained some NaNs.
- The custom backward preserves `G32 → G16 → multiply by 2 → D16`, FP16 dS and dB matmul outputs, then FP32 master B gradient conversion; dBase remains G32.
- Observed graph roots: baseline `AddBackward0` directly consumes `CloneBackward0` and `MulBackward0`; the toy custom root `BScaleAddBackward` receives S's `MmBackward0`, B's accumulation node, and the base clone. **Consolidating this autograd boundary does not itself fuse any Torch/GPU operations.**
- Three additional normal/subnormal/overflow cases checked installed CPU `autocast(dtype=float16)` + `F.linear(S16,B32)` against explicit B32→B16 casting. Outputs, finite gradients and nonfinite masks matched exactly. No CUDA autocast/real PEFT equivalence is inferred.

## SOUP-LORA-002: one saved-mask hook, not a broad hook context

- Registered only on the explicit CPU native-dropout node's `_raw_saved_result1`; lengths 1/7/8/9/257. Each graph was differentiated twice with a retained first backward.
- Outputs, input gradients and RNG endstates matched exactly; one pack and two unpack calls occurred per mask. Neighboring `sin` and `square` saved tensors retained their original pointers and were not intercepted.
- Explicit fallbacks cover no-grad/no-node, eval, unsupported nodes and metadata mismatch. Metadata mismatch keeps an identity-detached representation rather than packing an unsuitable tensor.
- Existing outer saved-tensor hooks remained operative; the recognized pre-registration conflict skipped our hook with **zero pack callback calls**.
- Two deliberate pack failures propagated: a ValueError and a RuntimeError with the **same text** as the known conflict. The wrapper checks both the exact conflict message **and that its pack callback was never entered**. Error-text matching alone would incorrectly hide the second failure after mutation. Failed test graphs were discarded, not silently reused.
- The final pack closure captures only expected metadata, not the output tensor/its graph, avoiding an unnecessary reference cycle. No original-mask debug clones are retained. Nevertheless, storage release, peak memory and speed were **not measured**.
- This is a **private, version-specific API** on an explicitly invoked CPU `aten.native_dropout` node. Selecting actual q/v PEFT nodes, actual CUDA dropout dispatch, compilation/checkpointing interactions, and GPU memory lifetime remain unvalidated.

## Reproduce and inspect

```bash
bash /redacted/local/soup_cycle/cpu_tests/cycle2b/run.sh
```

Artifacts in `cpu_tests/cycle2b/`:

- `run_cycle2b.py`: independently written toy custom autograd and selective hooks.
- `results.json`: full per-case gradient/error and hook evidence.
- `summary.json`: concise machine-readable outcomes.
- `run.log`, `console.log`: latest output; `results_initial2.json` / `run_initial2.log`: earlier two-group checkpoint.
- `cycle2_before.sha256`, `verification.txt`: earlier-cycle immutability and source integrity checks.

Python 3.12.14; Torch 2.15.0+cpu, actual build `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; CPU threads 2/2. No install, full model, dataset, CUDA, real GradScaler/optimizer step, remote write or official-test rerun. Both earlier result sets and the official source checkout remain unchanged.

**Stopping point:** the requested second-cycle CPU work is complete. Further promotion needs actual authorized T4 routing/resource/performance evidence, full numerical gates and owned-kernel share. No third CPU cycle is launched.
