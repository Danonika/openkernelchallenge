# Trial001 — original frozen-FP32 RMSNorm, measured standalone failure

**Reviewed result: NOT ADMITTED as a standalone candidate.** Useful own-kernel share was 1.205248976%, below 5%; this is the only failed automatic check (9/10 pass). All 79 patched Soup regressions and public quality checks passed. Hidden quality and final organizer acceptance remain UNKNOWN. The idea stays a useful conditional component, not a wholesale rejection.

| Harness-native metric | Paired base | Paired original norm |
|---|---:|---:|
| Median whole-command time, two repeats each | 83.62 s | 81.53 s |
| Maximum externally sampled GPU peak | 14345 MiB | 14343 MiB |
| First successful adapter public eval loss | 2.04967 | 2.04964 |

Time ratio 1.0256 and memory ratio 1.0001 are below the 1.05 scoring floor: no robust gain or earned score is claimed. Loop medians were 58.08 s and 60.10 s. These are one session's paired aggregates, not the earlier one-repeat baseline-v3 measurement. The PDF calls for averages; the frozen code uses median time and max memory. With two repeats median time equals mean, but max memory is not generally mean. Neither source is silently changed.

Read [the measured report](trial001_verified_report.md), [hash-bound handoff](trial001_verified_handoff.json), [official raw checks](runs_trial001/checks.json), [exact result](runs_trial001/result.json), and the [shared admission checklist](../../../../.agents/skills/soup-jury-admission/references/CHECKLIST.md). Existing catalog ID: `liger_castaware_norm`; no ID or earned point is added.

## Independent implementation

`solution/rmsnorm.py` independently implements one-row FP32 RMSNorm forward and a staged FP32 backward in Triton. It does not copy a reference kernel or activate the starter SwiGLU example. All matching Qwen2 norm modules may use it, guarded by exact source/class identity, width 1536, contiguous CUDA FP32 input/affine weight, same device, and frozen affine weight. Unsupported paths retain the original callable; an environment without Triton leaves the package patch unapplied.

Forward preserves square → true-width mean → instance epsilon → rsqrt → normalization → affine multiplication. Backward preserves the explicit direct and variance-path arithmetic stages; contraction is disabled. No residual-add fusion, precision narrowing, forced gradients, cache import, task switch, optimizer change or gradient-scaler manipulation. Saved input/weight/inverse-RMS use version-aware `save_for_backward` only when input gradients are needed; higher-order gradients are not claimed. No saved-reference-count memory saving is claimed.

## Numerical scope

This is a conditional tolerance-based experiment, **not universal floating-point identity**. Parallel reduction/rsqrt, native power versus a multiplication chain, and atomic backward accumulation can differ. Known extreme FP32 overflow and width-1536 norm-plus-bypass cancellation witnesses are explicitly recorded in `review/preflight_console_summary.json` (console-derived; individual fixture arrays unavailable); they are not hidden and are not automatically presented as whole-task failure.

The predeclared ordinary-case gate uses native expanded Torch norm arithmetic at widths/rows including [4,512,1536], ordinary activation scales 0.01/1/10, incoming gradient scales 1/128/65536, unit/nontrivial weights, both standalone and independent-bypass gradients, zero/tail controls, native fallbacks, mutation/version and no-grad behavior. Forward: allclose rtol 1e-4, atol 1e-5. Both backward results are divided by the known synthetic upstream scale before allclose rtol 1e-3, atol 2e-5. This scaling is fixture-only; training is unchanged. Raw error and finite/nonfinite classes are retained. These are engineering preflight thresholds, not the challenge's thresholds.


## Reproducibility

Official starter: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. Source hashes and the submitted runner hash are in `source_manifest.json`. The original three solution files, builder and generated credential-free runner are preserved byte-for-byte. Read the source before any authorized cloud execution; this publication does not launch or authorize a duplicate trial.

```bash
python verify_publication.py
# Generates a runner only; no GPU/network/installation:
python build_runner.py
# In a separately authorized pinned official clone with these solution files:
python -m solution.verify_trial001 --out gpu_preflight.json
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python task/prepare_data.py
python harness/run_task.py --repeats 2 --timeout 900 --out runs_trial001
python harness/check_rules.py --result runs_trial001/result.json
```

The actual measured order was base, solution, solution, base on one visible T4. Preserve cold compilation, normal launches, frozen task/harness/tests, output structure and the existing allocator. Do not slow or pad kernels, add redundant work, or rerun this same trial to seek a lucky ratio. Distinct useful kernels are the next research target.

## Evidence completeness and disclosure

The exact downloaded result/checks, both public evaluations and two base metric files are present and hash-bound. Per-solution metrics are embedded in the exact result. A recovered execution log established environment, preflight and wrapper summaries, but the full archive and standalone solution metrics/environment/preflight/session JSON were unavailable after an artifact-host policy block. Individual per-case preflight arrays are NOT claimed available. Known extreme cancellation (native0 versus regrouped1) remains recorded; ordinary-case preflight success is not universal bitwise identity.

This scientific package omits private account/notebook/launch/quota metadata, full control-plane/console logs, credentials, signed URLs, model/adapter weights, caches and private infrastructure/native-library provenance. Absolute private workspace prefixes in the historical source review were replaced by relative artifact references. Standard `/kaggle/working` paths in the original executable and exact scientific JSON are generic runtime paths, not account identifiers; retaining them preserves original hashes. Original private evidence is retained locally. No final-public-notebook, submission, leaderboard or official jury claim is made.

The dated pre-run contract is historical, not the current outcome. This measured snapshot supersedes its pending-result language without overwriting it. Verify all published hashes with `verify_publication.py`.
