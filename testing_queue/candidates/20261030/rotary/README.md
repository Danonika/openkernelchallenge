# Original FP32 RoPE application candidate — prepared, GPU untested

**Hypothesis:** replace Qwen2's half-split FP32 rotary application with original out-of-place Triton forward/adjoint kernels to reduce pointwise launches and transient traffic. Only RoPE application changes. No QK normalization, early FP16 cast, frequency/table cache, attention backend, model/task/LoRA, optimizer, data order, or precision policy changes.

**Current result:** 20 original CPU Triton-interpreter fixtures passed, including a real tiny Qwen2Attention routing fixture. The 79 unchanged official Soup CPU regression tests also passed with this solution applied in a temporary test tree. This machine has Torch **2.15.0+cpu** and Triton **3.9.0**, not target Torch 2.11.0+cu128 / Triton 3.6.0. There has been **no GPU execution, compilation to SM75, whole-task run, T4 speed/peak-memory measurement, quality evaluation, useful-share measurement, queue submission, or official admission**. Score is null; admission remains UNKNOWN. GPU preflight and training are future opt-in notebook steps.

## Files and use

- `candidate.ipynb`: cleared-output, self-contained notebook embedding the exact reviewed source ZIP and runner support. Configure the existing official checkout, public-only model cache, output root and backend; explicitly set `RUN_GPU = True` only on the authorized cloud T4.
- `candidate.zip`: worker upload artifact containing only `solution/` source. It is deterministic and has no credentials.
- `source_manifest.json`: exact per-source SHA256 hashes emitted by the queue's `bundle_candidate.py`.
- `submission.template.json`: compatible queue template. **Keep the GCP baseline ID unresolved until a real same-environment baseline is registered.** No queue operation was performed. The current worker supports `soup_screen_v1`; the notebook separately runs paired validation, not a fictional paired queue profile.
- `run_prepared.py`: notebook runner. `check_kaggle_environment.py` and `prepare_asset_manifest.py` are unmodified queue support, not claimed original optimization kernels.
- `official_frozen_manifest.json`: hashes of local official harness/task/tests at commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
- `validation.json`, `cpu_interpreter.log`, `cpu_native.log`, `official_cpu_regression.log`, `static_checks.log`, `runner_cleanup.log`: local evidence only.

No packages/model/data are installed or downloaded. The offline runner verifies exact direct pins, four-CPU affinity, one visible Tesla T4, official commit/frozen file hashes, existing public data hashes and cached model revision. It extracts a **new** local copy of the official commit and only replaces that copy's `solution/`; it never edits the prepared checkout. The official paired harness (`--repeats 2`, seed 1) is unchanged and creates fresh training processes/state/output/compiler caches, times whole training commands, externally samples nvidia-smi, performs public quality evaluation and measures owned kernels after two warmup steps. Preflight uses a separate process and compiler cache. The default native timeout is 900 seconds for either backend, not the historical 79-second Kaggle screen. A GCP run never inherits a Kaggle baseline or official admission.

The runner then invokes unchanged `harness/check_rules.py`, including original regression tests. Failed, partial or timed-out runs retain logs and cannot receive a score/admission PASS. Raw logs may contain official harness summaries, but `review_summary.json` independently rejects incomplete/failed nested results and always retains score null/admission UNKNOWN. Complete paired measurements are research evidence, not the final official Kaggle judgment. The canonical nonblocking `/tmp/soup-gpu0.lock` serializes the entire paired run across all six candidate notebooks. Short-lived GPU preflight exits before a compute-process idle check; the parent runner never imports Torch or holds a CUDA context. A Linux child-subreaper tracks owned descendants across the official harness's nested sessions and kills/reaps them on timeout or interrupt before releasing the GPU lock. Normal-exit orphan leakage is rejected too. Nested-session timeout, normal-parent-exit orphan and SIGTERM cleanup were CPU-tested using pidfd-safe signaling. All lanes check `/tmp/soup-gpu0.lock.blocked` before and after acquiring the lock. This runner writes an owner-token crash marker before subprocess work; normal clean termination can clear only its own token. **Outer timeout/interruption or uncertain cleanup retains quarantine even after observed process cleanup; operator reconciliation is required, never automatic clearing.** Marker persistence/refusal was CPU-tested with isolated fixture paths. Hard-killed runner/host failures likewise require operator resource inspection before any retry. Do not run multiple GPU notebooks concurrently.

## Implementation contract

`solution/rotary.py` owns two launched kernels, `rotary_apply_f32` and `rotary_adjoint_f32`, plus an inlined sign-bit helper. No starter or ready-made kernel is copied or credited. The starter SwiGLU patch is not installed; the baseline MLP remains native.

- CUDA FP32 Q/K/coefficient guard; production shapes B<=4, T<=512, Hq/Hk=12/2, D=128; dense transposed BTHD storage, contiguous coefficients with Bpos=1 or B; default unsqueeze axis only.
- Fresh outputs with input strides, unmodified inputs, frozen supplied coefficients, independent coefficient halves and Q/K requires-grad branches. No retained Q/K activation storage claim: only coefficients are saved, and the baseline already shares their storage.
- Shared activation storage, tensor subclasses, lazy negative/conjugate input views, trainable coefficients, unsupported layout/dtype/shape/axis, and CPU calls use the captured native function. Module installation is hash-guarded against the pinned Transformers 5.16.1 Qwen2 source blob. Unknown source fails installation, not training under an unidentified patch.
- Separate FP32 products and addition (`enable_fp_fusion=False`). Unary negation flips the FP32 sign bit, not `0-x`. Backward uses the **cross-half sine** coefficient and native slice-padding zero canonicalization `Z(z)=+0 if z==0 else z`. This preserves the audited native adjoint graph rather than substituting a generic inverse rotation.
- Backward accepts arbitrary upstream strides and resolves lazy negative/conjugate views. Unused outputs keep None gradients. Coefficients use `save_for_backward` for version checks. Higher-order requests use a differentiable native PyTorch adjoint instead of an opaque Triton second derivative.
- First and every 256th route/launch event print flushed `ROTARY_EVIDENCE` lines. They survive native timeouts in the official `train.log`. A `launch_returned` event is a host launch observation, **not proof of completed CUDA execution**; completed official metrics/quality checks are still required.

The GPU preflight requires exact arithmetic (including zero signs), exhaustive signed-zero boundary cases, nonfinite classes/subnormals, realistic full shapes, unchanged masked/unmasked SDPA autocast/GQA, real Qwen2Attention route, and PTX with separate FP32 multiply/add, no FMA and no FTZ. It exports PTX for human review. GPU tests are supplied but **never run here**. NaN payload-bit identity and full-model bitwise identity are not promised. Shared-ancestry accumulation can differ even when standalone RoPE is exact; integration tests compare raw scaled gradients with declared tolerances, not unscaled gradients.

## Scientific limits and rejection criteria

The strongest open blocker is eligibility: this narrow operation may be too small to provide the required **>=5% useful original-kernel wall share**. No artificial work is added to reach that threshold. Reject if routing is absent, useful share is insufficient, fallback dominates, PTX/numerics fail, full-task quality/outputs diverge, or complete time/memory is worse beyond official limits. There is no GPU-optimality, speedup, retained-Q/K savings or measured Pareto claim.

Research inputs inspected locally: `soup_cycle/research_rope_cycle3.md`, its native routing/CPU contracts; `soup_audit_1919/qwen.json`; `soup_audit_1919/handoff.json` (HANDOFF-ROPE-SIGNEDZERO); and supplied `soup_user_findings_20261030/test_rotary.py`. They inform requirements, not runtime ownership or inherited validation. All tests in this packet are independently authored, and earlier supplied tests were not relabeled as this implementation's results.

## Reproduce local CPU checks (no GPU)

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TRITON_INTERPRET=1 \
  /workspace/shared/soup_cloud_env/venv/bin/python -m pytest solution/test_rotary.py -q
CUDA_VISIBLE_DEVICES='' /workspace/shared/soup_cloud_env/venv/bin/python -m solution.cpu_contract
python validate_artifacts.py
```

The available interpreter emits one expected NumPy invalid-multiply warning on deliberate infinities/NaNs. Interpreter execution is not compilation, target-version parity or CUDA evidence. Do not install/downgrade packages to force those checks into a target claim.
