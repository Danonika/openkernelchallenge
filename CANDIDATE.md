# Soup candidate 1: RMSNorm, entropy, and MLP activation storage

## Current status

The `solution/` implementation, development checks, and complete service notebook are ready for a first GPU trial. **No GPU training run has been submitted or measured.** This is a development candidate, not a claim of official admission or a final jury submission.

The requested service at `https://35.200.1.242/` cannot currently be reached through a verified HTTPS connection from this environment. The Python download fails with `CERTIFICATE_VERIFY_FAILED` / IP address mismatch, and the browser reports `ERR_CERT_COMMON_NAME_INVALID`. The presented certificate in the local connection belongs to the environment's proxy and does not cover the service IP, so this evidence does not establish that the origin server's certificate itself is wrong. No access token was sent. Certificate verification remained enabled.

On 31 October 2026 the user supplied the missing service template and official API client in a connection bundle. Both attached files match the byte counts and SHA-256 hashes in that bundle's manifest. The template is now incorporated into `notebooks/soup_rms_entropy_half_v1.ipynb`: one solution installer is inserted before training, and all three supplied template cells remain unchanged. A separate `notebooks/soup_preflight_v1.ipynb` performs diagnostics before a measured submission.

The bundle's OpenSSL transport was tried with a fresh public template GET, certificate-chain verification, and IP verification enabled. It fails in this environment with `HTTP CONNECT failed, reason=426 Upgrade Required`, before TLS authentication or token transmission. The attached files unblock notebook preparation; they do not establish a working authenticated connection from this environment.

## What this candidate implements

| Component | Implementation | Numerical and behavior boundary | Contribution status |
| --- | --- | --- | --- |
| RMSNorm | Original row-wise Triton forward and input-gradient kernels | Contiguous CUDA FP32, width 1536, frozen FP32 gamma; native fallback elsewhere | Original useful kernels; runtime share unmeasured |
| Entropy | Original full-vocabulary tile reduction and merge | FP32 no-grad metric calls; original TRL masks, reductions, and logs remain | Original useful kernels; runtime share unmeasured |
| MLP saved activations | Retain half-origin gate/up values in FP16, using the unchanged supplied SwiGLU arithmetic | Original NF4 projections, FP32 activation output, and separate FP32 gradient joins remain | Supplied example kernels are explicitly excluded from original-kernel credit |
| Accuracy copy | Guarded removal of one redundant contiguous copy is available for investigation | Disabled in `install_patches()` | No kernel contribution claimed |

The patch hook is exactly the requested no-argument entry point:

```python
# solution/__init__.py
def apply():
    from .my_kernels import install_patches
    install_patches()
```

It installs patches before model and trainer construction. It does not construct a model, start training, read task rows, change the task configuration, or run development probes.

### RMSNorm

`solution/rmsnorm.py` computes the reference normalization formula with a row reduction and saves the actual forward reciprocal standard deviation for backward. The input-gradient expression keeps separate FP32 stages and disables fused multiply-add contraction for these kernels. Frozen gamma requires no gamma gradient. The patch falls back for trainable gamma, other dtypes or widths, noncontiguous inputs, CPU inputs, and empty tensors.

Different parallel reduction trees can produce different FP32 rounding. The implementation therefore does not claim bitwise identity with native CUDA. Its probe compares forward values and input gradients, including autocast and noncontiguous upstream gradients. Full task evaluation remains the deciding correctness check. Higher-order derivatives are outside the optimized path's supported use.

### Entropy

`solution/metrics.py` processes the entire vocabulary. Each tile stores a maximum, exponential sum, weighted displacement sum, and nonfinite flag. A second kernel merges those statistics to produce per-token Shannon entropy. It avoids materializing vocabulary-sized log-softmax and probability outputs for the metric. The row count is a runtime scalar used only for temporary-plane addressing, so different padded batch lengths do not each require their own constexpr kernel specialization.

For the maximum prepared shape `[4, 512, 151936]`, with tiles of 4096 logits, the statistics tensor contains `4 × 2048 × 38` FP32 values: 1,245,184 bytes, or 1.1875 MiB. This is a buffer-size calculation, not a measured whole-run memory result.

Both TRL entropy bindings are patched, including the name imported into `sft_trainer`. The wrapper falls back when gradients are enabled and its input requires gradients. This distinction matters because the ordinary metric call runs under `no_grad()` while its logits can still have `requires_grad=True`. The kernels preserve the reference entropy's NaN result for nonfinite logits. Metric masks, token counts, gathering, aggregation, logging, loss, accuracy, and rewind behavior remain in the original caller.

### MLP activation storage

In the pinned bitsandbytes implementation, the guarded bias-free frozen NF4 projections compute with FP16 and widen their results to the FP32 input dtype. Those widened values can be stored back in FP16 without losing effective values. `solution/mlp_storage.py` uses that representation property; it does not compress arbitrary FP32 activations.

The two original projection graphs remain separate. The activation output, incoming activation gradient, and returned gate/up gradients remain FP32. This preserves the original projection-input casts and the location of gradient accumulation. The unchanged starter kernels widen half operands in registers, so backward does not need full restored FP32 gate/up tensors.

For a full `[4, 512, 8960]` activation, retaining two operands in FP16 instead of FP32 saves 70 MiB per participating MLP layer in this particular saved-state pair. It is not valid to multiply this number by the layer count and call it a measured peak reduction: lifetimes, frozen prefixes, temporary buffers, and non-training phases affect the actual peak.

Eligibility requires the real Qwen2 MLP dimensions, frozen bias-free NF4 projections with FP16 compute, and the pinned SiLU activation. Output-changing hooks or instance overrides that can invalidate the representation argument use the native fallback. No tensor-value scan or GPU synchronization is added to the layer guard.

## What has and has not been validated

Local checks cover Python syntax, the zero-argument entry point, source imports, starter-file provenance, installer behavior, and notebook construction. The local machine has Python 3.12 and no Torch, Triton, or CUDA runtime. These checks cannot establish device correctness.

The following remain pending:

- Actual Triton compilation on the service's T4 / Triton 3.6.0 stack.
- GPU forward and backward comparisons for all three components.
- The official Soup regression tests in the supplied runtime.
- Complete measured training, evaluation, required metrics, and report export.
- Time and sampled GPU peak relative to baseline.
- Evaluation loss within the required tolerance.
- At least 5% original useful-kernel time from step 3 onward in every measured solution run.
- Paired jury runs on hidden rows and another seed.

Four original JIT definitions are present, but definition count does not prove runtime eligibility. RMSNorm and entropy may still fall below the required 5% share after fusion. The supplied SwiGLU kernels, their wrappers, and compiler-generated work cannot be counted to fill that gap. If the measured share is too low, the next candidate needs another useful original operation, followed by fresh validation.

## Run the development checks

From this candidate directory:

```bash
python dev/static_check.py --out reports/static_check.json
python -m unittest discover -s dev -p 'test_*.py' -v
```

On the prepared GPU runtime, run the synthetic probes as separate processes:

```bash
python dev/run_gpu_probes.py --out /kaggle/working/probe_reports
```

For the notebook service, `notebooks/soup_preflight_v1.ipynb` embeds the matching solution and development scripts, runs the official regression suite with `SOUP_APPLY_SOLUTION=1`, and runs the GPU probes in isolated child processes. It writes diagnostic reports to `SOUP_RESULTS` before reporting failure. This is a separate diagnostic job; it does not train, benchmark, or establish admission. The four-cell measured notebook follows it after those checks pass.

Add `--check-accuracy` only when investigating the optional copy removal. It remains disabled in the submitted patch hook regardless of that probe flag.

The probe runner's parent uses only the Python standard library. Each child gets fresh Triton and Inductor cache directories, and those directories are removed after the child exits. The probes do not run concurrently with measured training and do not alter harness allocator or measurement settings. Do not reuse their compiler cache for a measured run.

The RMSNorm probe covers the maximum batch shape, several input scales, backward, autocast, installation idempotency, and native fallbacks. The MLP probe compares to the unchanged starter exactly and exercises both a small batch and `[4,512,1536]`, including the bitsandbytes dispatch boundary. The entropy probe covers the full vocabulary, full batch, uniform and peaked distributions, nonfinite values, unsupported layouts/dtypes, and differentiable fallback.

## Build the real service notebook

The complete measured notebook is already built at `notebooks/soup_rms_entropy_half_v1.ipynb`. It has four code cells: original setup, our source installer, original training, and original report export. Every cell compiles, outputs are empty, and removing the inserted installer reproduces the supplied template JSON exactly.

The original training command contains `--timeout 900`. It is preserved exactly. The service's independent 80-second training cutoff still applies; this argument does not extend that policy. The attached bundle's other candidate and its timeout edit are not part of this six-file candidate.

To inspect or rebuild from the retained supplied template:

```bash
python dev/build_service_notebook.py \
  --template service/notebook-training-template.ipynb --list-cells
```

The correct insertion index for this template is **1**, after repository setup and before training. For a new output file, run:

```bash
python dev/build_service_notebook.py \
  --template service/notebook-training-template.ipynb \
  --insert-before 1 \
  --out notebooks/candidate_rebuilt.ipynb
```

The builder inserts one cell that writes the package to `Path(os.environ['SOUP_REPO']) / 'solution'`. It embeds the source files and verifies their SHA-256 digests. Removing that single cell must reconstruct the original template object exactly, including every harness and export cell.

The installer does not invoke `apply()` inside the notebook kernel. The preserved harness invokes it in its own training process. All implementation dependencies needed by that hook are under `solution/`.

Use the downloaded official API client with an environment-provided token or its prompt:

```bash
python service/notebook_client.py submit notebooks/soup_rms_entropy_half_v1.ipynb \
  --candidate duo-rms-entropy-half-v1 \
  --wait --out duo-rms-entropy-half-v1-results.zip
```

No token belongs in the notebook or repository. Keep HTTPS verification enabled. A larger notebook timeout does not extend the service's 80-second measured training cutoff. After submission, inspect the final job state, `result.error`, all metrics, and exported reports. Passing time/memory/loss badges alone do not establish notebook completion or jury admission.

## Files and provenance

Only `solution/` is needed by the jury hook. `dev/` contains validation and notebook construction tools, and `reports/` records local checks. `service/` retains the supplied template and client; `notebooks/` contains the prepared measured and diagnostic notebooks. `CREDITS.md` identifies the official example and the reference formulas. Actual GPU reports and a validated final Kaggle submission remain pending verified service access and execution.

The earlier research's n-gram/model-architecture ideas are not applied: this challenge fixes the model and task. This candidate transfers the useful general lesson—reduce intermediate storage and memory traffic—into the permitted implementation layer.
