# Soup candidate 1: RMSNorm, entropy, and MLP activation storage

## Current status

The `solution/` implementation and development checks are ready for a first GPU trial. **No GPU training run has been submitted or measured.** This is a development candidate, not a claim of official admission or a final jury submission.

The requested service at `https://35.200.1.242/` cannot currently be reached through a verified HTTPS connection from this environment. The Python download fails with `CERTIFICATE_VERIFY_FAILED` / IP address mismatch, and the browser reports `ERR_CERT_COMMON_NAME_INVALID`. The presented certificate in the local connection belongs to the environment's proxy and does not cover the service IP, so this evidence does not establish that the origin server's certificate itself is wrong. No access token was sent. Certificate verification remained enabled.

As a result, the exact service notebook template and official API client have not been downloaded. A builder is provided to insert the solution into that exact template once it is available. It does not invent harness cells or substitute the public Kaggle notebook for the service template.

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

`solution/metrics.py` processes the entire vocabulary. Each tile stores a maximum, exponential sum, weighted displacement sum, and nonfinite flag. A second kernel merges those statistics to produce per-token Shannon entropy. It avoids materializing vocabulary-sized log-softmax and probability outputs for the metric.

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

Add `--check-accuracy` only when investigating the optional copy removal. It remains disabled in the submitted patch hook regardless of that probe flag.

The probe runner's parent uses only the Python standard library. Each child gets fresh Triton and Inductor cache directories, and those directories are removed after the child exits. The probes do not run concurrently with measured training and do not alter harness allocator or measurement settings. Do not reuse their compiler cache for a measured run.

The RMSNorm probe covers the maximum batch shape, several input scales, backward, autocast, installation idempotency, and native fallbacks. The MLP probe compares to the unchanged starter exactly and exercises both a small batch and `[4,512,1536]`, including the bitsandbytes dispatch boundary. The entropy probe covers the full vocabulary, full batch, uniform and peaked distributions, nonfinite values, unsupported layouts/dtypes, and differentiable fallback.

## Build the real service notebook

Once the official `notebook-training-template.ipynb` has been downloaded through verified HTTPS or supplied directly, inspect its cells:

```bash
python dev/build_service_notebook.py \
  --template notebook-training-template.ipynb --list-cells
```

Choose the insertion index **after repository setup and any starter-solution creation, and before the first solution test or training invocation**. Then build:

```bash
python dev/build_service_notebook.py \
  --template notebook-training-template.ipynb \
  --insert-before CELL_INDEX \
  --out candidate.ipynb
```

Replace `CELL_INDEX` with the inspected zero-based index. The builder inserts one cell that writes the package to `Path(os.environ['SOUP_REPO']) / 'solution'`. It embeds the source files and verifies their SHA-256 digests. Removing that single cell must reconstruct the original template object exactly, including every harness and export cell.

The installer does not invoke `apply()` inside the notebook kernel. The preserved harness invokes it in its own training process. All implementation dependencies needed by that hook are under `solution/`.

Use the downloaded official API client with an environment-provided token or its prompt:

```bash
python notebook_client.py submit candidate.ipynb \
  --candidate duo-rms-entropy-half-v1 \
  --wait --out duo-rms-entropy-half-v1-results.zip
```

No token belongs in the notebook or repository. Keep HTTPS verification enabled. A larger notebook timeout does not extend the service's 80-second measured training cutoff. After submission, inspect the final job state, `result.error`, all metrics, and exported reports. Passing time/memory/loss badges alone do not establish notebook completion or jury admission.

## Files and provenance

Only `solution/` is needed by the jury hook. `dev/` contains validation and notebook construction tools, and `reports/` records local checks. `CREDITS.md` identifies the official example and the reference formulas. The full official template, API client, actual GPU reports, and a final Kaggle notebook are not present yet because the service connection is blocked.

The earlier research's n-gram/model-architecture ideas are not applied: this challenge fixes the model and task. This candidate transfers the useful general lesson—reduce intermediate storage and memory traffic—into the permitted implementation layer.
