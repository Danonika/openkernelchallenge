# Run the fused candidate on the Soup GPU service

## Current state

The fused implementation is ready for a first physical T4 correctness and training run. It has **not** demonstrated a time below 33 seconds. The earlier local evidence is preserved in `validation/summary.json`; it is CPU/interpreter and offline-compilation evidence.

`notebooks/Soup_Fusion_Service.ipynb` is the service-specific, offline notebook. It embeds 34 Python source files under `solution/` and the readiness helper. Its install cell writes the actual files under `Path(os.environ["SOUP_REPO"]) / "solution"`. `apply()` takes no arguments, installs patches, and does not start training.

All three cells from the service template are preserved exactly. The added cells verify and install source, then run physical-T4 readiness in child processes before the original measurement cell. The readiness compiler cache is temporary and removed before the measurement starts. The notebook parent does not import Torch or initialize CUDA. No package installation or network is needed inside the notebook.

The template and client were recovered through the connected GitHub repository, because the current execution environment could not verify HTTPS to the GPU IP. They are exact sources from:

- Repository: `BAITC-Hacks/hack-57faaa42-duo`.
- Service source commit: `3a4953f76288969bb91956f114eda5b7e541f104`.
- Template: `testing_queue/examples/notebook-training-template.ipynb`.
- Client: `testing_queue/notebook_client.py`.
- Template SHA-256: `bea1521d4c9d68b27591a483b19463b86ef49a0d7ea1fd8ec5e73b61001b81f8`.
- Client SHA-256: `7d7b5222b040c58af5d2f7a3d4846a7d04945809a3f6fd52d5ce73ca1312f4ff`.

These source pins do not certify which release is currently deployed. Verify the runtime endpoint before submitting from a connection where normal HTTPS verification succeeds.

## Submit one candidate

On the service website, connect with the token and upload `notebooks/Soup_Fusion_Service.ipynb` under **Run a notebook**. The token is not included in the notebook, source package, or reports.

Alternatively, use the unchanged official client. It prompts for the token without echo if `SOUP_NOTEBOOK_TOKEN` is not already set:

```bash
python tools/notebook_client.py runtime
python tools/notebook_client.py submit notebooks/Soup_Fusion_Service.ipynb --candidate fusion-cached128 --request-id 79e349aea4814d0e8cae71246bcc73b2 --wait --out fusion-cached128-results.zip
```

This is one bounded candidate, with a fixed cached-head chunk size of 128. Reuse that request ID **only with identical notebook bytes and identical timeout settings** when recovering a lost submission response. A new candidate or changed source needs a new request ID. Record the job ID printed by the client; status polling does not create another job. If that request ID already belongs to a different submission, do not silently choose a new ID before checking existing jobs.

Normal HTTPS certificate verification is required. Do not disable it, send the token to an alternate host, put it in a URL, or embed it in a notebook. The source service configuration identifies the IP origin as its only configured public endpoint.

## What runs and what is exported

1. The original setup cell selects the prepared `SOUP_REPO` and `SOUP_RESULTS` paths.
2. The added installer verifies the payload, writes source, checks the exact solution hash and all 15 protected starter files, and writes `fusion_source_manifest.json`.
3. The added readiness cell checks the installed stack and physical T4, runs the enabled candidate checks (including the combined real-NF4/Soup/TRL/GradScaler fixture and both NF4 dispatch paths), and runs all 79 unchanged Soup regressions with the patch installed. Missing or skipped required GPU cases fail readiness. Full logs and JUnit reports stay under `/kaggle/working/fusion-readiness`. A `finally` block exports small JSON/JSONL/log/YAML files and an export manifest to `SOUP_RESULTS/fusion_readiness`, including on readiness failure. Added readiness exports are limited to 20 files and 8 MiB in total, with each file smaller than 4 MiB. JUnit XML remains in working storage; its parsed counts and required-case outcomes appear in `readiness.json`.
4. The **unchanged** template invokes `harness/run_task.py --only solution` and exports its small JSON/JSONL/log/YAML reports. Large weights, adapters and compiler caches are not added to downloadable reports.

A readiness failure stops before measured training and still leaves its diagnostic files in `SOUP_RESULTS`. Before training, a synchronous IPython `post_run_cell` hook is registered to copy the same small report types if a training or reporting cell raises. It does no work while the measured subprocess is running. The original export cell remains unchanged. Inspect both the job's terminal state and `result.error`, not just time/memory/loss tags. Service termination can retain only the partial evidence already collected; a completed training result is never inferred from an incomplete run.

The template contains `--timeout 900`, but the service independently enforces an **80-second cutoff for each measured training command**. This notebook preserves that command and does not extend or bypass the server policy. Whole-notebook time also includes readiness, setup, warm-up, evaluation and reporting.

The source service documentation records a saved compatible public development reference of 71.85 s, 14,345 MiB and 2.04968 evaluation loss. The running service must confirm the active reference and its compatibility. Do not substitute the user's 77 s planning reference for the result's actual baseline, and do not treat saved-baseline development feedback as paired Kaggle/jury admission.

## Evidence to inspect next

Require a completed job, no nested error, all 64 training steps, a completed adapter evaluation, real metrics/Rewind, actual kernel launches, and the full command time/peak. Check the original-kernel share separately: the credited starter SwiGLU does not count. Keep all failed evidence.

If cached128 passes but misses the target, the first controlled follow-up is cached256. Select it in source **before** building a new immutable notebook, then submit it as a distinct candidate. No profile selection or tuning occurs inside measured training. Further changes should follow measured evidence.

The final competition entry still requires the platform repository, a public saved Kaggle notebook pinned to the final source, paired baseline/solution measurements, all automatic checks and jury review. The service notebook is development preparation, not the final submission.

## Rebuild and verify

```bash
python tools/build_service_notebook.py
```

The builder refuses an unreviewed template hash and only embeds ordinary Python source. `validation/service_preparation.json` records the filesystem-only install check, protected-file equality, entry-point inspection, and preserved template cells. It records **zero created jobs** and no physical GPU execution.

## Current connection blocker

Verified urllib/curl requests and the cloud browser reject `https://35.200.1.242/` with a certificate-name mismatch. The certificate presented through this workspace's proxy does not contain the GPU IP in its Subject Alternative Names. This is an observation of this workspace's connection, not proof that the service is down or that its origin certificate is wrong. No token was sent and no submission response or job ID exists. A trusted reachable connection must provide the first output ZIP before performance tuning can proceed.
