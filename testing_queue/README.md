# Soup testing queue

For automatic intake from Google Sheets, use [To Run](https://example.invalid/redacted): add a notebook, set a unique Request ID, and check Ready. The VM mirrors the site's result metrics, individual training runs and full job JSON directly to **VM Results**, **VM Runs**, **VM Details** and **VM Queue** in the same spreadsheet. [SHEETS_QUEUE.md](SHEETS_QUEUE.md) covers submission, result retrieval and the one-time Google writer authorization required for automatic updates.

The deployed service accepts authenticated notebook submissions at [https://35.200.1.242/](https://35.200.1.242/) and [https://35-200-1-242.sslip.io/](https://35-200-1-242.sslip.io/) and runs them automatically on the VM's Tesla T4. Both addresses serve the same queue and accept the same credentials without redirecting between addresses. Start with [NOTEBOOK_RUNNER.md](NOTEBOOK_RUNNER.md) for the current notebook workflow, API, limits and results.

This repository also retains the older reviewed solution-bundle lane. Its approval and source-allowlist requirements apply to bundle jobs, not to automatic notebook jobs.

## Choose a lane

| Lane | Input and access | Execution and results |
|---|---|---|
| Automatic notebooks | Python notebook and submitter token; no separate review or source allowlist | Isolated GPU execution; training time, GPU peak, eval loss, baseline comparison, logs and output ZIP |
| Legacy reviewed bundles | Exact reviewed solution ZIP and worker allowlist | Original bundle queue and runner contract, with its own baseline and evidence |

Both lanes share the host GPU lock. The API has no GPU or Docker privileges. Notebook containers receive no service credentials or network access.

## Notebook workflow

1. Open the dashboard and connect with a submitter token, or download [notebook_client.py](https://35-200-1-242.sslip.io/notebook_client.py).
2. Start with the [training template](https://35-200-1-242.sslip.io/notebook-training-template.ipynb) or [GPU smoke notebook](https://35-200-1-242.sslip.io/notebook-gpu-smoke.ipynb). Put candidate changes in the copied repository's `solution/` directory.
3. Submit, inspect the online experiment table, and download the output ZIP. The default training template runs only the solution and uses a compatible saved public baseline.
4. Use **Re-run** on a finished job to create a new job from its saved notebook and timeout settings. The original job remains available.

The client reads `SOUP_NOTEBOOK_TOKEN` from the agent's secret environment or prompts without echo. Operator role credentials live in root-owned files under `/etc/soup-queue/`; never publish their values in notebooks, source, logs or GitHub.

```bash
python notebook_client.py submit candidate.ipynb --candidate experiment-001 --wait --out results.zip
python notebook_client.py status n_JOB_ID
python notebook_client.py download n_JOB_ID --out another-results.zip
python notebook_client.py cancel n_JOB_ID
```

The client defaults to the IP address. To use the hostname, put `--url` before the subcommand:

```bash
python notebook_client.py --url https://35-200-1-242.sslip.io submit candidate.ipynb --candidate experiment-001 --wait --out results.zip
```

Keep HTTPS certificate verification enabled. Each address has its own certificate. The hostname resolves directly to the same VM; it is not a proxy and cannot guarantee access from an execution environment that blocks outbound connections to the VM.

A temporary Cloudflare HTTPS relay is also available for connectivity testing. See [HTTPS_RELAY.md](HTTPS_RELAY.md) for its current URL, a token-free reachability check, client commands and restart limitations.

## Runtime and comparisons

The fixed compute profile is one Tesla T4 on GPU 0, four usable logical CPUs, a 30 GiB container RAM ceiling with no swap, Python 3.13.15, PyTorch 2.11.0+cu128 and Triton 3.6.0. Each standard harness training command has an **80-second limit**, including compilation and adapter saving; warm-up and evaluation have separate notebook deadlines.

Speedup is baseline training seconds divided by solution training seconds. The selected public baseline is **71.85 s / 14,345 MiB / 2.04968 eval loss**, reused only when its runtime, harness, task and data bindings match. Paired runs remain supported. Failures retain available partial measurements; unfinished evaluation is shown as **Not evaluated**. A partial run cannot count as beating the baseline.

Storage and shared memory have separate limits from GPU memory, RAM and exported results. [NOTEBOOK_RUNNER.md](NOTEBOOK_RUNNER.md#storage-and-shared-memory) documents the v8 profile and how to verify activation. Its [storage capability probe](https://35-200-1-242.sslip.io/notebook-storage-probe.ipynb) is a diagnostic, not a training benchmark. The 80-second training cap remains unchanged.

These are development checks on public data. Completion is not official admission; admission stays `UNKNOWN` and score stays `null`. The jury's [pinned official rules](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#rules), hidden-data checks and original-kernel participation requirement still apply.

## Source and verification

- `notebook_queue.py`, `notebook_worker.py`, `notebook_runner.py`: authenticated automatic queue and isolated execution.
- `training_run_limit.py`, `notebook_failfast.py`, `notebook_reference.py`, `notebook_metrics.py`: limits, typed failures, compatible references and measured results.
- `notebook_client.py`, `web/notebooks.*`: agent client and dashboard.
- `build_release.py`, `install.py`, `deploy/`: source packaging, manifest verification and deployment templates. Source changes alone do not deploy a release.
- `candidates/`: historical submitted notebooks, source and evidence. These frozen archives are not blanket approval to rerun a candidate or claims of current correctness.

For a packaged release, `python3 install.py --verify-only` checks its manifest. For a source checkout, build a fresh release rather than treating an old manifest as verification of edited files. Notebook regression tests include `test_notebooks.py`, `test_training_run_limit.py`, `test_notebook_failfast.py`, `test_notebook_metrics.py`, `test_api_unit.py` and `test_notebook_browser.py`; browser tests require the configured Playwright environment. Tests and live acceptance evidence have different scopes.

## Legacy bundle documentation

[FULL_INSTRUCTIONS.md](FULL_INSTRUCTIONS.md), [HTTP_API.md](HTTP_API.md), [RUNNER_ADAPTER.md](RUNNER_ADAPTER.md) and [DEPLOYMENT_ADAPTER.md](DEPLOYMENT_ADAPTER.md) describe the original reviewed-bundle workflow and deployment preparation. [DELIVERY_REPORT.md](DELIVERY_REPORT.md) records that original delivery's tested scope. Their reviewed-only admission policy and historical deployment status do not describe the current automatic notebook lane. Use [NOTEBOOK_RUNNER.md](NOTEBOOK_RUNNER.md) for notebook endpoints and operating behavior.
