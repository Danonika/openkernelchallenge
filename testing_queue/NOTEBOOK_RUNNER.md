# Automatic GPU notebook service

Authenticated submitters can upload a Python notebook for automatic execution without a separate source review or allowlist. The legacy reviewed solution-bundle lane retains its own policy. Both workers share the host GPU lock, so only one job uses GPU 0 at a time.

The dashboard and API are available at both [https://35.200.1.242/](https://35.200.1.242/) and [https://35-200-1-242.sslip.io/](https://35-200-1-242.sslip.io/). Both addresses serve the same queue, jobs and downloads and accept the same credentials. Neither address redirects to the other. The API and persistent worker start on VM boot; an SSH session is not needed for a job to finish.

**Release v8:** the storage and shared-memory profile below is deployed on the VM. Inspect `GET /v1/notebook-runtime` for the active configuration.

## Agent quick start

Download [notebook_client.py](https://35-200-1-242.sslip.io/notebook_client.py) and the [training template](https://35-200-1-242.sslip.io/notebook-training-template.ipynb) or [GPU smoke notebook](https://35-200-1-242.sslip.io/notebook-gpu-smoke.ipynb).

The operator reads the required role credential from its root-owned file under `/etc/soup-queue/`, such as `submitter.env`, and supplies it through the agent's secret/environment mechanism. Do not publish its value or put it in a notebook. The client accepts `SOUP_NOTEBOOK_TOKEN` or prompts without echo. Submitters can submit, inspect, download, cancel and rerun notebook jobs, but cannot act as workers or reviewers or obtain SSH access. These are shared team credentials: authorized readers can see team jobs and results.

```bash
python notebook_client.py submit candidate.ipynb --candidate experiment-001 --wait --out results.zip
python notebook_client.py status n_JOB_ID
python notebook_client.py download n_JOB_ID --out another-results.zip
python notebook_client.py cancel n_JOB_ID
```

The client defaults to the IP address. To use the hostname, place `--url` before the subcommand:

```bash
python notebook_client.py --url https://35-200-1-242.sslip.io submit candidate.ipynb --candidate experiment-001 --wait --out results.zip
python notebook_client.py --url https://35-200-1-242.sslip.io status n_JOB_ID
```

Downloads and API paths are identical on both addresses. The hostname resolves directly to `35.200.1.242`; it does not supply a proxy route. It may help environments that require a hostname for HTTPS access, but an environment that blocks outbound connections to this VM still needs its own approved network route. Keep certificate verification enabled and never put credentials in the URL.

`--request-id` makes submission retries idempotent. Reusing an ID with different content or settings fails. Polling does not repeat execution. The client validates HTTPS and archive SHA-256, refuses redirects and refuses to overwrite an existing output file.

**Re-run** on a terminal job creates a new job from the saved original notebook, candidate name, whole-job timeout and cell timeout. It preserves the original job and requires submit permission. It uses the currently deployed runtime; it does not restore an obsolete container image.

## Notebook environment

| Resource | Configuration |
|---|---|
| GPU | GPU 0 only; Tesla T4, 15,360 MiB |
| Driver | NVIDIA 580.178.04 |
| CPU | Four usable logical CPUs, IDs 0Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р В РІР‚С™Р РЋРЎв„ў3 |
| RAM | 30 GiB container ceiling, no swap |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128; CUDA 12.8 is its build version |
| Triton / NumPy | 3.6.0 / 2.3.5 |
| Notebook execution | nbclient 0.10.4, nbformat 5.11.1, ipykernel 7.4.0 |

The VM has about 29.37 GiB of physical RAM shared with the OS and services; the container ceiling does not guarantee 30 GiB of free physical memory. Full resolved package pins and the immutable image ID are recorded with deployment evidence. The original bundle runner's 26 GiB baseline belongs to a different cohort.

Each job starts with fresh writable `/kaggle/working`. The public challenge is copied to `/kaggle/working/repo` (`SOUP_REPO`). Public task data is under `/kaggle/input/soup-data`. The challenge is pinned to commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`; the cached model is Qwen/Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

Python cells, IPython magics and shell cells execute sequentially in a fixed Python kernel, stopping on the first detected failure. Saved outputs, execution counts, widget metadata and skip/expected-error tags are removed before execution. Original and normalized notebook hashes are retained. Jobs do not share an interactive kernel.

The runtime is preinstalled and containers have no network. Use the prepared repository, model and data instead of package-install, clone or download cells. Write candidate changes into the copied `solution/` directory. Additional packages or assets need operator preparation. Containers run as UID 10001 with dropped capabilities, no-new-privileges, private IPC, PID limits and a read-only system filesystem. No credentials, Docker socket or private host directories are mounted.

## Storage and shared memory

The v8 profile removes the old small temporary-filesystem and shared-memory ceilings without changing the CPU, GPU, RAM or training-time limits.

| Path or limit | v8 behavior |
|---|---|
| `/kaggle/working` | Writable job workspace, including copied repository and training outputs |
| `/output/tmp` | Writable per-job temporary directory; `TMPDIR=/output/tmp` |
| `/tmp` | Bind-mounted to the same per-job directory as `/output/tmp`; no separate 512 MiB tmpfs ceiling |
| Compiler caches | Defaults under `/output/tmp/triton` and `/output/tmp/inductor`; the harness may use fresh per-run cache directories in `/kaggle/working`. Compilation and autotuning count in measured training time |
| `/dev/shm` | Private shared-memory filesystem with a maximum of 30 GiB |
| Worker disk pool | Expanded from 24 GiB to 38.49 GiB on the existing disk; shared by retained workspaces and cached assets/model files |
| Downloaded output ZIP | Separate 64 MiB / 128-file export bound; not the working-storage limit |

The 38.49 GiB disk pool is finite and is **not 38.49 GiB reserved for every job**. Retained jobs, compiler caches, temporary files and cached model assets consume that shared space. Available capacity can therefore be lower at submission time. New work pauses when free space is low; retained user results are not automatically deleted.

The `/dev/shm` maximum does not allocate 30 GiB in advance and does not add RAM. Pages written there count toward the same existing 30 GiB container memory budget as CPU tensors, offloaded activations, Python processes and other charged memory. A job can exhaust RAM before filling shared memory. Prefer disk-backed writable storage for large temporary files when RAM-backed sharing is unnecessary.

These settings remove incidental 512 MiB `/tmp` and 128 MiB `/dev/shm` restrictions; they do not make storage or memory unlimited. Test the actual candidate's peak RAM, temporary disk usage and worker count.

The runtime API reports `workspace_pool_gib:38.49`, `shared_memory_limit_gib:30` and `temporary_storage:"per-job disk directory shared by /tmp and /output/tmp"`. The [storage capability probe](https://35-200-1-242.sslip.io/notebook-storage-probe.ipynb), also in `examples/notebook-storage-probe.ipynb`, exercises a 2 GiB write through literal `/tmp`, 256 MiB of touched shared memory accessed by a child process, custom autograd with a CPU-saved activation, and fresh GPU `torch.compile`. It is a server capability diagnostic, not a training benchmark or proof that every larger workload fits.

## Candidate methods and official rules

The pinned [official README](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/README.md#rules) permits custom Triton kernels, autograd functions and patches, recomputation or offloading implemented in the candidate's own code, and `torch.compile`, subject to the fixed task and output requirements.

That permission does not authorize changing `harness/`, `task/`, `tests/`, the task's optimizer/model/data/batches/steps, or its Soup feature switches. Implementing recomputation or offloading yourself is distinct from enabling prohibited task-configuration switches. Runtime installation of packages, prebuilt compilation caches and methods that evade measurement remain disallowed.

Compiler-generated `torch.compile` kernels do not count toward the required original-kernel share. Useful candidate-defined Triton work must still meet the official 5% requirement under its prescribed timing method. All compilation, autotuning, recomputation and offloading costs inside the measured command count toward its runtime. Development checks do not replace the jury's tests or code review.

## Deadlines and early failure

Each standard harness training command has an **38.49-second wall-clock limit**, from process start through saved adapter, including compilation. `base_1`, `solution_1` and each repeat receive separate deadlines. A 38.49-second command passes this time check; a command exceeding 38.49 seconds fails. Warm-up and evaluation are outside this training cap but remain subject to notebook deadlines.

A read-only startup adapter wraps the official harness's `run_logged` without changing official source files. The worker also watches recognizable `_train.py` processes from the host; host enforcement can lag by its polling interval. A failure marker is honored even if notebook code catches the Python exception or exits zero.

The notebook stops on the first detected failure. `result.error` contains a typed code, message, phase and available run, measured value, limit, baseline and unit. Codes include `TIME_LIMIT_EXCEEDED`, `MEMORY_LIMIT_EXCEEDED`, `EVAL_LOSS_EXCEEDED`, `INVALID_EVAL_RESULT`, `BASELINE_REQUIRED`, `EXECUTION_ERROR` and `STORAGE_LIMIT_EXCEEDED`. Exit 137 alone is not proof of OOM; Docker OOM evidence or a recognized allocation error is required.

Other submission bounds are 8 MiB / 512 cells, at most 32 queued/running jobs, a default whole-notebook timeout of 3,600 seconds (maximum 14,400), and a default cell timeout of 1,800 seconds or the smaller whole-notebook timeout.

## Saved baseline and measurements

The default template runs only the solution. Its operator-selected public baseline uses performance measurements from job `n_89db8d59aa5f4666b8f9c09f906481a2`. Eval loss remains from the original job `n_6e7375b093154fca990312c240e88402`:

| Training wall time | GPU peak | Public eval loss |
|---|---|---|
| 38.49 s | 2,261 MiB | 2.04968 (original reference) |

Reuse requires matching runtime image, official harness hashes, task settings including seed and training-data hash, and evaluation-data hash. A changed task or runtime needs a compatible reference. Paired runs remain supported; rerunning the baseline is not required for each compatible solution submission.

- **Speedup:** baseline training seconds / solution training seconds.
- **GPU memory saved:** `100 Р В Р’В Р Р†Р вЂљРЎС™Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ (1 Р В Р’В Р В РІР‚В Р В Р вЂ Р Р†Р вЂљРЎв„ўР вЂ™Р’В¬Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС› solution_peak / baseline_peak)`.
- **Relative eval-loss change:** `100 Р В Р’В Р Р†Р вЂљРЎС™Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ (solution_loss / baseline_loss Р В Р’В Р В РІР‚В Р В Р вЂ Р Р†Р вЂљРЎв„ўР вЂ™Р’В¬Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС› 1)`.

The public screen rejects GPU peak above the hard limit of 2,261 MiB and completed eval loss above the original baseline * 1.005. The GPU limit covers host-sampled whole-notebook use as well as training. Missing or non-finite eval results fail validation; an unmatched reference does not produce a fabricated comparison. The 38.49-second cap remains a separate owner-set development limit.

Training elapsed time and sampled GPU peak are checkpointed. Completed eval loss is recorded before a quality failure is raised. Timeout, OOM, cancellation and process-error reports retain measurements available at termination. Host GPU sampling every 100 ms also records a whole-notebook peak when the kernel dies; this is labelled separately from a training-run peak.

Partial training time and peak are incomplete observations. Any partial speedup is an upper bound and cannot count as a win. If evaluation never completed, the table shows **Not evaluated**, not zero. A genuine sampled zero GPU peak is possible for a CPU-only or synthetic validation notebook.

The dashboard refreshes every five seconds and supports search, filters, sorting, expandable comparisons, output download, cancellation and Re-run. A win requires a completed compatible comparison with faster training, no extra GPU peak, passing available public checks and unchanged packages. Faster-but-higher-memory and lower-memory-but-slower runs are separate outcomes. Lifecycle failure overrides a claimed win.

These are public development measurements, not official hidden-data results or a contest score. Official judging still runs its own paired measurements and checks. Notebook-provided reports remain untrusted; automatic admission is `UNKNOWN` and score is `null`.

## Results, cleanup and retention

The output ZIP contains `executed.ipynb`, `execution.log`, host `runner.json` and selected regular files from `/kaggle/working/results` (`SOUP_RESULTS`). It also preserves available training/evaluation telemetry and bounded partial harness logs when export cells are skipped. The combined archive is limited to 64 MiB / 128 files; the executed notebook is limited to 32 MiB. Collection warnings identify skipped files. Large adapters and checkpoints remain in the retained workspace rather than the download.

The collector rejects symlinks, symlink ancestors and FIFOs. The API serves results as attachments and does not execute or render uploaded HTML. Metrics are parsed from bounded reports and cached against immutable archive/reference hashes, allowing existing evidence to be displayed without GPU reruns.

On timeout or cancellation the worker kills and removes the container, verifying cleanup before releasing GPU 0. Completed-cell checkpoints preserve partial notebooks where possible. After worker interruption, recovered running jobs become `interrupted` and are not silently retried. Uncertain cleanup quarantines the GPU lane for operator reconciliation.

`completed` means the notebook process exited zero and no screening failure was observed. It is not a certificate of scientific correctness.

## HTTP contract

Non-static routes require `Authorization: Bearer TOKEN`; credentials are never accepted in URLs. Nginx terminates verified HTTPS using separate IP and hostname certificates and forwards to the API on `127.0.0.1:8765`. HTTP port 80 serves certificate challenges only. Each dashboard uses API paths on its own origin; no CORS is enabled. The hostname virtual host maps only its exact browser Origin to the API's trusted IP origin, preserving other Origin values so the API rejects untrusted origins.

| Method and path | Purpose |
|---|---|
| `GET /v1/notebook-runtime` | Active runtime and limits |
| `POST /v1/notebooks` | Submit JSON containing `candidate_id`, `request_id`, `notebook_base64`, `contains_secrets:false`, optional `timeout_s` and `cell_timeout_s` |
| `GET /v1/notebooks` | Latest 100 jobs, including metrics |
| `GET /v1/notebooks/{id}` | Source hashes, lifecycle result, metrics and output hash |
| `GET /v1/notebooks/{id}/outputs` | ZIP with `X-Artifact-SHA256` |
| `POST /v1/notebooks/{id}/cancel` | Cancel using an empty JSON object |
| `POST /v1/notebooks/{id}/rerun` | Create a new job using `{"request_id":"a-new-unique-id"}` |

Re-run returns a new job with `rerun_of` set to the original ID. Retrying the same request ID returns that same new job rather than duplicating execution. Worker-only endpoints handle claims, input retrieval, heartbeats, uploads and completion. Existing `/v1/jobs` routes serve the legacy reviewed-bundle queue.

## Operations

- Services: `soup-api`, `soup-notebook-worker`, `nginx`; inspect with `systemctl status` and `journalctl`.
- Worker configuration: `/etc/soup-queue-worker/notebook-worker.json`.
- Public runtime configuration: `/etc/soup-notebook-runtime.json`.
- Role credential files: `/etc/soup-queue/*.env`, root-owned; read into the appropriate process environment without printing or publishing values.
- Durable queue state and output archives: `/var/lib/soup-queue/notebooks`.
- Retained job workspaces and receipts: `/var/lib/soup-worker/work/notebooks`.

Deploy a newly verified immutable release and reconcile active work before restarting services. Source edits alone do not change a running deployment. Check the active runtime endpoint and container mount/resource settings after activation.

The IP and hostname use separate certificates and renewal configurations. The existing `soup-certbot-renew.timer` renews eligible certificates with an Nginx reload hook, including the short-lived IP certificate. If the external IP changes, update both endpoint addresses, Nginx server names and exact Origin mapping, certificates and client configuration together; this hostname encodes the current IP. Keep certificate verification enabled and verify both endpoints after renewal or configuration changes.

The deployed hostname virtual host is recorded in `deploy/nginx_hostname.conf.example`. Enable it alongside the IP virtual host only after its hostname certificate exists. The IP configuration defines the shared request-rate zone. Both certificates are renewed by the existing Certbot timer.
