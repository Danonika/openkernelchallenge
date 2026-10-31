> **Scope: legacy reviewed solution-bundle lane.** This document describes the original reviewed bundle workflow. The deployed automatic notebook lane, its API, current limits, saved-baseline metrics and Re-run behavior are documented in [NOTEBOOK_RUNNER.md](NOTEBOOK_RUNNER.md). Review and allowlist requirements below apply to the legacy lane only.

# Reviewed worker: configuration and acceptance

This adapter completes the queue's **submit → review → claim → execute → result** path. It is not an official judge and does not execute arbitrary notebooks. Your PC controls the authorized cloud VM; GPU work runs only on that VM. No Docker, driver, GPU, deployment or training acceptance test has been performed by preparing this package.

Start with [FULL_INSTRUCTIONS.md](FULL_INSTRUCTIONS.md) for installation, API, HTTPS and pinned public assets. The example [`examples/worker_gcp.json`](examples/worker_gcp.json) is deliberately **disabled and incomplete**. `REPLACE_...` values are not measured evidence and must not be submitted as such.

## 1. Execution profiles and trust boundary

| Profile | Accepted artifact / entrypoint | Actual execution |
|---|---|---|
| `cpu_smoke_v1` | Reviewed ZIP / `cpu_contract_v1` | Fresh host Python process; **trusted bundled infrastructure test only**, not hostile-code isolation |
| `soup_screen_v1` | Reviewed `solution/` ZIP / `soup_screen_v1` | Fixed official harness inside the target-validated Docker boundary; one native 64-step run and public evaluation |

The queue also recognizes other historical entrypoint names, but this adapter does **not** implement paired runs, GPU preflight profiles or notebook execution. A recognized queue entrypoint is not permission or an executable command.

GPU candidates never run directly as the supervisor. The adapter requires:

- An existing immutable image reference; no automatic pulls, builds or package installation.
- A nonroot container UID different from the supervisor UID, dropped capabilities and no-new-privileges.
- No network, private process/IPC namespaces, and exactly the selected GPU 0.
- Read-only root, staged official/candidate source, launcher and selected public model cache.
- Only its unique `/output` bind mount writable, plus temporary container filesystems.
- No API database, bearer token, cloud credentials, home directory or Docker socket in candidate mounts or environment.
- One host GPU lock shared by baseline and candidate execution; uncertain container cleanup blocks the lane.

The worker supervisor's Docker access is powerful. It must not be granted to the web user. The target selftest checks namespace/mount policy; it is **not a proof against hostile GPU/kernel code**. Keep the service private to reviewed trusted researchers. The operator must review host security and establish explicit host storage and memory/resource limits before activation; these are not enforced by this adapter. A unique directory is not a disk quota, and the read bound on result JSON is not an output-space limit. Do not grant public arbitrary-code access.

## 2. Prepare a trusted worker configuration

Copy the example into a new administrator-owned configuration file outside candidate-readable storage. Do not let uploaders edit it. Fill the actual values after reviewing them:

| Field | Meaning |
|---|---|
| `queue_url` | Same-host API may use `http://127.0.0.1:8765`; remote clients require verified HTTPS |
| `workspace_root` | Dedicated retained worker directory; never the API state directory |
| `python` | Absolute installed supervisor interpreter; CPU smoke uses it, GPU execution uses the image interpreter |
| `repo` | Clean checkout of the pinned official commit, with the two prepared public data files |
| `cpu_affinity` | Four actual available logical CPU IDs, not assumed IDs or equivalent CPU hardware |
| `container_image` | Exact locally present `repository@sha256:<64 hex>` or local image ID `sha256:<64 hex>`; never a mutable tag |
| `container_uid` / `container_gid` | Dedicated nonzero candidate identity; UID differs from the supervisor |
| `hf_hub_cache` | Dedicated public-only cache prepared separately, on the same filesystem as the workspace for hardlinks |
| `model_cache_manifest*` | Exact output path and SHA256 from `prepare_asset_manifest.py` |
| `data_files` | Exact two public data hashes from that helper, not arbitrary caller-selected data |
| `baseline_id` | New named GCP cohort, not a copied Kaggle baseline |
| `resource` | Canonical resource name for this VM's GPU 0, used consistently by all clients |
| `lease_seconds` | Lease bound, 10–3600 seconds; example 600 |
| `baseline_timeout_s` | Explicit native train/evaluation budget, identical to the candidate cohort budget |
| `baseline_outer_timeout_s` | Larger whole-harness watchdog including warmup, train and evaluation |
| `profiles` | Empty initially; reviewer-owned exact ZIP hash → fixed execution profile |

The historical Kaggle screen budget is not a GCP deadline. The example uses a **proposed** 900-second native budget and 2400-second outer watchdog. These are configuration choices, not performance claims. Choose a sufficient budget before measuring the cohort, and preserve it for baseline and candidate. Do not silently lengthen it after seeing a failure.

Confirm `cpu_affinity` from the worker's actual available CPUs:

```bash
python3 -c 'import os; print(sorted(os.sched_getaffinity(0)))'
```

Use the full image SHA256, not a mutable tag. For an image built locally without a registry digest, read its immutable ID with `docker image inspect --format '{{.Id}}' soup-queue-kaggle-stack:local` and copy the returned `sha256:...` value into `container_image`; no registry push is needed. The supplied Dockerfile is only a build template. Verify the image contains the exact direct pins from `requirements-kaggle.txt`; unknown Kaggle OS/kernel/transitive versions remain unknown. Image package checks do not validate the host GPU driver. The adapter records actual inventory and requires baseline/candidate runtime agreement.

## 3. Freeze cohort fingerprints before the baseline

`environment_fingerprint` and `task_fingerprint` are operator-created immutable cohort IDs, not automatic claims that GCP equals Kaggle. Preserve the descriptor files used to derive them. Use SHA256 of the UTF-8 canonical JSON representation: sorted keys, separators `(',', ':')`, no NaN.

A suitable **environment descriptor** includes:

- Exact container image digest/ID and the image's installed package-freeze hash.
- The actual host GPU name/driver/inventory report and its file hash.
- Four CPU affinity IDs, the recorded host CPU model and relevant resource limits.
- The fixed allocator and visible-device policy, plus hashes of runner/helper files.

A suitable **task descriptor** includes:

- Official commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
- Hashes of unchanged `harness/`, `task/`, `tests/` files, plus both public data hashes.
- Public model ID, exact revision and the model-cache manifest SHA256.
- Seed 1, one native screen repetition, expected 64 steps, native timeout and outer watchdog.

Create these JSON files from the actual reviewed configuration/evidence; do not fill unknown fields with guesses. To obtain the canonical fingerprint for either file:

```bash
python3 - /absolute/path/environment-cohort.json <<'PY'
import hashlib, json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
print(hashlib.sha256(raw).hexdigest())
PY
```

Repeat with `task-cohort.json`, and copy the results into the config. Keep the descriptors alongside the protected configuration, not in candidate output. The worker independently checks the pinned source, cache and actual runtime against its measured baseline; merely typing a matching fingerprint cannot replace those checks.

The all-zero `baseline_fingerprint` in the example is **only a bootstrap sentinel** needed while running `--selftest` and `--baseline` before registration. It is not a baseline and must be replaced with the queue's registered hash before candidate consumption.

## 4. Target selftest — still disabled

Perform this under the dedicated worker identity on the authorized VM, with no API token needed. The worker account must have the reviewed Docker access required by the chosen installation. Do not run this command from your PC's local Python.

```bash
python3 /opt/soup-queue/runner_adapter.py \
  --config /absolute/protected/worker-bootstrap.json --selftest
```

Keep `gpu_execution_enabled: false` for this step. A fresh `boundary_<id>/` directory is created. The command returns a `container_validation_record` path and SHA256. Read that exact record and child checks. Acceptance requires `passed_target_container_selftest`, confirmed owned-container cleanup, and every policy check passing.

No candidate or training runs during this selftest, but it is an actual target container operation using the configured GPU runtime. A failure is a blocker, not permission to remove flags, use root, expose the Docker socket or fall back to host execution.

Only after owner acceptance:

1. Put the returned record path and hash into a new protected config version.
2. Confirm it is bound to this host, this image, the adapter and helper hashes, and the accepted UID/GID, Docker path and CPU-affinity policy.
3. Explicitly set `gpu_execution_enabled: true` in that owner-reviewed version.
4. Leave `profiles` empty while establishing the baseline.

The flag alone does not pass the record/hash/host/image gate. Changes to the accepted boundary require another target selftest. This package does not enable an unattended worker service.

## 5. Measure the GCP baseline

This is separately authorized real GPU work, not a local package test. Confirm the VM/GPU is idle, quota/budget permits the run and no other process bypasses the worker's lock. Then run one baseline:

```bash
python3 /opt/soup-queue/runner_adapter.py \
  --config /absolute/protected/worker-baseline.json --baseline
```

No queue token is needed. The command stages unchanged official source/data and public-cache bytes, rechecks the target stack/device, invokes the fixed official harness with `--only base --repeats 1 --seed 1`, and saves its own new output.

The result must say `measured_base_complete`. This requires successful native and outer exits, 64 verified steps, adapter files, finite successful public evaluation, no package mutation and confirmed cleanup. Retain the entire new baseline directory, raw result and record. `baseline_failed` is not usable; preserve it rather than replacing its evidence. Diagnosing/retrying a failed baseline requires explicit current run authority and a new output/cohort decision, never a blind loop.

## 6. Register that exact measured baseline

Set `baseline_record` and `baseline_record_sha256` from the successful baseline receipt. Build a **new** registration JSON using the config's unchanged fields and this record hash:

```json
{
  "baseline_id": "YOUR_NEW_GCP_COHORT_ID",
  "backend": "GCP",
  "environment_fingerprint": "YOUR_ENVIRONMENT_DESCRIPTOR_SHA256",
  "task_fingerprint": "YOUR_TASK_DESCRIPTOR_SHA256",
  "harness_commit": "c7c2fcd15972ce8b898d8d3ff732ee192c817ceb",
  "evidence_sha256": "YOUR_MEASURED_BASELINE_RECORD_SHA256"
}
```

Using only reviewer credentials securely loaded into `SOUP_QUEUE_AGENT_TOKEN`, register it:

```bash
python3 /opt/soup-queue/agent_client.py --url http://127.0.0.1:8765 \
  baseline --spec /absolute/protected/measured-baseline-registration.json
```

Save the receipt and copy its returned `hash` to `baseline_fingerprint` in a new protected worker config version. This hash is exactly:

```text
SHA256(canonical JSON of the six registration fields above)
```

The adapter recomputes that binding and requires the same host, image, source commit, public data/cache, CPU affinity, recorded environment and complete raw baseline result. Do not register synthetic files from `examples/` as measurements. Do not reuse this GCP baseline ID for Kaggle or another machine/image/cohort.

## 7. Allow one reviewed candidate

The packager returns two independent bindings: the ZIP's `source_sha256`, and the map of each path to its content SHA256. Review both the full source and exact ZIP bytes. An API approval alone is insufficient; an owner/reviewer must add the matching profile to the trusted config.

```json
"profiles": {
  "EXACT_REVIEWED_ZIP_SHA256": {
    "kind": "soup_screen_v1",
    "files": {
      "solution/__init__.py": "EXACT_REVIEWED_FILE_SHA256"
    },
    "support_files": {},
    "timeout_s": 900,
    "outer_timeout_s": 2400,
    "expected_steps": 64
  }
}
```

The candidate `timeout_s` and `outer_timeout_s` must exactly equal `baseline_timeout_s` and `baseline_outer_timeout_s`; this is enforced, not merely recommended. Replace `files` with the complete map from `bundle_candidate.py`, not only the entrypoint shown here. The staged source set must match exactly. Only `solution/` files may be replaced. `support_files`, if necessary, must explicitly list additional **unchanged pinned starter `solution/` files** by hash; they cannot overlap uploaded files. No implicit starter leftovers or arbitrary paths/commands are admitted.

Submit using `examples/http_submission.json`, edited to the measured `baseline_id` and a new candidate ID. The submitter must preserve the returned job ID/hash. A reviewer downloads and inspects the packet, confirms current execution authority, then approves that exact hash with a specific authorization reference. [HTTP_API.md](HTTP_API.md) gives the endpoint contract and client commands.

## 8. Consume once and return the real result

Give the supervisor **only** the worker role credential in `SOUP_QUEUE_WORKER_TOKEN`; never put token values in shell arguments, JSON configs, notebook cells or candidate-readable paths. Keep the API's all-role environment file accessible only to its intended service administrator/process. If running manually, the administrator can securely install only the worker role file into a worker-owned `0600` credentials directory and launch a shell that exports that file. The candidate container does not receive the supervisor environment.

As the authorized dedicated worker account, with its protected credential loaded:

```bash
python3 /opt/soup-queue/runner_adapter.py \
  --config /absolute/protected/worker-ready.json --once
```

This consumes at most one reviewed matching job. No matching job returns a no-work response, not a fake successful run. `--once` does not approve a job or retry ambiguous work. Keep it manual until the full target acceptance checklist has passed; no scheduling loop is supplied or enabled.

The submitter/viewer checks the terminal result using its own role token:

```bash
python3 /opt/soup-queue/agent_client.py --url https://YOUR_VERIFIED_IP \
  status --job q_RETURNED_JOB_ID
```

This returns actual queue JSON including `state` and `result` when terminal. A browser dashboard can display the same fields. Direct agent-to-VM connectivity must be tested from the actual agent environment; reachability from your PC proves only your PC's route. If the agent cannot reach it, use the offline reviewed-bundle/result handoff in `FULL_INSTRUCTIONS.md`; do not bypass a denied route.

## 9. Evidence and recovery

Every attempt has a unique directory with frozen claim/source, reviewed and executed manifests, process/container evidence, raw result and normalized result or a worker-error record. Logs and raw output may contain nonpublic research; they are not part of a public release. Do not delete failed/uncertain evidence or rerun into existing directories.

- Native timeout, signal or wrapper interruption stays censored/unknown as appropriate. No completed runtime is invented from a killed command's duration.
- Step 64, nested exit 0 or wrapper exit 0 alone is not success. Adapter creation and verified native completion are checked separately.
- Partial peak memory is marked partial. Native runtime and whole-wrapper duration remain separate.
- Evaluation is never taken from an incomplete/failed native training run. A missing or failed evaluation stays absent; it cannot establish quality.
- Route counters and useful own-kernel share remain unknown unless separately evidenced and reviewed. Automatic admission is `UNKNOWN`; automatic score is `null`.
- A single native screen and its GCP baseline do not replace a formal paired Kaggle run, unchanged official checks, hidden evaluation or jury admission.

On a claim/start/network/crash uncertainty the adapter attempts to mark the job `needs_reconciliation`, retains the resource lock, and does not retry. If owned-container cleanup is uncertain, `/tmp/soup-queue-gpu0.lock.blocked` also blocks subsequent GPU work. Inspect that exact container/process/output and confirm terminal state and cleanup before any administrator reconciliation; do not blindly remove markers or run `docker system prune`.

The fixed host lock coordinates this adapter's own baseline/candidate commands. It cannot stop unrelated users or tools from launching GPU work. Keep exclusive ownership of the authorized GPU lane and confirm idle start for every real run.
