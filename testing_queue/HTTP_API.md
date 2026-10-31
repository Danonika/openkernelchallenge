> **Scope: legacy reviewed solution-bundle lane.** This document describes the original reviewed bundle workflow. The deployed automatic notebook lane, its API, current limits, saved-baseline metrics and Re-run behavior are documented in [NOTEBOOK_RUNNER.md](NOTEBOOK_RUNNER.md). Review and allowlist requirements below apply to the legacy lane only.

# Local HTTP API and dashboard

**Prototype only: no hosted endpoint, public listener, TLS, real credentials or automatic runner has been deployed.** The static upload/status/results shell is public locally; every queue-data request is authenticated. The page does not review, claim or launch jobs. Source packets remain data.

## Authentication and scopes

Use `Authorization: Bearer <externally-provided-scoped-token>`, never a URL password, query parameter or cookie. `pass=00009999` is rejected. The server requires separate submitter/reviewer/worker tokens; an optional viewer token is read-only. Tokens must be 32–256 ASCII letters/digits/underscore/hyphen; this format check is **not proof of entropy**. Future production tokens must be randomly generated and held outside the repository, only after authorization. No real token is supplied here. Tests contain conspicuously public `TEST_ONLY_...` strings that must never be deployed.

| Role | Rights |
|---|---|
| submitter | Read jobs/results; upload packets |
| reviewer | Read/download source; register baselines; review/reconcile |
| worker | Read/download source; claim/start/heartbeat/observe/complete |
| viewer | Read jobs/results only |

One credential per role is the minimal private-team interface, not a per-agent identity provider or multi-tenant system. All authenticated roles can read all jobs. Same-OS-account access can bypass the HTTP boundary; deployment needs separate users/storage permissions. Tokens stay in dashboard memory/input only: no local/session storage, cookies, URL, access logs or third-party assets. Clearing disconnects and removes displayed data. The server logs neither requests nor credentials.

### Local launch interface (not run as a persistent service here)

The owner must supply `SOUP_QUEUE_SUBMITTER_TOKEN`, `SOUP_QUEUE_REVIEWER_TOKEN`, `SOUP_QUEUE_WORKER_TOKEN` and optionally `SOUP_QUEUE_VIEWER_TOKEN` securely via the environment. Do not put their values in command arguments, committed files or shell history.

```text
python api.py --root /approved/private/queue-state --port 8765 --serve-local
```

Binding is hardcoded to `127.0.0.1`; there is no public bind option. Startup prints only bind/port/mode. The dashboard is `/`. All data calls use the same origin. `--trusted-public-origin https://approved.example` only permits that browser Origin for a **future separately approved** reverse proxy; it does not publish anything or change binding. The proxy must rewrite upstream Host to `127.0.0.1:8765`, preserve Authorization, and reject alternate hosts.

## Endpoints

All POST bodies are JSON; unknown fields are rejected. Queue schemas and examples are shared with the CLI. Successful replies are JSON except source downloads. Validation errors are 400, missing/invalid credentials 401, wrong scope 403, missing route 404, oversized body 413. A duplicate submission currently returns 201 with `deduplicated:true` and the same job ID.

| Method/path | Scope | Body / purpose |
|---|---|---|
| GET `/v1/session` | read | Authenticated subject/role/scopes |
| GET `/v1/jobs` | read | Job list including structured results |
| GET `/v1/jobs/{job}` | read | One job, attempts and audit events |
| GET `/v1/jobs/{job}/artifact` | artifact | Download exact source data as attachment with SHA header |
| POST `/v1/jobs` | submit | Submission metadata + base64 packet, below |
| POST `/v1/baselines` | review | Immutable baseline schema |
| POST `/v1/jobs/{job}/review` | review | `source_sha256`, `approved`, `authorization_ref` |
| POST `/v1/claims` | work | `backend`, `baseline_id`, `resource`, optional `lease_seconds` |
| POST `/v1/leases/expire` | work | `{}`; quarantine, never blind rerun |
| POST `/v1/jobs/{job}/attempts/{attempt}/start` | work | `run_reference` after confirmed start |
| POST `…/heartbeat` | work | Optional `lease_seconds` |
| POST `…/observe` | work | Hash-bound incremental observation schema |
| POST `…/uncertain` | work | `reason`, `evidence_sha256` |
| POST `…/complete` | work | Full versioned result object |
| POST `…/reconcile` | review | `evidence_sha256`, `decision`, renewed `authorization_ref` |

Submission fields are exactly:
`candidate_id, backend, baseline_id, entrypoint, artifact_kind, priority, contains_secrets, artifact_base64`.
Use `artifact_kind=notebook` or `solution_bundle`; `contains_secrets=false` is the submitter attestation. No agent-supplied disk path, runner identity or command is accepted. The server creates a random temporary staging filename, validates the packet and removes staging afterward. Store the returned `job_id` and `source_sha256`, then poll the job endpoint. Agent credentials derive identity server-side; they cannot impersonate a reviewer by adding a field.

Agent request outline (credential value is not included):

```python
import json, os, urllib.request
req = urllib.request.Request(
    approved_base_url + "/v1/jobs/" + saved_job_id,
    headers={"Authorization": "Bearer " + os.environ["SOUP_QUEUE_AGENT_TOKEN"]},
)
with urllib.request.urlopen(req, timeout=15) as response:
    result = json.load(response)
# Never log the request headers/token. Do not retry a possibly started run.
```

## Bounds and verification

HTTP request body ≤12 MiB; decoded artifact ≤8 MiB; ZIP expanded text ≤32 MiB/128 files. No chunked upload, CORS, external redirect, URL credentials, symlink/archive traversal or automatic notebook execution. The conservative notebook scanner is not a sandbox. Manual exact-hash review remains mandatory, including secrets and lawful scientific scope.

The eight API tests use temporary state, ephemeral loopback listeners and dummy credentials. They test auth/scopes, query-password refusal, upload limits/paths, idempotent data submission and result completion, guarded source download, obvious unsafe notebook rejection, no credential logging, and static dashboard escaping/token-storage controls. JavaScript syntax and four fake-DOM state regression checks are tested separately, including late responses after Clear/token edits. Browser rendering/accessibility and real HTTPS/proxy behavior have **not** been tested. No scientific kernels execute in these tests.
