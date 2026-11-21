> **Scope: legacy reviewed solution-bundle lane.** This document describes the original reviewed bundle workflow. The deployed automatic notebook lane, its API, current limits, saved-baseline metrics and Re-run behavior are documented in [NOTEBOOK_RUNNER.md](NOTEBOOK_RUNNER.md). Review and allowlist requirements below apply to the legacy lane only.

# Proposed hosting adapter — not deployed

The requested short-project layout can use the **same eventual authorized GCP GPU VM**, without another paid server or unrelated Sites deployment. This is a deployment proposal, not a claim that a VM, domain, TLS certificate, firewall rule, account or budget exists. If that VM is stopped, the dashboard/API is offline. Persistent queue data must live on retained storage rather than ephemeral scratch.

```text
Agents/browser -- HTTPS + scoped Authorization --> reverse proxy
                                                --> loopback queue API
                                                     | SQLite + frozen packets
                                                     | reviewed claim/download/result protocol
                                              separate worker supervisor
                                                     | isolated reviewed candidate process
                                                     | unique private scratch/output
                                                     + existing authorized GPU runner
```

## Required separation even on one machine

- API/coordinator runs as an unprivileged service user; only it owns the database and immutable artifact store. No candidate code runs in the API process.
- Reviewer, submitter and worker have separate credentials. Worker control-plane credentials must never enter candidate environment, notebook metadata, output files or container mounts.
- Runner supervisor uses a different account/process boundary. Candidate execution needs its own constrained user/container, read-only reviewed source, unique scratch, no coordinator DB/auth files, no broad home-directory mounts and no inherited cloud/auth credentials. A Python subprocess under the coordinator UID is **not** isolation. The mandatory GPU runner container wrapper must be explicitly configured and validated on the target host; local static checks are not a live sandbox/GPU test. The web process never launches candidates.
- Future HTTPS proxy owns public networking, canonical Host/Origin, TLS, request-size and concurrency/rate limits. Only approved HTTPS ingress; upstream remains loopback. Disable request-body/Authorization/query logging. Do not expose SQLite, artifact storage or the loopback port directly.
- Use retained POSIX storage with backup/recovery and tested permissions; monitor disk quotas. SQLite here is a single-host coordinator, not a multi-host distributed queue.
- Keep the external runner's prepared session intact. A claim authorizes no session reset, package installation, compiler-cache reuse, provisioning, notebook mutation or launch by itself. Match backend/baseline, exact source + explicit pristine support allowlist, current user authority and one exclusive idle resource before action.
- Absent/uncertain submission and stale leases stay quarantined. Inspect the actual external state before any recovery. Queue leases alone do not prove a GPU is idle.

## Explicit next authorization and acceptance gates

Before deployment, obtain per-action approval for the target VM/project/region/budget, retained disk, domain/DNS/TLS and ingress policy, service/process isolation, production secret generation/storage/rotation, and runner activation. None of these changes was performed. No real credential was generated or persisted. No paid calls, firewall edits or service installation occurred.

After approval, deployment acceptance must verify HTTPS/auth and secret redaction, role boundaries, upload/disk/rate limits, proxy Host/Origin behavior, coordinator restart durability, sandbox denial of coordinator/credential access, worker crash reconciliation, and one bounded fake job before separately authorized real GPU consumption. Public exposure must not rely on Python's development `http.server` as the sole hardened edge; replace/harden the serving layer behind the proxy as part of that deployment review.

Current deliverable: usable local core + HTTP/UI prototype + explicit runner interface. Live deployment and sandbox acceptance, account quota enforcement, per-agent identities, automatic push notifications and persistent service operations remain **not implemented**. Polling returns structured results, including UNKNOWN/null/censored fields; it never invents a score.

See `FULL_INSTRUCTIONS.md` for the concrete user-run installation steps and `RUNNER_ADAPTER.md` for the now-implemented opt-in adapter. The original local data-only queue remains the control plane; a bundle never grants its own execution authority.

The adapter enforces PID and watchdog bounds, not host output-disk or memory quotas. Configure and test those limits separately before candidate acceptance. Container result reads reject symbolic links, special files, and oversized JSON/logs, but that is not an output-storage quota.
