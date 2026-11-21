> **Historical report for the initial reviewed-bundle delivery.** For the current deployed notebook service and verification, see [README.md](README.md) and [NOTEBOOK_RUNNER.md](NOTEBOOK_RUNNER.md).

# Delivery verification — 30 October 2026

**Source package complete; local infrastructure checks pass. Not deployed and not GPU-validated.**

## Verified on the final implementation

| Check | Actual scope and result |
|---|---|
| Python tests | **66 passed**: 27 durable queue, 8 HTTP/auth, 6 delivery/packaging, 25 runner tests |
| End-to-end smoke | **Passed**: submit → review exact hash → claim → real fresh CPU child → structured result → HTTP readback |
| Dashboard state | **4 shipped-JavaScript fake-DOM checks passed**, including stale responses after Clear/token edits and safe text rendering; not browser rendering |
| Static checks | Python compilation, JavaScript syntax and environment-install shell syntax passed |
| Archive/installer | Explicit source-file allowlist; SHA256 manifest; extraction and installer verification performed before handoff |
| Independent review | Bounded read-only review of auth, runner boundary, scientific schema, deployment instructions and release contents; fixes regression-tested |

Tests used temporary/ephemeral loopback state and conspicuously dummy credentials. The CPU child was genuinely executed, including nonzero exit, timeout and owned-descendant cleanup tests. GPU-shaped test records and Docker inspection objects are **mocks**, not measured GPU results. No real credential was generated, no external job was launched, and no user VM was contacted by this package preparation.

## Important defects repaired

- Container-controlled output could previously direct host parsing through symlinks or block it on special files. Reads now use no-follow directory-FD traversal, nonblocking regular-file validation and explicit size bounds. Intermediate-directory links, final links, FIFOs and oversized output have regression coverage.
- Baseline and candidate native/outer time budgets must match and are bound into the measured baseline record. Fixed public data hashes and the exact model revision are checked by the worker, not only a preparation helper.
- Local immutable Docker image IDs are supported without requiring a registry push. Mutable tags are rejected. Target acceptance binds the host, image, relevant code hashes and UID/GID/affinity policy.
- Late dashboard responses can no longer restore identity/data after Clear or token changes. Duplicate in-flight submissions are suppressed; queue-level content deduplication remains separate.
- Source bundling excludes generated Python bytecode without accepting secrets/state. A smoke failure caused by a prior compilation creating `__pycache__` was fixed and regression-tested; the subsequent full smoke passed.

## Not verified / remaining owner acceptance

- **Live Docker isolation, NVIDIA runtime, GPU execution, performance, actual package installation and GCP baseline/candidate consumption have not run.** GPU execution remains disabled in the template until an actual same-host/image selftest is reviewed and explicitly enabled.
- Actual Chromium launch was attempted but blocked before a page opened by this executor's Unix-socket restriction; an escalation then failed in the runtime wrapper. Browser rendering/accessibility remains **unverified**. `test_web_browser.py` is supplied for a prepared local test environment with Playwright/Chromium; it uses only dummy loopback state. No browser dependencies are added to the Kaggle environment pins.
- HTTPS/Certbot/nginx/systemd, service-user permissions, firewall rules, credential distribution, backup recovery, host disk quotas and memory limits require user-run configuration and acceptance. A unique output directory is not an enforced disk quota.
- Direct agent-to-VM API reachability remains unverified. PC reachability does not establish agent reachability; an offline reviewed-bundle/result handoff is documented.
- Historical Kaggle direct pins are preserved exactly. Full transitive freeze, OS/kernel/toolkit and uncaptured driver-advertised CUDA details remain unknown; they were not invented.
- This is a private reviewed-research queue, not a hostile public arbitrary-code service. A single native GCP screen is not a formal paired Kaggle comparison. Automatic own-share/admission remain unknown and score is null; no official success or speedup is claimed.

## Release contents and exclusions

`build_release.py` snapshots only a fixed explicit file list. The release contains code, tests, web assets, user-run deployment templates, full instructions, synthetic examples and a sanitized historical environment reference. It excludes databases, live queue state, credential files, raw logs, model/adapter weights, compiled caches, earlier private reports, private host/account identifiers and source-evidence workspace paths.

`PACKAGE_MANIFEST.json` binds every delivered source file. The handoff includes the ZIP and separate manifest SHA256. The manifest is a consistency check, not a digital signature: compare the ZIP hash through the trusted handoff channel before installation. Remote publication, deployment and GPU execution are separate steps, not effects of creating the ZIP.
