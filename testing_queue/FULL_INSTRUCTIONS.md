> **Scope: legacy reviewed solution-bundle lane.** This document describes the original reviewed bundle workflow. The deployed automatic notebook lane, its API, current limits, saved-baseline metrics and Re-run behavior are documented in [NOTEBOOK_RUNNER.md](NOTEBOOK_RUNNER.md). Review and allowlist requirements below apply to the legacy lane only.

# Full delivery: private Soup test queue, web UI and controlled runner

## 0. What runs where — read first

Your **PC controls/uploads to the cloud VM**; model training and GPU tests stay in the cloud. This package does not execute anything on your PC except an optional HTTP client/browser/SSH. No VM, port, service, real token or certificate has been created by the authors.

Set your own `INSTANCE`, `ZONE`, `VM_USER` and verified `PUBLIC_IP`; these account-specific values are deliberately not embedded in the published package. Ubuntu 22.04 and a single-T4 GCP host are **deployment examples, not verified host facts**. Recheck the current public IP and GPU before using commands. A changed ephemeral IP needs a new certificate. Reserving a static IP is optional and may cost money; this package does not reserve one.

**This is a private trusted-researcher queue, not a safe public arbitrary-code service.** Uploads are data. Reviewers turn notebook ideas into a frozen `solution/` source ZIP; a runner-owned allowlist binds the exact ZIP and every source file. An arbitrary notebook is never evaluated automatically. The GPU worker is disabled by default and requires the provided Docker boundary plus a successful self-test on the actual host/image before an owner explicitly enables it. There is no GPU host-execution fallback. The bundled CPU smoke is trusted local fixture code only, not a sandbox. Container isolation reduces exposure but is not a proof against hostile GPU code; admit only reviewed research code and complete host acceptance first. Never put cloud credentials or control-plane secrets in candidate-readable homes, source bundles or mounts.

### Ready now / not claimed

- Durable queue, header-token API, one-page upload/status/results dashboard, agent client, reviewed-source worker adapter, local tests and smoke job.
- Historical exact **direct** Kaggle package/Python pins, fixed official source commit, data/model hash binding and host checks. This is not a complete transitive package freeze or identical host/driver guarantee.
- Own GCP baseline required; GCP measurements never become official Kaggle results. Score stays null and automatic admission stays UNKNOWN. Worker output is evidence for review, not an official judge.
- Live GCP execution, package installation, GPU parity, nginx/TLS/systemd and browser rendering have not been exercised here. Do not label the local CPU tests GPU validation.

## 1. Verify and install the code on the VM

Transfer the release ZIP and its published SHA256 using your usual authenticated SSH/SCP route. On the VM, in a new directory:

```bash
# Replace paths; compare the displayed ZIP SHA256 with the supplied handoff hash.
sha256sum soup-test-queue-delivery.zip
unzip soup-test-queue-delivery.zip -d "$HOME/soup-queue-release"
cd "$HOME/soup-queue-release/soup-test-queue"
python3 install.py --verify-only
# No overwrite: choose a new prefix if this already exists.
sudo python3 install.py --prefix /opt/soup-queue
```

The queue/API itself uses Python standard-library features available in Python 3.10+. GPU tests use a **separate exact Python 3.13.15 environment** below. `install.py` copies only hash-verified package files; it never installs packages, changes users, opens ports or deletes an existing directory.

## 2. Run the small end-to-end smoke job first

```bash
cd /opt/soup-queue
python3 smoke_demo.py --out "$HOME/soup-queue-smoke-01"
```

This creates temporary local queue state, ephemeral loopback HTTP and conspicuously dummy credentials; it submits the bundled tiny arithmetic solution, explicitly approves its exact hash, claims it through the worker, executes a fresh CPU child, posts the structured result and reads it back. It exercises the actual protocol, **not a simulated success JSON**. It neither installs packages nor calls a GPU. Output must report completed CPU smoke with `score:null`, `admission:UNKNOWN`. Keep the output evidence. Existing output directories are not overwritten.

Infrastructure test commands (no scientific reruns), run from the **user-writable extracted checkout**, not the root-owned `/opt` installation:

```bash
python3 run_tests.py
python3 run_api_tests.py
python3 test_delivery.py
python3 test_runner_adapter.py
node test_web_state.js  # optional Node.js state-flow regression checks
```

## 3. Create separate service users, storage and credentials

The following are **administrator actions for you to review and run**, not actions already performed. If a user/path already exists, inspect it rather than deleting or recreating it.

```bash
sudo useradd --system --create-home --home-dir /var/lib/soup-queue --shell /usr/sbin/nologin soup-web
sudo useradd --system --create-home --home-dir /var/lib/soup-worker --shell /usr/sbin/nologin soup-worker
sudo install -d -m0700 -o soup-web -g soup-web /var/lib/soup-queue
sudo install -d -m0700 -o soup-worker -g soup-worker /var/lib/soup-worker
sudo python3 /opt/soup-queue/deploy/init_secrets.py \
  --directory /etc/soup-queue --confirm-create-secrets
```

The secret initializer fails if its directory exists, creates new random independent role tokens locally, writes 0600 files and **prints no token values**. It has not been run with real credentials here. Never use `00009999`, URL parameters, repository files or chat for passwords/tokens. Distribute only the relevant role file over a secure channel. API environment contains all role credentials; workers receive only the worker credential. Reviewer and submitter credentials must not be accessible to candidate processes.

## 4. HTTPS on the IP address — no domain purchase required

The supplied nginx/Certbot files are templates, not an installed service. The no-domain route uses Let's Encrypt short-lived IP certificates. Official source: [Certbot IP-certificate instructions,11 March 2026](https://letsencrypt.org/2026/03/11/shorter-certs-certbot.html). It requires Certbot 5.4+ and webroot issuance; do not assume the nginx plugin supports IP issuance. Certificates last about six days, so renewal is mandatory.

1. Verify the VM public IP. In GCP explicitly approve/open **TCP 80 for ACME validation and TCP 443 for HTTPS**, preferably restrict 443 to your researchers' source IPs if practical. Keep 8765 private/loopback. Review SSH access separately. This package makes no firewall changes.
2. Install nginx, Python venv support and Certbot in a separate admin environment. Example on Ubuntu, before any measured work:

```bash
sudo apt-get update
sudo apt-get install nginx python3-venv unzip
sudo python3 -m venv /opt/soup-certbot
sudo /opt/soup-certbot/bin/python -m pip install 'certbot>=5.4'
/opt/soup-certbot/bin/certbot --version  # must be5.4 or newer
sudo install -d -m0755 /var/www/acme
```

3. Set `PUBLIC_IP` to the current verified address and render the non-secret templates into a new review directory:

```bash
python3 /opt/soup-queue/deploy/render_templates.py --public-ip "$PUBLIC_IP" --out "$HOME/soup-rendered-01"
```

Set `PUBLIC_IP` to the verified address before this command. Render `deploy/nginx_acme_only.conf.template` for the verified IP. It configures an initial HTTP-only nginx server with `location /.well-known/acme-challenge/ { root /var/www/acme; }` and `location / { return 404; }`. Do not install the 443 block before certificate files exist. Port 80 is **ACME only**, never send a token there. Check `sudo nginx -t`, then reload nginx.
4. Issue the production certificate interactively, accepting terms yourself; do not hide agreement in an unattended script:

```bash
PUBLIC_IP=REPLACE_WITH_VERIFIED_PUBLIC_IP  # not a password or URL
sudo /opt/soup-certbot/bin/certbot certonly \
  --preferred-profile shortlived --webroot --webroot-path /var/www/acme \
  --ip-address "$PUBLIC_IP"
```

If you first use `--staging` to test issuance, that certificate is intentionally untrusted; **never bypass browser/TLS verification or send tokens to it**. Follow Certbot's documented production issuance path afterward and verify the production certificate. Do not assume a staging success is a usable HTTPS endpoint.

5. Render `deploy/nginx_ip.conf.template` replacing `__PUBLIC_IP__` with the verified address; review then install it into an appropriate nginx site/config location. Preserve unrelated sites. The certificate paths in the template are `/etc/letsencrypt/live/<IP>/fullchain.pem` and `privkey.pem`; confirm Certbot's actual emitted lineage before installation. `sudo nginx -t` must pass before reload. It proxies only to 127.0.0.1:8765, rewrites Host appropriately and disables request access logging.
6. Install the supplied reload hook and renewal service/timer after review:

```bash
sudo install -m0755 /opt/soup-queue/deploy/soup-reload-nginx /usr/local/sbin/soup-reload-nginx
sudo install -m0644 /opt/soup-queue/deploy/soup-certbot-renew.service.template /etc/systemd/system/soup-certbot-renew.service
sudo install -m0644 /opt/soup-queue/deploy/soup-certbot-renew.timer.template /etc/systemd/system/soup-certbot-renew.timer
sudo systemctl daemon-reload
sudo systemctl enable --now soup-certbot-renew.timer
sudo /opt/soup-certbot/bin/certbot renew --dry-run
systemctl list-timers soup-certbot-renew.timer
```

Review any existing Certbot timer to avoid conflicting duplicate renewal setup. The deploy hook validates nginx and reloads only after successful renewal. Monitor renewal failures and expiry; a short-lived certificate without working renewal will break access quickly.

## 5. Start the API, then open the single-page dashboard

Render `deploy/soup-api.service.template` with the verified IP, review and install as `/etc/systemd/system/soup-api.service`. The service runs as `soup-web`, has only `/var/lib/soup-queue` writable, and binds 127.0.0.1:8765. Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now soup-api.service
sudo systemctl status soup-api.service --no-pager
```

Open `https://<verified-IP>/` from your PC. Verify the browser shows a valid certificate; no warning bypass. Enter your submitter or viewer token in the page. It is held only in page memory and cleared by Clear; it is never included in a URL or local storage. All results/status data requires authentication. The upload form does not review or launch jobs. Review/worker actions use the scoped HTTP client/API.

The VM must remain running for the site to stay online; a stopped VM means the dashboard is offline. Keep queue DB/artifacts on retained disk and back them up with SQLite-aware backup procedures. Do not copy a live WAL database file alone and assume consistency.

## 6. Prepare a Kaggle-compatible GPU environment — fail on mismatch

`requirements-kaggle.txt` pins the observed direct versions:
Python 3.13.15; Torch 2.11.0+cu128; Triton 3.6.0; soup-cli[train]0.77.0; Transformers 5.16.1; PEFT 0.20.0; TRL 0.29.1; bitsandbytes 0.50.2; Accelerate 1.14.0; datasets 4.8.5.

Ubuntu 22.04's default Python is not 3.13.15. Supply a separately installed, trusted **exact Python 3.13.15** executable; this package does not replace system Python or silently select 3.10/3.12. If unavailable, stop and resolve that prerequisite. Driver installation is also a separate administrator prerequisite; do not change drivers during benchmark runs.

```bash
export PYTHON313=/absolute/path/to/python3.13
export VENV_DIR=/absolute/new/path/to/soup-kaggle-venv
bash /opt/soup-queue/deploy/install_environment.sh
```

The script refuses existing venv paths, installs only the specified environment, runs `pip check`, validates exact versions and saves an installed freeze. It never downgrades pins to make installation succeed. If any pin is unavailable, preserve the failure and stop; do not claim Kaggle compatibility. The checked direct pins do not prove all transitive packages or OS components match Kaggle.

Obtain the **unchanged** official source:

```bash
git clone https://github.com/MakazhanAlpamys/soup-kernel-challenge.git /absolute/new/path/to/soup-official
cd /absolute/new/path/to/soup-official
git checkout --detach c7c2fcd15972ce8b898d8d3ff732ee192c817ceb
```

No solution may modify `harness/`, `task/` or `tests/`. The original GPU memory sampler is retained. Record driver/OS/CPU differences; same software and T4 are **not identical hardware**, and a GCP baseline must be measured separately. Historical Kaggle had 4 usable logical CPUs, 2 physical cores, 30 GiB RAM limit and 2 physical T4s, with only GPU 0 visible to the task. Your intended GCP VM has different host resources; four-logical-CPU affinity is a control, not proof of equivalence.

Choose four CPUs from the actual process affinity, then validate before a GPU profile:

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c <four-valid-cpu-ids> "$VENV_DIR/bin/python" \
  /opt/soup-queue/check_kaggle_environment.py --gpu --require-four-cpus \
  --repo /absolute/path/to/soup-official --out /absolute/new/path/environment.json
```

Replace the angle-bracket placeholder; do not paste it literally. Exactly one visible Tesla T4 is required. Torch's compiled CUDA 12.8 is not the driver's advertised CUDA capability. Driver differences are recorded, not silently declared equal. No GPU check was run by package preparation. The sanitized historical reference is `reference/kaggle_environment.json`; OS/kernel fields intentionally remain null.

## 7. Prepare public assets once, outside measurement

Use the official pinned data preparation (public rows only):

```bash
cd /absolute/path/to/soup-official
"$VENV_DIR/bin/python" task/prepare_data.py
```

Use a dedicated public-only Hugging Face cache, with no tokens or adapter/compiler state. The separately invoked bootstrap below explicitly uses `token=False` for the public model and is never invoked by the queue or timed runner:

```bash
"$VENV_DIR/bin/python" /opt/soup-queue/prepare_public_model.py \
  --cache /absolute/new/public-hf-hub-cache --confirm-public-download
```

It downloads only the selected public model at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` using the installed Hugging Face client. Do not silently update `main`. `refs/main` and snapshot metadata must resolve that revision; retained snapshots whose links were stripped must be restored by a validated helper, not guessed. No model weights are in this ZIP.

```bash
"$VENV_DIR/bin/python" /opt/soup-queue/prepare_asset_manifest.py \
  --cache /absolute/public-hf-hub-cache --repo /absolute/path/to/soup-official \
  --out /absolute/new/path/model-manifest.json
```

This reads/hashes only the selected model and checks the two fixed public data hashes. Keep the resulting manifest SHA and `data_files` map in the trusted runner config. A mutable/unpinned/download-on-demand cache is not accepted as a known cohort. Public model/data caches can be reused; adapters, optimizer state and compiler caches cannot. Each measured child uses fresh output and compiler-cache directories; nothing is deleted unattended.

## 8. Configure the runner and establish its own GCP baseline

Default GPU worker execution is disabled. Do not expose automatic execution to arbitrary uploaders. GPU execution requires the supplied Docker mode: distinct nonroot candidate UID, network disabled, read-only source/cache/data, and no API credential, queue database, broad host directory, or Docker socket mounts. Run the target self-test before explicit owner activation; there is no same-UID GPU fallback. See `RUNNER_ADAPTER.md` and the supplied config examples for exact schema/commands. The worker config is **administrator/reviewer-owned**, never supplied by candidate uploads. It pins backend, baseline ID, environment/task fingerprints, official commit, interpreter, canonical resource, CPU affinity, data/model manifest, timeout and allowed source hashes. Supply the worker token as `SOUP_QUEUE_WORKER_TOKEN` via its protected environment only; the generated `worker.env` uses that name. Reviewer/submitter client files use `SOUP_QUEUE_AGENT_TOKEN`. Never run candidate code as root or the web user.

The adapter supports a bounded native screen, not automatic contest scoring. It invokes the fixed official harness with the reviewed solution source and fresh paths, checks native/nested completion, and suppresses evaluation/complete metrics for timeout or failure. A killed 79-second run is censored, not a successful 79-second runtime. **79 seconds is a historical Kaggle screen budget, not a universal GCP policy.** Choose and record a sufficient GCP budget explicitly (for example the official 900-second per-run default), and use the same budget/cohort for its baseline and candidate. Do not tune timeouts after seeing a result without creating a new cohort record.

Follow `RUNNER_ADAPTER.md`: create `examples/worker_gcp.json` as a protected host configuration, run `--selftest` while disabled, review its actual target record, and then explicitly enable GPU execution. Run the adapter's explicit `--baseline` command and keep its raw evidence/hash. Register only that matching baseline record with reviewer scope. Then permit one reviewed candidate at a time for the canonical GPU resource. The queue does not inspect GPUs. The runner checks host GPU idleness and serializes its own baseline/candidate work with one host lock, but cannot stop unrelated processes that bypass it; the operator must ensure exclusivity. GCP results remain GCP screening evidence. Formal paired Kaggle runs, unchanged official checks and jury review remain separate requirements; no local queue string can confer official admission.

## 9. Agents submit; reviewers approve; worker returns results

Prepare source from a notebook by manually extracting/reviewing only the original `solution/` module. Remove installation, downloads, shell cells, credentials, task overrides and unrelated code. The submitted notebook can remain a read-only research attachment; the initial worker accepts **reviewed ZIP solution bundles only**.

```bash
python3 /opt/soup-queue/bundle_candidate.py --solution-dir /path/to/reviewed/solution \
  --out /new/path/candidate.zip --manifest /new/path/candidate-files.json
```

An agent's submission JSON contains only candidate ID/backend/baseline ID/closed entrypoint/artifact kind/priority/no-secrets attestation; see `examples/http_submission.json`. Load the appropriate role credential securely into `SOUP_QUEUE_AGENT_TOKEN` without putting its value in command arguments/history. Example client operations:

```bash
python3 agent_client.py --url https://<verified-IP> submit --spec submission.json --artifact candidate.zip
python3 agent_client.py --url https://<verified-IP> status --job q_<returned-id>
# Reviewer credential, after inspecting the exact bundle and adding its file hashes to the trusted worker allowlist:
python3 agent_client.py --url https://<verified-IP> review --job q_<returned-id> \
  --sha256 <returned-source-sha256> --authorization-ref <specific-approval-id> --decision approve
```

The worker runs `runner_adapter.py --config <trusted-config> --once` under its own identity, not from a notebook cell. Run it manually first; do not enable unattended repeating GPU work until quota, isolation, crash recovery and evidence are accepted. It posts the actual result; agents poll the job. The UI displays runtime kind, peak completeness, evaluation, own share, admission and nullable score without inventing absent values.

## 10. Failure, recovery and limits

- HTTP/auth error: check endpoint/role, never disable TLS verification or print the token.
- Package/driver/task/source/hash mismatch: stop before GPU execution; preserve logs, do not silently substitute.
- Uncertain submission, lost worker, permission denial or expired lease: resource remains quarantined. Inspect real process/output state before any new attempt; no blind retry.
- Timeout/native nonzero: preserve observed duration/partial peak; no eval/share/score. Wrapper exit 0 or visible progress 64 is not completion.
- Restart API: SQLite state survives; it does not launch jobs. Started uncertain jobs need terminal evidence or owner investigation, not deleting locks/DB.
- Existing output/install paths: choose new paths; scripts do not recursively delete old evidence.
- Official quality/own-share checks require actual trustworthy task evidence. The queue validates identity/schema; it cannot cryptographically prove a result truthful. All automatic adapter scores are null and admission UNKNOWN.
- Logs, source and evidence may contain research information. Keep the service private; do not publish raw logs/weights/notebook identifiers unintentionally.

Initial acceptance: full CPU smoke succeeds; auth roles and file permissions reviewed; no public 8765; valid HTTPS/renewal; verified exact environment and pinned assets; idle one visible T4; own clean GCP baseline; one reviewed candidate; preserve failed as well as successful outcomes. Only then consider worker automation.

## Optional exact-Python build prerequisite

If your VM has no Python 3.13.15, obtain the exact CPython 3.13.15 release from the official Python release site, verify its published checksum/signature, and build it into a **new separate prefix**, not `/usr/bin/python3`. A typical Ubuntu source-build prerequisite list is `build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev uuid-dev`. After verification, `./configure --prefix=/new/absolute/python-3.13.15 --with-ensurepip=install`, `make -j2`, and `make install` install only into that selected prefix (use the owning account). Confirm `bin/python3.13 --version` prints exactly 3.13.15 before running the environment script. Package/system installation is a separate user-administrator action outside benchmark timing. If the exact official release or required wheel is unavailable, stop and report that blocker; do not silently replace it with a nearby version.

## Connectivity preflight and offline handoff

Before promising automated agent submissions, test reachability **from each actual submitting environment**: open the verified HTTPS page and run `agent_client.py ... session` with that submitter's scoped credential. Access from your PC does not prove an agent cloud environment can reach the VM. The authors' direct SSH/HTTPS route was not established; agent-to-VM connectivity is unverified. Do not weaken TLS or use a proxy/tunnel to evade a policy denial.

If an authorized agent cannot reach the endpoint, it can produce `candidate.zip`, the exact source SHA and a submission JSON **offline**, then hand those files to you over the approved artifact channel. You upload them from your reachable PC/client. The reviewer still approves the exact bytes, and results can be exported by you as JSON. This preserves the queue/review protocol without pretending unavailable direct automation works.

## Required GPU execution image (build is not performed here)

The native GPU worker requires the separately reviewed Docker isolation configuration in `RUNNER_ADAPTER.md`; it must not fall back to host same-UID execution. Install/configure Docker Engine and the NVIDIA Container Toolkit through their official administrator procedures on the authorized VM, and verify the GPU runtime before enabling a worker. Do not give the web user Docker-group/socket access. Docker supervisor privilege is itself powerful; the candidate container must never receive that socket.

Choose a **verified digest-pinned** official Python 3.13.15 base image compatible with Linux/amd64; no mutable or older-image fallback. Build only outside benchmark timing:

```bash
cd /opt/soup-queue
# Replace the complete placeholder with a locally verified image reference and SHA256 digest.
docker build --build-arg PYTHON_BASE_IMAGE='python:3.13.15-slim-bookworm@sha256:REPLACE_WITH_VERIFIED_DIGEST' \
  -f deploy/Dockerfile -t soup-queue-kaggle-stack:local .
docker image inspect soup-queue-kaggle-stack:local
```

The Dockerfile fails if Python/direct pins do not match and records its installed package freeze. The build may download large public CUDA wheels. It contains no model, token, adapter or compiler cache. Select the exact locally built image digest/ID supported by the runner configuration; do not assume a mutable tag is a reproducible pin. Image distribution/registry upload is a separate action, not done by these instructions. The worker must use an existing image (`--pull=never`) and enforce network none, nonroot, read-only root/source/cache/data, dropped capabilities, no-new-privileges and only unique writable output. Host disk quotas and memory limits remain administrator acceptance prerequisites, not limits enforced by this adapter; configure them before accepting candidate work. GPU host drivers are still separate and recorded; an image does not make GCP identical to Kaggle.

Docker argv/static tests are not live sandbox/GPU validation. Until the actual host configuration and one bounded isolated acceptance job are verified, keep GPU worker execution disabled. Do not enable a repeating worker service merely because the dashboard works.
