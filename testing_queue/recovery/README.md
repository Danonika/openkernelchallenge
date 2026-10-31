# Create one T4 recovery VM

Upload `create_one_t4.py` into Google Cloud Shell using its Upload file menu.
Python 3 and the authenticated `gcloud` installation are the only dependencies.

Preview:

```bash
python3 create_one_t4.py
```

Create a billable VM, trying the supplied T4 zone list one at a time until one succeeds:

```bash
python3 create_one_t4.py --execute
```

Default: Spot, `n1-standard-8`, one Tesla T4, 100 GB persistent boot disk,
fresh Ubuntu 22.04, default VPC and a new ephemeral public IP. Tokyo c/a are
tried first, followed by the other complete T4 entries from your supplied list.
The truncated us-west4-b entry was omitted. GPU location support does not
guarantee available capacity or project quota.

Your original VM, persistent disk and reserved IP are not modified. The fresh VM
still needs NVIDIA drivers, the pinned training runtime and queue deployment.
No startup script runs, no firewall rule is created and no service account is
attached by default. `gcloud compute ssh` can configure SSH access after creation.
The script reports RUNNING, not verified GPU readiness.

Spot may be stopped at any time. Its termination action is STOP. Deletion
protection is enabled and the boot disk has auto-delete disabled; any retained
disk continues to incur storage charges. This script never deletes resources.

Run only ONE Cloud Shell copy. It locks locally, uses fixed VM names, checks
all project zones for recovery VMs and persists intent before the API call.
It reuses the same Google Compute requestId after an uncertain response.
It never moves to another zone until a completed operation confirms a capacity
failure and inventory confirms no recovery VM or leftover boot disk exists.
Quota, IAM, billing, configuration, incomplete inventory and unknown failures
stop the script. A leftover disk also stops retries for inspection.

Ctrl+C stops local retries but cannot cancel an accepted cloud operation.
Resume the SAME command to reconcile. Keep
`~/.soup-t4-recovery/project-ceb435db-d81b-4fa9-b28/state.json`.
Do not delete state, launch another creator or change model/zones to bypass an
unresolved request. The local lock cannot coordinate copies on other computers.
Unresolved requests older than one hour require manual inspection before replay.

Choose only particular zones and optionally limit retry time:

```bash
python3 create_one_t4.py --execute \
  --zones asia-northeast1-c asia-northeast1-a \
  --max-minutes 120
```

An optional `--model STANDARD` uses Standard provisioning instead. Select this
before your first execution; saved configuration must match when resuming.
The default interval is 30 seconds between confirmed capacity failures.

To create a COPY from an already existing boot-disk snapshot, supply its full
Compute v1 selfLink with `--snapshot`. Creating the snapshot is a separate step.
An optional `--service-account EMAIL` attaches that account with the Sheets scope
at creation time; it requires `iam.serviceAccounts.actAs` and does not grant IAM
roles or spreadsheet sharing. A copied boot disk may start its installed services
automatically. Do not run the original and copied queue workers together without
checking their job coordination.

Reference:
- https://docs.cloud.google.com/compute/docs/reference/rest/v1/instances/insert
- https://docs.cloud.google.com/compute/docs/instances/spot
- https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones

Local mocked tests (no cloud resources):

```bash
python3 -m unittest -v test_create_one_t4.py
```
