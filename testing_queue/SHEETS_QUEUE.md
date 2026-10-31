# Google Sheets queue and results

Submit in [To Run](https://example.invalid/redacted). Read results in the same spreadsheet; agents do not need access to the VM website.

## Instructions to share with another agent

1. Prepare a Python `.ipynb` using [the repository training template](examples/notebook-training-template.ipynb). Write candidate code to `SOUP_REPO/solution/__init__.py`, exposing zero-argument `apply()`, then invoke the supplied harness. Keep its result-export cells.
2. Add a row to **To Run**, columns A:H, with a new **Request ID** and descriptive **Candidate ID**. Supply exactly one of **Notebook URL** or **Notebook JSON**. Set **Ready** to boolean `TRUE` after all inputs are complete.
3. The VM checks about every 30 seconds. Stay on **To Run**: columns I:U show status, outcome, training seconds, speedup, GPU peak, eval/training loss, limit checks, errors and per-run summaries.
4. Expand the grouped columns V:BS for queue errors, Job ID, encoded full JSON, hashes, baseline provenance and automation health. **VM Runs** retains every individual run; **VM Details** retains lossless multi-part records. **VM Results** and **VM Queue** are hidden data-feed tabs and remain required by automatic publication.
5. Poll every 30Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎС™60 seconds until status is `completed`, `failed`, `timed_out`, `interrupted`, `cancelled` or `rejected`. A completed notebook does not imply benchmark checks passed: also inspect Outcome and the loss/time/VRAM checks.

The training-command cutoff is **38.49 seconds** and the hard GPU peak limit is **2,261 MiB**, with no +10% allowance. Speed and memory compare against the selected best run; eval loss retains the original **2.04968** reference with +0.5% tolerance. Notebook and cell deadlines are separate.

| Input | Value |
|---|---|
| Request ID | Unique 1Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎС™80 letters/digits/dots/underscores/hyphens, starting with a letter/digit. Never reuse it for changed content. Reserved: `sheet-status`, `publisher-status`, `invalid-row-*`, and suffixes `.conflict` / `.rejected`. |
| Candidate ID | Display name, up to 100 letters/digits/dots/underscores/hyphens. |
| Notebook URL | Public Google Drive file or GitHub raw/blob `.ipynb` link. Empty when using JSON. |
| Notebook JSON | Complete notebook JSON as plain text, not a Python script; maximum 50,000 characters. Empty when using URL. |
| Ready | Boolean TRUE submits; false/blank leaves the row untouched. |
| Timeout (s) | Optional whole-notebook deadline, 10Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎС™14400; default 3600. |
| Cell timeout (s) | Optional, 1 through notebook deadline; default min(1800, notebook timeout). |
| Notes | Human notes; never executed. |

Use a fresh Request ID to rerun. Leaving Ready checked, reordering rows or restarting the poller does not create another job. Changing accepted inputs produces a conflict. Unchecking/deleting a submitted row does not cancel execution.

## Consolidated To Run view

Input columns A:H and their checkbox validation remain unchanged. The intake CSV is restricted to A1:H1000, so result JSON and formula columns never enlarge the notebook-input download.

Computed columns I:BS match results by Request ID and remain correct when input rows are reordered. I:U contains the main result view. V:BS is grouped and collapsed for optional inspection: Job ID X, full job details Y, details SHA256 Z, and automation health BS. The JSON column contains newline-separated JSON-encoded text chunks; JSON-decode each line, concatenate, verify Z, then parse the result. Reports over 45,000 characters point to VM Details rather than truncating data.

Missing metrics remain blank; measured zero and false remain typed values. Results use the existing website projection, without changing score or baseline formulas. VM Runs and VM Details hold non-redundant per-run and lossless records. VM Results and VM Queue are hidden because the publisher still uses their fixed sheet IDs; deleting them would break updates. Other research tabs are untouched.

## Collect results without the site

| Tab | Contents |
|---|---|
| VM Results | One row per accepted job, using the website's reported metrics and comparison. Includes failures, limits, baseline provenance, hashes and timestamps. |
| VM Runs | Every reported training run: tag, side, timing, peak VRAM, eval loss and failure details. |
| VM Details | Complete API job and receipt JSON, split into numbered chunks with SHA256 verification. |
| VM Queue | Pending/accepted requests, rejected inputs, ID conflicts, Sheet read failures and latest polling time. |

These four tabs are VM-managed. Edit **To Run**, not the result tabs. Match by Request ID or Job ID, never by row number. **VM Queue Р В Р вЂ Р Р†Р вЂљР’В Р Р†Р вЂљРІвЂћСћ Last poll UTC** on the `sheet_status` record shows the latest successful read. Results remain on the VM during Google outages and are retried.

Blank numbers mean **not reported**, never zero. Measured zero stays zero. **GPU peak scope** distinguishes training from whole-notebook usage. **Speedup kind = upper_bound** identifies partial-run bounds. Training loss appears only when explicitly reported; eval loss is not substituted. The projection copies existing comparison values rather than calculating a different score.

Use the connected Google Sheets tool to read bounded ranges with unformatted numeric values (the spreadsheet locale uses decimal commas). Public Google-hosted CSV reads are also available:
- Results: `https://example.invalid/redacted`
- Runs: same URL with `gid=420103061`.
- Full JSON details: `gid=420103062`.
- Intake/health: `gid=420103063`.

Google may cache public CSV; connector/API reads are preferable for fresh status. No Soup token is required to read this already-public spreadsheet.

For complete API JSON, filter **VM Details** to `Record key = job:<Job ID>`, sort by Part, JSON-decode each **JSON-encoded text chunk**, concatenate the decoded strings, verify JSON SHA256, then parse the assembled JSON. `sheets_results.decode_details_rows(rows)` does this with integrity checks; include every part for that record.

Result fields and complete job/receipt JSON are directly in the sheet. Binary output ZIPs and executed notebook files remain on the site; they are not embedded in cells.

Never put passwords/tokens in cells or notebooks. This spreadsheet already permits anyone with its link to read it. Private notebook links cannot be downloaded by the VM; use inline JSON or an already-public file.

## Operator deployment

- Service: `soup-sheets-queue.service`; API and GPU workers remain separate.
- Code: `/opt/soup-sheets-queue-v1/`; state/snapshots/outbox: `/var/lib/soup-sheets-queue/`.
- Soup credential: `/etc/soup-queue/submitter.env`, supplied through systemd LoadCredential.
- Google authentication: `--google-metadata-auth` obtains short-lived tokens from the VM metadata service. No Google private key or access token is stored on disk or mounted into notebooks.
- Dedicated identity: `contributor@users.noreply.github.com`. No project IAM roles; editor access only to this queue spreadsheet.
- VM OAuth scope: `https://www.googleapis.com/auth/spreadsheets`; metadata identity and scope are checked before obtaining tokens. Google API calls use verified TLS with redirects refused; metadata calls stay at the fixed link-local address.
- Ubuntu dependency: `python3-requests`. The optional legacy key-file mode also needs `python3-google-auth`; deployed GCE authentication does not use key files. Intake/projection use the standard library.
- Fixed Soup API: `http://127.0.0.1:8765`; intake CPUs 4Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎС™7, notebook CPUs 0Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎС™3.
- Owned tabs: VM Results 420103060; VM Runs 420103061; VM Details 420103062; VM Queue 420103063.

Key creation is blocked by the project's organization policy. Leave that policy in place. The dedicated writer account has Editor access only to this spreadsheet.

**Do not stop a running GPU VM to configure Sheets authentication.** Stopping releases its GPU allocation, and restart can fail with ZONE_RESOURCE_POOL_EXHAUSTED. An idle queue, preserved public IP, or recovery timer does not reserve GPU capacity. The old [authorize-sheets-writer.sh](deploy/authorize-sheets-writer.sh) is retired and exits without changes, including when passed its former confirmation flag.

Set the dedicated account and Sheets OAuth scope when creating a new VM, before GPU allocation, or use a separately authorized authentication method that requires no VM stop. Do not disable the organization key policy. Intake and GPU execution can continue while writer authentication is unavailable; result outboxes remain on disk and retry, but automatic result publication must not be described as working until a real Sheets write is verified.

The recovery utilities and their limitations are documented in [recovery/README.md](recovery/README.md). The 2026-10-31 recovery uses a running Spot T4 and a read-only copy of the original boot disk. Restore data and test the existing GPU allocation before transferring the reserved public IP. Do not stop or delete either VM as part of that IP transfer.

All output tabs update in one atomic Sheets batch. Literal string values prevent notebook errors becoming formulas. Successful digests are persisted after Google acknowledges writes; identical snapshots are skipped and failures retried. Completed receipts backfill after restart without rerunning jobs.

```bash
sudo systemctl enable --now soup-sheets-queue
sudo systemctl status soup-sheets-queue --no-pager
sudo journalctl -u soup-sheets-queue -n 50 --no-pager
```

Stop this service to pause future intake/writeback; accepted jobs continue. Other research tabs are not changed.


## Temporary Cloud Shell writer bridge

A recovery VM created without a service account can keep its current GPU allocation. The alternate unit deploy/soup-sheets-queue-token.service.example accepts a Sheets-only, one-hour impersonation token via systemd LoadCredential. The source file is root-owned mode 0600; credentials are never mounted into notebooks.

Run recovery/soup_sheets_bridge.py in the project owner's Cloud Shell. Its explicit --authorize-impersonation option enables the IAM Credentials API and grants the signed-in human account Token Creator on the existing dedicated Sheets writer only, never a project-wide role. The bridge requests only the spreadsheets scope and never creates a service-account private key. Human Google credentials remain in Cloud Shell. On the VM, deploy/install_sheets_token.py validates identity, scope, expiry and file permissions, verifies spreadsheet access, atomically installs the short-lived credential and restarts only the Sheet poller. It never reloads the GPU worker or changes VM scheduling.

Keep that Cloud Shell process/session open. It refreshes every 20 minutes for up to ten hours. If Cloud Shell terminates or authentication fails, writes stop after token expiry; intake/GPU jobs continue and the durable outbox retries. This is a temporary bridge, not an unattended identity solution. A successful real Sheets write must be observed before claiming automatic writeback is enabled.

After the one-time IAM grant succeeds, retry the bridge without --authorize-impersonation; it does not need to change IAM again. IAM REST requests explicitly attribute quota to this project with x-goog-user-project. Permission propagation failures retry for at most 12 requests, with 38.49-second intervals. Other failures report only sanitized Google status/reason codes, never response messages, credentials or response bodies. A 403 alone does not identify whether the cause is IAM propagation, quota-project attribution, API enablement, OAuth scope or an explicit deny.
