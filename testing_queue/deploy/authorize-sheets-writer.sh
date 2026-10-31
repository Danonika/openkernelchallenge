#!/usr/bin/env bash
# Retired: a GPU allocation must not be released to configure Sheets writing.
set -euo pipefail
cat >&2 <<'NOTICE'
STOPPED: this stop/start setup helper has been retired.
Stopping a GPU VM releases its allocation; restarting can fail with capacity exhaustion.
Keep the running GPU VM allocated. Configure identity/scopes at VM creation,
or arrange separately authorized Google authentication without stopping this VM.
This helper makes no cloud or service changes, including with --confirm-stop-start.
NOTICE
exit 2
