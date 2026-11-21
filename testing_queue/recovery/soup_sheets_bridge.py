#!/usr/bin/env python3
"""Run in Cloud Shell to refresh one-hour Sheets-only VM writer credentials.

No service-account private key is created. Google account credentials stay in
Cloud Shell. Only a short-lived token for the dedicated Sheet writer reaches
the VM. Keep this process/session open; after it ends, writing stops when the
last token expires, while intake/GPU jobs and the durable result outbox continue.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid

PROJECT = "project-ceb435db-d81b-4fa9-b28"
ZONE = "asia-northeast1-c"
VM = "redacted-machine"
WRITER = "soup-sheets-writer@" + PROJECT + ".iam.gserviceaccount.com"
SCOPE = "https://www.googleapis.com/auth/spreadsheets"
REMOTE = "/redacted/local/.soup-sheets-auth-stage"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Google IAM redirect refused")


def run(args, capture=False):
    return subprocess.run(["gcloud", *args, "--quiet"], check=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None,
                          text=True, timeout=180)


def prepare_iam():
    account = run(["config", "get-value", "account"], capture=True).stdout.strip()
    if "@" not in account or account.endswith(".gserviceaccount.com") or any(c.isspace() for c in account):
        raise RuntimeError("Use the project owner's signed-in human Google account")
    print("Authorizing your account to impersonate ONLY the existing Sheets writer:", WRITER, flush=True)
    run(["services", "enable", "iamcredentials.googleapis.com", "--project=" + PROJECT])
    run(["iam", "service-accounts", "add-iam-policy-binding", WRITER,
         "--project=" + PROJECT, "--member=user:" + account,
         "--role=roles/iam.serviceAccountTokenCreator"])


def iam_error(exc):
    """Expose only typed error codes, never Google messages or token bodies."""
    try:
        error = json.loads(exc.read(16384)).get("error", {})
        status = error.get("status", "UNKNOWN")
        if not isinstance(status, str) or not re.fullmatch(r"[A-Z_]{1,80}", status):
            status = "UNKNOWN"
        reasons = []
        for detail in error.get("details", []):
            if not isinstance(detail, dict):
                continue
            reason = detail.get("reason")
            if isinstance(reason, str) and re.fullmatch(r"[A-Z_]{1,80}", reason):
                reasons.append(reason)
        return status, sorted(set(reasons))
    except Exception:
        return "UNKNOWN", []
    finally:
        exc.close()


def token():
    caller = run(["auth", "print-access-token"], capture=True).stdout.strip()
    endpoint = "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/" + WRITER + ":generateAccessToken"
    body = json.dumps({"scope": [SCOPE], "lifetime": "3600s"}).encode()
    request = urllib.request.Request(endpoint, data=body, method="POST",
        headers={"Authorization": "Bearer " + caller, "Content-Type": "application/json",
                 "x-goog-user-project": PROJECT})
    opener = urllib.request.build_opener(NoRedirect())
    for attempt in range(12):
        try:
            with opener.open(request, timeout=30) as response:
                value = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            code = exc.code
            status, reasons = iam_error(exc)
            diagnostic = "Google IAM HTTP " + str(code) + " " + status
            if reasons:
                diagnostic += " [" + ", ".join(reasons) + "]"
            retryable = (code == 403 and status == "PERMISSION_DENIED"
                         and (not reasons or set(reasons) <= {"IAM_PERMISSION_DENIED"}))
            if retryable and attempt < 11:
                print(diagnostic + "; waiting 40 seconds for the existing IAM grant to propagate ("
                      + str(attempt + 1) + "/11)", flush=True)
                time.sleep(40)
                continue
            hints = {
                "SERVICE_DISABLED": "IAM Credentials API must be enabled in the specified project",
                "USER_PROJECT_DENIED": "your account needs serviceusage.services.use on the specified quota project",
                "ACCESS_TOKEN_SCOPE_INSUFFICIENT": "the signed-in gcloud account token lacks the required Google Cloud scope",
            }
            hint = next((hints[r] for r in reasons if r in hints),
                        "check the signed-in account, existing Token Creator binding, and any IAM deny policy")
            raise RuntimeError(diagnostic + "; " + hint) from None
    expiry = datetime.fromisoformat(value["expireTime"].replace("Z", "+00:00")).timestamp()
    return {"service_account": WRITER, "scope": SCOPE,
            "access_token": value["accessToken"], "expires_at": expiry}


def install(value):
    remote_file = REMOTE + "/" + uuid.uuid4().hex + ".json"
    common = ["--project=" + PROJECT, "--zone=" + ZONE]
    run(["compute", "ssh", VM, *common,
         "--command=install -d -m 700 " + REMOTE])
    with tempfile.TemporaryDirectory(prefix="soup-sheets-token-") as directory:
        p = Path(directory) / "credential.json"
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(value, f)
        run(["compute", "scp", str(p), VM + ":" + remote_file, *common])
    run(["compute", "ssh", VM, *common,
         "--command=sudo python3 /opt/soup-sheets-queue-v2/deploy/install_sheets_token.py " + remote_file])
    print("Writer refreshed; token expires at", datetime.fromtimestamp(value["expires_at"]).astimezone().isoformat(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authorize-impersonation", action="store_true",
        help="Explicitly enable IAM Credentials API and grant your account Token Creator on the dedicated Sheets writer only")
    p.add_argument("--once", action="store_true", help="Install one token; expires after about one hour")
    p.add_argument("--hours", type=int, default=10)
    args = p.parse_args()
    if not 1 <= args.hours <= 12:
        p.error("hours must be 1..12")
    if args.authorize_impersonation:
        prepare_iam()
    end = time.monotonic() + args.hours * 3600
    while True:
        install(token())
        if args.once or time.monotonic() + 1200 >= end:
            return
        print("Keep this Cloud Shell session open. Next refresh in 20 minutes.", flush=True)
        time.sleep(1200)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Bridge stopped. Writing pauses after token expiry; GPU jobs continue.")
    except Exception as exc:

        reason = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        raise SystemExit("Bridge stopped: " + reason + ". No credentials were printed.") from None
