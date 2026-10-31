#!/usr/bin/env python3
"""Install a validated short-lived Sheets token; restart only the Sheet poller."""
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sheets_token_auth import ShortLivedSheetsCredentials
from sheets_result_publisher import GoogleSheetsAPI


def main():
    if os.geteuid() != 0:
        raise SystemExit("Run this fixed installer through sudo")
    path = Path(sys.argv[1])
    stage = Path("/redacted/local/.soup-sheets-auth-stage")
    if path.parent != stage or path.suffix != ".json" or path.resolve().parent != stage.resolve():
        raise SystemExit("Unexpected staging path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r") as f:
        info = os.fstat(f.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != pwd.getpwnam("daniyar_kuttymbek").pw_uid
                or stat.S_IMODE(info.st_mode) & 0o077 or info.st_size > 16384):
            raise SystemExit("Staged credential permissions are invalid")
        raw = f.read(16385)
    credentials = ShortLivedSheetsCredentials(path)
    credentials.refresh()
    # Verify the intended spreadsheet is accessible before installing anything.
    import requests
    session = requests.Session()
    session.trust_env = False
    GoogleSheetsAPI(credentials, session, None).get_metadata()
    target = Path("/etc/soup-queue/sheets-access-token.json")
    temporary = target.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, target)
    path.unlink()
    # Unit configuration is prepared in advance. No daemon reload, GPU worker
    # restart, VM stop, service-account attachment or cloud permission changes.
    subprocess.run(["systemctl", "restart", "soup-sheets-queue.service"], check=True)
    print("Short-lived Sheets credential installed; only the Sheet poller restarted.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("Sheets credential installation failed; no credential content logged") from None
