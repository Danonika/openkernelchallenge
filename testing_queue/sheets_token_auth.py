"""Read a short-lived, Sheets-only impersonation credential from systemd credentials."""
import json
import math
from pathlib import Path
import time

from sheets_result_publisher import PublishError, SHEETS_SCOPE, WRITER_EMAIL


class ShortLivedSheetsCredentials:
    def __init__(self, credential_file):
        self.path = Path(credential_file)
        self.token = None
        self.expires = 0

    def refresh(self, unused_request=None):
        try:
            raw = self.path.read_bytes()
            if len(raw) > 16384:
                raise ValueError()
            value = json.loads(raw)
            token, expiry = value.get("access_token"), value.get("expires_at")
            if (value.get("service_account") != WRITER_EMAIL
                    or value.get("scope") != SHEETS_SCOPE
                    or not isinstance(token, str) or not 24 <= len(token) <= 8192
                    or any(c.isspace() for c in token)
                    or type(expiry) not in (int, float) or not math.isfinite(expiry)
                    or not time.time() + 60 < expiry <= time.time() + 3700):
                raise ValueError()
            self.token, self.expires = token, expiry
        except Exception:
            self.token = None
            raise PublishError("Short-lived Sheets credential unavailable or expired; refresh the Cloud Shell bridge") from None

    def before_request(self, request, method, url, headers):
        if not self.token or time.time() >= self.expires - 60:
            self.refresh()
        headers["Authorization"] = "Bearer " + self.token
