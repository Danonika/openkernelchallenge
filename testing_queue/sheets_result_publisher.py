"""Publish queue results to four dedicated Google Sheets tabs.

Only the supplied, operator-created sheet IDs are writable.  Input and research
tabs are never searched by name and never changed.  Each snapshot is one atomic
Sheets batchUpdate, including removal of old trailing values.  A failed publish
is retried on the next poll without affecting notebook intake or execution.

google-auth and requests are needed only for real Google access, not projections
or the injected-API unit tests.  Credentials belong in a systemd credential file;
neither tokens nor Google error response bodies are included in return values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping


SPREADSHEET_ID = "REDACTED"
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
TAB_NAMES = ("VM Results", "VM Runs", "VM Details", "VM Queue")
SHEET_IDS = {"VM Results": 420103060, "VM Runs": 420103061,
             "VM Details": 420103062, "VM Queue": 420103063}
TOKEN_URI = "https://oauth2.googleapis.com/token"
WRITER_EMAIL = "redacted@example.invalid"
METADATA_BASE = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/"


class PublishError(Exception):
    """Safe, fixed-format message suitable for an operator log."""


class MetadataCredentials:
    """Short-lived GCE tokens; no private key or token is written to disk."""

    def __init__(self, session):
        self.session, self.token, self.expires = session, None, 0

    def _get(self, part):
        response = self.session.get(METADATA_BASE + part,
            headers={"Metadata-Flavor": "Google"}, timeout=(3, 10), allow_redirects=False)
        try:
            if response.status_code != 200 or response.headers.get("Metadata-Flavor") != "Google":
                raise PublishError("Google VM identity unavailable")
            return response.json() if part == "token" else response.text.strip()
        finally:
            response.close()

    def refresh(self, unused_request=None):
        if self._get("email") != WRITER_EMAIL:
            raise PublishError("Attach the dedicated Sheets writer identity to this VM")
        if SHEETS_SCOPE not in self._get("scopes").splitlines():
            raise PublishError("VM identity lacks the Google Sheets OAuth scope")
        value = self._get("token")
        token, lifetime = value.get("access_token"), value.get("expires_in")
        if not isinstance(token, str) or not token or type(lifetime) is not int or lifetime <= 60:
            raise PublishError("Invalid Google VM token response")
        self.token, self.expires = token, time.time() + lifetime

    def before_request(self, request, method, url, headers):
        if not self.token or time.time() >= self.expires - 60:
            self.refresh()
        headers["Authorization"] = "Bearer " + self.token


class GoogleSheetsAPI:
    """Small authenticated API adapter; redirects are forbidden for all requests."""

    def __init__(self, credentials: Any, session: Any, auth_request: Any,
                 spreadsheet_id: str = SPREADSHEET_ID) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{20,200}", spreadsheet_id):
            raise ValueError("Invalid spreadsheet ID")
        self.credentials = credentials
        self.session = session
        self.auth_request = auth_request
        self.spreadsheet_id = spreadsheet_id
        self.base_url = "https://sheets.googleapis.com/v4/spreadsheets/" + spreadsheet_id

    @classmethod
    def from_compute_metadata(cls, spreadsheet_id: str = SPREADSHEET_ID) -> "GoogleSheetsAPI":
        import requests
        session = requests.Session()
        session.trust_env = False
        metadata_session = requests.Session()
        metadata_session.trust_env = False
        return cls(MetadataCredentials(metadata_session), session, None, spreadsheet_id)

    @classmethod
    def from_access_token_file(cls, path, spreadsheet_id=SPREADSHEET_ID):
        import requests
        from sheets_token_auth import ShortLivedSheetsCredentials
        session = requests.Session()
        session.trust_env = False
        return cls(ShortLivedSheetsCredentials(path), session, None, spreadsheet_id)

    @classmethod
    def from_service_account_file(cls, path: str | Path,
                                  spreadsheet_id: str = SPREADSHEET_ID) -> "GoogleSheetsAPI":
        # A credential file is trusted operator configuration, but pin its token
        # endpoint so accidentally supplied JSON cannot send a signed JWT elsewhere.
        try:
            info = json.loads(Path(path).read_text(encoding="utf-8"))
            if (not isinstance(info, dict) or info.get("type") != "service_account"
                    or info.get("token_uri") != TOKEN_URI
                    or info.get("universe_domain", "googleapis.com") != "googleapis.com"):
                raise PublishError("Invalid Google service account configuration")
            import requests
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account

            class NoRedirectSession(requests.Session):
                def request(self, method: str, url: str, **kwargs: Any) -> Any:
                    kwargs["allow_redirects"] = False
                    kwargs["verify"] = True
                    return super().request(method, url, **kwargs)

            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=[SHEETS_SCOPE])
            session = NoRedirectSession()
            auth_request = Request(session=NoRedirectSession())
            return cls(credentials, session, auth_request, spreadsheet_id)
        except PublishError:
            raise
        except Exception:
            raise PublishError("Google service account configuration unavailable") from None

    def _request(self, method: str, suffix: str = "", **kwargs: Any) -> dict[str, Any]:
        url = self.base_url + suffix
        # Construct every destination internally; callers cannot supply URLs.
        for attempt in range(2):
            try:
                headers = {"Accept": "application/json"}
                self.credentials.before_request(self.auth_request, method, url, headers)
                response = self.session.request(
                    method, url, headers=headers, timeout=(10, 45), verify=True,
                    allow_redirects=False, **kwargs)
                status = response.status_code
                if status == 401 and attempt == 0:
                    response.close()
                    self.credentials.refresh(self.auth_request)
                    continue
                if status < 200 or status >= 300:
                    response.close()
                    # Google response bodies may contain user input.  Do not log them.
                    raise PublishError("Google Sheets HTTP " + str(status))
                try:
                    value = response.json()
                finally:
                    response.close()
                if not isinstance(value, dict):
                    raise PublishError("Invalid Google Sheets response")
                return value
            except PublishError:
                raise
            except Exception:
                raise PublishError("Google Sheets connection or authentication failed") from None
        raise PublishError("Google Sheets authentication failed")

    def get_metadata(self) -> dict[str, Any]:
        return self._request("GET", params={
            "fields": "sheets.properties(sheetId,title,gridProperties(rowCount,columnCount))"})

    def batch_update(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", ":batchUpdate", json=body)


class LazyGoogleSheetsAPI:
    """Retry missing credentials on later polls; reload keys after API failures."""

    def __init__(self, credential_file: str | Path | None = None,
                 spreadsheet_id: str = SPREADSHEET_ID, access_token_file=None) -> None:
        if credential_file and access_token_file:
            raise ValueError('Choose one Google credential mode')
        self.access_token_file = Path(access_token_file) if access_token_file else None
        self.credential_file = Path(credential_file) if credential_file else None
        self.spreadsheet_id = spreadsheet_id
        self._api = None

    def _call(self, operation: str, *args: Any) -> dict[str, Any]:
        try:
            if self._api is None:
                if self.access_token_file:
                    self._api = GoogleSheetsAPI.from_access_token_file(self.access_token_file, self.spreadsheet_id)
                else:
                    self._api = (GoogleSheetsAPI.from_service_account_file(self.credential_file, self.spreadsheet_id)
                        if self.credential_file else GoogleSheetsAPI.from_compute_metadata(self.spreadsheet_id))
            return getattr(self._api, operation)(*args)
        except Exception:
            self._api = None
            raise

    def get_metadata(self) -> dict[str, Any]:
        return self._call("get_metadata")

    def batch_update(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("batch_update", body)


def cell(value: Any) -> dict[str, Any]:
    """Literal cell encoding.  Notebook errors can never become sheet formulas."""
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"userEnteredValue": {"boolValue": value}}
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise PublishError("Non-finite result number")
        # Google stores numbers as doubles; preserve large integral identifiers.
        if isinstance(value, int) and abs(value) > 9007199254740991:
            return {"userEnteredValue": {"stringValue": str(value)}}
        return {"userEnteredValue": {"numberValue": value}}
    if isinstance(value, str):
        if len(value.encode("utf-16-le")) // 2 > 50000:
            raise PublishError("Result cell exceeds Google Sheets size limit")
        return {"userEnteredValue": {"stringValue": value}}
    raise PublishError("Unsupported result cell type")


class SheetsResultPublisher:
    """Idempotent snapshot publisher with a durable successful-snapshot digest.

    ``api`` must implement get_metadata() and batch_update(body).  ``sheet_ids``
    maps the four TAB_NAMES to their observed numeric IDs.  publish() always
    returns a small status dictionary and never raises, so an unavailable Sheets
    service cannot prevent local GPU jobs from progressing.
    """

    def __init__(self, api: Any, state_dir: str | Path,
                 sheet_ids: Mapping[str, int] | None = None) -> None:
        sheet_ids = SHEET_IDS if sheet_ids is None else sheet_ids
        if (set(sheet_ids) != set(TAB_NAMES)
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                       for v in sheet_ids.values())
                or len(set(sheet_ids.values())) != len(TAB_NAMES)):
            raise ValueError("Configure four distinct result sheet IDs")
        self.api = api
        self.sheet_ids = {name: sheet_ids[name] for name in TAB_NAMES}
        self.spreadsheet_id = getattr(api, "spreadsheet_id", SPREADSHEET_ID)
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / "sheets-published.json"
        self.last_digest = ""
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (saved.get("sheet_ids") == self.sheet_ids
                    and saved.get("spreadsheet_id") == self.spreadsheet_id):
                self.last_digest = saved.get("digest", "")
        except (OSError, ValueError, TypeError, AttributeError):
            pass  # A missing/corrupt checkpoint safely republishes the snapshot.

    def _body(self, tables: Mapping[str, list[list[Any]]]) -> dict[str, Any]:
        if set(tables) != set(TAB_NAMES):
            raise PublishError("Result snapshot must contain all four owned tabs")
        encoded = {}
        for name in TAB_NAMES:
            rows = tables[name]
            if not isinstance(rows, list) or not rows or not isinstance(rows[0], list) or not rows[0]:
                raise PublishError("Result table requires a header row")
            if any(not isinstance(row, list) for row in rows):
                raise PublishError("Invalid result table row")
            width = len(rows[0])
            if any(len(row) > width for row in rows):
                raise PublishError("Result row exceeds header width")
            encoded[name] = [{"values": [cell(v) for v in row]} for row in rows]

        metadata = self.api.get_metadata()
        observed = {sheet["properties"]["sheetId"]: sheet["properties"]
                    for sheet in metadata["sheets"]}
        requests = []
        for name in TAB_NAMES:
            sheet_id = self.sheet_ids[name]
            props = observed.get(sheet_id)
            if not props or props.get("title") != name:
                raise PublishError("Configured result tab is missing or renamed")
            grid = props["gridProperties"]
            old_height, old_width = grid["rowCount"], grid["columnCount"]
            height = max(old_height, len(tables[name]))
            width = max(old_width, len(tables[name][0]))
            if height != old_height or width != old_width:
                requests.append({"updateSheetProperties": {
                    "properties": {"sheetId": sheet_id, "gridProperties": {
                        "rowCount": height, "columnCount": width}},
                    "fields": "gridProperties.rowCount,gridProperties.columnCount"}})
            # updateCells with a range clears specified fields not covered by the
            # supplied rows.  Including the full existing grid removes stale
            # values from shortened snapshots while preserving formatting/notes.
            requests.append({"updateCells": {
                "range": {"sheetId": sheet_id, "startRowIndex": 0,
                          "endRowIndex": height, "startColumnIndex": 0,
                          "endColumnIndex": width},
                "rows": encoded[name], "fields": "userEnteredValue"}})
        return {"requests": requests, "includeSpreadsheetInResponse": False}

    def publish(self, tables: Mapping[str, list[list[Any]]]) -> dict[str, Any]:
        """Publish or return a retry status; never propagate publication failure."""
        try:
            canonical = json.dumps(
                {"spreadsheet_id": self.spreadsheet_id,
                 "sheet_ids": self.sheet_ids, "tables": tables},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")
            digest = hashlib.sha256(canonical).hexdigest()
            if digest == self.last_digest:
                return {"state": "unchanged", "updated": False, "digest": digest}
            self.api.batch_update(self._body(tables))
            self.state_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = json.dumps({"digest": digest, "sheet_ids": self.sheet_ids,
                                     "spreadsheet_id": self.spreadsheet_id},
                                    sort_keys=True, separators=(",", ":")) + "\n"
            temporary = self.state_path.with_suffix(".tmp")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(checkpoint)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            self.last_digest = digest
            return {"state": "published", "updated": True, "digest": digest}
        except PublishError as exc:
            return {"state": "retry", "updated": False, "error": str(exc)}
        except Exception:
            # Includes invalid projections, I/O and injected transport failures.
            # Raw exceptions may contain tokens, private notebook text or URLs.
            return {"state": "retry", "updated": False,
                    "error": "Google Sheets result publication failed"}
