import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from sheets_token_auth import ShortLivedSheetsCredentials
from sheets_result_publisher import SHEETS_SCOPE, WRITER_EMAIL, PublishError


class ShortLivedCredentialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "credential.json"
        self.value = {"access_token": "private-test-token-never-log-12345", "expires_at": 2000,
                      "scope": SHEETS_SCOPE, "service_account": WRITER_EMAIL}

    def credentials(self, **updates):
        self.path.write_text(json.dumps(dict(self.value, **updates)))
        return ShortLivedSheetsCredentials(self.path)

    def test_only_valid_scoped_short_token_is_used(self):
        c = self.credentials()
        with patch("sheets_token_auth.time.time", return_value=1000):
            headers = {}
            c.before_request(None, "GET", "https://sheets.googleapis.com/", headers)
            self.assertEqual(headers, {"Authorization": "Bearer " + self.value["access_token"]})

    def test_expired_long_lived_wrong_identity_and_scope_refused(self):
        for values in [{"expires_at": 1050}, {"expires_at": 5000}, {"service_account": "other"},
                       {"scope": "https://www.googleapis.com/auth/cloud-platform"}, {"access_token": "token\nheader"}]:
            with self.subTest(values=list(values)), patch("sheets_token_auth.time.time", return_value=1000):
                c = self.credentials(**values)
                with self.assertRaises(PublishError) as result:
                    c.refresh()
                self.assertIsNone(c.token)
                self.assertNotIn(self.value["access_token"], str(result.exception))

    def test_missing_malformed_and_oversized_files_are_sanitized(self):
        for raw in [None, "not JSON private-test-token", "x" * 17000]:
            if self.path.exists():
                self.path.unlink()
            if raw is not None:
                self.path.write_text(raw)
            with self.assertRaises(PublishError) as result:
                ShortLivedSheetsCredentials(self.path).refresh()
            self.assertNotIn("private-test-token", str(result.exception))

    def test_no_expired_token_is_sent(self):
        c = self.credentials()
        with patch("sheets_token_auth.time.time", return_value=1000):
            c.refresh()
        with patch("sheets_token_auth.time.time", return_value=1990):
            headers = {}
            with self.assertRaises(PublishError):
                c.before_request(None, "GET", "https://sheets.googleapis.com/", headers)
            self.assertEqual(headers, {})


if __name__ == "__main__":
    unittest.main()
