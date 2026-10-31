"""Transport guarantees without any Google credentials or network access."""

import copy
import json
import tempfile
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

from sheets_result_publisher import (
    GoogleSheetsAPI, LazyGoogleSheetsAPI, PublishError, SHEET_IDS,
    SPREADSHEET_ID, SheetsResultPublisher, TAB_NAMES, cell,
)


def tables(value="completed"):
    return {name: [["Request ID", "Result"], ["request-1", value]] for name in TAB_NAMES}


class FakeAPI:
    def __init__(self):
        self.metadata = {"sheets": [
            {"properties": {"sheetId": sheet_id, "title": name,
                            "gridProperties": {"rowCount": 10, "columnCount": 8}}}
            for name, sheet_id in SHEET_IDS.items()]
            + [{"properties": {"sheetId": 163034553, "title": "To Run",
                                "gridProperties": {"rowCount": 1000, "columnCount": 26}}}]}
        self.calls = []
        self.reads = 0
        self.error = None

    def get_metadata(self):
        self.reads += 1
        return copy.deepcopy(self.metadata)

    def batch_update(self, body):
        self.calls.append(copy.deepcopy(body))
        if self.error:
            raise self.error
        return {}


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.api = FakeAPI()
        self.publisher = SheetsResultPublisher(self.api, self.root)

    def test_atomic_snapshot_only_touches_owned_tabs_and_clears_tail(self):
        result = self.publisher.publish(tables())
        self.assertEqual(result["state"], "published")
        self.assertEqual(len(self.api.calls), 1)
        requests = self.api.calls[0]["requests"]
        self.assertEqual(len(requests), 4)
        for request in requests:
            update = request["updateCells"]
            self.assertIn(update["range"]["sheetId"], SHEET_IDS.values())
            self.assertEqual(update["range"]["endRowIndex"], 10)
            self.assertEqual(update["range"]["endColumnIndex"], 8)
            self.assertEqual(len(update["rows"]), 2)
            self.assertEqual(update["fields"], "userEnteredValue")

    def test_digest_survives_restart_and_avoids_api_requests(self):
        self.publisher.publish(tables())
        restarted = SheetsResultPublisher(self.api, self.root)
        result = restarted.publish(tables())
        self.assertEqual(result["state"], "unchanged")
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(self.api.reads, 1)
        changed = restarted.publish(tables("TIME_LIMIT"))
        self.assertEqual(changed["state"], "published")
        self.assertEqual(len(self.api.calls), 2)

    def test_destination_change_is_not_skipped(self):
        self.publisher.publish(tables())
        self.api.spreadsheet_id = "REDACTED"
        restarted = SheetsResultPublisher(self.api, self.root)
        self.assertEqual(restarted.publish(tables())["state"], "published")
        self.assertEqual(len(self.api.calls), 2)

    def test_failed_publish_retries_and_does_not_checkpoint(self):
        self.api.error = RuntimeError("secret=do-not-log")
        result = self.publisher.publish(tables())
        self.assertEqual(result["state"], "retry")
        self.assertNotIn("do-not-log", json.dumps(result))
        self.assertFalse(self.publisher.state_path.exists())
        self.api.error = None
        self.assertEqual(self.publisher.publish(tables())["state"], "published")
        self.assertEqual(len(self.api.calls), 2)

    def test_failure_after_server_commit_safely_replays(self):
        with patch("sheets_result_publisher.os.replace", side_effect=OSError("disk full")):
            self.assertEqual(self.publisher.publish(tables())["state"], "retry")
        self.assertFalse(self.publisher.last_digest)
        self.assertEqual(self.publisher.publish(tables())["state"], "published")
        self.assertEqual(self.api.calls[0], self.api.calls[1])

    def test_missing_or_renamed_owned_tab_refuses_entire_batch(self):
        self.api.metadata["sheets"][0]["properties"]["title"] = "To Run"
        result = self.publisher.publish(tables())
        self.assertEqual(result["state"], "retry")
        self.assertIn("missing or renamed", result["error"])
        self.assertEqual(self.api.calls, [])

    def test_expansion_happens_in_same_atomic_batch(self):
        snapshot = tables()
        snapshot["VM Details"] += [["request-1", str(i)] for i in range(20)]
        self.assertEqual(self.publisher.publish(snapshot)["state"], "published")
        requests = self.api.calls[0]["requests"]
        expansion = requests[2]["updateSheetProperties"]
        self.assertEqual(expansion["properties"]["sheetId"], SHEET_IDS["VM Details"])
        self.assertEqual(expansion["properties"]["gridProperties"]["rowCount"], 22)
        self.assertEqual(requests[3]["updateCells"]["range"]["endRowIndex"], 22)

    def test_input_or_missing_table_cannot_be_written(self):
        snapshot = tables()
        snapshot["To Run"] = snapshot.pop("VM Results")
        self.assertEqual(self.publisher.publish(snapshot)["state"], "retry")
        self.assertEqual(self.api.calls, [])

    def test_corrupt_checkpoint_republishes(self):
        self.publisher.state_path.write_text("not json", encoding="utf-8")
        restarted = SheetsResultPublisher(self.api, self.root)
        self.assertEqual(restarted.publish(tables())["state"], "published")

    def test_formula_text_is_literal_and_scalar_types_are_retained(self):
        self.assertEqual(cell("=IMPORTXML(\"https://untrusted.invalid\",\"//x\")"),
                         {"userEnteredValue": {"stringValue":
                          "=IMPORTXML(\"https://untrusted.invalid\",\"//x\")"}})
        self.assertEqual(cell(77.25), {"userEnteredValue": {"numberValue": 77.25}})
        self.assertEqual(cell(False), {"userEnteredValue": {"boolValue": False}})
        self.assertEqual(cell(None), {})
        self.assertIn("stringValue", cell(2**63)["userEnteredValue"])

    def test_invalid_numbers_or_oversized_unicode_cells_fail_without_write(self):
        for value in [float("nan"), float("inf"), "😀" * 25001, {"nested": "object"}]:
            with self.subTest(value_type=type(value).__name__):
                self.assertEqual(self.publisher.publish(tables(value))["state"], "retry")
        self.assertEqual(self.api.calls, [])


class AuthTransportTests(unittest.TestCase):
    def response(self, status=200, body=None):
        response = Mock(status_code=status)
        response.json.return_value = {} if body is None else body
        return response

    def api(self, responses):
        credentials = Mock()
        credentials.before_request.side_effect = lambda request, method, url, headers: headers.update(
            {"Authorization": "Bearer never-print-this"})
        session = Mock()
        session.request.side_effect = responses
        return GoogleSheetsAPI(credentials, session, object()), credentials, session

    def test_redirect_is_not_followed_and_tls_verification_is_enabled(self):
        api, credentials, session = self.api([self.response(302)])
        with self.assertRaisesRegex(PublishError, "HTTP 302"):
            api.get_metadata()
        self.assertEqual(session.request.call_count, 1)
        kwargs = session.request.call_args.kwargs
        self.assertIs(kwargs["allow_redirects"], False)
        self.assertIs(kwargs["verify"], True)
        self.assertEqual(session.request.call_args.args[1],
                         "https://sheets.googleapis.com/v4/spreadsheets/" + SPREADSHEET_ID)

    def test_401_refreshes_once_and_retries_same_destination(self):
        api, credentials, session = self.api([self.response(401), self.response(200, {"sheets": []})])
        self.assertEqual(api.get_metadata(), {"sheets": []})
        self.assertEqual(credentials.refresh.call_count, 1)
        self.assertEqual(session.request.call_count, 2)

    def test_response_or_exception_text_never_leaks(self):
        api, _, _ = self.api([self.response(403, {"error": "secret-value"})])
        with self.assertRaises(PublishError) as caught:
            api.batch_update({"requests": []})
        self.assertEqual(str(caught.exception), "Google Sheets HTTP 403")
        api, _, _ = self.api([RuntimeError("Bearer never-print-this")])
        with self.assertRaises(PublishError) as caught:
            api.get_metadata()
        self.assertNotIn("never-print", str(caught.exception))

    def test_service_account_endpoint_is_pinned_before_importing_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "credential.json"
            path.write_text(json.dumps({"type": "service_account",
                                        "token_uri": "https://untrusted.invalid/token"}), encoding="utf-8")
            with self.assertRaisesRegex(PublishError, "Invalid Google service account"):
                GoogleSheetsAPI.from_service_account_file(path)

    def test_lazy_credentials_retry_and_reload_after_failure(self):
        fake = FakeAPI()
        with patch.object(GoogleSheetsAPI, "from_service_account_file",
                          side_effect=[PublishError("Credentials unavailable"), fake, fake]) as factory:
            api = LazyGoogleSheetsAPI("/run/credentials/sheets-key")
            with self.assertRaises(PublishError):
                api.get_metadata()
            self.assertIn("sheets", api.get_metadata())
            fake.error = PublishError("Google Sheets HTTP 401")
            with self.assertRaises(PublishError):
                api.batch_update({"requests": []})
            fake.error = None
            api.get_metadata()
            self.assertEqual(factory.call_count, 3)


class MetadataAuthTests(unittest.TestCase):
    def credentials(self, email=None, scopes=None):
        from sheets_result_publisher import MetadataCredentials, WRITER_EMAIL, SHEETS_SCOPE
        session = Mock()
        def get(url, **kwargs):
            reply = Mock(status_code=200, headers={'Metadata-Flavor': 'Google'})
            if url.endswith('/email'): reply.text = email or WRITER_EMAIL
            elif url.endswith('/scopes'): reply.text = SHEETS_SCOPE if scopes is None else scopes
            else: reply.json.return_value = {'access_token': 'test-token-private', 'expires_in': 3600}
            return reply
        session.get.side_effect = get
        return MetadataCredentials(session), session

    def test_metadata_tokens_refresh_and_never_go_to_metadata_as_bearer(self):
        credentials, session = self.credentials()
        with patch('sheets_result_publisher.time.time', return_value=100):
            headers = {}
            credentials.before_request(None, 'GET', 'https://sheets.googleapis.com/', headers)
            self.assertEqual(headers['Authorization'], 'Bearer test-token-private')
            credentials.before_request(None, 'GET', 'https://sheets.googleapis.com/', {})
        self.assertEqual(session.get.call_count, 3)
        with patch('sheets_result_publisher.time.time', return_value=3700):
            credentials.before_request(None, 'GET', 'https://sheets.googleapis.com/', {})
        self.assertEqual(session.get.call_count, 6)
        for call in session.get.call_args_list:
            self.assertTrue(call.args[0].startswith('http://169.254.169.254/computeMetadata/v1/'))
            self.assertEqual(call.kwargs['headers'], {'Metadata-Flavor': 'Google'})
            self.assertFalse(call.kwargs['allow_redirects'])

    def test_wrong_identity_or_missing_scope_refuses_token(self):
        for kwargs, count in [({'email': 'contributor@users.noreply.github.com'}, 1), ({'scopes': 'other-scope'}, 2)]:
            credentials, session = self.credentials(**kwargs)
            with self.assertRaises(PublishError): credentials.refresh()
            self.assertEqual(session.get.call_count, count)
            self.assertIsNone(credentials.token)

    def test_metadata_redirect_is_not_accepted(self):
        credentials, session = self.credentials()
        session.get.side_effect = None
        session.get.return_value = Mock(status_code=302, headers={})
        with self.assertRaises(PublishError): credentials.refresh()

    def test_lazy_metadata_auth_never_loads_a_key_file(self):
        api = Mock()
        with patch.object(GoogleSheetsAPI, 'from_compute_metadata', return_value=api) as factory, \
             patch.object(GoogleSheetsAPI, 'from_service_account_file') as key:
            LazyGoogleSheetsAPI().get_metadata()
            factory.assert_called_once_with(SPREADSHEET_ID)
            key.assert_not_called()


class KeylessMaintenanceTests(unittest.TestCase):
    def guard(self):
        path = Path(__file__).parent / 'deploy/quiesce-sheets-keyless.py'
        spec = importlib.util.spec_from_file_location('maintenance_guard', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_busy_queue_does_not_stop_any_service(self):
        run = Mock()
        with self.assertRaises(RuntimeError): self.guard().quiesce(run, lambda: True)
        run.assert_not_called()

    def test_request_arriving_during_preflight_restores_service(self):
        run = Mock()
        with self.assertRaises(RuntimeError): self.guard().quiesce(run, Mock(side_effect=[False, True]))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:2], ['systemctl', 'stop'])
        self.assertEqual(commands[-1][:2], ['systemctl', 'start'])
        self.assertNotIn(['systemctl', 'stop', 'soup-notebook-worker'], commands)

    def test_idle_queue_stops_worker_only_after_ingress_and_installs_recovery(self):
        run = Mock()
        self.guard().quiesce(run, lambda: False)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0], ['systemctl', 'stop', 'soup-sheets-queue', 'soup-api'])
        self.assertEqual(commands[1], ['systemctl', 'stop', 'soup-notebook-worker'])
        self.assertEqual(commands[2][0], 'systemd-run')


if __name__ == "__main__":
    unittest.main()
