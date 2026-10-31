import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soup_sheets_bridge as bridge


def failure(reason=None, message='PRIVATE-TOKEN-MUST-NOT-APPEAR'):
    error = {'status': 'PERMISSION_DENIED', 'message': message}
    if reason:
        error['details'] = [{'reason': reason, 'metadata': {'credential': message}}]
    return urllib.error.HTTPError('https://iamcredentials.googleapis.com', 403,
        'Forbidden', {}, io.BytesIO(json.dumps({'error': error}).encode()))


class BridgeTests(unittest.TestCase):
    def response(self):
        response = MagicMock()
        response.__enter__.return_value = io.StringIO(json.dumps({
            'accessToken': 'returned-private-token', 'expireTime': '2026-10-31T04:00:00Z'}))
        return response

    def attempt(self, responses):
        opener = MagicMock()
        opener.open.side_effect = responses
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(bridge, 'run', return_value=subprocess.CompletedProcess([], 0, 'caller-private-token')))
        stack.enter_context(patch.object(bridge.urllib.request, 'build_opener', return_value=opener))
        sleep = stack.enter_context(patch.object(bridge.time, 'sleep'))
        return opener, sleep

    def test_fixed_quota_project_and_sheets_scope(self):
        opener, sleep = self.attempt([self.response()])
        self.assertEqual(bridge.token()['access_token'], 'returned-private-token')
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header('X-goog-user-project'), bridge.PROJECT)
        self.assertEqual(json.loads(request.data), {'scope': [bridge.SCOPE], 'lifetime': '3600s'})
        self.assertEqual(request.full_url, 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/' + bridge.WRITER + ':generateAccessToken')
        sleep.assert_not_called()

    def test_propagation_retry_does_not_print_tokens(self):
        opener, sleep = self.attempt([failure('IAM_PERMISSION_DENIED'), self.response()])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            bridge.token()
        self.assertEqual(opener.open.call_count, 2)
        sleep.assert_called_once_with(40)
        self.assertIn('IAM_PERMISSION_DENIED', output.getvalue())
        self.assertNotIn('PRIVATE', output.getvalue())
        self.assertNotIn('private-token', output.getvalue())

    def test_other_403_reports_code_without_retry_or_secret(self):
        for reason in ['SERVICE_DISABLED', 'USER_PROJECT_DENIED', 'ACCESS_TOKEN_SCOPE_INSUFFICIENT']:
            with self.subTest(reason=reason):
                opener, sleep = self.attempt([failure(reason)])
                with self.assertRaises(RuntimeError) as caught:
                    bridge.token()
                self.assertIn(reason, str(caught.exception))
                self.assertNotIn('PRIVATE', str(caught.exception))
                sleep.assert_not_called()
                self.assertEqual(opener.open.call_count, 1)

    def test_permission_retry_is_bounded(self):
        opener, sleep = self.attempt([failure('IAM_PERMISSION_DENIED') for _ in range(12)])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            bridge.token()
        self.assertEqual(opener.open.call_count, 12)
        self.assertEqual(sleep.call_count, 11)

    def test_malformed_error_is_sanitized(self):
        error = urllib.error.HTTPError('https://iamcredentials.googleapis.com', 403, 'Forbidden', {}, io.BytesIO(b'PRIVATE-TOKEN'))
        self.assertEqual(bridge.iam_error(error), ('UNKNOWN', []))


if __name__ == '__main__':
    unittest.main()
